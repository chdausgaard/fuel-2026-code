"""T5 power (memo §8): pseudo assignments injected into the observed 2022 swings.

Each outer draw j permutes a cycle's state deviations within PADD groups and
assigns them to states: B̃^(j) = E_c × d^(j). The outcome is the observed 2022
swing plus β·B̃^(j), and RI with K inner permutations tests β = 0. Because
Y(β) − b·B̃^(j) = Y(0) − (b − β)·B̃^(j), the p-value for a true β is the
inverted test's p(−β) on Y(0), so one set of draws gives the whole power curve.

Sources:
- pseudo_2026 (memo T5): 2026 deviations (provisional until Nov 3) with the
  2026 design controls μ^pt and B̃^lag (2020 → 2024).
- pseudo_2022 (diagnostic): 2022 deviations and controls, i.e., how much
  power V_2022 had.

The memo also names the 2018 swings and control set C3; those need results
not yet built. C1 and C2 follow validate.C1 and validate.C2.

`substate` extends T5 to the three shock measures (w21, sa, ky12) and to the O6 two-layer assignment
(state deviations within PADD groups; cluster means of u within state × stratum), at four treatment
levels, for two shock sources and the 2022 and 2018 swings. Only permuted assignments meet outcomes.

    python -m fb26.models.power
    python -m fb26.models.power groupings
    python -m fb26.models.power substate
    python -m fb26.models.power house
"""
import functools
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from fb26.etl.exposure import with_market_oil
from fb26.models import burden, ri, validate
from fb26.util import paths, runlog

SEED = 20261026
J_OUTER = 500
K_INNER = 999
ALPHA = 0.05
MULTIPLES = np.linspace(-5, 5, 41)

# Assignment groupings compared for the variance-bias decision (PI, 2026-10-19).
# padd7 is the memo's design; census4 is R4 (Census Bureau regions).
CENSUS4 = {
    'NE': 'CT ME MA NH RI VT NJ NY PA',
    'MW': 'IL IN MI OH WI IA KS MN MO NE ND SD',
    'S': 'DE DC FL GA MD NC SC VA WV AL KY MS TN AR LA OK TX',
    'W': 'AZ CO ID MT NV NM UT WY AK CA HI OR WA',
}
GROUPINGS = ('padd7', 'padd5', 'census4', 'national')


def group_labels(shocks: pl.DataFrame, grouping: str) -> np.ndarray:
    padd = shocks['padd_group'].to_numpy()
    if grouping == 'padd7':
        return padd
    if grouping == 'padd5':
        return np.array([g[0] for g in padd])
    if grouping == 'census4':
        region = {s: r for r, members in CENSUS4.items() for s in members.split()}
        return shocks['state_abbr'].replace_strict(region).to_numpy()
    if grouping == 'national':
        return np.zeros(shocks.height, dtype=int).astype(str)
    raise ValueError(grouping)


def _demean(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, np.nan)
    for g in np.unique(groups):
        idx = (groups == g) & ~np.isnan(values)
        out[idx] = values[idx] - values[idx].mean()
    return out


def frame(source: str, grouping: str = 'padd7', outcome: int = 2022) -> tuple[pl.DataFrame, list[str], np.ndarray, np.ndarray]:
    """Outcome-cycle rows (default 2022) with the source cycle's design controls under a grouping."""
    df, _, _ = validate.frame(outcome, validate.PRIMARY)
    cycle = int(source.split('_')[1])
    if cycle == 2026:
        df = with_market_oil(df)
    if cycle != outcome:
        b = burden.build(cycle, validate.PRIMARY).select('geo_id', 'mu_pt')
        df = df.drop('mu_pt').join(b, on='geo_id', how='left', validate='m:1')
    return regroup(df, cycle, grouping)


def regroup(df: pl.DataFrame, cycle: int, grouping: str) -> tuple[pl.DataFrame, list[str], np.ndarray, np.ndarray]:
    """Rebuild B, B̃, B̃^lag and the group column of rows from a cycle's shocks under a grouping.

    Returns the rows, the 51-unit state order, the cycle's deviations and the
    group labels, so the treatment diagnostics describe that cycle.
    """
    shocks = (pl.read_parquet(paths.processed_dir() / 'shocks.parquet')
              .filter(pl.col('year') == cycle).sort('area_id'))
    groups = group_labels(shocks, grouping)
    dev = _demean(shocks['delta_p'].to_numpy(), groups)
    lag_dev = _demean(shocks['lag_delta_p'].to_numpy(), groups)
    if grouping == 'padd7':
        assert np.allclose(dev, shocks['deviation'].to_numpy())
    states = shocks['area_id'].to_list()
    by_state = pl.DataFrame({'state_fips': states, 'g': groups, 'dev': dev, 'lag_dev': lag_dev,
                             'dp': shocks['delta_p'].to_numpy()})
    df = (df.drop('padd_group', 'deviation', 'delta_p', strict=False)
          .join(by_state, on='state_fips', how='left', validate='m:1')
          .with_columns(pl.col('g').alias('padd_group'), pl.col('dev').alias('deviation'),
                        (pl.col('e_c') * pl.col('dp')).alias('b'),
                        (pl.col('e_c') * pl.col('dev')).alias('b_tilde'),
                        (pl.col('e_c') * pl.col('lag_dev')).alias('b_tilde_lag')))
    return df, states, dev, groups


def _within_race_sd(df: pl.DataFrame, x: np.ndarray) -> float:
    d = df.select('race_id', 'w').with_columns(pl.Series('x', x))
    w = pl.col('w')
    dm = d.select(pl.col('x') - (w * pl.col('x')).sum().over('race_id') / w.sum().over('race_id'), 'w')
    wv, xv = dm['w'].to_numpy(), dm['x'].to_numpy()
    return float(np.sqrt(np.sum(wv * xv ** 2) / wv.sum()))


def _within_race_cov(df: pl.DataFrame, x: np.ndarray, y: np.ndarray) -> float:
    """Weighted within-race covariance of x and y (both demeaned within race)."""
    d = df.select('race_id', 'w').with_columns(pl.Series('x', x), pl.Series('y', y))
    w = pl.col('w')
    dm = d.select([pl.col(c) - (w * pl.col(c)).sum().over('race_id') / w.sum().over('race_id') for c in ('x', 'y')] + ['w'])
    return float(np.sum(dm['w'].to_numpy() * dm['x'].to_numpy() * dm['y'].to_numpy()) / dm['w'].sum())


def simulate(source: str, control_set: str, j: int = J_OUTER, k: int = K_INNER,
             seed: int = SEED, grouping: str = 'padd7') -> tuple[pl.DataFrame, dict]:
    df, states, dev, groups = frame(source, grouping)
    precision = {'C0': (), 'C1': validate.C1, 'C2': validate.C2}[control_set]
    z = validate.design_matrix(df, precision=precision)
    state = df['state_fips'].replace_strict({s: i for i, s in enumerate(states)}).to_numpy()
    e = df['e_c'].to_numpy()
    design = ri.Design(df['dv'].to_numpy(), e, state, z, df['w'].to_numpy(), df['state_fips'].to_numpy(), dev.size)
    rng = np.random.default_rng(seed)
    pseudo = ri.permuted_deviations(dev, groups, j, rng)
    se, halfwidth, p = np.empty(j), np.empty(j), np.empty((j, MULTIPLES.size))
    invs = []
    for i in range(j):
        inv = design.inversion(pseudo[i], groups, k, rng)
        o = inv.obs
        se[i] = np.sqrt(o.f * o.s0[0]) / o.c[0]
        invs.append(inv)
        (lo, hi), _ = inv.ci(0.10, o.a[0] / o.c[0], se[i], n_grid=401)
        halfwidth[i] = (hi - lo) / 2
    scale = float(np.median(se))
    betas = MULTIPLES * scale
    for i, inv in enumerate(invs):
        p[i] = [inv.p(-b) for b in betas]
    power = (p <= ALPHA).mean(axis=0)
    sd_bt = _within_race_sd(df, e * dev[state])
    curve = pl.DataFrame({'source': source, 'control_set': control_set, 'beta': betas,
                          'beta_per_sd': betas * sd_bt, 'power': power})
    summary = {'source': source, 'control_set': control_set, 'j_outer': j, 'k_inner': k, 'seed': seed,
               'n_rows': df.height, 'se_cr1_median': scale, 'size_at_zero': float(power[MULTIPLES.size // 2]),
               'mde80_pos': _mde(betas, power, 1), 'mde80_neg': _mde(betas, power, -1),
               'ci90_halfwidth_median': float(np.median(halfwidth)), 'sd_bt_within_race': sd_bt}
    summary['mde80_pos_per_sd'] = summary['mde80_pos'] * sd_bt
    summary['mde80_neg_per_sd'] = summary['mde80_neg'] * sd_bt
    summary['ci90_halfwidth_per_sd'] = summary['ci90_halfwidth_median'] * sd_bt
    return curve, summary


def _mde(betas: np.ndarray, power: np.ndarray, sign: int) -> float:
    """Smallest |β| on one side with power ≥ 0.8, linearly interpolated; NaN if not reached."""
    side = betas * sign >= 0
    b, pw = np.abs(betas[side]), power[side]
    order = np.argsort(b)
    b, pw = b[order], pw[order]
    hit = np.flatnonzero(pw >= 0.8)
    if hit.size == 0:
        return float('nan')
    i = hit[0]
    if i == 0:
        return float(sign * b[0])
    return float(sign * np.interp(0.8, [pw[i - 1], pw[i]], [b[i - 1], b[i]]))


def compare_groupings() -> pl.DataFrame:
    """Treatment variation and C0/C1 power under each grouping (no V_t estimates are computed)."""
    rows = []
    for source in ('pseudo_2026', 'pseudo_2022'):
        for grouping in GROUPINGS:
            df, _, dev, groups = frame(source, grouping)
            in_sample = df['state_fips'].unique()
            for control_set in ('C0', 'C1'):
                _, s = simulate(source, control_set, grouping=grouping)
                rows.append({'grouping': grouping, 'n_groups': int(np.unique(groups).size),
                             'sd_dev_states': float(df.unique('state_fips')['deviation'].std()),
                             'n_states_in_sample': in_sample.len(), **validate.diagnostics(df), **s})
    return pl.DataFrame(rows)


# ---------------------------------------------------------------- sub-state levels (memo §5.1, O6, D22)

SUB_MEASURES = {'w21': 'shocks.parquet', 'sa': 'shocks_r9a.parquet', 'ky12': 'shocks_ky12.parquet'}
SUB_SOURCES = {'pseudo_2026': 2026, 'pseudo_2024': 2024}
SUB_LEVELS = ('state', 'metro', 'county_metro', 'county')
SUB_OUTCOMES = (2022, 2018)
MULTIPLES_WIDE = np.linspace(-10, 10, 81)     # contains MULTIPLES, so interpolation nodes are unchanged
EMPTY_COUNTIES = pl.DataFrame(schema={'geo_id': pl.Utf8, 'state_fips': pl.Utf8, 'u_c': pl.Float64, 'cluster': pl.Utf8,
                                      'stratum': pl.Utf8, 'src': pl.Utf8})
KY_PP_PER_DOLLAR = (0.2, 0.3)                 # Kim & Yang (2021), via prereg/lit/lit_design_fuel.md correction 1


@dataclass
class Assignment:
    """Both layers of one (measure, source, level): the pool that the pseudo and inner draws permute."""
    states: list[str]
    dev_state: np.ndarray
    clusters: pl.DataFrame      # cluster, state_fips, stratum, m (cluster mean of u), dev (m minus its stratum mean)
    counties: pl.DataFrame      # geo_id, u, cluster, src, resid (u minus cluster mean), unit (cluster index)
    groups: np.ndarray          # permutation groups of [states; clusters]
    dev: np.ndarray             # [dev_state; clusters.dev]
    state_table: pl.DataFrame


def state_table(measure: str, cycle: int) -> pl.DataFrame:
    """51 states' shock, lag shock, pass-through shock and their PADD deviations for a measure's state shock file."""
    # w21_hist (validation cycles at the metro level, `validate_metro`) is not one of the power runs' measures
    path = paths.processed_dir() / (SUB_MEASURES[measure] if measure != 'w21_hist' else 'shocks.parquet')
    sh = pl.read_parquet(path).filter(pl.col('year') == cycle).sort('area_id')
    if measure == 'w21' and cycle == 2024:
        # shocks.parquet holds the 2020 → 2024 change for 2024; the sub-state w21_2y rows are 2022 → 2024
        prev = (pl.read_parquet(paths.processed_dir() / 'shocks.parquet').filter(pl.col('year') == 2022)
                .select('area_id', pl.col('mean_post').alias('mean_prev')))
        sh = sh.join(prev, on='area_id', validate='1:1').with_columns((pl.col('mean_post') - pl.col('mean_prev')).alias('delta_p'))
    groups = sh['padd_group'].to_numpy()
    dev = _demean(sh['delta_p'].to_numpy(), groups)
    if not (measure == 'w21' and cycle == 2024):
        assert np.allclose(dev, sh['deviation'].to_numpy())
    lag = _demean(sh['lag_delta_p'].to_numpy(), groups)
    return pl.DataFrame({'state_fips': sh['area_id'], 'state_abbr': sh['state_abbr'], 'padd_group': sh['padd_group'],
                         'delta_p': sh['delta_p'], 'dev': dev, 'lag_dev': lag, 'g_pt': sh['g_pt'].cast(pl.Float64)}
                        ).with_columns(pl.col('lag_dev').fill_nan(None))


@functools.cache
def _interim_rows() -> pl.DataFrame:
    """In-memory sub-state shocks under the interim rules (post-window threshold 3 days); nothing is written."""
    from fb26.etl import substate_shocks as ss
    inputs, _, _ = ss.load_inputs()
    return ss.assemble(inputs, ss.RULESETS['interim'])


def county_rows_custom_window(pre_start, pre_end, post_start, post_end, min_days: int = 1) -> pl.DataFrame:
    """County-level sub-state rows from the daily price file with custom windows.

    Returns a DataFrame in the same schema as the county-level rows of the interim sub-state table,
    suitable for passing as `layer2_override` to `build_assignment`. Coverage requires `min_days`
    observed days in each window (default 1: any day is enough).
    """
    from datetime import date, timedelta
    from fb26.etl import substate_shocks as ss
    inputs, _, _ = ss.load_inputs()
    df = pl.read_parquet(paths.interim_dir() / 'prices_county_daily.parquet')
    pre = (df.filter((pl.col('as_of_date') >= pre_start) & (pl.col('as_of_date') <= pre_end))
           .group_by('county_fips').agg(pl.col('price_usd').mean().alias('mean_pre'), pl.len().alias('n_pre')))
    post = (df.filter((pl.col('as_of_date') >= post_start) & (pl.col('as_of_date') <= post_end))
            .group_by('county_fips').agg(pl.col('price_usd').mean().alias('mean_post'), pl.len().alias('n_post')))
    county = (pre.join(post, on='county_fips', how='inner')
              .with_columns((pl.col('mean_post') - pl.col('mean_pre')).alias('g_c'))
              .rename({'county_fips': 'geo_id'}))
    # Attach geography from the interim sub-state table
    interim = _interim_rows()
    geo = (interim.filter((pl.col('measure') == 'w21') & (pl.col('cycle') == 2026) & (pl.col('level') == 'county'))
           .select('geo_id', 'state_fips', 'cluster_id', 'stratum_id', 'cbsa_code', 'cbsa_type', 'padd_group'))
    # State prices from shocks.parquet
    sh = pl.read_parquet(paths.processed_dir() / 'shocks.parquet').filter(pl.col('year') == 2026)
    state_g = sh.select(pl.col('area_id').alias('state_fips'), pl.col('delta_p').alias('g_s'))
    result = (county.join(geo, on='geo_id', how='inner')
              .join(state_g, on='state_fips', how='left')
              .with_columns((pl.col('g_c') - pl.col('g_s')).alias('u_c'),
                            ((pl.col('n_pre') >= min_days) & (pl.col('n_post') >= min_days)).alias('coverage_ok'),
                            pl.lit('w21').alias('measure'), pl.lit(2026).alias('cycle'), pl.lit('county').alias('level'),
                            pl.lit(False).alias('provisional'), pl.lit('').alias('flags'))
              .rename({'n_pre': 'n_days_pre', 'n_post': 'n_days_post'}))
    return result


def layer2_rows(measure: str, cycle: int) -> pl.DataFrame:
    """Sub-state rows (county and metro) of a measure and cycle. 2026 w21 and sa come from the interim rules."""
    key = 'w21_2y' if (measure == 'w21' and cycle == 2024) else measure
    if cycle == 2026 and measure in ('w21', 'sa'):
        t = _interim_rows()
    else:
        t = pl.read_parquet(paths.processed_dir() / 'substate_shocks.parquet')
    return t.filter((pl.col('measure') == key) & (pl.col('cycle') == cycle))


def level_counties(rows: pl.DataFrame, level: str) -> pl.DataFrame:
    """Counties whose u comes from the level, with prefixed cluster and stratum ids (the two sources never share a pool).

    county: county-level u where coverage_ok. metro: metro-level u where coverage_ok. county_metro (D22 rule 1):
    county u where coverage_ok, else metro u where coverage_ok. state: none (u ≡ 0).
    """
    ok = pl.col('coverage_ok') & pl.col('u_c').is_not_null() & pl.col('cluster_id').is_not_null()

    def take(lv: str, src: str) -> pl.DataFrame:
        return (rows.filter((pl.col('level') == lv) & ok)
                .select('geo_id', 'state_fips', 'u_c', (src + ':' + pl.col('cluster_id')).alias('cluster'),
                        (src + ':' + pl.col('stratum_id')).alias('stratum'), pl.lit(src).alias('src')))

    empty = take('county', 'C').clear()
    if level == 'state':
        return empty
    county, metro = take('county', 'C'), take('metro', 'M')
    if level == 'county':
        return county
    if level == 'metro':
        return metro
    if level == 'county_metro':
        return pl.concat([county, metro.filter(~pl.col('geo_id').is_in(county['geo_id'].implode()))])
    raise ValueError(level)


def market_counties(rows: pl.DataFrame) -> pl.DataFrame:
    """Counties whose u comes from the county-market design: each market's u is the mean of its covered counties' u_c.

    Every county in a market with at least one covered county gets the market's layer 2.
    Counties without a covered market fall back to the state layer (not returned here).
    The market is the cluster_id from the sub-state table (CBSA within state, else CZ within state).
    """
    county = rows.filter(pl.col('level') == 'county')
    if county.height == 0:
        return level_counties(rows, 'state')
    ok = pl.col('coverage_ok') & pl.col('u_c').is_not_null() & pl.col('cluster_id').is_not_null()
    covered = county.filter(ok)
    if covered.height == 0:
        return level_counties(rows, 'state')
    # Market mean u from covered member counties
    market_u = covered.group_by('cluster_id').agg(pl.col('u_c').mean().alias('u_market'))
    # All counties in a market that has at least one covered county
    all_in_market = (county.filter(pl.col('cluster_id').is_in(market_u['cluster_id'].implode()))
                     .select('geo_id', 'state_fips', 'cluster_id', 'stratum_id')
                     .join(market_u, on='cluster_id', how='left', validate='m:1'))
    return all_in_market.select(
        'geo_id', 'state_fips', pl.col('u_market').alias('u_c'),
        ('K:' + pl.col('cluster_id')).alias('cluster'),
        ('K:' + pl.col('stratum_id')).alias('stratum'),
        pl.lit('K').alias('src'))


def _substate_config() -> dict:
    """Read the substate section from analysis.yaml; default to strata if absent."""
    import yaml
    path = paths.config_dir() / 'analysis.yaml'
    if not path.exists():
        return {'comparison': 'strata', 'source': 'county_markets'}
    cfg = yaml.safe_load(path.read_text())
    return cfg.get('substate', {'comparison': 'strata', 'source': 'county_markets'})


def substate_source_default() -> str:
    """The default sub-state source: analysis.yaml, substate.source (county_markets if absent)."""
    return _substate_config().get('source', 'county_markets')


def level_rule_default() -> str:
    """The default level rule: analysis.yaml, level.rule (per_market if absent)."""
    import yaml
    path = paths.config_dir() / 'analysis.yaml'
    if not path.exists():
        return 'per_market'
    cfg = yaml.safe_load(path.read_text())
    return (cfg.get('level') or {}).get('rule', 'per_market')


def metro_lag_default() -> bool:
    """Whether B̃^lag carries the metro layer: analysis.yaml, lag.metro_layer (False if absent: state-only lag)."""
    import yaml
    path = paths.config_dir() / 'analysis.yaml'
    if not path.exists():
        return False
    return bool((yaml.safe_load(path.read_text()).get('lag') or {}).get('metro_layer', False))


def _load_frozen_blocks() -> pl.DataFrame | None:
    """Read config/metro_blocks.csv, skipping the leading comment line.  Returns None if the file does not exist."""
    cfg = _substate_config()
    path = paths.repo_root() / cfg.get('blocks_file', 'config/metro_blocks.csv')
    if not path.exists():
        return None
    text = path.read_text()
    lines = text.splitlines()
    start = 0
    while start < len(lines) and lines[start].startswith('#'):
        start += 1
    import io
    return pl.read_csv(io.StringIO('\n'.join(lines[start:])), schema_overrides={'state_fips': pl.String})


def _load_frozen_market_blocks() -> pl.DataFrame | None:
    """Read config/market_blocks.csv, skipping the leading comment line.  Returns None if the file does not exist."""
    cfg = _substate_config()
    path = paths.repo_root() / cfg.get('market_blocks_file', 'config/market_blocks.csv')
    if not path.exists():
        return None
    text = path.read_text()
    lines = text.splitlines()
    start = 0
    while start < len(lines) and lines[start].startswith('#'):
        start += 1
    import io
    return pl.read_csv(io.StringIO('\n'.join(lines[start:])), schema_overrides={'state_fips': pl.String})


def _assign_blocks_to_clusters(clusters: pl.DataFrame, frozen: pl.DataFrame) -> np.ndarray:
    """Block labels for each row in `clusters`.  Rules (i)-(iv) from the design."""
    joined = clusters.select('cluster', 'stratum').join(
        frozen.select('cluster', 'block'), on='cluster', how='left')
    blocks = joined['block'].to_list()
    strata = clusters['stratum'].to_list()

    # (ii) unlisted clusters: join the nearest listed block of the same stratum
    for i, (b, st) in enumerate(zip(blocks, strata)):
        if b is not None:
            continue
        same = frozen.filter(pl.col('stratum') == st)
        if same.height == 0:
            blocks[i] = f'B:{st}:unlisted'
        else:
            blocks[i] = same['block'][0]

    blocks = np.array(blocks, dtype=object)

    # (iii) merge undersized blocks within each stratum
    strata_arr = np.array(strata)
    for stratum in np.unique(strata_arr):
        s_idx = np.flatnonzero(strata_arr == stratum)
        if len(s_idx) <= 1:
            continue
        s_blocks = blocks[s_idx]
        changed = True
        while changed:
            changed = False
            uniqs, counts = np.unique(s_blocks, return_counts=True)
            small = uniqs[counts < 2]
            big = uniqs[counts >= 2]
            if len(small) == 0 or len(big) == 0:
                break
            for sb in small:
                s_blocks[s_blocks == sb] = big[0]
                changed = True
        blocks[s_idx] = s_blocks

    return blocks


def build_assignment(measure: str, cycle: int, level: str,
                     comparison: str | None = None,
                     source: str | None = None,
                     layer2_override: pl.DataFrame | None = None) -> Assignment:
    """Build the two-layer assignment pool.

    comparison='strata' reproduces the original design (recentre and permute
    within state x CBSA-type strata).  comparison='blocks' recentres and
    permutes within the geographic blocks from config/metro_blocks.csv (metro)
    or config/market_blocks.csv (county_markets).
    None reads the default from config/analysis.yaml.

    source='county_markets': county prices grouped into local markets (CBSA/CZ
    within state); every county in a market with at least one covered county
    gets the market's layer 2. source='metro': the AAA metro design (the B1
    implementation). None reads the default.

    layer2_override: if provided, use these sub-state rows instead of reading from
    layer2_rows(). Must have the same schema as the county-level rows. Useful for
    running the design with custom treatment windows (e.g. the S1 stand-in).
    """
    if comparison is None:
        comparison = _substate_config().get('comparison', 'strata')
    if source is None:
        source = substate_source_default()
    st = state_table(measure, cycle)
    if layer2_override is not None and level != 'state':
        rows = layer2_override
    else:
        rows = layer2_rows(measure, cycle) if level != 'state' else None
    if rows is not None and layer2_override is None:
        gs = rows.filter(pl.col('g_s').is_not_null()).select('state_fips', 'g_s').unique('state_fips').join(st, on='state_fips')
        assert gs.height == 0 or (gs['g_s'] - gs['delta_p']).abs().max() < 1e-9, 'state shock differs from the sub-state table'
    use_market = source == 'county_markets' and level not in ('state', 'metro') and rows is not None
    if use_market:
        c = market_counties(rows).sort('geo_id')
    else:
        c = level_counties(rows, level).sort('geo_id') if rows is not None else EMPTY_COUNTIES
    c = c.with_columns(pl.col('u_c').mean().over('cluster').alias('m'))
    clusters = (c.group_by('cluster', maintain_order=True).agg(pl.col('state_fips').first(), pl.col('stratum').first(),
                                                              pl.col('stratum').n_unique().alias('n_strata'), pl.col('m').first())
                .sort('cluster'))
    assert clusters.height == 0 or clusters['n_strata'].max() == 1
    clusters = clusters.drop('n_strata')

    use_blocks = comparison == 'blocks' and clusters.height > 0
    if use_blocks:
        frozen = _load_frozen_market_blocks() if use_market else _load_frozen_blocks()
        if frozen is None:
            use_blocks = False   # fall back to strata when blocks file is missing
    if use_blocks:
        block_labels = _assign_blocks_to_clusters(clusters, frozen)
        clusters = clusters.with_columns(pl.Series('block', block_labels.tolist()))
        clusters = clusters.with_columns((pl.col('m') - pl.col('m').mean().over('block')).alias('dev'))
        l2_groups = clusters['block'].to_numpy()
    else:
        clusters = clusters.with_columns((pl.col('m') - pl.col('m').mean().over('stratum')).alias('dev'))
        l2_groups = clusters['stratum'].to_numpy()

    index = clusters.select('cluster', pl.int_range(pl.len(), dtype=pl.Int64).alias('unit'))
    c = (c.with_columns((pl.col('u_c') - pl.col('m')).alias('resid')).join(index, on='cluster', how='left')
         .rename({'u_c': 'u'}).drop('m'))
    # Block labels (B:...) are already unique; strata labels need 'S:' to avoid PADD clashes.
    if use_blocks:
        l2_prefix = l2_groups.astype(str)
    else:
        l2_prefix = np.char.add('S:', l2_groups.astype(str))
    groups = np.concatenate([('P:' + st['padd_group']).to_numpy(), l2_prefix])
    return Assignment(st['state_fips'].to_list(), st['dev'].to_numpy(), clusters, c, groups,
                      np.concatenate([st['dev'].to_numpy(), clusters['dev'].to_numpy()]), st)


def sub_frame(outcome: int, measure: str, cycle: int, level: str,
              comparison: str | None = None,
              metro_lag: bool | None = None,
              source: str | None = None,
              layer2_override: pl.DataFrame | None = None) -> tuple[pl.DataFrame, Assignment, list[str]]:
    """Outcome rows with B̃ and the design controls of the (measure, source cycle, level); returns the omitted controls.

    metro_lag: include the metro layer of B̃^lag (2020->2024 metro deviation, recentred within blocks or strata).
    None reads the registered setting (analysis.yaml, lag.metro_layer: off, the state-only lag); True is the
    robustness version.
    source: 'county_markets' | 'metro' | None (config default).
    layer2_override: pre-computed sub-state rows (custom treatment windows).
    """
    if metro_lag is None:
        metro_lag = metro_lag_default()
    df, _, _ = validate.frame(outcome, validate.PRIMARY)
    if cycle == 2026:
        df = with_market_oil(df)
    asg = build_assignment(measure, cycle, level, comparison=comparison, source=source,
                           layer2_override=layer2_override)
    assert df['c1_swing'].null_count() == 0
    st = asg.state_table.select('state_fips', 'padd_group', pl.col('dev').alias('dev_state'), 'lag_dev', 'g_pt')
    unit = asg.counties.select('geo_id', 'unit', pl.col('src'))
    dev2 = pl.DataFrame({'unit': np.arange(asg.clusters.height), 'dev2': asg.clusters['dev'].to_numpy()},
                        schema={'unit': pl.Int64, 'dev2': pl.Float64})
    df = (df.select('state_fips', 'geo_id', 'race_id', 'dv', 'w', 'e_c', 'c1_swing', 'oil_gas_emp_share')
          .join(st, on='state_fips', how='left', validate='m:1')
          .join(unit, on='geo_id', how='left', validate='m:1')
          .join(dev2, on='unit', how='left', validate='m:1')
          .with_columns(pl.col('unit').fill_null(-1), pl.col('dev2').fill_null(0.0)))
    # Metro layer of the lagged-burden control (2026 cycle, metro level only, any measure).
    # The lag is always the w21_hist 2020->2024 metro deviation, recentred within blocks or strata.
    # Validation cycles keep the state-only lag: their 2016 window predates the metro archive.
    lag_metro = np.zeros(df.height)
    if metro_lag and cycle == 2026 and level != 'state':
        from fb26.models.frame2026 import metro_lag_deviations
        lag_devs, _, _ = metro_lag_deviations(comparison)
        for i, geo in enumerate(df['geo_id'].to_list()):
            v = lag_devs.get(geo)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                lag_metro[i] = v
    df = (df.with_columns(pl.Series('_lag_metro', lag_metro))
          .with_columns((pl.col('e_c') * pl.col('dev2')).alias('b2'),
                        (pl.col('e_c') * (pl.col('dev_state') + pl.col('dev2'))).alias('b_tilde'),
                        (pl.col('e_c') * (pl.col('lag_dev') + pl.col('_lag_metro'))).alias('b_tilde_lag'),
                        (pl.col('e_c') * pl.col('g_pt')).alias('mu_pt'))
          .drop('_lag_metro'))
    omitted = [c for c in ('mu_pt', 'b_tilde_lag') if df[c].null_count()]
    return df, asg, omitted


def treatment_diagnostics(df: pl.DataFrame, asg: Assignment) -> dict:
    """Treatment-only quantities of the actual assignment (no outcomes): variation, layer-2 share, influence, counts."""
    e = df['e_c'].to_numpy()
    state = df['state_fips'].to_numpy()
    sd_bt = _within_race_sd(df, df['b_tilde'].to_numpy())
    share_l2 = _within_race_cov(df, df['b_tilde'].to_numpy(), df['b2'].to_numpy()) / sd_bt ** 2
    d = df.select('race_id', 'w', 'state_fips', 'e_c', 'b_tilde')
    w = pl.col('w')
    d = d.with_columns((pl.col('b_tilde') - (w * pl.col('b_tilde')).sum().over('race_id') / w.sum().over('race_id')).alias('dm'),
                       ((pl.col('e_c') - (w * pl.col('e_c')).sum().over('race_id') / w.sum().over('race_id')) ** 2 * w).alias('lam'))
    by = (d.group_by('state_fips', maintain_order=True).agg((w * pl.col('dm') ** 2).sum().alias('infl'), pl.col('lam').sum())
          .sort('infl', descending=True))
    infl = by['infl'].to_numpy()
    share = infl / infl.sum()
    abbr = dict(zip(asg.state_table['state_fips'], asg.state_table['state_abbr']))
    top = [(abbr[s], float(v)) for s, v in zip(by['state_fips'][:3], share[:3])]
    lam = by['lam'].to_numpy()
    cl = asg.clusters
    sizes = cl.group_by('stratum', maintain_order=True).len()
    ge2 = cl.join(sizes, on='stratum').filter(pl.col('len') >= 2)
    in_sample = df.filter(pl.col('unit') >= 0)['unit'].unique()
    ok_units = set(cl.with_row_index('i').join(ge2.select('cluster'), on='cluster')['i'].to_list())
    out = {'n_states_in_sample': int(df['state_fips'].n_unique()), 'n_cl_total': cl.height, 'n_cl_ge2': ge2.height,
           'n_cl_in_sample': int(in_sample.len()), 'n_cl_in_sample_ge2': int(sum(1 for u in in_sample.to_list() if u in ok_units)),
           'n_states_l2': int(cl['state_fips'].n_unique()), 'n_strata_ge2': int(ge2['stratum'].n_unique()),
           'share_rows_l2': float((df['unit'] >= 0).mean()), 'sd_bt_within_race': sd_bt,
           'share_var_l2': float(share_l2), 'n_eff_exposure': float(lam.sum() ** 2 / np.sum(lam ** 2)),
           'n_eff_influence': float(1 / np.sum(share ** 2))}
    for i in range(3):
        out[f'top{i + 1}_state'], out[f'top{i + 1}_share'] = top[i]
    return out


def verify_assignment(asg: Assignment) -> dict:
    """Treatment-only checks of the layer-2 part: stratum means of the pool are 0 and ū_c has the closed form."""
    cl, c = asg.clusters, asg.counties
    out = {'n_cl': cl.height, 'n_counties_l2': c.height}
    if cl.height == 0:
        return out | {'max_abs_stratum_mean': 0.0, 'max_abs_closed_form_gap': 0.0, 'sd_state_mean_u': 0.0, 'sd_state_mean_u_recentered': 0.0}
    out['max_abs_stratum_mean'] = float(cl.group_by('stratum', maintain_order=True).agg(pl.col('dev').mean().abs())['dev'].max())
    # ū_c = unweighted mean of the stratum's cluster means + the county's fixed residual; u_c - ū_c must equal the cluster's dev
    ubar = (cl.group_by('stratum', maintain_order=True).agg(pl.col('m').mean().alias('stratum_mean')))
    t = (c.join(cl.select('cluster', 'stratum', 'dev'), on='cluster').join(ubar, on='stratum')
         .with_columns((pl.col('u') - (pl.col('stratum_mean') + pl.col('resid'))).alias('u_minus_ubar')))
    out['max_abs_closed_form_gap'] = float((t['u_minus_ubar'] - t['dev']).abs().max())
    by = t.group_by('state_fips', maintain_order=True).agg(pl.col('u').mean().alias('mu'), pl.col('u_minus_ubar').mean().alias('mr'))
    out['sd_state_mean_u'] = float(by['mu'].std() or 0.0)
    out['sd_state_mean_u_recentered'] = float(by['mr'].std() or 0.0)
    return out


def ky_beta(multipliers=KY_PP_PER_DOLLAR) -> dict:
    """Kim & Yang (2021) per-dollar effect in β units: pp of vote share per pp of monthly income.

    E_c = 100 × (gallons per month) / (income / 12), so B_c = E_c × g is the extra monthly fuel cost in percent of
    monthly income; $1 per month is 100 / (Y / 12) = 1200 / Y percentage points of B at annual income Y.
    β_KY = (pp per $1/month) × Y / 1200, with Y the households-weighted median of county median household incomes.
    """
    ex = (pl.read_parquet(paths.processed_dir() / 'exposure_county.parquet').filter(pl.col('g_method') == validate.PRIMARY)
          .filter(pl.col('median_hh_income').is_not_null() & pl.col('households').is_not_null() & (pl.col('median_hh_income') > 0))
          .sort('median_hh_income'))
    inc, hh = ex['median_hh_income'].to_numpy(), ex['households'].to_numpy()
    cum = np.cumsum(hh) / hh.sum()
    q = {p: float(inc[np.searchsorted(cum, p)]) for p in (0.1, 0.5, 0.9)}
    return {'n_counties': ex.height, 'income_p10': q[0.1], 'income_median': q[0.5], 'income_p90': q[0.9],
            'dollar_in_pp_of_b': 1200 / q[0.5],
            'beta': {m: m * q[0.5] / 1200 for m in multipliers},
            'beta_p10': {m: m * q[0.1] / 1200 for m in multipliers}, 'beta_p90': {m: m * q[0.9] / 1200 for m in multipliers}}


def run_cell(df: pl.DataFrame, asg: Assignment, control_set: str, j: int, k: int, seed: int, ky: dict) -> dict:
    """T5 simulation for one assignment pool: pseudo draws permute both layers, each inner RI permutes both again."""
    precision = {'C0': (), 'C1': validate.C1, 'C2': validate.C2}[control_set]
    z = validate.design_matrix(df, precision=precision)
    states = asg.states
    state = df['state_fips'].replace_strict({s: i for i, s in enumerate(states)}).to_numpy()
    design = ri.TwoLayerDesign(df['dv'].to_numpy(), df['e_c'].to_numpy(), state, z, df['w'].to_numpy(), df['state_fips'].to_numpy(),
                               len(states), unit=df['unit'].to_numpy(), n_units=asg.clusters.height)
    rng = np.random.default_rng(seed)
    pseudo = ri.permuted_deviations(asg.dev, asg.groups, j, rng)
    se, halfwidth = np.empty(j), np.empty(j)
    invs = []
    for i in range(j):
        inv = design.inversion(pseudo[i], asg.groups, k, rng)
        o = inv.obs
        se[i] = np.sqrt(o.f * o.s0[0]) / o.c[0]
        invs.append(inv)
        (lo, hi), _ = inv.ci(0.10, o.a[0] / o.c[0], se[i], n_grid=401)
        halfwidth[i] = (hi - lo) / 2
    scale = float(np.median(se))
    betas = MULTIPLES_WIDE * scale
    power = np.array([np.mean([inv.p(-b) <= ALPHA for inv in invs]) for b in betas])
    sd_bt = _within_race_sd(df, df['b_tilde'].to_numpy())
    out = {'control_set': control_set, 'j_outer': j, 'k_inner': k, 'seed': seed, 'n_rows': df.height,
           'se_cr1_median': scale, 'size_at_zero': float(power[MULTIPLES_WIDE.size // 2]),
           'mde80_pos': _mde(betas, power, 1), 'mde80_neg': _mde(betas, power, -1),
           'ci90_halfwidth_median': float(np.median(halfwidth))}
    out['mde80_pos_per_sd'], out['mde80_neg_per_sd'] = out['mde80_pos'] * sd_bt, out['mde80_neg'] * sd_bt
    out['ci90_halfwidth_per_sd'] = out['ci90_halfwidth_median'] * sd_bt
    for m, b in ky['beta'].items():
        for sign, tag in ((1, 'pos'), (-1, 'neg')):
            out[f'power_ky{int(round(m * 10)):02d}_{tag}'] = float(np.mean([inv.p(-sign * b) <= ALPHA for inv in invs]))
    return out


def simulate_ubar_gap(asg: Assignment, n: int = 9_999, seed: int = SEED) -> dict:
    """Mean over n simulated layer-2 permutations of the cluster means vs the closed-form stratum mean (treatment only)."""
    cl = asg.clusters
    m, strata = cl['m'].to_numpy(), cl['stratum'].to_numpy()
    rng = np.random.default_rng(seed)
    total, total_sq = np.zeros(m.size), np.zeros(m.size)
    for start in range(0, n, 1000):
        d = ri.permuted_deviations(m, strata, min(1000, n - start), rng)
        total += d.sum(axis=0)
        total_sq += (d ** 2).sum(axis=0)
    mean = total / n
    closed = cl.select(pl.col('m').mean().over('stratum'))['m'].to_numpy()
    sd = np.sqrt(np.maximum(total_sq / n - mean ** 2, 0))
    varying = sd > 1e-12
    z = np.abs(mean - closed)[varying] / (sd[varying] / np.sqrt(n))
    return {'mc_n': n, 'mc_max_abs_gap': float(np.abs(mean - closed).max()), 'mc_max_abs_z': float(z.max()) if z.size else 0.0}


def cell_status(measure: str, cycle: int, level: str, cache: dict) -> tuple[str, str]:
    """('run', '') or (status, reason) for cells with no pool or whose pool equals another level's."""
    if level == 'state':
        return 'run', ''
    asg = cache.setdefault((measure, cycle, level), build_assignment(measure, cycle, level))
    if asg.clusters.height == 0:
        return 'empty', 'no covered counties at this level'
    if level == 'county_metro':
        for other in ('county', 'metro'):
            o = cache.setdefault((measure, cycle, other), build_assignment(measure, cycle, other))
            key = lambda a: a.counties.select('geo_id', 'u').sort('geo_id')  # noqa: E731
            if o.clusters.height and key(o).equals(key(asg)):
                return 'collapses', other
    return 'run', ''


def _cache_get(name: str):
    root = os.environ.get('FB26_POWER_CACHE')
    path = Path(root) / f'{name}.json' if root else None
    return json.loads(path.read_text()) if path and path.exists() else None


def _cache_put(name: str, row: dict) -> None:
    root = os.environ.get('FB26_POWER_CACHE')
    if root:
        Path(root).mkdir(parents=True, exist_ok=True)
        (Path(root) / f'{name}.json').write_text(json.dumps(row))


def substate_table(outcomes=SUB_OUTCOMES, j: int = J_OUTER, k: int = K_INNER, j_by_outcome: dict | None = None,
                   control_set: str = 'C1') -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict]:
    """Results rows, cell statuses, per-pool verification rows and the Kim & Yang conversion."""
    ky = ky_beta()
    rows, statuses, checks = [], [], []
    cache: dict = {}
    for source, cycle in SUB_SOURCES.items():
        for measure in SUB_MEASURES:
            for level in SUB_LEVELS:
                status, reason = cell_status(measure, cycle, level, cache)
                statuses.append({'measure': measure, 'source': source, 'level': level, 'status': status, 'reason': reason})
                if status != 'run':
                    continue
                checked = False
                for outcome in outcomes:
                    jj = (j_by_outcome or {}).get(outcome, j)
                    name = f'{measure}_{source}_{level}_{outcome}_{control_set}_{jj}_{k}'
                    row = _cache_get(name)
                    if row is None:
                        t0 = time.time()
                        df, asg, omitted = sub_frame(outcome, measure, cycle, level)
                        row = {'measure': measure, 'source': source, 'outcome': outcome, 'level': level,
                               **treatment_diagnostics(df, asg),
                               **run_cell(df, asg, control_set, jj, k, SEED, ky),
                               'controls_omitted': ';'.join(omitted), 'runtime_s': round(time.time() - t0, 1)}
                        if outcome == outcomes[0]:
                            row['_check'] = {'measure': measure, 'source': source, 'level': level,
                                             **verify_assignment(asg), **(simulate_ubar_gap(asg) if asg.clusters.height else {}),
                                             **mix_report(asg, measure, cycle, level)}
                        _cache_put(name, row)
                        print(measure, source, level, outcome, 'mde', round(row['mde80_pos'], 2), round(row['mde80_neg'], 2),
                              'size', row['size_at_zero'], f"{row['runtime_s']}s", flush=True)
                    if '_check' in row and not checked:
                        checks.append(row['_check'])
                        checked = True
                    rows.append({kk: vv for kk, vv in row.items() if kk != '_check'})
    return pl.DataFrame(rows), pl.DataFrame(statuses), pl.DataFrame(checks, infer_schema_length=None), ky


def mix_report(asg: Assignment, measure: str, cycle: int, level: str) -> dict:
    """Sources in a pool: counties and clusters by source; metros (state × CBSA) that hold counties of both sources."""
    c = asg.counties
    out = {'n_clusters_county_src': int(asg.clusters.filter(pl.col('cluster').str.starts_with('C:')).height),
           'n_clusters_metro_src': int(asg.clusters.filter(pl.col('cluster').str.starts_with('M:')).height),
           'n_clusters_mixed': 0, 'n_cbsa_split_across_sources': 0}
    if level == 'county_metro' and c.height:
        metro = (layer2_rows(measure, cycle).filter(pl.col('level') == 'metro').select('geo_id', pl.col('cluster_id').alias('metro_cluster'))
                 .drop_nulls())
        split = c.join(metro, on='geo_id').group_by('metro_cluster').agg(pl.col('src').n_unique().alias('n_src')).filter(pl.col('n_src') > 1)
        out['n_cbsa_split_across_sources'] = split.height
        out['n_clusters_mixed'] = int(c.group_by('cluster').agg(pl.col('src').n_unique().alias('n')).filter(pl.col('n') > 1).height)
    return out


LEVEL_LABEL = {'state': '(a) state', 'metro': '(b) metro', 'county_metro': '(c) county + metro', 'county': '(d) county'}


def _md(rows: list[list[str]], head: list[str]) -> str:
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '|'.join('---' for _ in head) + '|']
    return '\n'.join(lines + ['| ' + ' | '.join(r) + ' |' for r in rows])


def _f(x, n=2) -> str:
    return 'n/a' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.{n}f}'


def _pair(a, b, n=2) -> str:
    return f'{_f(a, n)} / {_f(b, n)}'


def _levels_present(tab: pl.DataFrame, source: str, outcome: int) -> list[dict]:
    order = {(m, lv): i for i, (m, lv) in enumerate((m, lv) for m in SUB_MEASURES for lv in SUB_LEVELS)}
    d = tab.filter((pl.col('source') == source) & (pl.col('outcome') == outcome)).to_dicts()
    return sorted(d, key=lambda r: order[(r['measure'], r['level'])])


def write_substate_report(tab: pl.DataFrame, status: pl.DataFrame, checks: pl.DataFrame, ky: dict, out: Path) -> None:
    j_all = sorted(set(tab['j_outer'].to_list()))
    k = tab['k_inner'][0]
    p = []
    p.append('<!-- role: result snapshot | as-of: 2026-10-19 | T5 power across shock measures and treatment levels (O6 two-layer RI, '
             'control set C1); superseded by the next run of python -m fb26.models.power substate -->\n')
    p.append('# T5 across measures and treatment levels (2026-10-19)\n')
    p.append('## Method\n')
    p.append('Same simulation as `T5_power.md` (memo §8, T5), extended to the three candidate shock measures and to the O6 two-layer '
             'assignment model (memo §5.1, D22; `prereg/o6_substate_assignment.md` §§1-4).\n')
    p.append(f'''- **Pseudo-assignments.** Each of J = {"/".join(map(str, j_all))} outer draws permutes **both layers** and builds B̃^(j) = E_c × [(g_s − ḡ_PADD) + (u_c − ū_c)]:
  state deviations within PADD groups, and the cluster means of u within state × CBSA-type strata (clusters: CBSA, else 2020 commuting zone, within state). Each county's within-cluster residual is fixed and drops out of B̃; counties with u ≡ 0 sit in layer 1 only.
- **Injection and test.** β·B̃^(j) is added to the observed swings (C1 plus design controls: race FE, E_c × PADD, μ^pt and B̃^lag where the measure has them, Oil_c). β = 0 is tested by RI with K = {k} inner draws that permute both layers again, independently. Statistic: studentized coefficient with CR1 SEs clustered by state. Two-sided, α = 0.05, seed {SEED}.
- **Blind rule.** Every estimate below regresses observed swings on a *permuted* pseudo-assignment. No estimate uses the actual B or B̃ of any measure or level (none is in the CSV). Descriptive treatment statistics (SD, shares, influence) use the actual assignment but no outcomes.
- **Units.** β is in percentage points of Republican two-party share per percentage point of monthly income in relative burden. "Per SD" multiplies by the within-race SD of B̃ of that row's actual assignment.
- **Outcomes.** 2022 swings (67 races) and 2018 swings, each against the baseline presidential election; sample and weights as in `validate.frame`.
- **Shock sources.** `pseudo_2026`: the provisional 2026 shocks (w21 and sa windows have 5 of 21 days; the w21 and sa sub-state rows are built in memory under the interim rules, post-window threshold 3 days; ky12 county rows use the primary rules). `pseudo_2024`: the 2024 shocks, which have 48-state county coverage. For w21 the 2024 state layer is the 2022 → 2024 two-year change (the `w21_2y` definition used by the sub-state rows), not the 2020 → 2024 change in `shocks.parquet`.
- **Levels.** (a) state only; (b) metro u where the metro row has `coverage_ok`; (c) county u where `coverage_ok`, else metro u where `coverage_ok` (D22 rule 1; cluster and stratum ids are prefixed C: / M:, so the two sources never share a pool); (d) county u wherever `coverage_ok`. Elsewhere u ≡ 0.
- Code: `src/fb26/models/power.py`, `src/fb26/models/ri.py` (`TwoLayerDesign`). Table: `reports/validation/T5_substate.csv`.
''')
    # tables
    p.append('## Results\n')
    for source in SUB_SOURCES:
        for outcome in SUB_OUTCOMES:
            rows = _levels_present(tab, source, outcome)
            if not rows:
                continue
            p.append(f'### {source} × {outcome} swings (C1)\n')
            body = []
            for r in rows:
                body.append([r['measure'], LEVEL_LABEL[r['level']], f"{r['n_rows']:,}", f"{r['n_cl_total']} ({r['n_cl_ge2']})",
                             _f(r['sd_bt_within_race'], 3), _f(r['share_var_l2'], 2), _f(r['size_at_zero'], 3),
                             _pair(r['mde80_pos'], r['mde80_neg']), _pair(r['mde80_pos_per_sd'], r['mde80_neg_per_sd']),
                             f"{_f(r['ci90_halfwidth_median'])} ({_f(r['ci90_halfwidth_per_sd'])})",
                             _pair(r['power_ky02_pos'], r['power_ky02_neg']), _pair(r['power_ky03_pos'], r['power_ky03_neg']),
                             r['controls_omitted'] or 'none'])
            p.append(_md(body, ['Measure', 'Level', 'Rows', 'Layer-2 clusters (in strata ≥ 2)', 'SD of B̃', 'Layer-2 share', 'Size at β = 0',
                                'MDE80 β > 0 / β < 0', 'MDE80 per SD', '90% CI half-width (per SD)',
                                f'Power at ±{ky["beta"][0.2]:.1f} (KY 0.2)', f'Power at ±{ky["beta"][0.3]:.1f} (KY 0.3)', 'Controls omitted']))
            p.append('')
    p.append('Layer-2 share is Cov(B̃, layer-2 part) / Var(B̃) within races, weighted, so the layer-1 share is one minus it. '
             'Power at ±β is shown as β > 0 / β < 0 ("KY" is the Kim & Yang benchmark below: an external, likely inflated estimate, not an expected effect).\n')
    p.append('### Treatment variation (2022-outcome sample, actual assignments, no outcomes)\n')
    for source in SUB_SOURCES:
        rows = _levels_present(tab, source, 2022)
        body = []
        for r in rows:
            body.append([r['measure'], LEVEL_LABEL[r['level']], str(r['n_states_in_sample']), str(r['n_states_l2']),
                         f"{r['n_cl_total']} / {r['n_cl_ge2']} / {r['n_cl_in_sample_ge2']}", _f(r['share_rows_l2'], 2),
                         _f(r['n_eff_exposure'], 1), _f(r['n_eff_influence'], 1),
                         ', '.join(f"{r[f'top{i}_state']} {r[f'top{i}_share']:.0%}" for i in (1, 2, 3))])
        p.append(f'**{source}**\n')
        p.append(_md(body, ['Measure', 'Level', 'States in sample', 'States with layer 2', 'Clusters: total / in strata ≥ 2 / in sample and strata ≥ 2',
                            'Share of rows with a layer-2 unit', 'n_eff (exposure)', 'n_eff (influence)', 'Top three states, share of within-race B̃ variation']))
        p.append('')
    p.append('n_eff (exposure) is (Σλ)² / Σλ² over states, λ_s = Σ w (E_c − Ē_race)², as in `T5_groupings.csv`. '
             'n_eff (influence) is 1 / Σ share², with a state\'s share = its part of Σ w (B̃ − race mean)², so it moves with the realized shocks and the layer-2 variation. '
             'A singleton stratum cannot permute: its cluster is in the total but contributes a zero layer-2 part.\n')
    # empty
    p.append('## Cells not run\n')
    skip = status.filter(pl.col('status') != 'run')
    if skip.height:
        body = [[r['measure'], r['source'], LEVEL_LABEL[r['level']],
                 'empty: ' + r['reason'] if r['status'] == 'empty' else f"collapses to {LEVEL_LABEL[r['reason']]} (identical pool)"]
                for r in skip.to_dicts()]
        p.append(_md(body, ['Measure', 'Source', 'Level', 'Status']))
    else:
        p.append('None.')
    p.append('')
    # KY
    p.append('## Kim & Yang benchmark\n')
    inc = ky['income_median']
    p.append(f'''**External estimate, different design (commute time × national prices, presidential vote), likely inflated by selection on significance. A benchmark, not an expected effect.**

1. Kim & Yang's interaction (0.25 pp per minute of commute per $1 of price) works out, per `prereg/lit/lit_design_fuel.md` (header correction 1), to about 0.2-0.3 pp of the president's party's vote share per $1 per month of extra fuel cost.
2. Our B is the extra monthly fuel cost in percent of monthly household income: E_c = 100 × (gallons per month) / (income / 12) (`src/fb26/etl/exposure.py`), B_c = E_c × g with g in $ per gallon. So $1 per month is 100 / (Y / 12) = 1200 / Y points of B, with Y the annual income.
3. Income figure: the households-weighted median of county median household incomes (ACS 2020-2024, {ky['n_counties']:,} counties) is Y = ${inc:,.0f} (10th and 90th percentiles: ${ky['income_p10']:,.0f} and ${ky['income_p90']:,.0f}). So $1 per month = 1200 / {inc:,.0f} = {ky['dollar_in_pp_of_b']:.4f} points of B.
4. β_KY = (pp per $1/month) / ({ky['dollar_in_pp_of_b']:.4f} points of B per $1/month) = (pp per $1/month) × {inc:,.0f} / 1200.
   - 0.2 pp: β_KY = 0.2 × {inc:,.0f} / 1200 = **{ky['beta'][0.2]:.2f}** (at the 10th / 90th percentile income: {ky['beta_p10'][0.2]:.1f} / {ky['beta_p90'][0.2]:.1f}).
   - 0.3 pp: β_KY = 0.3 × {inc:,.0f} / 1200 = **{ky['beta'][0.3]:.2f}** ({ky['beta_p10'][0.3]:.1f} / {ky['beta_p90'][0.3]:.1f}).
5. Sign: the effect is against the president's party, so β_KY > 0 when the president is a Democrat (2022) and β_KY < 0 when a Republican (2018, 2026). The tables show power at both signs.
6. Approximations: a constant β in B units implies an effect per dollar that scales with county income (a fixed dollar is a larger share of a poorer county's income); the conversion uses the median income. Vote share is taken as the same unit as the two-party share.
''')
    # verification
    p.append('## Verification of the layer-2 recentering (treatment data only)\n')
    body = []
    for r in checks.filter(pl.col('level') != 'state').to_dicts():
        g = {key: (r.get(key) or 0.0) for key in ('max_abs_stratum_mean', 'max_abs_closed_form_gap', 'mc_max_abs_gap', 'mc_max_abs_z')}
        body.append([r['measure'], r['source'], LEVEL_LABEL[r['level']], str(r['n_cl']), f"{g['max_abs_stratum_mean']:.1e}",
                     f"{g['max_abs_closed_form_gap']:.1e}", f"{g['mc_max_abs_gap']:.1e} (z {g['mc_max_abs_z']:.1f})",
                     _f(r.get('sd_state_mean_u'), 4), _f(r.get('sd_state_mean_u_recentered'), 4)])
    p.append(_md(body, ['Measure', 'Source', 'Level', 'Clusters', 'Max \\|mean of layer-2 part across clusters in a state × stratum\\|',
                        'Max \\|(u − ū) − (m_k − m̄)\\|', 'Max gap of ū to the mean over 9,999 simulated permutations (max z)',
                        'SD over states of the mean u', 'SD over states of the mean of u − ū']))
    sub = checks.filter(pl.col('level') != 'state')
    rng_u = f"{sub['sd_state_mean_u'].min():.3f}-{sub['sd_state_mean_u'].max():.3f}"
    rng_r = f"{sub['sd_state_mean_u_recentered'].min():.3f}-{sub['sd_state_mean_u_recentered'].max():.3f}"
    p.append(f'''
- **Mean zero.** The layer-2 part of B̃ is E_c × (m_k − m̄_stratum), the cluster mean minus the unweighted mean of the cluster means in the county's state × stratum. Across clusters its mean in every state × stratum is zero to rounding (column 5), and u_c − ū_c equals it (column 6) because the fixed residual enters u and ū alike.
- **Closed form vs simulation.** The closed form ū_c = (unweighted mean of the stratum's cluster means) + the county's residual matches the mean over 9,999 simulated layer-2 permutations (column 7; z is the gap in Monte Carlo standard errors, and the maximum over several hundred clusters is expected to be 3-4).
- **Where the state means of u go.** A state's mean u (column 8, SD across states {rng_u}, the `sd_state_mean_u` of `substate_shocks.md`) is carried by the stratum means of the cluster means, which are part of ū_c. Layer 2 permutes only deviations from those means, so that level difference is in μ_c = E_c × (ḡ_PADD + ū_c), not in B̃. The state layer takes the state's overall shock g_s through g_s − ḡ_PADD. The county-weighted state means of u − ū (column 9, SD {rng_r}) are not exactly zero, because clusters differ in size and the zero is across clusters, but they are smaller than the state means of u.
- μ_c's sub-state part is **not** a control in these regressions (memo/O6: optional robustness row); E_c × state dummies are excluded by design.
''')
    # mixed
    mixed = checks.filter(pl.col('level') == 'county_metro')
    if mixed.height:
        p.append('### Sources in level (c)\n')
        body = [[r['measure'], r['source'], str(r['n_clusters_county_src']), str(r['n_clusters_metro_src']), str(r['n_clusters_mixed']),
                 str(r['n_cbsa_split_across_sources'])] for r in mixed.to_dicts()]
        p.append(_md(body, ['Measure', 'Source', 'Clusters from county u', 'Clusters from metro u', 'Clusters mixing sources', 'Metros whose counties come from both sources']))
        p.append('\nClusters never mix sources (ids are prefixed and cluster means are recomputed over the counties of the cluster\'s own source). A metro that is split has some counties with county u (cluster C:) and the rest with metro u (cluster M:), which then permute in separate pools.\n')
    # regression
    p.append('## Regression check\n')
    old = pl.read_csv(paths.repo_root() / 'reports' / 'validation' / 'T5_power.csv').filter(
        (pl.col('source') == 'pseudo_2026') & (pl.col('control_set') == 'C1')).select('mde80_pos', 'mde80_neg', 'size_at_zero', 'ci90_halfwidth_median').unique()
    new = tab.filter((pl.col('measure') == 'w21') & (pl.col('source') == 'pseudo_2026') & (pl.col('level') == 'state') & (pl.col('outcome') == 2022))
    if old.height == 1 and new.height == 1 and int(new['j_outer'][0]) == J_OUTER:
        o, n = old.row(0, named=True), new.row(0, named=True)
        gaps = {c: abs(o[c] - n[c]) for c in o}
        exact = max(gaps.values()) < 1e-9
        p.append(f"Level (a) × w21 × pseudo_2026 × 2022 × C1 through the two-layer code gives MDE {n['mde80_pos']:.2f} / {n['mde80_neg']:.2f}, "
                 f"size {n['size_at_zero']:.3f}, expected 90% CI half-width {n['ci90_halfwidth_median']:.2f}; `T5_power.csv` has {o['mde80_pos']:.2f} / {o['mde80_neg']:.2f}, "
                 f"{o['size_at_zero']:.3f}, {o['ci90_halfwidth_median']:.2f}. " + ('**Exact match** (largest absolute difference below 1e-9): the RNG stream lines up.\n' if exact else
                 f"Largest absolute difference {max(gaps.values()):.2g}: not exact, so Monte Carlo noise only if small.\n"))
    else:
        p.append('Not computed in this run (the reference row was not run at the reference J).\n')
    # what it means
    p.append(what_it_means(tab, ky))
    out.write_text('\n'.join(p) + '\n')


def what_it_means(tab: pl.DataFrame, ky: dict) -> str:
    lines = []
    t = tab.filter(pl.col('outcome') == 2022)
    n_bad = tab.filter((pl.col('size_at_zero') < 0.03) | (pl.col('size_at_zero') > 0.07)).height
    lines.append(f'**Size.** Rejection rates at β = 0 are {tab["size_at_zero"].min():.3f}-{tab["size_at_zero"].max():.3f} across the {tab.height} rows; '
                 f'{n_bad} fall outside 0.03-0.07 (the Monte Carlo SE of a rate near 0.05 with J = 500 is about 0.01).')
    for source in SUB_SOURCES:
        s = t.filter(pl.col('source') == source)
        parts = []
        for m in SUB_MEASURES:
            r = s.filter(pl.col('measure') == m)
            if not r.height:
                continue
            base = r.filter(pl.col('level') == 'state')
            sub = r.filter(pl.col('level') != 'state')
            txt = f"{m}: state {_pair(base['mde80_pos'][0], base['mde80_neg'][0], 1)}"
            for d in sorted(sub.to_dicts(), key=lambda r: SUB_LEVELS.index(r['level'])):
                txt += f"; {LEVEL_LABEL[d['level']][:3]} {_pair(d['mde80_pos'], d['mde80_neg'], 1)}"
            parts.append(txt)
        lines.append(f'**MDE80 (β > 0 / β < 0), {source} × 2022 swings, C1.** ' + '; '.join(parts) + '.')
    # ratios
    ratios = []
    for d in t.filter(pl.col('level') != 'state').to_dicts():
        b = t.filter((pl.col('measure') == d['measure']) & (pl.col('source') == d['source']) & (pl.col('level') == 'state'))
        if b.height:
            ratios.append(((d['mde80_pos'] / b['mde80_pos'][0] + d['mde80_neg'] / b['mde80_neg'][0]) / 2, d))
    if ratios:
        lo, hi = min(r[0] for r in ratios), max(r[0] for r in ratios)
        lines.append(f'**Sub-state levels against the state level.** The mean of the two-sided MDE ratios to the same measure and source at the state level runs from {lo:.2f} to {hi:.2f} '
                     f'over {len(ratios)} cells (2022 swings); differences under about 10% are within Monte Carlo noise.')
    ps = t.select(pl.min_horizontal(pl.col('mde80_pos_per_sd'), -pl.col('mde80_neg_per_sd')).alias('lo'),
                  pl.max_horizontal(pl.col('mde80_pos_per_sd'), -pl.col('mde80_neg_per_sd')).alias('hi'))
    sd_state = t.filter(pl.col('level') == 'state')['sd_bt_within_race']
    sd_sub = t.filter(pl.col('level') != 'state')['sd_bt_within_race']
    lines.append(f'**Per SD of B̃.** The MDE per within-race SD of B̃ is {ps["lo"].min():.2f}-{ps["hi"].max():.2f} across all {t.height} 2022-swing rows, '
                 f'while the SD of B̃ is {sd_state.min():.3f}-{sd_state.max():.3f} at the state level and {sd_sub.min():.3f}-{sd_sub.max():.3f} at the sub-state levels. '
                 'The MDE in β units therefore differs across levels through the amount of treatment variation, not through precision per unit of variation.')
    k2 = t.filter((pl.col('mde80_pos') <= ky['beta'][0.3]) & (pl.col('mde80_neg') >= -ky['beta'][0.3])).height
    lines.append(f'**Kim & Yang benchmark.** β_KY is {ky["beta"][0.2]:.1f} (0.2 pp) to {ky["beta"][0.3]:.1f} (0.3 pp) at the median income. '
                 f'In {k2} of {t.height} 2022-swing rows both MDEs are below the upper value. This is an external, likely inflated benchmark.')
    a = tab.filter(pl.col('outcome') == 2018)
    if a.height:
        mean_ratio = float(np.mean([(r['mde80_pos'] / t.filter((pl.col('measure') == r['measure']) & (pl.col('source') == r['source']) & (pl.col('level') == r['level']))['mde80_pos'][0]) for r in a.to_dicts()]))
        lines.append(f'**2018 vs 2022 swings.** The 2018 MDE (β > 0) is on average {mean_ratio:.2f} times the 2022 MDE for the same row.')
    om = (tab.filter(pl.col('controls_omitted') != '').select('measure', 'source', 'controls_omitted').unique(maintain_order=True)
          .group_by('controls_omitted', maintain_order=True).agg(pl.concat_str(pl.col('measure'), pl.lit(' × '), pl.col('source')).str.join(', ').alias('rows')))
    omitted = '; '.join(f"{r['controls_omitted'].replace(';', ' and ')} omitted for {r['rows']}" for r in om.to_dicts()) or 'no control omitted'
    lines.append('**What the numbers rest on.** Any sub-state gain comes from layer 2 being exchangeable within state × stratum, as O6 assumes; '
                 'this run does not test that (persistence and spatial diagnostics are in `substate_shocks.md`). The pseudo_2026 w21 and sa sub-state shocks use 5 of 21 days with a 3-day threshold. '
                 f'Controls: {omitted}.')
    lines.append('**Monte Carlo.** J = 500: the SE of a power estimate is about 0.01 near 0.05 and about 0.018 near 0.8, so MDE differences under about 10% are noise.')
    return '## What it means\n\n' + '\n'.join(f'{i}. {line}' for i, line in enumerate(lines, 1))


def substate_main() -> None:
    t0 = time.time()
    tab, status, checks, ky = substate_table()
    out_dir = paths.repo_root() / 'reports' / 'validation'
    csv = tab.with_columns(pl.col('outcome').cast(pl.Int32))
    csv.write_csv(out_dir / 'T5_substate.csv')
    write_substate_report(tab, status, checks, ky, out_dir / 'T5_substate.md')
    runlog.append_run_log(
        run_id=uuid.uuid4().hex[:12], utc_timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        git_sha=runlog.git_sha(), git_dirty=runlog.git_dirty(), config_hash='', spec_id='T5_substate',
        data_snapshot_ids='exposure_county.parquet;shocks.parquet;shocks_r9a.parquet;shocks_ky12.parquet;substate_shocks.parquet;results_county.parquet',
        output_path='reports/validation/T5_substate.csv;reports/validation/T5_substate.md',
        notes=f'J={sorted(set(tab["j_outer"].to_list()))}; K={K_INNER}; seed={SEED}; C1; measures w21/sa/ky12; levels state/metro/county+metro/county; '
              f'sources pseudo_2026/pseudo_2024; outcomes 2022/2018; permuted assignments only; runtime {time.time() - t0:.0f}s')
    print(f'{time.time() - t0:.0f}s')


def house_analysis_frame(house: pl.DataFrame, asg: Assignment, columns: list[str]) -> pl.DataFrame:
    """Attach exactly the source assignment and controls used by sub_frame to House rows."""
    st = asg.state_table.select('state_fips', 'padd_group', pl.col('dev').alias('dev_state'), 'lag_dev', 'g_pt')
    unit = asg.counties.select('geo_id', 'unit', 'src')
    dev2 = pl.DataFrame({'unit': np.arange(asg.clusters.height), 'dev2': asg.clusters['dev'].to_numpy()},
                        schema={'unit': pl.Int64, 'dev2': pl.Float64})
    return (house.select('state_fips', 'geo_id', 'race_id', 'dv', 'w', 'e_c', 'c1_swing', 'oil_gas_emp_share')
            .join(st, on='state_fips', how='left', validate='m:1')
            .join(unit, on='geo_id', how='left', validate='m:1')
            .join(dev2, on='unit', how='left', validate='m:1')
            .with_columns(pl.col('unit').fill_null(-1), pl.col('dev2').fill_null(0.0))
            .with_columns((pl.col('e_c') * pl.col('dev2')).alias('b2'),
                          (pl.col('e_c') * (pl.col('dev_state') + pl.col('dev2'))).alias('b_tilde'),
                          (pl.col('e_c') * pl.col('lag_dev')).alias('b_tilde_lag'),
                          (pl.col('e_c') * pl.col('g_pt')).alias('mu_pt'))
            .select(columns))


def house_table() -> tuple[pl.DataFrame, list[dict]]:
    """24 independent, identically seeded T5 cells; no caches and no actual-assignment fit."""
    from fb26.models import house_frame, run2026

    houses, audit = {}, []
    for outcome in SUB_OUTCOMES:
        houses[outcome], stages = house_frame.frame(outcome)
        audit.extend(stages)
    rows, ky = [], ky_beta()
    for measure in ('w21', 'sa'):
        for level in ('state', 'metro'):
            for outcome in SUB_OUTCOMES:
                statewide, asg, _ = sub_frame(outcome, measure, 2026, level)
                h = house_analysis_frame(with_market_oil(houses[outcome]), asg, statewide.columns)
                stacked = house_frame.stack(statewide, h)
                # Check the actual presidential vote budget, not just internal normalization.
                s_base, _, _ = validate.frame(outcome, validate.PRIMARY)
                budgets = pl.concat([s_base.select('geo_id', 'v_c'), houses[outcome].select('geo_id', 'v_c')]).unique('geo_id')
                for d in (statewide, h, stacked):
                    sums = d.group_by('geo_id', maintain_order=True).agg(pl.col('w').sum()).join(budgets, on='geo_id')
                    assert np.allclose(sums['w'], sums['v_c'], rtol=1e-12, atol=1e-8), 'County weight differs from V_c'
                s_states = set(statewide['state_fips'].to_list())
                abbr = dict(zip(asg.state_table['state_fips'], asg.state_table['state_abbr']))
                for label, df in (('S', statewide), ('H', h), ('S+H', stacked)):
                    t0 = time.time()
                    added = sorted(set(df['state_fips'].to_list()) - s_states)
                    omitted = [c for c in ('mu_pt', 'b_tilde_lag') if df[c].null_count()]
                    row = {'measure': measure, 'source': 'pseudo_2026', 'outcome': outcome, 'level': level, 'frame': label,
                           'n_races': df['race_id'].n_unique(), 'n_counties': df['geo_id'].n_unique(),
                           'n_states_house_only': len(added), 'states_house_only': ';'.join(abbr[s] for s in added),
                           **treatment_diagnostics(df, asg),
                           **run_cell(df, asg, 'C1', J_OUTER, K_INNER, SEED, ky),
                           'controls_omitted': ';'.join(omitted), 'runtime_s': round(time.time() - t0, 1)}
                    for sign in ('pos', 'neg'):
                        row[f'mde80_{sign}_per_10_dollars'] = run2026.per_10_dollars(row[f'mde80_{sign}'])
                    rows.append(row)
                    print(measure, level, outcome, label, 'mde', round(row['mde80_pos'], 2), round(row['mde80_neg'], 2),
                          'size', row['size_at_zero'], f"{row['runtime_s']}s", flush=True)
    return pl.DataFrame(rows), audit


def house_reproduction(tab: pl.DataFrame) -> str:
    """Compare independently rerun S cells to every applicable reference column."""
    ref = pl.read_csv(paths.repo_root() / 'reports' / 'validation' / 'T5_substate.csv')
    columns = [c for c in ref.columns if c in tab.columns and c != 'runtime_s']
    exact, close, gaps = 0, 0, []
    for row in tab.filter(pl.col('frame') == 'S').to_dicts():
        old = ref.filter((pl.col('measure') == row['measure']) & (pl.col('source') == row['source']) &
                         (pl.col('level') == row['level']) & (pl.col('outcome') == row['outcome']))
        if old.height != 1:
            raise ValueError('Missing or duplicate T5_substate reference row')
        previous = old.row(0, named=True)
        same, near = True, True
        for c in columns:
            a, b = row[c], previous[c]
            if isinstance(a, float) and isinstance(b, (float, int)):
                same &= a == b
                near &= bool(np.isclose(a, b, rtol=1e-12, atol=1e-12, equal_nan=True))
                gaps.append(abs(a - b))
            else:
                same &= a == b
                near &= a == b
        exact += int(same)
        close += int(near)
    return (f'S-row reproduction: {exact}/8 rows exactly equal across all {len(columns)} shared reference columns '
            f'(runtime excluded); {close}/8 agree within rtol=atol=1e-12. '
            f'Maximum absolute numeric difference: {max(gaps):.3g}. '
            'S uses the unchanged sub_frame and run_cell, with the same seed reset and permutation stream in each cell; '
            'no reference results are substituted.')


def write_house_report(tab: pl.DataFrame, audit: list[dict], reproduction: str, out: Path) -> None:
    from fb26.models import run2026

    p = ['<!-- role: result snapshot | as-of: 2026-10-19 | T5 with U.S. House swings; superseded by the next run of python -m fb26.models.power house -->',
         '', '# T5 with U.S. House swings', '', '## Method', '',
         'Same simulation and units as `T5_substate.md`: pseudo_2026 only, w21 (two-year change) and sa (since January), '
         f'J = {J_OUTER}, K = {K_INNER}, seed {SEED}, C1, two-sided α = 0.05, CR1 clustered by state. '
         'Each outer draw permutes state deviations within PADD groups and metro cluster deviations within state × stratum; '
         'each inner RI draw permutes both layers again. State-only sets layer 2 to zero. Metro uses the unchanged '
         'interim 2026 sub-state rules (3-day threshold; 5 of 21 post-window days), metro coverage_ok, zero elsewhere. '
         'The assignment pools remain the full pools, including states/clusters without outcome rows.', '',
         '**Blind rule.** Only permuted pseudo-assignments meet vote swings in estimation. '
         'Actual-assignment variation and influence statistics use treatment and weights only, never outcomes. '
         'There is no actual-assignment estimate in these outputs.', '',
         '**Outcomes.** House units are county × district parts, keyed by race_id and stored unit_id '
         '(unit_id alone repeats across districts). Race IDs are HOUSE_state_district_year. '
         'Reference classification must be contested (exactly one non-write-in D and one R nominee), '
         'district_in_reference true, reconciliation_status exactly pass, and acs_match true with a non-null geo_id. '
         'Alaska is excluded. Fusion lines are already assigned to candidates in the input; votes are summed by candidate '
         'and then D/R within unit, excluding write-ins and minor-party candidates. '
         'r2 = 100 × R/(D+R), matching validate._r2; dv subtracts the county presidential r2 in 2020 for 2022 '
         'and 2016 for 2018. Zero-vote units and unavailable/nonpositive baselines are excluded.', '',
         '**County universe.** House includes all otherwise eligible counties, including states without an included '
         'statewide race. The eligible-universe filter requires a non-null outcome-cycle burden, exactly as validate.frame; '
         'it does not require membership in S. All retained rows have C1. The state additions below are relative to S '
         'in the same outcome year; they are the states contributed only by House.', '',
         '**Weights.** H allocates V_c, the baseline presidential D+R vote, across a county’s included House units '
         'in proportion to their House D+R votes (denominator uses included units only). S is unchanged. '
         'In S+H, each statewide race has raw weight V_c, the entire House block has raw weight V_c, '
         'and all are divided by the county’s number of included statewide races plus one if House is present. '
         'Thus the county’s total stacked weight is V_c; counties without House retain their statewide allocation. '
         'Each House district and each statewide race has its own fixed effect.', '',
         '**Controls.** County exposure, oil/gas employment (missing filled with zero), and C1 presidential swing '
         'follow validate.frame; assignment and source-cycle controls follow sub_frame. '
         'Controls are race FE, exposure × PADD, μ^pt and lagged burden when available, oil/gas employment, and C1. '
         'Repeated counties receive identical county controls.', '',
         '**Units.** β is vote percentage points per percentage point of monthly income in fuel burden. '
         f'Vote points per extra $10/month = β × 1000 / {run2026.monthly_income():,.0f}, the registered monthly income '
         '(analysis.yaml, translation.monthly_income). Negative MDEs retain their sign. '
         'Expected 90% CI half-width is the median across pseudo-assignments, in β units.', '',
         '## Filter audit', '',
         _md([[str(r['outcome']), r['filter'], str(r['districts']), str(r['units']), str(r['states'])] for r in audit],
             ['Year', 'Sequential filter', 'Districts kept', 'Units kept', 'States kept']), '',
         'Input classifications observed: contested, uncontested, other, null; reconciliation statuses: pass, fail, '
         'not_in_reference. Flags include excluded_ak, unassigned_county, district_not_in_reference, '
         'district_reconciliation_problem, CT/county recodes, masked_rows, and rcv_first_round_source_unverified. '
         'No blanket flag exclusion is applied: geographic recodes, masked observed sums and unverified Maine RCV '
         'first-round rows remain if they pass the stated filters. A district pass verifies totals within the source’s '
         '0.5% reconciliation tolerance, not the county allocation. Missing or masked votes are not imputed.', '',
         '## Sample reach', '',
         _md([[str(r['outcome']), r['frame'], str(r['n_counties']), str(r['n_states_in_sample']),
               str(r['n_states_house_only']), r['states_house_only'] or 'none']
              for r in tab.select('outcome', 'frame', 'n_counties', 'n_states_in_sample',
                                  'n_states_house_only', 'states_house_only').unique(maintain_order=True).to_dicts()],
             ['Year', 'Frame', 'Counties', 'States', 'States added only by House', 'Added states']), '',
         '## Results', '']
    for year in SUB_OUTCOMES:
        p += [f'### {year} swings', '', _md([
            [r['measure'], r['level'], r['frame'], str(r['n_rows']), str(r['n_races']), str(r['n_states_in_sample']),
             _f(r['size_at_zero'], 3), _pair(r['mde80_pos'], r['mde80_neg']),
             _pair(r['mde80_pos_per_10_dollars'], r['mde80_neg_per_10_dollars']),
             _f(r['ci90_halfwidth_median']), _f(r['n_eff_influence'], 1),
             ', '.join(f"{r[f'top{i}_state']} {r[f'top{i}_share']:.1%}" for i in (1, 2, 3)),
             r['controls_omitted'] or 'none'] for r in tab.filter(pl.col('outcome') == year).to_dicts()],
            ['Measure', 'Level', 'Frame', 'Rows', 'Races', 'States', 'Size at β=0', 'MDE80 β + / −',
             'MDE80 vote points per $10 + / −', '90% CI half-width β', 'Effective influential states',
             'Top-3 state shares', 'Controls omitted']), '']
    p += ['Influence shares are state shares of weighted within-race actual B̃ variation; '
          'effective influential states = 1 / Σ share². These are treatment-only diagnostics.', '',
          '## S-row reproduction', '', reproduction, '', '## What it means', '']
    ratios = []
    for s in tab.filter(pl.col('frame') == 'S').to_dicts():
        row = [s['measure'], s['level'], str(s['outcome'])]
        for label in ('H', 'S+H'):
            h = tab.filter((pl.col('measure') == s['measure']) & (pl.col('level') == s['level']) &
                           (pl.col('outcome') == s['outcome']) & (pl.col('frame') == label)).row(0, named=True)
            row.append(_pair(h['mde80_pos'] / s['mde80_pos'], h['mde80_neg'] / s['mde80_neg']))
        ratios.append(row)
    p += [_md(ratios, ['Measure', 'Level', 'Year', 'H / S MDE ratio + / −', 'S+H / S MDE ratio + / −']), '',
          f"Null rejection rates range from {tab['size_at_zero'].min():.3f} to {tab['size_at_zero'].max():.3f}. "
          'Ratios below one mean a smaller MDE than S, above one a larger MDE. '
          'Monte Carlo SE is about 0.01 near size 0.05 and 0.018 near power 0.8 at J=500; '
          'MDE differences under about 10% are noise. H covers only districts that pass reconciliation; '
          'repaired data could yield different results. These historical outcome samples do not establish 2026 House coverage.', '']
    out.write_text('\n'.join(p))


def house_main() -> None:
    t0 = time.time()
    tab, audit = house_table()
    reproduction = house_reproduction(tab)
    out_dir = paths.repo_root() / 'reports' / 'validation'
    tab.with_columns(pl.col('outcome').cast(pl.Int32)).write_csv(out_dir / 'T5_house.csv')
    write_house_report(tab, audit, reproduction, out_dir / 'T5_house.md')
    runlog.append_run_log(
        run_id=uuid.uuid4().hex[:12], utc_timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        git_sha=runlog.git_sha(), git_dirty=runlog.git_dirty(), config_hash='', spec_id='T5_house',
        data_snapshot_ids='house_county.parquet;exposure_county.parquet;shocks.parquet;shocks_r9a.parquet;'
                          'substate_shocks.parquet;results_county.parquet;interim substate inputs',
        output_path='reports/validation/T5_house.csv;reports/validation/T5_house.md',
        notes=f'J={J_OUTER}; K={K_INNER}; seed={SEED}; C1; pseudo_2026; w21/sa; state/metro; '
              f'2022/2018; S/H/S+H; all eligible House counties including additional states; '
              f'permuted assignments only; {reproduction}; runtime {time.time() - t0:.0f}s')
    print(reproduction, flush=True)


def main() -> None:
    if sys.argv[1:] == ['house']:
        house_main()
        return
    if sys.argv[1:] == ['substate']:
        substate_main()
        return
    if sys.argv[1:] == ['groupings']:
        out = paths.repo_root() / 'reports' / 'validation' / 'T5_groupings.csv'
        table = compare_groupings()
        table.write_csv(out)
        print(table.select('source', 'grouping', 'n_groups', 'sd_dev_states', 'kept_share_within_race', 'n_eff',
                           'max_realized_influence', 'corr_bt_btlag', 'mde80_pos', 'mde80_neg', 'mde80_pos_per_sd',
                           'size_at_zero'))
        runlog.append_run_log(
            run_id=uuid.uuid4().hex[:12],
            utc_timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            git_sha=runlog.git_sha(), git_dirty=runlog.git_dirty(), config_hash='', spec_id='T5_groupings',
            data_snapshot_ids='exposure_county.parquet;shocks.parquet;results_county.parquet',
            output_path=str(out.relative_to(paths.repo_root())),
            notes=f'groupings={GROUPINGS}; C0; J={J_OUTER}; K={K_INNER}; seed={SEED}; no V_t estimates')
        return
    curves, summaries = [], []
    for source in ('pseudo_2026', 'pseudo_2022'):
        for control_set in ('C0', 'C1', 'C2'):
            curve, summary = simulate(source, control_set)
            curves.append(curve)
            summaries.append(summary)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items()})
    out = paths.repo_root() / 'reports' / 'validation' / 'T5_power.csv'
    pl.concat(curves).join(pl.DataFrame(summaries), on=['source', 'control_set']).write_csv(out)
    runlog.append_run_log(
        run_id=uuid.uuid4().hex[:12], utc_timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        git_sha=runlog.git_sha(), git_dirty=runlog.git_dirty(), config_hash='', spec_id='T5_power',
        data_snapshot_ids='exposure_county.parquet;shocks.parquet;results_county.parquet',
        output_path=str(out.relative_to(paths.repo_root())),
        notes=f'J={J_OUTER}; K={K_INNER}; seed={SEED}; 2022 swings only; C0, C1, C2 (C3 needs RUCC, 2018); 2026 shocks provisional')


if __name__ == '__main__':
    main()
