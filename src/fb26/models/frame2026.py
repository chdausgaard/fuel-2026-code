"""2026 outcome frame, race/completeness sample rules and registered O6 assignments.

Inputs are supplied explicitly so tests never need election files. Shock pools
are built before outcome filtering; incomplete states remain in the RI pool.
"""
from dataclasses import asdict
import math

import numpy as np
import polars as pl
import yaml

from fb26.etl.exposure import with_market_oil
from fb26.etl.substate_shocks import Rules, ky_block, n_days
from fb26.models import power, validate
from fb26.util import paths


def level_config() -> dict:
    return yaml.safe_load((paths.config_dir() / 'analysis.yaml').read_text())['level']


def metro_lag_deviations(comparison: str | None = None) -> tuple[dict[str, float], int, int]:
    """Per-county 2020->2024 metro lag deviation, recentred within blocks or strata.

    Returns (geo_id -> deviation of the county's metro area, n_clusters_with_data, n_clusters_lacking). Keyed by county
    because the main design's clusters are county markets (K:...), which never equal the metro clusters here (M:...).
    Counties outside the w21_hist 2024 pool have no entry and get deviation 0 (state layer only).
    The validation cycles keep the state-only lag: their 2016 window predates
    the metro archive, so this function is only called for the 2026 cycle.
    """
    try:
        asg_lag = power.build_assignment('w21_hist', 2024, 'metro', comparison=comparison)
    except Exception:
        return {}, 0, 0
    devs = dict(zip(asg_lag.clusters['cluster'], asg_lag.clusters['dev'].to_list()))
    n_with = sum(1 for v in devs.values() if not (v is None or (isinstance(v, float) and np.isnan(v))))
    n_lack = len(devs) - n_with
    return {geo: devs.get(cluster) for geo, cluster in zip(asg_lag.counties['geo_id'], asg_lag.counties['cluster'])}, n_with, n_lack


def require_final(rows: pl.DataFrame, cause: str) -> None:
    if 'provisional' not in rows.columns or rows['provisional'].null_count() or rows['provisional'].any():
        raise ValueError('live guard: ' + cause)


def registered_rows(substate: pl.DataFrame, thresholds: dict | None = None,
                    measure: str = 'w21') -> pl.DataFrame:
    """Recheck the registered day thresholds; never accept interim coverage."""
    thresholds = level_config() if thresholds is None else thresholds
    rules = Rules(min_days_pre=thresholds['min_days'], min_days_post=thresholds.get('min_days_post', thresholds['min_days']))
    rows = substate.filter((pl.col('cycle') == 2026) & (pl.col('measure') == measure))
    pre, post = pl.lit(rules.min_days_pre), pl.lit(rules.min_days_post)
    if measure == 'sa':
        pre = pl.lit(rules.min_days_jan)
    elif measure == 'ky12':
        def minimum(year):
            days = n_days(ky_block(year))
            return (pl.when(pl.col('level') == 'metro').then(math.floor(rules.ky_share_metro * days))
                    .otherwise(math.ceil(rules.ky_share * days)))
        pre, post = minimum(2025), minimum(2026)
    return rows.with_columns((pl.col('coverage_ok').fill_null(False)
                              & (pl.col('n_days_pre') >= pre)
                              & (pl.col('n_days_post') >= post)).alias('coverage_ok'))


def measure_level(counties: pl.DataFrame, races: pl.DataFrame, substate: pl.DataFrame | None,
                  level: str, measure: str, thresholds: dict | None = None) -> str:
    """Retain the primary level only when this measure covers it; otherwise use state."""
    if level == 'state' or substate is None:
        return 'state'
    thresholds = level_config() if thresholds is None else thresholds
    states = races.filter(pl.col('proposed_in_sample') & (pl.col('state_fips') != '02'))['state_fips'].unique()
    universe = counties.filter(pl.col('state_fips').is_in(states.implode())).select('geo_id', 'state_fips').unique()
    covered = registered_rows(substate, thresholds, measure).filter(
        pl.col('coverage_ok') & pl.col('u_c').is_not_null() & pl.col('cluster_id').is_not_null())
    if level == 'metro':
        metros = covered.filter((pl.col('level') == 'metro') & pl.col('geo_id').is_in(universe['geo_id'].implode()))
        return level if set(states) <= set(metros['state_fips']) else 'state'
    ids = covered.filter(pl.col('level') == 'county')['geo_id'].unique()
    coverage = universe.with_columns(pl.col('geo_id').is_in(ids.implode()).alias('covered'))
    shares = coverage.group_by('state_fips').agg(pl.col('covered').mean().alias('share'))
    return level if (coverage.height and coverage['covered'].mean() >= thresholds['county_share']
                     and shares['share'].min() >= thresholds['state_county_share']
                     and set(states) == set(shares['state_fips'])) else 'state'


def build_assignment(shocks: pl.DataFrame, substate: pl.DataFrame | None, level: str,
                     mu_pt_source: str = 'seds', hf: pl.DataFrame | None = None,
                     thresholds: dict | None = None, measure: str = 'w21',
                     comparison: str | None = None,
                     source: str | None = None) -> power.Assignment:
    """All available states and registered covered clusters, independent of outcome completeness."""
    if mu_pt_source not in ('seds', 'hf_nodrift'):
        raise ValueError('mu_pt.source must be seds or hf_nodrift')
    if source is None:
        source = power.substate_source_default()
    sh = shocks.filter(pl.col('year') == 2026).sort('area_id')
    if mu_pt_source == 'hf_nodrift':
        if hf is None:
            raise ValueError('not_run: missing g_pt_hf.parquet for hf_nodrift')
        h = hf.filter((pl.col('year') == 2026) & (pl.col('measure') == measure))
        sh = sh.drop('g_pt').join(h.select('state_abbr', pl.col('g_pt_nodrift').alias('g_pt')),
                                on='state_abbr', how='left', validate='1:1')
    st = sh.select(pl.col('area_id').alias('state_fips'), 'state_abbr', 'padd_group', 'delta_p',
                   pl.col('deviation').alias('dev'), pl.col('lag_deviation').alias('lag_dev'), 'g_pt')
    st = st.filter(pl.col('dev').is_finite() & pl.col('delta_p').is_finite())
    if st.height == 0:
        raise ValueError('not_run: no current-cycle state shocks')
    use_market = source == 'county_markets' and level not in ('state', 'metro')
    if level == 'state':
        c = power.EMPTY_COUNTIES.clone()
    elif use_market:
        if substate is None:
            raise ValueError('not_run: missing substate_shocks.parquet')
        rows = registered_rows(substate, thresholds, measure).filter(pl.col('state_fips').is_in(st['state_fips'].implode()))
        c = power.market_counties(rows).sort('geo_id')
    else:
        if substate is None:
            raise ValueError('not_run: missing substate_shocks.parquet')
        rows = registered_rows(substate, thresholds, measure).filter(pl.col('state_fips').is_in(st['state_fips'].implode()))
        gs = rows.filter(pl.col('g_s').is_not_null()).select('state_fips', 'g_s').unique().join(st, on='state_fips')
        if gs.height and (gs['g_s'] - gs['delta_p']).abs().max() > 1e-9:
            raise ValueError('state shock differs from sub-state table')
        c = power.level_counties(rows, level).sort('geo_id')
    c = c.with_columns(pl.col('u_c').mean().over('cluster').alias('m'))
    clusters = (c.group_by('cluster').agg(pl.col('state_fips').first(), pl.col('stratum').first(),
                                        pl.col('stratum').n_unique().alias('n_strata'), pl.col('m').first())
                .sort('cluster'))
    if clusters.height and clusters['n_strata'].max() != 1:
        raise ValueError('cluster crosses assignment strata')
    clusters = clusters.drop('n_strata')
    if comparison is None:
        comparison = power._substate_config().get('comparison', 'strata')
    use_blocks = comparison == 'blocks' and clusters.height > 0
    if use_blocks:
        frozen = power._load_frozen_market_blocks() if use_market else power._load_frozen_blocks()
        if frozen is None:
            use_blocks = False
    if use_blocks:
        block_labels = power._assign_blocks_to_clusters(clusters, frozen)
        clusters = clusters.with_columns(pl.Series('block', block_labels.tolist()))
        clusters = clusters.with_columns((pl.col('m') - pl.col('m').mean().over('block')).alias('dev'))
        l2_groups = clusters['block'].to_numpy()
    else:
        clusters = clusters.with_columns((pl.col('m') - pl.col('m').mean().over('stratum')).alias('dev'))
        l2_groups = clusters['stratum'].to_numpy()
    index = clusters.select('cluster', pl.int_range(pl.len(), dtype=pl.Int64).alias('unit'))
    c = (c.with_columns((pl.col('u_c') - pl.col('m')).alias('resid')).join(index, on='cluster', how='left')
         .rename({'u_c': 'u'}).drop('m'))
    if use_blocks:
        l2_prefix = l2_groups.astype(str)
    else:
        l2_prefix = np.char.add('S:', l2_groups.astype(str))
    groups = np.concatenate([('P:' + st['padd_group']).to_numpy(), l2_prefix])
    return power.Assignment(st['state_fips'].to_list(), st['dev'].to_numpy(), clusters, c, groups,
                            np.concatenate([st['dev'].to_numpy(), clusters['dev'].to_numpy()]), st)


def select_races(results: pl.DataFrame, races: pl.DataFrame, mock: bool = False) -> tuple[list[str], list[dict]]:
    proposed = set(races.filter(pl.col('proposed_in_sample') & (pl.col('state_fips') != '02'))['race_id'].to_list())
    if mock:
        return sorted(proposed), []
    if 'candidate' not in results.columns:
        raise ValueError('not_run: live race rule requires candidate identities in ballot results')
    ballot = results.filter(pl.col('party').is_in(['D', 'R']) & (pl.col('state_fips') != '02'))
    if 'writein' in ballot.columns:
        ballot = ballot.filter(~pl.col('writein').fill_null(False))
    if ballot['candidate'].null_count():
        raise ValueError('not_run: missing ballot candidate identity')
    counts = ballot.group_by('race_id', 'geo_id').agg(
        pl.col('candidate').filter(pl.col('party') == 'D').n_unique().alias('n_d'),
        pl.col('candidate').filter(pl.col('party') == 'R').n_unique().alias('n_r'))
    counts = counts.group_by('race_id').agg(pl.col('n_d').max(), pl.col('n_r').max())
    actual = set(counts.filter((pl.col('n_d') == 1) & (pl.col('n_r') == 1))['race_id'].to_list())
    disagreements = [{'race_id': r, 'proposed_in_sample': r in proposed, 'included_d3': r in actual}
                     for r in sorted(proposed ^ actual)]
    return sorted(actual), disagreements


def turnout_outcome(results: pl.DataFrame, history: pl.DataFrame, mock: bool = False) -> pl.DataFrame:
    """All-candidate race votes / all-candidate presidential votes, in percent.

    The stored presidential table is long by candidate (no total_votes field).
    MOCK recovers all 2022 candidates using its source race, since mock rows keep only D/R.
    """
    keys = ['geo_id', 'race_id']
    if mock:
        if 'mock_source_race_id' not in results.columns:
            return results.select(keys).unique().with_columns(pl.lit(None, dtype=pl.Float64).alias('turnout'))
        source = (history.filter(pl.col('year') == 2022).group_by(keys)
                  .agg(pl.col('votes').sum().alias('_race_votes')).rename({'race_id': 'mock_source_race_id'}))
        totals = results.select(*keys, 'mock_source_race_id').unique().join(
            source, on=['geo_id', 'mock_source_race_id'], how='left', validate='m:1')
    else:
        totals = results.group_by(keys).agg(pl.col('votes').sum().alias('_race_votes'))
    baseline = (history.filter((pl.col('year') == (2020 if mock else 2024)) & (pl.col('office') == 'PRESIDENT'))
                .group_by('geo_id').agg(pl.col('votes').sum().alias('_pres_votes')))
    return (totals.join(baseline, on='geo_id', how='left', validate='m:1')
            .select(*keys, pl.when(pl.col('_pres_votes') > 0)
                    .then(100 * pl.col('_race_votes') / pl.col('_pres_votes')).otherwise(None).alias('turnout')))


def frame(results: pl.DataFrame, history: pl.DataFrame, completeness: pl.DataFrame, races: pl.DataFrame,
          exposure: pl.DataFrame, shocks: pl.DataFrame, substate: pl.DataFrame | None, level: str, *,
          mock: bool = False, mu_pt_source: str = 'seds', hf: pl.DataFrame | None = None,
          pct_expected_counted: float | None = None, thresholds: dict | None = None,
          g_method: str = validate.PRIMARY, measure: str = 'w21',
          comparison: str | None = None, metro_lag: bool | None = None,
          source: str | None = None) -> tuple[pl.DataFrame, power.Assignment, dict]:
    """Literal 2026 controls (2024 baseline, 2020→24 C1), including for MOCK.

    metro_lag: add the metro layer to B̃^lag; None reads the registered setting (analysis.yaml, lag.metro_layer).
    source: 'county_markets' | 'metro' | None (config default)."""
    if metro_lag is None:
        metro_lag = power.metro_lag_default()
    if pct_expected_counted is None:
        pct_expected_counted = yaml.safe_load((paths.config_dir() / 'analysis.yaml').read_text())['completeness']['pct_expected_counted']
    thresholds = level_config() if thresholds is None else thresholds
    assignment = build_assignment(shocks, substate, level, mu_pt_source, hf, thresholds, measure, comparison, source=source)
    cur = results.filter(pl.col('year') == 2026)
    if 'office' in cur.columns:
        cur = cur.filter(pl.col('office').is_in(['SENATE', 'GOVERNOR']))
    available_races = cur.select('race_id').unique()
    keep, disagreements = select_races(cur, races, mock)
    cur = cur.filter(pl.col('race_id').is_in(keep) & (pl.col('state_fips') != '02'))
    present = cur.select('state_fips', 'race_id').unique()
    # An absent listed race cannot be judged from ballot rows; retain it for completeness.
    absent = races.filter(pl.col('proposed_in_sample') & (pl.col('state_fips') != '02')).select(
        'state_fips', 'race_id').join(available_races, on='race_id', how='anti')
    selected = pl.concat([present, absent]).unique()
    keep = sorted(selected['race_id'].unique().to_list())
    snapshot_col = 'snapshot_utc' if 'snapshot_utc' in results.columns else 'retrieved_at_utc'
    cutoff = results[snapshot_col].str.to_datetime(time_zone='UTC').max()
    tracker = (completeness.with_columns(pl.col('snapshot_utc').str.to_datetime(time_zone='UTC').alias('_time'))
               .filter(pl.col('_time') <= cutoff).sort('_time').unique('state_po', keep='last'))
    state_names = shocks.select(pl.col('area_id').alias('state_fips'), pl.col('state_abbr').alias('state_po')).unique()
    missing_states = set(absent['state_fips'].to_list())
    check = (selected.select('state_fips').unique().join(state_names, on='state_fips', how='left')
             .join(tracker, on='state_po', how='left', validate='m:1').sort('state_fips'))
    state_status = []
    for row in check.iter_rows(named=True):
        reasons = []
        if row['state_fips'] in missing_states:
            reasons.append('race absent from snapshot')
        if row['pct_expected_counted'] is None:
            reasons.append('missing tracker row at snapshot')
        elif not row['pct_expected_counted'] >= pct_expected_counted:
            reasons.append('expected vote counted below threshold')
        state_status.append({'state_fips': row['state_fips'], 'state_po': row['state_po'],
                             'tracker_snapshot_utc': row['snapshot_utc'], 'reasons': reasons})
    states_in = [r['state_fips'] for r in state_status if not r['reasons']]
    states_out = [r['state_fips'] for r in state_status if r['reasons']]
    base = validate._r2(history.filter((pl.col('year') == 2024) & (pl.col('office') == 'PRESIDENT')), ['geo_id'])
    old = validate._r2(history.filter((pl.col('year') == 2020) & (pl.col('office') == 'PRESIDENT')), ['geo_id'])
    baseline = base.select('geo_id', pl.col('r2').alias('r2_base'), pl.col('two_party').alias('v_c'))
    c1 = base.select('geo_id', 'r2').join(old.select('geo_id', pl.col('r2').alias('r2_old')), on='geo_id').select(
        'geo_id', (pl.col('r2') - pl.col('r2_old')).alias('c1_swing'))
    out = (validate._r2(cur, ['state_fips', 'geo_id', 'race_id'])
           .join(baseline, on='geo_id', how='left', validate='m:1')
           .join(turnout_outcome(cur, history, mock), on=['geo_id', 'race_id'], how='left', validate='1:1'))
    if mock and 'mock_source_race_id' in cur.columns:
        out = out.join(cur.select('geo_id', 'race_id', 'mock_source_race_id').unique(),
                       on=['geo_id', 'race_id'], how='left', validate='1:1')
    votes = out.unique(['state_fips', 'geo_id'])
    denom = votes['v_c'].sum()
    vote_share = votes.filter(pl.col('state_fips').is_in(states_in))['v_c'].sum() / denom if denom else 0.0
    if mock and 'mock_dv' in cur.columns:
        standin = cur.select('geo_id', 'race_id', 'mock_dv').unique()
        out = out.join(standin, on=['geo_id', 'race_id'], how='left', validate='1:1')
    out = out.filter(pl.col('state_fips').is_in(states_in))
    exp = with_market_oil(exposure.filter(pl.col('g_method') == g_method)).drop('state_fips')
    st = assignment.state_table.rename({'dev': 'dev_state'})
    exp = exp.drop(*[c for c in st.columns if c in exp.columns and c != 'geo_id'])
    units = assignment.counties.select('geo_id', 'unit', 'src', 'u',
                                       (pl.col('u') - pl.col('resid')).alias('m_c'))
    dev2 = assignment.clusters.select(pl.int_range(pl.len(), dtype=pl.Int64).alias('unit'), pl.col('dev').alias('dev2'))
    df = (out.join(exp, on='geo_id', how='left', validate='m:1').join(c1, on='geo_id', how='left', validate='m:1')
          .join(st, on='state_fips', how='left', validate='m:1').join(units, on='geo_id', how='left', validate='m:1')
          .join(dev2, on='unit', how='left', validate='m:1')
          .with_columns(pl.col('unit').fill_null(-1), pl.col('dev2').fill_null(0.0), pl.col('u').fill_null(0.0),
                        pl.col('src').fill_null('state'), pl.col('m_c').fill_null(0.0), pl.col('oil_gas_emp_share').fill_null(0.0)))
    controls = {'w21': ('c1_swing', 'g_pt', 'lag_dev'), 'sa': ('c1_swing', 'g_pt')}.get(measure, ('c1_swing',))
    if df.height and any(df[c].null_count() or not df[c].is_finite().all() for c in controls):
        raise ValueError('not_run: required current-cycle controls are missing or nonfinite')
    invalid = df.filter((pl.col('two_party') <= 0) | pl.any_horizontal(
        [pl.col(c).is_null() | ~pl.col(c).is_finite() for c in ('r2_base', 'e_c', 'dev_state', *controls)]))
    df = df.join(invalid.select('geo_id', 'race_id'), on=['geo_id', 'race_id'], how='anti')
    # Metro layer of the lagged-burden control (2026 only, w21 measure).
    # The validation cycles keep the state-only lag: their 2016 window predates the metro archive.
    lag_metro_dev = np.zeros(df.height)
    n_lag_clusters, n_lag_lacking = 0, 0
    if metro_lag and level != 'state':
        lag_devs, n_lag_clusters, n_lag_lacking = metro_lag_deviations(comparison)
        for i, geo in enumerate(df['geo_id'].to_list()):
            v = lag_devs.get(geo)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                lag_metro_dev[i] = v
    df = df.with_columns(pl.Series('_lag_metro_dev', lag_metro_dev))
    df = (df.with_columns(pl.col('race_id').n_unique().over('state_fips').alias('n_s'))
          .with_columns((pl.col('v_c') / pl.col('n_s')).alias('w'), (pl.col('mock_dv') if mock and 'mock_dv' in df.columns else pl.col('r2') - pl.col('r2_base')).alias('dv'),
                        (pl.col('e_c') * pl.col('dev2')).alias('b2'),
                        (pl.col('e_c') * (pl.col('dev_state') + pl.col('dev2'))).alias('b_tilde'),
                        (pl.col('e_c') * (pl.col('delta_p') + pl.col('u'))).alias('b'),
                        (pl.col('e_c') * (pl.col('delta_p') + pl.col('m_c'))).alias('b_raw'),
                        (pl.col('e_c') * (pl.col('lag_dev') + pl.col('_lag_metro_dev'))).alias('b_tilde_lag'),
                        (pl.col('e_c') * pl.col('g_pt')).alias('mu_pt'), pl.col('dev_state').alias('deviation'))
          .drop('_lag_metro_dev').sort('race_id', 'geo_id'))
    info = {'states_in': states_in, 'states_out': states_out, 'state_status': state_status, 'baseline_vote_share': vote_share,
            'race_disagreements': disagreements, 'included_races': keep,
            'included_states': selected['state_fips'].unique().sort().to_list(), 'excluded_rows': invalid.height,
            'excluded_reason': 'not_run: incomplete baseline, exposure, controls or shock' if invalid.height else '',
            'mu_pt_source': mu_pt_source, 'measure': measure,
            'omitted_controls': [c for c in ('mu_pt', 'b_tilde_lag') if df[c].null_count()],
            'rules': asdict(Rules(min_days_pre=thresholds['min_days'], min_days_post=thresholds.get('min_days_post', thresholds['min_days']))),
            'states_missing_shock': sorted(set(shocks.filter(pl.col('year') == 2026)['area_id'].to_list()) - set(assignment.states)),
            'metro_lag': bool(metro_lag), 'metro_lag_clusters': n_lag_clusters, 'metro_lag_clusters_lacking': n_lag_lacking}
    return df, assignment, info
