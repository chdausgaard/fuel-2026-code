"""T4a placebo (memo §8): past swings regressed on the 2026 recentered burden.

B̃_2026 cannot cause swings that happened before 2026. A coefficient far from
zero means that states with high 2026 relative shocks already differed in how
their high- and low-exposure counties swung: a balance failure for that
grouping.

State-level placebos: outcomes include presidential swings and midterm swings.
RI permutes within PADD groups (or other groupings). Controls: C1.

Metro-level placebos: two-layer RI, forms F1/F5/F6, measures w21/sa, both
comparisons (blocks/strata). The outcome list is extensible.

Twelve-election battery (registered check): the actual 2026 B̃ regressed on the six midterm
swings (2002-2022) and the six presidential swings (2000→04 to 2020→24), at the level the D22
rule picks. Main design: the registered comparison (blocks), F1, w21, and on every row the C1
analog: the presidential swing into the election the row is measured from. Also strata (diagnostic), C2 and F5 (robustness). Calibration summary and chart
data come from the table of rows.

    python -m fb26.models.placebo
    python -m fb26.models.placebo metro [comparison]
    python -m fb26.models.placebo battery [comparison] [--write]
"""
import contextlib
import functools
import uuid
from datetime import datetime, timezone

import numpy as np
import polars as pl

from fb26.etl import results_hist, results_overrides
from fb26.etl.exposure import with_market_oil
from fb26.models import burden, p1, power, ri, validate
from fb26.util import paths, runlog

OUTCOMES = ('pres_2016_2020', 'pres_2020_2024', 'mid_2018', 'mid_2022')

# Metro placebo outcomes (extensible list; 2002-2014 added in B2)
METRO_OUTCOMES = ('mid_2018', 'mid_2022', 'pres_2016_2020', 'pres_2020_2024', 'pres_2012_2016')
METRO_MEASURES = ('w21', 'sa')
METRO_FORMS = ('F1', 'F5', 'F6')
METRO_SAMPLES = {'mid_2022': ('full', 'common'), 'mid_2018': ('all',),
                 'pres_2016_2020': ('all',), 'pres_2020_2024': ('all',),
                 'pres_2012_2016': ('all',)}


# ---------------------------------------------------------------- state-level placebos (original T4a)

def frame(outcome: str, grouping: str) -> tuple[pl.DataFrame, list[str], np.ndarray, np.ndarray]:
    if outcome.startswith('mid_'):
        return power.frame('pseudo_2026', grouping, int(outcome.split('_')[1]))
    start, end = (int(y) for y in outcome.split('_')[1:])
    sw = validate.pres_swing(start, end)
    df = (burden.build(2026, validate.PRIMARY)
          .join(sw, on='geo_id', how='inner')
          .filter((pl.col('state_fips') != '02') & pl.col('e_c').is_not_null())
          .with_columns(('PRES_' + pl.col('state_fips')).alias('race_id'),
                        pl.col('swing').alias('dv'), pl.col('v_start').alias('w'),
                        pl.col('oil_gas_emp_share').fill_null(0.0))
          .sort('race_id', 'geo_id'))
    return power.regroup(with_market_oil(df), 2026, grouping)


def estimate(outcome: str, grouping: str, k: int = ri.K_DRAWS) -> dict:
    df, states, dev, groups = frame(outcome, grouping)
    z = validate.design_matrix(df)
    y, x, w = df['dv'].to_numpy(), df['b_tilde'].to_numpy(), df['w'].to_numpy()
    fit = p1.fit(y, x, z, w, df['state_fips'].to_numpy())
    state = df['state_fips'].replace_strict({s: i for i, s in enumerate(states)}).to_numpy()
    test = ri.RandomizationTest(y, df['e_c'].to_numpy(), state, z, w, df['state_fips'].to_numpy(),
                                dev, groups, k=k)
    r = test.run(fit.beta, fit.se_cr1)
    sd = power._within_race_sd(df, x)
    return {'outcome': outcome, 'grouping': grouping, 'beta': fit.beta, 'se_cr1': fit.se_cr1, 't_cr1': fit.t,
            'p_ri': r.p_value, 'ci95_lo': r.ci95[0], 'ci95_hi': r.ci95[1], 'ci90_lo': r.ci90[0],
            'ci90_hi': r.ci90[1], 'eps_star': r.eps_star, 'sd_bt_within_race': sd,
            'beta_per_sd': fit.beta * sd, 'n_rows': fit.n, 'n_states': fit.n_clusters,
            'k_draws': r.k_draws, 'seed': r.seed, 'ri_flags': ';'.join(r.flags)}


# ---------------------------------------------------------------- metro-level placebos

def _build_extras(outcome_year: int) -> pl.DataFrame:
    """Per (geo_id, race_id): r2, r2_base and, for 2022, mapped 2018 race."""
    res = pl.read_parquet(paths.processed_dir() / 'results_county.parquet')
    out, _ = validate.outcomes(outcome_year)
    base = out.select('geo_id', 'race_id', 'state_fips', 'r2', 'r2_base')
    if outcome_year != 2022:
        return base
    # Map each 2022 race to a 2018 race
    src = res.filter((pl.col('year') == 2018) & pl.col('office').is_in(['SENATE', 'GOVERNOR'])
                     & pl.col('party').is_in(['D', 'R']) & pl.col('included_d3').fill_null(False))
    races = (pl.read_csv(paths.repo_root() / 'reports' / 'validation' / 'races_2022.csv',
                         schema_overrides={'state_fips': pl.String})
             .filter(pl.col('proposed_in_sample')).select('race_id', 'state_fips', 'office'))
    maps = []
    for r in races.iter_rows(named=True):
        cand = src.filter(pl.col('state_fips') == r['state_fips']).select('race_id', 'office', 'special').unique().sort('race_id')
        same = cand.filter(pl.col('office') == r['office'])
        opts = same if same.height else cand
        if opts.is_empty():
            maps.append({'race_id': r['race_id'], 'prev_race_id': None})
            continue
        reg = opts.filter(~pl.col('special').fill_null(False))
        opts = reg if reg.height else opts
        maps.append({'race_id': r['race_id'], 'prev_race_id': opts['race_id'][0]})
    maps = pl.DataFrame(maps, schema={'race_id': pl.String, 'prev_race_id': pl.String})
    prev = (validate._r2(src, ['geo_id', 'race_id']).filter(pl.col('two_party') > 0)
            .select('geo_id', pl.col('race_id').alias('prev_race_id'), pl.col('r2').alias('r2_prev')))
    return (base.join(maps, on='race_id', how='left', validate='m:1')
            .join(prev, on=['geo_id', 'prev_race_id'], how='left', validate='m:1')
            .drop('state_fips'))


def _race_slopes(df: pl.DataFrame, col: str) -> list[np.ndarray]:
    race = df['race_id'].to_numpy()
    x = df[col].to_numpy()
    return [x * (race == r) for r in np.unique(race)]


def apply_form(df: pl.DataFrame, form_name: str, outcome_year: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Outcome vector and extra control columns for a metro placebo form."""
    if form_name == 'F1':
        return df['dv'].to_numpy(), []
    if form_name == 'F5':
        return df['r2'].to_numpy(), _race_slopes(df, 'r2_base')
    if form_name == 'F6':
        return df['r2'].to_numpy(), _race_slopes(df, 'r2_base') + _race_slopes(df, 'r2_prev')
    raise ValueError(form_name)


def metro_estimate(outcome: str, measure: str, form_name: str, sample: str,
                   comparison: str, k: int = ri.K_DRAWS, seed: int = ri.SEED,
                   metro_lag: bool | None = None,
                   source: str | None = None) -> dict:
    """One metro-level placebo cell: two-layer RI on past swings."""
    outcome_year = int(outcome.split('_')[-1]) if outcome.startswith('mid_') else int(outcome.rsplit('_', 1)[-1])
    df, asg, omitted = power.sub_frame(outcome_year, measure, 2026, 'metro',
                                       comparison=comparison, metro_lag=metro_lag,
                                       source=source or 'metro')
    # Attach extras (r2, r2_base, r2_prev for F5/F6)
    ex = _build_extras(outcome_year)
    keep = ['geo_id', 'race_id', 'r2', 'r2_base']
    if outcome_year == 2022:
        keep.append('r2_prev')
    d = df.drop(*[c for c in keep if c in df.columns and c not in ('geo_id', 'race_id')])
    d = d.join(ex.select(keep), on=['geo_id', 'race_id'], how='left', validate='1:1')
    assert d.height == df.height
    if sample == 'common':
        d = d.filter(pl.col('r2_prev').is_not_null())
    d = d.sort('race_id', 'geo_id')
    y, extra = apply_form(d, form_name, outcome_year)
    assert not np.isnan(y).any()
    z = validate.design_matrix(d, precision=validate.C1)
    if extra:
        z = np.column_stack([z] + extra)
    fit = _fit_ri(d, asg, y, z, k, seed)
    return {'outcome': outcome, 'measure': measure, 'form': form_name, 'sample': sample,
            'comparison': comparison, 'level': 'metro', **fit, 'controls_omitted': ';'.join(omitted)}


def _fit_ri(d: pl.DataFrame, asg, y: np.ndarray, z: np.ndarray, k: int, seed: int) -> dict:
    """P1 fit and two-layer RI of B̃ on `y` given the controls `z`; `asg` is the treatment assignment pool.

    Works at every level: a state-level assignment has no clusters and the test reduces to the state layer."""
    x, e, w = d['b_tilde'].to_numpy(), d['e_c'].to_numpy(), d['w'].to_numpy()
    clusters = d['state_fips'].to_numpy()
    states = asg.states
    index = {s: i for i, s in enumerate(states)}
    state = np.array([index[s] for s in clusters])
    unit = d['unit'].to_numpy()
    expected = e * asg.dev[state]
    has = unit >= 0
    expected[has] += e[has] * asg.dev[len(states) + unit[has]]
    assert np.allclose(x, expected)
    fit = p1.fit(y, x, z, w, clusters)
    test = ri.TwoLayerRandomizationTest(y, e, state, z, w, clusters, asg.dev, asg.groups,
                                        len(states), unit, k=k, seed=seed)
    r = test.run(fit.beta, fit.se_cr1)
    sd = power._within_race_sd(d, x)
    return {'beta': fit.beta, 'se_cr1': fit.se_cr1, 't_cr1': fit.t,
            'p_ri': r.p_value, 'ci95_lo': r.ci95[0], 'ci95_hi': r.ci95[1],
            'ci90_lo': r.ci90[0], 'ci90_hi': r.ci90[1],
            'sd_bt_within_race': sd, 'beta_per_sd': fit.beta * sd,
            'n_rows': fit.n, 'n_races': int(d['race_id'].n_unique()), 'n_states': fit.n_clusters,
            'n_params': fit.n_params, 'k_draws': r.k_draws, 'seed': r.seed, 'ri_flags': ';'.join(r.flags)}


# ---------------------------------------------------------------- twelve-election battery (registered check)

# Sign of each election's president: I = +1 under a Democratic president, so I·β is the effect against the sitting
# president's party. Midterms are keyed by year; presidential swings by (start, end) with the president elected at start.
I_MIDTERM = {2002: -1, 2006: -1, 2010: +1, 2014: +1, 2018: -1, 2022: +1}
I_PRESIDENTIAL = {(2000, 2004): -1, (2004, 2008): -1, (2008, 2012): +1, (2012, 2016): +1, (2016, 2020): -1,
                  (2020, 2024): +1}
BASELINE = {2002: 2000, 2006: 2004, 2010: 2008, 2014: 2012, 2018: 2016, 2022: 2020}   # previous presidential election
# Every row controls for the presidential swing into the election it is measured from, as the 2020 -> 2024 swing does for 2026
# (PI, 2026-10-31). C1_WINDOW: the midterms the project's own frames do not cover; PRIOR_WINDOW: the presidential swings.
C1_WINDOW = {2002: (1996, 2000), 2006: (2000, 2004), 2010: (2004, 2008), 2014: (2008, 2012)}
PRIOR_WINDOW = {(2000, 2004): (1996, 2000), (2004, 2008): (2000, 2004), (2008, 2012): (2004, 2008),
                (2012, 2016): (2008, 2012), (2016, 2020): (2012, 2016), (2020, 2024): (2016, 2020)}
DEMOGRAPHICS = validate.C2[1:]
BATTERY_FORMS = ('F1', 'C2', 'F5')
BATTERY = {**{f'mid_{t}': {'kind': 'midterm', 'year': t, 'sign': s, 'label': f'{t} midterm'}
              for t, s in I_MIDTERM.items()},
           **{f'pres_{a}_{b}': {'kind': 'presidential', 'start': a, 'end': b, 'sign': s,
                                'label': f'{a}→{str(b)[2:]} presidential'}
              for (a, b), s in I_PRESIDENTIAL.items()}}


@functools.cache
def _exposure() -> pl.DataFrame:
    return (pl.read_parquet(paths.processed_dir() / 'exposure_county.parquet')
            .filter(pl.col('g_method') == validate.PRIMARY)
            .select('geo_id', 'state_fips', 'e_c', 'oil_gas_emp_share', *DEMOGRAPHICS))


@functools.cache
def _presidential(year: int) -> pl.DataFrame:
    """County two-party Republican share (points) and two-party votes of a presidential election. The project's county
    file starts in 2000; 1996, needed only for the 1996 -> 2000 swing, comes from the Amlani and Algara county file."""
    if year == 1996:
        return (results_overrides.aa_presidential(year)
                .with_columns((100 * pl.col('r') / (pl.col('r') + pl.col('d'))).alias('r2'), (pl.col('r') + pl.col('d')).alias('two_party'))
                .select('geo_id', 'r2', 'two_party'))
    res = pl.read_parquet(paths.processed_dir() / 'results_county.parquet')
    return (validate._r2(res.filter((pl.col('year') == year) & (pl.col('office') == 'PRESIDENT')), ['geo_id'])
            .select('geo_id', 'r2', 'two_party'))


def presidential_swing(start: int, end: int) -> pl.DataFrame:
    """R2(end) − R2(start) with the start year's votes; counties missing in either year (CT before 2012) drop out."""
    a, b = _presidential(start), _presidential(end)
    return (a.select('geo_id', pl.col('r2').alias('r2_start'), pl.col('two_party').alias('v_start'))
            .join(b.select('geo_id', pl.col('r2').alias('r2_end')), on='geo_id', how='inner')
            .with_columns((pl.col('r2_end') - pl.col('r2_start')).alias('swing')))


def historical_midterm_rows(year: int, hist: pl.DataFrame, exposure: pl.DataFrame, c1: pl.DataFrame) -> pl.DataFrame:
    """validate.frame's analysis rows for a 2002-2014 midterm: the usable county × race rows of the historical file
    against the previous presidential election (w = baseline two-party votes ÷ races in the state), with C1 from the
    prior presidential swing. Counties without it drop out."""
    cur = results_hist.two_party(hist.filter(pl.col('year') == year)).select('state_fips', 'geo_id', 'race_id', 'r2')
    base = _presidential(BASELINE[year]).select('geo_id', pl.col('r2').alias('r2_base'), pl.col('two_party').alias('v_c'))
    df = (cur.join(base, on='geo_id', how='left', validate='m:1')
          .with_columns((pl.col('r2') - pl.col('r2_base')).alias('dv'), pl.col('race_id').n_unique().over('state_fips').alias('n_s'))
          .with_columns((pl.col('v_c') / pl.col('n_s')).alias('w'))
          .filter(pl.col('r2_base').is_not_null()))
    df = (df.join(exposure.drop('state_fips'), on='geo_id', how='inner', validate='m:1')
          .join(c1.select('geo_id', pl.col('swing').alias('c1_swing')), on='geo_id', how='left', validate='m:1'))
    df = df.filter(pl.col('e_c').is_not_null() & pl.col('c1_swing').is_not_null() & (pl.col('state_fips') != '02'))
    return df.with_columns(pl.col('oil_gas_emp_share').fill_null(0.0)).sort('race_id', 'geo_id')


def presidential_rows(start: int, end: int, exposure: pl.DataFrame) -> pl.DataFrame:
    """Presidential swing rows (race = the state, weight = start-year two-party votes), with the prior window's swing
    as c1_swing. Counties without it drop out."""
    df = (exposure.select('geo_id', 'state_fips', 'e_c', 'oil_gas_emp_share', *DEMOGRAPHICS)
          .join(presidential_swing(start, end), on='geo_id', how='inner')
          .filter((pl.col('state_fips') != '02') & pl.col('e_c').is_not_null())
          .with_columns(('PRES_' + pl.col('state_fips')).alias('race_id'), pl.col('swing').alias('dv'),
                        pl.col('v_start').alias('w'), pl.col('oil_gas_emp_share').fill_null(0.0)))
    prior = PRIOR_WINDOW[(start, end)]
    df = (df.join(presidential_swing(*prior).select('geo_id', pl.col('swing').alias('c1_swing')), on='geo_id',
                  how='left', validate='m:1')
          .filter(pl.col('c1_swing').is_not_null()))
    return df.sort('race_id', 'geo_id')


def outcome_rows(outcome: str, form: str = 'F1') -> tuple[pl.DataFrame, bool]:
    """Outcome rows of a battery election for a form, and whether the form's design has a C1 control.

    Columns: the ones power.sub_frame selects, plus r2, r2_base (F5) and the C2 covariates. Every row has its C1 control.
    Midterms use the project's own frames for 2018 and 2022 and the historical file before."""
    spec = BATTERY[outcome]
    exposure = _exposure()
    if spec['kind'] == 'presidential':
        start, end = spec['start'], spec['end']
        df = presidential_rows(start, end, exposure)
        df = df.with_columns(pl.col('r2_start').alias('r2_base'), pl.col('r2_end').alias('r2'))
        return df, True
    year = spec['year']
    if year in (2018, 2022):
        df, _, _ = validate.frame(year, validate.PRIMARY)
        return df.sort('race_id', 'geo_id'), True
    return historical_midterm_rows(year, results_hist.load(), exposure, presidential_swing(*C1_WINDOW[year])), True


@contextlib.contextmanager
def _outcome_rows(df: pl.DataFrame):
    """power.sub_frame takes its outcome rows from validate.frame; serve it a prepared frame (this process only)."""
    original = validate.frame
    validate.frame = lambda cycle, g_method: (df, None, [])
    try:
        yield
    finally:
        validate.frame = original


def battery_design(d: pl.DataFrame, outcome: str, form: str, has_c1: bool) -> tuple[np.ndarray, np.ndarray]:
    """Outcome vector and control matrix of a battery form.

    F1: the swing, controls C1. C2: the swing, C1 plus the county covariates. F5: the Republican share in the race (presidential: end-year share), a slope per race on the
    previous presidential share (presidential: start-year share), C1. `has_c1` is true for every battery row since
    2026-10-31; a design without C1 stays possible for a caller that passes False."""
    if form not in BATTERY_FORMS:
        raise ValueError(form)
    if form == 'C2':
        precision = validate.C2 if has_c1 else DEMOGRAPHICS
        missing = {c: d[c].null_count() for c in DEMOGRAPHICS if d[c].null_count()}
        assert not missing, f'missing C2 covariates {missing}: not changing the sample silently'
    else:
        precision = validate.C1 if has_c1 else ()
    z = validate.design_matrix(d, precision=precision)
    if form == 'F5':
        assert d['r2'].null_count() == 0 and d['r2_base'].null_count() == 0
        return d['r2'].to_numpy(), np.column_stack([z] + _race_slopes(d, 'r2_base'))
    return d['dv'].to_numpy(), z


def battery_estimate(outcome: str, form: str = 'F1', comparison: str | None = None, *, measure: str = 'w21',
                     level: str = 'metro', k: int = ri.K_DRAWS, seed: int = ri.SEED, metro_lag: bool | None = None,
                     source: str | None = None,
                     layer2_override: pl.DataFrame | None = None) -> dict:
    """One battery cell: the actual 2026 B̃ at `level` on a past swing, two-layer RI under `comparison`.

    metro_lag: None reads the registered setting (analysis.yaml, lag.metro_layer: the state-only lag).
    source: 'county_markets' | 'metro' | None (config default).
    layer2_override: pre-computed sub-state rows (custom treatment windows)."""
    if metro_lag is None:
        metro_lag = power.metro_lag_default()
    spec = BATTERY[outcome]
    rows, has_c1 = outcome_rows(outcome, form)
    with _outcome_rows(rows):
        df, asg, omitted = power.sub_frame(spec.get('year', spec.get('end')), measure, 2026, level,
                                           comparison=comparison, metro_lag=metro_lag, source=source,
                                           layer2_override=layer2_override)
    extra = [c for c in ('r2', 'r2_base', *DEMOGRAPHICS) if c in rows.columns]
    d = (df.drop(*[c for c in extra if c in df.columns])
         .join(rows.select('geo_id', 'race_id', *extra), on=['geo_id', 'race_id'], how='left', validate='1:1')
         .sort('race_id', 'geo_id'))
    assert d.height == df.height
    y, z = battery_design(d, outcome, form, has_c1)
    fit = _fit_ri(d, asg, y, z, k, seed)
    comparison = comparison or power._substate_config().get('comparison', 'strata')
    return {'outcome': outcome, 'label': spec['label'], 'kind': spec['kind'], 'form': form, 'comparison': comparison,
            'measure': measure, 'level': level, 'metro_lag': bool(metro_lag), 'sign': spec['sign'], 'c1_control': has_c1, **fit,
            'beta_signed': spec['sign'] * fit['beta'], 't_signed': spec['sign'] * fit['t_cr1'],
            'controls_omitted': ';'.join(omitted)}


def battery(forms=('F1',), comparisons=('blocks',), *, level: str = 'metro', measure: str = 'w21', k: int = ri.K_DRAWS,
            seed: int = ri.SEED, metro_lag: bool | None = None, outcomes=tuple(BATTERY), progress: bool = False,
            source: str | None = None,
            layer2_override: pl.DataFrame | None = None) -> pl.DataFrame:
    """Rows of the battery: one per outcome × form × comparison."""
    rows = []
    for comparison in comparisons:
        for form in forms:
            for outcome in outcomes:
                rows.append(battery_estimate(outcome, form, comparison, measure=measure, level=level, k=k, seed=seed,
                                             metro_lag=metro_lag, source=source, layer2_override=layer2_override))
                if progress:
                    print(f"{outcome} {form} {comparison}: beta={rows[-1]['beta']:.4f} p={rows[-1]['p_ri']:.4f}", flush=True)
    return pl.DataFrame(rows, infer_schema_length=None)


def _summarize(t, se, p, beta, signed_beta, signed_t) -> dict:
    from scipy.stats import norm
    z_ri = norm.isf(np.asarray(p) / 2)
    n = len(t)
    return {'n': n, 'sum_t2': float(np.sum(t ** 2)), 'n_abs_t_gt_1_645': int(np.sum(np.abs(t) > 1.645)),
            'n_abs_t_gt_1_96': int(np.sum(np.abs(t) > 1.96)), 'mean_beta_raw': float(np.mean(beta)),
            'mean_beta_signed': float(np.mean(signed_beta)), 'sd_beta_raw': float(np.std(beta, ddof=1)),
            'sd_beta_signed': float(np.std(signed_beta, ddof=1)), 'mean_se': float(np.mean(se)),
            'sd_over_mean_se_raw': float(np.std(beta, ddof=1) / np.mean(se)),
            'sd_over_mean_se_signed': float(np.std(signed_beta, ddof=1) / np.mean(se)),
            'mean_t_signed': float(np.mean(signed_t)), 'sum_z2_from_ri_p': float(np.sum(z_ri ** 2)),
            'n_ri_p_le_05': int(np.sum(np.asarray(p) <= .05)), 'n_ri_p_le_10': int(np.sum(np.asarray(p) <= .10)),
            'n_signed_positive': int(np.sum(np.asarray(signed_beta) > 0)), 'min_p_ri': float(np.min(p))}


def calibration(table: pl.DataFrame) -> pl.DataFrame:
    """Calibration summary per form × comparison, for all outcomes and for the midterms and presidential swings alone.

    sum_t2: the sum of squared CR1 t (about n if the placebos are noise); counts of |t| above 1.645 and 1.96;
    sd_over_mean_se_*: the SD of β̂ across outcomes over the mean standard error (about 1 if the standard errors are
    right); the signed versions flip each β̂ by the sitting president's party (I)."""
    out = []
    for (form, comparison), group in table.group_by('form', 'comparison', maintain_order=True):
        for part, mask in (('all', pl.lit(True)), ('midterms', pl.col('kind') == 'midterm'),
                           ('presidential', pl.col('kind') == 'presidential')):
            d = group.filter(mask)
            if d.height < 2:
                continue
            out.append({'form': form, 'comparison': comparison, 'part': part,
                        **_summarize(d['t_cr1'].to_numpy(), d['se_cr1'].to_numpy(), d['p_ri'].to_numpy(),
                                     d['beta'].to_numpy(), d['beta_signed'].to_numpy(), d['t_signed'].to_numpy())})
    return pl.DataFrame(out)


def chart_rows(table: pl.DataFrame, income: float | None = None) -> pl.DataFrame:
    """One row per outcome of the main design (blocks, F1) for the website and the registration chart: β, its RI
    95% interval, t and RI p, in points per point of monthly income and in points per $10 a month."""
    from fb26.models import run2026
    income = run2026.monthly_income() if income is None else income
    d = table.filter((pl.col('form') == 'F1') & (pl.col('comparison') == 'blocks'))
    if d.is_empty():
        d = table.filter(pl.col('form') == 'F1')
    scale = run2026.per_10_dollars(1.0, income)
    return (d.select('outcome', 'label', 'kind', 'sign', 'beta', 'ci95_lo', 'ci95_hi', 't_cr1', 'p_ri', 'n_rows')
            .with_columns(*[(pl.col(c) * scale).alias(c + '_per_10_dollars') for c in ('beta', 'ci95_lo', 'ci95_hi')]))


def default_level() -> str:
    """The D22 level of the recorded decision (reports/2026/treatment_level_2026.json), else the registered default, metro."""
    import json
    import yaml
    config = yaml.safe_load((paths.config_dir() / 'analysis.yaml').read_text())
    path = paths.repo_root() / config['outputs']['level_file']
    if path.is_file():
        level = json.loads(path.read_text())
        if not level.get('forced', False) and level.get('level') in power.SUB_LEVELS:
            return level['level']
    return 'metro'


# ---------------------------------------------------------------- CLI

def main() -> None:
    import sys
    if sys.argv[1:] and sys.argv[1] == 'metro':
        comparison = sys.argv[2] if len(sys.argv) > 2 else None
        metro_main(comparison)
        return
    if sys.argv[1:] and sys.argv[1] == 'battery':
        if '--window' in sys.argv:
            import argparse
            from fb26.models.power_geo import battery_window_main
            parser = argparse.ArgumentParser(description='Production battery by treatment window')
            parser.add_argument('--window', choices=('S1', 'actual', 'both'), required=True)
            parser.add_argument('--actual-min-days', type=int)
            parser.add_argument('--write', action='store_true')
            args = parser.parse_args(sys.argv[2:])
            battery_window_main(args.window, write=args.write, actual_min_days=args.actual_min_days)
            return
        args = [a for a in sys.argv[2:] if not a.startswith('--')]
        battery_main(args[0] if args else None, write='--write' in sys.argv)
        return
    table = pl.DataFrame([estimate(o, g) for o in OUTCOMES for g in power.GROUPINGS])
    out = paths.repo_root() / 'reports' / 'validation' / 'T4a_placebo.csv'
    table.write_csv(out)
    runlog.append_run_log(
        run_id=uuid.uuid4().hex[:12], utc_timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        git_sha=runlog.git_sha(), git_dirty=runlog.git_dirty(), config_hash='', spec_id='T4a_placebo',
        data_snapshot_ids='exposure_county.parquet;shocks.parquet;results_county.parquet',
        output_path=str(out.relative_to(paths.repo_root())),
        notes=f'groupings={power.GROUPINGS}; outcomes={OUTCOMES}; K={ri.K_DRAWS}; 2026 shocks provisional (5 days)')
    print(table.select('outcome', 'grouping', 'beta', 'beta_per_sd', 'p_ri', 'ci95_lo', 'ci95_hi', 'n_rows', 'n_states'))


def battery_main(comparison: str | None = None, write: bool = False, level: str | None = None) -> None:
    """The registered battery (blocks, F1), the strata diagnostic, and C2 and F5 under the registered comparison.

    Nothing is written unless `write`: then the table, calibration summary and chart data go to reports/validation and
    the run is logged."""
    level = level or default_level()
    registered = comparison or power._substate_config().get('comparison', 'blocks')
    parts = [battery(('F1',), (registered,), level=level, progress=True)]
    if comparison is None:
        parts += [battery(('F1',), ('strata',), level=level, progress=True),
                  battery(('C2', 'F5'), (registered,), level=level, progress=True)]
    table = pl.concat(parts, how='diagonal_relaxed')
    summary = calibration(table)
    chart = chart_rows(table.filter(pl.col('comparison') == registered))
    with pl.Config(tbl_rows=60, tbl_cols=12, tbl_width_chars=200):
        print(summary.select('form', 'comparison', 'part', 'n', 'sum_t2', 'n_abs_t_gt_1_645', 'n_abs_t_gt_1_96',
                             'sd_over_mean_se_signed', 'min_p_ri'))
    if write:
        directory = paths.repo_root() / 'reports' / 'validation'
        outputs = {'placebo_battery.csv': table, 'placebo_battery_calibration.csv': summary,
                   'placebo_battery_chart.csv': chart}
        for name, frame in outputs.items():
            frame.write_csv(directory / name)
        runlog.append_run_log(
            run_id=uuid.uuid4().hex[:12], utc_timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            git_sha=runlog.git_sha(), git_dirty=runlog.git_dirty(), config_hash='', spec_id='placebo_battery',
            data_snapshot_ids='exposure_county.parquet;shocks.parquet;substate_shocks.parquet;results_county.parquet;'
                              'results_hist.parquet;metro_blocks.csv',
            output_path=';'.join(f'reports/validation/{n}' for n in outputs),
            notes=f'level={level}; K={ri.K_DRAWS}; seed={ri.SEED}; forms={sorted(set(table["form"]))}; '
                  f'comparisons={sorted(set(table["comparison"]))}')


def metro_main(comparison: str | None = None) -> None:
    rows = []
    for outcome in METRO_OUTCOMES:
        for measure in METRO_MEASURES:
            samples = METRO_SAMPLES.get(outcome, ('all',))
            for sample in samples:
                for form in METRO_FORMS:
                    for comp in ('blocks', 'strata') if comparison is None else (comparison,):
                        try:
                            row = metro_estimate(outcome, measure, form, sample, comp)
                            rows.append(row)
                            print(f"{outcome} {measure} {form} {sample} {comp}: "
                                  f"beta={row['beta']:.4f} p={row['p_ri']:.4f}")
                        except Exception as e:
                            print(f"{outcome} {measure} {form} {sample} {comp}: SKIP {e}")
    if rows:
        table = pl.DataFrame(rows, infer_schema_length=None)
        print(table.select('outcome', 'measure', 'form', 'sample', 'comparison', 'beta', 'p_ri', 'n_rows'))


if __name__ == '__main__':
    main()
