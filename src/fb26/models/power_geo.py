"""Production-window inputs and logged geography-aware precision/battery runs.

No exploratory modules are imported. Historical outcome rows come from the
registered battery; treatment assignments come from the 2026 runner, including
its coverage rule and configured pass-through control.
"""
import contextlib
import functools
import hashlib
import json
import uuid
import time
import argparse
from multiprocessing import get_context
from datetime import date, datetime, timezone

import numpy as np
import polars as pl
import yaml

from fb26.etl import shocks, substate_shocks as ss
from fb26.models import frame2026, placebo, power, ri, run2026
from fb26.util import paths, runlog

SCRATCH = paths.repo_root() / '.scratch/placebo_dig/impl_b3b'
REPORTS = paths.repo_root() / 'reports/validation'
WINDOWS = {
    'S1': (date(2026, 9, 13), date(2026, 10, 3)),
    'actual': (date(2026, 10, 13), date(2026, 11, 2)),
}
PRE = (date(2024, 10, 15), date(2024, 11, 4))
VERSIONS = (('county_markets_F1', 'county_markets', 'county', 'F1'),
            ('county_markets_C2', 'county_markets', 'county', 'C2'),
            ('county_markets_F5', 'county_markets', 'county', 'F5'),
            ('R_metro_F1', 'metro', 'metro', 'F1'))


def config():
    return yaml.safe_load((paths.config_dir() / 'analysis.yaml').read_text())


@functools.cache
def window_inputs(window, actual_min_days=None):
    """Rebuild both price layers over one window; keep pre-treatment fit slopes.

    The production state-price source precedence and geography are retained.
    The lag window stays 2020–24. HF no-drift prediction is the fitted slope
    times this window's national shock, as in pass_through.build.
    All counties retain geography rows, including counties lacking prices.
    """
    post = WINDOWS[window]
    cfg = config()
    # All requested windows are inside the parsed daily county panel. Avoid
    # load_inputs(): it also unpacks an irrelevant pre-2024 git archive in
    # .scratch/tmp, outside this job's approved scratch directory.
    extra = ss.report_inputs()
    universe = extra['exposure'].select(pl.col('geo_id').alias('county_fips'), 'state_fips')
    from fb26.util import rawstore
    types = ss.read_cbsa_types(rawstore.load_raw(ss.mp.cbsa_path()))
    cz = ss.read_cz(rawstore.load_raw(ss.latest_raw('commuting_zones', 'csv.gz')))
    geography = ss.build_geography(universe, types, cz, extra['states'])
    metro, metro_cbsa = ss.metro_daily(universe)
    county = pl.read_parquet(paths.interim_dir() / 'prices_county_daily.parquet')
    if county['as_of_date'].min() > PRE[0]:
        raise ValueError('Requested baseline predates parsed county panel')
    inputs = ss.Inputs(ss.load_windows(), geography, ss.to_daily(county), metro, metro_cbsa, {})
    prices = shocks.load_aaa_prices()
    means = []
    for label, bounds in (('pre', PRE), ('post', post)):
        daily = shocks.daily_values(prices.filter(pl.col('as_of_date').is_between(*bounds)))
        means.append(daily.group_by('state_abbr').agg(
            pl.col('value').mean().alias('mean_' + label),
            pl.col('as_of_date').n_unique().alias('n_days_' + label)))
    sh = (pl.read_parquet(paths.processed_dir() / 'shocks.parquet')
          .filter(pl.col('year') == 2026)
          .drop('mean_pre', 'mean_post', 'n_days_pre', 'n_days_post')
          .join(means[0], on='state_abbr', how='left', validate='1:1')
          .join(means[1], on='state_abbr', how='left', validate='1:1')
          .with_columns((pl.col('mean_post') - pl.col('mean_pre')).alias('delta_p')))
    if sh.height != 51 or sh['delta_p'].null_count() or not sh['delta_p'].is_finite().all():
        raise ValueError('Incomplete state price window; no imputation or unregistered fallback')
    nat = sh['delta_p'].mean()
    sh = sh.with_columns(
        (pl.col('delta_p') - pl.col('delta_p').mean().over('padd_group')).alias('deviation'),
        (pl.col('lag_delta_p') - pl.col('lag_delta_p').mean().over('padd_group')).alias('lag_deviation'),
        pl.lit(nat).alias('g_nat'),
        (pl.col('pt_intercept') + pl.col('pt_slope') * nat).alias('g_pt'))
    hf = pl.read_parquet(paths.processed_dir() / 'g_pt_hf.parquet').with_columns(
        pl.when((pl.col('year') == 2026) & (pl.col('measure') == 'w21'))
        .then(pl.col('b') * nat).otherwise(pl.col('g_pt_nodrift')).alias('g_pt_nodrift'))
    frames, horizons = [], {}
    minimum = cfg['level']['min_days']
    minimum_post = actual_min_days if window == 'actual' and actual_min_days is not None else minimum
    spec = ss.Spec('w21', 2026, post, PRE, minimum_post, minimum, ('county', 'metro'))
    for level, daily in (('county', inputs.county_daily), ('metro', inputs.metro_daily)):
        horizon = ss.horizon_of(daily)
        horizons[level] = str(daily.filter(pl.col('as_of_date').is_between(*post))['as_of_date'].max())
        geo = ss.level_geography(inputs, level)
        rows = (ss.spec_shocks(daily, spec, geo.select('county_fips', 'geo_flags'), horizon)
                .join(geo.drop('geo_flags'), on='county_fips', how='left', validate='1:1')
                .join(sh.select(pl.col('area_id').alias('state_fips'), pl.col('delta_p').alias('g_s')),
                      on='state_fips', how='left', validate='m:1')
                .with_columns((pl.col('g_c') - pl.col('g_s')).round(ss.DECIMALS).alias('u_c'),
                              pl.lit('w21').alias('measure'), pl.lit(2026).alias('cycle'),
                              pl.lit(level).alias('level'), pl.col('window_open').alias('provisional'))
                .rename({'county_fips': 'geo_id'}))
        frames.append(rows)
    substate = pl.concat(frames, how='diagonal_relaxed')
    metadata = dict(window=window, pre_start=str(PRE[0]), pre_end=str(PRE[1]),
                    post_start=str(post[0]), post_end=str(post[1]),
                    state_last_day=str(prices.filter(pl.col('as_of_date').is_between(*post))['as_of_date'].max()),
                    county_last_day=horizons['county'], metro_last_day=horizons['metro'],
                    state_min_pre_days=int(sh['n_days_pre'].min()),
                    state_min_post_days=int(sh['n_days_post'].min()),
                    min_days=minimum, min_days_post=minimum_post, mu_pt_source=cfg['mu_pt']['source'],
                    run_utc=datetime.now(timezone.utc).isoformat())
    return sh, substate, hf, metadata


def window_assignment(window, source, level, actual_min_days=None):
    sh, substate, hf, metadata = window_inputs(window, actual_min_days)
    cfg = config()
    asg = frame2026.build_assignment(sh, substate, level, cfg['mu_pt']['source'], hf,
                                     dict(cfg['level'], min_days_post=metadata['min_days_post']),
                                     'w21', 'blocks', source=source)
    if asg.state_table['g_pt'].null_count() or asg.state_table['lag_dev'].null_count():
        raise ValueError('Production controls missing')
    return asg, metadata


@contextlib.contextmanager
def assignment_context(asg):
    """Supply a production assignment to the existing historical-frame adapter.

    Like placebo._outcome_rows, this is process-local and must not run in
    threads. Restore the function even if a fit fails.
    """
    original = power.build_assignment
    power.build_assignment = lambda *args, **kwargs: asg
    try:
        yield
    finally:
        power.build_assignment = original


def log_run(spec_id, outputs, notes):
    """Log without invoking git (the B3b brief prohibits git operations)."""
    inputs = [paths.config_dir() / f for f in ('analysis.yaml', 'market_blocks.csv', 'metro_blocks.csv')]
    inputs += [paths.processed_dir() / f for f in (
        'exposure_county.parquet', 'shocks.parquet', 'substate_shocks.parquet',
        'results_county.parquet', 'results_hist.parquet', 'g_pt_hf.parquet')]
    inputs += [paths.interim_dir() / f for f in (
        'prices_county_daily.parquet', 'prices_state_daily_aaagas.parquet',
        'prices_state_daily_wayback.parquet', 'prices_metro_daily.parquet', 'prices_metro_wayback.parquet')]
    inputs += [paths.config_dir() / 'states.csv', paths.config_dir() / 'windows.yaml',
               REPORTS / 'metro_cbsa_crosswalk.csv', ss.mp.cbsa_path(),
               ss.latest_raw('commuting_zones', 'csv.gz')]
    snapshots = ';'.join(f'{p.relative_to(paths.repo_root())}:{hashlib.sha256(p.read_bytes()).hexdigest()}'
                         for p in inputs if p.is_file())
    run_id = uuid.uuid4().hex[:12]
    runlog.append_run_log(run_id=run_id, utc_timestamp=datetime.now(timezone.utc).isoformat(),
                         git_sha='not_queried', git_dirty='not_queried',
                         config_hash=runlog.config_hash(inputs[:3]), spec_id=spec_id,
                         data_snapshot_ids=snapshots, output_path=';'.join(outputs), notes=notes)
    return run_id


def battery_summary(table):
    parts = []
    for (window, version), group in table.group_by('window', 'version', maintain_order=True):
        parts.append(placebo.calibration(group).with_columns(pl.lit(window).alias('window'),
                                                           pl.lit(version).alias('version')))
    return pl.concat(parts)


def write_battery(table):
    stem = REPORTS / 'T4a_placebo_battery'
    table.write_csv(stem.with_suffix('.csv'))
    scale = run2026.per_10_dollars(1.0)
    chart = table.select('window', 'version', 'source', 'form', 'outcome', 'label', 'kind', 'sign',
                         'beta', 'ci95_lo', 'ci95_hi', 't_cr1', 'p_ri', 'n_rows',
                         'pre_start', 'pre_end', 'post_start', 'post_end', 'run_utc').with_columns(
        *[(pl.col(c) * scale).alias(c + '_per_10_dollars') for c in ('beta', 'ci95_lo', 'ci95_hi')])
    chart.write_csv(REPORTS / 'T4a_placebo_battery_chart.csv')
    summary = battery_summary(table).filter(pl.col('part') == 'all')
    rows = [[r['window'], r['version'], str(r['n']), f"{r['sum_t2']:.3f}",
             str(r['n_abs_t_gt_1_645']), str(r['n_abs_t_gt_1_96']),
             f"{r['sd_over_mean_se_raw']:.3f}", f"{r['min_p_ri']:.4f}"] for r in summary.to_dicts()]
    md = '# Twelve-election placebo battery\n\n'
    for window in table['window'].unique(maintain_order=True):
        r = table.filter(pl.col('window') == window).row(0, named=True)
        md += (f"{window}: {r['post_start']}–{r['state_last_day']} minus "
               f"{r['pre_start']}–{r['pre_end']}; minimum {r['min_days_post']} days "
               f"post and {r['min_days']} days pre. ")
    md += ('Both layers use the labelled window. Blocks; state-only lag; recorded coverage '
           'threshold and pass-through control. K=9,999, seed=20261019. '
           'Chart intervals are inverted two-sided RI intervals. Per-$10 factor: '
           f'{scale:.9f} at $6,659 monthly income.\n\n')
    md += power._md(rows, ['Window', 'Version', 'Outcomes', 'Sum t²', 'abs(t) > 1.645',
                           'abs(t) > 1.96', 'SD(beta)/mean SE', 'Min RI p']) + '\n\n'
    for win in table['window'].unique(maintain_order=True):
        main = summary.filter((pl.col('window') == win) & (pl.col('version') == 'county_markets_F1'))
        if main.height:
            r = main.row(0, named=True)
            md += (f"For {win}, the main county-market battery has sum t² {r['sum_t2']:.2f}, "
                   f"{r['n_abs_t_gt_1_96']} outcome(s) with abs(t) above 1.96, and minimum RI p "
                   f"{r['min_p_ri']:.4f}. ")
        minimum = table.filter(pl.col('window') == win).sort('p_ri').row(0, named=True)
        md += (f"The smallest p across the requested {win} versions is {minimum['p_ri']:.4f} "
               f"for {minimum['version']}, {minimum['label']}. ")
    md += '\n\n'
    md += ('These are checks for associations with elections that preceded the treatment. '
           'A sum of squared t around twelve and SD(beta)/mean SE around one are useful reference '
           'points, not a joint test: the outcomes overlap and are correlated. The table reports '
           'every requested version; smaller p-values and larger scatter warrant attention, '
           'but do not by themselves identify the source of an association. S1 is the primary '
           'basis for pre-registration claims; the actual window remains a sensitivity until complete.\n\n')
    md += ('The windowed CLI uses frame2026.build_assignment, including hf_nodrift and the '
           'coverage minima recorded below. Earlier exploratory/battery runs used power.state_table '
           '(SEDS control) and interim coverage. Those results are not exact reproduction targets. '
           'The historical outcome construction is unchanged. Early presidential county defects '
           '(ME 2004, MS 2004, NY 2004, NY 2008) are repaired from Amlani and Algara; '
           'AK 2004 is excluded (election-district rows only).\n\n')
    md += 'Window coverage and run dates:\n\n'
    for window in table['window'].unique(maintain_order=True):
        metadata = {k: table.filter(pl.col('window') == window)[k][0] for k in
                    ('window', 'pre_start', 'pre_end', 'post_start', 'post_end', 'state_last_day',
                     'county_last_day', 'metro_last_day', 'state_min_pre_days', 'state_min_post_days',
                     'min_days', 'min_days_post', 'mu_pt_source', 'run_utc')}
        md += (f"- {window}: minimum {metadata['min_days_post']} days in the 2026 window; "
               f"{metadata['min_days']} days in the 2024 baseline.\n")
        if window == 'actual' and metadata['min_days_post'] != metadata['min_days']:
            md += (f"This is a provisional run with a {metadata['min_days_post']}-day minimum for the "
                   '2026 window (PI, 2026-10-28) because county files are complete only from Oct 21; '
                   f"the registered rule ({metadata['min_days']} days) applies to the final run.\n")
        md += '- ' + json.dumps(metadata, sort_keys=True) + '\n'
    md += '\nAssignment support:\n\n'
    for r in table.unique(['window', 'version'], maintain_order=True).to_dicts():
        md += f"- {r['window']} {r['version']}: {r['n_units']} local units, {r['n_blocks']} blocks."
        if r['n_units'] == 0:
            md += ' No local units pass coverage; this is the state-layer fallback.'
        md += '\n'
    stem.with_suffix('.md').write_text(md)


def battery_window_main(window, write=False, actual_min_days=None):
    """P3: requested versions only, S1 first, then actual; no spec search."""
    windows = ('S1', 'actual') if window == 'both' else (window,)
    tables = []
    for win in windows:
        for version, source, level, form in VERSIONS:
            asg, metadata = window_assignment(win, source, level, actual_min_days)
            print(f'START {win} {version}: {asg.clusters.height} units', flush=True)
            with assignment_context(asg):
                table = placebo.battery((form,), ('blocks',), level=level, source=source,
                                        metro_lag=False, progress=True)
            table = table.with_columns(*[pl.lit(v).alias(k) for k, v in metadata.items()],
                                       pl.lit(version).alias('version'), pl.lit(source).alias('source'),
                                       pl.lit(asg.clusters.height).alias('n_units'),
                                       pl.lit(asg.clusters['block'].n_unique() if asg.clusters.height else 0).alias('n_blocks'))
            tables.append(table)
            if write:
                SCRATCH.mkdir(parents=True, exist_ok=True)
                checkpoint = SCRATCH / f'battery_{win}_{version}.csv'
                table.write_csv(checkpoint)
                write_battery(pl.concat(tables, how='diagonal_relaxed'))
                rid = log_run('T4a_placebo_battery', [str(checkpoint.relative_to(paths.repo_root())),
                    'reports/validation/T4a_placebo_battery.csv', 'reports/validation/T4a_placebo_battery.md',
                    'reports/validation/T4a_placebo_battery_chart.csv'],
                    f'window={win}; version={version}; K={ri.K_DRAWS}; seed={ri.SEED}; '
                    f'outcome_order={list(placebo.BATTERY)}; metadata={json.dumps(metadata)}; '
                    'git metadata not queried: brief prohibits git operations')
                print(f'LOG {rid}', flush=True)
    print(battery_summary(pl.concat(tables)).filter(pl.col('part') == 'all'))


# Independent streams match Job 1, including the infinite-range increment limit.
FIELD_SEED, STATE_SEED, INNER_SEED = 202610300, 202610301, 202610302
RHOS = (0.0, 400.0, 1600.0, np.inf)
BETA_GRID = np.r_[-100., -80., -60., -40., -30., np.linspace(-20, 20, 81),
                  30., 40., 60., 80., 100.]
PRECISION_VERSIONS = (('county_markets', 'county_markets', 'county'),
                      ('metro', 'metro', 'metro'), ('state', 'metro', 'state'))


def distances(lat, lon):
    """Great-circle centroid distances in km (Job 1 convention)."""
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    h = (np.sin((lat[:, None] - lat[None, :]) / 2) ** 2
         + np.cos(lat[:, None]) * np.cos(lat[None, :])
         * np.sin((lon[:, None] - lon[None, :]) / 2) ** 2)
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def nature(asg, dist, rho, n=500, field_seed=FIELD_SEED, state_seed=STATE_SEED):
    """Stratum-wide nature, BEFORE draw-specific within-block recentring.

    Preserve each stratum's multiset of m minus its stratum mean, not the
    observed block residuals. At rho=0 ranks are uniform permutations within
    strata; conditional on a draw's block pools, its allocation within blocks
    is uniform. Unconditional nature and the observed-pool block RI differ.
    A single field per state correlates ranks across its different strata.
    """
    if rho < 0 or np.isnan(rho):
        raise ValueError('rho must be nonnegative')
    rng = np.random.default_rng(field_seed)
    states = asg.clusters['state_fips'].to_numpy()
    if dist.shape != (len(states), len(states)):
        raise ValueError('distance matrix must align with assignment clusters')
    field = np.empty((n, len(states)))
    for state in np.unique(states):
        idx = np.flatnonzero(states == state)
        normals = rng.standard_normal((n, len(idx)))
        if rho > 0:
            dd = dist[np.ix_(idx, idx)]
            if np.isinf(rho):
                cov = dd[1:, 0, None] + dd[None, 0, 1:] - dd[1:, 1:]
                normals[:, 0] = 0.0
                if len(idx) > 1:
                    normals[:, 1:] = normals[:, 1:] @ np.linalg.cholesky(cov).T
            else:
                normals = normals @ np.linalg.cholesky(np.exp(-dd / rho)).T
        field[:, idx] = normals
    strata = asg.clusters['stratum'].to_numpy()
    dev = ri.center_within(asg.clusters['m'].to_numpy(), strata)
    draw = np.empty_like(field)
    for group in np.unique(strata):
        idx = np.flatnonzero(strata == group)
        order = np.argsort(field[:, idx], axis=1, kind='stable')
        ranks = np.argsort(order, axis=1)
        draw[:, idx] = np.sort(dev[idx])[ranks]
    state_draw = ri.permuted_deviations(asg.dev_state, asg.groups[:len(asg.states)],
                                        n, np.random.default_rng(state_seed))
    return np.column_stack([state_draw, draw])


def block_draws(draws, asg):
    """Production B-tilde uses the block means of EACH outer assignment."""
    out = draws.copy()
    s = len(asg.states)
    for group in np.unique(asg.groups[s:]):
        idx = s + np.flatnonzero(asg.groups[s:] == group)
        out[:, idx] -= out[:, idx].mean(axis=1, keepdims=True)
    return out


def precision_context(window, source, level, actual_min_days=None):
    from fb26.etl import metro_blocks
    asg, metadata = window_assignment(window, source, level, actual_min_days)
    metadata = metadata.copy()
    with assignment_context(asg):
        df, _, omitted = power.sub_frame(2022, 'w21', 2026, level,
                                         comparison='blocks', metro_lag=False, source=source)
    if omitted:
        raise ValueError(f'Controls omitted: {omitted}')
    df = df.sort('race_id', 'geo_id')
    z = placebo.validate.design_matrix(df, precision=placebo.validate.C1)
    state = df['state_fips'].replace_strict({s: i for i, s in enumerate(asg.states)}).to_numpy()
    design = ri.TwoLayerDesign(df['dv'].to_numpy(), df['e_c'].to_numpy(), state, z,
                              df['w'].to_numpy(), df['state_fips'].to_numpy(), len(asg.states),
                              df['unit'].to_numpy(), asg.clusters.height)
    if asg.clusters.height:
        cent = metro_blocks.cluster_centroids(asg)
        if cent['cluster'].to_list() != asg.clusters['cluster'].to_list():
            raise ValueError('Centroid order differs from assignment')
        dist = distances(cent['lat'].to_numpy(), cent['lon'].to_numpy())
    else:
        dist = np.empty((0, 0))
    metadata.update(n_rows=df.height, n_units=asg.clusters.height,
                    n_blocks=len(np.unique(asg.groups[len(asg.states):])),
                    rank=design.res.rank, outcome='2022', form='F1', controls='C1',
                    comparison='blocks', metro_lag=False)
    return asg, design, dist, metadata


_WORK = None


def _worker_init(design, groups, draws, k):
    global _WORK
    power._cache_get = lambda name: None
    power._cache_put = lambda name, row: None
    _WORK = design, groups, draws, k


def _precision_draw(j):
    design, groups, draws, k = _WORK
    seed = int(np.random.SeedSequence([INNER_SEED, j]).generate_state(1)[0])
    inv = design.inversion(draws[j], groups, k, np.random.default_rng(seed))
    o = inv.obs
    beta = float(o.a[0] / o.c[0])
    se = float(np.sqrt(o.f * o.s0[0]) / o.c[0])
    (lo, hi), flags = inv.ci(.10, beta, se, n_grid=401)
    row = dict(draw=j, inner_seed=seed, beta=beta, se_cr1=se,
               ci90_lo=lo, ci90_hi=hi, ci90_halfwidth=(hi-lo)/2,
               ri_flags=';'.join(flags))
    row.update({f'p_{i}': float(inv.p(-b)) for i, b in enumerate(BETA_GRID)})
    return row


def precision_summary(data, version, rho, metadata, j, k):
    rates = np.array([(data[f'p_{i}'].to_numpy() <= .05).mean() for i in range(len(BETA_GRID))])
    scale = run2026.per_10_dollars(1.0)
    size = float(rates[BETA_GRID == 0][0])
    common = dict(metadata, version=version, rho_km=float(rho),
                  J=j, K=k, field_seed=FIELD_SEED, state_seed=STATE_SEED, inner_seed=INNER_SEED,
                  size05=size, size_mcse=float(np.sqrt(size*(1-size)/j)),
                  rho0_gate_low=.05-3*np.sqrt(.05*.95/j), rho0_gate_high=.05+3*np.sqrt(.05*.95/j),
                  mde80_pos=power._mde(BETA_GRID, rates, 1), mde80_neg=power._mde(BETA_GRID, rates, -1),
                  ci90_halfwidth_mean=float(data['ci90_halfwidth'].mean()),
                  sd_beta=float(data['beta'].std()), n_ci_flagged=data.filter(pl.col('ri_flags') != '').height)
    for name in ('mde80_pos', 'mde80_neg', 'ci90_halfwidth_mean', 'sd_beta'):
        common[name + '_per_10_dollars'] = common[name] * scale
    return pl.DataFrame([dict(common, beta_injected=float(b), beta_injected_per_10_dollars=float(b*scale),
                               power05=float(p), power_mcse=float(np.sqrt(p*(1-p)/j)))
                         for b, p in zip(BETA_GRID, rates)])


def write_precision(table):
    stem = 'T5_power_geo_actual' if table['window'][0] == 'actual' else 'T5_power_geo'
    table.write_csv(REPORTS / (stem + '.csv'))
    write_precision_md(table)


def write_precision_md(table):
    summary = table.unique(['version', 'rho_km'], maintain_order=True)
    rows = []
    for r in summary.to_dicts():
        rows.append([r['version'], 'infinite' if np.isinf(r['rho_km']) else str(int(r['rho_km'])), f"{r['size05']:.3f}",
                     f"{r['mde80_neg']:.3f} / {r['mde80_pos']:.3f}",
                     f"{r['mde80_neg_per_10_dollars']:.3f} / {r['mde80_pos_per_10_dollars']:.3f}",
                     f"{r['ci90_halfwidth_mean']:.3f} / {r['ci90_halfwidth_mean_per_10_dollars']:.3f}",
                     f"{r['sd_beta']:.3f} / {r['sd_beta_per_10_dollars']:.3f}"])
    actual = table['window'][0] == 'actual'
    md = '# Geography-aware production precision' + (' — actual window (provisional)' if actual else '') + '\n\n'
    if actual:
        md += (f"Actual: {table['post_start'][0]} to {table['state_last_day'][0]} minus "
               f"{table['pre_start'][0]} to {table['pre_end'][0]}, both price layers; provisional. "
               f"Minimum: {table['min_days_post'][0]} days in the 2026 window and "
               f"{table['min_days'][0]} days in the 2024 baseline. ")
    else:
        md += 'S1: 2026-09-13–2026-10-03 minus 2024-10-15–2024-11-04, both price layers. '
    md += (
           'F1, C1, observed 2022 outcome as noise, state-only lag, HF no-drift pass-through, '
           'production coverage and frozen blocks. All three versions use the full eligible '
           '2022 frame. State draws permute within PADD; local values are stratum-centered '
           'cluster means rank-assigned along a state Gaussian field with exponential covariance. '
           'Infinite range uses the increment-covariance limit. Each outer draw is recentered '
           'within blocks, then tested by production within-block RI. Power injects beta times '
           'this production block-centered B-tilde. Job 1 used a different injection for its '
           'exploratory power, so only its size table is a reproduction gate.\n\n')
    md += (f"J={table['J'][0]}, K={table['K'][0]}, two-sided alpha=.05; independent seeds "
           f'{FIELD_SEED} (field), {STATE_SEED} (state), SeedSequence([{INNER_SEED}, draw]) (RI). '
           'Common random numbers across rho; the state-only rows are identical because rho '
           'has no state-layer role. The CSV contains every beta-grid power estimate and its '
           'Monte Carlo SE. Detectable effects linearly interpolate the first 80% crossing '
           'on each side; negative/positive values are reported separately.\n\n')
    md += power._md(rows, ['Version', 'rho km', 'Size', '80% beta (− / +)',
                           '80% points/$10 (− / +)', 'Mean 90% half-width beta / $10',
                           'SD(beta) beta / $10']) + '\n\n'
    md += ('The rule-out proxy is the **mean half-width of the inverted 90% RI interval at '
           'true beta=0**, not an observed bound or a coverage-adjusted guarantee. CI hulls use '
           'the existing inversion algorithm (401-point search and bisection); flagged intervals '
           'are counted in the CSV. Dollar translations use 1000/6659.\n\n'
           'At rho=0, nature is uniform within strata. It is not the unconditional RI '
           'distribution for the observed block pools: those pools change between outer draws. '
           'Conditional on each draw’s block pools, allocation is uniform within blocks. '
           'The empirical gate is a three-Monte-Carlo-SE band around .05, as in Job 1. '
           'No nature process is altered to satisfy this gate.\n\n')
    for r in summary.filter(pl.col('rho_km') == 0).to_dicts():
        passed = r['rho0_gate_low'] <= r['size05'] <= r['rho0_gate_high']
        md += (f"- {r['version']} rho=0: {r['size05']:.3f}; gate "
               f"[{r['rho0_gate_low']:.4f}, {r['rho0_gate_high']:.4f}]: {'PASS' if passed else 'FAIL'}.\n")
    md += ('\nJob 1 interim-treatment m4 reproduction (metro design on Job 1\'s interim treatment and frozen m4 '
           'blocks, 2022 common F1 and F6, 2018 all F1 and F5, rho 400/1600/infinity, J=500, K=999, same seeds, '
           'nature from `power_geo.nature` and `block_draws`): all 12 size rates equal Job 1\'s exactly; '
           'maximum absolute difference in RI p-values 0 and in beta 3e-15 (scratch '
           '`.scratch/placebo_dig/impl_b3b/job1_gate_results.csv`).\n')
    stem = 'T5_power_geo_actual' if actual else 'T5_power_geo'
    (REPORTS / (stem + '.md')).write_text(md)


def precision_main(workers=2, j=500, k=999, window='S1', actual_min_days=None):
    """Logged CLI, fixed specification order; checkpoints after every scenario."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    power._cache_get = lambda name: None
    power._cache_put = lambda name, row: None
    tables = []
    for version, source, level in PRECISION_VERSIONS:
        asg, design, dist, metadata = precision_context(window, source, level, actual_min_days)
        state_data = None
        for rho in RHOS:
            label = 'infinite' if np.isinf(rho) else str(int(rho))
            stem = f'P1_{window}_{version}_rho{label}'
            start = time.monotonic()
            print(f'START {stem}: rows={metadata["n_rows"]}, units={asg.clusters.height}', flush=True)
            raw_draws = nature(asg, dist, rho, j)
            draws = block_draws(raw_draws, asg)
            if level == 'state' and state_data is not None:
                data = state_data
                print('REUSE state rho=0: rho has no state-layer role', flush=True)
            else:
                rows = []
                with get_context('spawn').Pool(workers, initializer=_worker_init,
                                               initargs=(design, asg.groups, draws, k)) as pool:
                    for row in pool.imap(_precision_draw, range(j), chunksize=5):
                        rows.append(row)
                        if len(rows) % 25 == 0:
                            print(f'{stem}: {len(rows)}/{j} ({time.monotonic()-start:.0f}s)', flush=True)
                            (SCRATCH / (stem + '_progress.json')).write_text(json.dumps(
                                dict(completed=len(rows), J=j, elapsed_s=time.monotonic()-start)))
                data = pl.DataFrame(rows)
                if level == 'state':
                    state_data = data
            checkpoint = SCRATCH / (stem + '_draws.csv')
            data.write_csv(checkpoint)
            tab = precision_summary(data, version, rho, metadata, j, k)
            tables.append(tab)
            write_precision(pl.concat(tables))
            report_stem = 'T5_power_geo_actual' if window == 'actual' else 'T5_power_geo'
            rid = log_run(report_stem, [str(checkpoint.relative_to(paths.repo_root())),
                    f'reports/validation/{report_stem}.csv', f'reports/validation/{report_stem}.md'],
                    f'window={window}; version={version}; rho={label}; J={j}; K={k}; '
                    f'field_seed={FIELD_SEED}; state_seed={STATE_SEED}; inner_seed={INNER_SEED}; '
                    f'beta_grid={BETA_GRID.tolist()}; metadata={json.dumps(metadata)}; '
                    f'state_rho_reuse={level == "state" and rho != 0}; production block-centered injection')
            print(f'LOG {rid}: size={tab["size05"][0]:.3f}, MDE+={tab["mde80_pos"][0]:.3f}', flush=True)
            if rho == 0 and not tab['rho0_gate_low'][0] <= tab['size05'][0] <= tab['rho0_gate_high'][0]:
                raise RuntimeError('rho=0 size gate failed; process preserved; stop and report')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=['precision'])
    parser.add_argument('--window', choices=('S1', 'actual'), default='S1')
    parser.add_argument('--actual-min-days', type=int)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    precision_main(workers=args.workers, window=args.window, actual_min_days=args.actual_min_days)
