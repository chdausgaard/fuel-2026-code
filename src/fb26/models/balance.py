"""T4 balance checks (memo §8): T4a placebo on past swings, T4b shock-level balance.

T4a: the 2016 → 20 and 2020 → 24 presidential swings and the 2018 and 2022 midterm swings, regressed
on the 2026 recentered burden with P1's design controls (race FE, E_c × PADD, μ^pt, B̃^lag, Oil_c). No
precision control is used, so a placebo outcome is never also a control. It runs at the state level and,
when the D22 level has sub-state units, at that level with two-layer RI.

T4b: the 2026 state deviations regressed on state characteristics with PADD fixed effects. Covariates are
standardized, so a coefficient is the deviation change per SD. Robust (HC1) SEs; RI permutes the
deviations within PADD groups; the joint test is the HC1 Wald statistic of all covariates. Intervals for
the equivalence bound ε* are HC1 t-intervals. A covariate with no input (or input for fewer than
MIN_COVERAGE of the states) is dropped and recorded.

Historical outcomes and 2026 prices only; no 2026 outcomes. 2026 shocks are provisional until Nov 3.

    python -m fb26.models.balance
"""
import argparse
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from scipy import stats

from fb26.etl.pass_through import TAX_CLASS
from fb26.models import frame2026, level2026, p1, ri, validate
from fb26.util import paths, runlog

OUTCOMES = ('pres_2016_2020', 'pres_2020_2024', 'mid_2018', 'mid_2022')
COVARIATES = ('gov_rep', 'trump_margin_2024', 'grad_pres_2020_2024', 'grad_mid_2022',
              'oil_gas_share', 'pct_tax')
MIN_COVERAGE = 0.75
OUT = 'reports/validation/T4_balance_2026'


def load_outcomes(exposure: pl.DataFrame, catch: bool = False) -> dict[str, pl.DataFrame]:
    """Historical swing rows (geo_id, state_fips, race_id, dv, w) with E_c and Oil_c; reads the results."""
    exp = (exposure.filter(pl.col('g_method') == validate.PRIMARY)
           .select('geo_id', 'state_fips', 'e_c', 'oil_gas_emp_share'))
    out = {}
    for name in OUTCOMES:
        try:
            kind, *years = name.split('_')
            if kind == 'pres':
                sw = validate.pres_swing(*(int(y) for y in years))
                df = sw.select('geo_id', pl.col('swing').alias('dv'), pl.col('v_start').alias('w'))
                df = df.join(exp.select('geo_id', 'state_fips'), on='geo_id').with_columns(
                    ('PRES_' + pl.col('state_fips')).alias('race_id'))
                df = df.drop('state_fips')
            else:
                df = validate.outcomes(int(years[0]))[0].select('geo_id', 'race_id', 'dv', 'w')
            out[name] = (df.join(exp, on='geo_id', how='inner')
                         .filter((pl.col('state_fips') != '02') & pl.col('e_c').is_not_null())
                         .with_columns(pl.col('oil_gas_emp_share').fill_null(0.0)))
        except Exception:
            if not catch:
                raise
    return out


def design_rows(df: pl.DataFrame, assignment) -> pl.DataFrame:
    """Add the 2026 treatment and design controls of an assignment (state or D22 level) to outcome rows."""
    st = assignment.state_table.select('state_fips', 'padd_group', pl.col('dev').alias('deviation'),
                                       'lag_dev', 'g_pt')
    units = assignment.counties.select('geo_id', 'unit')
    dev2 = assignment.clusters.select(pl.int_range(pl.len(), dtype=pl.Int64).alias('unit'),
                                      pl.col('dev').alias('dev2'))
    out = (df.join(st, on='state_fips', how='inner').join(units, on='geo_id', how='left')
           .join(dev2, on='unit', how='left')
           .with_columns(pl.col('unit').fill_null(-1), pl.col('dev2').fill_null(0.0))
           .with_columns((pl.col('e_c') * (pl.col('deviation') + pl.col('dev2'))).alias('b_tilde'),
                         (pl.col('e_c') * pl.col('lag_dev')).alias('b_tilde_lag'),
                         (pl.col('e_c') * pl.col('g_pt')).alias('mu_pt'))
           .sort('race_id', 'geo_id'))
    if out.height == 0 or any(out[c].null_count() for c in ('b_tilde', 'b_tilde_lag', 'mu_pt')):
        raise ValueError('not_run: 2026 controls missing for the placebo rows')
    return out


def estimate_placebo(df: pl.DataFrame, assignment, k: int, seed: int) -> dict:
    """P1 on a historical swing with the 2026 design controls; two-layer RI (state layer only if no units)."""
    z = validate.design_matrix(df)
    y, e, x, w = (df[c].to_numpy() for c in ('dv', 'e_c', 'b_tilde', 'w'))
    clusters = df['state_fips'].to_numpy()
    index = {s: i for i, s in enumerate(assignment.states)}
    state = np.array([index[s] for s in clusters])
    fit = p1.fit(y, x, z, w, clusters)
    test = ri.TwoLayerRandomizationTest(y, e, state, z, w, clusters, assignment.dev, assignment.groups,
                                        len(assignment.states), df['unit'].to_numpy(), k=k, seed=seed)
    r = test.run(fit.beta, fit.se_cr1)
    return {'beta': fit.beta, 'se_cr1': fit.se_cr1, 't_cr1': fit.t, 'p_ri': r.p_value,
            'ci95_lo': r.ci95[0], 'ci95_hi': r.ci95[1], 'ci90_lo': r.ci90[0], 'ci90_hi': r.ci90[1],
            'eps_star': r.eps_star, 'n_rows': fit.n, 'n_states': fit.n_clusters, 'n_params': fit.n_params,
            'n_races': df['race_id'].n_unique(), 'n_counties': df['geo_id'].n_unique(),
            'k_draws': k, 'seed': seed, 'ri_flags': ';'.join(r.flags)}


def t4a(outcomes: dict, assignments: dict, k: int, seed: int, catch: bool = False) -> list[dict]:
    """One row per outcome and layer; catch turns a failure into a fixed not_run row."""
    rows = []
    for name in OUTCOMES:
        for layer, assignment in assignments.items():
            row = {'outcome': name, 'layer': layer}
            try:
                if name not in outcomes:
                    raise ValueError('not_run: outcome input unavailable')
                rows.append({**row, 'status': 'run',
                             **estimate_placebo(design_rows(outcomes[name], assignment), assignment, k, seed)})
            except Exception:
                if not catch:
                    raise
                rows.append({**row, 'status': 'not_run: placebo estimation failed'})
    return rows


def gradient(df: pl.DataFrame) -> pl.DataFrame:
    """Within-state slope of the swing on E_c (race-demeaned, weighted); null for one-county states."""
    w = pl.col('w')
    d = df.with_columns([(pl.col(c) - (w * pl.col(c)).sum().over('race_id') / w.sum().over('race_id')).alias(c + '_d')
                         for c in ('e_c', 'dv')])
    g = d.group_by('state_fips').agg((w * pl.col('e_c_d') * pl.col('dv_d')).sum().alias('num'),
                                     (w * pl.col('e_c_d') ** 2).sum().alias('den'))
    return g.select('state_fips', pl.when(pl.col('den') > 1e-12).then(pl.col('num') / pl.col('den'))
                    .alias('gradient'))


def state_covariates(state_table: pl.DataFrame, exposure: pl.DataFrame, history: pl.DataFrame,
                     outcomes: dict, incumbents: pl.DataFrame | None) -> pl.DataFrame:
    """One row per state in the shock pool (Alaska aside); null where a covariate has no input."""
    base = state_table.filter(pl.col('state_fips') != '02').select(
        'state_fips', 'state_abbr', 'padd_group', pl.col('dev').alias('deviation'))
    n = base.height
    null = pl.Series(values=[None] * n, dtype=pl.Float64)
    base = base.with_columns(null.alias(c) for c in COVARIATES)
    pres = (history.filter((pl.col('year') == 2024) & (pl.col('office') == 'PRESIDENT') & pl.col('party').is_in(['D', 'R']))
            .group_by('state_fips').agg(pl.col('votes').filter(pl.col('party') == 'R').sum().alias('r'),
                                        pl.col('votes').filter(pl.col('party') == 'D').sum().alias('d'))
            .select('state_fips', (100 * (pl.col('r') - pl.col('d')) / (pl.col('r') + pl.col('d'))).alias('trump_margin_2024')))
    exp = exposure.filter(pl.col('g_method') == validate.PRIMARY)
    oil = exp.group_by('state_fips').agg((pl.col('oil_gas_emp_share') * pl.col('households')).sum()
                                         / pl.col('households').sum()).rename({'oil_gas_emp_share': 'oil_gas_share'})
    parts = [pres, oil]
    for column, name in [('grad_pres_2020_2024', 'pres_2020_2024'), ('grad_mid_2022', 'mid_2022')]:
        if name in outcomes:
            parts.append(gradient(outcomes[name]).rename({'gradient': column}))
    if incumbents is not None and {'year', 'office', 'state_abbr', 'party'} <= set(incumbents.columns):
        gov = (incumbents.filter((pl.col('year') == 2026) & (pl.col('office') == 'GOVERNOR'))
               .select('state_abbr', pl.when(pl.col('party') == 'R').then(1.0).when(pl.col('party') == 'D')
                       .then(0.0).alias('gov_rep')))
        base = base.drop('gov_rep').join(gov, on='state_abbr', how='left')
    for part in parts:
        column = [c for c in part.columns if c != 'state_fips'][0]
        base = base.drop(column).join(part, on='state_fips', how='left')
    tax = [1.0 if TAX_CLASS.get(a, ('none_noted',))[0] == 'pct_retail' else 0.0 for a in base['state_abbr']]
    return base.with_columns(pl.Series('pct_tax', tax)).select(
        'state_fips', 'state_abbr', 'padd_group', 'deviation', *COVARIATES)


def t4b(covariates: pl.DataFrame, k: int, seed: int) -> dict:
    """Balance regression of the 2026 state deviations with PADD FE; RI permutes within PADD groups."""
    n0 = covariates.height
    kept, dropped = [], {}
    for name in COVARIATES:
        present = covariates[name].drop_nulls()
        if present.len() == 0:
            dropped[name] = f'not_run: {name} has no input'
        elif present.len() < MIN_COVERAGE * n0:
            dropped[name] = f'not_run: {name} has no input for most states'
        elif present.n_unique() < 2:
            dropped[name] = f'not_run: {name} has no variation'
        else:
            kept.append(name)
    sample = covariates.drop_nulls(kept)
    d = sample['deviation'].to_numpy()
    groups = sample['padd_group'].to_numpy()
    q, n = len(kept), sample.height
    fe = (groups[:, None] == np.unique(groups)[None, :]).astype(float)
    if q == 0 or n <= fe.shape[1] + q:
        raise ValueError('not_run: too few states or covariates for the balance regression')
    z = np.column_stack([sample[c].to_numpy() for c in kept])
    z = (z - z.mean(axis=0)) / z.std(axis=0)
    x = np.column_stack([fe, z])
    p = x.shape[1]
    proj = np.linalg.solve(x.T @ x, x.T)
    a = proj[-q:]
    factor = n / (n - p)

    def summary(draws):
        coef = draws @ proj.T
        resid = draws - coef @ x.T
        cov = factor * np.einsum('kn,jn,ln->kjl', resid ** 2, a, a)
        b = coef[:, -q:]
        t = b / np.sqrt(np.einsum('kjj->kj', cov))
        wald = np.einsum('kj,kj->k', b, np.linalg.solve(cov, b[..., None])[..., 0])
        return b, t, wald

    b, t, wald = (v[0] for v in summary(d[None, :]))
    draws = ri.permuted_deviations(d, groups, k, np.random.default_rng(seed))
    _, t_k, wald_k = summary(draws)
    se = b / t
    crit = {a_: stats.t.ppf(1 - a_ / 2, n - p) for a_ in (.05, .10)}
    rows = []
    for j, name in enumerate(kept):
        lo90, hi90 = b[j] - crit[.10] * se[j], b[j] + crit[.10] * se[j]
        rows.append({'covariate': name, 'beta': b[j], 'se_hc1': se[j], 't_hc1': t[j],
                     'p_ri': (1 + np.sum(np.abs(t_k[:, j]) >= abs(t[j]) * (1 - 1e-12))) / (k + 1),
                     'ci95_lo': b[j] - crit[.05] * se[j], 'ci95_hi': b[j] + crit[.05] * se[j],
                     'ci90_lo': lo90, 'ci90_hi': hi90, 'eps_star': max(abs(lo90), abs(hi90))})
    joint = {'wald': wald, 'p_ri': (1 + np.sum(wald_k >= wald * (1 - 1e-12))) / (k + 1), 'n_covariates': q}
    states_out = sorted(set(covariates['state_fips']) - set(sample['state_fips']))
    return {'covariates': rows, 'joint': joint, 'dropped': dropped, 'n_states': n, 'states_out': states_out,
            'k_draws': k, 'seed': seed}


def run(outcomes, assignments, exposure, history, incumbents, k, seed, catch=False) -> dict:
    """T4a and T4b together; the runner calls this with catch=True."""
    result = {'T4a': t4a(outcomes, assignments, k, seed, catch)}
    try:
        covariates = state_covariates(assignments['state'].state_table, exposure, history, outcomes, incumbents)
        result['T4b'] = {'status': 'run', **t4b(covariates, k, seed)}
    except Exception:
        if not catch:
            raise
        result['T4b'] = {'status': 'not_run: balance regression failed'}
    return result


def long_rows(result: dict) -> list[dict]:
    """T4a and T4b as one long table."""
    rows = []
    for r in result['T4a']:
        rows.append({'test': 'T4a', 'spec': r['outcome'], 'layer': r['layer'], 'term': 'recentered burden',
                     'estimate': r.get('beta'), 'se': r.get('se_cr1'), 'p_ri': r.get('p_ri'),
                     **{c: r.get(c) for c in ('ci95_lo', 'ci95_hi', 'ci90_lo', 'ci90_hi', 'eps_star', 'n_rows', 'n_states')},
                     'status': r['status']})
    b = result['T4b']
    for c in b.get('covariates', []):
        rows.append({'test': 'T4b', 'spec': 'state deviation', 'layer': 'state', 'term': c['covariate'],
                     'estimate': c['beta'], 'se': c['se_hc1'], 'p_ri': c['p_ri'],
                     **{x: c[x] for x in ('ci95_lo', 'ci95_hi', 'ci90_lo', 'ci90_hi', 'eps_star')},
                     'n_states': b['n_states'], 'status': 'run'})
    if 'joint' in b:
        rows.append({'test': 'T4b', 'spec': 'state deviation', 'layer': 'state', 'term': 'joint (Wald)',
                     'estimate': b['joint']['wald'], 'p_ri': b['joint']['p_ri'], 'n_states': b['n_states'],
                     'status': 'run'})
    for name, status in b.get('dropped', {}).items():
        rows.append({'test': 'T4b', 'spec': 'state deviation', 'layer': 'state', 'term': name, 'status': status})
    if 'covariates' not in b:
        rows.append({'test': 'T4b', 'spec': 'state deviation', 'layer': 'state', 'status': b['status']})
    return rows


def markdown(table: pl.DataFrame, stamp: str) -> str:
    def cell(v):
        return '' if v is None else f'{v:.3f}' if isinstance(v, float) else str(v)
    lines = ['# T4 balance checks, 2026', '', stamp, '',
             'T4a: past swings on the 2026 recentered burden (P1 design controls, no precision control; '
             'two-layer RI at the D22 level). T4b: 2026 state deviations on state characteristics with PADD FE '
             '(coefficient per SD of the covariate; HC1 intervals, within-PADD RI p). Historical outcomes only.', '']
    for test in ('T4a', 'T4b'):
        part = table.filter(pl.col('test') == test).drop('test')
        cols = part.columns
        lines += [f'## {test}', '', '| ' + ' | '.join(cols) + ' |', '| ' + ' | '.join('---' for _ in cols) + ' |']
        lines += ['| ' + ' | '.join(cell(row[c]) for c in cols) + ' |' for row in part.iter_rows(named=True)]
        lines.append('')
    return '\n'.join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=paths.config_dir() / 'analysis.yaml')
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    root, processed = paths.repo_root(), paths.processed_dir()
    shocks, substate = (pl.read_parquet(processed / f) for f in ('shocks.parquet', 'substate_shocks.parquet'))
    exposure, history = (pl.read_parquet(processed / f) for f in ('exposure_county.parquet', 'results_county.parquet'))
    races = pl.read_csv(paths.config_dir() / 'races_2026.csv', schema_overrides={'state_fips': pl.String})
    incumbents = pl.read_csv(paths.config_dir() / 'race_incumbents.csv')
    source = config['mu_pt']['source']
    hf = pl.read_parquet(processed / 'g_pt_hf.parquet') if source == 'hf_nodrift' else None
    level = level2026.choose_level(exposure, races, substate, thresholds=config['level'])['level']
    assignments = {'state': frame2026.build_assignment(shocks, substate, 'state', source, hf, config['level'])}
    if level != 'state':
        assignments['D22 level: ' + level] = frame2026.build_assignment(shocks, substate, level, source, hf, config['level'])
    k, seed = config['ri']['k'], config['ri']['seed']
    result = run(load_outcomes(exposure), assignments, exposure, history, incumbents, k, seed)
    provisional = bool(shocks.filter(pl.col('year') == 2026)['provisional'].any())
    stamp = (f'D22 level: {level}; K = {k}; seed {seed}. '
             + ('Provisional 2026 shocks (final after Nov 3).' if provisional else '2026 shocks final.'))
    table = pl.DataFrame(long_rows(result), infer_schema_length=None)
    out = root / (config.get('outputs', {}).get('balance', OUT))
    out.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(out.with_suffix('.csv'))
    out.with_suffix('.md').write_text(markdown(table, stamp), encoding='utf-8')
    runlog.append_run_log(
        run_id=uuid.uuid4().hex[:12], utc_timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        git_sha=runlog.git_sha(), git_dirty=runlog.git_dirty(), config_hash=runlog.config_hash([args.config]),
        spec_id='T4_balance_2026', data_snapshot_ids='exposure_county.parquet;shocks.parquet;results_county.parquet',
        output_path=str(out.with_suffix('.csv').relative_to(root)),
        notes=f'level={level}; K={k}; seed={seed}; provisional={provisional}')
    print(table.select('test', 'spec', 'layer', 'term', 'estimate', 'p_ri', 'status'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
