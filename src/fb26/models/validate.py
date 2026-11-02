"""Validation cycle V_t (memo §6.3): P1 re-estimated on a past cycle, with RI.

Sample (2022 midterm): races with `proposed_in_sample` in the race table;
counties outside Alaska with a recentered burden. Outcome and weights:

    Δv = R2(race) − R2(baseline presidential election), R2 = 100·R / (R + D)
    w  = V_c / n_s,  V_c = baseline two-party presidential votes, n_s = races in the state

Controls (C0 = design controls only): race FE, E_c × PADD dummies, μ^pt,
B̃^lag, Oil_c. Prediction for 2022 (Democratic president): β > 0.

    python -m fb26.models.validate 2022
"""
import hashlib
import sys
import uuid
from datetime import datetime, timezone

import numpy as np
import polars as pl

from fb26.models import burden, p1, ri
from fb26.util import paths, runlog

BASELINE = {2018: 2016, 2020: 2016, 2022: 2020, 2024: 2020}
PRESIDENTIAL = (2020, 2024)
PRIMARY = 'commute_vehicles'
# Memo §6.2 precision sets. C1: swing into the baseline election (for 2022, 2016 → 2020 presidential).
C1 = ('c1_swing',)
C2 = C1 + ('ba_share', 'hisp_share', 'nh_black_share', 'log_pop_density', 'median_age', 'log_median_income')


def pres_swing(start: int, end: int) -> pl.DataFrame:
    """County presidential swing R2(end) − R2(start), with the start year's two-party votes."""
    res = pl.read_parquet(paths.processed_dir() / 'results_county.parquet')
    a = _r2(res.filter((pl.col('year') == start) & (pl.col('office') == 'PRESIDENT')), ['geo_id'])
    b = _r2(res.filter((pl.col('year') == end) & (pl.col('office') == 'PRESIDENT')), ['geo_id'])
    return (a.select('geo_id', pl.col('r2').alias('r2_a'), pl.col('two_party').alias('v_start'))
            .join(b.select('geo_id', pl.col('r2').alias('r2_b')), on='geo_id', how='inner')
            .select('geo_id', (pl.col('r2_b') - pl.col('r2_a')).alias('swing'), 'v_start'))


def _r2(df: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    return (df.filter(pl.col('party').is_in(['D', 'R']))
            .group_by(keys).agg(pl.col('votes').filter(pl.col('party') == 'R').sum().alias('r'),
                                pl.col('votes').filter(pl.col('party') == 'D').sum().alias('d'))
            .with_columns((100 * pl.col('r') / (pl.col('r') + pl.col('d'))).alias('r2'),
                          (pl.col('r') + pl.col('d')).alias('two_party')))


def outcomes(cycle: int) -> tuple[pl.DataFrame, dict]:
    """Midterms: included Senate/governor races (races_{cycle}.csv). Presidential cycles: one race per state."""
    res = pl.read_parquet(paths.processed_dir() / 'results_county.parquet')
    not_ak = ~pl.col('flags').fill_null('').str.contains('excluded_ak')
    if cycle in PRESIDENTIAL:
        cur = (res.filter((pl.col('year') == cycle) & (pl.col('office') == 'PRESIDENT') & not_ak)
               .with_columns(('PRES_' + pl.col('state_fips')).alias('race_id')))
        keep = cur['race_id'].unique().to_list()
    else:
        races = pl.read_csv(paths.repo_root() / 'reports' / 'validation' / f'races_{cycle}.csv',
                            schema_overrides={'state_fips': pl.Utf8})
        keep = races.filter(pl.col('proposed_in_sample'))['race_id'].to_list()
        cur = res.filter((pl.col('year') == cycle) & pl.col('race_id').is_in(keep) & not_ak)
    filled = (cur.filter(pl.col('flags').fill_null('').str.contains('filled_from_medsl_county_level'))
              .select('geo_id', 'race_id').unique().with_columns(pl.lit(True).alias('filled')))
    cur = _r2(cur, ['state_fips', 'geo_id', 'race_id'])
    base = (_r2(res.filter((pl.col('year') == BASELINE[cycle]) & (pl.col('office') == 'PRESIDENT')), ['geo_id'])
            .select('geo_id', pl.col('r2').alias('r2_base'), pl.col('two_party').alias('v_c')))
    df = (cur.join(base, on='geo_id', how='left', validate='m:1')
          .join(filled, on=['geo_id', 'race_id'], how='left')
          .with_columns((pl.col('r2') - pl.col('r2_base')).alias('dv'),
                        pl.col('filled').fill_null(False),
                        pl.col('race_id').n_unique().over('state_fips').alias('n_s'))
          .with_columns((pl.col('v_c') / pl.col('n_s')).alias('w')))
    info = {'races': len(keep), 'rows_two_party_zero': df.filter(pl.col('two_party') == 0).height,
            'rows_no_baseline': df.filter(pl.col('r2_base').is_null()).height}
    return df.filter((pl.col('two_party') > 0) & pl.col('r2_base').is_not_null()), info


def frame(cycle: int, g_method: str) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    """Analysis rows and the 51-unit shock table (all states enter the permutations)."""
    out, _ = outcomes(cycle)
    b = burden.build(cycle, g_method)
    c2 = (pl.read_parquet(paths.processed_dir() / 'exposure_county.parquet')
          .filter(pl.col('g_method') == g_method).select('geo_id', *C2[1:]))
    c1 = pres_swing(BASELINE[cycle] - 4, BASELINE[cycle]).select('geo_id', pl.col('swing').alias('c1_swing'))
    df = (out.join(b.drop('state_fips'), on='geo_id', how='left', validate='m:1')
          .join(c2, on='geo_id', how='left', validate='m:1')
          .join(c1, on='geo_id', how='left', validate='m:1'))
    dropped = df.filter(pl.col('b_tilde').is_null())['geo_id'].unique().sort().to_list()
    df = (df.filter(pl.col('b_tilde').is_not_null())
          .with_columns(pl.col('oil_gas_emp_share').fill_null(0.0))
          .sort('race_id', 'geo_id'))
    shocks = (pl.read_parquet(paths.processed_dir() / 'shocks.parquet')
              .filter(pl.col('year') == cycle)
              .select(pl.col('area_id').alias('state_fips'), 'padd_group', 'deviation')
              .sort('state_fips'))
    return df, shocks, dropped


def design_matrix(df: pl.DataFrame, lag: bool = True, pt: bool = True,
                  precision: tuple[str, ...] = ()) -> np.ndarray:
    race = df['race_id'].to_numpy()
    padd = df['padd_group'].to_numpy()
    e = df['e_c'].to_numpy()
    cols = [(race == r).astype(float) for r in np.unique(race)]
    cols += [e * (padd == g) for g in np.unique(padd)]
    if pt and df['mu_pt'].null_count() == 0:
        cols.append(df['mu_pt'].to_numpy())
    if lag and df['b_tilde_lag'].null_count() == 0:
        cols.append(df['b_tilde_lag'].to_numpy())
    cols.append(df['oil_gas_emp_share'].to_numpy())
    cols += [df[c].to_numpy() for c in precision]
    return np.column_stack(cols)


def diagnostics(df: pl.DataFrame) -> dict:
    """T1 quantities on the estimation sample (treatment only, no outcomes)."""
    w = pl.col('w')
    d = df.with_columns(
        ((pl.col('e_c') - (w * pl.col('e_c')).sum().over('race_id') / w.sum().over('race_id')) ** 2 * w).alias('lam'),
        (pl.col('b') - (w * pl.col('b')).sum().over('race_id') / w.sum().over('race_id')).alias('b_w'),
        (pl.col('b_tilde') - (w * pl.col('b_tilde')).sum().over('race_id') / w.sum().over('race_id')).alias('bt_w'))
    lam = (d.group_by('state_fips', maintain_order=True).agg(pl.col('lam').sum(), pl.col('deviation').first())
           .sort('state_fips'))
    lam_v = lam['lam'].to_numpy()
    infl = lam_v * lam['deviation'].to_numpy() ** 2
    wv = d['w'].to_numpy()
    return {
        'n_eff': float(lam_v.sum() ** 2 / np.sum(lam_v ** 2)),
        'max_state_share': float(lam_v.max() / lam_v.sum()),
        'max_realized_influence': float(infl.max() / infl.sum()),
        'kept_share_within_race': float(np.sum(wv * d['bt_w'].to_numpy() ** 2) / np.sum(wv * d['b_w'].to_numpy() ** 2)),
        'corr_bt_btlag': float(np.corrcoef(d['b_tilde'], d['b_tilde_lag'])[0, 1]) if d['b_tilde_lag'].null_count() == 0 else None,
        'sd_bt_within_race': float(np.sqrt(np.sum(wv * d['bt_w'].to_numpy() ** 2) / wv.sum())),
    }


def estimate(df: pl.DataFrame, shocks: pl.DataFrame, k: int, **controls) -> dict:
    z = design_matrix(df, **controls)
    y, x, w = df['dv'].to_numpy(), df['b_tilde'].to_numpy(), df['w'].to_numpy()
    fit = p1.fit(y, x, z, w, df['state_fips'].to_numpy())
    index = {s: i for i, s in enumerate(shocks['state_fips'].to_list())}
    test = ri.RandomizationTest(y, df['e_c'].to_numpy(), df['state_fips'].replace_strict(index).to_numpy(),
                                z, w, df['state_fips'].to_numpy(),
                                shocks['deviation'].to_numpy(), shocks['padd_group'].to_numpy(), k=k)
    r = test.run(fit.beta, fit.se_cr1)
    return {'beta': fit.beta, 'se_cr1': fit.se_cr1, 't_cr1': fit.t, 'p_ri': r.p_value,
            'ci95_lo': r.ci95[0], 'ci95_hi': r.ci95[1], 'ci90_lo': r.ci90[0], 'ci90_hi': r.ci90[1],
            'eps_star': r.eps_star, 'n_rows': fit.n, 'n_states': fit.n_clusters, 'n_params': fit.n_params,
            'n_races': df['race_id'].n_unique(), 'n_counties': df['geo_id'].n_unique(),
            'k_draws': r.k_draws, 'seed': r.seed, 'ri_flags': ';'.join(r.flags)}


def run(cycle: int, k: int = ri.K_DRAWS) -> pl.DataFrame:
    """Primary = C1 (memo §6.2, PI 2026-10-19); C0 reported alongside. C1 is never silently replaced."""
    df, shocks, dropped = frame(cycle, PRIMARY)
    df_veh, _, _ = frame(cycle, 'vehicles')
    c1 = {'precision': C1}
    if df['c1_swing'].null_count():
        primary = ('primary_C1_unavailable', f"C1 missing for {df['c1_swing'].null_count()} rows; C0 shown",
                   df, {})
        c1 = {}
    else:
        primary = ('primary', 'C1; commute + vehicles exposure', df, c1)
    specs = [primary,
             ('C0', 'design controls only', df, {}),
             ('R19', 'vehicles-only exposure', df_veh, c1),
             ('R17', 'without lagged burden', df, {'lag': False, **c1})]
    if df['mu_pt'].null_count() == 0:
        specs.append(('R5', 'without pass-through expected burden', df, {'pt': False, **c1}))
    if df['filled'].any():
        specs.append(('filled_out', 'drop Senate cells filled from the MEDSL county file',
                      df.filter(~pl.col('filled')), c1))
    rows = []
    for spec, label, d, controls in specs:
        rows.append({'cycle': cycle, 'spec': spec, 'label': label, **estimate(d, shocks, k, **controls),
                     **diagnostics(d), 'dropped_no_burden': ';'.join(dropped)})
    return pl.DataFrame(rows)


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def main(argv: list[str]) -> int:
    for cycle in [int(a) for a in argv] or [2022]:
        table = run(cycle)
        out = paths.repo_root() / 'reports' / 'validation' / f'V_{cycle}.csv'
        table.write_csv(out)
        inputs = ['exposure_county.parquet', 'shocks.parquet', 'results_county.parquet']
        runlog.append_run_log(
            run_id=uuid.uuid4().hex[:12],
            utc_timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            git_sha=runlog.git_sha(), git_dirty=runlog.git_dirty(), config_hash='', spec_id=f'V_{cycle}',
            data_snapshot_ids=';'.join(f'{f}:{_sha(paths.processed_dir() / f)}' for f in inputs),
            output_path=str(out.relative_to(paths.repo_root())),
            notes=f'primary C1 (C0 alongside); K={ri.K_DRAWS}; seed={ri.SEED}; memo D18')
        print(cycle, table.select('spec', 'beta', 'se_cr1', 'p_ri', 'ci95_lo', 'ci95_hi', 'n_rows', 'n_races'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
