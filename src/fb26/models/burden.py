"""Burden, expected burden and recentered burden per county and cycle (memo §5.3, §5.4).

    B      = E_c × g_s,t                raw burden (percent of monthly income)
    μ      = E_c × ḡ_g,t                expected burden under within-PADD permutation
    B̃      = E_c × (g_s,t − ḡ_g,t)      recentered burden
    B̃^lag  = E_c × (g_s,τ − ḡ_g,τ)      lagged recentered burden
    μ^pt   = E_c × ĝ_s,t                pass-through expected burden (midterm cycles)

Alaska stays in its PADD group mean (memo §3: all 51 units enter the
assignment groups); its counties are flagged and left out of the sample later.

    python -m fb26.models.burden
"""
import uuid
from datetime import datetime, timezone

import polars as pl

from fb26.util import paths, runlog

CYCLES = (2018, 2020, 2022, 2024, 2026)
G_METHODS = ('commute_vehicles', 'vehicles')
OUT = 'burden_county.parquet'


def build(cycle: int, g_method: str) -> pl.DataFrame:
    exp = (pl.read_parquet(paths.processed_dir() / 'exposure_county.parquet')
           .filter(pl.col('g_method') == g_method))
    sh = (pl.read_parquet(paths.processed_dir() / 'shocks.parquet')
          .filter(pl.col('year') == cycle)
          .select(pl.col('area_id').alias('state_fips'), 'state_abbr', 'padd_group', 'delta_p',
                  'group_mean', 'deviation', 'lag_deviation', 'g_pt', 'provisional',
                  pl.col('flags').alias('shock_flags')))
    e = pl.col('e_c')
    return (exp.join(sh, on='state_fips', how='left', validate='m:1')
            .with_columns(pl.lit(cycle, pl.Int32).alias('year'))
            .with_columns(
                (e * pl.col('delta_p')).alias('b'),
                (e * pl.col('group_mean')).alias('mu'),
                (e * pl.col('deviation')).alias('b_tilde'),
                (e * pl.col('lag_deviation')).alias('b_tilde_lag'),
                (e * pl.col('g_pt')).alias('mu_pt'))
            .select('year', 'geo_id', 'state_fips', 'state_abbr', 'padd_group', 'g_method', 'e_c',
                    'delta_p', 'group_mean', 'deviation', 'lag_deviation', 'g_pt',
                    'b', 'mu', 'b_tilde', 'b_tilde_lag', 'mu_pt', 'oil_gas_emp_share',
                    'provisional', 'flags', 'shock_flags')
            .sort('geo_id'))


def main() -> None:
    df = pl.concat([build(c, m) for c in CYCLES for m in G_METHODS])
    out = paths.processed_dir() / OUT
    df.write_parquet(out)
    runlog.append_run_log(
        run_id=uuid.uuid4().hex[:12], utc_timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        git_sha=runlog.git_sha(), git_dirty=runlog.git_dirty(), config_hash='', spec_id='burden',
        data_snapshot_ids='exposure_county.parquet;shocks.parquet',
        output_path=f'data/processed/{OUT}', notes=f'rows={df.height}; cycles={list(CYCLES)}')
    print(f'{out}: {df.height} rows')


if __name__ == '__main__':
    main()
