"""Measurement error from thin pre-election windows (2016: 2 days, 2018: 5 days).

Uses the complete 2020, 2022 and 2024 windows as truth. For each, the state
window mean is recomputed from a subset of days, either the exact day
positions observed in 2016/2018 or random subsets, and compared with the
full-window mean. The error that matters for the design is in within-PADD
deviations, since recentering removes anything common to a PADD group.

Uses treatment data only (prices), no outcomes.

    python -m fb26.build.thin_windows
"""
from datetime import date, datetime, timedelta, timezone
import math
import random
import statistics
import uuid

import polars as pl
import yaml

from fb26.util import paths, runlog

SEED = 20261017
N_DRAWS = 2000
TRUTH_WINDOWS = (2020, 2022, 2024)
# 0-based day positions within the 21-day window that are observed for every state.
OBSERVED_PATTERNS = {'2016_actual': [15, 16], '2018_actual': [2, 6, 10, 14, 15]}
PRICE_FILES = ('prices_state_daily_aaagas.parquet', 'prices_state_daily_wayback.parquet')


def load_prices() -> dict[tuple[date, str], float]:
    frames = [pl.read_parquet(paths.interim_dir() / f) for f in PRICE_FILES]
    df = (pl.concat(frames, how='diagonal_relaxed')
          .filter((pl.col('product') == 'regular') & ~pl.col('derived_from_lag_field'))
          .group_by('as_of_date', 'state_abbr').agg(pl.col('price_usd').first()))
    return {(d, s): p for d, s, p in df.select('as_of_date', 'state_abbr', 'price_usd').iter_rows()}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def within_group(values: dict[str, float], groups: dict[str, str]) -> dict[str, float]:
    gmean = {g: mean([values[s] for s in values if groups[s] == g]) for g in set(groups.values())}
    return {s: values[s] - gmean[groups[s]] for s in values}


def main() -> None:
    states_df = pl.read_csv(paths.config_dir() / 'states.csv', schema_overrides={'state_fips': pl.Utf8})
    states = states_df['state_abbr'].to_list()
    groups = dict(zip(states, states_df['padd_group'].cast(pl.Utf8).to_list()))
    win = yaml.safe_load((paths.config_dir() / 'windows.yaml').read_text())['elections']
    prices = load_prices()
    rng = random.Random(SEED)

    # days x states, None where missing
    mats = {}
    for y in TRUTH_WINDOWS:
        start = datetime.fromisoformat(win[y]['start']).date()
        n = (datetime.fromisoformat(win[y]['end']).date() - start).days + 1
        mats[y] = [[prices.get((start + timedelta(i), s)) for s in states] for i in range(n)]

    rows = []
    full_means = {}
    for y, m in mats.items():
        full = {s: mean([row[j] for row in m]) for j, s in enumerate(states)}
        full_means[y] = full
        full_dev = within_group(full, groups)
        observed = [i for i, row in enumerate(m) if all(v is not None for v in row)]
        designs = {name: [pos] for name, pos in OBSERVED_PATTERNS.items()}
        for k in (2, 5):
            designs[f'random_{k}'] = [sorted(rng.sample(observed, k)) for _ in range(N_DRAWS)]
            designs[f'consecutive_{k}'] = [list(range(s, s + k)) for s in range(len(m) - k + 1)]
        for name, subsets in designs.items():
            raw, dev = [], []
            for pos in subsets:
                sub = {s: mean([m[i][j] for i in pos]) for j, s in enumerate(states)}
                sub_dev = within_group(sub, groups)
                raw += [sub[s] - full[s] for s in states]
                dev += [sub_dev[s] - full_dev[s] for s in states]
            rows.append(dict(truth_window=y, design=name, n_subsets=len(subsets), k=len(subsets[0]),
                             raw_mae_usd=mean([abs(x) for x in raw]), raw_max_abs_usd=max(abs(x) for x in raw),
                             dev_error_sd_usd=statistics.pstdev(dev), dev_max_abs_usd=max(abs(x) for x in dev)))

    # Signal: dispersion of within-PADD shock deviations in the fully observed cycles.
    signal = {}
    for t, b in ((2022, 2020), (2024, 2020)):
        g = {s: full_means[t][s] - full_means[b][s] for s in states}
        signal[f'{t}-{b}'] = statistics.pstdev(within_group(g, groups).values())

    out = pl.DataFrame(rows)
    err = {d: out.filter(pl.col('design') == d)['dev_error_sd_usd'].mean() for d in out['design'].unique()}
    summary = []
    for cycle, sd in signal.items():
        for label, var_err in (('2020-type: 2016 side thin (actual days)', err['2016_actual'] ** 2),
                               ('2018-type: both sides thin (actual days)',
                                err['2016_actual'] ** 2 + err['2018_actual'] ** 2)):
            summary.append(dict(signal_cycle=cycle, signal_dev_sd_usd=sd, scenario=label,
                                error_sd_usd=math.sqrt(var_err), reliability=sd ** 2 / (sd ** 2 + var_err)))
    summ = pl.DataFrame(summary)

    vdir = paths.repo_root() / 'reports' / 'validation'
    out.write_csv(vdir / 'thin_window_error.csv')
    summ.write_csv(vdir / 'thin_window_reliability.csv')
    runlog.append_run_log(run_id=uuid.uuid4().hex[:12],
                          utc_timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                          git_sha=runlog.git_sha(), git_dirty=runlog.git_dirty(),
                          config_hash=runlog.config_hash([paths.config_dir() / 'windows.yaml',
                                                          paths.config_dir() / 'states.csv']),
                          spec_id='diag_thin_windows', data_snapshot_ids=';'.join(PRICE_FILES),
                          output_path='reports/validation/thin_window_error.csv;reports/validation/thin_window_reliability.csv',
                          notes=f'seed={SEED}; draws={N_DRAWS}; treatment data only')
    with pl.Config(tbl_rows=40, tbl_cols=10, float_precision=4, fmt_str_lengths=50):
        print(out.group_by('design').agg(pl.col('raw_mae_usd', 'dev_error_sd_usd', 'dev_max_abs_usd').mean()).sort('design'))
        print(summ)


if __name__ == '__main__':
    main()
