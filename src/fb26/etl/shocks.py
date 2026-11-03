"""State gasoline-price shocks, PADD recentering and the pass-through prediction.

Design authority: memo §5.1 (shock, source precedence), §5.3 (PADD groups), §5.4 (deviation,
lagged deviation, mu^pt), §11.3 (schema). Build spec: .scratch/spec_v2022_slice.md, section B.

For each cycle t in CYCLES and each of the 51 units (50 states + DC):

    delta_p   = mean_W(t) p - mean_W(b(t)) p        W(ED) = [ED - 21, ED - 1]
    deviation = delta_p - mean of delta_p over the available members of the PADD sub-district
    g_pt      = a_s + b_s * gbar_nat,t                midterm cycles only

Daily price per state and date (AAA state average, regular, tax-inclusive):
  1. direct observations from any source (derived_from_lag_field == False); sources are collapsed
     first (one value per source), then averaged across sources. A state-date whose sources differ
     by more than TIE_TOL is flagged. (This mean is the proposed tie-break; the PI fixes it before freeze.)
  2. lag-field values only on dates with no direct value (shortest lag field wins), flagged.
  3. EIA weekly state series only for a state-window with no AAA observation at all.

Missing state-windows stay missing. Nothing is imputed.

    python -m fb26.etl.shocks            # build shocks.parquet and shock_sources.csv
    python -m fb26.etl.shocks sanity     # print the sanity checks for the return note
"""
from datetime import date, datetime, timedelta, timezone
import hashlib
import math
import statistics
import sys
import uuid

import polars as pl
import yaml

from fb26.util import paths, runlog

CODE_VERSION = 'shocks-0.1.0'
CYCLES = (2018, 2020, 2022, 2024, 2026)
PROVISIONAL_CYCLES = (2026,)
TIE_TOL = 0.001          # USD; direct sources within this are "agreeing"
JAN_KEY = 10000          # window key of the January window of year t in the S-a build: JAN_KEY + t
PEAK_KEY = 20000         # window key of the per-state peak window of year t in the S-b build: PEAK_KEY + t
KY_KEY = 30000           # window key of the Nov 1 (y-1) to Oct 31 (y) block in the 12-month build: KY_KEY + y
KY_MIN_SHARE = 0.8       # a 12-month block needs at least this share of its days (EIA: weeks) observed
KY_CYCLES = (2022, 2024, 2026)
KY_GAP_FLAG_DAYS = 30    # a run of this many missing days or more inside a block is flagged (seasonal bias risk)
PEAK_SPAN = 28           # S-b: highest mean over any 28 consecutive days of year t ...
PEAK_MIN_OBS = 21        # ... with at least 21 of the 28 days observed ...
PEAK_GAP = 22            # ... that ends by ED - 22 (no overlap with the 21-day run-up window)
DEFAULT_WINDOW_DAYS = 21  # W(ED) = [ED - N, ED - 1]; memo §5.1 fixes N = 21
THIN_DAYS = 15           # at N = 21: fewer observed days than this flags thin_window (as in the wayback audit)


def thin_days(n: int = DEFAULT_WINDOW_DAYS) -> int:
    """Thin-window cut scaled with the window length: 15 of 21 days (rounded up), i.e. 5 at N=7, 10 at N=14."""
    return -(-THIN_DAYS * n // DEFAULT_WINDOW_DAYS)


MIN_GROUP_SHARE = 0.5    # a PADD group mean needs at least this share of its members with a shock
SEDS_FIRST_YEAR = 1990   # memo §5.4
PRICE_FILES = ('prices_state_daily_aaagas.parquet', 'prices_state_daily_wayback.parquet')
EIA_FILE = 'prices_eia_weekly.parquet'
SEDS_FILE = 'seds_motor_gasoline_prices.parquet'
LAG_RANK = {'yesterday': 1, 'week_ago': 2, 'month_ago': 3, 'year_ago': 4}
SHOCK_PRICE_CONCEPT = 'pump'

OUT_SCHEMA = {
    'year': pl.Int32, 'area_type': pl.Utf8, 'area_id': pl.Utf8, 'state_abbr': pl.Utf8, 'product': pl.Utf8,
    'price_concept': pl.Utf8, 'window_id': pl.Utf8, 'cycle_type': pl.Utf8, 'baseline_year': pl.Int32,
    'window_years': pl.Int32, 'delta_p': pl.Float64, 'mean_post': pl.Float64, 'mean_pre': pl.Float64,
    'n_days_post': pl.Int32, 'n_days_pre': pl.Int32, 'delta_p_direct_only': pl.Float64, 'delta_p_common_days': pl.Float64,
    'source_used': pl.Utf8, 'archives_post': pl.Utf8, 'archives_pre': pl.Utf8,
    'padd_group': pl.Utf8, 'group_mean': pl.Float64, 'deviation': pl.Float64, 'group_n_used': pl.Int32,
    'group_mean_ex_ak': pl.Float64, 'deviation_ex_ak': pl.Float64,
    'lag_window_id': pl.Utf8, 'lag_delta_p': pl.Float64, 'lag_group_mean': pl.Float64,
    'lag_deviation': pl.Float64, 'lag_source_used': pl.Utf8,
    'lag_n_days_post': pl.Int32, 'lag_n_days_pre': pl.Int32,
    'g_nat': pl.Float64, 'g_pt': pl.Float64, 'pt_intercept': pl.Float64, 'pt_slope': pl.Float64,
    'pt_r2': pl.Float64, 'pt_n_pairs': pl.Int32,
    'provisional': pl.Boolean, 'flags': pl.Utf8, 'price_inputs': pl.Utf8, 'code_version': pl.Utf8,
}


# ---------------------------------------------------------------- inputs

def sha256_file(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config() -> dict:
    return yaml.safe_load((paths.config_dir() / 'windows.yaml').read_text())['elections']


def load_units() -> pl.DataFrame:
    """51 units with FIPS and PADD sub-district, from the S24 membership file."""
    padd = pl.read_csv(paths.interim_dir() / 'padd_membership.csv', schema_overrides={'padd_group': pl.Utf8})
    fips = pl.read_csv(paths.config_dir() / 'states.csv', schema_overrides={'state_fips': pl.Utf8})
    units = (padd.select(pl.col('state').alias('state_abbr'), 'padd_group')
             .join(fips.select('state_abbr', 'state_fips'), on='state_abbr', how='left')
             .sort('state_abbr'))
    assert units.height == 51 and units['state_fips'].null_count() == 0, 'expected 51 units with FIPS'
    return units


def load_aaa_prices() -> pl.DataFrame:
    frames = [pl.read_parquet(paths.interim_dir() / f) for f in PRICE_FILES]
    return (pl.concat(frames, how='diagonal_relaxed')
            .filter(pl.col('product') == 'regular')
            .select('source', 'as_of_date', 'state_abbr', 'price_usd', 'derived_from_lag_field', 'lag_field'))


def window_bounds(win: dict, year: int, n: int = DEFAULT_WINDOW_DAYS) -> tuple[date, date]:
    """[ED - n, ED - 1]; at n = 21 this equals the start/end in config/windows.yaml."""
    ed = date.fromisoformat(str(win[year]['election_day']))
    return ed - timedelta(days=n), ed - timedelta(days=1)


def window_open(aaa: pl.DataFrame, win: dict, year: int, n: int = DEFAULT_WINDOW_DAYS, ky: bool = False) -> bool:
    """A provisional cycle stays provisional until the AAA state prices reach the last day of its post window
    (W(ED) = [ED - n, ED - 1]; Oct 31 of the year for the 12-month block), as `substate_shocks` does for counties."""
    if year not in PROVISIONAL_CYCLES:
        return False
    end = date(year, 10, 31) if ky else window_bounds(win, year, n)[1]
    last = aaa['as_of_date'].max() if aaa.height else None
    return last is None or last < end


# ---------------------------------------------------------------- daily values

def daily_values(prices: pl.DataFrame, tol: float = TIE_TOL) -> pl.DataFrame:
    """One AAA price per (state, date) with its provenance.

    Columns: state_abbr, as_of_date, value, kind ('direct' | 'lag'), n_direct_sources, direct_range,
    within_source_range, lag_field_used, lag_range, lag_direct_diff.
    """
    direct = prices.filter(~pl.col('derived_from_lag_field'))
    per_source = direct.group_by('as_of_date', 'state_abbr', 'source', maintain_order=True).agg(
        pl.col('price_usd').mean().alias('p'),
        (pl.col('price_usd').max() - pl.col('price_usd').min()).alias('within'))
    d = per_source.group_by('as_of_date', 'state_abbr', maintain_order=True).agg(
        pl.col('p').mean().alias('direct_value'),
        pl.col('source').n_unique().alias('n_direct_sources'),
        (pl.col('p').max() - pl.col('p').min()).alias('direct_range'),
        pl.col('within').max().alias('within_source_range'))

    lag = prices.filter(pl.col('derived_from_lag_field')).with_columns(
        pl.col('lag_field').replace_strict(LAG_RANK, default=99).alias('rank'))
    lag_best = (lag.with_columns(pl.col('rank').min().over('as_of_date', 'state_abbr').alias('rmin'))
                .group_by('as_of_date', 'state_abbr', maintain_order=True).agg(
                    pl.col('price_usd').filter(pl.col('rank') == pl.col('rmin')).mean().alias('lag_value'),
                    pl.col('lag_field').filter(pl.col('rank') == pl.col('rmin')).first().alias('lag_field_used'),
                    (pl.col('price_usd').max() - pl.col('price_usd').min()).alias('lag_range')))

    out = d.join(lag_best, on=['as_of_date', 'state_abbr'], how='full', coalesce=True)
    return (out.with_columns(
        pl.when(pl.col('direct_value').is_not_null()).then(pl.col('direct_value')).otherwise(pl.col('lag_value')).alias('value'),
        pl.when(pl.col('direct_value').is_not_null()).then(pl.lit('direct')).otherwise(pl.lit('lag')).alias('kind'),
        (pl.col('direct_value') - pl.col('lag_value')).abs().alias('lag_direct_diff'))
        .select('state_abbr', 'as_of_date', 'value', 'kind', 'n_direct_sources', 'direct_range',
                'within_source_range', 'lag_field_used', 'lag_range', 'lag_direct_diff')
        .sort('state_abbr', 'as_of_date'))


def aaa_window_table(prices: pl.DataFrame, win: dict, year: int, tol: float = TIE_TOL,
                     n: int = DEFAULT_WINDOW_DAYS, bounds: tuple[date, date] | None = None) -> pl.DataFrame:
    """Per state: window mean and coverage diagnostics for one window year (AAA only)."""
    start, end = bounds or window_bounds(win, year, n)
    p = prices.filter(pl.col('as_of_date').is_between(start, end))
    if p.height == 0:
        return pl.DataFrame(schema={'state_abbr': pl.Utf8})
    dv = daily_values(p, tol)
    n_present = dv['state_abbr'].n_unique()
    common = dv.group_by('as_of_date', maintain_order=True).len().filter(pl.col('len') == n_present)['as_of_date'].to_list()
    stats = dv.group_by('state_abbr', maintain_order=True).agg(
        pl.col('value').filter(pl.col('as_of_date').is_in(common)).mean().alias('aaa_mean_common'),
        pl.col('as_of_date').is_in(common).sum().cast(pl.Int32).alias('aaa_n_common'),
        pl.col('value').mean().alias('aaa_mean'),
        pl.len().cast(pl.Int32).alias('aaa_n'),
        (pl.col('kind') == 'direct').sum().cast(pl.Int32).alias('aaa_n_direct'),
        (pl.col('kind') == 'lag').sum().cast(pl.Int32).alias('aaa_n_lag_only'),
        pl.col('value').filter(pl.col('kind') == 'direct').mean().alias('aaa_mean_direct_only'),
        (pl.col('direct_range') > tol + 1e-9).sum().cast(pl.Int32).alias('n_disagree_days'),
        pl.col('direct_range').max().alias('max_abs_disagreement'),
        (pl.col('within_source_range') > tol + 1e-9).sum().cast(pl.Int32).alias('n_within_source_conflict'),
        (pl.col('lag_direct_diff') > tol + 1e-9).sum().cast(pl.Int32).alias('n_lag_direct_conflict'),
        (pl.col('lag_range') > tol + 1e-9).sum().cast(pl.Int32).alias('n_lag_lag_conflict'),
        (pl.col('n_direct_sources') >= 2).sum().cast(pl.Int32).alias('multi_source_days'))
    # days observed by archive: direct by source, lag-derived by source
    cov = (p.group_by('state_abbr', 'source', 'derived_from_lag_field', maintain_order=True)
           .agg(pl.col('as_of_date').n_unique().cast(pl.Int32).alias('d'))
           .with_columns(pl.concat_str([pl.lit('n_days_'), pl.col('source'),
                                        pl.when(pl.col('derived_from_lag_field')).then(pl.lit('_lag')).otherwise(pl.lit(''))]).alias('col'))
           .pivot(on='col', index='state_abbr', values='d').fill_null(0))
    archives = (p.filter(~pl.col('derived_from_lag_field')).group_by('state_abbr', maintain_order=True)
                .agg(pl.col('source').unique().sort().str.join('+').alias('aaa_archives')))
    return stats.join(cov, on='state_abbr', how='left').join(archives, on='state_abbr', how='left')


def eia_window_table(eia: pl.DataFrame, win: dict, year: int, n: int = DEFAULT_WINDOW_DAYS,
                     bounds: tuple[date, date] | None = None) -> pl.DataFrame:
    """Per state with an EIA state series: mean of the weekly observations dated inside the window."""
    start, end = bounds or window_bounds(win, year, n)
    e = (eia.filter((pl.col('product') == 'regular') & pl.col('value').is_not_null()
                    & pl.col('period').is_between(start, end)
                    & pl.col('area_code').str.starts_with('S') & (pl.col('area_code').str.len_chars() == 3))
         .with_columns(pl.col('area_code').str.slice(1).alias('state_abbr')))
    return e.group_by('state_abbr', maintain_order=True).agg(pl.col('value').mean().alias('eia_mean'),
                                        pl.len().cast(pl.Int32).alias('eia_n'))


def build_window_table(aaa: pl.DataFrame, eia: pl.DataFrame, win: dict, years, units: pl.DataFrame,
                       n: int = DEFAULT_WINDOW_DAYS, extra: dict | None = None) -> pl.DataFrame:
    """Long table: one row per (window key, unit), AAA and EIA columns side by side. `extra` adds windows with
    explicit bounds under integer keys of their own (the January windows of S-a use JAN_KEY + year)."""
    frames = []
    keys = [(y, None) for y in years] + [(k, b) for k, b in sorted((extra or {}).items())]
    for y, bnd in keys:
        a = aaa_window_table(aaa, win, y, n=n, bounds=bnd)
        e = eia_window_table(eia, win, y, n=n, bounds=bnd)
        t = units.select('state_abbr', 'state_fips', 'padd_group').join(a, on='state_abbr', how='left')
        t = t.join(e, on='state_abbr', how='left')
        if 'aaa_n' not in t.columns:
            t = t.with_columns(pl.lit(None, pl.Int32).alias('aaa_n'), pl.lit(None, pl.Float64).alias('aaa_mean'))
        frames.append(t.with_columns(pl.lit(y).cast(pl.Int32).alias('window_year')))
    return _finalize_window_table(pl.concat(frames, how='diagonal_relaxed'))


def _finalize_window_table(out: pl.DataFrame) -> pl.DataFrame:
    return out.with_columns(
        pl.col('aaa_n').fill_null(0).cast(pl.Int32), pl.col('eia_n').fill_null(0).cast(pl.Int32),
        pl.when(pl.col('aaa_n').fill_null(0) > 0).then(pl.lit('aaa'))
        .when(pl.col('eia_n').fill_null(0) > 0).then(pl.lit('eia_weekly')).otherwise(pl.lit('none')).alias('window_source'))


def peak_spans(dv: pl.DataFrame, states: list[str], year: int, ed: date, span: int = PEAK_SPAN,
               min_obs: int = PEAK_MIN_OBS, gap: int = PEAK_GAP) -> dict:
    """R9 S-b: per state, the 28-day window of year t with the highest mean of the daily values, among windows that
    start on or after January 1 and end on or before ED - gap, and have at least min_obs observed days. Ties go
    to the earliest window. Returns {state: ((first day, last day) or None, share of candidate windows that qualify)};
    a low share means the peak search ran over a year with large holes and the maximum may have been missed."""
    first = date(year, 1, 1)
    n = (ed - timedelta(days=gap) - first).days + 1
    series: dict[str, list] = {}
    for st, d, v in dv.select('state_abbr', 'as_of_date', 'value').iter_rows():
        i = (d - first).days
        if 0 <= i < n:
            series.setdefault(st, [None] * n)[i] = v
    out = {}
    for st in states:
        arr = series.get(st)
        best = None
        n_ok = 0
        if arr is not None:
            cs, cc = [0.0], [0]
            for v in arr:
                cs.append(cs[-1] + (v if v is not None else 0.0))
                cc.append(cc[-1] + (v is not None))
            for i in range(n - span + 1):
                c = cc[i + span] - cc[i]
                if c >= min_obs:
                    n_ok += 1
                    m = (cs[i + span] - cs[i]) / c
                    if best is None or m > best[0] + 1e-12:
                        best = (m, i)
        found = None if best is None else (first + timedelta(days=best[1]), first + timedelta(days=best[1] + span - 1))
        out[st] = (found, n_ok / max(n - span + 1, 1))
    return out


def peak_window_rows(aaa: pl.DataFrame, win: dict, units: pl.DataFrame, t: int, n: int) -> pl.DataFrame:
    """Window-table rows (key PEAK_KEY + t) for each state's own peak window; states without a qualifying window get
    an empty row. Statistics come from aaa_window_table over the chosen span, as for any other window."""
    ed = date.fromisoformat(str(win[t]['election_day']))
    states = units['state_abbr'].to_list()
    lo, hi = date(t, 1, 1), ed - timedelta(days=PEAK_GAP)
    spans = peak_spans(daily_values(aaa.filter(pl.col('as_of_date').is_between(lo, hi))), states, t, ed)
    frames = []
    for span in sorted({v[0] for v in spans.values() if v[0] is not None}):
        who = [s for s in states if spans[s][0] == span]
        a = aaa_window_table(aaa, win, t, n=n, bounds=span).filter(pl.col('state_abbr').is_in(who))
        part = units.filter(pl.col('state_abbr').is_in(who)).select('state_abbr', 'state_fips', 'padd_group').join(a, on='state_abbr', how='left')
        frames.append(part.with_columns(pl.lit(span[0]).alias('peak_start'), pl.lit(span[1]).alias('peak_end')))
    none = [s for s in states if spans[s][0] is None]
    if none:
        frames.append(units.filter(pl.col('state_abbr').is_in(none)).select('state_abbr', 'state_fips', 'padd_group'))
    share = pl.DataFrame({'state_abbr': states, 'peak_qualifying_share': [spans[s][1] for s in states]})
    out = pl.concat(frames, how='diagonal_relaxed').join(share, on='state_abbr', how='left').with_columns(
        pl.lit(PEAK_KEY + t).cast(pl.Int32).alias('window_year'), pl.lit(0).cast(pl.Int32).alias('eia_n'))
    if 'aaa_n' not in out.columns:
        out = out.with_columns(pl.lit(None, pl.Int32).alias('aaa_n'))
    return _finalize_window_table(out)


# ---------------------------------------------------------------- shocks and recentering

def state_shock(w: dict, post: int, pre: int, abbr: str, thin: int | tuple[int, int] = THIN_DAYS) -> dict:
    """Shock for one state from the window table rows (dicts keyed by (window_year, abbr))."""
    a, b = w[(post, abbr)], w[(pre, abbr)]
    flags = []
    out = dict(delta_p=None, mean_post=None, mean_pre=None, n_post=None, n_pre=None, source_used=None,
               archives_post=None, archives_pre=None, delta_direct_only=None, delta_common=None)
    if a['aaa_n'] > 0 and b['aaa_n'] > 0:
        out.update(source_used='aaa', mean_post=a['aaa_mean'], mean_pre=b['aaa_mean'],
                   n_post=a['aaa_n'], n_pre=b['aaa_n'],
                   archives_post=a.get('aaa_archives'), archives_pre=b.get('aaa_archives'))
        out['delta_p'] = a['aaa_mean'] - b['aaa_mean']
        if a.get('aaa_mean_direct_only') is not None and b.get('aaa_mean_direct_only') is not None:
            out['delta_direct_only'] = a['aaa_mean_direct_only'] - b['aaa_mean_direct_only']
        if a.get('aaa_mean_common') is not None and b.get('aaa_mean_common') is not None:
            out['delta_common'] = a['aaa_mean_common'] - b['aaa_mean_common']
        for (tag, r), cut in zip((('post', a), ('pre', b)), thin if isinstance(thin, tuple) else (thin, thin)):
            if r['aaa_n'] < cut:
                flags.append(f'thin_window_{tag}')
            if r['aaa_n_lag_only']:
                flags.append(f'lag_days_used_{tag}')
            if r['n_disagree_days']:
                flags.append(f'direct_disagree_{tag}')
            if r['n_within_source_conflict']:
                flags.append(f'within_source_conflict_{tag}')
    elif a['eia_n'] > 0 and b['eia_n'] > 0 and (a['aaa_n'] == 0 or b['aaa_n'] == 0):
        # Same-source difference only: AAA and EIA levels differ by a state-specific constant (S1b audit).
        out.update(source_used='eia_weekly', mean_post=a['eia_mean'], mean_pre=b['eia_mean'],
                   n_post=a['eia_n'], n_pre=b['eia_n'])
        out['delta_p'] = a['eia_mean'] - b['eia_mean']
        flags.append('eia_fallback')
        for tag, r in (('post', a), ('pre', b)):
            if r['aaa_n'] == 0:
                flags.append(f'no_aaa_{tag}')
    else:
        flags.append('shock_missing')
        for tag, r in (('post', a), ('pre', b)):
            if r['aaa_n'] == 0 and r['eia_n'] == 0:
                flags.append(f'no_prices_{tag}')
        if (a['aaa_n'] > 0) != (b['aaa_n'] > 0) and (a['eia_n'] > 0 or b['eia_n'] > 0):
            flags.append('mixed_source_not_used')
    out['flags'] = flags
    return out


def within_group(delta: dict, groups: dict, exclude: frozenset = frozenset(),
                 min_share: float = MIN_GROUP_SHARE) -> dict:
    """Group mean (unweighted, over available members) and deviation per unit.

    A group with fewer than min_share of its members available gets no mean (deviation missing).
    `exclude` removes units from the group mean only (used for the Alaska variant); excluded units
    still receive a deviation from the mean of the others.
    """
    members: dict[str, list[str]] = {}
    for s, g in groups.items():
        members.setdefault(g, []).append(s)
    out = {}
    for g, ms in members.items():
        use = [s for s in ms if delta.get(s) is not None and s not in exclude]
        denom = len([s for s in ms if s not in exclude])
        gm = statistics.fmean(delta[s] for s in use) if use and len(use) >= min_share * denom else None
        for s in ms:
            d = delta.get(s)
            out[s] = dict(group_mean=gm, deviation=(d - gm) if (d is not None and gm is not None) else None,
                          n_used=len(use) if gm is not None else 0)
    return out


def shock_table(w: dict, units: pl.DataFrame, post: int, pre: int, thin: int | tuple[int, int] = THIN_DAYS) -> dict:
    """All units: shock, group mean, deviation (and the ex-Alaska variant) for one window pair."""
    abbrs = units['state_abbr'].to_list()
    groups = dict(zip(abbrs, units['padd_group'].to_list()))
    sh = {s: state_shock(w, post, pre, s, thin) for s in abbrs}
    delta = {s: sh[s]['delta_p'] for s in abbrs}
    wg = within_group(delta, groups)
    wg_ak = within_group(delta, groups, exclude=frozenset({'AK'}))
    for s in abbrs:
        sh[s].update(wg[s])
        sh[s]['group_mean_ex_ak'] = wg_ak[s]['group_mean']
        sh[s]['deviation_ex_ak'] = wg_ak[s]['deviation']
        if sh[s]['delta_p'] is not None and sh[s]['group_mean'] is None:
            sh[s]['flags'].append('group_coverage_low')
    return sh


# ---------------------------------------------------------------- pass-through (mu^pt)

def seds_two_year_changes(seds: pl.DataFrame, end_year: int, first_year: int = SEDS_FIRST_YEAR) -> pl.DataFrame:
    """Overlapping 2-year changes (y -> y+2) with y >= first_year and y+2 <= end_year, 51 units, wide by pair."""
    s = (seds.filter((pl.col('state') != 'US') & pl.col('price_usd_per_gal').is_not_null()
                     & pl.col('year').is_between(first_year, end_year))
         .select('state', 'year', 'price_usd_per_gal'))
    a = s.with_columns((pl.col('year') - 2).alias('y0')).select('state', 'y0', pl.col('price_usd_per_gal').alias('p1'))
    b = s.select('state', pl.col('year').alias('y0'), pl.col('price_usd_per_gal').alias('p0'))
    return (a.join(b, on=['state', 'y0'], how='inner')
            .with_columns((pl.col('p1') - pl.col('p0')).alias('d2'))
            .select('state', 'y0', 'd2').sort('state', 'y0'))


def passthrough_fit(changes: pl.DataFrame) -> dict:
    """Per-state OLS of the state's 2-year change on the unweighted mean 2-year change (closed form)."""
    nat = changes.group_by('y0', maintain_order=True).agg(pl.col('d2').mean().alias('x'), pl.len().alias('n'))
    full = nat.filter(pl.col('n') == nat['n'].max())  # pairs where all units are present
    fits = {}
    for state, grp in changes.join(full.select('y0', 'x'), on='y0').sort('state', 'y0').group_by('state', maintain_order=True):
        xs, ys = grp['x'].to_list(), grp['d2'].to_list()
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        syy = sum((y - my) ** 2 for y in ys)
        slope = sxy / sxx
        icpt = my - slope * mx
        r2 = (sxy * sxy) / (sxx * syy) if syy > 0 else None
        fits[state[0]] = dict(intercept=icpt, slope=slope, r2=r2, n=n)
    return fits


# ---------------------------------------------------------------- assembly

def longest_gaps(prices: pl.DataFrame, bounds: tuple[date, date]) -> dict:
    """Per state: the longest run of consecutive days inside `bounds` without an AAA daily value (all days if none)."""
    a, b = bounds
    dv = daily_values(prices.filter(pl.col('as_of_date').is_between(a, b)))
    have: dict[str, set] = {}
    for st, d in dv.select('state_abbr', 'as_of_date').iter_rows():
        have.setdefault(st, set()).add(d)
    out = {}
    for st in prices['state_abbr'].unique().to_list():
        run = best = 0
        for i in range((b - a).days + 1):
            run = 0 if (a + timedelta(days=i)) in have.get(st, ()) else run + 1
            best = max(best, run)
        out[st] = best
    return out


def apply_coverage_rule(wt: pl.DataFrame, min_share: float = KY_MIN_SHARE) -> pl.DataFrame:
    """Blocks with `block_days`: an AAA (EIA: weekly) mean counts only if at least min_share of the block's days (weeks)
    were observed; otherwise the source is treated as absent for that block (shock missing, nothing filled)."""
    t = wt.with_columns(
        pl.when(pl.col('block_days').is_null() | (pl.col('aaa_n') >= min_share * pl.col('block_days')))
        .then(pl.col('aaa_n')).otherwise(0).cast(pl.Int32).alias('aaa_n'),
        pl.when(pl.col('block_days').is_null() | (pl.col('eia_n') >= min_share * pl.col('block_days') / 7))
        .then(pl.col('eia_n')).otherwise(0).cast(pl.Int32).alias('eia_n'))
    return _finalize_window_table(t)


def window_dicts(wt: pl.DataFrame) -> dict:
    return {(r['window_year'], r['state_abbr']): r for r in wt.iter_rows(named=True)}


def build(aaa: pl.DataFrame, eia: pl.DataFrame, seds: pl.DataFrame, win: dict, units: pl.DataFrame,
          price_inputs: str = '', window_days: int = DEFAULT_WINDOW_DAYS,
          baseline: str = 'year', cycles: tuple | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (shocks, window_table). window_days = N in W(ED) = [ED - N, ED - 1]; the default 21 is the memo's.

    baseline='year' (default): the baseline is W(b(t)), the same window before the previous presidential election.
    baseline='ky12' (Kim & Yang 2021): the shock is the mean over Nov 1 (t-1) to Oct 31 (t) minus the mean over Nov 1 (t-2)
    to Oct 31 (t-1), cycles 2022, 2024 and 2026; a block needs 80% of its days. The lagged control is the same change
    two years earlier. g_pt is null (flag pt_pending_hf). Block windows are calendar blocks, not N-day windows.
    baseline='peak' (R9 S-b): the baseline is each state's highest 28-day mean in year t among windows with at least 21
    observed days that end by ED - 22; the chosen span is returned in peak_start / peak_end. g_pt is null, the lagged
    control keeps its primary definition, and the EIA fallback is not applied to the peak window.
    baseline='jan' (R9 S-a, AGENT_BRIEF §6.3): the baseline is the January 1-31 mean of year t itself; the shock is
    a within-year change, so window_years is 0 and the 2-year pass-through prediction is not defined (g_pt null).
    The lagged control keeps its primary definition (4-year change of N-day windows).
    """
    jan = baseline == 'jan'
    peak = baseline == 'peak'
    ky = baseline == 'ky12'
    cycles = tuple(cycles or (KY_CYCLES if ky else CYCLES))
    thin = thin_days(window_days)
    wyears = [] if ky else sorted({y for t in cycles for y in (t, win[t]['baseline'], *win[t]['lag_window'])})
    extra = {JAN_KEY + t: (date(t, 1, 1), date(t, 1, 31)) for t in cycles} if jan else None
    if ky:
        extra = {KY_KEY + y: (date(y - 1, 11, 1), date(y, 10, 31)) for y in sorted({y for t in cycles for y in range(t - 3, t + 1)})}
    wt = build_window_table(aaa, eia, win, wyears, units, window_days, extra)
    if peak:
        wt = pl.concat([wt, *(peak_window_rows(aaa, win, units, t, window_days) for t in cycles)], how='diagonal_relaxed')
    w_raw = window_dicts(wt)
    if ky:
        days = {k: (b - a).days + 1 for k, (a, b) in extra.items()}
        wt = wt.with_columns(pl.col('window_year').replace_strict(days, default=None, return_dtype=pl.Int64).alias('block_days'))
        gaps = pl.DataFrame([dict(window_year=k, state_abbr=st, max_gap_days=g) for k, bnd in extra.items()
                             for st, g in longest_gaps(aaa, bnd).items()],
                            schema={'window_year': pl.Int32, 'state_abbr': pl.Utf8, 'max_gap_days': pl.Int64})
        wt = wt.join(gaps, on=['window_year', 'state_abbr'], how='left')
        w = window_dicts(apply_coverage_rule(wt))
        w_raw = window_dicts(wt)
    else:
        w = w_raw
    abbrs = units['state_abbr'].to_list()
    fips = dict(zip(abbrs, units['state_fips']))
    groups = dict(zip(abbrs, units['padd_group']))
    rows = []
    for t in cycles:
        open_t = window_open(aaa, win, t, window_days, ky)
        if ky:
            post_key, b = KY_KEY + t, KY_KEY + t - 1
            hi, lo = KY_KEY + t - 2, KY_KEY + t - 3
            sh = shock_table(w, units, post_key, b, 0)
            lag = shock_table(w, units, hi, lo, 0)
        else:
            b = JAN_KEY + t if jan else PEAK_KEY + t if peak else win[t]['baseline']
            lo, hi = win[t]['lag_window']
            sh = shock_table(w, units, t, b, (thin, thin_days(31)) if jan else (thin, thin_days(PEAK_SPAN)) if peak else thin)
            lag = shock_table(w, units, hi, lo, thin)
        valid = [sh[s]['delta_p'] for s in abbrs if sh[s]['delta_p'] is not None]
        g_nat = statistics.fmean(valid) if valid else None
        midterm = win[t]['type'] == 'midterm' and not (jan or peak or ky)
        fits = passthrough_fit(seds_two_year_changes(seds, b)) if midterm else {}
        for s in abbrs:
            r, lg = sh[s], lag[s]
            flags = list(r['flags'])
            if open_t:
                flags.append('provisional_window_open')
            if s == 'AK':
                flags.append('alaska_not_in_sample')
            if lg['delta_p'] is not None:
                if lg['source_used'] == 'eia_weekly':
                    flags.append('lag_eia_fallback')
                if any(f.startswith('thin_window') for f in lg['flags']):
                    flags.append('lag_thin_window')
            if lg['deviation'] is None:
                flags.append('lag_deviation_missing')
                if lg['delta_p'] is not None:
                    flags.append('lag_group_coverage_low')
            g_pt = icpt = slope = r2 = n_pairs = None
            if midterm:
                f = fits.get(s)
                if f and g_nat is not None:
                    icpt, slope, r2, n_pairs = f['intercept'], f['slope'], f['r2'], f['n']
                    g_pt = icpt + slope * g_nat
                if g_nat is not None and len(valid) < 51:
                    flags.append('g_nat_partial_units')
            elif ky:
                flags.append('pt_pending_hf')
                for tag, key in (('post', post_key), ('pre', b)):
                    raw, ok = w_raw[(key, s)], w[(key, s)]
                    if (raw['aaa_n'] > 0 and ok['aaa_n'] == 0) or (raw['eia_n'] > 0 and ok['eia_n'] == 0):
                        flags.append(f'ky_coverage_below_80pct_{tag}')
                    if r['source_used'] == 'aaa' and raw['max_gap_days'] >= KY_GAP_FLAG_DAYS:
                        flags.append(f'ky_gap_30d_or_more_{tag}')
                for tag, key in (('post', hi), ('pre', lo)):
                    raw, ok = w_raw[(key, s)], w[(key, s)]
                    if (raw['aaa_n'] > 0 and ok['aaa_n'] == 0) or (raw['eia_n'] > 0 and ok['eia_n'] == 0):
                        flags.append(f'lag_ky_coverage_below_80pct_{tag}')
            elif jan or peak:
                flags.append('pt_not_defined_s_a' if jan else 'pt_not_defined_s_b')
                if peak and w[(b, s)]['aaa_n'] == 0:
                    flags.append('no_peak_window')
                if peak and w[(b, s)]['peak_qualifying_share'] < OK_SHARE:
                    flags.append('peak_search_incomplete')
            else:
                flags.append('pt_window_unspecified')
            rows.append(dict(
                year=t, area_type='state', area_id=fips[s], state_abbr=s, product='regular',
                price_concept=SHOCK_PRICE_CONCEPT, window_id=f'ky12-{t}' if ky else f'{t}-jan' if jan else f'{t}-peak' if peak else f'{t}-{b}',
                cycle_type=win[t]['type'], baseline_year=t - 1 if ky else t if (jan or peak) else b,
                window_years=1 if ky else 0 if (jan or peak) else t - b, delta_p=r['delta_p'], mean_post=r['mean_post'],
                mean_pre=r['mean_pre'], n_days_post=r['n_post'], n_days_pre=r['n_pre'],
                delta_p_direct_only=r['delta_direct_only'], delta_p_common_days=r['delta_common'],
                source_used=r['source_used'],
                archives_post=r['archives_post'], archives_pre=r['archives_pre'],
                padd_group=groups[s], group_mean=r['group_mean'], deviation=r['deviation'],
                group_n_used=r['n_used'], group_mean_ex_ak=r['group_mean_ex_ak'],
                deviation_ex_ak=r['deviation_ex_ak'], lag_window_id=f'ky12-{hi - KY_KEY}-{lo - KY_KEY}' if ky else f'{hi}-{lo}',
                lag_delta_p=lg['delta_p'], lag_group_mean=lg['group_mean'], lag_deviation=lg['deviation'],
                lag_source_used=lg['source_used'], lag_n_days_post=lg['n_post'], lag_n_days_pre=lg['n_pre'],
                g_nat=g_nat, g_pt=g_pt, pt_intercept=icpt, pt_slope=slope, pt_r2=r2, pt_n_pairs=n_pairs,
                provisional=open_t, flags=';'.join(sorted(set(flags))),
                price_inputs=price_inputs, code_version=CODE_VERSION))
    out = pl.DataFrame(rows, schema=OUT_SCHEMA).sort('year', 'state_abbr')
    if window_days != DEFAULT_WINDOW_DAYS and not ky:  # the default file keeps its schema (byte-identical rebuilds)
        out = out.with_columns(pl.lit(window_days).cast(pl.Int32).alias('window_days'))
    if ky:
        cov = []
        for t, s, src in zip(out['year'], out['state_abbr'], out['source_used']):
            row = []
            for key in (KY_KEY + t, KY_KEY + t - 1):
                r = w_raw[(key, s)]
                row.append(None if src is None else r['aaa_n'] / r['block_days'] if src == 'aaa' else r['eia_n'] / (r['block_days'] / 7))
            cov.append(row)
        gap = [[w_raw[(KY_KEY + t - k, s)]['max_gap_days'] if src == 'aaa' else None for k in (0, 1)]
               for t, s, src in zip(out['year'], out['state_abbr'], out['source_used'])]
        out = out.with_columns(pl.lit('KY12').alias('window_spec'),
                               pl.Series('max_gap_days_post', [g[0] for g in gap], dtype=pl.Int64),
                               pl.Series('max_gap_days_pre', [g[1] for g in gap], dtype=pl.Int64),
                               pl.Series('coverage_post', [c[0] for c in cov], dtype=pl.Float64),
                               pl.Series('coverage_pre', [c[1] for c in cov], dtype=pl.Float64))
    if jan or peak:
        out = out.with_columns(pl.lit(window_days).cast(pl.Int32).alias('window_days'),
                               pl.lit('S-a' if jan else 'S-b').alias('window_spec'))
    if peak:
        span = {(y, s): (w[(PEAK_KEY + y, s)].get('peak_start'), w[(PEAK_KEY + y, s)].get('peak_end'),
                         w[(PEAK_KEY + y, s)].get('peak_qualifying_share'))
                for y in cycles for s in abbrs}
        keys = list(zip(out['year'].to_list(), out['state_abbr'].to_list()))
        out = out.with_columns(pl.Series('peak_start', [span[k][0] for k in keys], dtype=pl.Date),
                               pl.Series('peak_end', [span[k][1] for k in keys], dtype=pl.Date),
                               pl.Series('peak_qualifying_share', [span[k][2] for k in keys], dtype=pl.Float64))
    return out, wt


def sources_table(wt: pl.DataFrame, win: dict, window_days: int = DEFAULT_WINDOW_DAYS) -> pl.DataFrame:
    """Per state-window: days observed by source, disagreements, lag-derived days (shock_sources.csv)."""
    cols = sorted(c for c in wt.columns if c.startswith('n_days_'))
    base = ['window_year', 'block_days', 'max_gap_days', 'peak_start', 'peak_end', 'peak_qualifying_share', 'state_abbr', 'state_fips', 'padd_group', 'window_source', 'aaa_n', 'aaa_n_direct',
            'aaa_n_lag_only', 'aaa_n_common', 'multi_source_days', 'n_disagree_days', 'max_abs_disagreement',
            'n_within_source_conflict', 'n_lag_direct_conflict', 'n_lag_lag_conflict', 'eia_n', 'aaa_archives']
    t = wt.select([c for c in base if c in wt.columns] + cols)
    if 'aaa_archives' not in t.columns:
        t = t.with_columns(pl.lit(None, pl.Utf8).alias('aaa_archives'))
    starts = {y: window_bounds(win, y, window_days)[0].isoformat() for y in win}
    ends = {y: window_bounds(win, y, window_days)[1].isoformat() for y in win}
    cut = {y: thin_days(window_days) for y in win}
    jan_keys = [k for k in wt['window_year'].unique().to_list() if k >= JAN_KEY]
    for k in jan_keys:
        if k >= KY_KEY:    # 12-month block Nov 1 (y-1) to Oct 31 (y), y = k - KY_KEY
            a, b = date(k - KY_KEY - 1, 11, 1), date(k - KY_KEY, 10, 31)
            starts[k], ends[k], cut[k] = a.isoformat(), b.isoformat(), -(-int(KY_MIN_SHARE * 100) * ((b - a).days + 1) // 100)
        elif k < PEAK_KEY:   # S-a: January window of year k - JAN_KEY
            starts[k], ends[k], cut[k] = date(k - JAN_KEY, 1, 1).isoformat(), date(k - JAN_KEY, 1, 31).isoformat(), thin_days(31)
        else:              # S-b: per-state peak window; dates are filled from peak_start / peak_end below
            starts[k], ends[k], cut[k] = '', '', PEAK_MIN_OBS
    t = t.with_columns(pl.col('window_year').replace_strict(starts, return_dtype=pl.Utf8).alias('window_start'),
                       pl.col('window_year').replace_strict(ends, return_dtype=pl.Utf8).alias('window_end'),
                       (pl.col('aaa_n') < pl.col('window_year').replace_strict(cut, return_dtype=pl.Int64)).alias('thin_window'),
                       (pl.col('window_source') == 'none').alias('no_prices'))
    for c in ('n_disagree_days', 'n_within_source_conflict', 'n_lag_direct_conflict', 'n_lag_lag_conflict',
              'multi_source_days', 'aaa_n_direct', 'aaa_n_lag_only', 'aaa_n_common', *cols):
        if c in t.columns:
            t = t.with_columns(pl.col(c).fill_null(0))
    if 'peak_start' in t.columns:
        t = t.with_columns(pl.when(pl.col('peak_start').is_not_null()).then(pl.col('peak_start').dt.to_string('%Y-%m-%d'))
                           .otherwise(pl.col('window_start')).alias('window_start'),
                           pl.when(pl.col('peak_end').is_not_null()).then(pl.col('peak_end').dt.to_string('%Y-%m-%d'))
                           .otherwise(pl.col('window_end')).alias('window_end')).drop('peak_start', 'peak_end')
    if jan_keys:
        t = t.with_columns(pl.when(pl.col('window_year') >= KY_KEY).then(pl.lit('ky12_block'))
                           .when(pl.col('window_year') >= PEAK_KEY).then(pl.lit('peak_28d'))
                           .when(pl.col('window_year') >= JAN_KEY).then(pl.lit('january')).otherwise(pl.lit('pre-election'))
                           .alias('window_kind'),
                           pl.when(pl.col('window_year') >= JAN_KEY).then(pl.col('window_year') % JAN_KEY)
                           .otherwise(pl.col('window_year')).alias('window_year'))
    return t.sort(*(['window_year', 'window_kind'] if jan_keys else ['window_year']), 'state_abbr')


def main(window_days: int = DEFAULT_WINDOW_DAYS, baseline: str = 'year') -> None:
    win = load_config()
    units = load_units()
    aaa = load_aaa_prices()
    eia = pl.read_parquet(paths.interim_dir() / EIA_FILE)
    seds = pl.read_parquet(paths.interim_dir() / SEDS_FILE)
    files = [*PRICE_FILES, EIA_FILE, SEDS_FILE]
    inputs = ';'.join(f'{f}:{sha256_file(paths.interim_dir() / f)[:16]}' for f in files)

    shocks, wt = build(aaa, eia, seds, win, units, price_inputs=inputs, window_days=window_days, baseline=baseline)
    src = sources_table(wt, win, window_days)

    suffix = '_r9a' if baseline == 'jan' else '_r9b' if baseline == 'peak' else '_ky12' if baseline == 'ky12' else '' if window_days == DEFAULT_WINDOW_DAYS else f'_w{window_days}'
    out = paths.processed_dir() / f'shocks{suffix}.parquet'
    out.parent.mkdir(parents=True, exist_ok=True)
    shocks.write_parquet(out, compression='zstd', statistics=False)
    vdir = paths.repo_root() / 'reports' / 'validation'
    vdir.mkdir(parents=True, exist_ok=True)
    src.write_csv(vdir / f'shock_sources{suffix}.csv')

    cfg_files = [paths.config_dir() / 'windows.yaml', paths.config_dir() / 'states.csv',
                 paths.interim_dir() / 'padd_membership.csv']
    runlog.append_run_log(
        run_id=uuid.uuid4().hex[:12], utc_timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        git_sha=runlog.git_sha(), git_dirty=runlog.git_dirty(), config_hash=runlog.config_hash(cfg_files),
        spec_id='ws4_shocks', data_snapshot_ids=inputs,
        output_path=f'data/processed/shocks{suffix}.parquet;reports/validation/shock_sources{suffix}.csv',
        notes=f'{CODE_VERSION}; rows={shocks.height}; cycles={list(CYCLES)}; window_days={window_days}; baseline={baseline}; '
              f'tie_tol={TIE_TOL}; 2026 provisional={bool(shocks.filter(pl.col("year") == 2026)["provisional"].any())}')
    print(f'wrote {out} ({shocks.height} rows) and shock_sources{suffix}.csv ({src.height} rows)')


# ---------------------------------------------------------------- sanity checks (printed, for the return note)

def _corr(x, y) -> float | None:
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxy = sum((a - mx) * (b - my) for a, b in pairs)
    sxx, syy = sum((a - mx) ** 2 for a in xs), sum((b - my) ** 2 for b in ys)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else None


def sanity() -> None:
    from fb26.build import thin_windows
    shocks = pl.read_parquet(paths.processed_dir() / 'shocks.parquet')
    src = pl.read_csv(paths.repo_root() / 'reports' / 'validation' / 'shock_sources.csv',
                      schema_overrides={'state_fips': pl.Utf8})
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_float_precision(4)

    print('== national mean shock (USD/gal) per cycle')
    print(shocks.group_by('year', 'window_id', maintain_order=True).agg(
        pl.col('delta_p').mean().alias('mean'), pl.col('delta_p').min().alias('min'),
        pl.col('delta_p').max().alias('max'), pl.col('delta_p').null_count().alias('n_missing')).sort('year'))

    print('== deviation distribution by cycle and PADD group (sd, min, max)')
    print(shocks.group_by('year', 'padd_group', maintain_order=True).agg(
        pl.len().alias('n'), pl.col('deviation').std().alias('sd'), pl.col('deviation').min().alias('min'),
        pl.col('deviation').max().alias('max')).sort('year', 'padd_group'))

    print('== correlation of deviations across cycles (pairwise complete)')
    wide = shocks.pivot(on='year', index='state_abbr', values='deviation').sort('state_abbr')
    ys = [str(y) for y in CYCLES]
    for i, a in enumerate(ys):
        print(a, [None if _corr(wide[a], wide[b]) is None else round(_corr(wide[a], wide[b]), 3) for b in ys])
    print('== correlation of delta_p across cycles')
    wide = shocks.pivot(on='year', index='state_abbr', values='delta_p').sort('state_abbr')
    for a in ys:
        print(a, [None if _corr(wide[a], wide[b]) is None else round(_corr(wide[a], wide[b]), 3) for b in ys])
    print('== correlation of deviation with its own lagged deviation, by cycle')
    for y in CYCLES:
        s = shocks.filter(pl.col('year') == y)
        print(y, _corr(s['deviation'].to_list(), s['lag_deviation'].to_list()))

    print('== g_pt fit: per-state R2 and slope, by midterm cycle')
    print(shocks.filter(pl.col('g_pt').is_not_null()).group_by('year', maintain_order=True).agg(
        pl.col('pt_r2').min().alias('r2_min'), pl.col('pt_r2').median().alias('r2_med'),
        pl.col('pt_r2').max().alias('r2_max'), pl.col('pt_slope').min().alias('b_min'),
        pl.col('pt_slope').median().alias('b_med'), pl.col('pt_slope').max().alias('b_max'),
        pl.col('pt_intercept').median().alias('a_med'), pl.col('g_nat').first().alias('g_nat'),
        pl.col('g_pt').mean().alias('g_pt_mean'), pl.col('g_pt').min().alias('g_pt_min'),
        pl.col('g_pt').max().alias('g_pt_max'), pl.col('pt_n_pairs').first().alias('n_pairs')).sort('year'))
    print('== corr(g_pt, delta_p) and corr(g_pt - g_nat, deviation), midterm cycles')
    for y in (2018, 2022, 2026):
        s = shocks.filter(pl.col('year') == y)
        print(y, _corr(s['g_pt'].to_list(), s['delta_p'].to_list()),
              _corr((s['g_pt'] - s['g_nat']).to_list(), s['deviation'].to_list()))

    print('== cross-check 2022 deviation vs thin_windows.py (direct-only, first-source, own windows)')
    states_df = pl.read_csv(paths.config_dir() / 'states.csv', schema_overrides={'state_fips': pl.Utf8})
    groups = dict(zip(states_df['state_abbr'], states_df['padd_group'].cast(pl.Utf8)))
    win = load_config()
    prices = thin_windows.load_prices()

    def tw_mean(y):
        s0, e0 = window_bounds(win, y)
        out = {}
        for st in groups:
            v = [prices.get((s0 + timedelta(i), st)) for i in range((e0 - s0).days + 1)]
            v = [x for x in v if x is not None]
            out[st] = statistics.fmean(v) if v else None
        return out
    m22, m20 = tw_mean(2022), tw_mean(2020)
    g = {s: m22[s] - m20[s] for s in groups if m22[s] is not None and m20[s] is not None}
    dev = thin_windows.within_group(g, groups)
    mine = dict(zip(*shocks.filter(pl.col('year') == 2022).select('state_abbr', 'deviation').to_dict(as_series=False).values()))
    diffs = [abs(dev[s] - mine[s]) for s in dev if mine.get(s) is not None]
    print('n states', len(diffs), 'max |diff| USD', max(diffs), 'mean |diff|', statistics.fmean(diffs))
    print('delta_p diffs: max',
          max(abs(g[s] - v) for s, v in zip(*shocks.filter(pl.col('year') == 2022).select('state_abbr', 'delta_p').to_dict(as_series=False).values()) if s in g))

    print('== coverage: AAA days per window (min / median / max over states) and missing state-windows')
    print(src.group_by('window_year', maintain_order=True).agg(
        pl.col('aaa_n').min().alias('min'), pl.col('aaa_n').median().alias('median'), pl.col('aaa_n').max().alias('max'),
        pl.col('aaa_n_direct').median().alias('direct_med'), pl.col('aaa_n_lag_only').sum().alias('lag_only_cells'),
        pl.col('n_disagree_days').sum().alias('disagree_days'), pl.col('max_abs_disagreement').max().alias('max_disagree'),
        pl.col('n_within_source_conflict').sum().alias('within_src_conflict'),
        pl.col('n_lag_direct_conflict').sum().alias('lag_vs_direct_conflict'),
        (pl.col('window_source') == 'none').sum().alias('no_prices'),
        (pl.col('window_source') == 'eia_weekly').sum().alias('eia_only')).sort('window_year'))
    print('== flags (count of rows carrying each flag)')
    fl = shocks.select(pl.col('year'), pl.col('flags').str.split(';').alias('f')).explode('f').filter(pl.col('f') != '')
    print(fl.group_by('year', 'f', maintain_order=True).len().sort('year', 'f'))
    print('== sensitivity of delta_p to the day set: |delta_p - variant|, by cycle (n = states differing by > $0.0005)')
    for col in ('delta_p_direct_only', 'delta_p_common_days'):
        print(col)
        print(shocks.filter(pl.col(col).is_not_null()).group_by('year', maintain_order=True).agg(
            (pl.col('delta_p') - pl.col(col)).abs().max().alias('max_abs'),
            ((pl.col('delta_p') - pl.col(col)).abs() > 0.0005).sum().alias('n_differ')).sort('year'))
    print('== deviation sensitivity: max |deviation(common days) - deviation|, by cycle')
    for y in CYCLES:
        s_ = shocks.filter(pl.col('year') == y)
        groups = dict(zip(s_['state_abbr'], s_['padd_group']))
        dc = within_group(dict(zip(s_['state_abbr'], s_['delta_p_common_days'])), groups)
        print(y, max(abs(dc[a]['deviation'] - d) for a, d in zip(s_['state_abbr'], s_['deviation'])))

    print('== EIA state series vs AAA (9 states with an EIA state series): shocks by cycle')
    eia = pl.read_parquet(paths.interim_dir() / EIA_FILE)
    aaa = load_aaa_prices()
    units = load_units()
    wt = build_window_table(aaa, eia, win, [2016, 2018, 2020, 2022, 2024, 2026], units)
    w = window_dicts(wt)
    for t in CYCLES:
        b = win[t]['baseline']
        d = [(s_, w[(t, s_)]['aaa_mean'] - w[(b, s_)]['aaa_mean'], w[(t, s_)]['eia_mean'] - w[(b, s_)]['eia_mean'])
             for s_ in units['state_abbr'] if w[(t, s_)]['eia_n'] and w[(b, s_)]['eia_n']]
        if not d:
            print(t, 'no EIA state observations in both windows')
            continue
        print(t, len(d), 'corr', _corr([x[1] for x in d], [x[2] for x in d]),
              'mean(AAA-EIA) change', statistics.fmean(x[1] - x[2] for x in d),
              'max abs', max(abs(x[1] - x[2]) for x in d))


# ---------------------------------------------------------------- Internet Archive inventory (CDX only) and feasibility

INV_YEARS = (2016, 2018, 2020, 2022, 2024)   # 2026 is not a baseline year; 2024 has a 2024-04-17..09-11 hole in the scrapers
INV_PAGE_TYPES = ('gp_home', 'gp_sga', 'gp_state')
INV_LAG_ROUTES = ('current', 'yesterday', 'week_ago', 'month_ago', 'year_ago')
FEASIBILITY_N = (7, 14, 21, 28, 42, 63)
OK_SHARE = THIN_DAYS / DEFAULT_WINDOW_DAYS   # share of required days a state needs to count as "not thin"
PEAK_DAYS = 28
INV_FILE = 'wayback_inventory_extended.csv'


def _chunks(frm: date, to: date, months: int = 3):
    """Disjoint, inclusive date ranges of at most `months` calendar months."""
    from fb26.scrape.wayback import shift_months
    cur = frm
    while cur <= to:
        nxt = min(shift_months(cur, months) - timedelta(days=1), to)
        yield cur, nxt
        cur = nxt + timedelta(days=1)


def collect_captures(client, win: dict) -> dict:
    """CDX listings per (target year, page type). Index queries only; nothing is downloaded."""
    from fb26.scrape import wayback as wb
    out = {}
    for y in INV_YEARS:
        ed = date.fromisoformat(str(win[y]['election_day']))
        for pt in INV_PAGE_TYPES:
            first, last = wb.PAGE_TYPE_YEARS[pt]
            frm = date(y, 1, 1)
            to = wb.shift_months(ed, 12) if pt == 'gp_state' else ed   # state pages reach back a year via year_ago
            frm, to = max(frm, date(first, 1, 1)), min(to, date(last, 12, 31))
            snaps = {}
            if frm <= to:
                for a, b in _chunks(frm, to):
                    for sn in wb.cdx(client, pt, a, b):
                        snaps[(sn.timestamp, sn.original)] = sn
            out[(y, pt)] = sorted(snaps.values(), key=lambda sn: (sn.timestamp, sn.original))
    return out


def cdx_cells(snaps: list, pt: str, target: tuple[date, date], states: list[str]) -> dict:
    """route -> {date: {state: n_captures}} for target dates inside `target`. Home and state-averages pages give
    all states for their page date; a state page gives its state for the dates of its five rows."""
    from fb26.scrape import wayback as wb
    routes: dict[str, dict] = {}
    for sn in snaps:
        if pt in ('gp_home', 'gp_sga'):
            fields = {'current': sn.page_day}
            who = states
        else:
            fields = {r: wb.LAG_RULES[r](sn.page_day) for r in INV_LAG_ROUTES}
            who = [sn.state]
        for route, day in fields.items():
            if target[0] <= day <= target[1]:
                cell = routes.setdefault(route, {}).setdefault(day, {})
                for st in who:
                    cell[st] = cell.get(st, 0) + 1
    return routes


def _union(maps: list[dict]) -> dict:
    out: dict = {}
    for m in maps:
        for day, sts in m.items():
            cell = out.setdefault(day, {})
            for st, k in sts.items():
                cell[st] = cell.get(st, 0) + k
    return out


def _month_rows(y: int, pt: str, route: str, cov: dict, target: tuple[date, date], n_states: int) -> list[dict]:
    rows = []
    months = sorted({(d.year, d.month) for d in (target[0] + timedelta(days=i) for i in range((target[1] - target[0]).days + 1))})
    tot = dict(days=0, any=0, all51=0, caps=0, st_sum=0)
    for ym in months:
        days = [target[0] + timedelta(days=i) for i in range((target[1] - target[0]).days + 1)]
        days = [d for d in days if (d.year, d.month) == ym]
        covered = [d for d in days if d in cov]
        sizes = [len(cov[d]) for d in covered]
        # captures: for home/state-averages pages one capture covers every state, so count per state-max
        caps = sum(max(cov[d].values()) for d in covered) if covered else 0
        row = dict(target_year=y, month=f'{ym[0]}-{ym[1]:02d}', page_type=pt, route=route, days_in_range=len(days),
                   days_any=len(covered), days_all_states=sum(k >= n_states for k in sizes), captures=caps,
                   mean_states_on_covered_days=round(sum(sizes) / len(sizes), 1) if sizes else None)
        rows.append(row)
        tot['days'] += len(days); tot['any'] += len(covered); tot['all51'] += row['days_all_states']
        tot['caps'] += caps; tot['st_sum'] += sum(sizes)
    rows.append(dict(target_year=y, month='ALL', page_type=pt, route=route, days_in_range=tot['days'],
                     days_any=tot['any'], days_all_states=tot['all51'], captures=tot['caps'],
                     mean_states_on_covered_days=round(tot['st_sum'] / tot['any'], 1) if tot['any'] else None))
    return rows


def existing_cells(aaa: pl.DataFrame) -> dict:
    """date -> {state: 1} for every state-date with any AAA value (direct or lag-derived) in the interim tables."""
    dv = daily_values(aaa)
    out: dict = {}
    for st, d in dv.select('state_abbr', 'as_of_date').iter_rows():
        out.setdefault(d, {})[st] = 1
    return out


def _share(cells: dict, state: str, days: list[date]) -> float:
    return sum(state in cells.get(d, ()) for d in days) / len(days) if days else 0.0


def _min_share(cells: dict, states: list[str], days: list[date]) -> float:
    return min(_share(cells, s, days) for s in states)


def _label(share: float, thin_cut: float = OK_SHARE) -> str:
    return 'NONE' if share == 0 else 'OK' if share >= thin_cut - 1e-9 else 'THIN'


def feasibility(win: dict, now: dict, potential: dict, states: list[str]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Which cycles can be computed for each window length N and for R9's S-a and S-b.

    `now` and `potential` are {date: {state: n}}; `potential` already includes `now`. A window is "OK" when every
    state has at least 15/21 of its days (the thin cut of the 21-day window, scaled), "THIN" when every state has
    at least one day but some have fewer, "NONE" when some state has none. The cell is the worse of the two windows
    a shock needs (election window and baseline) as min over states of observed/required days.
    """
    def days_of(y, n):
        a, b = window_bounds(win, y, n)
        return [a + timedelta(days=i) for i in range((b - a).days + 1)]

    rows_p, rows_l = [], []
    for n in FEASIBILITY_N:
        for t in CYCLES:
            b = win[t]['baseline']
            lo, hi = win[t]['lag_window']
            for tbl, pair in ((rows_p, (t, b)), (rows_l, (hi, lo))):
                rec = dict(window_days=n, cycle=t, windows=f'{pair[0]}-{pair[1]}')
                for tag, cells in (('now', now), ('fetch', potential)):
                    shares = [_min_share(cells, states, days_of(y, n)) for y in pair]
                    rec[f'{tag}_min_share_post'] = round(shares[0], 2)
                    rec[f'{tag}_min_share_pre'] = round(shares[1], 2)
                    rec[tag] = _label(min(shares))
                tbl.append(rec)
    # R9: S-a (election window vs January mean of the same year) and S-b (election window vs highest 28-day mean
    # earlier in the same year; required: the days Jan 1 .. ED-22, and at least one 28-day block)
    for t in CYCLES:
        ed = date.fromisoformat(str(win[t]['election_day']))
        jan = [date(t, 1, 1) + timedelta(days=i) for i in range(31)]
        early = [date(t, 1, 1) + timedelta(days=i) for i in range((ed - timedelta(days=22) - date(t, 1, 1)).days + 1)]
        rec = dict(window_days=0, cycle=t, windows='S-a')
        for tag, cells in (('now', now), ('fetch', potential)):
            shares = [_min_share(cells, states, days_of(t, DEFAULT_WINDOW_DAYS)), _min_share(cells, states, jan)]
            rec[f'{tag}_min_share_post'], rec[f'{tag}_min_share_pre'] = round(shares[0], 2), round(shares[1], 2)
            rec[tag] = _label(min(shares))
        rows_p.append(rec)
        rec = dict(window_days=0, cycle=t, windows='S-b')
        for tag, cells in (('now', now), ('fetch', potential)):
            shares = [_min_share(cells, states, days_of(t, DEFAULT_WINDOW_DAYS)), _min_share(cells, states, early)]
            rec[f'{tag}_min_share_post'], rec[f'{tag}_min_share_pre'] = round(shares[0], 2), round(shares[1], 2)
            rec[tag] = _label(min(shares))
        rows_p.append(rec)
    return pl.DataFrame(rows_p), pl.DataFrame(rows_l)


def fetch_cost(win: dict, now: dict, caps: dict, states: list[str], target: tuple[date, date], y: int) -> dict:
    """Snapshots needed to fill the state-days of `target` (year y) that the current tables lack, using the CDX
    listing: one home or state-averages capture per missing page day, then a greedy cover of what is left with state
    pages through their lag rows (the wayback module's own planner)."""
    from fb26.scrape import wayback as wb
    days = [target[0] + timedelta(days=i) for i in range((target[1] - target[0]).days + 1)]
    missing_cells = {(d, s) for d in days for s in states if s not in now.get(d, ())}
    direct = {}
    for pt in ('gp_sga', 'gp_home'):
        for sn in caps.get((y, pt), []):
            if target[0] <= sn.page_day <= target[1]:
                direct.setdefault(sn.page_day, sn)
    direct_days = {d for d in direct if any((d, s) in missing_cells for s in states)}
    left = {c for c in missing_cells if c[0] not in direct_days}
    plan = wb.plan_lag(caps.get((y, 'gp_state'), []), target, left)
    covered = set()
    for sn in plan:
        covered |= wb.lag_cells(sn, target) & left
    return dict(missing_state_days=len(missing_cells), page_day_captures=len(direct_days),
                state_page_captures=len(plan), snapshots=len(direct_days) + len(plan),
                state_days_unreachable=len(left - covered))


def inventory_extended() -> None:
    """CDX queries (cached, through PoliteClient) -> reports/validation/wayback_inventory_extended.csv and printed
    feasibility tables. No snapshot is downloaded."""
    from fb26.scrape import wayback as wb
    win = load_config()
    states = load_units()['state_abbr'].to_list()
    client = wb.PoliteClient()
    caps = collect_captures(client, win)
    now = existing_cells(load_aaa_prices())

    rows = []
    potential = {d: dict(c) for d, c in now.items()}
    for y in INV_YEARS:
        ed = date.fromisoformat(str(win[y]['election_day']))
        target = (date(y, 1, 1), ed - timedelta(days=1))
        per_type = {}
        for pt in INV_PAGE_TYPES:
            routes = cdx_cells(caps[(y, pt)], pt, target, states)
            per_type[pt] = routes
            if pt == 'gp_state':
                for r in INV_LAG_ROUTES:
                    rows += _month_rows(y, pt, r, routes.get(r, {}), target, len(states))
                rows += _month_rows(y, pt, 'any_row', _union(list(routes.values())), target, len(states))
            else:
                rows += _month_rows(y, pt, 'current', routes.get('current', {}), target, len(states))
            if not caps[(y, pt)]:
                rows[-1]['mean_states_on_covered_days'] = None
        every = _union([m for routes in per_type.values() for m in routes.values()])
        rows += _month_rows(y, 'ALL_CDX', 'union', every, target, len(states))
        restricted_now = {d: c for d, c in now.items() if target[0] <= d <= target[1]}
        rows += _month_rows(y, 'EXISTING_TABLES', 'direct+lag', restricted_now, target, len(states))
        for d, c in every.items():
            potential.setdefault(d, {}).update({s: 1 for s in c})
        rows += _month_rows(y, 'EXISTING_PLUS_CDX', 'union', {d: v for d, v in potential.items() if target[0] <= d <= target[1]},
                            target, len(states))
    inv = pl.DataFrame(rows, schema_overrides={'mean_states_on_covered_days': pl.Float64})
    vdir = paths.repo_root() / 'reports' / 'validation'
    inv.write_csv(vdir / INV_FILE)
    print(f'wrote {vdir / INV_FILE} ({inv.height} rows)')

    n_caps = {(y, pt): len(v) for (y, pt), v in caps.items()}
    print('CDX captures listed (target year, page type):', n_caps)
    with pl.Config(tbl_rows=200, tbl_cols=20, tbl_width_chars=220):
        print(inv.filter(pl.col('month') == 'ALL').drop('month'))
        fp, fl = feasibility(win, now, potential, states)
        for title, tbl in (('PRIMARY SHOCK', fp), ('LAGGED CONTROL', fl)):
            print(title, '(label min share of required days over states, election window / baseline window)')
            for r in tbl.iter_rows(named=True):
                print(f"N={r['window_days'] or r['windows']!s:>4} {r['cycle']} {r['windows']:<10} "
                      f"now {r['now']:<4} {r['now_min_share_post']:.2f}/{r['now_min_share_pre']:.2f}   "
                      f"fetch {r['fetch']:<4} {r['fetch_min_share_post']:.2f}/{r['fetch_min_share_pre']:.2f}")
    print('FETCH COST (3 s per request floor)')
    for y in INV_YEARS:
        ed = date.fromisoformat(str(win[y]['election_day']))
        for label, tgt in [(f'N={n}', (ed - timedelta(days=n), ed - timedelta(days=1))) for n in (28, 42, 63)] + [
                ('Jan1-ED', (date(y, 1, 1), ed - timedelta(days=1)))]:
            c = fetch_cost(win, now, caps, states, tgt, y)
            print(y, label, c, f"~{c['snapshots'] * 3 / 60:.0f} min at 3 s")


def cli(argv: list[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(prog='python -m fb26.etl.shocks', description=__doc__.split('\n')[0])
    ap.add_argument('command', nargs='?', default='build', choices=['build', 'sanity', 'inventory'])
    ap.add_argument('--baseline', choices=['year', 'jan', 'peak', 'ky12'], default='year',
                    help="'jan' = R9 S-a: baseline is the January 1-31 mean of year t (*_r9a); "
                         "'peak' = R9 S-b: baseline is the highest 28-day mean of year t ending by ED - 22 (*_r9b); "
                         "'ky12' = Kim & Yang 12-month blocks Nov-Oct for 2022, 2024, 2026 (*_ky12)")
    ap.add_argument('--window-days', type=int, default=DEFAULT_WINDOW_DAYS,
                    help='N in W(ED) = [ED - N, ED - 1] (default 21; other values write *_w{N} outputs)')
    args = ap.parse_args(argv)
    if args.window_days < 1:
        ap.error('--window-days must be at least 1')
    if args.command == 'sanity':
        sanity()
    elif args.command == 'inventory':
        inventory_extended()
    else:
        main(args.window_days, args.baseline)


if __name__ == '__main__':
    cli()
