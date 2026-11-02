"""High-frequency pass-through control (mu^pt) from daily AAA state prices.

Design authority: memo §5.4 (mu^pt), §5.1 (shock measures), §9 R9 (S-a), §12 open decision 1 (SEDS excludes sales
taxes). Replaces the SEDS annual fit in shocks.py for the measures w21, sa and ky12. Treatment side only: no outcomes.

    monthly price   = mean of the daily AAA values (shocks.daily_values rules), month needs >= MIN_DAYS observed days
    national price  = unweighted mean of the 51 monthly state prices (month missing unless all 51 are present)
    per state       : OLS  dp_s,m = a_s + b_s * dp_nat,m   over consecutive-month changes
    fit window      : months strictly before the first day of the month that contains the start of the earliest
                      price block the measure uses (no look-ahead): w21 = baseline window, sa = January of t,
                      ky12 = Nov 1 of t-2
    prediction      : g_pt = a_s * k + b_s * g_nat,   k = months spanned by the shock's change (drift scales with horizon)
                      g_nat = unweighted mean of the measure's state shocks (shocks*.parquet), 51 units

    python -m fb26.etl.pass_through           # build g_pt_hf.parquet and append the run log
    python -m fb26.etl.pass_through report    # print diagnostics for the report (no files written)
"""
from datetime import date, datetime, timezone
import math
import statistics
import sys
import uuid

import polars as pl

from fb26.etl import shocks
from fb26.util import paths, runlog

CODE_VERSION = 'pass-through-0.1.0'
MIN_DAYS = 15            # observed days a state-month needs
MIN_FIT_MONTHS = 24      # fewer monthly changes than this flags short_fit
HARD_MIN_FIT = 8         # fewer than this: no estimate (a, b, g_pt null)
MIN_RUN = 6              # a run of consecutive monthly changes shorter than this is dropped (isolated 2020 wayback pair)
MIN_NAT_SHARE = 0.9      # g_nat is null when fewer than this share of the units have a shock (46 of 51)
DAYS_PER_MONTH = 365.25 / 12
MEASURES = ('w21', 'sa', 'ky12')
SHOCK_FILES = {'w21': 'shocks.parquet', 'sa': 'shocks_r9a.parquet', 'ky12': 'shocks_ky12.parquet'}
KY_CYCLES = (2022, 2024, 2026)

OUT_SCHEMA = {
    'state_abbr': pl.Utf8, 'measure': pl.Utf8, 'year': pl.Int32, 'cycle_type': pl.Utf8,
    'a': pl.Float64, 'b': pl.Float64, 'se_b': pl.Float64, 'r2': pl.Float64, 'n_months': pl.Int32,
    'fit_window': pl.Utf8, 'fit_cutoff': pl.Date, 'span_months': pl.Float64,
    'g_nat': pl.Float64, 'g_pt': pl.Float64, 'g_pt_nodrift': pl.Float64,
    'provisional': pl.Boolean, 'flags': pl.Utf8, 'price_inputs': pl.Utf8, 'code_version': pl.Utf8,
}


# ---------------------------------------------------------------- monthly series

def month_start(d: date) -> date:
    return date(d.year, d.month, 1)


def next_month(d: date) -> date:
    return date(d.year + (d.month == 12), d.month % 12 + 1, 1)


def monthly_prices(dv: pl.DataFrame, min_days: int = MIN_DAYS) -> pl.DataFrame:
    """State-month mean of the daily values; a month with fewer than min_days observed days is dropped.

    Columns: state_abbr, month (first day), price, n_days. dv is shocks.daily_values output.
    """
    return (dv.with_columns(pl.col('as_of_date').dt.truncate('1mo').alias('month'))
            .group_by('state_abbr', 'month')
            .agg(pl.col('value').mean().alias('price'), pl.len().alias('n_days'))
            .filter(pl.col('n_days') >= min_days)
            .sort('state_abbr', 'month'))


def national_monthly(monthly: pl.DataFrame, n_units: int = 51) -> pl.DataFrame:
    """Unweighted mean over the units; a month is kept only if all n_units are present."""
    return (monthly.group_by('month').agg(pl.col('price').mean().alias('nat'), pl.len().alias('n'))
            .filter(pl.col('n') == n_units).select('month', 'nat').sort('month'))


def monthly_changes(monthly: pl.DataFrame, nat: pl.DataFrame) -> pl.DataFrame:
    """Month-on-month changes of each state and of the national mean, for consecutive calendar months only.

    Columns: state_abbr, month (the later month), dp, dp_nat.
    """
    s = monthly.select('state_abbr', 'month', 'price')
    prev = s.select('state_abbr', pl.col('month').dt.offset_by('1mo').alias('month'), pl.col('price').alias('price_prev'))
    n_prev = nat.select(pl.col('month').dt.offset_by('1mo').alias('month'), pl.col('nat').alias('nat_prev'))
    return (s.join(prev, on=['state_abbr', 'month'], how='inner')
            .join(nat, on='month', how='inner').join(n_prev, on='month', how='inner')
            .select('state_abbr', 'month', (pl.col('price') - pl.col('price_prev')).alias('dp'),
                    (pl.col('nat') - pl.col('nat_prev')).alias('dp_nat'))
            .sort('state_abbr', 'month'))


# ---------------------------------------------------------------- per-state OLS

def ols(xs: list[float], ys: list[float]) -> dict:
    """OLS of y on a constant and x with the HC1 standard error of the slope."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    syy = sum((y - my) ** 2 for y in ys)
    b = sxy / sxx
    a = my - b * mx
    res = [y - a - b * x for x, y in zip(xs, ys)]
    meat = sum(((x - mx) ** 2) * e * e for x, e in zip(xs, res))
    se = math.sqrt(n / (n - 2) * meat) / sxx
    r2 = sxy * sxy / (sxx * syy) if syy > 0 else None
    return dict(a=a, b=b, se_b=se, r2=r2, n=n)


def fit_states(changes: pl.DataFrame, cutoff: date, hard_min: int = HARD_MIN_FIT) -> dict:
    """Per-state fit on the changes whose later month lies strictly before cutoff.

    Runs of fewer than MIN_RUN consecutive changes are dropped. Returns {state: dict(a, b, se_b, r2, n, fit_window,
    n_gaps)}; a state with fewer than hard_min changes gets a = b = se_b = r2 = None.
    """
    sub = changes.filter(pl.col('month') < cutoff)
    out = {}
    for (state,), grp in sub.group_by('state_abbr', maintain_order=True):
        grp = grp.sort('month')
        keep = {m for run in _runs(grp['month'].to_list()) if len(run) >= MIN_RUN for m in run}
        grp = grp.filter(pl.col('month').is_in(keep))
        months = grp['month'].to_list()
        if len(months) < hard_min:
            out[state] = dict(a=None, b=None, se_b=None, r2=None, n=len(months), fit_window=_window(months), n_gaps=0)
            continue
        f = ols(grp['dp_nat'].to_list(), grp['dp'].to_list())
        f.update(fit_window=_window(months), n_gaps=_gaps(months))
        out[state] = f
    return out


def _runs(months: list[date]) -> list[list[date]]:
    """Split sorted change months into runs of consecutive calendar months."""
    runs = []
    for m in months:
        if runs and m == next_month(runs[-1][-1]):
            runs[-1].append(m)
        else:
            runs.append([m])
    return runs


def _window(months: list[date]) -> str | None:
    """Contiguous runs of price months used, e.g. '2021-10..2022-10,2023-01..2024-09' (a change reaches back one month)."""
    if not months:
        return None
    return ','.join(f'{_prev_month(r[0]):%Y-%m}..{r[-1]:%Y-%m}' for r in _runs(months))


def _prev_month(m: date) -> date:
    return date(m.year - (m.month == 1), (m.month - 2) % 12 + 1, 1)


def _gaps(months: list[date]) -> int:
    """Calendar months missing between the first and last change month."""
    span = (months[-1].year - months[0].year) * 12 + months[-1].month - months[0].month + 1
    return span - len(months)


# ---------------------------------------------------------------- measure specifications

def measure_specs(win: dict) -> list[dict]:
    """One spec per (measure, cycle): fit cutoff, months spanned by the change, cycle type and shock file."""
    specs = []
    for t in shocks.CYCLES:
        ed = date.fromisoformat(str(win[t]['election_day']))
        b = win[t]['baseline']
        ed_b = date.fromisoformat(str(win[b]['election_day']))
        # w21: window midpoints are ED - 11, so the span is the distance between the two election days
        specs.append(dict(measure='w21', year=t, cycle_type=win[t]['type'],
                          cutoff=month_start(date.fromordinal(ed_b.toordinal() - shocks.DEFAULT_WINDOW_DAYS)),
                          span=(ed - ed_b).days / DAYS_PER_MONTH))
        # sa: midpoint of W(ED) (ED - 11) against the midpoint of January 1-31 (Jan 16)
        mid = date.fromordinal(ed.toordinal() - 11)
        specs.append(dict(measure='sa', year=t, cycle_type=win[t]['type'], cutoff=date(t, 1, 1),
                          span=(mid - date(t, 1, 16)).days / DAYS_PER_MONTH))
    for t in KY_CYCLES:
        # ky12: Nov (t-1)..Oct (t) block minus Nov (t-2)..Oct (t-1) block: 12 months apart
        specs.append(dict(measure='ky12', year=t, cycle_type=win[t]['type'], cutoff=date(t - 2, 11, 1), span=12.0))
    return specs


def load_shock_file(measure: str, directory=None) -> pl.DataFrame | None:
    path = (directory or paths.processed_dir()) / SHOCK_FILES[measure]
    return pl.read_parquet(path) if path.exists() else None


def national_shock(sh: pl.DataFrame | None, year: int, n_units: int = 51) -> tuple[float | None, int, bool, list[str]]:
    """Unweighted mean of the cycle's state shocks, the number of states, provisional, and notes."""
    if sh is None:
        return None, 0, False, ['shock_file_missing']
    sub = sh.filter((pl.col('year') == year) & pl.col('delta_p').is_not_null())
    if sub.height == 0:
        return None, 0, False, ['no_shocks_for_cycle']
    prov = bool(sh.filter(pl.col('year') == year)['provisional'].any())
    notes = []
    if sub.height < n_units:
        notes.append('g_nat_partial_units')
    if sub.height < math.ceil(MIN_NAT_SHARE * n_units):
        return None, sub.height, prov, notes + ['g_nat_too_few_units']
    return statistics.fmean(sub['delta_p'].to_list()), sub.height, prov, notes


# ---------------------------------------------------------------- assembly

def build(monthly: pl.DataFrame, units: pl.DataFrame, win: dict, shock_frames: dict, price_inputs: str = '') -> pl.DataFrame:
    """The full grid: 51 units x (w21 x 5 cycles, sa x 5, ky12 x 3)."""
    nat = national_monthly(monthly, units.height)
    changes = monthly_changes(monthly, nat)
    rows = []
    for spec in measure_specs(win):
        fits = fit_states(changes, spec['cutoff'])
        g_nat, n_units, prov, notes = national_shock(shock_frames.get(spec['measure']), spec['year'], units.height)
        for s in units['state_abbr'].to_list():
            f = fits.get(s) or dict(a=None, b=None, se_b=None, r2=None, n=0, fit_window=None, n_gaps=0)
            flags = list(notes)
            if f['b'] is None:
                flags.append('no_fit')
            elif f['n'] < MIN_FIT_MONTHS:
                flags.append('short_fit')
            if f['n_gaps']:
                flags.append('fit_gap')
            if spec['measure'] == 'w21' and spec['cycle_type'] == 'presidential':
                flags.append('presidential_span_not_used')
            if prov:
                flags.append('provisional_window_open')
            g_pt = nodrift = None
            if f['b'] is not None and g_nat is not None:
                g_pt = f['a'] * spec['span'] + f['b'] * g_nat
                nodrift = f['b'] * g_nat
            rows.append(dict(
                state_abbr=s, measure=spec['measure'], year=spec['year'], cycle_type=spec['cycle_type'],
                a=f['a'], b=f['b'], se_b=f['se_b'], r2=f['r2'], n_months=f['n'], fit_window=f['fit_window'],
                fit_cutoff=spec['cutoff'], span_months=spec['span'], g_nat=g_nat, g_pt=g_pt, g_pt_nodrift=nodrift,
                provisional=prov, flags=';'.join(sorted(set(flags))), price_inputs=price_inputs, code_version=CODE_VERSION))
    return pl.DataFrame(rows, schema=OUT_SCHEMA).sort('measure', 'year', 'state_abbr')


# ---------------------------------------------------------------- diagnostics (treatment side only)

def pearson(x: list[float], y: list[float]) -> float | None:
    n = len(x)
    if n < 3:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(sxx * syy)


def _ranks(v: list[float]) -> list[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x: list[float], y: list[float]) -> float | None:
    return pearson(_ranks(x), _ranks(y)) if len(x) >= 3 else None


def demean_within(values: dict, groups: dict) -> dict:
    """Subtract the unweighted group mean over the states present in values."""
    by = {}
    for s, v in values.items():
        by.setdefault(groups[s], []).append(v)
    mean = {g: sum(v) / len(v) for g, v in by.items()}
    return {s: v - mean[groups[s]] for s, v in values.items()}


def predictive_correlations(pt: pl.DataFrame, shock_frames: dict, units: pl.DataFrame,
                            exclude: frozenset = frozenset()) -> pl.DataFrame:
    """Per (measure, year): correlation of the predicted state component with the actual within-PADD deviation.

    raw     : corr(g_pt - g_nat, deviation)                      (as in the brief)
    within  : corr(g_pt - PADD mean of g_pt, deviation)          (the part the E_c x group dummies leave)
    nodrift : same as within, with g_pt_nodrift = b * g_nat
    seds    : same as raw / within for the SEDS g_pt in shocks.parquet (w21, midterm cycles)
    exclude : states dropped from every correlation (sensitivity to outliers and tax holidays)
    """
    groups = dict(zip(units['state_abbr'], units['padd_group']))
    rows = []
    for (measure, year), grp in pt.group_by('measure', 'year', maintain_order=True):
        sh = shock_frames.get(measure)
        if sh is None:
            continue
        s = sh.filter(pl.col('year') == year).select('state_abbr', 'deviation', 'g_pt', 'g_nat')
        dev = {r['state_abbr']: r['deviation'] for r in s.iter_rows(named=True)
               if r['deviation'] is not None and r['state_abbr'] not in exclude}
        hf = {r['state_abbr']: r for r in grp.iter_rows(named=True) if r['g_pt'] is not None}
        seds = {r['state_abbr']: r for r in s.iter_rows(named=True) if r['g_pt'] is not None}
        row = dict(measure=measure, year=year)
        row.update(_corrs('hf', hf, dev, groups, 'g_pt'))
        row.update(_corrs('hf_nodrift', hf, dev, groups, 'g_pt_nodrift'))
        row.update(_corrs('seds', seds, dev, groups, 'g_pt'))
        rows.append(row)
    return pl.DataFrame(rows).sort('measure', 'year') if rows else pl.DataFrame()


def _corrs(tag: str, src: dict, dev: dict, groups: dict, col: str) -> dict:
    keys = sorted(set(src) & set(dev))
    out = {f'{tag}_n': len(keys), f'{tag}_raw': None, f'{tag}_within': None, f'{tag}_within_spearman': None}
    if len(keys) < 10:
        return out
    g_nat = next(iter(src.values()))['g_nat']
    raw = pearson([src[k][col] - g_nat for k in keys], [dev[k] for k in keys])
    pred = demean_within({k: src[k][col] for k in keys}, groups)
    out[f'{tag}_raw'] = raw
    out[f'{tag}_within'] = pearson([pred[k] for k in keys], [dev[k] for k in keys])
    out[f'{tag}_within_spearman'] = spearman([pred[k] for k in keys], [dev[k] for k in keys])
    return out


def backtest(monthly: pl.DataFrame, units: pl.DataFrame, cutoffs: list[date], horizons=(12, 24)) -> pl.DataFrame:
    """Pseudo-cycles inside the monthly data: fit before a cutoff, predict h-month changes of the monthly price.

    For each cutoff and horizon, every start month m0 >= cutoff with m0 + h observed (all 51 units) is one evaluation
    change. Prediction a*h + b*dnat; actual within-PADD deviation of the change. Evaluation changes overlap, so n_eval
    overstates the information; the reported range shows the spread.
    """
    groups = dict(zip(units['state_abbr'], units['padd_group']))
    nat = national_monthly(monthly, units.height)
    changes = monthly_changes(monthly, nat)
    price = {(r['state_abbr'], r['month']): r['price'] for r in monthly.iter_rows(named=True)}
    avail = set(nat['month'].to_list())
    states = units['state_abbr'].to_list()
    rows = []
    for cutoff in cutoffs:
        fits = fit_states(changes, cutoff)
        if any(fits[s]['b'] is None for s in states):
            continue
        n_fit = min(fits[s]['n'] for s in states)
        for h in horizons:
            res = {'hf': [], 'hf_nodrift': [], 'b_only_unit_nat': []}
            m0 = cutoff
            last = max(avail)
            while m0 <= last:
                m1 = date(m0.year + (m0.month - 1 + h) // 12, (m0.month - 1 + h) % 12 + 1, 1)
                if m0 in avail and m1 in avail:
                    chg = {s: price[(s, m1)] - price[(s, m0)] for s in states}
                    g_nat = sum(chg.values()) / len(chg)
                    act = demean_within(chg, groups)
                    for tag, fn in (('hf', lambda s: fits[s]['a'] * h + fits[s]['b'] * g_nat),
                                    ('hf_nodrift', lambda s: fits[s]['b'] * g_nat)):
                        pred = demean_within({s: fn(s) for s in states}, groups)
                        res[tag].append(pearson([pred[s] for s in states], [act[s] for s in states]))
                m0 = next_month(m0)
            for tag in ('hf', 'hf_nodrift'):
                v = [x for x in res[tag] if x is not None]
                if v:
                    rows.append(dict(cutoff=cutoff, horizon=h, variant=tag, n_fit=n_fit, n_eval=len(v),
                                     mean_corr=sum(v) / len(v), min_corr=min(v), max_corr=max(v)))
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def split_half(changes: pl.DataFrame, split: date, end: date) -> dict:
    """Stability of b: fit on changes before `split` and on changes from `split` to before `end`; correlate the slopes."""
    a = fit_states(changes.filter(pl.col('month') < split), split)
    b = fit_states(changes.filter(pl.col('month') >= split), end)
    keys = sorted(k for k in a if a[k]['b'] is not None and b.get(k, {}).get('b') is not None)
    x, y = [a[k]['b'] for k in keys], [b[k]['b'] for k in keys]
    return dict(n_states=len(keys), n_first=min(a[k]['n'] for k in keys), n_second=min(b[k]['n'] for k in keys),
                pearson=pearson(x, y), spearman=spearman(x, y))


def slope_reliability(pt: pl.DataFrame, measure: str, year: int) -> dict:
    """Share of the cross-state variance of b that is signal: 1 - mean(se_b^2) / var(b)."""
    s = pt.filter((pl.col('measure') == measure) & (pl.col('year') == year) & pl.col('b').is_not_null())
    if s.height < 3:
        return {}
    b, se = s['b'].to_list(), s['se_b'].to_list()
    var = statistics.variance(b)
    return dict(n=s.height, mean_b=statistics.fmean(b), sd_b=math.sqrt(var), median_se=statistics.median(se),
                reliability=1 - statistics.fmean([e * e for e in se]) / var)


def group_difference(pt: pl.DataFrame, measure: str, year: int, treated: set, others: set | None = None,
                     seed: int = 20261019, draws: int = 20000) -> dict:
    """Mean b in the treated (percentage-tax) states against `others` (default: all other states), permutation p (two-sided)."""
    import random
    s = pt.filter((pl.col('measure') == measure) & (pl.col('year') == year) & pl.col('b').is_not_null())
    b = dict(zip(s['state_abbr'], s['b']))
    if others is not None:
        b = {k: v for k, v in b.items() if k in treated or k in others}
    tr = [b[k] for k in b if k in treated]
    ot = [b[k] for k in b if k not in treated]
    if len(tr) < 2 or len(ot) < 2:
        return {}
    diff = statistics.fmean(tr) - statistics.fmean(ot)
    pool = list(b.values())
    rng = random.Random(seed)
    hits = 0
    for _ in range(draws):
        rng.shuffle(pool)
        if abs(statistics.fmean(pool[:len(tr)]) - statistics.fmean(pool[len(tr):])) >= abs(diff) - 1e-12:
            hits += 1
    return dict(n_treated=len(tr), n_other=len(ot), mean_treated=statistics.fmean(tr), mean_other=statistics.fmean(ot),
                diff=diff, p_perm=(hits + 1) / (draws + 1))


# ---------------------------------------------------------------- tax cross-check (public sources, see the report)

# pct_retail   : a sales tax or ad valorem charge applies to the retail price of gasoline in 2024-2026
# pct_wholesale: percentage tax on first-sale gross earnings (CT)
# variable_unspecified: excise rate varies by formula; the base (wholesale price, CPI, ...) is not established by the
#                       sources used, so these states are left out of the group comparison
# none_noted   : no percentage tax on gasoline reported in FHWA MF-121T (2023) or API via Wikipedia (July 2025)
# Sources: FHWA = Highway Statistics 2023 Table MF-121T (rates at 31 Dec 2023); API = American Petroleum Institute
# state tax table (July 2025) as reproduced in Wikipedia 'Fuel taxes in the United States'; IN = Indiana DOR
# Gasoline Use Tax page (retrieved 2026-10-19).
TAX_CLASS = {
    'CA': ('pct_retail', 'FHWA 6.0% on sales price incl. fuel taxes; API 2.25% on gasoline (rate differs, existence agrees)'),
    'GA': ('pct_retail', 'FHWA: 4% sales tax, 3% second motor fuel tax, 1-cent sales tax on sales price; API: local sales tax'),
    'HI': ('pct_retail', 'FHWA 4% on sales price excl. fuel taxes; API: additional sales tax'),
    'IL': ('pct_retail', 'API 6.25% state sales tax plus local; not listed in FHWA MF-121T (single source)'),
    'IN': ('pct_retail', 'Indiana DOR: gasoline use tax = 7% of average retail price (ex tax), rate reset monthly; suspended 2026-04-08..11-04'),
    'MI': ('pct_retail', 'FHWA and API: 6% sales tax on gasoline through 2025; a 2026 change is not verified here'),
    'NY': ('pct_retail', 'FHWA 4% on the first $2 per gallon, plus local percentage rates; API same'),
    'CT': ('pct_wholesale', 'FHWA: 5% of gross earnings from first sale (API: 8.1%)'),
    'DE': ('variable_unspecified', 'FHWA: 0.9% HSCA tax on gross receipts of petroleum sales (small); excise fixed'),
    'KY': ('variable_unspecified', 'FHWA: rate variable, adjusted quarterly; base not stated'),
    'NE': ('variable_unspecified', 'FHWA: rate variable, adjusted quarterly; base not stated'),
    'PA': ('variable_unspecified', 'FHWA: Oil Franchise Tax variable, adjusted annually; base not stated'),
    'WV': ('variable_unspecified', 'FHWA: rate variable, adjusted annually; base not stated'),
    'NC': ('variable_unspecified', 'FHWA: rate variable, adjusted semiannually; base not stated'),
    'FL': ('variable_unspecified', 'FHWA: rate variable, adjusted annually; base not stated'),
    'ME': ('variable_unspecified', 'FHWA: CPI-indexed, adjusted annually'),
    'WI': ('variable_unspecified', 'FHWA: rate variable, adjusted annually; base not stated'),
}


def tax_group(state: str) -> str:
    return TAX_CLASS.get(state, ('none_noted', ''))[0]


# ---------------------------------------------------------------- CLI

def _inputs() -> tuple[pl.DataFrame, pl.DataFrame, dict, str, dict]:
    win = shocks.load_config()
    units = shocks.load_units()
    dv = shocks.daily_values(shocks.load_aaa_prices())
    monthly = monthly_prices(dv)
    frames = {m: load_shock_file(m) for m in MEASURES}
    files = [*shocks.PRICE_FILES, *(SHOCK_FILES[m] for m in MEASURES if frames[m] is not None)]
    inputs = ';'.join(
        f'{f}:{shocks.sha256_file((paths.interim_dir() if f.startswith("prices") else paths.processed_dir()) / f)[:16]}'
        for f in files)
    return monthly, units, frames, inputs, win


def main() -> None:
    monthly, units, frames, inputs, win = _inputs()
    out = build(monthly, units, win, frames, inputs)
    path = paths.processed_dir() / 'g_pt_hf.parquet'
    out.write_parquet(path, compression='zstd', statistics=False)
    cfg_files = [paths.config_dir() / 'windows.yaml', paths.config_dir() / 'states.csv',
                 paths.interim_dir() / 'padd_membership.csv']
    missing = [m for m in MEASURES if frames[m] is None]
    runlog.append_run_log(
        run_id=uuid.uuid4().hex[:12], utc_timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        git_sha=runlog.git_sha(), git_dirty=runlog.git_dirty(), config_hash=runlog.config_hash(cfg_files),
        spec_id='pass_through_hf', data_snapshot_ids=inputs, output_path='data/processed/g_pt_hf.parquet',
        notes=f'{CODE_VERSION}; rows={out.height}; min_days={MIN_DAYS}; min_fit={MIN_FIT_MONTHS}; hard_min={HARD_MIN_FIT}; '
              f'shock files missing={missing or "none"}; treatment data only')
    print(f'wrote {path} ({out.height} rows)')


def report() -> None:
    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(220)
    monthly, units, frames, inputs, win = _inputs()
    pt = build(monthly, units, win, frames, inputs)
    print(pt.group_by('measure', 'year').agg(
        pl.col('b').is_not_null().sum().alias('n_fit_states'), pl.col('n_months').min().alias('n_min'),
        pl.col('n_months').max().alias('n_max'), pl.col('b').mean().alias('mean_b'), pl.col('b').std().alias('sd_b'),
        pl.col('b').min().alias('min_b'), pl.col('b').max().alias('max_b'), pl.col('se_b').median().alias('med_se'),
        pl.col('r2').median().alias('med_r2'), pl.col('a').mean().alias('mean_a'), pl.col('span_months').first(),
        pl.col('g_nat').first(), pl.col('g_pt').count().alias('n_gpt'), pl.col('fit_window').first()).sort('measure', 'year'))
    for label, ex in (('all', frozenset()), ('ex AK HI', frozenset({'AK', 'HI'})), ('ex AK HI IN', frozenset({'AK', 'HI', 'IN'}))):
        print(label)
        print(predictive_correlations(pt, frames, units, ex).select(
            'measure', 'year', 'hf_n', 'hf_raw', 'hf_within', 'hf_within_spearman', 'hf_nodrift_within', 'seds_n', 'seds_raw', 'seds_within',
            'seds_within_spearman').filter(pl.col('hf_n') + pl.col('seds_n') > 0))
    cut = [date(2024, 1, 1), date(2024, 7, 1), date(2025, 1, 1)]
    print(backtest(monthly, units, cut))
    for m, y in (('w21', 2026), ('sa', 2026), ('ky12', 2026), ('sa', 2024), ('ky12', 2024)):
        print(m, y, slope_reliability(pt, m, y))
        pr = {k for k, v in TAX_CLASS.items() if v[0] == 'pct_retail'}
        none = {s for s in units['state_abbr'] if tax_group(s) == 'none_noted'}
        print('  pct_retail vs none_noted', group_difference(pt, m, y, pr, none))
        print('  pct_retail ex HI vs none_noted', group_difference(pt, m, y, pr - {'HI'}, none))
        print('  retail+CT vs none_noted', group_difference(pt, m, y, pr | {'CT'}, none))
        print('  pct_retail vs all others', group_difference(pt, m, y, pr))
    nat = national_monthly(monthly, units.height)
    ch = monthly_changes(monthly, nat)
    print('split-half 2023-07', split_half(ch, date(2023, 7, 1), date(2026, 10, 1)))
    print('split-half 2023-01', split_half(ch, date(2023, 1, 1), date(2026, 10, 1)))
    w = pt.filter((pl.col('measure') == 'sa') & (pl.col('year') == 2026)).select('state_abbr', 'b', 'se_b').sort('b')
    print(w.head(6), w.tail(8))


if __name__ == '__main__':
    {'report': report}.get(sys.argv[1] if len(sys.argv) > 1 else '', main)()
