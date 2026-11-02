"""Sub-state (county and metro) gasoline-price shocks with the clusters and strata of the O6 assignment model.

Design authority: memo §5.1 (D22 treatment level; O6 sub-state assignment model), `.scratch/o6_substate_assignment_draft.md`.
AAA prices stay in the interim parquets; the output holds shocks only. Reports hold aggregates only (memo D19).

    g_c = mean_post p_c - mean_pre p_c        g_s: state shock of the same measure        u_c = g_c - g_s

Measures (a change in the mean daily regular price between two blocks):
    w21     W(ED_t) minus W(ED_t-2), W(ED) = [ED-21, ED-1]            (2026)
    sa      W(ED_t) minus January 1-31 of t                           (2024, 2026; memo R9 S-a)
    ky12    Nov 1 (t-1) to Oct 31 (t) minus Nov 1 (t-2) to Oct 31 (t-1)   (2024, 2026; Kim & Yang 2021)
    w21_2y  W(ED_t) minus W(ED_t-2) for t = 2024, i.e. the 2022->2024 change (persistence diagnostic only)
    w21_hist  W(ED_t) minus W(ED_2020) for the validation cycles t = 2022, 2024 (the shocks.parquet change); metro only

Coverage: 21-day blocks need MIN_DAYS observed days, January MIN_DAYS_JAN, 12-month blocks KY_SHARE of their days
(KY_SHARE_METRO at the metro level); w21_hist needs HIST_MIN_DAYS in each window (PI 2026-10-22).
A unit that fails gets a null shock and a flag; nothing is imputed.

County prices before 2024-10-15 are not in `prices_county_daily.parquet`; they are read here from the stored
nathanqthai bundle with the parser of `county_prices.py` (same 11:00-13:59 UTC rule) and are not written anywhere.
Metro prices before the daily metro panel starts come from the archived AAA state pages (`prices_metro_wayback.parquet`).

    python -m fb26.etl.substate_shocks [all|build|report]
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import sys
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from fb26.etl import county_prices as cp
from fb26.etl import metro_prices as mp
from fb26.etl.shocks import LAG_RANK
from fb26.scrape.aaagas import _git, read_blobs, open_bundle, latest_record, load_states
from fb26.util import paths, rawstore, runlog

VERSION = 'substate-shocks-0.2.0'
MIN_DAYS = 7           # 7 of 21 days in each price window (PI 2026-10-30; was 10, memo D22)
MIN_DAYS_JAN = 15      # 15 of 31 days
KY_SHARE = 0.8         # 80% of the days of a 12-month block
KY_SHARE_METRO = 0.7   # metro level: the metro archive has gaps (memo D23, PI 2026-10-19)
HIST_MIN_DAYS = 1      # w21_hist: any archived day (PI 2026-10-22; the 2020 window has a median of 3 per state)
HIST_CYCLES = {2022: 2020, 2024: 2020}   # validation cycle: baseline (windows.yaml)
WAYBACK_ALIASES = {('25', 'Seakonk'): 'Seekonk', ('54', 'Beckly'): 'Beckley'}   # archived spellings -> crosswalk labels
EARLY_START = date(2022, 9, 30)   # first day of the nathanqthai county series
ALASKA = '02'
SEED = 20261019
N_PERM = 999
DECIMALS = 10        # prices have 3 decimals; rounding removes last-bit differences from parallel summation
LEVELS = ('county', 'metro')
OUT_COLUMNS = ['measure', 'cycle', 'level', 'geo_id', 'state_fips', 'g_c', 'g_s', 'u_c', 'n_days_post', 'n_days_pre',
               'coverage_ok', 'cbsa_code', 'cbsa_type', 'cluster_id', 'cluster_mean_u', 'within_cluster_resid',
               'provisional', 'flags', 'padd_group', 'stratum_id', 'cluster_n_valid']
STATE_FILES = {'w21': 'shocks.parquet', 'w21_2y': 'shocks.parquet', 'sa': 'shocks_r9a.parquet', 'ky12': 'shocks_ky12.parquet',
               'w21_hist': 'shocks.parquet'}


@dataclass(frozen=True)
class Rules:
    min_days_post: int = MIN_DAYS    # 21-day post window of the 2026 cycle (lowered only for the interim report)
    min_days_pre: int = MIN_DAYS
    min_days_jan: int = MIN_DAYS_JAN
    ky_share: float = KY_SHARE
    ky_share_metro: float = KY_SHARE_METRO


INTERIM = Rules(min_days_post=3)
RULESETS = {'primary': Rules(), 'interim': INTERIM}


@dataclass(frozen=True)
class Spec:
    measure: str
    cycle: int
    post: tuple[date, date]
    pre: tuple[date, date]
    min_post: int
    min_pre: int
    levels: tuple[str, ...]


# ---------------------------------------------------------------- specs

def load_windows(path: Path | None = None) -> dict[int, tuple[date, date]]:
    config = yaml.safe_load((path or paths.config_dir() / 'windows.yaml').read_text())
    return {int(y): (date.fromisoformat(str(v['start'])), date.fromisoformat(str(v['end']))) for y, v in config['elections'].items()}


def ky_block(year: int) -> tuple[date, date]:
    """Nov 1 (year-1) to Oct 31 (year)."""
    return date(year - 1, 11, 1), date(year, 10, 31)


def n_days(window: tuple[date, date]) -> int:
    return (window[1] - window[0]).days + 1


def make_specs(windows: dict[int, tuple[date, date]], rules: Rules = Rules()) -> list[Spec]:
    specs = [Spec('w21', 2026, windows[2026], windows[2024], rules.min_days_post, rules.min_days_pre, LEVELS),
             Spec('w21_2y', 2024, windows[2024], windows[2022], MIN_DAYS, rules.min_days_pre, ('county',))]
    for cycle, levels in ((2026, LEVELS), (2024, ('county',))):
        jan = (date(cycle, 1, 1), date(cycle, 1, 31))
        specs.append(Spec('sa', cycle, windows[cycle], jan, rules.min_days_post if cycle == 2026 else MIN_DAYS,
                          rules.min_days_jan, levels))
        post, pre = ky_block(cycle), ky_block(cycle - 1)
        for level in levels:   # one spec per level: metro takes 70% rounded down (255 of 365; PI 2026-10-19)
            share, rnd = (rules.ky_share_metro, math.floor) if level == 'metro' else (rules.ky_share, math.ceil)
            specs.append(Spec('ky12', cycle, post, pre, rnd(share * n_days(post)),
                              rnd(share * n_days(pre)), (level,)))
    for cycle, baseline in HIST_CYCLES.items():
        specs.append(Spec('w21_hist', cycle, windows[cycle], windows[baseline], HIST_MIN_DAYS, HIST_MIN_DAYS, ('metro',)))
    return sorted(specs, key=lambda s: (s.measure, s.cycle))


# ---------------------------------------------------------------- daily prices

def to_daily(df: pl.DataFrame) -> pl.DataFrame:
    """One equal-weight value per county-day across sources (as in D21 and `county_prices.county_shocks`)."""
    valid = df.filter(pl.col('county_fips').is_not_null() & pl.col('price_usd').is_not_null() & pl.col('as_of_date').is_not_null())
    suspect = pl.col('flags').str.contains('equals_state_average|widely_repeated_price')
    return (valid.group_by('county_fips', 'state_fips', 'as_of_date')
            .agg(pl.col('price_usd').mean(), pl.col('source').unique().sort().str.join('+').alias('sources'),
                 suspect.any().alias('suspect'))
            .sort('county_fips', 'as_of_date'))


def early_county_rows(end_exclusive: date) -> pl.DataFrame:
    """nathanqthai county rows from EARLY_START up to the day before the parquet starts (same filters as `county_prices.build`)."""
    _, lookup = cp.county_universe()
    _, by_abbr = load_states()
    record = latest_record('nathanqthai_gas_prices_repo')
    scratch = paths.repo_root() / '.scratch/tmp'
    scratch.mkdir(parents=True, exist_ok=True)
    frames, state_rows = [], []
    with tempfile.TemporaryDirectory(prefix='substate_shocks_', dir=scratch) as tmp:
        repo = open_bundle(record, Path(tmp))
        listing = _git(['ls-tree', '-r', '--name-only', 'refs/heads/main', '--', 'prices/states', 'prices/national'], git_dir=repo).decode().splitlines()
        selected = []
        for name in listing:
            try:
                captured = datetime.fromisoformat(Path(name).stem)
            except ValueError:
                continue
            if EARLY_START <= captured.date() < end_exclusive and 11 <= captured.hour < 14:
                selected.append((name, captured))
        for i in range(0, len(selected), 40):
            batch = selected[i:i + 40]
            blobs = read_blobs(repo, ['refs/heads/main:' + n for n, _ in batch])
            rows = []
            for name, captured in batch:
                content = blobs['refs/heads/main:' + name]
                if name.startswith('prices/states/'):
                    rows.extend(cp.parse_nathan(content, captured, f'{record.sha256}:{name}', by_abbr, lookup))
                else:
                    for rec in csv.reader(io.StringIO(content.decode())):
                        value, _ = cp.price_value(rec[2])
                        state_rows.append(dict(source=cp.NATHAN, as_of_date=cp.scrape_date(captured), state_fips=by_abbr[rec[0].upper()], _state_price=value))
            if rows:
                frames.append(pl.DataFrame(rows, schema=cp.SCHEMA))
    states = pl.DataFrame(state_rows).group_by('source', 'as_of_date', 'state_fips').agg(pl.col('_state_price').mean())
    return cp.annotate_placeholders(cp.dedupe(pl.concat(frames)), states)


def county_daily() -> pl.DataFrame:
    current = pl.read_parquet(paths.interim_dir() / 'prices_county_daily.parquet')
    first = current['as_of_date'].min()
    return to_daily(pl.concat([early_county_rows(first), current]))


def wayback_metro_daily(wb: pl.DataFrame, before: date) -> pl.DataFrame:
    """One price per state, metro label and day from the archived AAA state pages, for days before `before`.

    A price printed for the day beats one read from a lag column, and the shortest lag wins (the state rule in
    `shocks.py`); the page date is taken as printed. Two archived spellings are mapped to the crosswalk labels."""
    rank = pl.when(pl.col('derived_from_lag_field')).then(pl.col('lag_field').replace_strict(LAG_RANK, default=99)).otherwise(0)
    aliases = pl.DataFrame([(s, raw, label) for (s, raw), label in WAYBACK_ALIASES.items()],
                           schema=['state_fips', 'metro_raw', '_label'], orient='row')
    valid = (wb.filter((pl.col('product') == 'regular') & pl.col('price_usd').is_not_null() & (pl.col('as_of_date') < before))
             .join(aliases, on=['state_fips', 'metro_raw'], how='left')
             .with_columns(pl.coalesce('_label', 'metro_raw').alias('metro_raw'), rank.alias('_rank')))
    key = ['state_fips', 'metro_raw', 'as_of_date']
    return (valid.filter(pl.col('_rank') == pl.col('_rank').min().over(key))
            .group_by(key).agg(pl.col('price_usd').mean())
            .with_columns(pl.lit(None, dtype=pl.String).alias('cbsa_code')).sort(key))


def metro_daily(universe: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """County-day prices inherited from the matched metro label (narrowest label wins; see `metro_prices.assign_counties`),
    and the CBSA code of the label that supplies each county. Days before the daily panel starts come from the archive."""
    df = pl.read_parquet(paths.interim_dir() / 'prices_metro_daily.parquet')
    wb = wayback_metro_daily(pl.read_parquet(paths.interim_dir() / 'prices_metro_wayback.parquet'), df['as_of_date'].min())
    cw = pl.read_csv(paths.repo_root() / 'reports/validation/metro_cbsa_crosswalk.csv',
                     schema_overrides={'state_fips': pl.String, 'cbsa_code': pl.String, 'county_fips': pl.String,
                                       'candidate_cbsa_codes': pl.String, 'evidence': pl.String, 'scope_size': pl.Float64})
    counties = universe.select('state_fips', 'county_fips')
    assigned = pl.concat([mp.assign_counties(df, cw, counties).with_columns(pl.lit('metro').alias('sources')),
                          mp.assign_counties(wb, cw, counties).with_columns(pl.lit('metro_wayback').alias('sources'))])
    codes = assigned.group_by('county_fips').agg(pl.col('cbsa_code').n_unique().alias('_n'), pl.col('cbsa_code').min())
    if codes['_n'].max() > 1:
        raise ValueError('A county takes prices from labels of different CBSAs')
    return (assigned.select('county_fips', 'state_fips', 'as_of_date', 'price_usd', 'sources', pl.lit(False).alias('suspect'))
            .sort('county_fips', 'as_of_date'),
            codes.select('county_fips', 'cbsa_code').sort('county_fips'))


# ---------------------------------------------------------------- geography: CBSA type, commuting zones, adjacency

def sheet_rows(content: bytes, header_key: str) -> list[dict[str, str]]:
    """Rows of the first worksheet of an xlsx as dicts keyed by the header row (the row that contains `header_key`)."""
    ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        strings = []
        if 'xl/sharedStrings.xml' in z.namelist():
            strings = [''.join(n.itertext()) for n in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si', ns)]
        sheet = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
    header, out = None, []
    for row in sheet.findall('s:sheetData/s:row', ns):
        cells = {}
        for c in row:
            col = re.sub(r'\d', '', c.attrib['r'])
            val = c.findtext('s:v', default='', namespaces=ns)
            if c.attrib.get('t') == 's':
                val = strings[int(val)]
            elif c.attrib.get('t') == 'inlineStr':
                val = ''.join(c.find('s:is', ns).itertext())
            cells[col] = val.strip()
        if header is None:
            if header_key in cells.values():
                header = cells
            continue
        out.append({header[k]: v for k, v in cells.items() if k in header})
    return out


def read_cbsa_types(content: bytes) -> pl.DataFrame:
    """County -> CBSA code and type ('metropolitan' | 'micropolitan') from OMB List 1 (July 2023)."""
    rows = []
    for r in sheet_rows(content, 'CBSA Code'):
        kind = r.get('Metropolitan/Micropolitan Statistical Area', '').lower()
        code, state, county = r.get('CBSA Code', ''), r.get('FIPS State Code', ''), r.get('FIPS County Code', '')
        if not (re.fullmatch(r'\d{5}', code) and state.isdigit() and county.isdigit() and kind):
            continue
        rows.append(dict(county_fips=state.zfill(2) + county.zfill(3), cbsa_code=code,
                         cbsa_type='metropolitan' if kind.startswith('metro') else 'micropolitan'))
    df = pl.DataFrame(rows).unique().sort('county_fips')
    if df['county_fips'].n_unique() != df.height:
        raise ValueError('County belongs to several CBSAs')
    return df


def read_cz(content: bytes) -> pl.DataFrame:
    """County -> 2020 commuting zone (USDA ERS)."""
    df = pl.read_csv(io.BytesIO(content), schema_overrides={'FIPStxt': pl.String, 'CZ2020': pl.String})
    return df.select(pl.col('FIPStxt').alias('county_fips'), pl.col('CZ2020').alias('cz')).unique().sort('county_fips')


def read_adjacency(content: bytes) -> pl.DataFrame:
    """Undirected county pairs (a < b) from the Census adjacency file; self pairs dropped; blank-leading rows inherit the county."""
    pairs, current = set(), None
    for rec in csv.reader(io.StringIO(content.decode('utf-8-sig')), delimiter='|'):
        if len(rec) != 4 or rec[1] == 'County GEOID' and rec[0] == 'County Name':
            continue
        if rec[1].strip():
            current = rec[1].strip()
        other = rec[3].strip()
        if current and other and other != current:
            pairs.add((min(current, other), max(current, other)))
    return pl.DataFrame(sorted(pairs), schema={'a': pl.String, 'b': pl.String}, orient='row')


def latest_raw(source: str, suffix: str) -> Path:
    files = sorted((paths.raw_dir() / source).glob(f'**/*{suffix}'))
    if not files:
        raise FileNotFoundError(f'No stored raw file for {source}')
    return files[-1]


def build_geography(universe: pl.DataFrame, cbsa_types: pl.DataFrame, cz: pl.DataFrame, states: pl.DataFrame) -> pl.DataFrame:
    """One row per county: CBSA, type, cluster (CBSA, else commuting zone, within state) and stratum (state x type)."""
    g = (universe.select('county_fips', 'state_fips')
         .join(cbsa_types, on='county_fips', how='left').join(cz, on='county_fips', how='left')
         .join(states.select('state_fips', 'padd_group'), on='state_fips', how='left')
         .with_columns(pl.col('cbsa_type').fill_null('none')))
    cluster = (pl.when(pl.col('cbsa_code').is_not_null()).then(pl.col('state_fips') + pl.lit('-CBSA') + pl.col('cbsa_code'))
               .when(pl.col('cz').is_not_null()).then(pl.col('state_fips') + pl.lit('-CZ') + pl.col('cz'))
               .otherwise(pl.col('state_fips') + pl.lit('-C') + pl.col('county_fips')))
    return (g.with_columns(cluster.alias('cluster_id'), (pl.col('state_fips') + pl.lit('-') + pl.col('cbsa_type')).alias('stratum_id'),
                           pl.concat_str(pl.when(pl.col('cbsa_code').is_null() & pl.col('cz').is_null()).then(pl.lit('no_cluster_source'))).alias('geo_flags'))
            .drop('cz').sort('county_fips'))


# ---------------------------------------------------------------- state shocks

def state_shock_tables(processed: Path) -> dict[tuple[str, int], pl.DataFrame]:
    """(measure, cycle) -> state_fips, g_s, provisional, from the state shock parquets. A missing file leaves its measures out."""
    out = {}
    cache = {}
    for measure, name in STATE_FILES.items():
        path = processed / name
        if not path.exists():
            continue
        df = cache.setdefault(name, pl.read_parquet(path).filter(pl.col('area_type') == 'state'))
        if measure == 'w21_2y':
            post = df.filter(pl.col('year') == 2024).select('area_id', pl.col('mean_post').alias('a'))
            pre = df.filter(pl.col('year') == 2022).select('area_id', pl.col('mean_post').alias('b'))
            t = post.join(pre, on='area_id').select(pl.col('area_id').alias('state_fips'), (pl.col('a') - pl.col('b')).alias('g_s'),
                                                    pl.lit(False).alias('provisional'))
            out[(measure, 2024)] = t
            continue
        # w21 is a two-year change only in 2026 (the 2024 shock in shocks.parquet is a four-year change); w21_hist takes that change
        cycles = {'w21': [2026], 'w21_hist': list(HIST_CYCLES)}.get(measure) or sorted(df['year'].unique().to_list())
        for cycle in cycles:
            out[(measure, cycle)] = (df.filter(pl.col('year') == cycle)
                                     .select(pl.col('area_id').alias('state_fips'), pl.col('delta_p').alias('g_s'), 'provisional'))
    return out


def padd_deviation(states: pl.DataFrame, shocks: pl.DataFrame, exclude_ak: bool = False) -> pl.DataFrame:
    """State shock minus the mean of the available states in its PADD group (Alaska optionally dropped from sample and mean)."""
    t = shocks.join(states.select('state_fips', 'padd_group'), on='state_fips').filter(pl.col('g_s').is_not_null())
    if exclude_ak:
        t = t.filter(pl.col('state_fips') != ALASKA)
    return t.with_columns((pl.col('g_s') - pl.col('g_s').mean().over('padd_group')).alias('deviation'))


# ---------------------------------------------------------------- shocks

def block_stats(daily: pl.DataFrame, window: tuple[date, date], suffix: str) -> pl.DataFrame:
    d = daily.filter(pl.col('as_of_date').is_between(*window))
    base = d.group_by('county_fips').agg(pl.col('price_usd').mean().alias(f'mean_{suffix}'), pl.len().alias(f'n_{suffix}'),
                                         pl.col('suspect').mean().alias(f'suspect_{suffix}'))
    sources = (d.select('county_fips', pl.col('sources').str.split('+')).explode('sources').group_by('county_fips')
               .agg(pl.col('sources').unique().sort().str.join('+').alias(f'src_{suffix}')))
    return base.join(sources, on='county_fips', how='left')


def spec_shocks(daily: pl.DataFrame, spec: Spec, units: pl.DataFrame, horizon: date) -> pl.DataFrame:
    """Unit-level county shock for one spec. `units` holds the county_fips rows of the level; every unit keeps a row."""
    post, pre = block_stats(daily, spec.post, 'post'), block_stats(daily, spec.pre, 'pre')
    t = (units.join(post, on='county_fips', how='left').join(pre, on='county_fips', how='left')
         .with_columns(pl.col('n_post', 'n_pre').fill_null(0)))
    ok = (pl.col('n_post') >= spec.min_post) & (pl.col('n_pre') >= spec.min_pre)
    flags = pl.concat_str(
        [pl.when(pl.col('n_post') == 0).then(pl.lit('no_prices_post')).when(pl.col('n_post') < spec.min_post).then(pl.lit('thin_post')),
         pl.when(pl.col('n_pre') == 0).then(pl.lit('no_prices_pre')).when(pl.col('n_pre') < spec.min_pre).then(pl.lit('thin_pre')),
         pl.when(ok & (pl.col('src_post') != pl.col('src_pre'))).then(pl.lit('source_switch')),
         pl.when(ok & (((pl.col('suspect_post') > .5) & (pl.col('n_post') >= 3)) | ((pl.col('suspect_pre') > .5) & (pl.col('n_pre') >= 3))))
         .then(pl.lit('placeholder_suspect')),
         pl.when(pl.lit(spec.post[1] > horizon)).then(pl.lit('provisional_window_open')),
         pl.col('geo_flags')],
        separator=';', ignore_nulls=True)
    return t.select('county_fips', pl.when(ok).then((pl.col('mean_post') - pl.col('mean_pre')).round(DECIMALS)).alias('g_c'),
                    pl.col('n_post').alias('n_days_post'), pl.col('n_pre').alias('n_days_pre'), ok.alias('coverage_ok'),
                    pl.lit(spec.post[1] > horizon).alias('window_open'), flags.alias('flags'))


def add_cluster_columns(t: pl.DataFrame) -> pl.DataFrame:
    """Cluster mean of u over covered members, each member's fixed within-cluster residual, and the member count."""
    valid = pl.col('coverage_ok') & pl.col('u_c').is_not_null()
    key = ['measure', 'cycle', 'level', 'cluster_id']
    return (t.with_columns(pl.when(valid).then(pl.col('u_c')).mean().over(key).alias('_m'),
                           valid.cast(pl.Int32).sum().over(key).alias('cluster_n_valid'))
            .with_columns(pl.when(valid).then(pl.col('_m').round(DECIMALS)).alias('cluster_mean_u'))
            .with_columns(pl.when(valid).then(pl.col('u_c') - pl.col('cluster_mean_u')).alias('within_cluster_resid')).drop('_m'))


@dataclass
class Inputs:
    windows: dict
    geography: pl.DataFrame           # build_geography()
    county_daily: pl.DataFrame
    metro_daily: pl.DataFrame
    metro_cbsa: pl.DataFrame          # county_fips, cbsa_code of the metro label that supplies the county
    state_shocks: dict                # (measure, cycle) -> state_fips, g_s, provisional


def horizon_of(daily: pl.DataFrame) -> date:
    return daily['as_of_date'].max()


def level_geography(inputs: Inputs, level: str) -> pl.DataFrame:
    """Geography of the units of a level. Metro level: counties of matched metros only; the CBSA, type and cluster are those of the
    metro label that supplies the price (differs from the OMB county assignment only where a label names a county outside its CBSA)."""
    geo = inputs.geography
    if level == 'county':
        return geo
    types = geo.filter(pl.col('cbsa_code').is_not_null()).select(pl.col('cbsa_code').alias('label_cbsa'), pl.col('cbsa_type').alias('label_type')).unique(subset='label_cbsa')
    g = geo.join(inputs.metro_cbsa.rename({'cbsa_code': 'label_cbsa'}), on='county_fips').join(types, on='label_cbsa', how='left')
    differs = pl.col('cbsa_code').is_null() | (pl.col('cbsa_code') != pl.col('label_cbsa'))
    return (g.with_columns(pl.concat_str(pl.col('geo_flags'), pl.when(differs).then(pl.lit('metro_county_outside_omb_cbsa')), separator=';', ignore_nulls=True).alias('geo_flags'),
                           pl.col('label_cbsa').alias('cbsa_code'), pl.col('label_type').fill_null('none').alias('cbsa_type'),
                           (pl.col('state_fips') + pl.lit('-CBSA') + pl.col('label_cbsa')).alias('cluster_id'))
            .with_columns((pl.col('state_fips') + pl.lit('-') + pl.col('cbsa_type')).alias('stratum_id'))
            .drop('label_cbsa', 'label_type'))


def assemble(inputs: Inputs, rules: Rules = Rules()) -> pl.DataFrame:
    daily = {'county': inputs.county_daily, 'metro': inputs.metro_daily}
    geo = {level: level_geography(inputs, level) for level in LEVELS}
    horizons = {level: horizon_of(d) for level, d in daily.items()}
    frames = []
    for spec in make_specs(inputs.windows, rules):
        state = inputs.state_shocks.get((spec.measure, spec.cycle))
        for level in spec.levels:
            s = spec_shocks(daily[level], spec, geo[level].select('county_fips', 'geo_flags'), horizons[level])
            s = s.join(geo[level].drop('geo_flags'), on='county_fips', how='left')
            if state is None:
                s = s.with_columns(pl.lit(None, dtype=pl.Float64).alias('g_s'), pl.lit(True).alias('state_provisional'))
            else:
                s = s.join(state, on='state_fips', how='left').rename({'provisional': 'state_provisional'})
            frames.append(s.with_columns(pl.lit(spec.measure).alias('measure'), pl.lit(spec.cycle, dtype=pl.Int32).alias('cycle'), pl.lit(level).alias('level')))
    t = pl.concat(frames)
    t = (t.with_columns((pl.col('g_c') - pl.col('g_s')).round(DECIMALS).alias('u_c'))
         .with_columns(pl.concat_str([pl.col('flags'),
                                      pl.when(pl.col('g_s').is_null()).then(pl.lit('state_shock_missing')),
                                      pl.when(pl.col('state_fips') == ALASKA).then(pl.lit('alaska_not_in_sample'))],
                                     separator=';', ignore_nulls=True).alias('flags'),
                       (pl.col('window_open') | pl.col('state_provisional').fill_null(True)).alias('provisional')))
    t = add_cluster_columns(t)
    return (t.rename({'county_fips': 'geo_id'}).select(OUT_COLUMNS)
            .with_columns(pl.col('n_days_post', 'n_days_pre').cast(pl.Int32))
            .with_columns(pl.col('flags').str.strip_chars(';')).sort('measure', 'cycle', 'level', 'geo_id'))


# ---------------------------------------------------------------- diagnostics helpers

def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def icc1(values: np.ndarray, groups: np.ndarray) -> tuple[float | None, int, int]:
    """One-way ANOVA ICC(1) with unequal group sizes over groups with >= 2 members: (icc, groups used, observations used)."""
    _, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    keep = counts[inv] >= 2
    x, g = values[keep], np.unique(inv[keep], return_inverse=True)[1]
    k, n = (g.max() + 1 if len(g) else 0), len(x)
    if k < 2 or n <= k:
        return None, int(k), int(n)
    size = np.bincount(g)
    means = np.bincount(g, x) / size
    msb = (size * (means - x.mean()) ** 2).sum() / (k - 1)
    msw = ((x - means[g]) ** 2).sum() / (n - k)
    n0 = (n - (size ** 2).sum() / n) / (k - 1)
    denom = msb + (n0 - 1) * msw
    return (float((msb - msw) / denom) if denom > 0 else None), int(k), int(n)


def moran_i(z: np.ndarray, i: np.ndarray, j: np.ndarray, groups: np.ndarray | None = None, seed: int = SEED,
            n_perm: int = N_PERM) -> dict:
    """Moran's I with binary weights over undirected edges (i, j) and a permutation p-value (two-sided).
    Values are permuted across all nodes, or within `groups` when given."""
    z = z - z.mean() if groups is None else z
    n = len(z)
    if n < 3 or len(i) == 0 or not np.any(z):
        return dict(n=n, edges=len(i), i=None, expected=None, p=None)
    denom = (z ** 2).sum()
    def stat(v):
        return n / len(i) * (v[i] * v[j]).sum() / denom
    observed = stat(z)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_perm)
    order = np.argsort(groups, kind='stable') if groups is not None else None
    bounds = None
    if groups is not None:
        sorted_groups = groups[order]
        bounds = np.flatnonzero(np.r_[True, sorted_groups[1:] != sorted_groups[:-1], True])
    for b in range(n_perm):
        if groups is None:
            v = rng.permutation(z)
        else:
            v = z.copy()
            for lo, hi in zip(bounds[:-1], bounds[1:]):
                idx = order[lo:hi]
                v[idx] = z[rng.permutation(idx)]
        draws[b] = stat(v)
    p = (1 + np.sum(np.abs(draws - draws.mean()) >= abs(observed - draws.mean()))) / (n_perm + 1)
    return dict(n=n, edges=len(i), i=float(observed), expected=float(draws.mean()), p=float(p))


def valid_rows(t: pl.DataFrame, measure: str, cycle: int, level: str) -> pl.DataFrame:
    return t.filter((pl.col('measure') == measure) & (pl.col('cycle') == cycle) & (pl.col('level') == level)
                    & pl.col('coverage_ok') & pl.col('u_c').is_not_null() & (pl.col('state_fips') != ALASKA))


def demean(df: pl.DataFrame, col: str, by: str | list[str], out: str) -> pl.DataFrame:
    return df.with_columns((pl.col(col) - pl.col(col).mean().over(by)).alias(out))


def edge_index(valid: pl.DataFrame, adjacency: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    position = {g: k for k, g in enumerate(valid['geo_id'].to_list())}
    e = adjacency.filter(pl.col('a').is_in(list(position)) & pl.col('b').is_in(list(position)))
    return (np.array([position[a] for a in e['a']], dtype=int), np.array([position[b] for b in e['b']], dtype=int))


def spatial_stats(valid: pl.DataFrame, adjacency: pl.DataFrame) -> dict:
    out = {}
    v = demean(demean(valid, 'u_c', 'state_fips', 'u_dev_state'), 'u_c', 'stratum_id', 'u_dev_stratum')
    u = v['u_c'].to_numpy()
    cluster = v['cluster_id'].to_numpy()
    for label, col in (('raw', 'u_c'), ('stratum', 'u_dev_stratum')):
        out[f'icc_{label}'], k, n = icc1(v[col].to_numpy(), cluster)
    out['icc_clusters'], out['icc_obs'] = k, n
    out['clusters'] = v['cluster_id'].n_unique()
    out['share_counties_singleton_cluster'] = 1 - n / v.height if v.height else None
    i, j = edge_index(v, adjacency)
    state = v['state_fips'].to_numpy()
    m = moran_i(u, i, j)
    out.update(moran_raw=m['i'], moran_raw_expected=m['expected'], moran_raw_p=m['p'], adj_edges=m['edges'])
    same = state[i] == state[j]
    iw, jw = i[same], j[same]
    m = moran_i(v['u_dev_state'].to_numpy(), iw, jw, groups=state)
    out.update(moran_within_state=m['i'], moran_within_state_expected=m['expected'], moran_within_state_p=m['p'], adj_edges_within_state=m['edges'])
    # correlation of cluster means between adjacent clusters
    cl = v.group_by('cluster_id').agg(pl.col('state_fips').first(), pl.col('u_c').mean().alias('m'), pl.col('geo_id'))
    cl = demean(cl, 'm', 'state_fips', 'm_dev')
    index = {c: k for k, c in enumerate(cl['cluster_id'].to_list())}
    owner = {g: index[c] for c, gs in zip(cl['cluster_id'], cl['geo_id']) for g in gs}
    pairs = {(min(owner[a], owner[b]), max(owner[a], owner[b])) for a, b in zip(*[adjacency[c].to_list() for c in ('a', 'b')])
             if a in owner and b in owner and owner[a] != owner[b]}
    pa = np.array(sorted(pairs), dtype=int).reshape(-1, 2)
    m_all, m_dev, st = cl['m'].to_numpy(), cl['m_dev'].to_numpy(), cl['state_fips'].to_numpy()
    def sym(vals, rows):
        return pearson(np.r_[vals[rows[:, 0]], vals[rows[:, 1]]], np.r_[vals[rows[:, 1]], vals[rows[:, 0]]]) if len(rows) else None
    within = pa[st[pa[:, 0]] == st[pa[:, 1]]] if len(pa) else pa
    out.update(adj_cluster_pairs=len(pa), r_adj_clusters_raw=sym(m_all, pa),
               adj_cluster_pairs_within_state=len(within), r_adj_clusters_within_state=sym(m_dev, within))
    return out


def persistence(a: pl.DataFrame, b: pl.DataFrame) -> tuple[dict, list[dict]]:
    """Within-state correlation of county u between two valid frames (matched counties); pooled and per state."""
    j = a.select('geo_id', 'state_fips', pl.col('u_c').alias('u0')).join(b.select('geo_id', pl.col('u_c').alias('u1')), on='geo_id')
    j = demean(demean(j, 'u0', 'state_fips', 'd0'), 'u1', 'state_fips', 'd1')
    pooled = dict(n=j.height, states=j['state_fips'].n_unique(),
                  r_within_state=pearson(j['d0'].to_numpy(), j['d1'].to_numpy()) if j.height else None,
                  r_raw=pearson(j['u0'].to_numpy(), j['u1'].to_numpy()) if j.height else None)
    per = []
    for state, s in j.group_by('state_fips', maintain_order=True):
        per.append(dict(state=state[0], n=s.height, r=pearson(s['u0'].to_numpy(), s['u1'].to_numpy())))
    return pooled, sorted(per, key=lambda r: r['state'])


# ---------------------------------------------------------------- report

INT_COLUMNS = {'counties_in_level', 'counties_covered', 'counties_valid', 'states_with_any', 'states_with_75pct', 'need_days_post', 'max_days_post',
               'need_days_pre', 'max_days_pre', 'n', 'clusters', 'adj_cluster_pairs', 'adj_cluster_pairs_within_state', 'adj_edges',
               'adj_edges_within_state', 'states', 'states_with_n10', 'n_states_dev', 'n_density', 'n_income', 'n_ec'}
CORRELATES = {'log_pop_density': 'density', 'log_median_income': 'income', 'e_c': 'ec'}
PERSISTENCE_PAIRS = (('w21_2y', 2024, 'w21', 2026, 'primary: 21-day change 2022->2024 vs 2024->2026'),
                     ('sa', 2024, 'sa', 2026, 'extra: S-a 2024 vs 2026'),
                     ('ky12', 2024, 'ky12', 2026, 'extra: ky12 2024 vs 2026'))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def report_inputs() -> dict:
    root = paths.repo_root()
    exposure = (pl.read_parquet(paths.processed_dir() / 'exposure_county.parquet').filter(pl.col('g_method') == 'commute_vehicles')
                .select(pl.col('geo_id'), 'state_fips', 'log_pop_density', 'log_median_income', 'e_c'))
    votes = (pl.scan_parquet(paths.processed_dir() / 'results_county.parquet')
             .filter((pl.col('year') == 2024) & (pl.col('office') == 'PRESIDENT') & pl.col('party').is_in(['D', 'R']))
             .select('geo_id', 'votes').collect().group_by('geo_id').agg(pl.col('votes').sum()))
    states = pl.read_csv(root / 'config/states.csv', schema_overrides={'state_fips': pl.String, 'padd_group': pl.String})
    return dict(exposure=exposure, votes=votes, states=states)


def diagnostics(tables: dict[str, pl.DataFrame], inputs: Inputs, extra: dict, adjacency: pl.DataFrame) -> list[dict]:
    """Aggregate records for every table ('primary' and optionally 'interim'). No unit-level values."""
    records = []
    states, exposure, votes = extra['states'], extra['exposure'], extra['votes']
    geo = inputs.geography.filter(pl.col('state_fips') != ALASKA).join(votes.rename({'geo_id': 'county_fips'}), on='county_fips', how='left') \
        .with_columns(pl.col('votes').fill_null(0))
    total_votes = geo['votes'].sum()
    def add(section, rules, measure, cycle, level, state='US', **metrics):
        for metric, value in metrics.items():
            if isinstance(value, (np.floating, np.integer)):
                value = value.item()
            if isinstance(value, float) and not math.isfinite(value):
                value = None
            records.append(dict(section=section, rules=rules, measure=measure, cycle=cycle, level=level, state=state, metric=metric, value=value))
    for rules, t in tables.items():
        thresholds = {(sp.measure, sp.cycle, lv): (sp.min_post, sp.min_pre)
                      for sp in make_specs(inputs.windows, RULESETS[rules]) for lv in sp.levels}
        combos = t.select('measure', 'cycle', 'level').unique().sort('measure', 'cycle', 'level').iter_rows()
        for measure, cycle, level in combos:
            if rules == 'interim' and not (cycle == 2026 and measure in ('w21', 'sa')):
                continue
            sub = t.filter((pl.col('measure') == measure) & (pl.col('cycle') == cycle) & (pl.col('level') == level) & (pl.col('state_fips') != ALASKA))
            valid = valid_rows(t, measure, cycle, level)
            # 1. coverage
            by_state = (geo.group_by('state_fips').agg(pl.len().alias('all')).join(
                valid.group_by('state_fips').agg(pl.len().alias('n')), on='state_fips', how='left').with_columns(pl.col('n').fill_null(0)))
            vv = valid.join(geo.select(pl.col('county_fips').alias('geo_id'), 'votes'), on='geo_id', how='left')['votes'].sum()
            add('coverage', rules, measure, cycle, level, counties_in_level=sub.height, counties_covered=sub['coverage_ok'].sum(),
                counties_valid=valid.height, share_of_all_counties=valid.height / geo.height, states_with_any=int((by_state['n'] > 0).sum()),
                states_with_75pct=int((by_state['n'] / by_state['all'] >= .75).sum()),
                min_state_share=float((by_state['n'] / by_state['all']).min()), vote_share=vv / total_votes,
                max_days_post=sub['n_days_post'].max(), max_days_pre=sub['n_days_pre'].max(),
                need_days_post=thresholds[(measure, cycle, level)][0], need_days_pre=thresholds[(measure, cycle, level)][1])
            if valid.height < 3:
                continue
            # 2. within-state SD against the state deviations within PADD groups
            v = demean(demean(valid, 'u_c', 'state_fips', 'u_dev_state'), 'u_c', 'stratum_id', 'u_dev_stratum')
            cl = v.group_by('cluster_id').agg(pl.col('u_c').mean().alias('m'), pl.col('stratum_id').first())
            cl = demean(cl, 'm', 'stratum_id', 'm_dev')
            state_mean = v.group_by('state_fips').agg(pl.col('u_c').mean().alias('m'))
            st = inputs.state_shocks.get((measure, cycle))
            sd_state = {}
            if st is not None:
                for label, ex in (('with_ak', False), ('ex_ak', True)):
                    dev = padd_deviation(states, st, ex)
                    sd_state[label] = (dev['deviation'].std(ddof=0), dev.height)
            add('sd', rules, measure, cycle, level, n=valid.height, sd_u=v['u_c'].std(ddof=0), sd_u_within_state=v['u_dev_state'].std(ddof=0),
                sd_u_within_state_stratum=v['u_dev_stratum'].std(ddof=0), sd_cluster_mean_within_stratum=cl['m_dev'].std(ddof=0),
                sd_within_cluster_resid=v['within_cluster_resid'].std(ddof=0), sd_state_mean_u=state_mean['m'].std(ddof=0),
                sd_state_dev_padd_with_ak=sd_state.get('with_ak', (None, 0))[0], sd_state_dev_padd_ex_ak=sd_state.get('ex_ak', (None, 0))[0],
                n_states_dev=sd_state.get('ex_ak', (None, 0))[1],
                ratio_within_state_to_state_dev=(v['u_dev_state'].std(ddof=0) / sd_state['ex_ak'][0]) if sd_state.get('ex_ak', (None,))[0] else None)
            for state, s in v.group_by('state_fips', maintain_order=True):
                add('sd_state', rules, measure, cycle, level, state=state[0], n=s.height, sd_u_within_state=s['u_c'].std(ddof=0) if s.height > 1 else None)
            # 3. spatial dependence
            add('spatial', rules, measure, cycle, level, **spatial_stats(valid, adjacency))
            # 5. correlates
            j = v.join(exposure.select('geo_id', 'log_pop_density', 'log_median_income', 'e_c'), on='geo_id')
            for var, short in CORRELATES.items():
                p = demean(j.filter(pl.col(var).is_finite()), var, 'state_fips', 'x_dev')
                add('correlate', rules, measure, cycle, level, **{f'n_{short}': p.height,
                    f'r_pooled_{short}': pearson(p['u_c'].to_numpy(), p[var].to_numpy()),
                    f'r_within_{short}': pearson(p['u_dev_state'].to_numpy(), p['x_dev'].to_numpy())})
        # 4. persistence (county level)
        for m0, c0, m1, c1, label in PERSISTENCE_PAIRS:
            if rules == 'interim' and m1 not in ('w21', 'sa'):
                continue
            pooled, per = persistence(valid_rows(t, m0, c0, 'county'), valid_rows(t, m1, c1, 'county'))
            add('persistence', rules, pair_label(m0, c0, m1, c1), c1, 'county', **pooled)
            usable = [r for r in per if r['r'] is not None and r['n'] >= 10]
            rs = np.array([r['r'] for r in usable])
            add('persistence_states', rules, pair_label(m0, c0, m1, c1), c1, 'county', states_with_n10=len(usable),
                median_r=float(np.median(rs)) if len(rs) else None, q25_r=float(np.quantile(rs, .25)) if len(rs) else None,
                q75_r=float(np.quantile(rs, .75)) if len(rs) else None, share_positive=float((rs > 0).mean()) if len(rs) else None)
            for r in per:
                add('persistence_state', rules, pair_label(m0, c0, m1, c1), c1, 'county', state=r['state'], n=r['n'], r=r['r'])
    return records


def records_to_frame(records: list[dict]) -> pl.DataFrame:
    return (pl.DataFrame(records, schema={'section': pl.String, 'rules': pl.String, 'measure': pl.String, 'cycle': pl.Int32, 'level': pl.String,
                                          'state': pl.String, 'metric': pl.String, 'value': pl.Float64}, strict=False)
            .with_columns(pl.col('value').round(9)))


def pair_label(m0, c0, m1, c1) -> str:
    return f'{m0}[{c0}] vs {m1}[{c1}]'


def as_ints(rows: list[dict]) -> list[dict]:
    return [{k: int(v) if k in INT_COLUMNS and v is not None else v for k, v in r.items()} for r in rows]


def wide(records: pl.DataFrame, section: str, rules: str = 'primary', keys=('measure', 'cycle', 'level'), state: str = 'US') -> list[dict]:
    r = records.filter((pl.col('section') == section) & (pl.col('rules') == rules) & (pl.col('state') == state))
    if not r.height:
        return []
    return as_ints(r.pivot(on='metric', index=list(keys), values='value', aggregate_function='first', maintain_order=True).sort(*keys).to_dicts())


def write_report(records: pl.DataFrame, meta: dict, out: Path) -> None:
    md = cp.md_table
    c = {k: wide(records, k) for k in ('coverage', 'sd', 'spatial', 'correlate')}
    interim = {k: wide(records, k, 'interim') for k in ('coverage', 'sd', 'spatial')}
    persist = as_ints(records.filter(pl.col('section') == 'persistence').pivot(on='metric', index=['rules', 'measure'], values='value', aggregate_function='first', maintain_order=True).to_dicts())
    pstates = as_ints(records.filter(pl.col('section') == 'persistence_states').pivot(on='metric', index=['rules', 'measure'], values='value', aggregate_function='first', maintain_order=True).to_dicts())
    lines = [
        '# Sub-state gasoline-price shocks: treatment-only diagnostics', '',
        'Treatment data only. Shocks are differences of archived AAA county and metro prices (memo D19: not redistributed); this report holds aggregates, no price values. '
        'No election outcome or swing enters; 2024 two-party presidential vote counts are used only to weight coverage shares. '
        'Alaska (memo D3) is kept in the parquet, flagged `alaska_not_in_sample`, and left out of every statistic below.', '',
        f'Builder `{VERSION}`; command `python -m fb26.etl.substate_shocks`. Price data end {meta["county_through"]} (county) and {meta["metro_through"]} (metro); '
        'every 2026 row is provisional. Input hashes are in `logs/run_log.csv`.', '',
        '## Rules and definitions', '',
        '- **Measures.** `w21`: W(ED_t) minus W(ED_t-2); `sa`: W(ED_t) minus January of t; `ky12`: Nov 1 (t-1)..Oct 31 (t) minus the year before; '
        '`w21_2y` (2024 only): W(2024) minus W(2022), for the persistence diagnostic; '
        '`w21_hist` (2022 and 2024, metro only): W(ED_t) minus W(2020), the validation-cycle change of `shocks.parquet`. W(ED) = [ED-21, ED-1].',
        f'- **Coverage.** 21-day blocks need {MIN_DAYS} observed days; January {MIN_DAYS_JAN} of 31; 12-month blocks {int(KY_SHARE * 100)}% of their days ({int(KY_SHARE_METRO * 100)}% at the metro level, rounded down, memo D23); '
        f'`w21_hist` {HIST_MIN_DAYS} day in each window (PI 2026-10-22). Failing units have null shocks. No imputation.',
        '- **Daily price.** Equal-weight mean across sources per county-day, then equal-weight observed days. Metro level: a county takes the price of its narrowest matched metro label. '
        'Metro days before the daily panel starts (2023-12-27) come from the archived AAA state pages (`prices_metro_wayback.parquet`): a price printed for the day beats a lag-column value, the shortest lag wins; '
        'source `metro_wayback`, so the 2024 `w21_hist` rows carry `source_switch`.',
        '- **u_c = g_c - g_s**, with g_s the state shock of the same measure from `shocks.parquet` (w21, w21_2y, w21_hist), `shocks_r9a.parquet` (sa) and `shocks_ky12.parquet` (ky12). '
        'County and state prices come from different archives, so the mean of u_c within a state is not exactly zero (`sd_state_mean_u` below).',
        '- **Clusters** are the CBSA (OMB July 2023) for counties inside one, else the county\'s 2020 commuting zone, intersected with the state; **strata** are state x CBSA type.',
        '- **SDs** are population SDs (ddof 0) over covered counties, equal weights. "State dev (PADD)" is the SD of state shocks minus their PADD-group mean (ex-AK, available states).', '',
        '## 1. Coverage', '',
        'Vote share is the share of 2024 D+R presidential votes (Alaska excluded) in covered counties. `states_with_75pct`: states in which at least 75% of counties are covered. '
        'At the metro level a county is "in the level" only if it lies in a matched metro, so shares are relative to all counties.', '',
        md(c['coverage'], ['measure', 'cycle', 'level', 'counties_in_level', 'counties_covered', 'counties_valid', 'share_of_all_counties', 'states_with_any',
                           'states_with_75pct', 'min_state_share', 'vote_share', 'need_days_post', 'max_days_post', 'need_days_pre', 'max_days_pre']), '']
    if interim['coverage']:
        lines += ['**Interim view of the open 2026 windows** (post-window threshold lowered from 10 to 3 days; the pre-window rules are unchanged). '
                  'Not a primary result; the windows close on 2026-11-02 and the builder must be rerun.', '',
                  md(interim['coverage'], ['measure', 'cycle', 'level', 'counties_valid', 'share_of_all_counties', 'states_with_any', 'states_with_75pct', 'vote_share', 'max_days_post']), '']
    lines += ['## 2. Within-state variation of u_c', '',
              'Compare `sd_u_within_state` with `sd_state_dev_padd_ex_ak` (the variation the state-level design uses). `sd_cluster_mean_within_stratum` is the part the sub-state permutation layer exchanges; '
              '`sd_within_cluster_resid` stays fixed. Per-state SDs are in the CSV.', '',
              md(c['sd'], ['measure', 'cycle', 'level', 'n', 'sd_u', 'sd_u_within_state', 'sd_u_within_state_stratum', 'sd_cluster_mean_within_stratum', 'sd_within_cluster_resid',
                           'sd_state_mean_u', 'sd_state_dev_padd_with_ak', 'sd_state_dev_padd_ex_ak', 'ratio_within_state_to_state_dev']), '']
    if interim['sd']:
        lines += ['Interim (open 2026 windows, post threshold 3 days):', '',
                  md(interim['sd'], ['measure', 'cycle', 'level', 'n', 'sd_u', 'sd_u_within_state', 'sd_cluster_mean_within_stratum', 'sd_within_cluster_resid',
                                     'sd_state_dev_padd_ex_ak', 'ratio_within_state_to_state_dev']), '']
    lines += ['## 3. Spatial dependence', '',
              '- `icc_raw`: ICC(1) of u_c within clusters (clusters with at least 2 covered counties). `icc_stratum`: the same after removing state x CBSA-type means. At the metro level u_c is constant within a metro label, so the ICC is mechanical.',
              '- `r_adj_clusters_raw`: correlation of cluster means across adjacent clusters (any pair of adjacent counties in different clusters, across states too). '
              '`r_adj_clusters_within_state`: the same within states after removing the state mean of cluster means.',
              '- `moran_raw`: Moran\'s I of u_c over adjacent covered counties (binary Census adjacency, 2024). `moran_within_state`: u_c minus its state mean, same-state pairs only, permuted within states. '
              f'Permutation p from {N_PERM} draws (seed {SEED}); `*_expected` is the permutation mean.', '',
              md(c['spatial'], ['measure', 'cycle', 'level', 'clusters', 'icc_raw', 'icc_stratum', 'share_counties_singleton_cluster', 'adj_cluster_pairs', 'r_adj_clusters_raw',
                                'adj_cluster_pairs_within_state', 'r_adj_clusters_within_state']), '',
              md(c['spatial'], ['measure', 'cycle', 'level', 'adj_edges', 'moran_raw', 'moran_raw_expected', 'moran_raw_p', 'adj_edges_within_state', 'moran_within_state',
                                'moran_within_state_expected', 'moran_within_state_p']), '']
    if interim['spatial']:
        lines += ['Interim (open 2026 windows):', '',
                  md(interim['spatial'], ['measure', 'cycle', 'level', 'clusters', 'icc_raw', 'icc_stratum', 'r_adj_clusters_within_state', 'moran_raw', 'moran_within_state', 'moran_within_state_p']), '']
    lines += ['## 4. Persistence of within-state county deviations', '',
              'Descriptive; not a decision rule. The pooled correlation demeans each period\'s u_c by state over the counties covered in both periods. Per-state correlations (states with at least 10 matched counties) are in the CSV.', '',
              '**Caution on the primary pair.** The 2022->2024 and 2024->2026 changes share the 2024 window, with opposite signs. Any noise in a county\'s 2024 window mean enters one change as +noise and the other as -noise, '
              'so the correlation is biased downward by construction even if true deviations persist. The S-a and ky12 pairs use non-overlapping blocks and do not have this problem. '
              'A negative primary correlation therefore cannot be read as mean reversion without that qualification. County coverage also differs: the 2026 side is limited to the states with county prices in 2026.', '',
              md(persist, ['rules', 'measure', 'n', 'states', 'r_within_state', 'r_raw']), '',
              md(pstates, ['rules', 'measure', 'states_with_n10', 'median_r', 'q25_r', 'q75_r', 'share_positive']), '',
              '## 5. Correlates of u_c (treatment-only)', '',
              'Pearson correlations of u_c with log population density, log median household income and E_c (`g_method == commute_vehicles`). `r_within_state` demeans both sides by state. Correlations only.', '',
              md(c['correlate'], ['measure', 'cycle', 'level'] + [f'{k}_{v}' for v in CORRELATES.values() for k in ('n', 'r_pooled', 'r_within')]), '']
    lines += ['## Notes and limits', '',
              '- **Open 2026 windows.** County and metro prices end on 2026-10-18/19, so the 2026 21-day windows (Oct 13 to Nov 2) have 5 observed days and fail the 10-day rule everywhere. '
              'The primary 2026 `w21` and `sa` rows are therefore null; only the interim rows above show what the data look like so far. Rerun after 2026-11-03.',
              '- **2026 county coverage is 17 states.** Since 2026-10-04 the county archive holds only states with FIPS 01-23 (Connecticut and DC have no matched county prices), and ky12 additionally needs 80% of the 2026 block, which counties in other states lack after 2026-03-27.',
              '- **Metro ky12 2026** is null everywhere: the metro archive has no more than 278 (post block) and 255 (pre block) observed days against the 292 needed.',
              '- **County vs state source.** County and metro prices come from different archives than the state series, so u_c carries a state-level component (`sd_state_mean_u`) that is not sampling noise. The assignment model removes it through the expected-deviation recentering, not here.',
              '- **No placeholder exclusion.** Counties whose prices equal the state average on most days are flagged `placeholder_suspect` in the parquet, not dropped.',
              '- **w21_hist rests on sparse archive days.** The 2020 window has a median of 3 archived days per state and the 2022 window 9; reliability of the within-state deviations, '
              'before and after the validation controls: `reports/validation/metro_wayback_coverage.md` §4 and `reports/validation/metro_resid_reliability.csv`.', '']
    out.write_text('\n'.join(lines))
    records.sort('section', 'rules', 'measure', 'cycle', 'level', 'state', 'metric').write_csv(out.with_suffix('.csv'))


# ---------------------------------------------------------------- build

def load_inputs() -> tuple[Inputs, pl.DataFrame, dict]:
    extra = report_inputs()
    universe = extra['exposure'].select(pl.col('geo_id').alias('county_fips'), 'state_fips')
    types = read_cbsa_types(rawstore.load_raw(mp.cbsa_path()))
    cz = read_cz(rawstore.load_raw(latest_raw('commuting_zones', 'csv.gz')))
    adjacency = read_adjacency(rawstore.load_raw(latest_raw('county_adjacency', 'txt.gz')))
    geography = build_geography(universe, types, cz, extra['states'])
    metro, metro_cbsa = metro_daily(universe)
    inputs = Inputs(windows=load_windows(), geography=geography, county_daily=county_daily(), metro_daily=metro, metro_cbsa=metro_cbsa,
                    state_shocks=state_shock_tables(paths.processed_dir()))
    return inputs, adjacency, extra


def main():
    command = sys.argv[1] if sys.argv[1:] else 'all'
    if command not in ('all', 'build', 'report') or sys.argv[2:]:
        raise SystemExit('Usage: python -m fb26.etl.substate_shocks [all|build|report]')
    root = paths.repo_root()
    inputs, adjacency, extra = load_inputs()
    primary = assemble(inputs)
    out = paths.processed_dir() / 'substate_shocks.parquet'
    if command in ('all', 'build'):
        primary.write_parquet(out)
        print(f'Wrote {primary.height:,} rows to {out.relative_to(root)}')
    if command in ('all', 'report'):
        interim = assemble(inputs, INTERIM)
        records = records_to_frame(diagnostics({'primary': primary, 'interim': interim}, inputs, extra, adjacency))
        meta = dict(county_through=horizon_of(inputs.county_daily), metro_through=horizon_of(inputs.metro_daily))
        write_report(records, meta, root / 'reports/validation/substate_shocks.md')
    snapshot = {p.name: sha(p) for p in [paths.interim_dir() / 'prices_county_daily.parquet', paths.interim_dir() / 'prices_metro_daily.parquet',
                                         *[paths.processed_dir() / n for n in sorted(set(STATE_FILES.values()))],
                                         paths.processed_dir() / 'exposure_county.parquet', paths.processed_dir() / 'results_county.parquet'] if p.exists()}
    snapshot.update(nathan_bundle=latest_record('nathanqthai_gas_prices_repo').sha256, cbsa=sha(mp.cbsa_path()),
                    commuting_zones=sha(latest_raw('commuting_zones', 'csv.gz')), county_adjacency=sha(latest_raw('county_adjacency', 'txt.gz')))
    runlog.append_run_log(run_id=f'substate_shocks_{hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()[:12]}',
                          utc_timestamp=rawstore.utc_iso(datetime.now(timezone.utc)), git_sha=runlog.git_sha(), git_dirty=runlog.git_dirty(),
                          config_hash=runlog.config_hash([root / 'config/states.csv', root / 'config/windows.yaml', Path(__file__)]),
                          spec_id='substate_shocks', data_snapshot_ids=json.dumps(snapshot, sort_keys=True),
                          output_path=str(out.relative_to(root)), notes=f'{command}; treatment-only; no election outcomes')


if __name__ == '__main__':
    main()
