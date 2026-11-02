"""Offline, conservative AAA metro feasibility. Restricted prices stay in interim parquet.

python -m fb26.etl.metro_prices [build|feasibility]
No network, mirror writes, imputation, fuzzy geographic assignments, or design changes.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import sys
import unicodedata
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from fb26.scrape.aaagas import _git, read_blobs, load_states
from fb26.etl.county_prices import WINDOWS, price_value, md_table
from fb26.util import paths, rawstore, runlog

VERSION = 'metro-prices-0.2.0'
PREFIX = 'data/metro-daily-averages'
# Primary first: where both archives carry a valid value for a day and metro the primary value is kept.
SOURCES = {'lykmapipo_metro': 'lykmapipo_mirror.git', 'lykfork_metro': 'lykfork_mirror.git'}
DECISIONS = ('match', 'state_part', 'no_cbsa', 'ambiguous')
OVERRIDE_COLUMNS = ['state_fips', 'metro_raw', 'cbsa_code', 'county_fips_list', 'decision', 'evidence']
SCHEMA = {'source': pl.String, 'as_of_date': pl.Date, 'state_fips': pl.String,
          'metro_raw': pl.String, 'cbsa_code': pl.String, 'price_usd': pl.Float64,
          'commit': pl.String, 'parser_version': pl.String, 'flags': pl.String}


def read_cbsa(content: bytes) -> pl.DataFrame:
    """Read OMB List 1 using the standard library; preserve leading-zero FIPS."""
    ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        strings = []
        if 'xl/sharedStrings.xml' in z.namelist():
            strings = [''.join(n.itertext()) for n in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si', ns)]
        sheet = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
    rows, header = [], None
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
        if 'CBSA Code' in cells.values():
            header = {v: k for k, v in cells.items()}
            continue
        if header is None:
            continue
        def get(name):
            return cells.get(header[name], '')
        if not re.fullmatch(r'\d{5}', get('CBSA Code')):
            continue
        state, county = get('FIPS State Code'), get('FIPS County Code')
        if not state.isdigit() or not county.isdigit():
            continue
        rows.append(dict(cbsa_code=get('CBSA Code'), cbsa_title=get('CBSA Title'),
                         state_fips=state.zfill(2), county_fips=state.zfill(2) + county.zfill(3)))
    if not rows:
        raise ValueError('No CBSA county rows in workbook')
    df = pl.DataFrame(rows).unique().sort('cbsa_code', 'county_fips')
    if df.group_by('county_fips').agg(pl.col('cbsa_code').n_unique().alias('n')).filter(pl.col('n') > 1).height:
        raise ValueError('County belongs to multiple CBSAs')
    return df


def cbsa_path() -> Path:
    files = sorted((paths.raw_dir() / 'cbsa/2026/10/19').glob('*.xlsx.gz'))
    if len(files) != 1:
        raise ValueError('Expected one supplied July 2023 CBSA workbook')
    return files[0]


def normalize(label: str) -> str:
    label = re.sub(r'\((?:[A-Z]{2} only|Entire Metro Average)\)', '', label, flags=re.I)
    label = unicodedata.normalize('NFKD', label).encode('ascii', 'ignore').decode().lower()
    label = re.sub(r'\bsaint\b', 'st', label)
    return re.sub(r'[^a-z0-9]', '', label)


def read_overrides(path: Path | None = None) -> pl.DataFrame:
    """Reviewed label decisions (config/metro_cbsa_overrides.csv); one row per state and label."""
    path = path or paths.config_dir() / 'metro_cbsa_overrides.csv'
    df = pl.read_csv(path, schema_overrides={c: pl.String for c in OVERRIDE_COLUMNS}).with_columns(
        pl.col('cbsa_code', 'county_fips_list', 'evidence').fill_null(''))
    if df.columns != OVERRIDE_COLUMNS:
        raise ValueError(f'Override columns must be {OVERRIDE_COLUMNS}')
    return df


def validate_overrides(overrides: pl.DataFrame, cbsa: pl.DataFrame) -> None:
    """Fail closed: unknown decisions, CBSAs without a county in the label's state, malformed county lists."""
    if overrides.select('state_fips', 'metro_raw').is_duplicated().any():
        raise ValueError('Duplicate override labels')
    in_state = set(cbsa.select('cbsa_code', 'state_fips').iter_rows())
    for row in overrides.iter_rows(named=True):
        who = f"{row['state_fips']} {row['metro_raw']}"
        if row['decision'] not in DECISIONS:
            raise ValueError(f'{who}: unknown decision {row["decision"]!r}')
        counties = [c for c in row['county_fips_list'].split(';') if c]
        if row['decision'] in ('match', 'state_part'):
            if (row['cbsa_code'], row['state_fips']) not in in_state:
                raise ValueError(f'{who}: CBSA {row["cbsa_code"]!r} has no county in the state')
        elif row['cbsa_code'] or counties:
            raise ValueError(f'{who}: {row["decision"]} must leave CBSA and counties blank')
        if row['decision'] == 'state_part' and not counties:
            raise ValueError(f'{who}: state_part needs counties')
        if row['decision'] != 'state_part' and counties:
            raise ValueError(f'{who}: only state_part carries counties')
        if any(not re.fullmatch(r'\d{5}', c) or c[:2] != row['state_fips'] for c in counties):
            raise ValueError(f'{who}: counties must be five-digit FIPS inside the state')
        if not row['evidence'].strip():
            raise ValueError(f'{who}: evidence is required')


def match_labels(labels: pl.DataFrame, cbsa: pl.DataFrame, overrides: pl.DataFrame | None = None) -> pl.DataFrame:
    """Automatic exact/normalized title match, then reviewed overrides on top (which win)."""
    areas = {}
    for row in cbsa.iter_rows(named=True):
        area = areas.setdefault(row['cbsa_code'], dict(title=row['cbsa_title'], states=set(), counties=set()))
        area['states'].add(row['state_fips'])
        area['counties'].add(row['county_fips'])
    reviewed = {}
    if overrides is not None and overrides.height:
        validate_overrides(overrides, cbsa)
        reviewed = {(r['state_fips'], r['metro_raw']): r for r in overrides.iter_rows(named=True)}
    rows = []
    for row in labels.unique().sort('state_fips', 'metro_raw').iter_rows(named=True):
        state, label = row['state_fips'], row['metro_raw']
        available = {k: v for k, v in areas.items() if state in v['states']}
        exact = [k for k, v in available.items() if label == v['title'].split(',')[0]]
        normalized = [k for k, v in available.items() if normalize(label) == normalize(v['title'].split(',')[0])]
        candidates = exact or normalized
        code = candidates[0] if len(candidates) == 1 else None
        match_type = ('exact' if exact else 'normalized') if code else 'manual-review needed'
        if not candidates:
            # Suggestions only, NEVER used for assignment: partial city-name overlap.
            key = normalize(label)
            candidates = [k for k, v in available.items() if key and
                          (normalize(v['title'].split(',')[0]).startswith(key) or
                           any(normalize(t) == key for t in v['title'].split(',')[0].split('-')) or
                           bool({normalize(t) for t in re.sub(r'\([^)]*\)', '', label).split('-')} &
                                {normalize(t) for t in v['title'].split(',')[0].split('-')}))]
        area = areas.get(code)
        counties = sorted(area['counties']) if area else []
        decision, evidence = ('match' if area else 'unreviewed'), ''
        if (state, label) in reviewed:
            ov = reviewed[(state, label)]
            decision, evidence, match_type = ov['decision'], ov['evidence'].strip(), f"override:{ov['decision']}"
            code = ov['cbsa_code'] or None
            area = areas.get(code)
            counties = ([c for c in ov['county_fips_list'].split(';') if c] if decision == 'state_part'
                        else sorted(area['counties']) if area else [])
        rows.append({**row, 'cbsa_code': code, 'cbsa_title': area['title'] if area else None,
                     'match_type': match_type, 'decision': decision, 'county_fips': ';'.join(counties),
                     # Narrowest label wins a county; an explicit county list outranks a whole-CBSA name match of equal width.
                     'scope_size': len(counties) + (0.5 if decision in ('match', 'unreviewed') else 0),
                     'candidate_cbsa_codes': ';'.join(sorted(candidates)),
                     'cross_state': len(area['states']) > 1 if area else False, 'evidence': evidence})
    return pl.DataFrame(rows, schema_overrides={'cbsa_code': pl.String, 'cbsa_title': pl.String})


def parse_rows(content: bytes, file_date: date, commit: str, by_abbr: dict,
               rewritten: bool = False, source: str = 'lykmapipo_metro') -> list[dict]:
    """The fork renamed Regular to Regular-Current on 2026-08-20; both carry the same figure."""
    rows = []
    for rec in csv.DictReader(io.StringIO(content.decode('utf-8-sig'))):
        abbr = rec['State-Abbreviation']
        if abbr not in by_abbr:
            raise ValueError(f'Unknown state abbreviation: {abbr}')
        flags = ['rewritten_after_first_commit'] if rewritten else []
        try:
            day = date.fromisoformat(rec['Date'])
        except (ValueError, TypeError):
            day = None
            flags.append('missing_or_invalid_row_date')
        if day and day != file_date:
            flags.append('date_ne_filename')
        value, invalid = price_value(rec['Regular'] if 'Regular' in rec else rec['Regular-Current'])
        if invalid:
            flags.append(invalid)
        if rec.get('Currency', 'U.S Dollar') != 'U.S Dollar' or rec.get('Unit', 'US Gallon') != 'US Gallon':
            value = None
            flags.append('unit_unexpected')
        if day and WINDOWS[2026][0] <= day <= WINDOWS[2026][1]:
            flags.append('provisional_2026_window')
        rows.append(dict(source=source, as_of_date=day, state_fips=by_abbr[abbr],
                         metro_raw=rec['Metro-Name'].strip(), cbsa_code=None, price_usd=value,
                         commit=commit, parser_version=VERSION, flags=';'.join(flags)))
    return rows


def file_history(repo: Path, ref: str) -> tuple[dict, Counter]:
    """Same last-writing-commit approach as aaagas.lyk_history, including deleted files."""
    log = _git(['log', '--reverse', '--name-status', '--no-renames', '--format=@@C%x09%H%x09%cI',
                ref, '--', PREFIX], git_dir=repo, timeout=600).decode()
    last, writes, commit = {}, Counter(), None
    for line in log.splitlines():
        if line.startswith('@@C\t'):
            _, sha, when = line.split('\t')
            commit = (sha, when)
        elif line and commit:
            status, _, name = line.partition('\t')
            if status in ('A', 'M') and re.fullmatch(re.escape(PREFIX) + r'/\d{4}-\d{2}-\d{2}\.csv', name):
                last[name] = commit
                writes[name] += 1
    return last, writes


def read_source(source: str, repo: Path) -> pl.DataFrame:
    """One archive: every dated file at its last writing commit; latest capture wins duplicate keys."""
    tip = _git(['rev-parse', 'HEAD'], git_dir=repo).decode().strip()
    last, writes = file_history(repo, tip)
    names, frames = sorted(last), []
    by_abbr = load_states()[1]
    for start in range(0, len(names), 40):
        batch = names[start:start + 40]
        specs = [f'{last[n][0]}:{n}' for n in batch]
        blobs = read_blobs(repo, specs)
        rows = []
        for n, spec in zip(batch, specs):
            for r in parse_rows(blobs[spec], date.fromisoformat(Path(n).stem), last[n][0], by_abbr, writes[n] > 1, source):
                r['_capture'] = last[n][1]
                rows.append(r)
        if rows:
            frames.append(pl.DataFrame(rows, schema={**SCHEMA, '_capture': pl.String}))
    df = pl.concat(frames)
    key = ['state_fips', 'metro_raw', 'as_of_date']
    return (df.with_columns(pl.len().over(key).alias('_n')).sort(key + ['_capture', 'commit'])
            .unique(subset=key, keep='last', maintain_order=True)
            .with_columns(pl.when(pl.col('_n') > 1).then(pl.col('flags') + ';duplicate_latest_capture')
                          .otherwise(pl.col('flags')).alias('flags')).drop('_capture', '_n', 'cbsa_code'))


def merge_sources(frames: list[pl.DataFrame]) -> pl.DataFrame:
    """One row per state, metro and date. `source` lists every archive carrying the key. The value comes from
    the first archive (SOURCES order) with a valid price; fallbacks and disagreements are flagged."""
    order = {s: i for i, s in enumerate(SOURCES)}
    key = ['state_fips', 'metro_raw', 'as_of_date']
    both = (pl.concat(frames).with_columns(pl.col('source').replace_strict(order, return_dtype=pl.Int64).alias('_rank'))
            .with_columns(pl.col('price_usd').is_not_null().alias('_valid')))
    summary = both.group_by(key).agg(pl.col('source').sort_by('_rank').str.join('+').alias('_sources'),
                                     pl.col('_rank').min().alias('_first'),
                                     pl.col('price_usd').drop_nulls().n_unique().alias('_values'),
                                     (pl.len() > 1).alias('_shared'), pl.col('_valid').all().alias('_all_valid'))
    chosen = (both.sort(key + ['_valid', '_rank'], descending=[False] * len(key) + [True, False])
              .unique(subset=key, keep='first', maintain_order=True).join(summary, on=key))
    fallback = pl.col('_valid') & (pl.col('_rank') > pl.col('_first'))
    return (chosen.with_columns(
        pl.when(fallback).then(pl.col('flags') + ';value_from_' + pl.col('source') + '_only').otherwise(pl.col('flags')).alias('flags'),
        pl.col('_sources').alias('source'))
        .with_columns(pl.when(pl.col('_shared') & pl.col('_valid') & ~pl.col('_all_valid') & ~fallback)
                      .then(pl.col('flags') + ';other_source_invalid').otherwise(pl.col('flags')).alias('flags'))
        .with_columns(pl.when(pl.col('_values') > 1).then(pl.col('flags') + ';source_conflict').otherwise(pl.col('flags')).alias('flags'))
        .with_columns(pl.col('flags').str.strip_chars(';'), pl.lit(None, dtype=pl.String).alias('cbsa_code'))
        .select(list(SCHEMA)).sort(key))


def build() -> Path:
    frames = [read_source(source, paths.interim_dir() / mirror) for source, mirror in SOURCES.items()]
    df = merge_sources(frames)
    key = ['state_fips', 'metro_raw', 'as_of_date']
    cbsa = read_cbsa(rawstore.load_raw(cbsa_path()))
    cw = match_labels(df.select('state_fips', 'metro_raw'), cbsa, read_overrides())
    df = (df.drop('cbsa_code').join(cw.select('state_fips', 'metro_raw', 'cbsa_code'), on=['state_fips', 'metro_raw'])
          .with_columns(pl.when(pl.col('cbsa_code').is_null()).then(pl.col('flags') + ';manual_review_needed')
                        .otherwise(pl.col('flags')).str.strip_chars(';').alias('flags')).select(list(SCHEMA)).sort(key))
    out = paths.interim_dir() / 'prices_metro_daily.parquet'
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    cw.select('state_fips', 'metro_raw', 'cbsa_code', 'cbsa_title', 'match_type', 'decision', 'county_fips', 'scope_size',
              'candidate_cbsa_codes', 'cross_state', 'evidence').write_csv(paths.repo_root() / 'reports/validation/metro_cbsa_crosswalk.csv')
    print(f'Built {df.height:,} rows from {len(SOURCES)} archives; {cw.filter(pl.col("cbsa_code").is_not_null()).height}/{cw.height} labels matched.')
    return out


def assign_counties(daily: pl.DataFrame, crosswalk: pl.DataFrame, universe: pl.DataFrame) -> pl.DataFrame:
    """County-day prices from matched labels, state-scoped. Where several labels cover a county the narrowest
    (smallest scope_size) wins for every day; remaining ties collapse only if that day's prices agree."""
    members = (crosswalk.filter(pl.col('cbsa_code').is_not_null())
               .select('state_fips', 'metro_raw', 'cbsa_code', 'scope_size', pl.col('county_fips').str.split(';'))
               .explode('county_fips').filter(pl.col('county_fips').str.slice(0, 2) == pl.col('state_fips'))
               .filter(pl.col('scope_size') == pl.col('scope_size').min().over('state_fips', 'county_fips')))
    valid = daily.filter(pl.col('price_usd').is_not_null() & pl.col('as_of_date').is_not_null()).drop('cbsa_code')
    assigned = valid.join(members, on=['state_fips', 'metro_raw']).join(universe.select('state_fips', 'county_fips'), on=['state_fips', 'county_fips'])
    key = ['state_fips', 'county_fips', 'as_of_date']
    return (assigned.group_by(key).agg(pl.col('price_usd').n_unique().alias('_n'), pl.col('price_usd').first(),
                                       pl.col('cbsa_code').min(), pl.col('metro_raw').min())
            .filter(pl.col('_n') == 1).drop('_n').select(key + ['cbsa_code', 'metro_raw', 'price_usd']).sort(key))


def source_agreement(df: pl.DataFrame) -> list[dict]:
    """How the two archives relate on the merged keys, from the flags set by merge_sources."""
    def has(pattern):
        return pl.col('flags').str.contains(pattern)
    rows = []
    for state in ['US'] + sorted(df['state_fips'].unique().to_list()):
        a = df if state == 'US' else df.filter(pl.col('state_fips') == state)
        shared = a.filter(pl.col('source').str.contains(r'\+'))
        days = a.group_by('as_of_date').agg(*[pl.col('source').str.contains(k).any().alias(k) for k in SOURCES])
        lyk, fork = (days[k] for k in SOURCES)
        rows.append(dict(state=state, keys=a.height, keys_both=shared.height,
                         both_equal=shared.filter(pl.col('price_usd').is_not_null() & ~has('source_conflict|other_source_invalid|value_from_')).height,
                         both_conflict=shared.filter(has('source_conflict')).height,
                         one_source_invalid=shared.filter(has('other_source_invalid|value_from_')).height,
                         both_invalid=shared.filter(pl.col('price_usd').is_null()).height,
                         dates_both=int((lyk & fork).sum()), dates_only_lykmapipo=int((lyk & ~fork).sum()),
                         dates_only_fork=int((~lyk & fork).sum())))
    return rows


def window_means(daily: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    frames = []
    for year, window in WINDOWS.items():
        frames.append(daily.filter(pl.col('as_of_date').is_between(*window)).group_by(keys)
                      .agg(pl.col('price_usd').mean().alias(f'p{year}'),
                           pl.col('as_of_date').n_unique().alias(f'n{year}')))
    return frames[0].join(frames[1], on=keys).with_columns((pl.col('p2026') - pl.col('p2024')).alias('shock'))


def demean(df: pl.DataFrame, col: str, out: str = 'deviation') -> pl.DataFrame:
    return df.with_columns((pl.col(col) - pl.col(col).mean().over('state_fips')).alias(out))


def correlation(df: pl.DataFrame, x: str, y: str) -> float | None:
    df = df.filter(pl.col(x).is_finite() & pl.col(y).is_finite())
    if df.height < 2 or not df[x].std(ddof=0) or not df[y].std(ddof=0):
        return None
    value = df.select(pl.corr(x, y)).item()
    return value if math.isfinite(value) else None


def ratio(a, b):
    return a / b if a is not None and b is not None and b > 0 else None


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def feasibility(*, log_run: bool = True) -> None:
    root = paths.repo_root()
    input_path = paths.interim_dir() / 'prices_metro_daily.parquet'
    df = pl.read_parquet(input_path)
    cw_path = root / 'reports/validation/metro_cbsa_crosswalk.csv'
    overrides_path = paths.config_dir() / 'metro_cbsa_overrides.csv'
    cw = pl.read_csv(cw_path, schema_overrides={'state_fips': pl.String, 'cbsa_code': pl.String, 'county_fips': pl.String, 'candidate_cbsa_codes': pl.String,
                                                'evidence': pl.String, 'scope_size': pl.Float64})
    exposure_path = paths.processed_dir() / 'exposure_county.parquet'
    exposure = (pl.scan_parquet(exposure_path).filter(pl.col('g_method') == 'commute_vehicles')
                .select(pl.col('geo_id').alias('county_fips'), 'state_fips', 'log_pop_density', 'log_median_income', 'e_c').collect())
    if exposure['county_fips'].n_unique() != exposure.height:
        raise ValueError('Exposure county universe is not unique')
    universe = exposure.select('county_fips', 'state_fips')
    votes_path = paths.processed_dir() / 'results_county.parquet'
    votes = (pl.scan_parquet(votes_path).filter((pl.col('year') == 2024) & (pl.col('office') == 'PRESIDENT') & pl.col('party').is_in(['D', 'R']))
             .select(pl.col('geo_id').alias('county_fips'), 'votes').collect().group_by('county_fips').agg(pl.col('votes').sum()))
    votes = votes.with_columns(pl.col('county_fips').str.slice(0, 2).alias('state_fips'))
    states = pl.read_csv(root / 'config/states.csv', schema_overrides={'state_fips': pl.String, 'padd_group': pl.String})
    state_ids = ['US'] + states['state_fips'].to_list()
    assigned = assign_counties(df, cw, universe)
    valid = df.filter(pl.col('price_usd').is_not_null() & pl.col('as_of_date').is_not_null())
    records, sections = [], []
    def add(section, state='US', period='', **metrics):
        for metric, value in metrics.items():
            if isinstance(value, float) and not math.isfinite(value):
                value = None
            records.append(dict(section=section, state=state, period=str(period), metric=metric, value=value))
    def subset(frame, state):
        return frame if state == 'US' else frame.filter(pl.col('state_fips') == state)
    def table(title, rows, cols):
        sections.extend(['', f'## {title}', '', md_table(rows, cols)])
    # Deterministic observation horizon from the pinned source commits, not the wall clock.
    tips, horizons = {}, []
    for source, mirror in SOURCES.items():
        repo = paths.interim_dir() / mirror
        tips[source] = _git(['rev-parse', 'HEAD'], git_dir=repo).decode().strip()
        horizons.append(datetime.fromisoformat(_git(['show', '-s', '--format=%cI', tips[source]], git_dir=repo).decode().strip()).date())
    horizon = max(horizons)
    periods = {str(y): w for y, w in WINDOWS.items()}
    periods.update({'August 2026': (date(2026, 8, 1), date(2026, 8, 31)), 'since October 4': (date(2026, 10, 4), max(date(2026, 10, 4), horizon))})
    daily_rows = []
    for period, (start, end) in periods.items():
        day = start
        while day <= end:
            s = valid.filter(pl.col('as_of_date') == day)
            raw = df.filter(pl.col('as_of_date') == day)
            status = 'beyond_snapshot' if day > horizon else ('missing' if not s.height else 'observed')
            daily_rows.append(dict(period=period, date=day, states=s['state_fips'].n_unique(), metros=s.select('state_fips', 'metro_raw').unique().height, status=status))
            for state in state_ids:
                a = subset(s, state)
                add('daily_coverage', state, day, metros=a.height, states=a['state_fips'].n_unique(),
                    invalid_prices=subset(raw, state)['price_usd'].null_count(), beyond_snapshot=int(day > horizon))
            day += timedelta(days=1)
    gap_rows = []
    for period, (start, end) in periods.items():
        observed = [r for r in daily_rows if r['period'] == period and r['status'] == 'observed']
        missing = [r['date'].isoformat() for r in daily_rows if r['period'] == period and r['status'] == 'missing']
        gap_rows.append(dict(period=period, observed_days=len(observed), missing_days=len(missing),
                             missing_dates=', '.join(missing) or 'none',
                             min_metros=min((r['metros'] for r in observed), default=0),
                             max_metros=max((r['metros'] for r in observed), default=0)))
        add('period_coverage', period=period, observed_days=len(observed), missing_days=len(missing))
    table('Archive gap summary', gap_rows, list(gap_rows[0]))
    # Period overlap produces identical records: remove duplicates before CSV output.
    table('Daily archive coverage (valid regular prices; metros are state × label)', daily_rows, ['period', 'date', 'states', 'metros', 'status'])
    agreement = source_agreement(df)
    for r in agreement:
        add('source_agreement', r['state'], **{k: v for k, v in r.items() if k != 'state'})
    table('Source agreement: lykmapipo vs fork (state × metro × date keys; national; per-state in CSV)', [r for r in agreement if r['state'] == 'US'], list(agreement[0])[1:])
    match_rows = []
    for state in state_ids:
        s = subset(cw, state)
        matched = s.filter(pl.col('cbsa_code').is_not_null())
        r = dict(state=state, labels=s.height, matched=matched.height, match_rate=ratio(matched.height, s.height),
                 cross_state=matched.filter(pl.col('cross_state')).height,
                 unresolved=s.filter(pl.col('cbsa_code').is_null()).height,
                 reviewed=s.filter(pl.col('match_type').str.starts_with('override:')).height,
                 state_part=s.filter(pl.col('decision') == 'state_part').height,
                 no_cbsa=s.filter(pl.col('decision') == 'no_cbsa').height,
                 ambiguous=s.filter(pl.col('decision') == 'ambiguous').height,
                 multiple_candidates=s.filter(pl.col('candidate_cbsa_codes').fill_null('').str.contains(';')).height)
        match_rows.append(r)
        add('matching', state, **{k: v for k, v in r.items() if k != 'state'})
    table('Matching', match_rows, list(match_rows[0]))
    coverage_rows = []
    for year, window in WINDOWS.items():
        counts = assigned.filter(pl.col('as_of_date').is_between(*window)).group_by('county_fips').agg(pl.col('as_of_date').n_unique().alias('days'))
        u = universe.join(counts, on='county_fips', how='left').with_columns(pl.col('days').fill_null(0)).join(votes.select('county_fips', 'votes'), on='county_fips', how='left')
        for state in state_ids:
            a = subset(u, state)
            denom = subset(votes, state)['votes'].sum()
            r = dict(year=year, state=state, counties=a.height, covered=a.filter(pl.col('days') > 0).height,
                     share=ratio(a.filter(pl.col('days') > 0).height, a.height),
                     vote_share=ratio(a.filter(pl.col('days') > 0)['votes'].sum(), denom),
                     no_metro_price=a.filter(pl.col('days') == 0).height,
                     min15=a.filter(pl.col('days') >= 15).height,
                     vote_share15=ratio(a.filter(pl.col('days') >= 15)['votes'].sum(), denom))
            coverage_rows.append(r)
            add('county_coverage', state, year, **{k: v for k, v in r.items() if k not in ['state', 'year']})
    table('County and two-party vote coverage', coverage_rows, list(coverage_rows[0]))
    days_rows = []
    for state in state_ids:
        r = dict(state=state)
        for year, window in WINDOWS.items():
            w = subset(valid.filter(pl.col('as_of_date').is_between(*window)), state)
            cover = next(c for c in coverage_rows if c['year'] == year and c['state'] == state)
            r.update({f'days{year}': w['as_of_date'].n_unique(), f'labels{year}': w.select('state_fips', 'metro_raw').unique().height,
                      f'counties{year}': cover['covered'], f'vote_share{year}': cover['vote_share']})
        days_rows.append(r)
        add('window_days', state, **{k: v for k, v in r.items() if k != 'state'})
    table('Observed days, labels and covered counties per state and window (valid prices; labels are state × metro name)', days_rows, list(days_rows[0]))
    shocks = window_means(assigned, ['state_fips', 'county_fips', 'cbsa_code', 'metro_raw'])
    variation_rows, corr_rows = [], []
    for minimum in (1, 15):
        sample = shocks.filter((pl.col('n2024') >= minimum) & (pl.col('n2026') >= minimum))
        county = demean(sample, 'shock')
        metro = demean(sample.unique(subset=['state_fips', 'metro_raw']), 'shock')
        for state in state_ids:
            a, m = subset(county, state), subset(metro, state)
            r = dict(sample=f'min{minimum}', state=state, counties=a.height, metros=m.height,
                     county_weighted_within_sd=a['deviation'].std(ddof=0), metro_weighted_within_sd=m['deviation'].std(ddof=0),
                     within_variance_share=ratio(a['deviation'].var(ddof=0), a['shock'].var(ddof=0)))
            variation_rows.append(r)
            add('shock_variation', state, f'min{minimum}', **{k: v for k, v in r.items() if k not in ['state', 'sample']})
        a = county.join(exposure, on=['county_fips', 'state_fips'])
        for state in state_ids:
            for variable in ['log_pop_density', 'log_median_income', 'e_c']:
                s = subset(a, state).filter(pl.col(variable).is_finite() & pl.col('shock').is_finite())
                s = demean(s, 'shock')
                r = dict(sample=f'min{minimum}', state=state, variable=variable, n=s.height, correlation=correlation(s, 'deviation', variable))
                corr_rows.append(r)
                add('correlates', state, f'min{minimum}:{variable}', n=s.height, correlation=r['correlation'])
    table('Shock variation', variation_rows, list(variation_rows[0]))
    table('Treatment-only correlates (national; per-state values in CSV)', [r for r in corr_rows if r['state'] == 'US'], list(corr_rows[0]))
    county_path = paths.interim_dir() / 'prices_county_daily.parquet'
    county_daily = (pl.scan_parquet(county_path).filter(pl.col('county_fips').is_not_null() & pl.col('price_usd').is_not_null() & pl.col('as_of_date').is_not_null())
                    .select('state_fips', 'county_fips', 'as_of_date', 'price_usd').collect()
                    .group_by('state_fips', 'county_fips', 'as_of_date').agg(pl.col('price_usd').mean()))
    # Fixed 19-state cohort requested by the task; separate partial Montana day is excluded.
    comparison_states = (pl.scan_parquet(county_path)
                         .filter(pl.col('as_of_date').is_between(*WINDOWS[2026]) & (pl.col('state_fips') <= '23'))
                         .select('state_fips').unique().collect()['state_fips'].sort().to_list())
    paired = assigned.join(county_daily, on=['state_fips', 'county_fips', 'as_of_date'], suffix='_county')
    agreement_rows = []
    for year, window in WINDOWS.items():
        pairs = paired.filter(pl.col('as_of_date').is_between(*window))
        if year == 2026:
            pairs = pairs.filter(pl.col('state_fips').is_in(comparison_states))
        for state in state_ids:
            a = subset(pairs, state).with_columns((pl.col('price_usd') - pl.col('price_usd_county')).alias('difference'))
            r = dict(year=year, state=state, county_days=a.height, counties=a['county_fips'].n_unique(),
                     correlation=correlation(a, 'price_usd', 'price_usd_county'), bias=a['difference'].mean(),
                     mae=a['difference'].abs().mean(), rmse=(a['difference'].pow(2).mean() ** .5 if a.height else None))
            agreement_rows.append(r)
            add('level_agreement', state, year, **{k: v for k, v in r.items() if k not in ['state', 'year']})
    window_level_rows = []
    for year, window in WINDOWS.items():
        left = assigned.filter(pl.col('as_of_date').is_between(*window)).group_by('state_fips', 'county_fips').agg(pl.col('price_usd').mean())
        right = county_daily.filter(pl.col('as_of_date').is_between(*window)).group_by('state_fips', 'county_fips').agg(pl.col('price_usd').mean())
        pairs = left.join(right, on=['state_fips', 'county_fips'], suffix='_county')
        if year == 2026:
            pairs = pairs.filter(pl.col('state_fips').is_in(comparison_states))
        for state in state_ids:
            a = subset(pairs, state).with_columns((pl.col('price_usd') - pl.col('price_usd_county')).alias('difference'))
            r = dict(year=year, state=state, counties=a.height,
                     correlation=correlation(a, 'price_usd', 'price_usd_county'), bias=a['difference'].mean(),
                     mae=a['difference'].abs().mean(), rmse=(a['difference'].pow(2).mean() ** .5 if a.height else None))
            window_level_rows.append(r)
            add('window_level_agreement', state, year, **{k: v for k, v in r.items() if k not in ['state', 'year']})
    table('Observed-window mean levels (different available dates; national; per-state in CSV)',
          [r for r in window_level_rows if r['state'] == 'US'], list(window_level_rows[0]))
    table('Metro vs county levels on identical county-days (national; per-state in CSV)', [r for r in agreement_rows if r['state'] == 'US'], list(agreement_rows[0]))
    comparison_rows = []
    county_shocks = window_means(county_daily, ['state_fips', 'county_fips'])
    common = shocks.join(county_shocks, on=['state_fips', 'county_fips'], suffix='_county').filter(pl.col('state_fips').is_in(comparison_states))
    # Also compute both window means using identical observed dates per county.
    synced_m = window_means(paired, ['state_fips', 'county_fips', 'cbsa_code', 'metro_raw'])
    synced_c = window_means(paired.drop('price_usd').rename({'price_usd_county': 'price_usd'}), ['state_fips', 'county_fips'])
    synced = synced_m.join(synced_c, on=['state_fips', 'county_fips'], suffix='_county').filter(pl.col('state_fips').is_in(comparison_states))
    for label, frame in [('observed_windows', common), ('identical_days', synced)]:
        frame = demean(demean(frame, 'shock', 'metro_dev'), 'shock_county', 'county_dev')
        for state in state_ids:
            a = subset(frame, state)
            r = correlation(a, 'metro_dev', 'county_dev')
            residual = a['county_dev'] - a['metro_dev']
            sst = a['county_dev'].pow(2).sum()
            # Group projection R² is separate from the actual AAA metro-price fit.
            grouped = a.with_columns(pl.col('county_dev').mean().over(['state_fips', 'cbsa_code']).alias('group_mean'))
            result = dict(sample=label, state=state, counties=a.height,
                          shock_correlation=correlation(a, 'shock', 'shock_county'), within_correlation=r,
                          fitted_variance_share=r * r if r is not None else None,
                          assignment_variance_ratio=ratio(a['metro_dev'].var(ddof=0), a['county_dev'].var(ddof=0)),
                          direct_capture=1 - residual.pow(2).sum() / sst if sst > 0 else None,
                          cbsa_group_variance_share=ratio(grouped['group_mean'].pow(2).sum(), sst))
            comparison_rows.append(result)
            add('shock_agreement', state, label, **{k: v for k, v in result.items() if k not in ['state', 'sample']})
    table('Metro vs county shocks (national; per-state in CSV)', [r for r in comparison_rows if r['state'] == 'US'], list(comparison_rows[0]))
    input_files = [input_path, cw_path, overrides_path, cbsa_path(), exposure_path, votes_path, county_path, root / 'config/states.csv', Path(__file__), root / 'src/fb26/etl/county_prices.py', root / 'src/fb26/scrape/aaagas.py']
    hashes = {str(p.relative_to(root)): sha(p) for p in input_files}
    missing_votes = votes.join(universe, on=['county_fips', 'state_fips'], how='anti')['votes'].sum()
    unmatched = cw.filter(pl.col('cbsa_code').is_null())
    duplicate_cbsa = cw.filter(pl.col('cbsa_code').is_not_null()).group_by('state_fips', 'cbsa_code').len().filter(pl.col('len') > 1).height
    lines = ['# AAA metro-price feasibility — provisional 2026', '',
             'Treatment-only feasibility; no change to the preregistered state-level treatment. No individual metro/county prices or shocks are published. All dollar statistics below are state or national aggregates.', '',
             f'Parser `{VERSION}`. Read-only mirror HEADs {", ".join(f"{k} `{v}`" for k, v in tips.items())}; snapshot horizon {horizon}. Observed row dates {valid["as_of_date"].min()}–{valid["as_of_date"].max()}.',
             'Commands: `python -m fb26.etl.metro_prices` then `python -m fb26.etl.metro_prices feasibility`. Build reads each dated CSV at its last writing commit, including files deleted from HEAD, as in the existing state parser. The row Date is authoritative; filename discrepancies, rewrites, invalid values and duplicates are flagged. Latest commit timestamp wins duplicate keys (commit hash breaks ties). No missing day is carried forward. Two archives are read: lykmapipo (stopped after 2026-10-13) and its fork, which kept scraping and renamed `Regular` to `Regular-Current` from 2026-08-20 (both carry the same figure). One row per state, metro and date is kept; where both archives have a valid price the lykmapipo value is used, and a fork value fills a day where lykmapipo is missing or invalid. The section on source agreement lists how often the two differ.', '',
             'The archive gap summary below is decisive for feasibility. If the 2026 same-day comparison has zero pairs, identical-day shock agreement is unavailable; any observed-window agreement then uses different dates and cannot establish measurement agreement.', '',
             '## Geographic rules and limitations', '',
             'Match within the AAA-listed state to the CBSA title before its state suffix. Exact equality or normalization of case, accents, punctuation, Saint/St and explicit parenthetical state-only/entire-metro qualifiers is accepted. No abbreviated, partial, legacy, division or county-group name is automatically expanded. Micro- and metropolitan CBSAs are both allowed. Labels that this rule does not resolve were reviewed by hand in `config/metro_cbsa_overrides.csv` (decision match, state_part, no_cbsa or ambiguous, each with a one-line evidence note), using only the July 2023 OMB file and the label text; the override wins over the automatic result and the crosswalk records the match type. A matching label is evidence of name correspondence, not verification of AAA sampling boundaries.',
             'The crosswalk contains every label, accepted CBSA/title, the counties the label is assigned to (`county_fips`, `scope_size`), cross-state indicators, candidate codes for unresolved names and, for reviewed labels, the evidence note; candidates are suggestions only. state_part labels cover part of a CBSA (a state-only qualifier, a metropolitan division, named counties, or a label naming several CBSAs, in which case `cbsa_code` is the first CBSA). no_cbsa and ambiguous labels are left unassigned. No fuzzy match is used.',
             'For a cross-state CBSA, assign only counties in the AAA-listed state, even for Entire Metro Average. Never borrow another state’s listing. Where several labels cover one county the narrowest (smallest `scope_size`) is used for every day of that county; labels of equal width are collapsed for a county-day only when values agree, and conflicting values leave that day unassigned. This conservative proxy requires boundary validation before analysis.',
             f'{unmatched.height} labels have no CBSA ({unmatched.filter(pl.col("decision") == "no_cbsa").height} reviewed no_cbsa, {unmatched.filter(pl.col("decision") == "ambiguous").height} reviewed ambiguous, {unmatched.filter(pl.col("decision") == "unreviewed").height} unreviewed); {duplicate_cbsa} state-CBSA combinations have multiple accepted labels. County membership uses July 2023 OMB delineations, intersected with the existing exposure county universe ({universe.height} counties); no boundary interpolation.',
             f'Counties without a valid assigned price are flagged by the `no_metro_price` aggregate; they remain missing in memory and no extra county-level file is written. Vote coverage denominator includes all 2024 PRESIDENT D+R votes, including {missing_votes:,} votes outside the exposure geography. Missing vote records are not imputed.', '',
             '## Statistical definitions', '',
             'All estimates are provisional observed-day means, m = mean(2026-10-13–11-02) minus mean(2024-10-15–11-04). min1 is exploratory; min15 retains the county report’s 15/21-day diagnostic without relaxing it for an open window. Daily coverage counts valid prices, including unmatched labels. Days beyond the mirror commit date are marked beyond_snapshot, not treated as observed gaps or necessarily as future calendar days.',
             'County-weighted moments give each assigned county equal weight; metro-weighted moments give each state and AAA label equal weight. SD uses population moments (ddof=0). Within variance share = Var(m minus its sample state mean)/Var(m). County vs metro levels report dollar bias, MAE and RMSE, both on identical county-days and on separately observed window means (the latter can confound geographic differences with timing). The 2026 comparison restricts to the 19 FIPS 01–23 states with county data; the separate partial Montana observation is excluded. County data are not literally complete in 2024: legacy geography and missing values limit the paired sample.',
             'Shock comparisons report both independent observed windows and identical paired dates. State means are recomputed on each paired county sample. Fitted variance share is squared within-state correlation (a fitted slope); direct capture is 1 − sum((county deviation − metro deviation)^2)/sum(county deviation^2), which can be negative. Assignment variance ratio is Var(metro deviation)/Var(county deviation), not explained variance. CBSA group variance share is the R² of state-CBSA group means of county shocks (by `cbsa_code`, which is the anchor CBSA for state_part labels); it measures geographic grouping, not AAA-price accuracy.',
             'Correlates use only commute_vehicles log density, log income and E_c. Pairwise finite samples, state-demeaned metro assignment and raw covariates; no election outcomes enter apart from vote counts used for coverage. Singleton states have mechanically zero within-state variation; zero-variance correlations are NA. PADD groups are loaded from config/states.csv and do not replace within-state centering.',
             *sections, '', '## Blockers', '',
             'The open 2026 window, any missing archive dates above, no_cbsa and ambiguous labels, hand-reviewed label geography and unverified AAA sampling boundaries prevent a validated replacement treatment. Conservative matches give a coverage lower bound under this rule, not an estimate of full AAA county reach. Comparison results concern the matched subset of the 19 county-data states; they do not establish nationwide agreement. Rebuild after November 2 and review geography before any treatment-design decision.', '',
             '## Input SHA-256 snapshots', '', *[f'- `{k}`: `{v}`' for k, v in sorted(hashes.items())], '']
    output = root / 'reports/validation/metro_price_feasibility.md'
    output.write_text('\n'.join(lines))
    pl.DataFrame(records, schema_overrides={'value': pl.Float64}).unique().with_columns(pl.col('value').round(9)).sort('section', 'state', 'period', 'metric').write_csv(output.with_suffix('.csv'))
    if log_run:
        runlog.append_run_log(run_id=f'metro_prices_{hashes[str(input_path.relative_to(root))][:12]}',
                              utc_timestamp=rawstore.utc_iso(datetime.now(timezone.utc)), git_sha=runlog.git_sha(), git_dirty=runlog.git_dirty(),
                              config_hash=runlog.config_hash([root / 'config/states.csv', overrides_path, Path(__file__)]), spec_id='metro_prices',
                              data_snapshot_ids=json.dumps({**{f'{k}_head': v for k, v in tips.items()}, **hashes}, sort_keys=True),
                              output_path=str(output.relative_to(root)), notes='Offline provisional metro feasibility; lykmapipo + fork; reviewed CBSA overrides; 2024 D/R votes used only for coverage.')
    print('Coverage:', [r for r in coverage_rows if r['state'] == 'US'])
    print('Matching:', match_rows[0])
    print('Variation:', [r for r in variation_rows if r['state'] == 'US'])
    print('Levels:', [r for r in agreement_rows if r['state'] == 'US'])
    print('Window levels:', [r for r in window_level_rows if r['state'] == 'US'])
    print('Shocks:', [r for r in comparison_rows if r['state'] == 'US'])
    print('Correlates:', [r for r in corr_rows if r['state'] == 'US' and r['sample'] == 'min1'])


def main():
    if sys.argv[1:] == ['feasibility']:
        feasibility()
    elif not sys.argv[1:] or sys.argv[1:] == ['build']:
        build()
    else:
        raise SystemExit('Usage: python -m fb26.etl.metro_prices [build|feasibility]')


if __name__ == '__main__':
    main()
