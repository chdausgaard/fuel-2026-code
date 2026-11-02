"""Offline AAA county-price feasibility; no raw data or existing parser is modified.

Build: python -m fb26.etl.county_prices
Report: python -m fb26.etl.county_prices feasibility
The parquet is restricted data. Reports contain aggregates only, never county prices.
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
import unicodedata
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from fb26.etl import results
from fb26.scrape import medsl
from fb26.scrape.aaagas import _git, read_blobs, open_bundle, latest_record, load_states, parse_price
from fb26.scrape.wayback import _first_commit_per_blob
from fb26.util import paths, rawstore, runlog

VERSION = 'county-prices-0.3.0'
NATHAN = 'nathanqthai_aaa'
AAAGAS = 'aaagas_county'
CUTOFF = date(2026, 3, 27)
WINDOWS = {2024: (date(2024, 10, 15), date(2024, 11, 4)),
           2026: (date(2026, 10, 13), date(2026, 11, 2))}
SCHEMA = {'source': pl.String, 'as_of_date': pl.Date, 'date_rule': pl.String,
          'state_fips': pl.String, 'county_fips': pl.String, 'county_name_raw': pl.String,
          'price_usd': pl.Float64, 'commit_or_file': pl.String,
          'captured_at_utc': pl.Datetime('us', 'UTC'), 'parser_version': pl.String, 'flags': pl.String}
CT_LEGACY = {'fairfield', 'hartford', 'litchfield', 'middlesex', 'new haven', 'new london', 'tolland', 'windham'}


def normalize_name(name: str) -> str:
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode().lower().strip()
    name = re.sub(r'\bsaint\b', 'st', name)
    name = re.sub(r'\s+(county|parish|borough|census area|municipality)$', '', name)
    return re.sub(r'[^a-z0-9]', '', name)


def match_county(state_fips: str, name: str, lookup: dict) -> tuple[str | None, str]:
    key = normalize_name(name)
    if state_fips == '09' and key in {normalize_name(n) for n in CT_LEGACY}:
        return None, 'ct_legacy_county'
    value = lookup.get((state_fips, key))
    return value, '' if value else 'unmatched_county'


def scrape_date(timestamp: datetime) -> date:
    if timestamp.tzinfo is None:
        raise ValueError('scrape timestamp must be timezone-aware')
    return timestamp.astimezone(ZoneInfo('America/New_York')).date()


def county_universe() -> tuple[pl.DataFrame, dict]:
    files = sorted((paths.raw_dir() / 'census_geo').rglob('*.json.gz'))
    candidates = []
    for file in files:
        data = json.loads(rawstore.load_raw(file))
        if data and all(c in data[0] for c in ('NAME', 'state', 'county')):
            candidates.append((file, data))
    if not candidates:
        raise FileNotFoundError('2024 Census county name list is required')
    file, data = candidates[-1]
    rows = [dict(zip(data[0], r)) for r in data[1:]]
    allowed = set(load_states()[1].values())
    rows = [dict(state_fips=r['state'], county_fips=r['state'] + r['county'],
                 county_name=r['NAME'].split(',')[0]) for r in rows if r['state'] in allowed]
    df = pl.DataFrame(rows).sort('county_fips')
    assert set(df.filter(pl.col('state_fips') == '09')['county_fips']) == {f'09{i}' for i in range(110, 191, 10)}
    lookup = {}
    for r in rows:
        key = (r['state_fips'], normalize_name(r['county_name']))
        if key in lookup:
            raise ValueError(f'Ambiguous normalized Census name: {key}')
        lookup[key] = r['county_fips']
    return df, lookup


def price_value(text: str) -> tuple[float | None, str]:
    try:
        value = parse_price(text)
    except (ValueError, TypeError):
        return None, 'invalid_price'
    if value is None or not math.isfinite(value):
        return None, 'missing_price'
    if not 1 <= value <= 8:
        return None, 'out_of_range'
    return value, ''


def row(source, day, rule, state, name, price, captured, ref, lookup, extra=''):
    fips, flag = match_county(state, name, lookup)
    value, invalid = price_value(price)
    return dict(source=source, as_of_date=day, date_rule=rule, state_fips=state,
                county_fips=fips, county_name_raw=name, price_usd=value, commit_or_file=ref,
                captured_at_utc=captured, parser_version=VERSION,
                flags=';'.join(f for f in (extra, flag, invalid) if f))


def parse_nathan(content: bytes, captured: datetime, ref: str, by_abbr: dict, lookup: dict) -> list[dict]:
    if captured.astimezone(timezone.utc).date() > CUTOFF:
        return []
    rows = []
    for rec in csv.reader(io.StringIO(content.decode('utf-8-sig'))):
        if len(rec) < 4 or rec[0].upper() not in by_abbr:
            raise ValueError('Unexpected Nathan county CSV row')
        rows.append(row(NATHAN, scrape_date(captured), 'scrape_US_Eastern_calendar_day',
                        by_abbr[rec[0].upper()], rec[2], rec[3], captured, ref, lookup, 'date_from_scrape_time'))
    return rows


def parse_realcounty(content: bytes, captured: datetime, ref: str, by_abbr: dict, lookup: dict) -> list[dict]:
    rows = []
    by_name = load_states()[0]
    for rec in csv.DictReader(io.StringIO(content.decode('utf-8-sig'))):
        if 'abbreviation' in rec:
            state, name, value, day = by_abbr[rec['abbreviation'].upper()], rec['name'], rec['price'], rec['date']
        elif 'Region' in rec:
            state = by_abbr[rec['State']] if rec['State'] in by_abbr else by_name[rec['State']][1]
            name, value, day = rec['Region'], rec['Price'], rec.get('Date')
        else:
            raise ValueError('Unexpected RealCounty header')
        rows.append(row(AAAGAS, date.fromisoformat(day) if day else None, 'repository_row_date_runner_calendar',
                        state, name, value, captured, ref, lookup,
                        ('date_from_runner' if day else 'missing_row_date') + ';capture_is_first_commit'))
    return rows


def dedupe(frame: pl.DataFrame) -> pl.DataFrame:
    key = ['source', 'as_of_date', 'state_fips', 'county_name_raw']
    return (frame.with_columns(pl.len().over(key).alias('_n'))
            .sort(key + ['captured_at_utc', 'commit_or_file'])
            .unique(subset=key, keep='last', maintain_order=True)
            .with_columns(pl.when(pl.col('_n') > 1).then(pl.col('flags') + ';duplicate_latest_capture')
                          .otherwise(pl.col('flags')).alias('flags')).drop('_n').sort(key))


def ct_weights(towns: pl.DataFrame, regions: pl.DataFrame) -> pl.DataFrame:
    """Planning region x legacy county weights: the share of the region's votes cast in towns of that legacy county.
    `towns`: town_code, county_name (legacy), votes. `regions`: town_code, geo_id (as `results.ct_town_regions`)."""
    joined = towns.join(regions, on='town_code', how='left')
    if joined['geo_id'].null_count() or joined['town_code'].n_unique() != joined.height:
        raise ValueError('Connecticut towns must each map to exactly one planning region')
    legacy = joined['county_name'].map_elements(normalize_name, return_dtype=pl.String)
    if set(legacy) != {normalize_name(n) for n in CT_LEGACY}:
        raise ValueError(f'Unexpected Connecticut legacy counties: {sorted(set(legacy))}')
    return (joined.with_columns(legacy.alias('legacy'))
            .group_by(pl.col('geo_id').alias('county_fips'), 'legacy').agg(pl.col('votes').sum())
            .with_columns((pl.col('votes') / pl.col('votes').sum().over('county_fips')).alias('weight'))
            .select('county_fips', 'legacy', 'weight').sort('county_fips', 'legacy'))


def ct_region_weights() -> pl.DataFrame:
    """Weights from the 2024 presidential vote by town (TOTAL rows, all candidates): the MEDSL 2024 Connecticut file
    gives each town's legacy county, the Census 2022 relationship file its planning region (as in the results build)."""
    cfg = results.config()['connecticut']
    raw, _ = results.read_source('pres_2024_ct', office='US PRESIDENT', state='CT')
    towns = (raw.filter(pl.col('mode') == 'TOTAL')
             .group_by(pl.col('jurisdiction_fips').str.slice(-5).alias('town_code'), 'county_name')
             .agg(pl.col('votes').cast(pl.Int64).sum()))
    regions = results.ct_town_regions(rawstore.load_raw(medsl.raw_record(cfg['region_file_key']).path))
    if towns.height != cfg['expected_towns'] or regions['geo_id'].n_unique() != cfg['expected_regions']:
        raise ValueError('Connecticut town or region count differs from config/geography.yaml')
    return ct_weights(towns, regions)


def add_ct_regions(df: pl.DataFrame, weights: pl.DataFrame) -> pl.DataFrame:
    """Appends one row per source, day and Connecticut planning region: the weighted mean of the legacy-county prices
    of its towns, flagged ct_legacy_crosswalk. AAA reports Connecticut by its 8 legacy counties; the Census and this
    project use the 9 planning regions. A region-day needs a valid price for every legacy county it draws on.
    The legacy rows themselves stay unmatched."""
    key = ['source', 'as_of_date']
    legacy = (df.filter(pl.col('flags').str.contains('ct_legacy_county') & pl.col('price_usd').is_not_null()
                        & pl.col('as_of_date').is_not_null())
              .with_columns(pl.col('county_name_raw').map_elements(normalize_name, return_dtype=pl.String).alias('legacy'))
              .group_by(key + ['legacy'])
              .agg(pl.col('price_usd').mean(), pl.col('date_rule').first(), pl.col('commit_or_file').unique(),
                   pl.col('captured_at_utc').max(), pl.col('flags').str.split(';').list.explode().unique()))
    need = weights.group_by('county_fips').agg(pl.len().alias('_need'))
    derived = (weights.join(legacy, on='legacy')
               .group_by(key + ['county_fips'])
               .agg((pl.col('price_usd') * pl.col('weight')).sum().alias('price_usd'), pl.len().alias('_n'),
                    pl.col('date_rule').first(), pl.col('commit_or_file').list.explode().unique().sort().str.join(';'),
                    pl.col('captured_at_utc').max(), pl.col('flags').list.explode().unique().sort())
               .join(need, on='county_fips').filter(pl.col('_n') == pl.col('_need'))
               .with_columns(pl.lit('09').alias('state_fips'),
                             (pl.lit('CT planning region ') + pl.col('county_fips')).alias('county_name_raw'),
                             pl.lit(VERSION).alias('parser_version'),
                             pl.col('flags').list.set_difference(pl.lit(['ct_legacy_county', ''])).list.sort()
                             .list.concat(pl.lit(['ct_legacy_crosswalk'])).list.join(';').alias('flags'))
               .select(list(SCHEMA)).cast(SCHEMA))
    return pl.concat([df, derived]).sort(['source', 'as_of_date', 'state_fips', 'county_name_raw'])


def history_blobs(repo: Path, prefix: str, rev: str = 'HEAD'):
    log = _git(['log', '--raw', '--no-abbrev', '-r', '--no-renames', '--format=C %H %aI', rev, '--', prefix], git_dir=repo).decode()
    return _first_commit_per_blob(log)


def county_blobs(repo: Path) -> dict[str, tuple[str, str, bool]]:
    """AAAGas county blobs on every branch: blob -> (first introducing commit, its time, whether HEAD's history has it).
    Upstream tests scraper fixes on side branches (dev, 2026-10-21), so HEAD (main) alone misses those files."""
    prefixes = ('RealCounty/Data', 'RealCounty/RealCounty.csv')
    on_head = {blob for prefix in prefixes for blob in history_blobs(repo, prefix)}
    every = {blob: first for prefix in prefixes for blob, first in history_blobs(repo, prefix, '--branches').items()}
    return {blob: (commit, when, blob in on_head) for blob, (commit, when) in every.items()}


def annotate_placeholders(df: pl.DataFrame, states: pl.DataFrame) -> pl.DataFrame:
    keys = ['source', 'as_of_date', 'state_fips']
    # A broad repeated-price screen: at least five counties and 20% of valid counties in a state-day.
    valid = df.filter(pl.col('county_fips').is_not_null() & pl.col('price_usd').is_not_null() & pl.col('as_of_date').is_not_null())
    repeats = (valid.group_by(keys + ['price_usd']).agg(pl.col('county_fips').n_unique().alias('_same'))
               .join(valid.group_by(keys).agg(pl.col('county_fips').n_unique().alias('_total')), on=keys))
    return (df.join(repeats, on=keys + ['price_usd'], how='left').join(states, on=keys, how='left')
            .with_columns((pl.col('flags')
                           + pl.when((pl.col('price_usd') - pl.col('_state_price')).abs() < 1e-9)
                             .then(pl.lit(';equals_state_average')).otherwise(pl.lit(''))
                           + pl.when((pl.col('_same') >= 5) & (pl.col('_same') / pl.col('_total') >= .2))
                             .then(pl.lit(';widely_repeated_price')).otherwise(pl.lit(''))).alias('flags'))
            .select(list(SCHEMA)).sort(['source', 'as_of_date', 'state_fips', 'county_name_raw']))


def build() -> Path:
    _, lookup = county_universe()
    by_name, by_abbr = load_states()
    root = paths.repo_root()
    scratch = root / '.scratch/tmp'
    scratch.mkdir(parents=True, exist_ok=True)
    frames, state_rows = [], []
    record = latest_record('nathanqthai_gas_prices_repo')
    with tempfile.TemporaryDirectory(prefix='county_prices_', dir=scratch) as tmp:
        repo = open_bundle(record, Path(tmp))
        listing = _git(['ls-tree', '-r', '--name-only', 'refs/heads/main', '--', 'prices/states', 'prices/national'], git_dir=repo).decode().splitlines()
        selected = []
        for name in listing:
            try:
                captured = datetime.fromisoformat(Path(name).stem)
            except ValueError:
                continue
            # Existing state archive audit found the scheduled 11:00–13:59 UTC runs best aligned.
            if WINDOWS[2024][0] <= captured.date() <= CUTOFF and 11 <= captured.hour < 14:
                selected.append((name, captured))
        for i in range(0, len(selected), 40):
            batch = selected[i:i + 40]
            blobs = read_blobs(repo, ['refs/heads/main:' + n for n, _ in batch])
            rows = []
            for name, captured in batch:
                content = blobs['refs/heads/main:' + name]
                ref = f'{record.sha256}:{name}'
                if name.startswith('prices/states/'):
                    rows.extend(parse_nathan(content, captured, ref, by_abbr, lookup))
                else:
                    for rec in csv.reader(io.StringIO(content.decode())):
                        value, _ = price_value(rec[2])
                        state_rows.append(dict(source=NATHAN, as_of_date=scrape_date(captured), state_fips=by_abbr[rec[0].upper()], _state_price=value))
            if rows:
                frames.append(pl.DataFrame(rows, schema=SCHEMA))
    repo = paths.interim_dir() / 'aaagas_mirror.git'
    tip = _git(['rev-parse', 'HEAD'], git_dir=repo).decode().strip()
    items = sorted(county_blobs(repo).items())
    for i in range(0, len(items), 40):
        batch = items[i:i + 40]
        blobs = read_blobs(repo, [b for b, _ in batch])
        rows = []
        for blob, (commit, when, on_head) in batch:
            parsed = parse_realcounty(blobs[blob], datetime.fromisoformat(when).astimezone(timezone.utc),
                                      f'{commit}:{blob}', by_abbr, lookup)
            if not on_head:
                for rec in parsed:
                    rec['flags'] += ';not_on_main_branch'
            rows.extend(parsed)
        frames.append(pl.DataFrame(rows, schema=SCHEMA))
    master = read_blobs(repo, [f'{tip}:Prices/MasterGas.csv'])[f'{tip}:Prices/MasterGas.csv']
    for rec in csv.DictReader(io.StringIO(master.decode())):
        value, _ = price_value(rec['Regular'])
        state_rows.append(dict(source=AAAGAS, as_of_date=date.fromisoformat(rec['Date']), state_fips=by_name[rec['State']][1], _state_price=value))
    states = pl.DataFrame(state_rows).group_by('source', 'as_of_date', 'state_fips').agg(pl.col('_state_price').mean())
    df = annotate_placeholders(add_ct_regions(dedupe(pl.concat(frames)), ct_region_weights()), states)
    output = paths.interim_dir() / 'prices_county_daily.parquet'
    df.write_parquet(output)
    print(f'Built {df.height:,} rows; {df["county_fips"].n_unique() - int(df["county_fips"].null_count() > 0):,} matched counties.')
    return output


def placeholder_counties(df: pl.DataFrame) -> pl.DataFrame:
    valid = df.filter(pl.col('county_fips').is_not_null() & pl.col('price_usd').is_not_null() & pl.col('as_of_date').is_not_null())
    # Diagnose source-window separately so a long clean history cannot hide a bad target window.
    sets = []
    for label, subset in [('all', valid)] + [(str(y), valid.filter(pl.col('as_of_date').is_between(*w))) for y, w in WINDOWS.items()]:
        sets.append(subset.group_by('source', 'state_fips', 'county_fips').agg(
            pl.len().alias('days'),
            pl.col('flags').str.contains('equals_state_average').mean().alias('state_equal_share'),
            pl.col('flags').str.contains('widely_repeated_price').mean().alias('repeated_share'))
                    .with_columns(pl.lit(label).alias('period')))
    return pl.concat(sets).with_columns(((pl.col('days') >= 3) & ((pl.col('state_equal_share') > .5) | (pl.col('repeated_share') > .5))).alias('placeholder'))


def county_shocks(df: pl.DataFrame, excluded: set[str] | None = None) -> pl.DataFrame:
    valid = df.filter(pl.col('county_fips').is_not_null() & pl.col('price_usd').is_not_null() & pl.col('as_of_date').is_not_null())
    if excluded:
        valid = valid.filter(~pl.col('county_fips').is_in(sorted(excluded)))
    # One equal-weight source value per county-day, then equal-weight observed days, as in D21.
    daily = valid.group_by('county_fips', 'state_fips', 'as_of_date').agg(pl.col('price_usd').mean())
    means = []
    for year, window in WINDOWS.items():
        means.append(daily.filter(pl.col('as_of_date').is_between(*window)).group_by('county_fips', 'state_fips')
                     .agg(pl.col('price_usd').mean().alias(f'p{year}'), pl.len().alias(f'n{year}')))
    return (means[0].join(means[1], on=['county_fips', 'state_fips'])
            .with_columns((pl.col('p2026') - pl.col('p2024')).alias('shock'))
            .with_columns((pl.col('shock') - pl.col('shock').mean().over('state_fips')).alias('deviation')))


def population_sd(values: pl.Series) -> float | None:
    return values.std(ddof=0) if len(values) else None


def format_value(x):
    if x is None or isinstance(x, float) and not math.isfinite(x):
        return 'NA'
    return f'{x:.6f}' if isinstance(x, float) else str(x)


def md_table(rows: list[dict], columns: list[str]) -> str:
    return '\n'.join(['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |'] +
                     ['| ' + ' | '.join(format_value(r.get(c)) for c in columns) + ' |' for r in rows])


def feasibility(*, log_run: bool = True) -> None:
    root = paths.repo_root()
    input_path = paths.interim_dir() / 'prices_county_daily.parquet'
    df = pl.read_parquet(input_path)
    universe, _ = county_universe()
    states = pl.read_csv(root / 'config/states.csv', schema_overrides={'state_fips': pl.String, 'padd_group': pl.String})
    # Strictly project before collecting: no candidate/outcome columns used beyond D/R coverage votes.
    votes = (pl.scan_parquet(paths.processed_dir() / 'results_county.parquet')
             .filter((pl.col('year') == 2024) & (pl.col('office') == 'PRESIDENT') & pl.col('party').is_in(['D', 'R']))
             .select(pl.col('geo_id').alias('county_fips'), 'votes').collect()
             .group_by('county_fips').agg(pl.col('votes').sum()))
    universe = universe.join(votes, on='county_fips', how='left').with_columns(pl.col('votes').fill_null(0))
    unmapped_votes = votes.join(universe.select('county_fips'), on='county_fips', how='anti')['votes'].sum()
    records = []
    def add(section, source='', year=None, state='US', sample='', **metrics):
        for metric, value in metrics.items():
            records.append(dict(section=section, source=source, year=year, state=state, sample=sample, metric=metric, value=value))
    coverage = []
    valid = df.filter(pl.col('county_fips').is_not_null() & pl.col('price_usd').is_not_null() & pl.col('as_of_date').is_not_null())
    for source in [NATHAN, AAAGAS, 'combined']:
        for year, window in WINDOWS.items():
            sub = valid.filter(pl.col('as_of_date').is_between(*window))
            if source != 'combined':
                sub = sub.filter(pl.col('source') == source)
            counts = sub.group_by('county_fips').agg(pl.col('as_of_date').n_unique().alias('days'))
            covered = universe.join(counts, on='county_fips', how='left').with_columns(pl.col('days').fill_null(0))
            for state in ['US'] + states['state_fips'].to_list():
                u = covered if state == 'US' else covered.filter(pl.col('state_fips') == state)
                denominator = u['votes'].sum() + (unmapped_votes if state == 'US' else votes.filter(pl.col('county_fips').str.starts_with(state)).join(universe.select('county_fips'), on='county_fips', how='anti')['votes'].sum())
                result = dict(source=source, year=year, state=state, all_counties=u.height, max_days=u['days'].max())
                for threshold in [1, 15]:
                    s = u.filter(pl.col('days') >= threshold)
                    result.update({f'n{threshold}': s.height, f'share{threshold}': s.height / u.height,
                                   f'vote_share{threshold}': s['votes'].sum() / denominator if denominator else None})
                coverage.append(result)
                add('coverage', source, year, state, **{k: v for k, v in result.items() if k not in ['source', 'year', 'state']})
    overlap = valid.filter(pl.col('as_of_date').is_between(date(2024, 11, 16), CUTOFF))
    join_keys = ['county_fips', 'state_fips', 'as_of_date']
    left = overlap.filter(pl.col('source') == NATHAN).select(join_keys + ['price_usd'])
    right = overlap.filter(pl.col('source') == AAAGAS).select(join_keys + ['price_usd'])
    agreement = []
    for shift in [-1, 0, 1]:
        joined = left.join(right.with_columns(pl.col('as_of_date') + pl.duration(days=shift)), on=join_keys, suffix='_b')
        joined = joined.with_columns((pl.col('price_usd') - pl.col('price_usd_b')).abs().alias('absdiff'))
        for state in ['US'] + states['state_fips'].to_list():
            s = joined if state == 'US' else joined.filter(pl.col('state_fips') == state)
            d = dict(state=state, shift=shift, pairs=s.height, exact_share=(s['absdiff'] < 1e-9).mean(), mad=s['absdiff'].mean())
            agreement.append(d)
            add('agreement', state=state, sample=f'aaagas_date_shift_{shift}', pairs=d['pairs'], exact_share=d['exact_share'], mad=d['mad'])
    placeholders = placeholder_counties(df)
    bad = set(placeholders.filter(pl.col('placeholder'))['county_fips'])
    placeholder_rows = []
    for state in ['US'] + states['state_fips'].to_list():
        sub = placeholders if state == 'US' else placeholders.filter(pl.col('state_fips') == state)
        result = dict(state=state, suspected=sub.filter(pl.col('placeholder'))['county_fips'].n_unique(),
                      state_equal=sub.filter((pl.col('days') >= 3) & (pl.col('state_equal_share') > .5))['county_fips'].n_unique(),
                      repeated=sub.filter((pl.col('days') >= 3) & (pl.col('repeated_share') > .5))['county_fips'].n_unique())
        placeholder_rows.append(result)
        add('placeholders', state=state, **{k: v for k, v in result.items() if k != 'state'})
    diagnostics, correlations = [], []
    exposure = (pl.read_parquet(paths.processed_dir() / 'exposure_county.parquet')
                .filter(pl.col('g_method') == 'commute_vehicles')
                .select(pl.col('geo_id').alias('county_fips'), 'log_pop_density', 'log_median_income', 'e_c'))
    for label, excluded in [('all', set()), ('exclude_placeholders', bad)]:
        shocks = county_shocks(df, excluded)
        for threshold in [1, 15]:
            sample = f'{label}_min{threshold}'
            s = shocks.filter((pl.col('n2024') >= threshold) & (pl.col('n2026') >= threshold))
            # Recenter after filtering: each reported sample has its own state means.
            s = s.with_columns((pl.col('shock') - pl.col('shock').mean().over('state_fips')).alias('deviation'))
            for state in ['US'] + states['state_fips'].to_list():
                a = s if state == 'US' else s.filter(pl.col('state_fips') == state)
                within = population_sd(a['deviation']) if a.height else None
                total = population_sd(a['shock']) if a.height else None
                means = a.group_by('state_fips').agg(pl.col('shock').mean()).join(states.select('state_fips', 'padd_group'), on='state_fips')
                between = population_sd(means.with_columns((pl.col('shock') - pl.col('shock').mean().over('padd_group')).alias('d'))['d']) if state == 'US' and means.height else None
                r = dict(sample=sample, state=state, counties=a.height, within_sd=within,
                         total_sd=total, within_variance_share=within ** 2 / total ** 2 if total else None,
                         state_within_padd_sd=between)
                diagnostics.append(r)
                add('shock_variation', state=state, sample=sample, **{k: v for k, v in r.items() if k not in ['sample', 'state']})
            joined = s.join(exposure, on='county_fips')
            for variable in ['log_pop_density', 'log_median_income', 'e_c']:
                pair = joined.filter(pl.col(variable).is_finite() & pl.col('deviation').is_finite())
                corr = pair.select(pl.corr('deviation', variable)).item() if pair.height >= 3 else None
                correlations.append(dict(sample=sample, variable=variable, n=pair.height, correlation=corr))
                add('correlation', sample=sample, source=variable, n=pair.height, correlation=corr)
    unmatched = (df.filter(pl.col('county_fips').is_null()).group_by('source', 'state_fips', 'county_name_raw')
                 .agg(pl.len().alias('rows'), pl.col('as_of_date').min().alias('first_date'), pl.col('as_of_date').max().alias('last_date'),
                      pl.col('flags').str.contains('ct_legacy_county').any().alias('ct_legacy')).sort('source', 'state_fips', 'county_name_raw'))
    repo = paths.interim_dir() / 'aaagas_mirror.git'
    tip = _git(['rev-parse', 'HEAD'], git_dir=repo).decode().strip()
    metro_spec = f'{tip}:CountyPrices/MasterMergedGas.csv'
    metro = list(csv.DictReader(io.StringIO(read_blobs(repo, [metro_spec])[metro_spec].decode())))
    metro_dates = sorted({r['Date'] for r in metro if r['Date'].startswith('2026')})
    city_paths = _git(['ls-tree', '-r', '--name-only', tip, '--', 'City Scrape/Data'], git_dir=repo).decode().splitlines()
    city_2026_files = sum('2026-' in name for name in city_paths)
    source_summary = valid.group_by('source').agg(pl.col('as_of_date').min().alias('first'), pl.col('as_of_date').max().alias('last'), pl.col('as_of_date').n_unique().alias('dates'), pl.col('county_fips').n_unique().alias('counties')).sort('source').to_dicts()
    snapshot = hashlib.sha256(input_path.read_bytes()).hexdigest()
    input_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [paths.processed_dir() / 'results_county.parquet',
                              paths.processed_dir() / 'exposure_county.parquet',
                              *sorted((paths.raw_dir() / 'census_geo').rglob('*.json.gz'))]}
    current = df.filter(pl.col('as_of_date').is_between(*WINDOWS[2026]))
    current_dates = (current.group_by('as_of_date').agg(pl.len().alias('rows'),
                     pl.col('state_fips').n_unique().alias('states_present'),
                     pl.col('price_usd').null_count().alias('missing_or_invalid_prices'))
                     .sort('as_of_date').to_dicts())
    for r in current_dates:
        add('daily_2026_availability', sample=str(r['as_of_date']),
            **{k: v for k, v in r.items() if k != 'as_of_date'})
    lines = ['# AAA county-price feasibility — provisional 2026', '',
             'Exploratory treatment-data feasibility only; this does not change the preregistered state-level treatment. No county prices or county shocks are redistributed here.', '',
             '## Inputs and reproducibility', '',
             f'Parser: `{VERSION}`. Parquet SHA-256: `{snapshot}`. AAAGas mirror HEAD: `{tip}`.',
             f'Nathan bundle SHA-256: `{latest_record("nathanqthai_gas_prices_repo").sha256}`. Census names use 2024 ACS geography, 50 states plus DC; Connecticut planning regions are the denominator.',
             'Build and report are offline. Commands: `python -m fb26.etl.county_prices` then `python -m fb26.etl.county_prices feasibility`.', '',
             md_table(source_summary, ['source', 'first', 'last', 'dates', 'counties']), '',
             'Input file hashes (SHA-256):', '', *[f'- `{name}`: `{digest}`' for name, digest in sorted(input_hashes.items())], '',
             '## Date, matching and aggregation rules', '',
             'Nathan: retain scheduled scrapes at 11:00–13:59 UTC, assign America/New_York calendar date, latest capture per county-day; retain UTC scrape dates only through 2026-03-27 (AAA cutoff). Earlier runs are excluded because AAA may not yet have turned over. This follows the existing state archive parser’s timing restriction; it is not proof of the county page’s printed date. Data before the 2024 window are outside this extract.',
             'AAAGas: read all unique historical blobs under RealCounty/Data and RealCounty/RealCounty.csv on every branch, including deleted early snapshots; blobs absent from main’s history (upstream test runs on side branches) are flagged not_on_main_branch. Use the row’s runner calendar date, not the commit date; captured_at_utc is the first introducing commit’s author time, flagged as a proxy. Latest capture wins duplicate source/county/date records, with lexical reference as tie-break. The county series starts later than the state series. An undated early snapshot is retained with missing_row_date and excluded from diagnostics; commit time is not substituted for a missing row date.',
             'Name matching is within state, removes legal county-equivalent suffixes, punctuation and accents, and normalizes Saint/St. City distinctions remain. No fuzzy matches. Legacy Connecticut counties stay unmatched; each planning region gets a derived row (flag ct_legacy_crosswalk), the mean of its towns\' legacy-county prices weighted by town 2024 presidential votes, on days with a price for every legacy county it draws on. Invalid/out-of-range regular prices stay null and flagged.',
             'Daily values are averaged across available sources after source deduplication. Window means use observed days only; neither missing days nor counties are imputed. All statistics weight counties equally except explicitly vote-weighted coverage. Variance uses population moments (ddof=0); within-state variance is mean squared county residual. State-within-PADD SD weights available state means equally.', '',
             '## 1. Coverage', '',
             'Windows: 2024-10-15–2024-11-04 and 2026-10-13–2026-11-02. The latter is open; 15/21 is retained as the full-window criterion and is not relaxed. Shares are fractions, not percentages.',
             f'All-county denominator: {universe.height}. D/R presidential votes attached to 2024-vintage counties: {universe["votes"].sum():,}; votes with no matching county in that universe: {unmapped_votes:,}, retained in vote-share denominators. Missing vote records contribute no invented votes.', '',
             md_table([r for r in coverage if r['state'] == 'US'], ['source', 'year', 'all_counties', 'max_days', 'n1', 'share1', 'vote_share1', 'n15', 'share15', 'vote_share15']), '',
             'Observed 2026-window dates and raw parsed availability (missing dates are not filled):', '',
             md_table(current_dates, ['as_of_date', 'rows', 'states_present', 'missing_or_invalid_prices']), '',
             f'The 2026 archive covers {min((r["states_present"] for r in current_dates), default=0)}–{max((r["states_present"] for r in current_dates), default=0)} states on observed dates; the per-state table distinguishes geographic gaps from the still-open window.', '',
             '### Per-state coverage (FIPS)', '', md_table([r for r in coverage if r['state'] != 'US'], ['source', 'year', 'state', 'all_counties', 'max_days', 'n1', 'share1', 'vote_share1', 'n15', 'share15', 'vote_share15']), '',
             '## 2. Cross-source agreement', '',
             'Requested interval: 2024-11-16–2026-03-27; pairs require both sources on the county-day. Exact means equal reported precision (numerical tolerance 1e-9). MAD is dollars per gallon. Date-shift checks move the AAAGas date by ±1; they are sensitivity diagnostics, not a fitted correction.', '',
             md_table([r for r in agreement if r['state'] == 'US'], ['shift', 'pairs', 'exact_share', 'mad']), '',
             'Worst states by same-day MAD (all states in CSV):', '',
             md_table(sorted([r for r in agreement if r['state'] != 'US' and r['shift'] == 0 and r['pairs']], key=lambda r: -r['mad'])[:10], ['state', 'pairs', 'exact_share', 'mad']), '',
             '## 3. Suspected placeholders', '',
             'A county is screened when it equals the same-source state average on >50% of valid days, OR shares its exact price with at least five counties and at least 20% of observed counties in its state on >50% of days. Require at least three days. Evaluate separately by source over the full extract and each election window; flag if any qualifies. Equality alone does not establish fabrication: common prices can be real. This deliberately conservative diagnostic can overexclude small states. Exclude the flagged county from both windows for the sensitivity results.', '',
             md_table(placeholder_rows, ['state', 'suspected', 'state_equal', 'repeated']), '',
             '## 4. County-shock variation', '',
             'g_c = observed 2026-window mean minus observed 2024-window mean. No individual g_c is published. Compare the pooled within-state SD below with the provisional state-level within-PADD benchmark $0.124. The state-within-PADD statistic here is based on means of covered county shocks, not the official state price series. A singleton state has zero residual mechanically; empty samples are NA.', '',
             md_table([r for r in diagnostics if r['state'] == 'US'], ['sample', 'counties', 'within_sd', 'total_sd', 'within_variance_share', 'state_within_padd_sd']), '',
             '### Per-state within-state SD', '',
             md_table([r for r in diagnostics if r['state'] != 'US' and r['sample'].endswith('min1')], ['sample', 'state', 'counties', 'within_sd']), '',
             '## 5. Treatment-only correlates', '',
             'Pearson correlations of the county shock minus its sample-specific state mean with existing log population density, log median household income, and e_c (commute_vehicles only). Pairwise finite observations; no outcomes enter these calculations. Covariates themselves are not demeaned. Descriptive correlations do not establish exogeneity.', '',
             md_table(correlations, ['sample', 'variable', 'n', 'correlation']), '',
             '## 6. Metro / proximity limitation', '',
             f'AAAGas CountyPrices/MasterMergedGas.csv is actually a city/metro series; it contains {len(metro_dates)} distinct 2026 dates' + (f' ({metro_dates[0]}–{metro_dates[-1]}).' if metro_dates else '.'),
             f'The separate City Scrape/Data directory has {city_2026_files} files dated 2026 at this mirror snapshot.',
             'A verified county-to-metro membership or adjacency/coordinate crosswalk is not supplied in the allowed inputs. Metro labels can span counties and states; joining county names to city names would be unjustified. The shared-metro / nearby-county fraction and county-versus-metro shock correlation are therefore not estimated. Metro g also needs a compatible 2024-window baseline. This remains a feasibility limitation.', '',
             '## Unmatched names (every source/name; no prices)', '',
             md_table(unmatched.to_dicts(), ['source', 'state_fips', 'county_name_raw', 'rows', 'first_date', 'last_date', 'ct_legacy']), '',
             '## Assessment', '',
             'County shocks are computable for the covered sample, but remain provisional and depend on different archives across the two windows. The 15-day final criterion cannot yet be met in 2026. Missing counties, legacy Connecticut geography, uncertainty in the repeated-price screen, unverified printed-date timing and unmeasured spatial dependence block treating this as a validated replacement for the preregistered state shock. Rebuild after the window closes, verify timing and screen validity, then assess coverage and spatial dependence before making any design decision.', '']
    out = root / 'reports/validation/county_price_feasibility.md'
    out.write_text('\n'.join(lines))
    pl.DataFrame(records, schema_overrides={'year': pl.Int32, 'value': pl.Float64}).with_columns(pl.col('value').round(9)).sort('section', 'source', 'year', 'state', 'sample', 'metric').write_csv(out.with_suffix('.csv'))
    if log_run:
        runlog.append_run_log(run_id=f'county_prices_{snapshot[:12]}', utc_timestamp=rawstore.utc_iso(datetime.now(timezone.utc)),
                              git_sha=runlog.git_sha(), git_dirty=runlog.git_dirty(),
                              config_hash=runlog.config_hash([root / 'config/states.csv', Path(__file__)]), spec_id='county_prices',
                              data_snapshot_ids=json.dumps({'county_parquet': snapshot, 'aaagas_head': tip,
                                                            'nathan_bundle': latest_record('nathanqthai_gas_prices_repo').sha256,
                                                            **input_hashes}, sort_keys=True),
                              output_path=str(out.relative_to(root)), notes='Offline provisional county feasibility; no election outcomes except 2024 D/R coverage counts.')
    print(md_table([r for r in coverage if r['state'] == 'US'], ['source', 'year', 'n1', 'n15', 'vote_share1']))
    print('Agreement:', [r for r in agreement if r['state'] == 'US' and r['shift'] == 0])
    print('Placeholders:', len(bad))
    print('Variation:', [r for r in diagnostics if r['state'] == 'US' and r['sample'].endswith('min1')])
    print('Correlates:', [r for r in correlations if r['sample'] == 'exclude_placeholders_min1'])


def main():
    if sys.argv[1:] == ['feasibility']:
        feasibility()
    elif not sys.argv[1:] or sys.argv[1:] == ['build']:
        build()
    else:
        raise SystemExit('Usage: python -m fb26.etl.county_prices [build|feasibility]')


if __name__ == '__main__':
    main()
