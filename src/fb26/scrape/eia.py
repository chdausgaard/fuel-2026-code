"""EIA weekly retail gasoline/diesel, SHOPP and spot prices, and PADD membership (WS1c; sources S3, S4, S24).

Keyless route: the EIA bulk file PET.zip (www.eia.gov/opendata/bulk). The API v2 path is
written for when EIA_API_KEY exists but is not used by default. Fetching goes through
PoliteClient only; parsing reads from the raw store and records PARSER_VERSION.

    python -m fb26.scrape.eia            # fetch (cached 6 h) + parse + coverage + PADD table
    python -m fb26.scrape.eia fetch|parse|shopp
"""
from __future__ import annotations

from datetime import date, datetime
import io
import json
import os
from pathlib import Path
import re
import sys
import zipfile

import polars as pl
import yaml

from fb26.util import paths, rawstore
from fb26.util.http import FetchError, PoliteClient

PARSER_VERSION = 'eia-0.1.0'
SOURCE = 'eia'
BULK_URL = 'https://www.eia.gov/opendata/bulk/PET.zip'
MANIFEST_URL = 'https://www.eia.gov/opendata/bulk/manifest.txt'
DOC_URLS = {  # (url, ext) of documentation and cross-check pages kept in the raw store
    'gasdiesel_page': ('https://www.eia.gov/petroleum/gasdiesel/', 'html'),
    'shopp_page': ('https://www.eia.gov/petroleum/heatingoilpropane/', 'html'),
    'shopp_schedule': ('https://www.eia.gov/petroleum/heatingoilpropane/schedule.php', 'html'),
    'gasdiesel_tbldef': ('https://www.eia.gov/dnav/pet/TblDefs/pet_pri_gnd_tbldef2.asp', 'html'),
    'shopp_tbldef': ('https://www.eia.gov/dnav/pet/TblDefs/pet_pri_wfr_tbldef2.asp', 'html'),
    'copyright': ('https://www.eia.gov/about/copyrights_reuse.php', 'html'),
}
HISTORY_XLS = ('https://www.eia.gov/petroleum/gasdiesel/xls/pswrgvwall.xls',
               'https://www.eia.gov/petroleum/gasdiesel/xls/psw18vwall.xls')
API_ROOT = 'https://api.eia.gov/v2'
API_ROUTES = {'gnd': '/petroleum/pri/gnd/data/', 'wfr': '/petroleum/pri/wfr/data/',
              'spt': '/petroleum/pri/spt/data/'}
CACHE_TTL_S = 6 * 3600

# bulk series prefix -> product label and brief §8 range ($/gal)
PRODUCTS = {'EMM_EPMR_PTE': ('regular', 1.00, 8.00),
            'EMD_EPD2D_PTE': ('diesel', 1.50, 9.00),
            'EMD_EPD2DXL0_PTE': ('diesel_ulsd', 1.50, 9.00)}
SERIES_RE = re.compile(r'^PET\.(?P<prefix>[A-Z0-9]+_[A-Z0-9]+_[A-Z]+)_(?P<area>[A-Z0-9]+)_DPG\.W$')
NAME_RE = re.compile(r' (Regular All Formulations Retail Gasoline Prices|No 2 Diesel(?: Ultra Low Sulfur \(0-15 ppm\))?'
                     r' Retail Prices), Weekly$')
PADD_CODES = {'R10', 'R20', 'R30', 'R40', 'R50', 'R5XCA'}
SUBPADD_CODES = {'R1X', 'R1Y', 'R1Z'}
WEEKLY_COLUMNS = ['series_id', 'area_code', 'area_name', 'product', 'period', 'value', 'units',
                  'fetched_at_utc', 'raw_file', 'raw_sha256', 'parser_version', 'flag', 'geography']

# Memo §5.3 assignment groups (compared with EIA's own definition in padd_membership()).
MEMO_PADD = {
    '1A': 'CT ME MA NH RI VT', '1B': 'DE DC MD NJ NY PA', '1C': 'FL GA NC SC VA WV',
    '2': 'IL IN IA KS KY MI MN MO NE ND OH OK SD TN WI', '3': 'AL AR LA MS NM TX',
    '4': 'CO ID MT UT WY', '5': 'AK AZ CA HI NV OR WA'}
PADD_BULK_CODE = {'1A': 'R1X', '1B': 'R1Y', '1C': 'R1Z', '2': 'R20', '3': 'R30', '4': 'R40', '5': 'R50'}
PADD_DEFINITION_URL = DOC_URLS['gasdiesel_tbldef'][0]


def area_type(code: str) -> str:
    if code == 'NUS':
        return 'national'
    if code in PADD_CODES:
        return 'padd'
    if code in SUBPADD_CODES:
        return 'sub_padd'
    if re.fullmatch(r'S[A-Z]{2}', code):
        return 'state'
    if code.startswith('Y'):
        return 'city'
    raise ValueError(f'Unclassified EIA area code {code}')


def latest_raw(source: str, url: str) -> rawstore.RawRecord:
    """Newest raw-store record for `url` (matched on the stored, redacted URL)."""
    records = [rawstore.record_from_meta(m) for m in (paths.raw_dir() / source).glob('**/*.meta.json')]
    records = [r for r in records if r.url == url]
    if not records:
        raise FileNotFoundError(f'No raw record for {url}; run the fetch step first')
    return max(records, key=lambda r: r.fetched_at_utc)


def fetch(client: PoliteClient | None = None) -> dict:
    """Fetch bulk file, manifest, documentation pages and the two history spreadsheets."""
    client = client or PoliteClient()
    out = {'manifest': client.get(MANIFEST_URL, source=SOURCE, ext='txt', cache_ttl_s=CACHE_TTL_S),
           'bulk': client.get(BULK_URL, source=SOURCE, ext='zip', cache_ttl_s=CACHE_TTL_S)}
    for key, (url, ext) in DOC_URLS.items():
        out[key] = client.get(url, source=SOURCE, ext=ext, cache_ttl_s=CACHE_TTL_S)
    for url in HISTORY_XLS:  # availability check only; the parser does not read .xls (no reader dependency)
        try:
            out[url.rsplit('/', 1)[-1]] = client.get(url, source=SOURCE, ext='xls', cache_ttl_s=CACHE_TTL_S)
        except FetchError as exc:
            print(f'{url}: {exc}', file=sys.stderr)
    return out


def fetch_api_v2(client: PoliteClient, route: str, facets: dict[str, list[str]],
                 api_key: str | None = None, page_size: int = 5000) -> list[dict]:
    """Page through an API v2 route; needs EIA_API_KEY. Returns the raw `response.data` rows.

    Caveat (audit S03): PoliteClient requests https://api.eia.gov/robots.txt first, and that URL
    answers 403 API_KEY_MISSING, so this path raises FetchError until util.http handles it.
    """
    key = api_key or os.getenv('EIA_API_KEY')
    if not key:
        raise RuntimeError('EIA_API_KEY is not set')
    rows, offset = [], 0
    while True:
        params = {'api_key': key, 'frequency': 'weekly', 'data[0]': 'value', 'offset': offset,
                  'length': page_size, 'sort[0][column]': 'period', 'sort[0][direction]': 'asc'}
        for name, values in facets.items():
            for i, value in enumerate(values):
                params[f'facets[{name}][{i}]'] = value
        result = client.get(API_ROOT + API_ROUTES[route], source=SOURCE, params=params, ext='json')
        response = json.loads(result.content)['response']
        rows.extend(response['data'])
        offset += page_size
        if offset >= int(response['total']):
            return rows


def parse_api_v2(payload_rows: list[dict], *, fetched_at_utc: str, raw_file: str, raw_sha256: str) -> pl.DataFrame:
    """API v2 `gnd` rows (series, duoarea, area-name, product, period, value, units) to the weekly schema."""
    records = []
    for row in payload_rows:
        match = SERIES_RE.match('PET.' + row['series'] + '.W')
        if not match or match['prefix'] not in PRODUCTS:
            continue
        records.append(_record(row['series'], match['area'], row.get('area-name') or match['area'],
                               match['prefix'], datetime.strptime(row['period'], '%Y-%m-%d').date(),
                               None if row['value'] in (None, '') else float(row['value']),
                               row.get('units') or 'Dollars per Gallon', None,
                               fetched_at_utc, raw_file, raw_sha256))
    return _frame(records)


def _record(series_id, area_code, area_name, prefix, period, value, units, geography,
            fetched_at_utc, raw_file, raw_sha256):
    product, lo, hi = PRODUCTS[prefix]
    flag = 'missing' if value is None else ('out_of_range' if not lo <= value <= hi else '')
    return dict(series_id=series_id, area_code=area_code, area_name=area_name, product=product,
                period=period, value=value, units=units, fetched_at_utc=fetched_at_utc,
                raw_file=raw_file, raw_sha256=raw_sha256, parser_version=PARSER_VERSION,
                flag=flag, geography=geography)


def _frame(records: list[dict]) -> pl.DataFrame:
    schema = {'series_id': pl.Utf8, 'area_code': pl.Utf8, 'area_name': pl.Utf8, 'product': pl.Utf8,
              'period': pl.Date, 'value': pl.Float64, 'units': pl.Utf8, 'fetched_at_utc': pl.Utf8,
              'raw_file': pl.Utf8, 'raw_sha256': pl.Utf8, 'parser_version': pl.Utf8, 'flag': pl.Utf8,
              'geography': pl.Utf8}
    return pl.DataFrame(records, schema=schema).sort(['product', 'area_code', 'period'])


def parse_pet_lines(lines, *, fetched_at_utc: str, raw_file: str, raw_sha256: str) -> pl.DataFrame:
    """Weekly retail regular gasoline / diesel rows from the lines of PET.txt (one JSON object per line)."""
    records = []
    for line in lines:
        if not line.strip():
            continue
        series = json.loads(line)
        match = SERIES_RE.match(series.get('series_id', ''))
        if not match or match['prefix'] not in PRODUCTS or series.get('f') != 'W':
            continue
        code = match['area']
        name = NAME_RE.sub('', series['name'])
        for period, value in series['data']:
            records.append(_record(series['series_id'], code, name, match['prefix'],
                                   datetime.strptime(period, '%Y%m%d').date(),
                                   None if value in (None, '') else float(value),
                                   series.get('units', ''), series.get('geography'),
                                   fetched_at_utc, raw_file, raw_sha256))
    return _frame(records)


def bulk_lines(content: bytes):
    """Yield the lines of PET.txt inside the PET.zip bytes."""
    with zipfile.ZipFile(io.BytesIO(content)) as archive, archive.open('PET.txt') as handle:
        yield from handle


def load_windows() -> dict[int, tuple[date, date]]:
    cfg = yaml.safe_load((paths.config_dir() / 'windows.yaml').read_text())['elections']
    return {int(y): (date.fromisoformat(v['start']), date.fromisoformat(v['end'])) for y, v in cfg.items()}


def coverage_table(df: pl.DataFrame, windows: dict[int, tuple[date, date]]) -> pl.DataFrame:
    """One row per series: first/last period, n_obs, and non-missing obs inside each election window."""
    keys = ['series_id', 'area_code', 'area_name', 'product', 'geography']
    valid = df.filter(pl.col('value').is_not_null())
    aggs = [pl.col('period').min().alias('first_period'), pl.col('period').max().alias('last_period'),
            pl.len().alias('n_obs')]
    aggs += [pl.col('period').is_between(lo, hi).sum().cast(pl.Int64).alias(f'n_w{year}')
             for year, (lo, hi) in sorted(windows.items())]
    out = valid.group_by(keys).agg(aggs)
    out = out.with_columns(pl.col('area_code').map_elements(area_type, return_dtype=pl.Utf8).alias('area_type'))
    order = {'national': 0, 'padd': 1, 'sub_padd': 2, 'state': 3, 'city': 4}
    out = out.with_columns(pl.col('area_type').replace_strict(order, return_dtype=pl.Int8).alias('_o'))
    cols = ['area_type', 'area_name', 'area_code', 'product', 'series_id', 'first_period', 'last_period',
            'n_obs'] + [f'n_w{y}' for y in sorted(windows)] + ['geography']
    return out.sort(['product', '_o', 'area_name']).select(cols)


def padd_membership(lines) -> pl.DataFrame:
    """State membership of each PADD/sub-district from the `geography` field of the EIA regular-gasoline
    series, with a flag for agreement with memo §5.3."""
    geo = {}
    for line in lines:
        if b'EMM_EPMR_PTE_R' not in line:
            continue
        series = json.loads(line)
        match = SERIES_RE.match(series.get('series_id', ''))
        if match and match['prefix'] == 'EMM_EPMR_PTE' and match['area'] in PADD_BULK_CODE.values():
            geo[match['area']] = sorted(g.removeprefix('USA-') for g in series['geography'].split('+'))
    rows = []
    for group, memo_states in MEMO_PADD.items():
        eia_states = set(geo[PADD_BULK_CODE[group]])
        for state in sorted(eia_states | set(memo_states.split())):
            rows.append(dict(padd_group=group, padd=group[0], state=state, eia_series_code=PADD_BULK_CODE[group],
                             in_eia=state in eia_states, in_memo_5_3=state in memo_states.split(),
                             definition_url=PADD_DEFINITION_URL))
    return pl.DataFrame(rows).sort(['padd_group', 'state'])


def shopp_summary(lines, season_start: str = '20250901') -> pl.DataFrame:
    """Weekly residential heating-oil series: area, first/last period, obs since `season_start`."""
    rows = []
    for line in lines:
        if b'W_EPD2F_PRS_' not in line:
            continue
        series = json.loads(line)
        match = SERIES_RE.match(series.get('series_id', ''))
        if not match or match['prefix'] != 'W_EPD2F_PRS':
            continue
        periods = [p for p, v in series['data'] if v is not None]
        if not periods:
            continue
        rows.append(dict(series_id=series['series_id'], area_code=match['area'], area_name=series['name'],
                         first_period=min(periods), last_period=max(periods),
                         n_since_season_start=sum(p >= season_start for p in periods)))
    return pl.DataFrame(rows).sort('area_code')


def parse(out_dir: Path | None = None, coverage_path: Path | None = None) -> pl.DataFrame:
    """Read the newest bulk file from the raw store; write weekly parquet, coverage CSV and PADD table."""
    record = latest_raw(SOURCE, BULK_URL)
    content = rawstore.load_raw(record.path)
    stamp = rawstore.utc_iso(record.fetched_at_utc)
    df = parse_pet_lines(bulk_lines(content), fetched_at_utc=stamp, raw_file=str(record.path),
                         raw_sha256=record.sha256)
    out_dir = out_dir or paths.interim_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_dir / 'prices_eia_weekly.parquet')
    coverage_table(df, load_windows()).write_csv(
        coverage_path or paths.repo_root() / 'reports/audit/S03_state_city_coverage.csv')
    padd_membership(bulk_lines(content)).write_csv(out_dir / 'padd_membership.csv')
    return df


def main(argv: list[str]) -> int:
    step = argv[1] if len(argv) > 1 else 'all'
    if step in ('fetch', 'all'):
        for key, result in fetch().items():
            print(f'{key}: {result.record.sha256[:12]} {result.record.path.name}')
    if step in ('parse', 'all'):
        df = parse()
        print(f'parsed {df.height} rows, {df["series_id"].n_unique()} series; '
              f'flags: {df.group_by("flag").len().sort("flag").rows()}')
    if step == 'shopp':
        record = latest_raw(SOURCE, BULK_URL)
        print(shopp_summary(bulk_lines(rawstore.load_raw(record.path))))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
