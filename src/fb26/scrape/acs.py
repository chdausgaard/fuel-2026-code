"""ACS 5-year county tables for exposure and the C2 controls (WS3; source S5) and the Census Gazetteer (land area).

B08136 (aggregate travel time by means) is in the 2024 release but the API returns null for about 57% of
counties (also in the 2022 and 2023 vintages); B08134 (means by travel-time bin) is complete. See the WS3 return.

Counties for all states and DC come from `for=county:*&in=state:*`; in the 2022+ vintages the Census API
returns Connecticut as its 9 planning regions (09110-09190), so no town aggregation is needed (S05, S19). Puerto
Rico (state 72) is returned by the API and dropped at parse time with a count. Raw responses keep the API
sentinel codes (negative values); parsing turns them into nulls and counts them.

    python -m fb26.scrape.acs              # fetch + parse
    python -m fb26.scrape.acs fetch|parse
"""
from __future__ import annotations

import io
import json
import os
import sys
import zipfile

from dotenv import load_dotenv
import polars as pl
import requests

from fb26.scrape.eia import latest_raw
from fb26.util import paths, rawstore
from fb26.util.http import PoliteClient

PARSER_VERSION = 'acs-0.1.0'
SOURCE = 'acs'
VINTAGE = 2024
API_ROOT = 'https://api.census.gov/data/{vintage}/acs/acs5'
GAZETTEER_URL = ('https://www2.census.gov/geo/docs/maps-data/data/gazetteer/'
                 '{vintage}_Gazetteer/{vintage}_Gaz_counties_national.zip')
CACHE_TTL_S = 24 * 3600

# Estimate variables by request (memo §5.2, §6.2; brief Appendix B). The API allows 50 variables per call, so
# the travel-time bins (B08134) are a second request, joined on geo_id.
REQUESTS = {
    'core': {
        'B08301': [1, 2, 3, 4, 5, 6, 7, 8, 9, 21],  # means of transportation to work; 21 = worked from home
        'B08136': [1, 3, 4],  # aggregate travel time to work (minutes): total, drove alone, carpooled
        'B25046': [1],  # aggregate number of vehicles available
        'B11001': [1],  # households
        'B19013': [1],  # median household income
        'B15003': [1, 22, 23, 24, 25],  # attainment 25+: total, bachelor's, master's, professional, doctorate
        'B03002': [1, 3, 4, 12],  # total, not Hispanic white alone, not Hispanic Black alone, Hispanic or Latino
        'B01002': [1],  # median age
        'B01003': [1],  # total population
    },
    'bins': {
        # B08134 by travel-time bin (<10, 10-14, 15-19, 20-24, 25-29, 30-34, 35-44, 45-59, 60+): drove alone
        # 022-030 and carpooled 032-040. Complete for every county, unlike B08136 (see module note).
        'B08134': list(range(22, 31)) + list(range(32, 41)),
    },
}
# Margins of error, only where a reliability measure is built from them.
MARGINS = {'core': {'B19013': [1], 'B25046': [1], 'B11001': [1]}, 'bins': {}}
BIN_COLUMNS = {'alone': [f'B08134_{i:03d}E' for i in range(22, 31)],
               'carpool': [f'B08134_{i:03d}E' for i in range(32, 41)]}


def variable_names(request: str = 'core') -> list[str]:
    est = [f'{t}_{i:03d}E' for t, ids in REQUESTS[request].items() for i in ids]
    moe = [f'{t}_{i:03d}M' for t, ids in MARGINS[request].items() for i in ids]
    return est + moe


def api_key() -> str:
    load_dotenv(paths.repo_root() / '.env')
    key = os.getenv('CENSUS_API_KEY', '').strip()
    if not key:
        raise ValueError('CENSUS_API_KEY is required (set in .env)')
    return key


def county_url(vintage: int = VINTAGE) -> str:
    return API_ROOT.format(vintage=vintage)


def county_params(key: str, request: str = 'core') -> dict:
    names = variable_names(request)
    if len(names) + 1 > 50:
        raise ValueError('Too many variables for one Census API request')
    return {'get': ','.join(['NAME'] + names), 'for': 'county:*', 'in': 'state:*', 'key': key}


def fetch(client: PoliteClient | None = None, vintage: int = VINTAGE) -> dict:
    """County-level pulls (one request per entry of REQUESTS) and the Gazetteer zip."""
    client = client or PoliteClient()
    key = api_key()
    out = {r: client.get(county_url(vintage), source=SOURCE, params=county_params(key, r), ext='json',
                         cache_ttl_s=CACHE_TTL_S) for r in REQUESTS}
    out['gazetteer'] = client.get(GAZETTEER_URL.format(vintage=vintage), source=SOURCE, ext='zip',
                                  cache_ttl_s=CACHE_TTL_S)
    return out


def latest_county_record(vintage: int, request: str) -> rawstore.RawRecord:
    """Newest raw record of exactly this request (matched on the key-redacted URL, so a changed variable list
    never silently reuses an older pull)."""
    prepared = requests.Request('GET', county_url(vintage), params=county_params('REDACTED', request)).prepare().url
    return latest_raw(SOURCE, rawstore.redact_url(prepared))


def parse_county(content: bytes) -> tuple[pl.DataFrame, dict]:
    """API JSON (header row + rows) to one row per county with numeric columns; negative sentinels become null.

    Returns the table and, per value column, how many cells held a negative Census annotation code."""
    rows = json.loads(content)
    header, body = rows[0], rows[1:]
    df = pl.DataFrame(body, schema=header, orient='row', infer_schema_length=None)
    value_cols = [c for c in header if c not in ('NAME', 'state', 'county')]
    # Census annotation codes (-666666666 etc.) are negative; the median-income margin of error also
    # takes -555555555 / -333333333 for controlled or top-coded estimates. All become null.
    df = df.with_columns(
        [pl.col(c).cast(pl.Float64, strict=False).alias(c) for c in value_cols])
    sentinel_counts = {c: int(df.select(pl.col(c) < 0).to_series().sum()) for c in value_cols}
    df = df.with_columns([pl.when(pl.col(c) < 0).then(None).otherwise(pl.col(c)).alias(c) for c in value_cols])
    df = df.with_columns((pl.col('state') + pl.col('county')).alias('geo_id'))
    df = df.rename({'NAME': 'name', 'state': 'state_fips'}).drop('county')
    return df.sort('geo_id'), sentinel_counts


def parse_gazetteer(content: bytes) -> pl.DataFrame:
    """County land area (square miles) from the Gazetteer zip (tab-delimited, latin-1)."""
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        names = [n for n in zf.namelist() if n.endswith('.txt')]
        if len(names) != 1:
            raise ValueError(f'Expected one .txt in the Gazetteer zip, found {names}')
        text = zf.read(names[0]).decode('latin-1')
    df = pl.read_csv(io.StringIO(text), separator='\t', infer_schema_length=0)
    df = df.rename({c: c.strip() for c in df.columns})
    return df.select(pl.col('GEOID').str.zfill(5).alias('geo_id'),
                     pl.col('ALAND_SQMI').cast(pl.Float64).alias('land_sqmi')).sort('geo_id')


def parse(vintage: int = VINTAGE) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Newest raw pulls to (county table incl. travel-time bins, land area, provenance and sentinel counts)."""
    info = {'sentinels': {}}
    county = None
    for request in REQUESTS:
        rec = latest_county_record(vintage, request)
        part, sentinels = parse_county(rawstore.load_raw(rec.path))
        info['sentinels'].update(sentinels)
        info[f'{request}_raw'], info[f'{request}_sha256'] = str(rec.path), rec.sha256
        county = part if county is None else county.join(part.drop('name', 'state_fips'), on='geo_id', how='full',
                                                         coalesce=True)
    gaz_rec = latest_raw(SOURCE, GAZETTEER_URL.format(vintage=vintage))
    land = parse_gazetteer(rawstore.load_raw(gaz_rec.path))
    info['gazetteer_raw'], info['gazetteer_sha256'] = str(gaz_rec.path), gaz_rec.sha256
    return county.sort('geo_id'), land, info


def main(argv: list[str]) -> int:
    step = argv[1] if len(argv) > 1 else 'all'
    if step in ('fetch', 'all'):
        for key, result in fetch().items():
            print(f'{key}: {result.record.sha256[:12]} {result.record.path.name}')
    if step in ('parse', 'all'):
        county, land, info = parse()
        print(f'parsed {county.height} county rows, {land.height} gazetteer rows; '
              f'negative-sentinel counts: { {k: v for k, v in info["sentinels"].items() if v} }')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
