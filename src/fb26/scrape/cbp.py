"""Census County Business Patterns, county employment for the Oil_c control (WS3; source S9).

Oil_c = employment in NAICS 211 + 213111 + 213112 over total county employment (memo §5.4). One request per
NAICS code (plus the all-industry total, code 00), all-size establishments (EMPSZES=001), all legal forms
(LFO=001). A county missing from a NAICS response has no establishments in that code and counts as zero
employment; a county present with its employment flagged (EMP_F) is NOT imputed, only counted and kept.

CBP flags kept per cell: EMP_F (employment flag, e.g. withheld), EMP_N (noise range) and EMP_N_F (noise flag
G < 2%, H 2 to < 5%, J >= 5% noise, per the variables metadata of the release).

    python -m fb26.scrape.cbp              # fetch + parse
    python -m fb26.scrape.cbp fetch|parse
"""
from __future__ import annotations

import json
import sys

import polars as pl
import requests

from fb26.scrape.acs import api_key
from fb26.scrape.eia import latest_raw
from fb26.util import rawstore
from fb26.util.http import PoliteClient

PARSER_VERSION = 'cbp-0.1.0'
SOURCE = 'cbp'
YEAR = 2023  # latest CBP year listed by the API on 2026-10-19 (2024 returns 404)
API_URL = 'https://api.census.gov/data/{year}/cbp'
TOTAL_CODE = '00'
OIL_GAS_CODES = ('211', '213111', '213112')
GET = 'NAME,NAICS2017,EMP,EMP_F,EMP_N,EMP_N_F,ESTAB,EMPSZES'
CACHE_TTL_S = 24 * 3600
SCHEMA = {'geo_id': pl.Utf8, 'naics': pl.Utf8, 'emp': pl.Float64, 'emp_f': pl.Utf8, 'emp_n': pl.Utf8,
          'emp_n_f': pl.Utf8, 'estab': pl.Float64}


def params(naics: str, key: str, level: str = 'county') -> dict:
    geo = {'for': 'county:*', 'in': 'state:*'} if level == 'county' else {'for': 'state:*'}
    return {'get': GET, **geo, 'NAICS2017': naics, 'EMPSZES': '001', 'LFO': '001', 'key': key}


def fetch(client: PoliteClient | None = None, year: int = YEAR) -> dict:
    """County pulls for the four codes, plus state pulls used to count establishments missing from the county
    file (ESTAB is a count, so state ESTAB minus the county sum is the number of unpublished establishments)."""
    client = client or PoliteClient()
    key = api_key()
    return {(level, code): client.get(API_URL.format(year=year), source=SOURCE, params=params(code, key, level),
                                      ext='json', cache_ttl_s=CACHE_TTL_S)
            for level in ('county', 'state') for code in (TOTAL_CODE, *OIL_GAS_CODES)}


def parse_cell_file(content: bytes, naics: str, level: str = 'county') -> pl.DataFrame:
    """One API response to one row per county (or state) for this NAICS code, flags kept as strings, nothing
    imputed. `geo_id` is the 5-digit county FIPS, or the 2-digit state FIPS at state level."""
    rows = json.loads(content)
    header, body = rows[0], rows[1:]
    # the header repeats NAICS2017 and EMPSZES (predicate echoes); keep the first occurrence of each
    seen, keep = set(), []
    for i, name in enumerate(header):
        if name not in seen:
            seen.add(name)
            keep.append(i)
    df = pl.DataFrame([[r[i] for i in keep] for r in body], schema=[header[i] for i in keep], orient='row',
                      infer_schema_length=None) if body else pl.DataFrame(schema={header[i]: pl.Utf8 for i in keep})
    got = set(df['NAICS2017']) if df.height else set()
    if got - {naics}:
        raise ValueError(f'Unexpected NAICS codes {got} in response for {naics}')
    if df.height and set(df['EMPSZES']) != {'001'}:
        raise ValueError('CBP response contains non-total size classes')
    return df.select(
        (pl.col('state') + pl.col('county') if level == 'county' else pl.col('state')).alias('geo_id'),
        pl.lit(naics).alias('naics'),
        pl.col('EMP').cast(pl.Float64, strict=False).alias('emp'), pl.col('EMP_F').alias('emp_f'),
        pl.col('EMP_N').alias('emp_n'), pl.col('EMP_N_F').alias('emp_n_f'),
        pl.col('ESTAB').cast(pl.Float64, strict=False).alias('estab')).cast(SCHEMA).sort('geo_id')


def latest_cell_record(year: int, naics: str, level: str = 'county') -> rawstore.RawRecord:
    prepared = requests.Request('GET', API_URL.format(year=year),
                                params=params(naics, 'REDACTED', level)).prepare().url
    return latest_raw(SOURCE, rawstore.redact_url(prepared))


def parse(year: int = YEAR) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """(county cells, state cells, provenance); both long by NAICS code."""
    frames, info = {'county': [], 'state': []}, {'year': year}
    for level in frames:
        for code in (TOTAL_CODE, *OIL_GAS_CODES):
            rec = latest_cell_record(year, code, level)
            frames[level].append(parse_cell_file(rawstore.load_raw(rec.path), code, level))
            info[f'raw_{level}_{code}'], info[f'sha256_{level}_{code}'] = str(rec.path), rec.sha256
    return pl.concat(frames['county']), pl.concat(frames['state']), info


def main(argv: list[str]) -> int:
    step = argv[1] if len(argv) > 1 else 'all'
    if step in ('fetch', 'all'):
        for key, result in fetch().items():
            print(f'{key}: {result.record.sha256[:12]} {result.record.path.name}')
    if step in ('parse', 'all'):
        df, _, _ = parse()
        print(df.group_by('naics').agg(pl.len().alias('counties'), pl.col('emp').sum(), pl.col('estab').sum())
              .sort('naics'))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
