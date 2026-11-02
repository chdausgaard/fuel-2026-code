"""MEDSL election returns (Harvard Dataverse) and Census geography files for the results build (WS5; S11, S12, S19).

Dataverse puts every MEDSL file behind a guestbook. Scripted access works with the account's API token: a POST of an
empty guestbook response to the file's access URL returns a signed URL, which answers 303 to a pre-signed S3 object.
`DataverseSession` does those steps inside one logical GET so that `PoliteClient` still owns throttling, robots,
retries and the raw store, and the raw record is keyed to the canonical Dataverse URL. One download writes one
guestbook record under the token holder's account; PoliteClient's cache avoids repeat records.

    python -m fb26.scrape.medsl                  # fetch every file in CATALOG + the Census files
    python -m fb26.scrape.medsl fetch <key>...   # only these catalog keys
    python -m fb26.scrape.medsl census           # Census files only
"""
from __future__ import annotations

import json
import os
import sys
import time
from urllib.parse import urlsplit

import requests
from dotenv import load_dotenv

from fb26.scrape.eia import latest_raw
from fb26.util import paths, rawstore
from fb26.util.http import FetchError, PoliteClient

SOURCE = 'medsl'
CENSUS_SOURCE = 'census_geo'
DATAVERSE_HOST = 'dataverse.harvard.edu'
DATAFILE_URL = f'https://{DATAVERSE_HOST}/api/access/datafile/' + '{file_id}'
# Hosts a Dataverse 303 may point to. Anything else is refused (PoliteClient's own rule: audited destinations only).
REDIRECT_HOSTS = ('dvn-cloud-iqss.s3.amazonaws.com',)
CACHE_TTL_S = 30 * 24 * 3600

# key: (doi, dataverse file id, file name, request original format?, use)
CATALOG = {
    'pres_county_2000_2024': ('10.7910/DVN/VOQCHQ', 13573089, 'countypres_2000-2024.tab', True,
                              '2020 presidential county baseline (S11)'),
    'senate_2022_precinct': ('10.7910/DVN/IAD3XR', 13996913, 'SENATE_precinct_general.tab', True,
                             '2022 Senate precinct returns (S12)'),
    'state_2022_precinct': ('10.7910/DVN/OAARCY', 14249788, 'STATE_precinct_general.csv', False,
                            '2022 state-office (governor) precinct returns (S12)'),
    # Governor races missing from the combined state-offices file (verified in results.py); 2022 by-state release.
    'bystate_ar': ('10.7910/DVN/UYQIEP', 14240723, '2022-ar-local-precinct-general.zip', False,
                   'Arkansas 2022 by-state file (zip) for the governor race'),
    'bystate_az': ('10.7910/DVN/UYQIEP', 14249794, '2022-az-local-precinct-general.tab', True,
                   'Arizona 2022 by-state file for the governor race'),
    'bystate_md': ('10.7910/DVN/UYQIEP', 14249804, '2022-md-local-precinct-general.tab', True,
                   'Maryland 2022 by-state file for the governor race'),
    'bystate_mi': ('10.7910/DVN/UYQIEP', 14248930, '2022-mi-local-precinct-general.zip', False,
                   'Michigan 2022 by-state file (zip) for the governor race'),
    'bystate_ca': ('10.7910/DVN/UYQIEP', 10855141, 'CA_2022_final.csv', False,
                   'California 2022 by-state file; checked: no US SENATE rows (kept as evidence, not used)'),
    'bystate_in': ('10.7910/DVN/UYQIEP', 10855180, '2022-in-local-precinct-general.tab', True,
                   'Indiana 2022 by-state file; checked: same 39-county Senate rows as the combined file (not used)'),
    'bystate_tn': ('10.7910/DVN/UYQIEP', 10855152, 'tn22_cleaned.tab', True,
                   'Tennessee 2022 by-state file for the governor race'),
    'pres_2020_precinct': ('10.7910/DVN/JXPREB', 13582897, 'PRESIDENT_precinct_general.csv', False,
                           '2020 presidential precinct returns: Connecticut towns and Kansas City, MO (S12)'),
    'pres_2016_precinct': ('10.7910/DVN/LYWX3D', 3345331, '2016-precinct-president.tab', True,
                           '2016 presidential precinct returns: Connecticut towns (county file has legacy counties)'),
    'pres_2024_ct': ('10.7910/DVN/XDJYKC', 13731178, '2024-ct-precinct-general.tab', True,
                     '2024 Connecticut precinct returns (towns), by-state file of the 2024 release'),
    'pres_2024_me': ('10.7910/DVN/XDJYKC', 13731129, '2024-me-precinct-general.tab', True,
                     '2024 Maine precinct returns; checked: same county totals as the county file, no UOCAVA unit (not used)'),
    'senate_2018_precinct': ('10.7910/DVN/DGNAFS', 6692658, 'SENATE_precinct_general.tab', True,
                             '2018 Senate precinct returns (S12)'),
    'state_2018_precinct': ('10.7910/DVN/ZFXEJU', 6692663, 'STATE_precinct_general.csv', False,
                            '2018 state-office (governor) precinct returns (S12)'),
    'bystate_2018_ia': ('10.7910/DVN/NVQYMG', 6692582, '2018-ia-precinct-general.tab', True,
                        'Iowa 2018 by-state file: governor race absent from the combined 2018 state-offices file'),
    'bystate_2018_ny': ('10.7910/DVN/NVQYMG', 6692578, '2018-ny-precinct-general.tab', True,
                        '2018 NY by-state file, checked against the combined 2018 files'),
    'bystate_2018_in': ('10.7910/DVN/NVQYMG', 6692610, '2018-in-precinct-general.tab', True,
                        '2018 IN by-state file, checked against the combined 2018 files'),
    'bystate_2018_nj': ('10.7910/DVN/NVQYMG', 6692604, '2018-nj-precinct-general.tab', True,
                        '2018 NJ by-state file, checked against the combined 2018 files'),
    'bystate_2018_pa': ('10.7910/DVN/NVQYMG', 6692580, '2018-pa-precinct-general.tab', True,
                        '2018 PA by-state file, checked against the combined 2018 files'),
    'bystate_2018_wv': ('10.7910/DVN/NVQYMG', 6692603, '2018-wv-precinct-general.tab', True,
                        '2018 WV by-state file, checked against the combined 2018 files'),
    'bystate_2018_ms': ('10.7910/DVN/NVQYMG', 6692579, '2018-ms-precinct-general.tab', True,
                        '2018 MS by-state file, checked against the combined 2018 files'),
    'bystate_2018_or': ('10.7910/DVN/NVQYMG', 6692598, '2018-or-precinct-general.tab', True,
                        '2018 OR by-state file, checked against the combined 2018 files'),
    'bystate_2018_vt': ('10.7910/DVN/NVQYMG', 6692606, '2018-vt-precinct-general.tab', True,
                        '2018 VT by-state file, checked against the combined 2018 files'),
    # U.S. House feasibility check (download only; no ETL yet)
    'house_2018_precinct': ('10.7910/DVN/IVIXLK', 6692624, 'HOUSE_precinct_general.tab', True,
                            '2018 U.S. House precinct returns, House-only release'),
    'house_2020_precinct': ('10.7910/DVN/VLGF2M', 13582893, 'HOUSE_precinct_general.tab', True,
                            '2020 U.S. House precinct returns, House-only release'),
    'house_2022_precinct': ('10.7910/DVN/EOKNGW', 13996912, 'HOUSE_precinct_general.tab', True,
                            '2022 U.S. House precinct returns, House-only release'),
    'house_2024_precinct': ('10.7910/DVN/USBYR4', 13731101, 'HOUSE_precinct_general.csv', False,
                            '2024 U.S. House precinct returns, House-only release'),
    'house_1976_2024_district': ('10.7910/DVN/IG0UN2', 13592823, '1976-2024-house.tab', False,
                                 'District-level U.S. House returns 1976-2024: reconciliation reference'),
    'house_codebook_2018': ('10.7910/DVN/IVIXLK', 6692626, 'codebook.md', False, 'codebook for the 2018 House file'),
    'house_codebook_2020': ('10.7910/DVN/VLGF2M', 6100385, '2020-precincts-codebook.md', False, 'codebook 2020'),
    'house_codebook_2024': ('10.7910/DVN/USBYR4', 13731100, '2024-precincts-codebook.md', False, 'codebook 2024'),
    'house_readme_2024': ('10.7910/DVN/USBYR4', 13731099, 'README.md', False, 'README of the 2024 House release'),
    'house_codebook_district': ('10.7910/DVN/IG0UN2', 12066703, 'codebook-us-house-1976\u20132024.md', False,
                                'codebook of the district-level House file'),
    'senate_2022_county': ('10.7910/DVN/YB60EJ', 7412054, 'senate_2022.tab', True,
                           'MEDSL county-level 2022 Senate, cross-check of the precinct aggregation'),
    'senate_statewide': ('10.7910/DVN/PEJ5QU', 13887039, '1976-2024-senate-state.tab', True,
                         'Reference totals: Senate statewide 1976-2024'),
    'president_statewide': ('10.7910/DVN/42MVDX', 13887042, '1976-2024-president.csv', False,
                            'Reference totals: president statewide 1976-2024'),
}
CENSUS_FILES = {
    # ACS 2024 5-year county list (the matching universe); the key is redacted in the raw store.
    'acs2024_counties': ('https://api.census.gov/data/2024/acs/acs5?get=NAME&for=county:*', 'json', True),
    # Connecticut town -> planning region (county codes 110-190), 2022-vintage ACS relationship file (S19).
    'ct_cousub22_tract22': ('https://www2.census.gov/geo/docs/maps-data/data/rel2022/acs22_cousub22_tract22_st09.txt',
                            'txt', False),
}


class DataverseSession(requests.Session):
    """requests.Session that completes the Dataverse guestbook + redirect dance for datafile URLs."""

    def __init__(self, token: str | None = None, *, sleep=time.sleep, min_interval_s: float = 3.0) -> None:
        super().__init__()
        self._token = token or os.getenv('DATAVERSE_API_TOKEN', '')
        self._sleep, self._interval = sleep, min_interval_s

    def get(self, url, **kwargs):
        parts = urlsplit(url)
        if parts.hostname != DATAVERSE_HOST or not parts.path.startswith('/api/access/datafile/'):
            return super().get(url, **kwargs)
        if not self._token:
            raise FetchError('DATAVERSE_API_TOKEN is not set')
        kwargs['allow_redirects'] = False
        auth = {'X-Dataverse-key': self._token}
        posted = super().post(url, headers={**auth, 'Content-Type': 'application/json'},
                              data=json.dumps({'guestbookResponse': {}}), timeout=kwargs.get('timeout'),
                              allow_redirects=False)
        if posted.status_code != 200:
            raise FetchError(f'Dataverse guestbook POST failed with status {posted.status_code}')
        signed = posted.json()['data']['signedUrl']
        self._sleep(self._interval)
        response = super().get(signed, **kwargs)
        if response.status_code == 303:
            target = response.headers.get('Location', '')
            if urlsplit(target).hostname not in REDIRECT_HOSTS:
                raise FetchError('Dataverse redirect to an unaudited host')
            self._sleep(self._interval)
            return super().get(target, **kwargs)
        return response


def client() -> PoliteClient:
    load_dotenv(paths.repo_root() / '.env')  # PoliteClient loads it too, but the session reads the token first
    return PoliteClient(session=DataverseSession())


def catalog_url(key: str) -> str:
    _, file_id, _, original, _ = CATALOG[key]
    return DATAFILE_URL.format(file_id=file_id) + ('?format=original' if original else '')


def catalog_ext(key: str) -> str:
    name, original = CATALOG[key][2], CATALOG[key][3]
    if name.endswith('.zip'):
        return 'zip'
    if name.endswith('.md'):
        return 'md'
    return 'csv' if original or name.endswith('.csv') else 'tsv'


def raw_record(key: str) -> rawstore.RawRecord:
    """Newest raw record of a catalog file or Census file."""
    if key in CATALOG:
        return latest_raw(SOURCE, catalog_url(key))
    return latest_raw(CENSUS_SOURCE, _census_stored_url(key))


def _census_request(key: str) -> tuple[str, dict | None]:
    url = CENSUS_FILES[key][0]
    if 'api.census.gov' not in url:
        return url, None
    base, query = url.split('?', 1)
    params = dict(p.split('=', 1) for p in query.split('&'))
    params['key'] = os.getenv('CENSUS_API_KEY', '')
    return base, params


def _census_stored_url(key: str) -> str:
    """The URL as the raw store records it (secret parameters redacted)."""
    base, params = _census_request(key)
    if params is None:
        return base
    params = {**params, 'key': 'x'}
    return rawstore.redact_url(requests.Request('GET', base, params=params).prepare().url)


def fetch_catalog(keys: list[str] | None = None, pc: PoliteClient | None = None) -> dict:
    pc = pc or client()
    out = {}
    for key in keys or list(CATALOG):
        out[key] = pc.get(catalog_url(key), source=SOURCE, ext=catalog_ext(key), cache_ttl_s=CACHE_TTL_S)
    return out


def fetch_census(pc: PoliteClient | None = None) -> dict:
    pc = pc or PoliteClient()
    out = {}
    for key, (_, ext, _) in CENSUS_FILES.items():
        base, params = _census_request(key)
        out[key] = pc.get(base, source=CENSUS_SOURCE, params=params, ext=ext, cache_ttl_s=CACHE_TTL_S)
    return out


def main(argv: list[str]) -> int:
    step = argv[1] if len(argv) > 1 else 'all'
    if step in ('fetch', 'all'):
        keys = argv[2:] or None
        for key, result in fetch_catalog(keys).items():
            print(f'{key}: {result.record.sha256[:12]} {result.record.n_bytes} bytes {result.record.path.name}')
    if step in ('census', 'all'):
        for key, result in fetch_census().items():
            print(f'{key}: {result.record.sha256[:12]} {result.record.n_bytes} bytes {result.record.path.name}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
