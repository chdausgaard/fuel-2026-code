"""Amlani & Algara county-level presidential returns, 1868-2020 (Harvard Dataverse, doi:10.7910/DVN/DGUMFI, CC0).

Second source for the five defective MEDSL county presidential state-years (2004 ME, MS, NY; 2008 NY; 2004 AK).
This file has no guestbook, so the guestbook POST of `medsl.DataverseSession` returns an HTML page instead of a signed
URL; the session below goes straight to the datafile URL with the token and follows the audited S3 redirect.

    python -m fb26.scrape.county_pres_hist     # fetch the Rdata file into the raw store (source county_returns_hist)
"""
from __future__ import annotations

import sys
from urllib.parse import urlsplit

from fb26.scrape import medsl
from fb26.util import rawstore
from fb26.util.http import FetchError, PoliteClient

SOURCE = 'county_returns_hist'
DOI = '10.7910/DVN/DGUMFI'
FILE_ID = 5028532
FILE_NAME = 'dataverse_shareable_presidential_county_returns_1868_2020.Rdata'
LICENSE = 'CC0 1.0'
URL = medsl.DATAFILE_URL.format(file_id=FILE_ID)
SHA256 = 'c3ad5e87a6dfd8aaf25725ca43b47e86c1cfe828958f6371fe096543906c1687'  # 1,459,576 bytes


class NoGuestbookSession(medsl.DataverseSession):
    """DataverseSession for a file without a guestbook: token GET, then the audited redirect."""

    def get(self, url, **kwargs):
        parts = urlsplit(url)
        if parts.hostname != medsl.DATAVERSE_HOST or not parts.path.startswith('/api/access/datafile/'):
            return super(medsl.DataverseSession, self).get(url, **kwargs)
        if not self._token:
            raise FetchError('DATAVERSE_API_TOKEN is not set')
        kwargs['allow_redirects'] = False
        response = super(medsl.DataverseSession, self).get(url, headers={'X-Dataverse-key': self._token}, **kwargs)
        if response.status_code == 303:
            target = response.headers.get('Location', '')
            if urlsplit(target).hostname not in medsl.REDIRECT_HOSTS:
                raise FetchError('Dataverse redirect to an unaudited host')
            self._sleep(self._interval)
            return super(medsl.DataverseSession, self).get(target, **kwargs)
        return response


def client() -> PoliteClient:
    medsl.load_dotenv(medsl.paths.repo_root() / '.env')
    return PoliteClient(session=NoGuestbookSession())


def raw_record() -> rawstore.RawRecord:
    """The stored Rdata file. Pinned by SHA-256: the raw store also holds one HTML page that a first, failed fetch
    stored under the same URL (6,023 bytes)."""
    records = [rawstore.record_from_meta(m) for m in (medsl.paths.raw_dir() / SOURCE).glob('**/*.meta.json')]
    records = [r for r in records if r.url == URL and r.sha256 == SHA256]
    if not records:
        raise FileNotFoundError(f'No raw record of {URL} with sha256 {SHA256[:12]}; run the fetch step first')
    return max(records, key=lambda r: r.fetched_at_utc)


def fetch(pc: PoliteClient | None = None):
    pc = pc or client()
    result = pc.get(URL, source=SOURCE, ext='rdata', cache_ttl_s=medsl.CACHE_TTL_S)
    if result.record.sha256 != SHA256:
        raise FetchError(f'unexpected content for {URL}: {result.record.n_bytes} bytes, sha256 {result.record.sha256[:12]}')
    return result


def main(argv: list[str]) -> int:
    result = fetch()
    print(result.record.path, result.record.sha256, result.record.n_bytes, 'cached' if result.from_cache else 'fetched')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
