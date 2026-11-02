from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import time
import hashlib
import json
from typing import Callable
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

from dotenv import load_dotenv
import os
import re
import requests
import yaml

from . import paths, rawstore, runlog


class RobotsDisallowed(RuntimeError):
    pass


class FetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class FetchResult:
    raw: rawstore.RawRecord
    content: bytes
    http_status: int
    from_cache: bool = False

    @property
    def record(self) -> rawstore.RawRecord:
        return self.raw


class PoliteClient:
    def __init__(self, *, session: requests.Session | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep,
                 now: Callable[[], datetime] | None = None) -> None:
        load_dotenv(paths.repo_root() / '.env')
        email = os.getenv('UA_CONTACT_EMAIL', '').strip()
        if not email:
            raise ValueError('UA_CONTACT_EMAIL is required')
        self.config = yaml.safe_load((paths.config_dir() / 'scrape.yaml').read_text())
        self.session = session if session is not None else requests.Session()
        self.user_agent = self.config['user_agent_template'].format(email=email)
        self.session.headers.update({'User-Agent': self.user_agent})
        self.clock, self.sleep = clock, sleep
        self.now = now or (lambda: datetime.now(timezone.utc))
        self._last: dict[str, float] = {}
        self._robots: dict[str, RobotFileParser] = {}
        self._crawl_delay: dict[str, float] = {}
        self._cache: dict[tuple[str, str], rawstore.RawRecord] = {}

    def _request(self, url: str) -> requests.Response:
        host = urlsplit(url).hostname or ''
        # robots.txt Crawl-delay raises the interval; it never lowers the 3 s floor.
        interval = max(3.0, float(self.config.get('host_min_intervals', {}).get(
            host, self.config['default_min_interval_s'])), self._crawl_delay.get(host, 0.0))
        for attempt in range(int(self.config['max_retries']) + 1):
            if host in self._last:
                delay = interval - (self.clock() - self._last[host])
                if delay > 0:
                    self.sleep(delay)
            self._last[host] = self.clock()
            response = None
            try:
                response = self.session.get(url, timeout=self.config['timeout_s'], allow_redirects=False)
                retry = response.status_code == 429 or 500 <= response.status_code < 600
                if not retry:
                    if 300 <= response.status_code < 400:
                        raise FetchError('Redirect requires an explicitly audited destination URL')
                    return response
            except (requests.ConnectionError, requests.Timeout):
                pass
            if attempt == int(self.config['max_retries']):
                raise FetchError('HTTP retry budget exhausted') from None
            delay = min(float(self.config['backoff_max_s']),
                        float(self.config['backoff_base_s']) * 2 ** attempt)
            if response is not None and response.headers.get('Retry-After'):
                retry_after = response.headers['Retry-After']
                try:
                    wait = float(retry_after)
                except ValueError:
                    try:
                        wait = (parsedate_to_datetime(retry_after) - self.now()).total_seconds()
                    except (ValueError, TypeError, OverflowError):
                        wait = 0
                delay = max(delay, wait)
            self.sleep(max(0, delay))
        raise AssertionError('Unreachable')

    def _store(self, url: str, source: str, response: requests.Response, ext: str) -> rawstore.RawRecord:
        record = rawstore.store_raw(source, url, response.content, fetched_at_utc=self.now(),
                                   http_status=response.status_code, headers=dict(response.headers), ext=ext)
        runlog.append_scrape_log(utc_timestamp=rawstore.utc_iso(record.fetched_at_utc), source=source,
                                job='http.get', url=record.url, http_status=response.status_code,
                                raw_file=str(record.path), raw_sha256=record.sha256, n_bytes=record.n_bytes,
                                rows_parsed='', status='ok', message='')
        return record

    def _check_robots(self, url: str) -> None:
        parts = urlsplit(url)
        origin = f'{parts.scheme}://{parts.netloc}'
        if origin not in self._robots:
            robots_url = origin + '/robots.txt'
            response = self._request(robots_url)
            parser = RobotFileParser(robots_url)
            # RFC 9309 §2.3.1.3: a 4xx robots.txt means no restrictions apply
            # (api.eia.gov answers 403 without a key). 429 is retried in _request.
            if 400 <= response.status_code < 500:
                parser.parse(['User-agent: *', 'Disallow:'])
            elif 200 <= response.status_code < 300:
                self._store(robots_url, 'robots', response, 'txt')
                # A leading byte-order mark hides the first rule from RobotFileParser (seen on elections.il.gov)
                parser.parse(response.text.lstrip('﻿').splitlines())
            else:
                raise FetchError(f'robots.txt failed with status {response.status_code}')
            crawl_delay = parser.crawl_delay(self.user_agent)
            if crawl_delay:
                self._crawl_delay[parts.hostname or ''] = float(crawl_delay)
            self._robots[origin] = parser
        if not self._robots[origin].can_fetch(self.user_agent, url):
            raise RobotsDisallowed('robots.txt disallows this URL')

    def get(self, url: str, *, source: str, params: dict | None = None,
            ext: str = 'html', cache_ttl_s: float | None = None) -> FetchResult:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', source):
            raise ValueError('Unsafe source name')
        prepared = requests.Request('GET', url, params=params).prepare().url
        if prepared is None or urlsplit(prepared).scheme not in {'http', 'https'}:
            raise ValueError('An HTTP(S) URL is required')
        key = (source, prepared)
        if cache_ttl_s is not None:
            records = [self._cache[key]] if key in self._cache else []
            # A one-way request fingerprint distinguishes URL+params without
            # retaining API credentials in metadata or filenames.
            fingerprint = hashlib.sha256(prepared.encode()).hexdigest()
            for meta in (paths.raw_dir() / source).glob('**/*.meta.json'):
                metadata = json.loads(meta.read_text())
                if metadata.get('request_sha256') == fingerprint:
                    records.append(rawstore.record_from_meta(meta))
            if records:
                record = max(records, key=lambda item: item.fetched_at_utc)
                age = (self.now() - record.fetched_at_utc).total_seconds()
                if 0 <= age < cache_ttl_s:
                    return FetchResult(record, rawstore.load_raw(record.path), 200, True)
        self._check_robots(prepared)
        response = self._request(prepared)
        if not 200 <= response.status_code < 300:
            raise FetchError(f'HTTP status {response.status_code}')
        record = self._store(prepared, source, response, ext)
        self._cache[key] = record
        return FetchResult(record, response.content, response.status_code)
