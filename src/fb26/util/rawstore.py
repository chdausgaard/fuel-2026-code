from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .paths import raw_dir


SECRET_PARAMS = {'api_key', 'key', 'token'}


def redact_url(url: str) -> str:
    parts = urlsplit(url)
    query = urlencode([(k, 'REDACTED' if k.lower() in SECRET_PARAMS else v)
                       for k, v in parse_qsl(parts.query, keep_blank_values=True)])
    host = parts.netloc.rsplit('@', 1)[-1]
    return urlunsplit((parts.scheme, host, parts.path, query, ''))


def utc_iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError('A timezone-aware timestamp is required')
    return value.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')


@dataclass(frozen=True)
class RawRecord:
    path: Path
    meta_path: Path
    sha256: str
    n_bytes: int
    fetched_at_utc: datetime
    url: str


def meta_path_for(path: Path) -> Path:
    return path.with_suffix('').with_suffix('.meta.json')


def record_from_meta(meta_path: Path) -> RawRecord:
    meta = json.loads(meta_path.read_text())
    return RawRecord(Path(meta['raw_path']), meta_path, meta['sha256'], meta['n_bytes'],
                     datetime.fromisoformat(meta['fetched_at_utc'].replace('Z', '+00:00')), meta['url'])


def _publish(path: Path, content: bytes) -> None:
    # The enclosing directory flock serializes writers across the existence
    # check and atomic rename. Existing destinations are never replaced.
    fd, name = tempfile.mkstemp(dir=path.parent, prefix='.raw-')
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o444)
        if path.exists():
            raise FileExistsError(path)
        os.rename(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def store_raw(source: str, url: str, content: bytes, *, fetched_at_utc: datetime,
              http_status: int, headers: dict, ext: str) -> RawRecord:
    if not re.fullmatch(r'[A-Za-z0-9_-]+', source):
        raise ValueError('Unsafe source name')
    if not re.fullmatch(r'[A-Za-z0-9]+', ext):
        raise ValueError('Unsafe extension')
    timestamp = utc_iso(fetched_at_utc)
    fetched_at_utc = fetched_at_utc.astimezone(timezone.utc)
    safe_url = redact_url(url)
    parts = urlsplit(safe_url)
    slug = re.sub(r'[^A-Za-z0-9_-]+', '-', parts.netloc + parts.path + '?' + parts.query).strip('-')[:80]
    digest = hashlib.sha256(content).hexdigest()
    directory = raw_dir() / source / fetched_at_utc.strftime('%Y/%m/%d')
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{fetched_at_utc:%Y%m%dT%H%M%SZ}_{slug}_{digest[:12]}.{ext}.gz'
    meta_path = meta_path_for(path)
    selected = {k.lower(): v for k, v in headers.items()
                if k.lower() in {'content-type', 'content-length', 'last-modified', 'etag', 'date'}}
    meta = dict(url=safe_url, request_sha256=hashlib.sha256(url.encode()).hexdigest(),
                fetched_at_utc=timestamp, http_status=http_status,
                headers=selected, sha256=digest, n_bytes=len(content), raw_path=str(path))
    directory_fd = os.open(directory, os.O_RDONLY)
    try:
        fcntl.flock(directory_fd, fcntl.LOCK_EX)
        if path.exists():
            existing_bytes = gzip.decompress(path.read_bytes())
            if hashlib.sha256(existing_bytes).hexdigest() != digest:
                raise ValueError('Raw path collision: different SHA-256')
        else:
            _publish(path, gzip.compress(content, mtime=0))
        if not meta_path.exists():
            _publish(meta_path, json.dumps(meta, sort_keys=True, indent=2).encode())
        record = record_from_meta(meta_path)
        if record.sha256 != digest:
            raise ValueError('Metadata SHA-256 collision')
        load_raw(path)
        return record
    finally:
        fcntl.flock(directory_fd, fcntl.LOCK_UN)
        os.close(directory_fd)


def load_raw(path: str | Path) -> bytes:
    path = Path(path)
    record = record_from_meta(meta_path_for(path))
    content = gzip.decompress(path.read_bytes())
    if hashlib.sha256(content).hexdigest() != record.sha256 or len(content) != record.n_bytes:
        raise ValueError('Raw content failed SHA-256/length verification')
    return content
