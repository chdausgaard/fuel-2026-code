import csv
import fcntl
import hashlib
from pathlib import Path
import subprocess
from typing import Iterable, Mapping, Any

from .paths import logs_dir, repo_root
from .rawstore import redact_url

RUN_COLUMNS = 'run_id utc_timestamp git_sha git_dirty config_hash spec_id data_snapshot_ids output_path notes'.split()
SCRAPE_COLUMNS = 'utc_timestamp source job url http_status raw_file raw_sha256 n_bytes rows_parsed status message'.split()


def _append(name: str, columns: list[str], row: Mapping[str, Any]) -> None:
    path = logs_dir() / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', newline='', encoding='utf-8') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            writer = csv.DictWriter(handle, fieldnames=columns)
            if path.stat().st_size == 0:
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def append_run_log(row: Mapping[str, Any] | None = None, **fields: Any) -> None:
    _append('run_log.csv', RUN_COLUMNS, {**(row or {}), **fields})


def append_scrape_log(row: Mapping[str, Any] | None = None, **fields: Any) -> None:
    values = {**(row or {}), **fields}
    if values.get('status') not in {'ok', 'warn', 'fail'}:
        raise ValueError('Invalid scrape status')
    if 'url' in values:
        values['url'] = redact_url(values['url'])
    _append('scrape_log.csv', SCRAPE_COLUMNS, values)


def _git(*args: str) -> str:
    return subprocess.check_output(['git', *args], cwd=repo_root(), text=True,
                                   stderr=subprocess.DEVNULL).strip()


def git_sha() -> str:
    return _git('rev-parse', 'HEAD')


def git_dirty() -> bool:
    return bool(_git('status', '--porcelain'))


def config_hash(paths: Iterable[str | Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(map(Path, paths), key=str):
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, 'big'))
        digest.update(content)
    return digest.hexdigest()
