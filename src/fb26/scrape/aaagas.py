"""AAA state gasoline averages from third-party GitHub archives (WS1b-repo; source S1b).

jgreathouse9/AAAGas scrapes AAA's state-averages page daily (GitHub Actions) into Prices/MasterGas.csv.
gueyenono/ScrapeUSGasPrices is an older, inactive scraper (data/state/DATE-usa_gas_price-state.csv).
Repo contents are untrusted data: nothing from them is ever executed or checked out. Files are read
with `git cat-file` / `git log` against bare repositories only.

    python -m fb26.scrape.aaagas mirror      # ls-remote, fetch if changed, incremental bundle into the raw store
    python -m fb26.scrape.aaagas restore DIR # replay base + incremental bundles into a bare repo
    python -m fb26.scrape.aaagas parse       # newest stored bundles -> prices_state_daily_aaagas.parquet
    python -m fb26.scrape.aaagas overlap     # AAAGas vs lykmapipo date alignment -> reports/validation/github_sources_overlap.csv
    python -m fb26.scrape.aaagas eia-check   # compare with EIA weekly -> reports/validation/aaagas_vs_eia.csv

Date semantics (see reports/audit/S1b_github_aaa.md): the `Date` column is the runner's calendar date
(UTC) at scrape time, not an AAA "as of" date. Only AAA's "current" column is archived.
"""
from __future__ import annotations

import csv
import fcntl
import hashlib
import io
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import polars as pl
import yaml

from fb26.util import alerts, paths, rawstore, runlog

PARSER_VERSION = 'aaagas-0.1.0'
SOURCE_AAAGAS, SOURCE_GUEY = 'aaagas_repo', 'gueyenono_repo'   # raw-store source names
TAG_AAAGAS, TAG_GUEY = 'aaagas_github', 'gueyenono_github'     # `source` values in prices_state_daily
REPO_URL = {SOURCE_AAAGAS: 'https://github.com/jgreathouse9/AAAGas.git',
            SOURCE_GUEY: 'https://github.com/gueyenono/ScrapeUSGasPrices.git',
            'lykmapipo_repo': 'https://github.com/lykmapipo/US-Gas-Prices.git',
            'lykfork_repo': 'https://github.com/yjb8hktcjf-commits/US-Gas-Prices.git'}
BRANCH = {SOURCE_AAAGAS: 'refs/heads/main', SOURCE_GUEY: 'refs/heads/master', 'lykmapipo_repo': 'refs/heads/main',
          'lykfork_repo': 'refs/heads/main'}
MASTER_PATH = 'Prices/MasterGas.csv'
GUEY_DIR = 'data/state'
EXPECTED_UNITS = 51
STALE_AFTER_DAYS = 2

OUT_PARQUET = 'prices_state_daily_aaagas.parquet'
OUT_EIA_CSV = 'aaagas_vs_eia.csv'
SOURCE_MASTERGAS = 'aaagas_mastergas'   # raw copies of Prices/MasterGas.csv, stored when its blob changes
MASTERGAS_URL = 'https://github.com/jgreathouse9/AAAGas/blob/main/Prices/MasterGas.csv'
SOURCE_LYK, SOURCE_LYK_CSV, TAG_LYK = 'lykmapipo_repo', 'lykmapipo_state_csv', 'lykmapipo_github'
# Fork of lykmapipo that kept scraping metro averages after upstream stopped (2026-10-13); mirrored only, not parsed.
SOURCE_LYKFORK, SOURCE_LYKFORK_CSV = 'lykfork_repo', 'lykfork_state_csv'
LYK_DIR = 'data/state-daily-averages'

RANGES = {'regular': (1.00, 8.00), 'midgrade': (1.00, 8.00), 'premium': (1.00, 8.00), 'diesel': (1.50, 9.00)}
MASTER_PRODUCTS = {'Regular': 'regular', 'Mid-Grade': 'midgrade', 'Premium': 'premium', 'Diesel': 'diesel'}
GUEY_PRODUCTS = {'regular': 'regular', 'mid': 'midgrade', 'premium': 'premium', 'diesel': 'diesel'}
EIA_THRESHOLD = 0.15
EIA_SHIFTS = (-3, -2, -1, 1, 2, 3)

SCHEMA = {
    'source': pl.String, 'as_of_date': pl.Date, 'state_abbr': pl.String, 'state_fips': pl.String,
    'product': pl.String, 'price_usd': pl.Float64, 'lag_field': pl.String,
    'derived_from_lag_field': pl.Boolean, 'raw_file': pl.String, 'raw_sha256': pl.String,
    'capture_ref': pl.String, 'captured_at_utc': pl.Datetime('us', 'UTC'),
    'parser_version': pl.String, 'flags': pl.String,
}


# ---------------------------------------------------------------- git plumbing

def user_agent() -> str:
    from dotenv import load_dotenv
    load_dotenv(paths.repo_root() / '.env')
    email = os.getenv('UA_CONTACT_EMAIL', '').strip()
    if not email:
        raise ValueError('UA_CONTACT_EMAIL is required')
    template = yaml.safe_load((paths.config_dir() / 'scrape.yaml').read_text())['user_agent_template']
    return template.format(email=email)


def _git(args: list[str], *, git_dir: Path | str | None = None, input: bytes | None = None,
         timeout: int = 300, ua: str | None = None) -> bytes:
    cmd = ['git']
    if ua:
        cmd += ['-c', f'http.userAgent={ua}']
    cmd += ['-c', 'core.hooksPath=/dev/null', '-c', 'protocol.version=2', '-c', 'transfer.fsckObjects=true']
    if git_dir is not None:
        cmd += ['--git-dir', str(git_dir)]
    env = {**os.environ, 'GIT_TERMINAL_PROMPT': '0', 'GIT_CONFIG_NOSYSTEM': '1', 'LC_ALL': 'C'}
    done = subprocess.run(cmd + args, input=input, capture_output=True, timeout=timeout, env=env)
    if done.returncode != 0:
        raise RuntimeError(f'git {" ".join(args[:2])} failed ({done.returncode}): '
                           f'{done.stderr.decode(errors="replace").strip()[:500]}')
    return done.stdout


def read_blobs(git_dir: Path, specs: list[str]) -> dict[str, bytes]:
    """`<rev>:<path>` -> bytes via one `git cat-file --batch` process."""
    out = _git(['cat-file', '--batch'], git_dir=git_dir, input=''.join(s + '\n' for s in specs).encode())
    blobs, pos = {}, 0
    for spec in specs:
        end = out.index(b'\n', pos)
        header = out[pos:end].decode().split()
        if len(header) != 3 or header[1] != 'blob':
            raise FileNotFoundError(f'{spec}: {out[pos:end].decode()}')
        size = int(header[2])
        blobs[spec] = out[end + 1:end + 1 + size]
        pos = end + 1 + size + 1
    return blobs


def heads(git_dir: Path) -> dict[str, str]:
    out = _git(['for-each-ref', '--format=%(refname) %(objectname)', 'refs/heads'], git_dir=git_dir)
    return dict(line.split(' ') for line in out.decode().splitlines())


def parse_ls_remote(output: str) -> dict[str, str]:
    refs = {}
    for line in output.splitlines():
        sha, _, ref = line.partition('\t')
        if ref.startswith('refs/heads/'):
            refs[ref] = sha
    return refs


def latest_record(source: str) -> rawstore.RawRecord:
    records = [rawstore.record_from_meta(m) for m in (paths.raw_dir() / source).glob('**/*.meta.json')]
    if not records:
        raise FileNotFoundError(f'No stored bundle for {source}; run `mirror` first')
    return max(records, key=lambda r: r.fetched_at_utc)


def open_bundle(record: rawstore.RawRecord, workdir: Path) -> Path:
    """Verify the stored bundle, extract it into a bare repo under `workdir` (outside the project)."""
    bundle = workdir / 'repo.bundle'
    bundle.write_bytes(rawstore.load_raw(record.path))
    repo = workdir / 'repo.git'
    _git(['init', '--bare', '--quiet', str(repo)])
    _git(['fetch', '--quiet', str(bundle), '+refs/heads/*:refs/heads/*'], git_dir=repo)
    return repo


# ---------------------------------------------------------------- mirror

# The first stored bundle of a repo is a full base; every later one holds only objects added since the previous
# tips. A per-repo append-only manifest (data/interim/<key>_bundles.jsonl) records, per stored bundle, its kind,
# prerequisites and new tips, so base + increments can be replayed in order (`restore`). A non-fast-forward
# upstream change stores a new full bundle (with a warning), and replay starts from the latest full one.
# `csv_only` entries record a raw copy of a published file without a bundle. Entry field `copies` lists raw copies
# ({path, blob, raw_file, sha256}); legacy entries carry master_blob / master_raw_file / master_sha256 instead.
FULL_KINDS = ('base', 'full')


@dataclass(frozen=True)
class RepoSpec:
    key: str
    title: str                     # name used in alert subjects
    source: str                    # raw-store source of the bundles
    base_sources: tuple            # raw-store sources searched for the base bundle (legacy names included)
    url: str
    branch: str
    copy_source: str               # raw-store source of single-file copies
    copy_url: str                  # URL template, `{path}` filled in
    copy_paths: Callable           # (git_dir) -> paths to keep raw copies of, read at `branch`
    latest_units: Callable         # (git_dir) -> (latest as-of date, units on that date)


def _master_paths(git_dir: Path) -> list[str]:
    return [MASTER_PATH]


def _lyk_paths(git_dir: Path) -> list[str]:
    names = _git(['ls-tree', '-r', '--name-only', BRANCH[SOURCE_LYK], LYK_DIR + '/'], git_dir=git_dir).decode().splitlines()
    return sorted(n for n in names if re.fullmatch(re.escape(LYK_DIR) + r'/\d{4}-\d{2}-\d{2}\.csv', n))


def latest_date_units(git_dir: Path, ref: str = BRANCH[SOURCE_AAAGAS]) -> tuple[date, int]:
    spec = f'{ref}:{MASTER_PATH}'
    counts: dict[str, int] = defaultdict(int)
    for row in csv.DictReader(io.StringIO(read_blobs(git_dir, [spec])[spec].decode('utf-8'))):
        counts[row['Date']] += 1
    latest = max(counts)
    return date.fromisoformat(latest), counts[latest]


def lyk_latest_units(git_dir: Path) -> tuple[date, int]:
    newest = _lyk_paths(git_dir)[-1]
    spec = f'{BRANCH[SOURCE_LYK]}:{newest}'
    rows = list(csv.DictReader(io.StringIO(read_blobs(git_dir, [spec])[spec].decode('utf-8'))))
    return max(date.fromisoformat(r['Date']) for r in rows), len(rows)


AAAGAS = RepoSpec('aaagas', 'AAAGas', SOURCE_AAAGAS, (SOURCE_AAAGAS,), REPO_URL[SOURCE_AAAGAS], BRANCH[SOURCE_AAAGAS],
                  SOURCE_MASTERGAS, MASTERGAS_URL, _master_paths, latest_date_units)
LYKMAPIPO = RepoSpec('lykmapipo', 'lykmapipo', SOURCE_LYK, (SOURCE_LYK, 'lykmapipo_US-Gas-Prices_repo'), REPO_URL[SOURCE_LYK],
                     BRANCH[SOURCE_LYK], SOURCE_LYK_CSV,
                     'https://github.com/lykmapipo/US-Gas-Prices/blob/main/{path}', _lyk_paths, lyk_latest_units)
LYKFORK = RepoSpec('lykfork', 'lykmapipo fork (yjb8hktcjf-commits)', SOURCE_LYKFORK, (SOURCE_LYKFORK,),
                   REPO_URL[SOURCE_LYKFORK], BRANCH[SOURCE_LYKFORK], SOURCE_LYKFORK_CSV,
                   'https://github.com/yjb8hktcjf-commits/US-Gas-Prices/blob/main/{path}', _lyk_paths, lyk_latest_units)
SPECS = {s.key: s for s in (AAAGAS, LYKMAPIPO, LYKFORK)}


def default_mirror_dir(spec: RepoSpec) -> Path:
    return paths.interim_dir() / f'{spec.key}_mirror.git'


def manifest_path(mirror_dir: Path) -> Path:
    name = Path(mirror_dir).name
    stem = name[:-len('_mirror.git')] + '_bundles.jsonl' if name.endswith('_mirror.git') else name + '.bundles.jsonl'
    return Path(mirror_dir).parent / stem


def read_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_manifest(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(json.dumps(entry, sort_keys=True) + '\n')
        handle.flush()
        fcntl.flock(handle, fcntl.LOCK_UN)


def entry_copies(entry: dict) -> list[dict]:
    copies = list(entry.get('copies') or [])
    if entry.get('master_blob'):   # legacy field names
        copies.append(dict(path=MASTER_PATH, blob=entry['master_blob'], raw_file=entry['master_raw_file'],
                           sha256=entry['master_sha256']))
    return copies


def _bundle_heads(content: bytes) -> dict[str, str]:
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / 'b.bundle'
        bundle.write_bytes(content)
        out = _git(['bundle', 'list-heads', str(bundle)]).decode()
    return {ref: sha for sha, ref in (line.split(' ', 1) for line in out.splitlines())
            if ref.startswith('refs/heads/')}


def base_entry(spec: RepoSpec = AAAGAS) -> dict | None:
    """Manifest entry for the newest stored bundle, assumed to be a full one (the orchestrator's bundle)."""
    records = []
    for source in spec.base_sources:
        try:
            records.append(latest_record(source))
        except FileNotFoundError:
            pass
    if not records:
        return None
    rec = max(records, key=lambda r: r.fetched_at_utc)
    return dict(kind='base', created_utc=_iso(rec.fetched_at_utc), raw_file=str(rec.path), raw_sha256=rec.sha256,
                n_bytes=rec.n_bytes, prerequisites={}, tips=_bundle_heads(rawstore.load_raw(rec.path)), copies=[])


def chain_entries(mpath: Path, spec: RepoSpec = AAAGAS) -> list[dict]:
    """Manifest entries, or the seed entry derived from the base bundle if the manifest is empty."""
    entries = read_manifest(mpath)
    if entries:
        return entries
    base = base_entry(spec)
    return [{**base, 'seq': 0}] if base else []


def restore(dest: Path, entries: list[dict], branch: str = BRANCH[SOURCE_AAAGAS]) -> None:
    """Replay the latest full bundle and the increments after it into the bare repo `dest` (created if missing)."""
    if not (Path(dest) / 'HEAD').exists():
        _git(['init', '--bare', '--quiet', str(dest)])
        _git(['symbolic-ref', 'HEAD', branch], git_dir=dest)
    start = max((i for i, e in enumerate(entries) if e['kind'] in FULL_KINDS), default=None)
    if start is None:
        return
    for entry in (e for e in entries[start:] if e.get('raw_file')):   # csv_only entries carry no bundle
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / 'b.bundle'
            bundle.write_bytes(rawstore.load_raw(entry['raw_file']))
            _git(['bundle', 'verify', '--quiet', str(bundle)], git_dir=dest)
            _git(['fetch', '--quiet', str(bundle), '+refs/heads/*:refs/heads/*'], git_dir=dest)


def commit_bundles(repo: Path, entries: list[dict]) -> dict[str, dict]:
    """commit SHA -> manifest entry of the stored bundle that first carries it (replay order)."""
    start = max((i for i, e in enumerate(entries) if e['kind'] in FULL_KINDS), default=None)
    if start is None:
        return {}
    seen, mapping = [], {}
    for entry in (e for e in entries[start:] if e.get('raw_file')):
        tips = list(entry['tips'].values())
        args = ['rev-list', *tips] + (['--not', *seen] if seen else [])
        for sha in _git(args, git_dir=repo).decode().split():
            mapping[sha] = entry
        seen += tips
    return mapping


def _bootstrap_mirror(mirror_dir: Path, spec: RepoSpec = AAAGAS) -> None:
    mirror_dir.parent.mkdir(parents=True, exist_ok=True)
    restore(mirror_dir, chain_entries(manifest_path(mirror_dir), spec), spec.branch)   # delta-only first fetch
    if not (mirror_dir / 'HEAD').exists():
        raise RuntimeError('mirror bootstrap failed')


def _iso(value: datetime) -> str:
    return rawstore.utc_iso(value)


def _is_fast_forward(git_dir: Path, prev: dict[str, str], new: dict[str, str]) -> bool:
    for ref, old in prev.items():
        if ref not in new:
            return False
        if new[ref] != old:
            try:
                _git(['merge-base', '--is-ancestor', old, new[ref]], git_dir=git_dir)
            except RuntimeError:
                return False
    return True


def _make_bundle(mirror_dir: Path, prev: dict[str, str], new: dict[str, str]) -> tuple[bytes, str, bool]:
    """(bundle bytes, kind, forced_full). Incremental when every previous tip is an ancestor of its new tip."""
    forced = False
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / 'out.bundle'
        if prev:
            if _is_fast_forward(mirror_dir, prev, new):
                try:
                    _git(['bundle', 'create', '--quiet', str(out), '--branches', '--not', *prev.values()],
                         git_dir=mirror_dir)
                    _git(['bundle', 'verify', '--quiet', str(out)], git_dir=mirror_dir)
                    return out.read_bytes(), 'incremental', False
                except RuntimeError:
                    pass
            forced = True
        _git(['bundle', 'create', '--quiet', str(out), '--branches'], git_dir=mirror_dir)
        _git(['bundle', 'verify', '--quiet', str(out)], git_dir=mirror_dir)
        return out.read_bytes(), 'full', forced


def mirror(remote: str | None = None, mirror_dir: Path | None = None, now=None, spec: RepoSpec = AAAGAS) -> dict:
    """One idempotent run for one repo: ls-remote; fetch only if refs changed; store an incremental bundle (and raw
    copies of the published file(s)) only if refs differ from the tips recorded in the manifest."""
    remote = remote or spec.url
    mirror_dir = Path(mirror_dir) if mirror_dir else default_mirror_dir(spec)
    mpath = manifest_path(mirror_dir)
    now = now or (lambda: datetime.now(timezone.utc))
    result = {'repo': spec.key, 'fetched': False, 'bundled': False, 'bundle_kind': '', 'raw_file': '',
              'raw_sha256': '', 'n_bytes': 0, 'copies_stored': 0, 'units_latest': None, 'latest_date': None}
    warnings = []
    try:
        ua = user_agent()
        if not mirror_dir.exists():
            _bootstrap_mirror(mirror_dir, spec)
        lock_fd = os.open(mirror_dir, os.O_RDONLY)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(lock_fd)
            result['message'] = 'another mirror run holds the lock; skipped'
            runlog.append_scrape_log(utc_timestamp=_iso(now()), source=spec.source, job='mirror', url=remote,
                                     status='warn', message=result['message'])
            return result
        try:
            remote_refs = parse_ls_remote(_git(['ls-remote', '--heads', remote], ua=ua, timeout=120).decode())
            if not remote_refs:
                raise RuntimeError('ls-remote returned no branch refs')
            if remote_refs != heads(mirror_dir):
                _git(['fetch', '--quiet', '--no-tags', '--prune', remote, '+refs/heads/*:refs/heads/*'],
                     git_dir=mirror_dir, ua=ua, timeout=900)
                result['fetched'] = True
            local = heads(mirror_dir)
            entries = read_manifest(mpath)
            if not entries:
                seed = base_entry(spec)
                if seed:
                    append_manifest(mpath, {**seed, 'seq': 0})
                    entries = [{**seed, 'seq': 0}]
            prev = entries[-1]['tips'] if entries else {}
            entry = dict(seq=len(entries), kind='csv_only', created_utc=_iso(now()), raw_file=None, raw_sha256=None,
                         n_bytes=0, prerequisites={}, tips=local, copies=[])
            if local != prev:
                content, kind, forced = _make_bundle(mirror_dir, prev, local)
                if forced:
                    warnings.append('previous tips are not ancestors of the new tips (force-push or ref removal); '
                                    'stored a full bundle')
                rec = rawstore.store_raw(spec.source, spec.url, content, fetched_at_utc=now(), http_status=200,
                                         headers={'content-type': 'application/x-git-bundle'}, ext='bundle')
                entry.update(kind=kind, raw_file=str(rec.path), raw_sha256=rec.sha256, n_bytes=rec.n_bytes,
                             prerequisites=dict(prev) if kind == 'incremental' else {})
                result.update(bundled=True, bundle_kind=kind, raw_file=str(rec.path), raw_sha256=rec.sha256,
                              n_bytes=rec.n_bytes)
            # raw copies of the published file(s) whenever a blob is new (also on the first run after seeding)
            known = {c['blob'] for e in entries for c in entry_copies(e)}
            for path in spec.copy_paths(mirror_dir):
                blob_spec = f'{spec.branch}:{path}'
                blob = _git(['rev-parse', blob_spec], git_dir=mirror_dir).decode().strip()
                if blob in known:
                    continue
                csv_rec = rawstore.store_raw(spec.copy_source, spec.copy_url.format(path=path),
                                             read_blobs(mirror_dir, [blob_spec])[blob_spec], fetched_at_utc=now(),
                                             http_status=200, headers={'content-type': 'text/csv'}, ext='csv')
                entry['copies'].append(dict(path=path, blob=blob, raw_file=str(csv_rec.path),
                                            sha256=csv_rec.sha256))
                known.add(blob)
                result['copies_stored'] += 1
            if result['bundled'] or entry['copies']:
                append_manifest(mpath, entry)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
        latest, units = spec.latest_units(mirror_dir)
        result.update(latest_date=latest.isoformat(), units_latest=units)
        status, notes = 'ok', ['fetched' if result['fetched'] else 'refs unchanged',
                               f"bundled ({result['bundle_kind']}, {result['n_bytes']} bytes)"
                               if result['bundled'] else 'no new bundle']
        if result['copies_stored']:
            notes.append(f"{result['copies_stored']} raw file copies stored")
        if warnings:
            status = 'warn'
            notes += warnings
            alerts.send_alert(f'{spec.title} mirror: full bundle', '; '.join(warnings), level='warn')
        if units != EXPECTED_UNITS:
            alerts.check_row_count(f'{spec.key}_latest_date_units', units, EXPECTED_UNITS, tol=0.0)
            status = 'warn'
            notes.append(f'{units} units on {latest}, expected {EXPECTED_UNITS}')
        if (now().date() - latest).days > STALE_AFTER_DAYS:
            alerts.send_alert(f'{spec.title} stale', f'Latest date in {spec.title} data is {latest}', level='warn')
            status = 'warn'
            notes.append(f'latest date {latest} is stale')
        result['message'] = '; '.join(notes) + f'; latest date {latest}'
        runlog.append_scrape_log(utc_timestamp=_iso(now()), source=spec.source, job='mirror', url=remote,
                                 raw_file=result['raw_file'], raw_sha256=result['raw_sha256'],
                                 n_bytes=result['n_bytes'] or '', rows_parsed=units, status=status,
                                 message=result['message'])
        return result
    except Exception as exc:
        message = f'{type(exc).__name__}: {exc}'[:500]
        alerts.send_alert(f'{spec.title} mirror failed', message)
        runlog.append_scrape_log(utc_timestamp=_iso(now()), source=spec.source, job='mirror', url=remote,
                                 status='fail', message=message)
        raise


def mirror_all(now=None) -> list[dict]:
    """Mirror every configured repo; one repo failing does not stop the others (exit status reports it)."""
    results, failed = [], False
    for spec in SPECS.values():
        try:
            results.append(mirror(spec=spec, now=now))
        except Exception as exc:
            failed = True
            results.append({'repo': spec.key, 'error': f'{type(exc).__name__}: {exc}'})
    if failed:
        raise MirrorFailed(results)
    return results


class MirrorFailed(RuntimeError):
    def __init__(self, results):
        super().__init__('; '.join(r['error'] for r in results if 'error' in r))
        self.results = results


# ---------------------------------------------------------------- parsing

def load_states() -> tuple[dict[str, tuple[str, str]], dict[str, str]]:
    """name -> (abbr, fips); abbr -> fips."""
    with (paths.config_dir() / 'states.csv').open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_UNITS:
        raise ValueError(f'states.csv has {len(rows)} rows, expected {EXPECTED_UNITS}')
    by_name = {r['state_name']: (r['state_abbr'], r['state_fips']) for r in rows}
    by_abbr = {r['state_abbr']: r['state_fips'] for r in rows}
    return by_name, by_abbr


def parse_price(text: str | None) -> float | None:
    if text is None:
        return None
    text = text.strip().lstrip('$').replace(',', '')
    if text in ('', 'NA', 'N/A', 'nan', 'NaN'):
        return None
    return float(text)


def price_flags(product: str, price: float | None) -> list[str]:
    if price is None or math.isnan(price):
        return ['missing_price']
    lo, hi = RANGES[product]
    return [] if lo <= price <= hi else ['out_of_range']


def to_utc(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def timing_flags(as_of: date, captured: datetime | None) -> list[str]:
    if captured is None:
        return ['commit_unknown']
    if captured.date() > as_of:
        return ['captured_after_asof_day']
    if captured.date() < as_of:
        return ['captured_before_asof_day']
    return []


def master_history(git_dir: Path, ref: str = BRANCH[SOURCE_AAAGAS], path: str = MASTER_PATH) -> dict:
    """Walk every commit touching `path`; record, per (state name, date), each version of the row that was
    added (values, commit, commit time) so the capture commit of the *final* value can be identified."""
    out = _git(['log', '--reverse', '-p', '-U0', '--no-color', '--no-ext-diff',
                '--format=@@C%x09%H%x09%cI%x09%s', ref, '--', path], git_dir=git_dir, timeout=900).decode()
    adds: dict[tuple[str, str], list[tuple[tuple, str, datetime]]] = defaultdict(list)
    commit = None
    n_removed = 0
    for line in out.splitlines():
        if line.startswith('@@C\t'):
            _, sha, when, _subject = line.split('\t', 3)
            commit = (sha, to_utc(when))
        elif line.startswith('+') and not line.startswith('+++'):
            fields = next(csv.reader([line[1:]]))
            if len(fields) == 6 and fields[0] != 'State':
                adds[(fields[0], fields[5])].append((tuple(fields[1:5]), *commit))
        elif line.startswith('-') and not line.startswith('---'):
            n_removed += 1
    return {'adds': adds, 'n_removed_lines': n_removed}


def capture_for(history: dict, key: tuple[str, str], final_values: tuple) -> tuple[str | None, datetime | None, bool]:
    """(commit sha, commit time UTC, revised?) for the first commit adding `final_values` for `key`."""
    versions = history['adds'].get(key, [])
    revised = any(v[0] != final_values for v in versions)
    for values, sha, when in versions:
        if values == final_values:
            return sha, when, revised
    return None, None, revised


def parse_master(content: bytes, history: dict, raw_file: str, raw_sha256: str,
                 by_name: dict[str, tuple[str, str]]) -> list[dict]:
    parsed = list(csv.DictReader(io.StringIO(content.decode('utf-8'))))
    seen: dict[tuple[str, str], int] = defaultdict(int)
    for rec in parsed:
        seen[(rec['State'], rec['Date'])] += 1
    rows = []
    for rec in parsed:
        name = rec['State']
        if name not in by_name:
            raise ValueError(f'Unmapped state name in {MASTER_PATH}: {name!r}')
        abbr, fips = by_name[name]
        as_of = date.fromisoformat(rec['Date'])
        final = tuple(rec[c] for c in ('Regular', 'Mid-Grade', 'Premium', 'Diesel'))
        sha, when, revised = capture_for(history, (name, rec['Date']), final)
        base = (['revised'] if revised else []) + timing_flags(as_of, when)
        if seen[(name, rec['Date'])] > 1:
            base.append('duplicate_row')
        for column, product in MASTER_PRODUCTS.items():
            price = parse_price(rec[column])
            rows.append(_row(TAG_AAAGAS, as_of, abbr, fips, product, price, raw_file, raw_sha256, sha, when,
                             base + price_flags(product, price)))
    return rows


def _row(source, as_of, abbr, fips, product, price, raw_file, raw_sha256, sha, when, flags) -> dict:
    return dict(source=source, as_of_date=as_of, state_abbr=abbr, state_fips=fips, product=product,
                price_usd=price, lag_field='current', derived_from_lag_field=False, raw_file=raw_file,
                raw_sha256=raw_sha256, capture_ref=sha, captured_at_utc=when, parser_version=PARSER_VERSION,
                flags=';'.join(flags))


def guey_history(git_dir: Path, ref: str = BRANCH[SOURCE_GUEY]) -> dict:
    """Per data/state file: the commit that wrote its current content (`last`) and any rewrite commits."""
    out = _git(['log', '--reverse', '--name-status', '--no-renames', '--format=@@C%x09%H%x09%cI', ref, '--',
                GUEY_DIR], git_dir=git_dir, timeout=300).decode()
    last, modified, commit = {}, defaultdict(list), None
    for line in out.splitlines():
        if line.startswith('@@C\t'):
            _, sha, when = line.split('\t')
            commit = (sha, to_utc(when))
        elif line and commit:
            status, _, name = line.partition('\t')
            if status in ('A', 'M'):
                last[name] = commit
            if status == 'M':
                modified[name].append(commit)
    return {'last': last, 'modified': modified}


def parse_guey(git_dir: Path, history: dict, raw_file: str, raw_sha256: str,
               by_abbr: dict[str, str], ref: str = BRANCH[SOURCE_GUEY]) -> list[dict]:
    names = _git(['ls-tree', '-r', '--name-only', ref, GUEY_DIR + '/'], git_dir=git_dir).decode().splitlines()
    pattern = re.escape(GUEY_DIR) + r'/\d{4}-\d{2}-\d{2}-usa_gas_price-state\.csv'
    names = [n for n in names if re.fullmatch(pattern, n)]
    blobs = read_blobs(git_dir, [f'{ref}:{n}' for n in names])
    rows = []
    for name in sorted(names):
        file_date = date.fromisoformat(Path(name).name[:10])
        sha, when = history['last'].get(name, (None, None))
        file_flags = ['rewritten_after_first_commit'] if name in history['modified'] else []
        parsed = list(csv.DictReader(io.StringIO(blobs[f'{ref}:{name}'].decode('utf-8'))))
        counts: dict[str, int] = defaultdict(int)
        for rec in parsed:
            counts[rec['state']] += 1
        for rec in parsed:
            abbr = rec['state']
            if abbr not in by_abbr:
                raise ValueError(f'Unmapped state abbreviation in {name}: {abbr!r}')
            as_of = date.fromisoformat(rec['date'])
            base = file_flags + timing_flags(as_of, when)
            if as_of != file_date:
                base.append('date_ne_filename')
            if counts[abbr] > 1:
                base.append('duplicate_row')
            for column, product in GUEY_PRODUCTS.items():
                price = parse_price(rec[column])
                rows.append(_row(TAG_GUEY, as_of, abbr, by_abbr[abbr], product, price, raw_file, raw_sha256,
                                 sha, when, base + price_flags(product, price)))
    return rows


def lyk_history(git_dir: Path, ref: str = BRANCH[SOURCE_LYK]) -> dict:
    """Per state-daily file: the commit that last added or modified it (files are deleted again by the next day's
    commit, so the content only exists in history), and whether it was ever rewritten."""
    out = _git(['log', '--reverse', '--name-status', '--no-renames', '--format=@@C%x09%H%x09%cI', ref, '--',
                LYK_DIR], git_dir=git_dir, timeout=600).decode()
    last, writes, commit = {}, defaultdict(int), None
    for line in out.splitlines():
        if line.startswith('@@C\t'):
            _, sha, when = line.split('\t')
            commit = (sha, to_utc(when))
        elif line and commit:
            status, _, name = line.partition('\t')
            if status in ('A', 'M'):
                last[name] = commit
                writes[name] += 1
    return {'last': last, 'writes': writes}


def parse_lyk(git_dir: Path, history: dict, by_commit: dict[str, dict], copies: dict[str, str],
              by_name: dict[str, tuple[str, str]], default_raw: tuple[str, str] = ('', '')) -> list[dict]:
    """State-daily files of lykmapipo/US-Gas-Prices read at the commit that wrote them. `by_commit` maps a commit
    to the stored bundle carrying it; `copies` maps a file's sha256 to its stored raw csv copy."""
    pattern = re.compile(re.escape(LYK_DIR) + r'/(\d{4}-\d{2}-\d{2})\.csv')
    names = sorted(n for n in history['last'] if pattern.fullmatch(n))
    specs = [f'{history["last"][n][0]}:{n}' for n in names]
    blobs = read_blobs(git_dir, specs)
    abbr_ok = {abbr: (fips, name) for name, (abbr, fips) in by_name.items()}
    rows = []
    for name, spec in zip(names, specs):
        content = blobs[spec]
        sha, when = history['last'][name]
        file_date = date.fromisoformat(pattern.fullmatch(name).group(1))
        digest = hashlib.sha256(content).hexdigest()
        if digest in copies:
            raw_file, raw_sha = copies[digest], digest
        elif sha in by_commit:
            raw_file, raw_sha = by_commit[sha]['raw_file'], by_commit[sha]['raw_sha256']
        else:
            raw_file, raw_sha = default_raw
        file_flags = ['rewritten_after_first_commit'] if history['writes'][name] > 1 else []
        for rec in csv.DictReader(io.StringIO(content.decode('utf-8'))):
            legacy = 'State-Abbreviation' not in rec          # one early file has columns State,Regular,...,Date
            if legacy:
                if rec['State'] not in by_name:
                    raise ValueError(f'Unmapped state name in {name}: {rec["State"]!r}')
                abbr = by_name[rec['State']][0]
            else:
                abbr = rec['State-Abbreviation']
            if abbr not in abbr_ok:
                raise ValueError(f'Unmapped state abbreviation in {name}: {abbr!r}')
            fips, official = abbr_ok[abbr]
            as_of = date.fromisoformat(rec['Date'])
            base = list(file_flags) + timing_flags(as_of, when)
            if as_of != file_date:
                base.append('date_ne_filename')
            if legacy:
                base.append('legacy_format')
            else:
                if rec['State-Name'] != official:
                    base.append('name_mismatch')
                if rec['Currency'] != 'U.S Dollar' or rec['Unit'] != 'US Gallon':
                    base.append('unit_unexpected')
            for column, product in MASTER_PRODUCTS.items():
                price = parse_price(rec[column])
                rows.append(_row(TAG_LYK, as_of, abbr, fips, product, price, raw_file, raw_sha, sha, when,
                                 base + price_flags(product, price)))
    seen: dict[tuple, int] = defaultdict(int)
    for r in rows:
        seen[(r['state_abbr'], r['as_of_date'], r['product'])] += 1
    for r in rows:
        if seen[(r['state_abbr'], r['as_of_date'], r['product'])] > 1:
            r['flags'] = ';'.join(f for f in (r['flags'], 'duplicate_row') if f)
    return rows


def parse_lyk_rows(tmp: str, by_name: dict[str, tuple[str, str]], interim: Path | None = None) -> list[dict]:
    entries = chain_entries(manifest_path((interim or paths.interim_dir()) / 'lykmapipo_mirror.git'), LYKMAPIPO)
    if not entries:
        raise FileNotFoundError(f'No stored bundle for {SOURCE_LYK}; run `mirror` first')
    repo = _sub(tmp, 'lyk') / 'repo.git'
    restore(repo, entries, LYKMAPIPO.branch)
    copies = {c['sha256']: c['raw_file'] for e in entries for c in entry_copies(e)}
    default = (entries[-1]['raw_file'] or '', entries[-1]['raw_sha256'] or '')
    return parse_lyk(repo, lyk_history(repo), commit_bundles(repo, entries), copies, by_name, default)


def frame(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=SCHEMA).sort(['source', 'as_of_date', 'state_abbr', 'product'])


def _sub(tmp: str, name: str) -> Path:
    path = Path(tmp) / name
    path.mkdir()
    return path


def master_provenance(entries: list[dict], content: bytes) -> tuple[str, str]:
    """The stored MasterGas.csv copy if it is byte-identical to what the repo holds, else the newest bundle."""
    digest = hashlib.sha256(content).hexdigest()
    for entry in reversed(entries):
        for copy in entry_copies(entry):
            if copy['sha256'] == digest:
                return copy['raw_file'], digest
    last = next(e for e in reversed(entries) if e.get('raw_file'))
    return last['raw_file'], last['raw_sha256']


def parse_aaagas_rows(tmp: str, by_name: dict[str, tuple[str, str]], interim: Path | None = None) -> list[dict]:
    """Rows from MasterGas.csv as restored from base + incremental bundles (replayed under `tmp`)."""
    entries = chain_entries(manifest_path((interim or paths.interim_dir()) / 'aaagas_mirror.git'), AAAGAS)
    if not entries:
        raise FileNotFoundError(f'No stored bundle for {SOURCE_AAAGAS}; run `mirror` first')
    repo = _sub(tmp, 'aaagas') / 'repo.git'
    restore(repo, entries)
    spec = f'{BRANCH[SOURCE_AAAGAS]}:{MASTER_PATH}'
    content = read_blobs(repo, [spec])[spec]
    raw_file, raw_sha = master_provenance(entries, content)
    return parse_master(content, master_history(repo), raw_file, raw_sha, by_name)


def parse(out_path: Path | None = None) -> pl.DataFrame:
    by_name, by_abbr = load_states()
    with tempfile.TemporaryDirectory() as tmp:
        rows = parse_aaagas_rows(tmp, by_name)
        rows += parse_lyk_rows(tmp, by_name)
        rec = latest_record(SOURCE_GUEY)
        repo = open_bundle(rec, _sub(tmp, 'guey'))
        rows += parse_guey(repo, guey_history(repo), str(rec.path), rec.sha256, by_abbr)
    df = frame(rows)
    out_path = Path(out_path) if out_path else paths.interim_dir() / OUT_PARQUET
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    return df


def coverage(df: pl.DataFrame) -> pl.DataFrame:
    """Per source: first/last date, days present, calendar days missing inside the span, rows."""
    out = []
    for source in df['source'].unique().sort():
        days = df.filter((pl.col('source') == source) & (pl.col('product') == 'regular'))['as_of_date'].unique().sort()
        span = {days[0] + timedelta(n) for n in range((days[-1] - days[0]).days + 1)}
        out.append(dict(source=source, first=days[0], last=days[-1], n_days=len(days),
                        n_missing=len(span - set(days)), missing_days=[str(d) for d in sorted(span - set(days))],
                        n_rows=df.filter(pl.col('source') == source).height))
    return pl.DataFrame(out)


# ---------------------------------------------------------------- EIA cross-check

def _pearson(x: list[float], y: list[float]) -> float | None:
    n = len(x)
    if n < 3:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sxx, syy = sum((a - mx) ** 2 for a in x), sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(sxx * syy)


def _sd(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def _variants(prices: dict[date, float], monday: date) -> dict[str, float]:
    out = {}
    if monday in prices:
        out['same_monday'] = prices[monday]
    window = [monday - timedelta(k) for k in range(7)]   # preceding Tue .. Mon
    if all(d in prices for d in window):
        out['prev_tue_mon_mean'] = sum(prices[d] for d in window) / 7
    for k in EIA_SHIFTS:
        if monday + timedelta(k) in prices:
            out[f'monday_shift_{k:+d}d'] = prices[monday + timedelta(k)]
    return out


def eia_compare(prices: pl.DataFrame, eia: pl.DataFrame, threshold: float = EIA_THRESHOLD) -> pl.DataFrame:
    """Weekly rows and per-state summaries; `prices` is prices_state_daily, `eia` the EIA weekly table."""
    eia = eia.filter((pl.col('product') == 'regular') & pl.col('area_code').str.contains(r'^S[A-Z]{2}$')
                     & pl.col('value').is_not_null())
    states = sorted(eia['area_code'].unique().str.slice(1).to_list())
    weeks = []
    for source in sorted(prices['source'].unique().to_list()):
        for st in states:
            p = prices.filter((pl.col('source') == source) & (pl.col('state_abbr') == st)
                              & (pl.col('product') == 'regular') & (pl.col('lag_field') == 'current')
                              & pl.col('price_usd').is_not_null())
            daily = dict(zip(p['as_of_date'].to_list(), p['price_usd'].to_list()))
            if not daily:
                continue
            for monday, value in eia.filter(pl.col('area_code') == 'S' + st).select('period', 'value').iter_rows():
                for name, price in _variants(daily, monday).items():
                    weeks.append(dict(row_type='week', source=source, comparison=name, state_abbr=st,
                                      week_monday=monday, eia_price=value, aaa_price=round(price, 6),
                                      diff=round(price - value, 6), flagged=abs(round(price - value, 6)) > threshold))
    wk = pl.DataFrame(weeks)
    summaries = []
    for (source, comparison), g in wk.group_by(['source', 'comparison'], maintain_order=True):
        for st in sorted(g['state_abbr'].unique().to_list()) + ['ALL']:
            s = g if st == 'ALL' else g.filter(pl.col('state_abbr') == st)
            diffs = s['diff'].to_list()
            x, y = [], []
            for _, h in s.group_by('state_abbr'):
                m = dict(zip(h['week_monday'].to_list(), zip(h['aaa_price'].to_list(), h['eia_price'].to_list())))
                for mon, (a, e) in m.items():
                    prev = m.get(mon - timedelta(7))
                    if prev:
                        x.append(a - prev[0])
                        y.append(e - prev[1])
            summaries.append(dict(row_type='summary', source=source, comparison=comparison, state_abbr=st,
                                  n_weeks=len(diffs), mean_diff=sum(diffs) / len(diffs),
                                  mean_abs_diff=sum(abs(d) for d in diffs) / len(diffs),
                                  sd_diff=_sd(diffs), max_abs_diff=max(abs(d) for d in diffs),
                                  corr_wow=_pearson(x, y),
                                  n_flagged=int(s['flagged'].sum())))
    cols = ['row_type', 'source', 'comparison', 'state_abbr', 'week_monday', 'eia_price', 'aaa_price', 'diff',
            'flagged', 'n_weeks', 'mean_diff', 'mean_abs_diff', 'sd_diff', 'max_abs_diff', 'corr_wow', 'n_flagged']
    return pl.concat([pl.DataFrame(summaries), wk], how='diagonal').select(cols)


def eia_check(prices_path: Path | None = None, eia_path: Path | None = None,
              out_path: Path | None = None) -> pl.DataFrame:
    prices = pl.read_parquet(prices_path or paths.interim_dir() / OUT_PARQUET)
    eia = pl.read_parquet(eia_path or paths.interim_dir() / 'prices_eia_weekly.parquet')
    result = eia_compare(prices, eia)
    out_path = Path(out_path) if out_path else paths.repo_root() / 'reports' / 'validation' / OUT_EIA_CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_csv(out_path)
    return result


# ---------------------------------------------------------------- AAAGas vs lykmapipo overlap

OUT_OVERLAP_CSV = 'github_sources_overlap.csv'
OVERLAP_SHIFTS = (-2, -1, 0, 1, 2)
PRECISION_CHANGE = date(2026, 8, 5)       # AAAGas prices carry 4 decimals from this date (audit S1b)
OVERLAP_COLS = ['row_type', 'shift_days', 'period', 'state_abbr', 'n', 'exact_share', 'within_0005_share',
                'within_005_share', 'mean_diff', 'mean_abs_diff', 'max_abs_diff']


def _overlap_stats(df: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    return df.group_by(keys).agg(
        n=pl.len(),
        exact_share=(pl.col('diff').abs() < 1e-9).mean(),
        within_0005_share=(pl.col('diff').abs() <= 0.0005 + 1e-9).mean(),
        within_005_share=(pl.col('diff').abs() <= 0.005 + 1e-9).mean(),
        mean_diff=pl.col('diff').mean(), mean_abs_diff=pl.col('diff').abs().mean(),
        max_abs_diff=pl.col('diff').abs().max())


def overlap_compare(prices: pl.DataFrame) -> pl.DataFrame:
    """AAAGas (scrape date d) against lykmapipo (AAA's own as-of date d+k) for regular gasoline, k = -2..2.

    `shift` rows: all overlap days and the pre/post 2026-08-05 split. `state`, `month`, `hour` (AAAGas capture hour,
    UTC) and `day` rows follow for the best shift, plus shifts -1/0/+1 for `hour` and `day`."""
    reg = prices.filter((pl.col('product') == 'regular') & pl.col('price_usd').is_not_null()
                        & (pl.col('lag_field') == 'current'))
    a = reg.filter(pl.col('source') == TAG_AAAGAS).select(
        'state_abbr', d='as_of_date', a='price_usd', hour=pl.col('captured_at_utc').dt.hour().cast(pl.Int32))
    lyk = reg.filter(pl.col('source') == TAG_LYK).select('state_abbr', 'as_of_date', l='price_usd')
    if a.is_empty() or lyk.is_empty():
        raise ValueError('Both aaagas_github and lykmapipo_github rows are required')
    parts = []
    for k in OVERLAP_SHIFTS:
        j = a.join(lyk.with_columns(d=pl.col('as_of_date') - pl.duration(days=k)).drop('as_of_date'),
                   on=['state_abbr', 'd'], how='inner').with_columns(
            diff=pl.col('a') - pl.col('l'), shift_days=pl.lit(k, dtype=pl.Int32))
        parts.append(j)
    j = pl.concat(parts).with_columns(
        period=pl.when(pl.col('d') < PRECISION_CHANGE).then(pl.lit(f'pre_{PRECISION_CHANGE}'))
        .otherwise(pl.lit(f'post_{PRECISION_CHANGE}')), month=pl.col('d').dt.strftime('%Y-%m'))
    out = []
    shift = pl.concat([_overlap_stats(j, ['shift_days']).with_columns(period=pl.lit('all')),
                       _overlap_stats(j, ['shift_days', 'period'])], how='diagonal')
    out.append(shift.with_columns(row_type=pl.lit('shift')))
    allrows = shift.filter(pl.col('period') == 'all')
    best = allrows.sort(['exact_share', 'within_005_share'], descending=True)['shift_days'][0]
    jb = j.filter(pl.col('shift_days') == best)
    out.append(_overlap_stats(jb, ['state_abbr']).with_columns(row_type=pl.lit('state'), shift_days=pl.lit(best, dtype=pl.Int32),
                                                              period=pl.lit('all')))
    out.append(_overlap_stats(jb, ['month']).rename({'month': 'period'}).with_columns(
        row_type=pl.lit('month'), shift_days=pl.lit(best, dtype=pl.Int32)))
    near = j.filter(pl.col('shift_days').is_in([-1, 0, 1]))
    out.append(_overlap_stats(near, ['shift_days', 'hour']).with_columns(
        row_type=pl.lit('hour'), period=pl.col('hour').cast(pl.String).str.zfill(2)).drop('hour'))
    out.append(_overlap_stats(near, ['shift_days', 'd']).with_columns(
        row_type=pl.lit('day'), period=pl.col('d').cast(pl.String)).drop('d'))
    res = pl.concat([o.with_columns(pl.col('shift_days').cast(pl.Int32)) for o in out], how='diagonal')
    return res.select([pl.col(c) if c in res.columns else pl.lit(None).alias(c) for c in OVERLAP_COLS]).sort(
        ['row_type', 'shift_days', 'period', 'state_abbr'], nulls_last=True)


def overlap_check(prices_path: Path | None = None, out_path: Path | None = None) -> pl.DataFrame:
    prices = pl.read_parquet(prices_path or paths.interim_dir() / OUT_PARQUET)
    result = overlap_compare(prices)
    out_path = Path(out_path) if out_path else paths.repo_root() / 'reports' / 'validation' / OUT_OVERLAP_CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_csv(out_path)
    return result


# ---------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    command = args[0] if args else ''
    if command == 'mirror':
        try:
            results = mirror_all()
        except MirrorFailed as exc:
            for r in exc.results:
                print(json.dumps(r, default=str, sort_keys=True))
            return 1
        for r in results:
            print(json.dumps(r, default=str, sort_keys=True))
        return 0
    if command == 'restore' and len(args) == 2:
        restore(Path(args[1]), chain_entries(manifest_path(paths.interim_dir() / 'aaagas_mirror.git'), AAAGAS))
        return 0
    if command == 'parse':
        print(coverage(parse()))
        return 0
    if command == 'eia-check':
        res = eia_check()
        print(res.filter((pl.col('row_type') == 'summary')
                         & pl.col('comparison').is_in(['same_monday', 'prev_tue_mon_mean'])))
        return 0
    if command == 'overlap':
        res = overlap_check()
        print(res.filter((pl.col('row_type') == 'shift')))
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
