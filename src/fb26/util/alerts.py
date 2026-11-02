from datetime import datetime, timezone
import fcntl
import logging
import os
import shutil
import subprocess

from .paths import logs_dir


def send_alert(subject: str, body: str, level: str = 'error') -> None:
    path = logs_dir() / 'alerts.log'
    path.parent.mkdir(parents=True, exist_ok=True)
    message = f'{datetime.now(timezone.utc).isoformat()} [{level}] {subject}: {body}\n'
    with path.open('a', encoding='utf-8') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(message)
        handle.flush()
        fcntl.flock(handle, fcntl.LOCK_UN)
    if shutil.which('osascript'):
        script = ('on run argv\n display notification (item 2 of argv) '
                  'with title (item 1 of argv)\nend run')
        try:
            subprocess.run(['osascript', '-e', script, subject, body], check=True,
                           capture_output=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            logging.getLogger(__name__).warning('macOS notification could not be delivered')
    # PI channel hook: deliberately a no-op until the channel and transport
    # are selected and implemented, even when ALERT_CHANNEL is populated.
    _channel = os.getenv('ALERT_CHANNEL', '')


def check_row_count(job: str, n: int, expected: int, tol: float = 0.10) -> None:
    if expected < 0 or tol < 0:
        raise ValueError('expected and tol must be nonnegative')
    if (expected == 0 and n != 0) or (expected > 0 and abs(n - expected) / expected > tol):
        send_alert(f'Row count: {job}', f'Observed {n}; expected {expected}; tolerance {tol:.0%}')
