from pathlib import Path
import os


def repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / 'pyproject.toml').is_file():
            return parent
    raise RuntimeError('Cannot locate repository root (pyproject.toml)')


def data_dir() -> Path:
    override = os.getenv('FB26_DATA_DIR')
    return Path(override).expanduser().resolve() if override else repo_root() / 'data'


def raw_dir() -> Path:
    return data_dir() / 'raw'


def interim_dir() -> Path:
    return data_dir() / 'interim'


def processed_dir() -> Path:
    return data_dir() / 'processed'


def logs_dir() -> Path:
    return repo_root() / 'logs'


def config_dir() -> Path:
    return repo_root() / 'config'
