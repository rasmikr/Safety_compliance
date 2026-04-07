from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def state_dir() -> Path:
    # Keep env-only to avoid circular imports with config/settings.py
    # Default is <repo>/state
    raw = os.getenv("STATE_DIR", "").strip()
    if raw:
        return Path(raw).expanduser()
    return project_root() / "state"


def config_path() -> Path:
    return state_dir() / "config.yaml"


def registry_dir() -> Path:
    return state_dir() / "registry"


def folders_registry_path() -> Path:
    return registry_dir() / "folders.csv"


def files_registry_path() -> Path:
    return registry_dir() / "files.csv"


def locks_dir() -> Path:
    return state_dir() / "locks"


def lock_path(name: str) -> Path:
    safe = "".join(c if c.isalnum() or c in {"-", "_", "."} else "_" for c in name)
    return locks_dir() / f"{safe}.lock"

