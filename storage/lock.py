from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import fcntl

from storage.atomic import ensure_parent_dir


@contextmanager
def file_lock(lock_file: Path):
    """
    Cross-process advisory lock using flock (Linux).
    Holds lock until context exits.
    """
    ensure_parent_dir(lock_file)
    with open(lock_file, "a+") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass

