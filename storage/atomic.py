from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """
    Atomic file write: write temp file in same dir, fsync, then os.replace.
    """
    ensure_parent_dir(path)
    directory = str(path.parent)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=directory)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        # Best-effort fsync directory to persist rename
        try:
            dir_fd = os.open(directory, os.O_DIRECTORY)
        except Exception:
            dir_fd = None
        if dir_fd is not None:
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass


@contextmanager
def atomic_write_text(path: Path, encoding: str = "utf-8"):
    """
    Context manager that yields a temp file path to write, then atomically replaces.
    Usage:
      with atomic_write_text(path) as tmp:
          tmp.write_text("...", encoding="utf-8")
    """
    ensure_parent_dir(path)
    directory = str(path.parent)
    tmp = Path(
        tempfile.mkstemp(prefix=f".{path.name}.", dir=directory)[1]
    )
    try:
        yield tmp
        data = tmp.read_bytes()
        atomic_write_bytes(path, data)
    finally:
        try:
            tmp.unlink(missing_ok=True)  # py3.8+; safe on 3.10
        except Exception:
            pass

