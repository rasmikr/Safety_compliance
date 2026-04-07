from __future__ import annotations

import csv
import json
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

from storage.atomic import atomic_write_bytes
from storage.lock import file_lock
from storage.paths import (
    files_registry_path,
    folders_registry_path,
    lock_path,
    registry_dir,
)


def parse_folder_processing_json(raw: str | None) -> dict[str, Any]:
    if not raw or not str(raw).strip():
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def folder_row_for_api(row: dict[str, str]) -> dict[str, Any]:
    proc = parse_folder_processing_json(row.get("processing"))
    return {
        "folder_id": row.get("folder_id", ""),
        "input_path": row.get("input_path", ""),
        "output_path": row.get("output_path", ""),
        "status": row.get("status", ""),
        "created_at": row.get("created_at", ""),
        "updated_at": row.get("updated_at", ""),
        "processing": proc if proc else None,
    }


def _normalize_folder_row(r: dict[str, str]) -> dict[str, str]:
    return {
        "folder_id": r.get("folder_id", ""),
        "input_path": r.get("input_path", ""),
        "output_path": r.get("output_path", ""),
        "status": r.get("status", ""),
        "created_at": r.get("created_at", ""),
        "updated_at": r.get("updated_at", ""),
        "processing": r.get("processing", ""),
    }


FOLDERS_HEADER = [
    "folder_id",
    "input_path",
    "output_path",
    "status",  # active/inactive
    "created_at",
    "updated_at",
    "processing",  # JSON object: per-folder overrides merged over global config
]

# Stable id for the auto-seeded ``settings.INPUT_DIR`` / ``OUTPUT_DIR`` pair (until unregistered).
BUILTIN_INBOX_FOLDER_ID = "__builtin_inbox__"

# Rows created by ``POST /process/single`` (not folder-scanned).
API_SINGLE_FILE_FOLDER_ID = "__api_single__"

FILES_HEADER = [
    "file_path",
    "mtime_ns",
    "folder_id",
    "processed_at",
    "output_artifact_path",
    "status",  # success/failed
    "error",
]


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_registry_layout() -> None:
    registry_dir().mkdir(parents=True, exist_ok=True)
    if not folders_registry_path().exists():
        # Write directly to avoid recursion (write_folders calls ensure_registry_layout)
        atomic_write_bytes(folders_registry_path(), (",".join(FOLDERS_HEADER) + "\n").encode("utf-8"))
    if not files_registry_path().exists():
        atomic_write_bytes(files_registry_path(), (",".join(FILES_HEADER) + "\n").encode("utf-8"))


def _load_folders_table() -> list[dict[str, str]]:
    with folders_registry_path().open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return [_normalize_folder_row({k: (v or "") for k, v in row.items()}) for row in reader if row]


def _save_folders_table(rows: Iterable[dict[str, Any]]) -> None:
    lines = [",".join(FOLDERS_HEADER) + "\n"]
    for r in rows:
        vals = [str(r.get(h, "")) for h in FOLDERS_HEADER]
        lines.append(",".join(_csv_escape(vals)) + "\n")
    atomic_write_bytes(folders_registry_path(), "".join(lines).encode("utf-8"))


def read_folders() -> list[dict[str, str]]:
    import config.settings as settings

    ensure_registry_layout()
    default_in = str(settings.INPUT_DIR.resolve())
    default_out = str(settings.OUTPUT_DIR.resolve())

    with file_lock(lock_path("folders_registry")):
        rows = _load_folders_table()
        if not any(r.get("input_path") == default_in for r in rows):
            now = _now_iso()
            rows.append(
                {
                    "folder_id": BUILTIN_INBOX_FOLDER_ID,
                    "input_path": default_in,
                    "output_path": default_out,
                    "status": "active",
                    "created_at": now,
                    "updated_at": now,
                    "processing": "",
                }
            )
            _save_folders_table(rows)
        return rows


def write_folders(rows: Iterable[dict[str, Any]]) -> None:
    ensure_registry_layout()
    with file_lock(lock_path("folders_registry")):
        _save_folders_table(rows)


def register_folder(
    input_path: str,
    output_path: str,
    *,
    processing: dict[str, Any] | None = None,
) -> dict[str, str]:
    proc_cell = (
        json.dumps(processing, sort_keys=True, separators=(",", ":")) if processing else ""
    )
    rows = read_folders()
    now = _now_iso()
    for r in rows:
        if r.get("input_path") == input_path:
            r["output_path"] = output_path
            r["processing"] = proc_cell
            r["status"] = "active"
            r["updated_at"] = now
            write_folders(rows)
            return _normalize_folder_row(r)
    new_row = {
        "folder_id": str(uuid.uuid4()),
        "input_path": input_path,
        "output_path": output_path,
        "status": "active",
        "created_at": now,
        "updated_at": now,
        "processing": proc_cell,
    }
    rows.append(new_row)
    write_folders(rows)
    return _normalize_folder_row(new_row)


def unregister_folder(input_path: str) -> bool:
    rows = read_folders()
    now = _now_iso()
    changed = False
    for r in rows:
        if r.get("input_path") == input_path and r.get("status") != "inactive":
            r["status"] = "inactive"
            r["updated_at"] = now
            changed = True
    if changed:
        write_folders(rows)
    return changed


def read_files() -> list[dict[str, str]]:
    ensure_registry_layout()
    with file_lock(lock_path("files_registry")):
        path = files_registry_path()
        with path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows: list[dict[str, str]] = []
            for row in reader:
                if not row:
                    continue
                rows.append({k: (v or "") for k, v in row.items()})
            return rows


def write_files(rows: Iterable[dict[str, Any]]) -> None:
    ensure_registry_layout()
    with file_lock(lock_path("files_registry")):
        out = []
        out.append(",".join(FILES_HEADER) + "\n")
        for r in rows:
            line = []
            for h in FILES_HEADER:
                val = r.get(h, "")
                line.append(str(val))
            out.append(",".join(_csv_escape(line)) + "\n")
        atomic_write_bytes(files_registry_path(), "".join(out).encode("utf-8"))


def is_file_already_processed_success(file_path: str, mtime_ns: int) -> bool:
    """True if files.csv has a success row for this path and mtime (skip unchanged re-runs)."""
    mkey = str(mtime_ns)
    for r in read_files():
        if r.get("file_path") == file_path and r.get("mtime_ns") == mkey and r.get("status") == "success":
            return True
    return False


def append_file_record(
    file_path: str,
    mtime_ns: int,
    folder_id: str,
    output_artifact_path: str,
    status: str,
    error: str = "",
) -> None:
    rows = list(read_files())
    rows.append(
        {
            "file_path": file_path,
            "mtime_ns": str(mtime_ns),
            "folder_id": folder_id,
            "processed_at": _now_iso(),
            "output_artifact_path": output_artifact_path,
            "status": status,
            "error": error,
        }
    )
    write_files(rows)


def _csv_escape(values: list[str]) -> list[str]:
    # Very small CSV escaper (we write simple CSV without relying on writer to preserve atomicity).
    out: list[str] = []
    for v in values:
        if any(c in v for c in [",", "\"", "\n", "\r"]):
            out.append("\"" + v.replace("\"", "\"\"") + "\"")
        else:
            out.append(v)
    return out

