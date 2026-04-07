from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from config.settings import (
    DEFAULT_CONF,
    DEFAULT_FRAME_SKIP,
    DEFAULT_IMGSZ,
    DEFAULT_IOU,
    DEFAULT_MODEL,
    VIDEO_EXTENSIONS,
)
from storage.atomic import atomic_write_bytes, ensure_parent_dir
from storage.lock import file_lock
from storage.paths import config_path, lock_path, state_dir

# Defaults for state/config.yaml; inference numbers match config.settings.
DEFAULT_CONFIG: dict[str, Any] = {
    "default_input_path": None,
    "default_output_path": None,
    "poll_interval_seconds": 30,
    "model_path": DEFAULT_MODEL,
    "device": "",
    "conf": DEFAULT_CONF,
    "imgsz": DEFAULT_IMGSZ,
    "iou": DEFAULT_IOU,
    "video": {"frame_skip": DEFAULT_FRAME_SKIP},
    "image": {"format": "jpg"},
    "supported_media": {
        "video_exts": sorted(VIDEO_EXTENSIONS),
        "image_exts": [".jpg", ".jpeg", ".png", ".bmp", ".webp"],
    },
    "display_classes": [],
    "strict_overlays": {
        "show_inferred_violations": True,
        "filter_inferred_by_display_classes": True,
    },
}


def ensure_state_layout() -> None:
    (state_dir() / "registry").mkdir(parents=True, exist_ok=True)
    (state_dir() / "locks").mkdir(parents=True, exist_ok=True)
    ensure_parent_dir(config_path())


def load_config() -> dict[str, Any]:
    ensure_state_layout()
    path = config_path()
    with file_lock(lock_path("config")):
        if not path.exists():
            return dict(DEFAULT_CONFIG)
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            return dict(DEFAULT_CONFIG)
        merged = dict(DEFAULT_CONFIG)
        merged.update(data)
        return merged


def save_config(cfg: dict[str, Any]) -> None:
    ensure_state_layout()
    body = yaml.safe_dump(cfg, sort_keys=False).encode("utf-8")
    with file_lock(lock_path("config")):
        atomic_write_bytes(config_path(), body)

