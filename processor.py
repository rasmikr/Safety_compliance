#!/usr/bin/env python3
"""
Poll active folders in the registry (includes builtin files/input → files/output), detect new
media, write under ``<output>/videos/...`` and ``<output>/images/...``, append ``files.csv``.

Logs: same timestamped lines go to stdout and to ``config.settings.LOG_FILE`` (default
``logs/processed.log``; Docker often sets this to ``<output>/processed.log``).

Usage:
  python3 processor.py                  # poll forever (interval from state/config.yaml)
  python3 processor.py --once           # one scan then exit
  python3 processor.py --model /path   # override model_path for this process
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

import config.settings as settings
from processing.worker import (
    ProcessingRuntime,
    merge_effective_config,
    get_model_path_for_config,
    media_suffix_sets,
    output_dir_for_media,
    process_image,
    process_video,
    resolve_watch_pairs,
)
from storage.config_io import DEFAULT_CONFIG, ensure_state_layout, load_config
from storage.registry_io import (
    append_file_record,
    ensure_registry_layout,
    is_file_already_processed_success,
)

_cached_model_path: str | None = None
_cached_model: YOLO | None = None


def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        path = settings.LOG_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def _get_model(cfg: dict) -> YOLO:
    global _cached_model_path, _cached_model
    path = get_model_path_for_config(cfg)
    if _cached_model is None or _cached_model_path != path:
        _log(f"Loading model: {path}")
        _cached_model = YOLO(path)
        _cached_model_path = path
    return _cached_model


def run_scan(cfg: dict) -> None:
    pairs = resolve_watch_pairs(cfg)
    for in_root, out_root, folder_id, folder_proc in pairs:
        eff = merge_effective_config(cfg, folder_proc)
        model = _get_model(eff)
        ve, ie = media_suffix_sets(eff)
        vid_cfg = eff.get("video") or {}
        frame_skip = int(vid_cfg.get("frame_skip", DEFAULT_CONFIG["video"]["frame_skip"]))
        _ofps = vid_cfg.get("output_fps")
        output_fps = float(_ofps) if _ofps is not None else None
        conf = float(eff.get("conf", DEFAULT_CONFIG["conf"]))
        imgsz = int(eff.get("imgsz", DEFAULT_CONFIG["imgsz"]))
        device = str(eff.get("device", "") or "")
        iou = float(eff.get("iou", DEFAULT_CONFIG["iou"]))
        rt = ProcessingRuntime.from_effective_config(eff)
        if not in_root.exists():
            _log(f"No input directory (skipped): {in_root}")
            continue
        for f in sorted(in_root.rglob("*")):
            if not f.is_file():
                continue
            suf = f.suffix.lower()
            if suf not in ve and suf not in ie:
                continue
            abs_s = str(f.resolve())
            try:
                mtime_ns = f.stat().st_mtime_ns
            except OSError as e:
                _log(f"stat failed {f}: {e}")
                continue
            if is_file_already_processed_success(abs_s, mtime_ns):
                continue

            _log(f"Processing {f}")
            if suf in ve:
                out_dir = output_dir_for_media(in_root, out_root, f, "video")
                out_vid = out_dir / f"{f.stem}_processed.mp4"
                report = out_dir / f"{f.stem}_report.csv"
                ok, err = process_video(
                    model,
                    f,
                    out_video_path=out_vid,
                    report_csv_path=report,
                    frame_skip=frame_skip,
                    conf=conf,
                    imgsz=imgsz,
                    device=device,
                    iou=iou,
                    rt=rt,
                    output_fps=output_fps,
                )
                if ok:
                    append_file_record(abs_s, mtime_ns, folder_id, str(out_vid.resolve()), "success", "")
                    _log(f"  → {out_vid}")
                else:
                    append_file_record(abs_s, mtime_ns, folder_id, "", "failed", err)
                    _log(f"  failed: {err}")
            else:
                img_cfg = eff.get("image") or {}
                fmt = str(img_cfg.get("format", "jpg")).lower().lstrip(".")
                out_dir = output_dir_for_media(in_root, out_root, f, "image")
                out_img = out_dir / f"annotated.{fmt}"
                ok, err, _meta = process_image(
                    model,
                    f,
                    out_image_path=out_img,
                    conf=conf,
                    imgsz=imgsz,
                    device=device,
                    iou=iou,
                    rt=rt,
                )
                if ok:
                    append_file_record(abs_s, mtime_ns, folder_id, str(out_img.resolve()), "success", "")
                    _log(f"  → {out_img}")
                else:
                    append_file_record(abs_s, mtime_ns, folder_id, "", "failed", err)
                    _log(f"  failed: {err}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPE registry processor loop")
    p.add_argument("--once", action="store_true", help="Run one scan and exit")
    p.add_argument("--model", type=str, default=None, help="Override config model_path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_state_layout()
    ensure_registry_layout()

    _log("PPE processor started")
    _log(f"Log file: {settings.LOG_FILE.resolve()}")
    _log(f"Code default I/O (if unused): {settings.INPUT_DIR} → {settings.OUTPUT_DIR}")

    while True:
        cfg = load_config()
        if args.model:
            cfg = dict(cfg)
            cfg["model_path"] = args.model
        try:
            run_scan(cfg)
        except Exception as e:
            _log(f"Scan error: {e!r}")

        if args.once:
            _log("Single scan complete (--once).")
            break

        interval = float(cfg.get("poll_interval_seconds", DEFAULT_CONFIG["poll_interval_seconds"]))
        _log(f"Sleep {interval}s until next scan")
        time.sleep(interval)


if __name__ == "__main__":
    main()
