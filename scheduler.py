"""
scheduler.py — Monitor files/input for new videos and process them automatically.

For every new video file found in the input directory (recursively):
  1. Run PPE detection with frame skipping
  2. Save an annotated video  → files/output/<video_stem>/<video_stem>_processed.mp4
  3. Save a CSV report        → files/output/<video_stem>/<video_stem>_report.csv
  4. Log the processed file   → logs/processed.log

Already-processed files are tracked in the log and will not be reprocessed.

Usage:
    python3 scheduler.py                          # poll every 30s (default)
    python3 scheduler.py --poll 10                # poll every 10s
    python3 scheduler.py --frame-skip 5           # process every 5th frame
    python3 scheduler.py --once                   # single scan, then exit
"""
# final code

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

import config.settings as settings
from config.settings import (
    DEFAULT_CONF,
    DEFAULT_IMGSZ,
    DEFAULT_DEVICE,
    DEFAULT_FRAME_SKIP,
    VIDEO_EXTENSIONS,
    SHOW_INFERRED_VIOLATIONS,
    FILTER_INFERRED_BY_DISPLAY_CLASSES,
    DISPLAY_CLASSES,
    DISPLAY_MISSING_LABELS,
)
from processing.worker import DETECT_IMAGE_REPORT_FIELDNAMES, csv_row_for_detect_frame
from utils.visualization import annotate_frame
from utils.compliance_logic import check_compliance_strict


def _filter_inferred_label(label: str) -> str:
    """Filter inferred no_* labels based on env settings."""
    if not SHOW_INFERRED_VIOLATIONS:
        return ""
    if not FILTER_INFERRED_BY_DISPLAY_CLASSES or not DISPLAY_CLASSES:
        return label

    parts = [p.strip() for p in label.split(",") if p.strip()]
    filtered = [p for p in parts if p in DISPLAY_MISSING_LABELS]
    return ", ".join(filtered)

# ──────────────────────────────────────────────────────────────
# Log helpers
# ──────────────────────────────────────────────────────────────

def _load_processed_set() -> set[str]:
    """Load the set of already-processed file paths from the log."""
    processed = set()
    log_file = settings.LOG_FILE
    if log_file.exists():
        with open(log_file, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Format: <timestamp> | <absolute_path>
                parts = line.split(" | ", maxsplit=1)
                if len(parts) == 2:
                    processed.add(parts[1].strip())
    return processed


def _append_log(video_path: Path):
    """Append a processed-file entry to the log."""
    log_file = settings.LOG_FILE
    log_file.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as fh:
        fh.write(f"{timestamp} | {video_path.resolve()}\n")


def _log_print(msg: str):
    """Print with a timestamp prefix."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ──────────────────────────────────────────────────────────────
# Discovery
# ──────────────────────────────────────────────────────────────

def _discover_new_videos(processed: set[str]) -> list[Path]:
    """Recursively find video files in INPUT_DIR that haven't been processed."""
    input_dir = settings.INPUT_DIR
    new_files = []
    if not input_dir.exists():
        _log_print(f"  ⚠️  Input dir does not exist: {input_dir}")
        return new_files
    
    all_files = list(input_dir.rglob("*"))
    # _log_print(f"  🔍 Scanned {len(all_files)} total items in {input_dir}")
    
    for f in sorted(all_files):
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS:
            if str(f.resolve()) not in processed:
                new_files.append(f)
    return new_files


# ──────────────────────────────────────────────────────────────
# Processing
# ──────────────────────────────────────────────────────────────

def process_video(
    model: YOLO,
    video_path: Path,
    frame_skip: int,
    conf: float,
    imgsz: int,
    device: str,
):
    """Process a single video: produce annotated video + CSV report."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        _log_print(f"  ❌ Cannot open: {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare output directory: files/output/<subdir>/<stem>/
    # Preserve input subdirectory structure to avoid collisions
    # e.g. input/AA/part1.mp4 → output/AA/part1/
    try:
        relative = video_path.resolve().relative_to(settings.INPUT_DIR.resolve())
        # relative = AA/part1.mp4 → parent = AA, stem = part1
        out_dir = settings.OUTPUT_DIR / relative.parent / video_path.stem
    except ValueError:
        # Fallback if video is not under INPUT_DIR
        out_dir = settings.OUTPUT_DIR / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    video_out_path = out_dir / f"{video_path.stem}_processed.mp4"
    csv_out_path = out_dir / f"{video_path.stem}_report.csv"

    _log_print(f"  📹 {video_path.name}  →  {out_dir}")
    _log_print(f"     FPS={fps:.1f}  Resolution={width}x{height}  Frames={total}  FrameSkip={frame_skip}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_out_path), fourcc, fps / frame_skip, (width, height))

    csv_rows: list[dict] = []
    frame_idx = 0
    processed_count = 0

    # Keep track if we saw any violations across the video
    total_violations = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            results = model.predict(
                source=frame,
                conf=conf,
                imgsz=imgsz,
                device=device if device else None,
                verbose=False,
            )
            result = results[0]

            # Use STRICT LOGIC to separate detections and find violations
            summary, annotations = check_compliance_strict(result)

            csv_rows.append(
                csv_row_for_detect_frame(frame_idx, frame_skip, fps, result, summary)
            )

            # Annotate and write frame
            annotated = annotate_frame(frame, result, conf_threshold=conf)

            # ---------------------------------------------------------
            # DRAW INFERRED VIOLATIONS (Missing PPE)
            # ---------------------------------------------------------
            for ann in annotations:
                # ann = {'box': [x1, y1, x2, y2], 'label': 'no_helmet', 'color': (0,0,220)}
                filtered_label = _filter_inferred_label(ann["label"])
                if not filtered_label:
                    continue
                x1, y1, x2, y2 = map(int, ann["box"])
                color = ann["color"]
                label = filtered_label
                
                # Draw box (maybe dashed or thinner to distinguish? or just red)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # Draw Label
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                # Place label just below the top line of the box, or above if space
                cv2.rectangle(annotated, (x1, y1), (x1 + tw + 4, y1 + th + 4), color, -1)
                cv2.putText(
                    annotated, label, (x1 + 2, y1 + th),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
                )
            # ---------------------------------------------------------

            # Simple Text Overlay for Violation status on video
            status_text = "COMPLIANT" if summary["is_compliant"] else "VIOLATION"
            status_color = (0, 200, 0) if summary["is_compliant"] else (0, 0, 255)

            # Add top-left status
            cv2.putText(annotated, f"Status: {status_text}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

            if not summary["is_compliant"]:
                total_violations += 1
                # List missing items
                missing_labels = list(summary["missing_ppe"].keys())
                if SHOW_INFERRED_VIOLATIONS:
                    if FILTER_INFERRED_BY_DISPLAY_CLASSES and DISPLAY_CLASSES:
                        missing_labels = [x for x in missing_labels if x in DISPLAY_MISSING_LABELS]
                    if missing_labels:
                        missing_str = ", ".join(missing_labels)
                        cv2.putText(annotated, f"Missing: {missing_str}", (20, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            writer.write(annotated)
            processed_count += 1

            # Progress every 50 processed frames
            if processed_count % 50 == 0:
                _log_print(f"     Processed {processed_count} frames (frame #{frame_idx}/{total})")

        frame_idx += 1

    cap.release()
    writer.release()

    try:
        with open(csv_out_path, "w", newline="") as fh:
            csv_writer = csv.DictWriter(fh, fieldnames=DETECT_IMAGE_REPORT_FIELDNAMES)
            csv_writer.writeheader()
            csv_writer.writerows(csv_rows)
    except PermissionError:
        _log_print(f"❌ Permission Error writing CSV to {csv_out_path}. File might be open.")
        return False

    _log_print(f"  ✅ Done — {processed_count} frames processed")
    _log_print(f"     Video → {video_out_path}")
    _log_print(f"     CSV   → {csv_out_path}  ({len(csv_rows)} rows)")
    
    if total_violations > 0:
        _log_print(f"     ⚠️  CAUTION: {total_violations} frames had PPE violations!")

    return True


# ──────────────────────────────────────────────────────────────
# Scheduler loop
# ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scheduler: watch files/input for new videos and process them",
    )
    parser.add_argument("--poll", type=int, default=30,
                        help="Polling interval in seconds (default: 30)")
    parser.add_argument("--frame-skip", type=int, default=DEFAULT_FRAME_SKIP,
                        help=f"Process every Nth frame (default: {DEFAULT_FRAME_SKIP})")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF,
                        help=f"Confidence threshold (default: {DEFAULT_CONF})")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ,
                        help=f"Inference image size (default: {DEFAULT_IMGSZ})")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE,
                        help="Device for inference")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model weights (.pt)")
    parser.add_argument("--once", action="store_true",
                        help="Run a single scan and exit (no polling)")
    return parser.parse_args()


def _resolve_model(model_arg: str | None) -> str:
    """Resolve model path: CLI arg → default (best.pt if present, else pretrained name)."""
    if model_arg:
        return model_arg
    from config.settings import default_model_path

    return default_model_path()


def main():
    args = parse_args()
    model_path = _resolve_model(args.model)

    # Ensure directories exist
    settings.INPUT_DIR.mkdir(parents=True, exist_ok=True)
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    settings.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    _log_print("=" * 60)
    _log_print("  PPE Detection — Scheduler")
    _log_print("=" * 60)
    _log_print(f"  Model       : {model_path}")
    _log_print(f"  Input Dir   : {settings.INPUT_DIR}")
    _log_print(f"  Output Dir  : {settings.OUTPUT_DIR}")
    _log_print(f"  Log File    : {settings.LOG_FILE}")
    _log_print(f"  Frame Skip  : {args.frame_skip}")
    _log_print(f"  Confidence  : {args.conf}")
    _log_print(f"  Poll Interval: {args.poll}s")
    _log_print("=" * 60)

    # Load model once
    _log_print("Loading model...")
    model = YOLO(model_path)
    _log_print("Model loaded ✅")

    while True:
        processed_set = _load_processed_set()
        new_videos = _discover_new_videos(processed_set)

        if new_videos:
            _log_print(f"📂 Found {len(new_videos)} new video(s)")
            for video_path in new_videos:
                _log_print(f"\n▶ Processing: {video_path}")
                success = process_video(
                    model=model,
                    video_path=video_path,
                    frame_skip=args.frame_skip,
                    conf=args.conf,
                    imgsz=args.imgsz,
                    device=args.device,
                )
                if success:
                    _append_log(video_path)
        else:
            _log_print("💤 No new videos found.")

        if args.once:
            _log_print("Single scan complete (--once). Exiting.")
            break

        _log_print(f"⏳ Next scan in {args.poll}s...")
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
