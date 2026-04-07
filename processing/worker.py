from __future__ import annotations

import csv
import base64
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from ultralytics import YOLO

from config.settings import (
    CLASS_NAMES,
    DEFAULT_PROJECT,
    DISPLAY_TO_MISSING_LABEL,
    default_model_path,
)
from storage.config_io import DEFAULT_CONFIG
from storage.registry_io import parse_folder_processing_json, read_folders
from utils.compliance_logic import check_compliance_strict
from utils.visualization import annotate_frame, resolve_class_label


@dataclass(frozen=True)
class ProcessingRuntime:
    show_inferred_violations: bool
    filter_inferred_by_display_classes: bool
    display_classes: tuple[str, ...]
    display_missing_labels: frozenset[str]

    @staticmethod
    def from_effective_config(effective: dict[str, Any]) -> "ProcessingRuntime":
        dc = effective.get("display_classes") or []
        if not isinstance(dc, list):
            dc = []
        names = tuple(str(x).strip().lower() for x in dc if str(x).strip())
        missing = frozenset(
            DISPLAY_TO_MISSING_LABEL[n]
            for n in names
            if n in DISPLAY_TO_MISSING_LABEL
        )
        so = effective.get("strict_overlays") or {}
        if not isinstance(so, dict):
            so = {}
        show = bool(so.get("show_inferred_violations", DEFAULT_CONFIG["strict_overlays"]["show_inferred_violations"]))
        filt = bool(
            so.get(
                "filter_inferred_by_display_classes",
                DEFAULT_CONFIG["strict_overlays"]["filter_inferred_by_display_classes"],
            )
        )
        return ProcessingRuntime(
            show_inferred_violations=show,
            filter_inferred_by_display_classes=filt,
            display_classes=names,
            display_missing_labels=missing,
        )


def get_model_path_for_config(effective: dict[str, Any]) -> str:
    mp = effective.get("model_path")
    if mp and str(mp).strip():
        p = Path(str(mp).strip()).expanduser()
        if p.is_file():
            return str(p.resolve())
        return str(p)
    return default_model_path()


def allowed_class_ids_from_display_names(names: tuple[str, ...]) -> list[int]:
    """Map display class names to YOLO ids (worn + dataset ``no_*`` ids)."""
    if not names:
        return []
    _map = {name.lower(): cid for cid, name in CLASS_NAMES.items()}
    out: list[int] = []
    for n in names:
        nl = str(n).strip().lower()
        if nl in _map:
            out.append(_map[nl])
        miss = DISPLAY_TO_MISSING_LABEL.get(nl)
        if miss and miss in _map:
            out.append(_map[miss])
    seen: set[int] = set()
    uniq: list[int] = []
    for i in out:
        if i not in seen:
            seen.add(i)
            uniq.append(i)
    return uniq


def _annotate_allowed_class_ids(rt: ProcessingRuntime) -> list[int]:
    """Empty ``display_classes``: draw all boxes (``[]`` in ``annotate_frame``). Else whitelist + Person."""
    if not rt.display_classes:
        return []
    ids = allowed_class_ids_from_display_names(tuple(rt.display_classes))
    if 6 not in ids:
        ids = ids + [6]
    return ids


def _csv_raw_class_filter(rt: ProcessingRuntime) -> frozenset[int] | None:
    """Restrict ``raw_detections`` column to these ids; ``None`` = list every box."""
    if not rt.display_classes:
        return None
    ids = list(allowed_class_ids_from_display_names(tuple(rt.display_classes)))
    if 6 not in ids:
        ids.append(6)
    return frozenset(ids)


def _filter_inferred_label(label: str, rt: ProcessingRuntime) -> str:
    if not rt.show_inferred_violations:
        return ""
    if not rt.filter_inferred_by_display_classes or not rt.display_classes:
        return label
    parts = [p.strip() for p in label.split(",") if p.strip()]
    filtered = [p for p in parts if p in rt.display_missing_labels]
    return ", ".join(filtered)


def _build_csv_row(frame_no: int, frame_skip: int, fps: float, result, summary: dict) -> dict:
    persons = []
    if result.boxes is not None:
        for box in result.boxes:
            if int(box.cls[0]) == 6:
                persons.append(f"Person:{float(box.conf[0]):.3f}")
    worn_strings = []
    for name, count in summary["worn_ppe"].items():
        worn_strings.append(name if count == 1 else f"{name}({count})")
    missing_strings = []
    for name, count in summary["missing_ppe"].items():
        missing_strings.append(name if count == 1 else f"{name}({count})")
    return {
        "frame_no": frame_no,
        "frame_skip_rate": frame_skip,
        "video_fps": round(fps, 2),
        "person_count": summary["total_persons"],
        "person_details": "; ".join(persons) if persons else "",
        "worn_ppe": "; ".join(worn_strings) if worn_strings else "",
        "missing_ppe": "; ".join(missing_strings) if missing_strings else "",
    }


DETECT_IMAGE_REPORT_FIELDNAMES = [
    "frame_no",
    "frame_skip_rate",
    "video_fps",
    "person_count",
    "person_details",
    "worn_ppe",
    "missing_ppe",
    "is_compliant",
    "raw_detections",
]


def csv_row_for_detect_frame(
    frame_no: int,
    frame_skip: int,
    fps: float,
    result,
    summary: dict,
    *,
    raw_class_ids_filter: frozenset[int] | None = None,
) -> dict:
    """One CSV row for API detect reports (per frame for video, or frame 0 for image)."""
    row = _build_csv_row(frame_no, frame_skip, fps, result, summary)
    row["is_compliant"] = "yes" if summary.get("is_compliant") else "no"
    raw_parts: list[str] = []
    if result.boxes is not None:
        for box in result.boxes:
            cid = int(box.cls[0])
            if raw_class_ids_filter is not None and cid not in raw_class_ids_filter:
                continue
            name = resolve_class_label(result, cid)
            raw_parts.append(f"{name}:{float(box.conf[0]):.3f}")
    row["raw_detections"] = "; ".join(raw_parts)
    return row


def csv_row_for_detect_image(result, summary: dict) -> dict:
    """One CSV row for a single-frame API image report."""
    return csv_row_for_detect_frame(0, 1, 0.0, result, summary)


def _missing_strings_from_summary(summary: dict, rt: ProcessingRuntime) -> list[str]:
    missing_labels = list(summary["missing_ppe"].keys())
    if not rt.show_inferred_violations:
        return []
    if rt.filter_inferred_by_display_classes and rt.display_classes:
        missing_labels = [x for x in missing_labels if x in rt.display_missing_labels]
    return missing_labels


def _draw_annotation_overlays(annotated: np.ndarray, annotations: list, rt: ProcessingRuntime) -> None:
    for ann in annotations:
        filtered_label = _filter_inferred_label(ann["label"], rt)
        if not filtered_label:
            continue
        x1, y1, x2, y2 = map(int, ann["box"])
        color = ann["color"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(filtered_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1), (x1 + tw + 4, y1 + th + 4), color, -1)
        cv2.putText(
            annotated,
            filtered_label,
            (x1 + 2, y1 + th),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def resolve_watch_pairs(
    cfg: dict[str, Any],
) -> list[tuple[Path, Path, str, dict[str, Any]]]:
    rows = read_folders()
    active = [r for r in rows if r.get("status") == "active"]
    pairs: list[tuple[Path, Path, str, dict[str, Any]]] = []
    for r in active:
        proc = parse_folder_processing_json(r.get("processing"))
        pairs.append(
            (
                Path(r["input_path"]).expanduser().resolve(),
                Path(r["output_path"]).expanduser().resolve(),
                r.get("folder_id", ""),
                proc,
            )
        )
    if pairs:
        return pairs
    din = cfg.get("default_input_path")
    dout = cfg.get("default_output_path")
    if din and dout:
        pairs.append(
            (
                Path(str(din)).expanduser().resolve(),
                Path(str(dout)).expanduser().resolve(),
                "",
                {},
            )
        )
    return pairs


def output_dir_for_media(
    watch_input: Path,
    output_base: Path,
    media_path: Path,
    kind: Literal["video", "image"],
) -> Path:
    try:
        rel = media_path.resolve().relative_to(watch_input.resolve())
    except ValueError:
        rel = Path(media_path.name)
    stem = media_path.stem
    parent = rel.parent
    sub = "videos" if kind == "video" else "images"
    return (output_base / sub / parent / stem).resolve()


def process_video(
    model: YOLO,
    video_path: Path,
    *,
    out_video_path: Path,
    report_csv_path: Path,
    frame_skip: int,
    conf: float,
    imgsz: int,
    device: str,
    iou: float,
    rt: ProcessingRuntime,
    output_fps: float | None = None,
) -> tuple[bool, str]:
    draw_ids = _annotate_allowed_class_ids(rt)
    csv_raw_filter = _csv_raw_class_filter(rt)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, "cannot_open_video"

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_video_path.parent.mkdir(parents=True, exist_ok=True)
    report_csv_path.parent.mkdir(parents=True, exist_ok=True)

    target_suffix = out_video_path.suffix.lower()
    # OpenCV can't encode WebM reliably; for .webm we write a temp .mp4 then transcode with ffmpeg.
    write_path = out_video_path
    needs_ffmpeg_webm = target_suffix == ".webm"
    if needs_ffmpeg_webm:
        write_path = out_video_path.with_name(out_video_path.stem + "_tmp.mp4")

    # Select a reasonable OpenCV codec/container pair.
    # Note: this is about the intermediate file if we later transcode to webm.
    fourcc = cv2.VideoWriter_fourcc(*("XVID" if write_path.suffix.lower() == ".avi" else "mp4v"))
    playback_fps = (
        float(output_fps)
        if output_fps is not None and float(output_fps) > 0
        else (fps / frame_skip)
    )
    writer = cv2.VideoWriter(str(write_path), fourcc, playback_fps, (width, height))

    csv_rows: list[dict] = []
    frame_idx = 0
    processed_count = 0
    total_violations = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip == 0:
                results = model.predict(
                    source=frame,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    device=device if device else None,
                    verbose=False,
                )
                result = results[0]
                summary, annotations = check_compliance_strict(result)
                csv_rows.append(
                    csv_row_for_detect_frame(
                        frame_idx,
                        frame_skip,
                        fps,
                        result,
                        summary,
                        raw_class_ids_filter=csv_raw_filter,
                    )
                )

                # display_classes empty → draw all; non-empty → only those classes + Person (see _annotate_allowed_class_ids).
                annotated = annotate_frame(
                    frame,
                    result,
                    conf_threshold=conf,
                    allowed_class_ids=draw_ids,
                )

                _draw_annotation_overlays(annotated, annotations, rt)

                status_text = "COMPLIANT" if summary["is_compliant"] else "VIOLATION"
                status_color = (0, 200, 0) if summary["is_compliant"] else (0, 0, 255)
                cv2.putText(
                    annotated,
                    f"Status: {status_text}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    status_color,
                    2,
                )
                if not summary["is_compliant"]:
                    total_violations += 1
                    missing_labels = _missing_strings_from_summary(summary, rt)
                    if missing_labels:
                        missing_str = ", ".join(missing_labels)
                        cv2.putText(
                            annotated,
                            f"Missing: {missing_str}",
                            (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                        )
                writer.write(annotated)
                processed_count += 1
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    if needs_ffmpeg_webm:
        try:
            # VP9/Opus gives good compatibility. "-y" overwrites requested output path.
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(write_path),
                    "-c:v",
                    "libvpx-vp9",
                    "-crf",
                    "32",
                    "-b:v",
                    "0",
                    "-c:a",
                    "libopus",
                    str(out_video_path),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            return False, "ffmpeg_not_installed_for_webm_output"
        except subprocess.CalledProcessError:
            return False, "ffmpeg_transcode_failed_webm_output"
        finally:
            try:
                write_path.unlink(missing_ok=True)
            except OSError:
                pass

    if processed_count == 0:
        return False, "no_frames_processed"

    try:
        with open(report_csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=DETECT_IMAGE_REPORT_FIELDNAMES)
            w.writeheader()
            w.writerows(csv_rows)
    except OSError as e:
        return False, f"csv_write_failed:{e}"

    _ = (total, total_violations)  # could log
    return True, ""


def process_image(
    model: YOLO,
    image_path: Path,
    *,
    out_image_path: Path,
    conf: float,
    imgsz: int,
    device: str,
    iou: float,
    rt: ProcessingRuntime,
) -> tuple[bool, str, dict[str, Any]]:
    def _parse_svg_render_size(svg: bytes, *, default_width_px: int = 640) -> tuple[int, int]:
        """
        Decide a rasterization size for an SVG.

        This matters for SVGs that use normalized units like viewBox="0 0 1 1" and width/height="100%",
        where rasterizers may otherwise emit a 1x1 PNG.
        """
        text = svg.decode("utf-8", errors="ignore")

        # viewBox gives us aspect ratio even when width/height are percentages.
        vb = re.search(r'viewBox\s*=\s*"[^"]*"', text, flags=re.IGNORECASE)
        vb_w = vb_h = None
        if vb:
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", vb.group(0))
            if len(nums) >= 4:
                try:
                    vb_w = float(nums[2])
                    vb_h = float(nums[3])
                except ValueError:
                    vb_w = vb_h = None

        def _len_to_px(val: str) -> float | None:
            v = val.strip().lower()
            if not v or v.endswith("%"):
                return None
            m = re.match(r"^([-+]?\d*\.?\d+)([a-z]+)?$", v)
            if not m:
                return None
            num = float(m.group(1))
            unit = (m.group(2) or "px").lower()
            # Best-effort conversions. For our use, the exact DPI doesn't matter much.
            if unit == "px":
                return num
            if unit == "pt":
                return num * (96.0 / 72.0)
            if unit == "in":
                return num * 96.0
            if unit == "cm":
                return num * (96.0 / 2.54)
            if unit == "mm":
                return num * (96.0 / 25.4)
            return None

        w_attr = re.search(r'\bwidth\s*=\s*"([^"]+)"', text, flags=re.IGNORECASE)
        h_attr = re.search(r'\bheight\s*=\s*"([^"]+)"', text, flags=re.IGNORECASE)
        w_px = _len_to_px(w_attr.group(1)) if w_attr else None
        h_px = _len_to_px(h_attr.group(1)) if h_attr else None

        if w_px and h_px and w_px > 1 and h_px > 1:
            return int(round(w_px)), int(round(h_px))

        # Derive from aspect ratio + default width.
        ratio = 1.0
        if vb_w and vb_h and vb_w > 0 and vb_h > 0:
            ratio = vb_w / vb_h
        width = max(2, int(default_width_px))
        height = max(2, int(round(width / ratio))) if ratio > 0 else width
        # Avoid absurd sizes if viewBox is weird.
        height = min(height, 4096)
        width = min(width, 4096)
        return width, height

    def _imread_any(path: Path) -> np.ndarray | None:
        suf = path.suffix.lower()
        if suf != ".svg":
            return cv2.imread(str(path))

        try:
            svg_bytes = path.read_bytes()
        except OSError:
            return None

        # Prefer Python rasterization if available (works without extra binaries).
        try:
            import cairosvg  # type: ignore

            out_w, out_h = _parse_svg_render_size(svg_bytes)
            # Render onto a white background so "black-only" SVGs (e.g. potrace output)
            # don't become an all-black image when alpha is dropped.
            png_bytes = cairosvg.svg2png(
                bytestring=svg_bytes,
                background_color="white",
                output_width=out_w,
                output_height=out_h,
            )
            buf = np.frombuffer(png_bytes, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
            if img is None:
                return None
            if img.ndim == 3 and img.shape[2] == 4:
                # Composite onto white.
                bgr = img[:, :, :3].astype(np.float32)
                a = (img[:, :, 3:4].astype(np.float32) / 255.0)
                out = bgr * a + 255.0 * (1.0 - a)
                return out.astype(np.uint8)
            return img
        except Exception:
            pass

        # Fallback to system binary if present (librsvg).
        try:
            out_w, out_h = _parse_svg_render_size(svg_bytes)
            p = subprocess.run(
                ["rsvg-convert", "-f", "png", "-w", str(out_w), "-h", str(out_h), str(path)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            buf = np.frombuffer(p.stdout, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
            if img is None:
                return None
            if img.ndim == 3 and img.shape[2] == 4:
                bgr = img[:, :, :3].astype(np.float32)
                a = (img[:, :, 3:4].astype(np.float32) / 255.0)
                out = bgr * a + 255.0 * (1.0 - a)
                return out.astype(np.uint8)
            return img
        except Exception:
            return None

    def _write_svg_with_embedded_png(path: Path, bgr: np.ndarray) -> bool:
        ok, png = cv2.imencode(".png", bgr)
        if not ok:
            return False
        b64 = base64.b64encode(png.tobytes()).decode("ascii")
        h, w = bgr.shape[:2]
        svg = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n'
            f'  <rect width="{w}" height="{h}" x="0" y="0" fill="white"/>\n'
            f'  <image width="{w}" height="{h}" x="0" y="0" '
            f'xlink:href="data:image/png;base64,{b64}"/>\n'
            "</svg>\n"
        )
        try:
            path.write_text(svg, encoding="utf-8")
            return True
        except OSError:
            return False

    frame = _imread_any(image_path)
    if frame is None:
        return False, "cannot_read_image", {}

    results = model.predict(
        source=frame,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device if device else None,
        verbose=False,
    )
    result = results[0]
    summary, annotations = check_compliance_strict(result)

    draw_ids = _annotate_allowed_class_ids(rt)
    annotated = annotate_frame(
        frame,
        result,
        conf_threshold=conf,
        allowed_class_ids=draw_ids,
    )

    _draw_annotation_overlays(annotated, annotations, rt)

    out_image_path.parent.mkdir(parents=True, exist_ok=True)
    if out_image_path.suffix.lower() == ".svg":
        ok = _write_svg_with_embedded_png(out_image_path, annotated)
    else:
        ok = cv2.imwrite(str(out_image_path), annotated)
    if not ok:
        return False, "imwrite_failed", {}

    meta = {
        "is_compliant": summary["is_compliant"],
        "total_persons": summary["total_persons"],
        "worn_ppe": summary["worn_ppe"],
        "missing_ppe": summary["missing_ppe"],
    }
    return True, "", meta


_NESTED_KEYS = frozenset({"strict_overlays", "video", "image", "supported_media"})


def merge_effective_config(cfg: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    if not overrides:
        return dict(cfg)
    out = dict(cfg)
    for k, v in overrides.items():
        if v is None:
            continue
        if k in _NESTED_KEYS and isinstance(v, dict):
            merged = dict(out.get(k) or {})
            merged.update({sk: sv for sk, sv in v.items() if sv is not None})
            out[k] = merged
        else:
            out[k] = v
    return out


def media_suffix_sets(cfg: dict[str, Any]) -> tuple[set[str], set[str]]:
    sm = cfg.get("supported_media") or DEFAULT_CONFIG["supported_media"]
    ve = {str(x).lower() for x in (sm.get("video_exts") or [])}
    ie = {str(x).lower() for x in (sm.get("image_exts") or [])}
    return ve, ie


def process_single_media(
    model: YOLO,
    *,
    input_file_path: Path,
    output_path: Path,
    cfg: dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Process one file; ``output_path`` is the annotated media file to write (e.g. ``.mp4`` or ``.jpg``)."""
    effective = merge_effective_config(cfg, overrides)
    rt = ProcessingRuntime.from_effective_config(effective)
    ve, ie = media_suffix_sets(effective)

    conf = float(effective.get("conf", DEFAULT_CONFIG["conf"]))
    imgsz = int(effective.get("imgsz", DEFAULT_CONFIG["imgsz"]))
    device = str(effective.get("device", "") or "")
    iou = float(effective.get("iou", DEFAULT_CONFIG["iou"]))
    vid_cfg = effective.get("video") or {}
    frame_skip = int(vid_cfg.get("frame_skip", DEFAULT_CONFIG["video"]["frame_skip"]))
    _ofps = vid_cfg.get("output_fps")
    output_fps = float(_ofps) if _ofps is not None else None
    img_cfg = effective.get("image") or {}
    img_format = str(img_cfg.get("format", "jpg")).lower().lstrip(".")

    inp = input_file_path.expanduser().resolve()
    out = output_path.expanduser().resolve()
    if not inp.is_file():
        return {
            "status": "failed",
            "media_type": "unknown",
            "input_file_path": str(inp),
            "output_path": str(out),
            "report_path": None,
            "summary": None,
            "error": "input_not_found",
        }

    suf = inp.suffix.lower()
    if suf in ve:
        if frame_skip < 1:
            return {
                "status": "failed",
                "media_type": "video",
                "input_file_path": str(inp),
                "output_path": str(out),
                "report_path": None,
                "summary": None,
                "error": "invalid_frame_skip",
            }
        report_path = out.with_name(out.stem + "_report.csv")
        ok, err = process_video(
            model,
            inp,
            out_video_path=out,
            report_csv_path=report_path,
            frame_skip=frame_skip,
            conf=conf,
            imgsz=imgsz,
            device=device,
            iou=iou,
            rt=rt,
            output_fps=output_fps,
        )
        return {
            "status": "success" if ok else "failed",
            "media_type": "video",
            "input_file_path": str(inp),
            "output_path": str(out),
            "report_path": str(report_path) if ok else None,
            "summary": None,
            "error": err or None,
        }
    if suf in ie:
        ok, err, meta = process_image(
            model,
            inp,
            out_image_path=out,
            conf=conf,
            imgsz=imgsz,
            device=device,
            iou=iou,
            rt=rt,
        )
        return {
            "status": "success" if ok else "failed",
            "media_type": "image",
            "input_file_path": str(inp),
            "output_path": str(out),
            "report_path": None,
            "summary": meta if ok else None,
            "error": err or None,
        }
    return {
        "status": "failed",
        "media_type": "unknown",
        "input_file_path": str(inp),
        "output_path": str(out),
        "report_path": None,
        "summary": None,
        "error": "unsupported_media_type",
    }
