"""
Central project settings for the PPE Detection project.
"""

import os
from pathlib import Path


def _env_csv(key: str, default: list[str]) -> list[str]:
    raw = os.getenv(key, "")
    if not raw.strip():
        return default
    return [x.strip() for x in raw.split(",") if x.strip()]


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATASET_YAML = CONFIG_DIR / "dataset.yaml"
RUNS_DIR = PROJECT_ROOT / "runs"

# Fine-tuned PPE weights (11 classes) after `yolo train ...`; used as default when present.
PPE_BEST_WEIGHTS = RUNS_DIR / "detect" / "ppe_detection" / "weights" / "best.pt"
# Ultralytics pretrained checkpoint name when no fine-tuned `best.pt` is on disk.
_FALLBACK_PRETRAINED = "yolo26n.pt"


def default_model_path() -> str:
    """Prefer on-disk PPE ``best.pt``; otherwise return downloadable pretrained name."""
    if PPE_BEST_WEIGHTS.is_file():
        return str(PPE_BEST_WEIGHTS.resolve())
    return _FALLBACK_PRETRAINED


# Scheduler directories
INPUT_DIR = PROJECT_ROOT / "files" / "input"
OUTPUT_DIR = PROJECT_ROOT / "files" / "output"
LOG_FILE = PROJECT_ROOT / "logs" / "processed.log"

# Supported video extensions
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

# ──────────────────────────────────────────────
# Model Defaults
# ──────────────────────────────────────────────
DEFAULT_MODEL = default_model_path()  # best.pt when trained weights exist; else yolo26n.pt
DEFAULT_EPOCHS = 100
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 16
DEFAULT_CONF = 0.25                # Confidence threshold for inference
DEFAULT_IOU = 0.45                 # IoU threshold for NMS
DEFAULT_DEVICE = ""                # "" = auto-detect (CUDA if available, else CPU)
DEFAULT_FRAME_SKIP = 10            # Process every Nth frame for video inference
DEFAULT_PROJECT = str(RUNS_DIR / "detect")
DEFAULT_NAME = "ppe_detection"

# ──────────────────────────────────────────────
# Class Definitions
# ──────────────────────────────────────────────
CLASS_NAMES = {
    0: "helmet",
    1: "gloves",
    2: "vest",
    3: "boots",
    4: "goggles",
    5: "none",
    6: "Person",
    7: "no_helmet",
    8: "no_goggle",
    9: "no_gloves",
    10: "no_boots",
}

# Optional display filtering for overlays:
# DISPLAY_CLASSES="" (or unset) => show all
# DISPLAY_CLASSES="helmet,boots" => only show selected classes and their no_* overlays
DISPLAY_CLASSES = _env_csv("DISPLAY_CLASSES", [])
_CLASS_NAME_TO_ID = {name.lower(): cid for cid, name in CLASS_NAMES.items()}
DISPLAY_CLASS_IDS = {
    _CLASS_NAME_TO_ID[name.lower()]
    for name in DISPLAY_CLASSES
    if name.lower() in _CLASS_NAME_TO_ID
}

# Inferred (strict) violation overlay controls.
'''
show_inferred_violations
----------------------------------
If true (default): draw the inferred violation overlays (e.g. no_helmet) when compliance logic marks them.  
If false: those inferred violation overlays are hidden (you may still see raw detection boxes depending on display_classes / visualization).

filter_inferred_by_display_classes
----------------------------------
Only matters when display_classes is non-empty.  

If true: among inferred violation labels, only show those that match your chosen display classes’ mapped no_* labels (same idea as DISPLAY_MISSING_LABELS in settings).  
If false: inferred violation text is not filtered that way (broader labeling on screen).
'''
SHOW_INFERRED_VIOLATIONS = _env_bool("SHOW_INFERRED_VIOLATIONS", True)
FILTER_INFERRED_BY_DISPLAY_CLASSES = _env_bool("FILTER_INFERRED_BY_DISPLAY_CLASSES", True)

# Map selected wearable PPE to inferred violation labels.
DISPLAY_TO_MISSING_LABEL = {
    "helmet": "no_helmet",
    "gloves": "no_gloves",
    "boots": "no_boots",
    "vest": "no_vest",
}
DISPLAY_MISSING_LABELS = {
    DISPLAY_TO_MISSING_LABEL[name.lower()]
    for name in DISPLAY_CLASSES
    if name.lower() in DISPLAY_TO_MISSING_LABEL
}

# Classes that represent WORN PPE (compliant)
WORN_PPE_CLASSES = {0, 1, 2, 3, 4}  # helmet, gloves, vest, boots, goggles

# Classes that represent MISSING PPE (non-compliant / violation)
MISSING_PPE_CLASSES = {7, 8, 9, 10}  # no_helmet, no_goggle, no_gloves, no_boots

# Neutral classes
NEUTRAL_CLASSES = {5, 6}  # none, Person

# ──────────────────────────────────────────────
# Visualization Colors  (BGR format for OpenCV)
# ──────────────────────────────────────────────
COLOR_COMPLIANT = (0, 200, 0)       # Green  — worn PPE
COLOR_VIOLATION = (0, 0, 220)       # Red    — missing PPE
COLOR_NEUTRAL = (200, 180, 0)       # Cyan   — Person / none
