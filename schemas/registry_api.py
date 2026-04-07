"""JSON bodies for folder registry endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from config.settings import DEFAULT_MODEL
from storage.config_io import DEFAULT_CONFIG

_fs = int(DEFAULT_CONFIG["video"]["frame_skip"])

# OpenAPI request samples (used by FastAPI Body(openapi_examples=...)).
REGISTER_FOLDER_OPENAPI_EXAMPLES: dict[str, dict[str, Any]] = {
    "full": {
        "summary": "Default sample (tenant layout + fps)",
        "description": (
            "`fps` sets encoded MP4 playback speed (optional). Omit `fps` → use source_fps/frame_skip. "
            "Inference tuning (`conf`, `imgsz`, `iou`, …) is PATCH `/config` only."
        ),
        "value": {
            "input_path": "files/input",
            "output_path": "files/output",
            "model_path": DEFAULT_MODEL,
            "frame_skip": _fs,
            "fps": 2.0,
            "display_classes": ["helmet", "vest", "goggles"],
        },
    },
    "paths_only": {
        "summary": "Paths only (defaults from /config)",
        "description": "All processing fields fall back to GET /config.",
        "value": {
            "input_path": "files/input/siteA/cam01",
            "output_path": "files/output/siteA/cam01",
        },
    },
}


class RegisterFolderBody(BaseModel):
    """Register or update a watch folder. Optional processing fields default to global `/config` when omitted."""

    # Do not set JSON Schema `examples` here — Swagger UI picks that array over `example`
    # and shows the minimal sample first. Use Body(openapi_examples=...) on the route instead.
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        json_schema_extra={
            "example": REGISTER_FOLDER_OPENAPI_EXAMPLES["full"]["value"],
        },
    )

    input_path: str = Field(description="Filesystem path to watch (scanned recursively).")
    output_path: str | None = Field(
        None,
        description="Output root for this folder; if omitted, uses config `default_output_path` or `files/output`.",
    )
    model_path: str | None = Field(
        default=None,
        description=(
            f"YOLO weights; omit to use global `model_path` from `/config` "
            f"(code default name: {DEFAULT_MODEL!r})."
        ),
    )
    frame_skip: int | None = Field(
        default=None,
        ge=1,
        description=(
            f"Run video detection every Nth frame; omit to use global `video.frame_skip` ({_fs})."
        ),
    )
    fps: float | None = Field(
        default=None,
        gt=0,
        validation_alias=AliasChoices("fps", "output_fps"),
        serialization_alias="fps",
        description=(
            "Annotated output video playback FPS (encoded MP4 frame rate). Omit to use "
            "source_fps / frame_skip. Alias `output_fps` is accepted for compatibility."
        ),
    )
    display_classes: list[str] | None = Field(
        default=None,
        description=(
            "Non-empty: annotated media and CSV ``raw_detections`` only include these classes plus Person; "
            "also filters strict overlay labels when `filter_inferred_by_display_classes` is true. "
            "`[]` = show all classes. Omit = use global `/config`."
        ),
    )


class UnregisterFolderBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_path: str = Field(description="Must match the registered folder input_path (resolved path).")


def processing_overrides_from_parts(
    *,
    model_path: str | None = None,
    frame_skip: int | None = None,
    output_fps: float | None = None,
    display_classes: list[str] | None = None,
) -> dict[str, Any]:
    """Folder `processing` JSON / `process/single` overrides; omit keys = use global `/config`.

    Only: model_path, video.frame_skip, video.output_fps, display_classes.
    Other inference settings stay on global config (PATCH /config).
    """
    out: dict[str, Any] = {}
    if model_path is not None:
        out["model_path"] = model_path
    vid: dict[str, Any] = {}
    if frame_skip is not None:
        vid["frame_skip"] = frame_skip
    if output_fps is not None:
        vid["output_fps"] = output_fps
    if vid:
        out["video"] = vid
    if display_classes is not None:
        out["display_classes"] = list(display_classes)
    return out


def register_folder_processing_from_body(body: RegisterFolderBody) -> dict[str, Any]:
    """Stored as JSON on the folder row; only non-None fields (omit = use global defaults)."""
    return processing_overrides_from_parts(
        model_path=body.model_path,
        frame_skip=body.frame_skip,
        output_fps=body.fps,
        display_classes=body.display_classes,
    )
