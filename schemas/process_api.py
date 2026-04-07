"""Request/response models for ``POST /process/single``."""

from __future__ import annotations

from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from config.settings import DEFAULT_MODEL
from schemas.registry_api import processing_overrides_from_parts
from storage.config_io import DEFAULT_CONFIG

_F_SKIP = int(DEFAULT_CONFIG["video"]["frame_skip"])

PROCESS_SINGLE_OPENAPI_EXAMPLES: dict[str, dict[str, Any]] = {
    "full": {
        "summary": "Paths + processing (incl. fps)",
        "description": (
            "Optional processing keys match register-folder (`model_path`, `frame_skip`, `output_fps`, "
            "`display_classes`). Omit any for global /config. `conf`, `imgsz`, `iou`, etc. → PATCH `/config`."
        ),
        "value": {
            "input_file_path": "files/input/siteA/capture1.png",
            "output_path": "files/output/siteA/capture1_annotated.png",
            "model_path": DEFAULT_MODEL,
            "frame_skip": _F_SKIP,
            "output_fps": 2.0,
            "display_classes": ["helmet", "vest", "goggles"],
        },
    },
    "paths_only": {
        "summary": "Input file path only (output auto-derived)",
        "description": "If output_path is omitted, it is derived from GET /config defaults.",
        "value": {
            "input_file_path": "files/input/test.mp4",
        },
    },
}


class ProcessSingleRequest(BaseModel):
    """Process one file; optional processing fields match ``RegisterFolderBody`` (except paths are file paths)."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        json_schema_extra={
            "example": PROCESS_SINGLE_OPENAPI_EXAMPLES["full"]["value"],
        },
    )

    input_file_path: str = Field(description="Absolute or resolvable path to one image or video file.")
    output_path: str | None = Field(
        default=None,
        description=(
            "Optional. If omitted, derived from GET /config `default_output_path` and input name. "
            "Video → `<default_output>/videos/<rel>/<stem>/<stem>_annotated.mp4`. "
            "Image → `<default_output>/images/<rel>/<stem>/annotated.<image.format>`."
        ),
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
            f"Video only: detection every Nth frame; omit for global `video.frame_skip` ({_F_SKIP})."
        ),
    )
    fps: float | None = Field(
        default=None,
        gt=0,
        validation_alias=AliasChoices("fps", "output_fps"),
        serialization_alias="fps",
        description=(
            "Video only: encoded output MP4 FPS. Omit uses source_fps / frame_skip. "
            "Alias `output_fps` accepted for compatibility."
        ),
    )
    display_classes: list[str] | None = Field(
        default=None,
        description=(
            "Non-empty: draw only these YOLO classes (+ Person) and filter CSV ``raw_detections`` to them; "
            "also narrows strict violation overlay text when `filter_inferred_by_display_classes` is true. "
            "Empty list: show every class (and match `/detect/*`-style full output). Omit = use `/config`."
        ),
    )

    def to_processing_overrides(self) -> dict[str, Any]:
        return processing_overrides_from_parts(
            model_path=self.model_path,
            frame_skip=self.frame_skip,
            output_fps=self.fps,
            display_classes=self.display_classes,
        )


class ProcessSingleResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    media_type: str
    input_file_path: str
    output_path: str
    report_path: str | None = None
    summary: dict | None = None
    error: str | None = None
