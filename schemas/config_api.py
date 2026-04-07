"""
OpenAPI-friendly config models for /config endpoints.

PATCH semantics: only fields you send are applied (merged into existing YAML).
Omitted fields leave the stored config unchanged.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from config.settings import DEFAULT_IOU, DEFAULT_MODEL
from storage.config_io import DEFAULT_CONFIG


class VideoConfigPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_skip: int | None = Field(
        None,
        ge=1,
        description="For video processing: run detection every Nth frame.",
        examples=[10],
    )
    output_fps: float | None = Field(
        None,
        gt=0,
        description=(
            "Annotated output video FPS when set; otherwise source_fps / frame_skip. "
            "Registry / process/single can override per folder or per request."
        ),
    )


class ImageConfigPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: str | None = Field(
        None,
        description="Annotated image output format (e.g. jpg, png).",
        examples=["jpg"],
    )


class SupportedMediaPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video_exts: list[str] | None = Field(
        None,
        description="File extensions (lowercase, with dot) treated as video.",
    )
    image_exts: list[str] | None = Field(
        None,
        description="File extensions (lowercase, with dot) treated as image.",
    )


class StrictOverlaysPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    show_inferred_violations: bool | None = Field(
        None,
        description="If false, do not draw heuristic no_* violation overlays.",
    )
    filter_inferred_by_display_classes: bool | None = Field(
        None,
        description="If true with display_classes set, only show no_* labels matching those classes.",
    )


class ConfigInitOrUpdateBody(BaseModel):
    """
    Partial update body for `state/config.yaml`.

    Programmer defaults (when file missing) match `storage.config_io.DEFAULT_CONFIG`.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                # Same layout as code defaults (Docker can use absolute paths like /data/inbox instead).
                "default_input_path": "files/input",
                "default_output_path": "files/output",
                "poll_interval_seconds": 30,
                "model_path": "runs/detect/ppe_detection/weights/best.pt",
                "device": "",
                "conf": 0.25,
                "imgsz": 640,
                "iou": 0.45,
                "video": {"frame_skip": 10},
                "image": {"format": "jpg"},
                "supported_media": {
                    "video_exts": [".mp4", ".avi", ".mov", ".mkv"],
                    "image_exts": [".jpg", ".jpeg", ".png"],
                },
                "display_classes": ["helmet", "boots"],
                "strict_overlays": {
                    "show_inferred_violations": True,
                    "filter_inferred_by_display_classes": True,
                },
            }
        },
    )

    default_input_path: str | None = Field(
        None,
        description=(
            "Default watch folder when registry has no active folders (used with default_output_path). "
            "Empty/unset falls back to code paths (e.g. files/input)."
        ),
    )
    default_output_path: str | None = Field(
        None,
        description="Default output root paired with default_input_path.",
    )
    poll_interval_seconds: int | None = Field(
        None,
        ge=1,
        description="Background processor poll interval in seconds.",
        examples=[DEFAULT_CONFIG["poll_interval_seconds"]],
    )
    model_path: str | None = Field(
        None,
        description=(
            "YOLO ``.pt`` path or pretrained name. Omit to use defaults: "
            "``runs/detect/ppe_detection/weights/best.pt`` when present, else ``yolo26n.pt``."
        ),
        examples=[DEFAULT_CONFIG["model_path"]],
    )
    device: str | None = Field(
        None,
        description="CUDA device id, 'cpu', or empty string for auto.",
        examples=[""],
    )
    conf: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Detection confidence threshold.",
        examples=[DEFAULT_CONFIG["conf"]],
    )
    imgsz: int | None = Field(
        None,
        ge=32,
        description="Inference image size (square).",
        examples=[DEFAULT_CONFIG["imgsz"]],
    )
    iou: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="YOLO NMS IoU.",
        examples=[DEFAULT_IOU],
    )
    video: VideoConfigPatch | None = Field(
        None,
        description="Video-specific options.",
    )
    image: ImageConfigPatch | None = Field(
        None,
        description="Image-specific options.",
    )
    supported_media: SupportedMediaPatch | None = Field(
        None,
        description="Which extensions count as video vs image.",
    )
    display_classes: list[str] | None = Field(
        None,
        description=(
            "Non-empty: processor draws only these classes (+ Person) and filters CSV `raw_detections`. "
            "`[]`: all classes. With `filter_inferred_by_display_classes`, also limits no_* overlay text."
        ),
    )
    strict_overlays: StrictOverlaysPatch | None = Field(
        None,
        description="Controls heuristic no_* overlays.",
    )


class StoredConfigResponse(BaseModel):
    """Full merged config as returned by GET /config and POST /config/init_or_update."""

    model_config = ConfigDict(extra="allow")

    default_input_path: str | None = None
    default_output_path: str | None = None
    poll_interval_seconds: int = 30
    model_path: str = DEFAULT_MODEL
    device: str = ""
    conf: float = 0.25
    imgsz: int = 640
    iou: float = DEFAULT_IOU
    video: dict = Field(default_factory=dict)
    image: dict = Field(default_factory=dict)
    supported_media: dict = Field(default_factory=dict)
    display_classes: list[str] = Field(default_factory=list)
    strict_overlays: dict = Field(default_factory=dict)


def patch_to_merge_dict(body: ConfigInitOrUpdateBody) -> dict:
    """Convert PATCH body to a plain dict; omit unset keys for merge."""
    data = body.model_dump(exclude_none=True)
    # Nested models become dicts automatically
    return data
