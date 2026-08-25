"""Pydantic v2 configuration models for the MOT counting pipeline.

Every pipeline parameter is defined here.  Frame-dimension validation and
class-name-vs-model validation are intentionally absent — those checks run
later, once the video is opened (§7.3) and the YOLO model is loaded (§4.1),
respectively.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class VideoConfig(BaseModel):
    """Video I/O configuration.

    Attributes:
        path: Path to the input video file (relative or absolute).
        output_dir: Directory where all pipeline outputs are written.
    """

    path: str = Field(..., description="Path to the input video file.")
    output_dir: str = Field(..., description="Directory for all pipeline outputs.")


class DetectionConfig(BaseModel):
    """Object detector configuration.

    Attributes:
        model_variant: YOLO26 variant identifier.  Valid values:
            ``yolo26n``, ``yolo26s``, ``yolo26m``, ``yolo26l``, ``yolo26x``.
        imgsz: Inference image size in pixels.  Must be a positive integer;
            typically a multiple of 32 (e.g. 640).
        confidence_threshold: Global minimum detector confidence in the range
            ``(0, 1]``.  Detections below this value are discarded before
            tracking.
        classes: List of class-name strings to keep (e.g. ``["person", "car"]``).
            Validated against the loaded model's ``model.names`` mapping in the
            composition root (§4.1, §7.1) — **not** here.
    """

    model_variant: str = Field(
        "yolo26m",
        description="YOLO26 variant: yolo26n | yolo26s | yolo26m | yolo26l | yolo26x.",
    )
    imgsz: int = Field(640, description="Inference image size in pixels (positive integer).")
    confidence_threshold: float = Field(
        0.4,
        description="Global confidence threshold in (0, 1].",
    )
    classes: list[str] = Field(
        default_factory=lambda: ["person", "car"],
        description="Class names to detect.  Validated against the loaded model at startup.",
    )

    @field_validator("confidence_threshold")
    @classmethod
    def _validate_confidence(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError(f"confidence_threshold must be in (0, 1], got {v!r}.")
        return v

    @field_validator("imgsz")
    @classmethod
    def _validate_imgsz(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"imgsz must be a positive integer, got {v!r}.")
        return v


class TrackerConfig(BaseModel):
    """Multi-object tracker configuration.

    Attributes:
        type: Tracker implementation to use.  ``"bytetrack"`` is the primary
            production path; ``"botsort"`` is a stretch-goal alternative.
        track_thresh: Detection confidence threshold for initialising new
            tracks.
        match_thresh: IoU threshold used for track–detection association.
        track_buffer: Number of frames a lost track is kept alive before it
            is removed.
    """

    type: Literal["bytetrack", "botsort"] = Field(
        "bytetrack",
        description="Tracker type: bytetrack | botsort.",
    )
    track_thresh: float = Field(0.5, description="Confidence threshold for new tracks.")
    match_thresh: float = Field(0.8, description="IoU threshold for track–detection association.")
    track_buffer: int = Field(30, description="Frames to keep a lost track alive.")


class LineConfig(BaseModel):
    """Configuration for a single virtual counting line.

    Attributes:
        line_id: Unique string identifier for this line, used in event logs
            and visualizations.
        point_a: ``[x, y]`` pixel coordinate of the first endpoint of the
            line, in absolute pixel units matching the source video resolution.
        point_b: ``[x, y]`` pixel coordinate of the second endpoint of the
            line, in absolute pixel units matching the source video resolution.
        positive_direction: Which crossing direction is considered the
            positive (IN) direction relative to the A→B line vector.
            ``"A_to_B"`` means crossing from the left/below side of A→B counts
            as IN; ``"B_to_A"`` inverts this.
    """

    line_id: str = Field(..., description="Unique identifier for this counting line.")
    point_a: list[int] = Field(
        ...,
        description="[x, y] pixel coordinate of endpoint A (absolute pixels).",
    )
    point_b: list[int] = Field(
        ...,
        description="[x, y] pixel coordinate of endpoint B (absolute pixels).",
    )
    positive_direction: Literal["A_to_B", "B_to_A"] = Field(
        "A_to_B",
        description="Crossing direction treated as IN: A_to_B | B_to_A.",
    )

    @field_validator("point_a", "point_b")
    @classmethod
    def _validate_point(cls, v: list[int]) -> list[int]:
        if len(v) != 2:  # noqa: PLR2004
            raise ValueError(f"Each point must be [x, y] (2 integers), got {v!r}.")
        return v


class CrossingLogicConfig(BaseModel):
    """Crossing-logic state-machine configuration.

    Attributes:
        reference_point: Which point on the bounding box is used for
            side-of-line computation.  ``"bottom_center"`` (default) is more
            stable for pedestrians and vehicles.
        history_length: Number of frames in the raw side-history sliding
            window.  Must be ≥ 1.
        confirmation_majority_threshold: Fraction of the history window that
            must vote for the new side before a crossing is confirmed.  Must be
            in ``(0.5, 1.0]`` to guarantee a decisive majority.
        cooldown_seconds: Minimum time in seconds between two successive
            crossing events for the same ``(track_id, line_id)`` pair.  Must
            be ≥ 0.
        stale_track_timeout_seconds: Seconds of inactivity after which a
            ``(track_id, line_id)`` state entry is purged.
        min_displacement_px: Optional minimum Euclidean displacement (pixels)
            of the reference point since the pair's last event.  ``None``
            disables the check.
        min_velocity_px_per_s: Optional minimum velocity component
            perpendicular to the line (pixels per second).  ``None`` disables
            the check.
    """

    reference_point: Literal["bottom_center", "box_center"] = Field(
        "bottom_center",
        description="Reference point used for side computation: bottom_center | box_center.",
    )
    history_length: int = Field(
        8,
        description="Raw side-history window length (≥ 1 frames).",
    )
    confirmation_majority_threshold: float = Field(
        0.7,
        description="Fraction of window required to confirm a crossing, in (0.5, 1.0].",
    )
    cooldown_seconds: float = Field(
        1.5,
        description="Minimum seconds between successive events per (track_id, line_id) pair (≥ 0).",
    )
    stale_track_timeout_seconds: float = Field(
        2.0,
        description="Seconds of inactivity before a (track_id, line_id) state entry is removed.",
    )
    min_displacement_px: float | None = Field(
        None,
        description="Minimum reference-point displacement (pixels) since last event.  null = disabled.",
    )
    min_velocity_px_per_s: float | None = Field(
        None,
        description="Minimum velocity perpendicular to the line (px/s).  null = disabled.",
    )

    @field_validator("history_length")
    @classmethod
    def _validate_history_length(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"history_length must be ≥ 1, got {v!r}.")
        return v

    @field_validator("confirmation_majority_threshold")
    @classmethod
    def _validate_majority_threshold(cls, v: float) -> float:
        if not (0.5 < v <= 1.0):
            raise ValueError(f"confirmation_majority_threshold must be in (0.5, 1.0], got {v!r}.")
        return v

    @field_validator("cooldown_seconds")
    @classmethod
    def _validate_cooldown(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"cooldown_seconds must be ≥ 0, got {v!r}.")
        return v


class EventsConfig(BaseModel):
    """Event persistence configuration.

    Attributes:
        output_csv: Path to the output CSV file where crossing events are
            written by the ``IEventRepository`` implementation.
    """

    output_csv: str = Field(..., description="Path for the output crossing-events CSV file.")


class EvaluationConfig(BaseModel):
    """Evaluation configuration.

    Attributes:
        matching_tolerance_seconds: Maximum time difference (in seconds)
            between a predicted event and a ground-truth event for them to be
            considered a match during bipartite evaluation.
    """

    matching_tolerance_seconds: float = Field(
        1.0,
        description="Temporal tolerance (seconds) for GT–prediction event matching.",
    )


class VisualizationConfig(BaseModel):
    """Video annotation and visualization configuration.

    Attributes:
        output_video: Path for the annotated output video file.
        draw_trails: Whether to draw short trajectory trails on tracked
            objects.  Disabled by default (stretch goal).
    """

    output_video: str = Field(..., description="Path for the annotated output video.")
    draw_trails: bool = Field(False, description="Draw short trajectory trails on tracked objects.")


# ---------------------------------------------------------------------------
# Root model
# ---------------------------------------------------------------------------


class AppConfig(BaseModel):
    """Top-level application configuration loaded from a YAML file.

    Attributes:
        video: Video I/O settings.
        detection: Detector and class-filter settings.
        tracker: Tracker hyperparameters.
        lines: Non-empty list of virtual counting lines, each with a unique
            ``line_id``.
        crossing_logic: Crossing-logic state-machine parameters.
        events: Event persistence settings.
        evaluation: Evaluation matching parameters.
        visualization: Output video and drawing settings.
    """

    video: VideoConfig
    detection: DetectionConfig
    tracker: TrackerConfig
    lines: list[LineConfig] = Field(..., description="One or more virtual counting lines.")
    crossing_logic: CrossingLogicConfig
    events: EventsConfig
    evaluation: EvaluationConfig
    visualization: VisualizationConfig

    @field_validator("lines")
    @classmethod
    def _validate_lines_nonempty(cls, v: list[LineConfig]) -> list[LineConfig]:
        if not v:
            raise ValueError("lines must contain at least one counting line.")
        return v

    @model_validator(mode="after")
    def _validate_unique_line_ids(self) -> AppConfig:
        ids = [line.line_id for line in self.lines]
        seen: set[str] = set()
        duplicates: set[str] = set()
        for lid in ids:
            if lid in seen:
                duplicates.add(lid)
            seen.add(lid)
        if duplicates:
            raise ValueError(
                f"Duplicate line_id values detected: {sorted(duplicates)!r}.  "
                "Each counting line must have a unique line_id."
            )
        return self


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_config(path: str) -> AppConfig:
    """Load and validate a YAML configuration file.

    Relative paths in ``path`` are resolved from the current working directory
    (callers should ``os.chdir`` to the project root first, or pass an
    absolute path).

    Args:
        path: Filesystem path to the YAML configuration file.

    Returns:
        A fully-validated :class:`AppConfig` instance.

    Raises:
        FileNotFoundError: If the file does not exist at ``path``.
        ValueError: If the file cannot be parsed as valid YAML.
        pydantic.ValidationError: If the parsed YAML does not satisfy the
            schema (e.g. invalid field values, duplicate line IDs).
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path.resolve()!s}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML configuration at {config_path!s}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(
            f"Configuration file must contain a YAML mapping at the top level, "
            f"got {type(raw).__name__!r}: {config_path!s}"
        )

    return AppConfig.model_validate(raw)
