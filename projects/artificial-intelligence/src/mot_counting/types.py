"""Core domain types for detection, tracking, and crossing events."""

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class Detection:
    """Single-frame object detection produced by the detector.

    Attributes:
        xyxy: Axis-aligned bounding box as ``(x1, y1, x2, y2)`` in pixel
            coordinates, where ``(x1, y1)`` is the top-left corner and
            ``(x2, y2)`` is the bottom-right corner.
        confidence: Detector confidence score in the range ``[0.0, 1.0]``.
        class_id: Integer class identifier from the detector's class mapping.
        class_name: Human-readable class label (for example, ``"person"``).
    """

    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


@dataclass(frozen=True)
class Track:
    """Persistent object track with identity assigned by the tracker.

    Attributes:
        track_id: Unique track identifier assigned by the multi-object tracker.
        bbox: Axis-aligned bounding box as ``(x1, y1, x2, y2)`` in pixel
            coordinates, where ``(x1, y1)`` is the top-left corner and
            ``(x2, y2)`` is the bottom-right corner.
        class_id: Integer class identifier from the detector's class mapping.
        class_name: Human-readable class label (for example, ``"car"``).
        score: Tracker confidence or association score in the range ``[0.0, 1.0]``.
    """

    track_id: int
    bbox: tuple[float, float, float, float]
    class_id: int
    class_name: str
    score: float


class Direction(str, Enum):
    """Direction of a validated line crossing relative to the counting line.

    Values serialize cleanly to CSV and JSON as the strings ``"IN"`` and
    ``"OUT"``.
    """

    IN = "IN"
    """Object crossed the line in the configured IN direction."""

    OUT = "OUT"
    """Object crossed the line in the configured OUT direction."""


@dataclass(frozen=True)
class CrossingEvent:
    """Validated crossing of a virtual counting line by a tracked object.

    Attributes:
        frame_idx: Zero-based index of the video frame where the crossing
            was detected.
        timestamp_seconds: Elapsed time from the start of the video in seconds.
        track_id: Identifier of the track that crossed the line.
        class_id: Integer class identifier of the tracked object.
        class_name: Human-readable class label of the tracked object.
        direction: Crossing direction (``Direction.IN`` or ``Direction.OUT``).
        line_id: Identifier of the counting line that was crossed.
        confidence: Optional detector confidence at crossing time; ``None`` if
            not recorded.
        bbox: Optional axis-aligned bounding box as ``(x1, y1, x2, y2)`` in
            pixel coordinates at crossing time; ``None`` if not recorded.
        video_name: Optional source video filename or path; ``None`` if not
            recorded.
    """

    frame_idx: int
    timestamp_seconds: float
    track_id: int
    class_id: int
    class_name: str
    direction: Direction
    line_id: str
    confidence: float | None = None
    bbox: tuple[float, float, float, float] | None = None
    video_name: str | None = None
