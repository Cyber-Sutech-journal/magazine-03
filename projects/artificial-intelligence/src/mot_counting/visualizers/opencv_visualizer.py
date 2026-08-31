"""OpenCV-based visualizer implementing IVisualizer interface.

Draws annotated output frames: per-track bounding boxes with labels,
configured counting lines with identifiers, and live per-line counters
overlaid as a HUD card (§7.6).

Coloring strategy: consistent per-class colors (CLASS_COLORS) — clearer for
demo videos; unknown classes fall back to a deterministic track-id palette.

The input frame is never mutated: ``draw()`` always returns a fresh copy.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from mot_counting.interfaces.visualizer import IVisualizer
from mot_counting.types import Track

# ==============================================================================
# Drawing constants (§12.5) — no magic numbers inside methods.
# ==============================================================================
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 1
TEXT_PADDING = 4  # horizontal padding around label text
LABEL_HEIGHT_OFFSET = 5  # vertical gap between box edge and text chip

BOX_THICKNESS = 2
LINE_THICKNESS = 2

COLOR_TEXT = (255, 255, 255)  # primary HUD text (white)
COLOR_TEXT_DARK = (0, 0, 0)  # text on top of solid color chips (black)
COLOR_LINE_DEFAULT = (0, 0, 255)  # counting lines (red)
COLOR_OVERLAY_BG = (20, 20, 20)  # HUD card background (dark gray)
OVERLAY_ALPHA = 0.6  # HUD card opacity (0..1)

OVERLAY_MARGIN = 10
OVERLAY_LINE_HEIGHT = 20
OVERLAY_BOX_WIDTH = 240
OVERLAY_HEADER_PAD = 15
OVERLAY_TEXT_OFFSET = 8  # left indent of text inside the HUD card
OVERLAY_INDENT = "  "  # indentation for per-line nested counters

DEFAULT_LINE_LABEL = "line"

# Consistent per-class palette (BGR) — aligned with common COCO demo classes.
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (0, 255, 0),  # person
    1: (255, 128, 0),  # bicycle
    2: (0, 255, 255),  # car
    3: (255, 0, 255),  # motorcycle
    4: (0, 165, 255),  # bus
    5: (255, 255, 0),  # truck
}

# Deterministic fallback palette for unknown classes (cycled by track_id).
TRACK_ID_PALETTE: tuple[tuple[int, int, int], ...] = (
    (255, 0, 0),
    (0, 128, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 128),
    (128, 0, 255),
    (0, 200, 200),
    (200, 200, 0),
)

__all__ = ["OpenCvVisualizer", "format_counters_overlay", "get_color_for_track"]


# ==============================================================================
# Pure helpers (unit-testable in isolation — recommended by T16).
# ==============================================================================


def _count(direction_counts: dict[str, Any], direction: str) -> int:
    """Extract a count from a direction dict, tolerating 'in'/'IN'/'In' keys."""
    for key in (direction, direction.upper(), direction.lower(), direction.capitalize()):
        if key in direction_counts:
            return int(direction_counts[key])
    return 0


def format_counters_overlay(counters: dict[Any, Any]) -> list[str]:
    """Format a counters dict into human-readable HUD lines.

    Supports three shapes:
    * Canonical tuple: {(class_name, line_id, direction): count}
    * Flat dict:       {"person": {"in": 2, "out": 1}}
    * Nested dict:     {"Gate-1": {"person": {"in": 2, ...}}}
    """
    if not counters:
        return []

    lines: list[str] = []
    first_key = next(iter(counters.keys()))

    # Case 1: Canonical Spec §7.4 tuple-key mapping: (class_name, line_id, direction) -> count
    if isinstance(first_key, tuple) and len(first_key) == 3:
        grouped: dict[str, dict[str, dict[str, int]]] = {}
        for (cls_name, line_id, direction), count in counters.items():
            dir_str = str(getattr(direction, "value", direction)).lower()
            cls_str = str(cls_name)
            lid_str = str(line_id)
            grouped.setdefault(lid_str, {}).setdefault(cls_str, {})[dir_str] = int(count)

        for line_id, class_map in sorted(grouped.items()):
            lines.append(f"[{line_id}]")
            for cls_name, directions in sorted(class_map.items()):
                lines.append(
                    f"{OVERLAY_INDENT}{cls_name.capitalize()} "
                    f"IN: {_count(directions, 'in')} OUT: {_count(directions, 'out')}"
                )
        return lines

    # Case 2 & 3: Flat or nested dictionary
    for section, value in counters.items():
        if isinstance(value, dict) and value and all(isinstance(v, dict) for v in value.values()):
            # Nested shape: line_id -> class -> {in/out}
            lines.append(f"[{section}]")
            for cls_name, directions in value.items():
                lines.append(
                    f"{OVERLAY_INDENT}{cls_name.capitalize()} "
                    f"IN: {_count(directions, 'in')} OUT: {_count(directions, 'out')}"
                )
        elif isinstance(value, dict):
            # Flat shape: class -> {in/out}
            lines.append(
                f"{section.capitalize()} IN: {_count(value, 'in')} OUT: {_count(value, 'out')}"
            )
        else:
            # Scalar fallback: "key: value"
            lines.append(f"{section}: {value}")
    return lines


def get_color_for_track(track: Track) -> tuple[int, int, int]:
    """Pick a stable BGR color for a track.

    Primary key: ``class_id`` — consistent per-class coloring (demo-friendly).
    Fallback: deterministic palette indexed by ``track_id`` — stable across
    frames, no RNG, no global state.
    """
    class_id = getattr(track, "class_id", None)
    if class_id in CLASS_COLORS:
        return CLASS_COLORS[class_id]
    track_id = getattr(track, "track_id", 0)
    return TRACK_ID_PALETTE[int(track_id) % len(TRACK_ID_PALETTE)]


def _as_point(value: Any) -> tuple[int, int] | None:
    """Convert a 2D coordinate into an int tuple, or None if malformed."""
    try:
        x, y = value
        return int(round(x)), int(round(y))
    except (TypeError, ValueError):
        return None


def _parse_line(line: Any) -> tuple[tuple[int, int], tuple[int, int], str] | None:
    """Normalize a counting-line spec into ((x1, y1), (x2, y2), line_id).

    Accepts dicts, objects with ``point_a``/``point_b`` or ``pt1``/``pt2`` attributes,
    and (pt1, pt2[, line_id]) sequences. Returns None for unsupported shapes.
    """
    if isinstance(line, dict):
        pt1 = _as_point(line.get("point_a", line.get("pt1")))
        pt2 = _as_point(line.get("point_b", line.get("pt2")))
        line_id = str(line.get("line_id", line.get("id", DEFAULT_LINE_LABEL)))
    elif hasattr(line, "point_a") and hasattr(line, "point_b"):
        pt1 = _as_point(line.point_a)
        pt2 = _as_point(line.point_b)
        line_id = str(getattr(line, "line_id", getattr(line, "id", DEFAULT_LINE_LABEL)))
    elif hasattr(line, "pt1") and hasattr(line, "pt2"):
        pt1 = _as_point(line.pt1)
        pt2 = _as_point(line.pt2)
        line_id = str(getattr(line, "line_id", getattr(line, "id", DEFAULT_LINE_LABEL)))
    elif isinstance(line, (tuple, list)) and len(line) >= 2:
        pt1 = _as_point(line[0])
        pt2 = _as_point(line[1])
        line_id = str(line[2]) if len(line) > 2 else DEFAULT_LINE_LABEL
    else:
        return None

    if pt1 is None or pt2 is None:
        return None
    return pt1, pt2, line_id


# ==============================================================================
# Concrete visualizer
# ==============================================================================


class OpenCvVisualizer(IVisualizer):
    """Renders annotated frames (boxes, labels, lines, counters) with OpenCV."""

    def __init__(self) -> None:
        self.last_annotated_frame: np.ndarray | None = None

    def draw(
        self,
        frame: np.ndarray,
        tracks: list[Track],
        lines: list,
        counters: dict,
    ) -> np.ndarray:
        """Annotate *frame* and return a new copy.

        Documented contract: the input frame is **never mutated** — callers
        can safely reuse it after the call.
        """
        annotated = frame.copy()
        self._draw_lines(annotated, lines)
        self._draw_tracks(annotated, tracks)
        self._draw_counters(annotated, counters)
        self.last_annotated_frame = annotated
        return annotated

    def _draw_tracks(self, frame: np.ndarray, tracks: list[Track]) -> None:
        """Draw bounding boxes + class/track-id label chips."""
        for track in tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            if x2 <= x1 or y2 <= y1:  # skip degenerate boxes
                continue

            color = get_color_for_track(track)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)

            class_name = getattr(track, "class_name", None)
            if not class_name:
                class_name = f"class_{getattr(track, 'class_id', 0)}"
            label = f"{class_name} #{getattr(track, 'track_id', 0)}"

            (text_w, text_h), baseline = cv2.getTextSize(
                label, FONT_FACE, FONT_SCALE, FONT_THICKNESS
            )
            chip_y = max(y1, text_h + LABEL_HEIGHT_OFFSET + baseline)
            chip_top = chip_y - text_h - LABEL_HEIGHT_OFFSET

            # Solid color chip behind the text for readability
            cv2.rectangle(
                frame,
                (x1, chip_top),
                (x1 + text_w + 2 * TEXT_PADDING, chip_y + baseline),
                color,
                thickness=cv2.FILLED,
            )
            cv2.putText(
                frame,
                label,
                (x1 + TEXT_PADDING, chip_y),
                FONT_FACE,
                FONT_SCALE,
                COLOR_TEXT_DARK,
                FONT_THICKNESS,
                lineType=cv2.LINE_AA,
            )

    def _draw_lines(self, frame: np.ndarray, lines: list) -> None:
        """Draw every configured counting line with its line_id label."""
        for raw_line in lines:
            parsed = _parse_line(raw_line)
            if parsed is None:
                continue
            (pt1, pt2, line_id) = parsed

            cv2.line(
                frame,
                pt1,
                pt2,
                COLOR_LINE_DEFAULT,
                LINE_THICKNESS,
                lineType=cv2.LINE_AA,
            )

            mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
            cv2.putText(
                frame,
                f"Line: {line_id}",
                (mid[0] + TEXT_PADDING, mid[1] - TEXT_PADDING),
                FONT_FACE,
                FONT_SCALE,
                COLOR_LINE_DEFAULT,
                FONT_THICKNESS,
                lineType=cv2.LINE_AA,
            )

    def _draw_counters(self, frame: np.ndarray, counters: dict) -> None:
        """Render a semi-transparent HUD card with live per-line counts."""
        overlay_lines = format_counters_overlay(counters)
        if not overlay_lines:
            return

        box_w = OVERLAY_BOX_WIDTH
        box_h = len(overlay_lines) * OVERLAY_LINE_HEIGHT + OVERLAY_HEADER_PAD
        top_left = (OVERLAY_MARGIN, OVERLAY_MARGIN)
        bottom_right = (OVERLAY_MARGIN + box_w, OVERLAY_MARGIN + box_h)

        # Semi-transparent background card (60% opacity)
        card = frame.copy()
        cv2.rectangle(card, top_left, bottom_right, COLOR_OVERLAY_BG, thickness=cv2.FILLED)
        cv2.addWeighted(card, OVERLAY_ALPHA, frame, 1.0 - OVERLAY_ALPHA, 0, frame)

        for i, text in enumerate(overlay_lines):
            y = OVERLAY_MARGIN + OVERLAY_LINE_HEIGHT + i * OVERLAY_LINE_HEIGHT
            cv2.putText(
                frame,
                text,
                (OVERLAY_MARGIN + OVERLAY_TEXT_OFFSET, y),
                FONT_FACE,
                FONT_SCALE,
                COLOR_TEXT,
                FONT_THICKNESS,
                lineType=cv2.LINE_AA,
            )

    def update(
        self,
        frame: np.ndarray,
        tracks: list[Track] | None = None,
        lines: list | None = None,
        counters: dict | None = None,
    ) -> np.ndarray:
        """Observer callback conforming to T07 / T16 contract."""
        return self.draw(
            frame=frame,
            tracks=tracks if tracks is not None else [],
            lines=lines if lines is not None else [],
            counters=counters if counters is not None else {},
        )
