"""OpenCV-based visualizer for bounding boxes, tracking IDs, lines, and counter overlays."""

from __future__ import annotations

import cv2
import numpy as np

from mot_counting.interfaces.visualizer import IVisualizer
from mot_counting.types import Track

# Color palette mapped by canonical class name with fallback to class_id
CLASS_COLORS: dict[str | int, tuple[int, int, int]] = {
    "person": (0, 255, 0),  # Green
    "car": (255, 0, 0),  # Blue
    "bicycle": (0, 255, 255),  # Yellow
    "motorcycle": (255, 255, 0),  # Cyan
    "bus": (0, 165, 255),  # Orange
    "truck": (128, 0, 128),  # Purple
}

DEFAULT_PALETTE = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (0, 255, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 165, 255),
    (128, 0, 128),
]


def _get_color(class_name: str | None = None, class_id: int | None = None) -> tuple[int, int, int]:
    """Retrieve BGR color prioritising class_name, falling back to class_id, then palette."""
    if class_name is not None:
        normalized_name = str(class_name).strip().lower()
        if normalized_name in CLASS_COLORS:
            return CLASS_COLORS[normalized_name]

    if class_id is not None:
        if class_id in CLASS_COLORS:
            return CLASS_COLORS[class_id]
        return DEFAULT_PALETTE[class_id % len(DEFAULT_PALETTE)]

    return (0, 255, 0)


def _parse_line(line: object) -> tuple[tuple[int, int], tuple[int, int], str] | None:
    """Extract (point_a, point_b, line_id) from dicts, tuples, or objects."""
    if isinstance(line, dict):
        pt_a = line.get("point_a") or line.get("pt1")
        pt_b = line.get("point_b") or line.get("pt2")
        line_id = str(line.get("line_id", "Line"))
        if pt_a is not None and pt_b is not None:
            return (tuple(pt_a), tuple(pt_b), line_id)
        return None

    if hasattr(line, "point_a") and hasattr(line, "point_b"):
        pt_a = line.point_a
        pt_b = line.point_b
        line_id = str(getattr(line, "line_id", "Line"))
        return (tuple(pt_a), tuple(pt_b), line_id)

    if isinstance(line, (tuple, list)) and len(line) >= 2:
        pt_a = line[0]
        pt_b = line[1]
        line_id = str(line[2]) if len(line) > 2 else "Line"
        return (tuple(pt_a), tuple(pt_b), line_id)

    return None


def format_counters_overlay(counters: dict) -> list[str]:
    """Format counting dictionary into readable text overlay lines.

    Handles canonical Spec tuple keys: (class_name, line_id, direction),
    multi-line nested dicts: {line_id: {class_name: {in: x, out: y}}},
    and flat dicts: {class_name: {in: x, out: y}}.
    """
    if not counters:
        return []

    lines: list[str] = []

    # Check if keys are canonical tuples: (class_name, line_id, direction)
    first_key = next(iter(counters))
    if isinstance(first_key, tuple) and len(first_key) >= 3:
        # Group by line_id -> class_name -> direction
        grouped: dict[str, dict[str, dict[str, int]]] = {}
        for (c_name, l_id, direction), count in counters.items():
            l_id_str = str(l_id)
            c_name_str = str(c_name).capitalize()
            dir_str = str(direction).upper()
            if l_id_str not in grouped:
                grouped[l_id_str] = {}
            if c_name_str not in grouped[l_id_str]:
                grouped[l_id_str][c_name_str] = {"IN": 0, "OUT": 0}
            grouped[l_id_str][c_name_str][dir_str] = count

        for l_id_str in sorted(grouped.keys()):
            lines.append(f"[{l_id_str}]")
            for c_name_str in sorted(grouped[l_id_str].keys()):
                in_cnt = grouped[l_id_str][c_name_str].get("IN", 0)
                out_cnt = grouped[l_id_str][c_name_str].get("OUT", 0)
                lines.append(f"  {c_name_str} IN: {in_cnt} OUT: {out_cnt}")
        return lines

    # Check for nested line dictionary: {line_id: {class_name: {...}}}
    if isinstance(first_key, str) and isinstance(counters[first_key], dict):
        sub_first = next(iter(counters[first_key].values()), None)
        if isinstance(sub_first, dict):
            for l_id, sub_dict in counters.items():
                lines.append(f"[{l_id}]")
                for c_name, dirs in sub_dict.items():
                    in_cnt = dirs.get("in", dirs.get("IN", 0))
                    out_cnt = dirs.get("out", dirs.get("OUT", 0))
                    lines.append(f"  {str(c_name).capitalize()} IN: {in_cnt} OUT: {out_cnt}")
            return lines

        # Flat dict: {class_name: {in: x, out: y}}
        for c_name, dirs in counters.items():
            in_cnt = dirs.get("in", dirs.get("IN", 0))
            out_cnt = dirs.get("out", dirs.get("OUT", 0))
            lines.append(f"{str(c_name).capitalize()} IN: {in_cnt} OUT: {out_cnt}")
        return lines

    # Primitive scalar key-value fallback
    for k, v in counters.items():
        lines.append(f"{k}: {v}")

    return lines


class OpenCvVisualizer(IVisualizer):
    """Visualizes tracked objects, lines, and counting statistics using OpenCV."""

    def __init__(
        self,
        line_color: tuple[int, int, int] = (0, 255, 255),
        text_color: tuple[int, int, int] = (255, 255, 255),
        line_thickness: int = 2,
        font_scale: float = 0.6,
    ) -> None:
        self.line_color = line_color
        self.text_color = text_color
        self.line_thickness = line_thickness
        self.font_scale = font_scale
        self.last_annotated_frame: np.ndarray | None = None

    def draw(
        self,
        frame: np.ndarray,
        tracks: list[Track] | None = None,
        lines: list | None = None,
        counters: dict | None = None,
    ) -> np.ndarray:
        """Render tracks, counting lines, and counter overlay on a copy of the frame."""
        annotated = frame.copy()

        # 1. Draw counting lines
        if lines:
            for line_spec in lines:
                parsed = _parse_line(line_spec)
                if parsed is None:
                    continue
                pt_a, pt_b, line_id = parsed
                cv2.line(annotated, pt_a, pt_b, self.line_color, self.line_thickness)
                mid_x = (pt_a[0] + pt_b[0]) // 2
                mid_y = (pt_a[1] + pt_b[1]) // 2
                cv2.putText(
                    annotated,
                    line_id,
                    (mid_x + 5, mid_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    self.line_color,
                    1,
                    cv2.LINE_AA,
                )

        # 2. Draw tracks (bounding boxes and IDs)
        if tracks:
            for track in tracks:
                x1, y1, x2, y2 = map(int, track.bbox)
                color = _get_color(
                    getattr(track, "class_name", None), getattr(track, "class_id", None)
                )

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                label_parts = [f"ID:{track.track_id}"]
                if getattr(track, "class_name", None):
                    label_parts.append(str(track.class_name))
                if getattr(track, "score", None) is not None:
                    label_parts.append(f"{track.score:.2f}")

                label = " ".join(label_parts)
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                text_y = max(y1 - 5, th + 5)
                cv2.rectangle(
                    annotated,
                    (x1, text_y - th - 3),
                    (x1 + tw + 2, text_y + baseline - 1),
                    color,
                    -1,
                )
                cv2.putText(
                    annotated,
                    label,
                    (x1 + 1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

        # 3. Draw counters overlay in the top-left corner
        if counters:
            overlay_lines = format_counters_overlay(counters)
            y_offset = 25
            for line_txt in overlay_lines:
                cv2.putText(
                    annotated,
                    line_txt,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_scale,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    line_txt,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_scale,
                    self.text_color,
                    1,
                    cv2.LINE_AA,
                )
                y_offset += int(25 * self.font_scale / 0.6)

        self.last_annotated_frame = annotated
        return annotated

    def update(
        self,
        frame: np.ndarray,
        tracks: list[Track] | None = None,
        lines: list | None = None,
        counters: dict | None = None,
    ) -> None:
        """Observer callback; performs rendering and stores output in self.last_annotated_frame."""
        self.draw(
            frame=frame,
            tracks=tracks if tracks is not None else [],
            lines=lines if lines is not None else [],
            counters=counters if counters is not None else {},
        )
