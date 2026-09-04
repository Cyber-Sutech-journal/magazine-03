"""Unit tests for OpenCvVisualizer and helper formatting utilities."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from mot_counting.interfaces.visualizer import IVisualizer
from mot_counting.observers.base import Observer
from mot_counting.types import Direction, Track
from mot_counting.visualizers.opencv_visualizer import (
    OpenCvVisualizer,
    _get_color,
    _parse_line,
    format_counters_overlay,
)


@pytest.fixture
def blank_frame() -> np.ndarray:
    """Fixture providing a blank 3-channel 480x640 BGR image."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_track() -> Track:
    """Fixture providing a sample Track object with all required positional attributes."""
    return Track(
        track_id=1,
        bbox=(50, 50, 150, 150),
        class_id=0,
        class_name="person",
        score=0.92,
    )


# ---------------------------------------------------------------------------
# 1. Tests for OpenCvVisualizer Initialization and Interface
# ---------------------------------------------------------------------------


def test_visualizer_instantiation() -> None:
    """Verify default initialization and inheritance from IVisualizer."""
    vis = OpenCvVisualizer()
    assert isinstance(vis, IVisualizer)
    assert vis.line_color == (0, 255, 255)
    assert vis.text_color == (255, 255, 255)
    assert vis.line_thickness == 2
    assert vis.font_scale == 0.6
    assert vis.last_annotated_frame is None


def test_visualizer_custom_params() -> None:
    """Verify custom visualizer constructor parameters."""
    vis = OpenCvVisualizer(
        line_color=(255, 0, 0),
        text_color=(0, 255, 0),
        line_thickness=3,
        font_scale=0.8,
    )
    assert vis.line_color == (255, 0, 0)
    assert vis.text_color == (0, 255, 0)
    assert vis.line_thickness == 3
    assert vis.font_scale == 0.8


# ---------------------------------------------------------------------------
# 2. Tests for _get_color logic (Class Name Priority & Fallbacks)
# ---------------------------------------------------------------------------


def test_get_color_by_canonical_name() -> None:
    """Verify canonical class names return expected palette colors."""
    assert _get_color(class_name="person") == (0, 255, 0)
    assert _get_color(class_name="CAR") == (255, 0, 0)
    assert _get_color(class_name=" bicycle ") == (0, 255, 255)
    assert _get_color(class_name="bus") == (0, 165, 255)


def test_get_color_fallback_to_id() -> None:
    """Verify fallback to class_id when class_name is absent or unknown."""
    # Unknown class name with valid id fallback
    assert _get_color(class_name="alien", class_id=0) == (0, 255, 0)
    # None class name with valid id fallback
    assert _get_color(class_name=None, class_id=1) == (255, 0, 0)


def test_get_color_default_fallback() -> None:
    """Verify fallback to default green if neither is resolvable."""
    assert _get_color(class_name=None, class_id=None) == (0, 255, 0)


# ---------------------------------------------------------------------------
# 3. Tests for _parse_line Utility
# ---------------------------------------------------------------------------


def test_parse_line_from_dict() -> None:
    """Verify parsing dictionary configurations with point_a/b or pt1/pt2."""
    line_dict1 = {"point_a": (0, 100), "point_b": (200, 100), "line_id": "Gate1"}
    assert _parse_line(line_dict1) == ((0, 100), (200, 100), "Gate1")

    line_dict2 = {"pt1": (10, 20), "pt2": (30, 40)}
    assert _parse_line(line_dict2) == ((10, 20), (30, 40), "Line")


def test_parse_line_from_tuple_or_list() -> None:
    """Verify parsing tuple and list representations."""
    line_tup = ((0, 50), (100, 50), "L1")
    assert _parse_line(line_tup) == ((0, 50), (100, 50), "L1")

    line_list = [(0, 50), (100, 50)]
    assert _parse_line(line_list) == ((0, 50), (100, 50), "Line")


def test_parse_line_invalid() -> None:
    """Verify invalid line definitions return None gracefully."""
    assert _parse_line({}) is None
    assert _parse_line("invalid") is None
    assert _parse_line([10]) is None


# ---------------------------------------------------------------------------
# 4. Tests for format_counters_overlay
# ---------------------------------------------------------------------------


def test_format_counters_canonical_tuples() -> None:
    """Verify overlay formatting with canonical tuple keys: (class_name, line_id, direction)."""
    counters = {
        ("person", "line_1", "in"): 5,
        ("person", "line_1", "out"): 2,
        ("car", "line_1", "in"): 10,
        ("car", "line_2", "in"): 1,
    }
    overlay = format_counters_overlay(counters)
    assert "[line_1]" in overlay
    assert "[line_2]" in overlay
    assert "  Person IN: 5 OUT: 2" in overlay
    assert "  Car IN: 10 OUT: 0" in overlay
    assert "  Car IN: 1 OUT: 0" in overlay


def test_format_counters_nested_dict() -> None:
    """Verify overlay formatting with nested dict structure {line_id: {class_name: {in/out: count}}}."""
    counters = {
        "line_A": {
            "person": {"in": 3, "out": 1},
            "car": {"in": 0, "out": 4},
        }
    }
    overlay = format_counters_overlay(counters)
    assert "[line_A]" in overlay
    assert "  Person IN: 3 OUT: 1" in overlay
    assert "  Car IN: 0 OUT: 4" in overlay


def test_format_counters_flat_dict() -> None:
    """Verify overlay formatting with flat dictionary {class_name: {in: x, out: y}}."""
    counters = {
        "person": {"in": 12, "out": 4},
    }
    overlay = format_counters_overlay(counters)
    assert "Person IN: 12 OUT: 4" in overlay


def test_format_counters_empty() -> None:
    """Verify empty dictionary returns empty list."""
    assert format_counters_overlay({}) == []


def test_format_counters_real_direction_enum() -> None:
    """Live overlay must read Direction.IN/OUT keys from get_counters()."""
    counters = {
        ("person", "main_line", Direction.IN): 4,
        ("person", "main_line", Direction.OUT): 2,
        ("car", "main_line", Direction.IN): 1,
    }
    overlay = format_counters_overlay(counters)
    assert "[main_line]" in overlay
    assert "  Person IN: 4 OUT: 2" in overlay
    assert "  Car IN: 1 OUT: 0" in overlay
    assert all("DIRECTION.IN" not in line for line in overlay)


# ---------------------------------------------------------------------------
# 5. Tests for Rendering (`draw` and `update` Methods)
# ---------------------------------------------------------------------------


def test_draw_renders_and_returns_new_frame(blank_frame: np.ndarray, sample_track: Track) -> None:
    """Verify draw returns an annotated frame without mutating original frame in-place."""
    vis = OpenCvVisualizer()
    original_copy = blank_frame.copy()

    lines = [{"point_a": (10, 10), "point_b": (100, 10), "line_id": "L1"}]
    counters = {("person", "L1", "in"): 3}

    result = vis.draw(
        frame=blank_frame,
        tracks=[sample_track],
        lines=lines,
        counters=counters,
    )

    # Frame is modified and returned
    assert result is not None
    assert result.shape == blank_frame.shape
    assert np.any(result != original_copy)
    # Original frame remains pristine
    assert np.array_equal(blank_frame, original_copy)
    # last_annotated_frame state is updated
    assert vis.last_annotated_frame is not None
    assert np.array_equal(vis.last_annotated_frame, result)


def test_visualizer_observer_inheritance_and_update_signature() -> None:
    """Visualizer must be an Observer whose update() matches the §4.3 contract."""
    vis = OpenCvVisualizer()
    assert isinstance(vis, Observer)
    assert isinstance(vis, IVisualizer)
    assert list(inspect.signature(OpenCvVisualizer.update).parameters) == list(
        inspect.signature(Observer.update).parameters
    )


def test_update_executes_pipeline_and_caches_frame(
    blank_frame: np.ndarray, sample_track: Track
) -> None:
    """Observer.update renders via the bound frame and construction-time lines."""
    vis = OpenCvVisualizer(lines=[((0, 0), (50, 50), "TestLine")])
    vis.set_frame(blank_frame)
    vis.update(
        frame_idx=0,
        tracks=[sample_track],
        events=[],
        counters={"person": {"in": 1, "out": 0}},
    )

    assert vis.last_annotated_frame is not None
    assert isinstance(vis.last_annotated_frame, np.ndarray)
    assert vis.last_annotated_frame.shape == (480, 640, 3)


def test_draw_with_empty_inputs(blank_frame: np.ndarray) -> None:
    """Verify draw handles None/empty lists gracefully."""
    vis = OpenCvVisualizer()
    result = vis.draw(frame=blank_frame, tracks=None, lines=None, counters=None)
    assert result.shape == blank_frame.shape
    assert np.array_equal(result, blank_frame)
