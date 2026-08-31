"""Unit tests for OpenCvVisualizer and related helpers (T16 / Observer T07)."""

from types import SimpleNamespace

import numpy as np

from mot_counting.interfaces.visualizer import IVisualizer
from mot_counting.types import Track
from mot_counting.visualizers.opencv_visualizer import (
    OpenCvVisualizer,
    _parse_line,
    format_counters_overlay,
)


def test_visualizer_is_i_visualizer():
    """Ensure OpenCvVisualizer conforms to the IVisualizer interface."""
    assert issubclass(OpenCvVisualizer, IVisualizer)


def test_draw_does_not_mutate_input_frame():
    """Ensure input frame is not modified in-place."""
    vis = OpenCvVisualizer()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    original = frame.copy()

    out = vis.draw(frame, tracks=[], lines=[], counters={})

    assert np.array_equal(frame, original)
    assert out is not frame


def test_draw_with_empty_inputs_returns_frame():
    """Verify that drawing with empty inputs preserves shape and dtype."""
    vis = OpenCvVisualizer()
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    out = vis.draw(frame, tracks=[], lines=[], counters={})

    assert out.shape == frame.shape
    assert out.dtype == frame.dtype
    assert np.array_equal(out, frame)


def test_draw_tracks_modifies_pixels():
    """Verify that rendering tracks modifies pixel values on the output frame."""
    vis = OpenCvVisualizer()
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    track = Track(
        track_id=1,
        bbox=(20, 20, 80, 80),
        score=0.9,
        class_id=0,
        class_name="person",
    )

    out = vis.draw(frame, tracks=[track], lines=[], counters={})

    assert not np.array_equal(out, frame)
    assert np.any(out > 0)


def test_draw_lines_modifies_pixels():
    """Verify that drawing counting lines modifies pixel values."""
    vis = OpenCvVisualizer()
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    line_spec = {"point_a": (10, 100), "point_b": (190, 100), "line_id": "Gate-1"}

    out = vis.draw(frame, tracks=[], lines=[line_spec], counters={})

    assert not np.array_equal(out, frame)
    assert np.any(out > 0)


def test_parse_line_variants():
    """Test line parsing with point_a/point_b attributes and dict keys."""
    # Dict with point_a / point_b
    d_line = {"point_a": (0, 50), "point_b": (100, 50), "line_id": "L1"}
    assert _parse_line(d_line) == ((0, 50), (100, 50), "L1")

    # Object / LineConfig with point_a / point_b
    obj_line = SimpleNamespace(point_a=(10, 20), point_b=(30, 40), line_id="L2")
    assert _parse_line(obj_line) == ((10, 20), (30, 40), "L2")

    # Legacy pt1 / pt2
    legacy_line = {"pt1": (5, 5), "pt2": (15, 15), "line_id": "L3"}
    assert _parse_line(legacy_line) == ((5, 5), (15, 15), "L3")

    # Tuple sequence
    tuple_line = ((1, 2), (3, 4), "L4")
    assert _parse_line(tuple_line) == ((1, 2), (3, 4), "L4")

    # Invalid line
    assert _parse_line("invalid") is None


def test_format_counters_overlay_canonical_spec():
    """Test formatting canonical Spec §7.4 tuple-key mapping."""
    counters = {
        ("person", "Gate-A", "IN"): 5,
        ("person", "Gate-A", "OUT"): 2,
        ("car", "Gate-A", "IN"): 1,
    }
    lines = format_counters_overlay(counters)
    assert lines == [
        "[Gate-A]",
        "  Car IN: 1 OUT: 0",
        "  Person IN: 5 OUT: 2",
    ]


def test_format_counters_overlay_flat():
    """Test formatting flat dictionary."""
    counters = {"person": {"in": 2, "out": 1}}
    assert format_counters_overlay(counters) == ["Person IN: 2 OUT: 1"]


def test_format_counters_overlay_nested():
    """Test formatting nested dictionary."""
    counters = {
        "Gate-1": {
            "person": {"in": 2, "out": 1},
            "car": {"IN": 3, "OUT": 4},
        }
    }
    lines = format_counters_overlay(counters)
    assert lines == [
        "[Gate-1]",
        "  Person IN: 2 OUT: 1",
        "  Car IN: 3 OUT: 4",
    ]


def test_format_counters_overlay_scalar_fallback():
    """Test fallback on primitive key-value pairs."""
    counters = {"total": 7}
    assert format_counters_overlay(counters) == ["total: 7"]


def test_update_observer_contract():
    """Test visualizer update method as an Observer subscriber."""
    vis = OpenCvVisualizer()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    track = Track(
        track_id=1,
        bbox=(10, 10, 50, 50),
        score=0.8,
        class_id=0,
        class_name="person",
    )
    lines = [{"point_a": (0, 0), "point_b": (50, 50), "line_id": "L1"}]
    counters = {"person": {"in": 1, "out": 0}}

    # Call update with explicit arguments
    out = vis.update(frame=frame, tracks=[track], lines=lines, counters=counters)

    assert out is not None
    assert out.shape == frame.shape
    assert not np.array_equal(out, frame)
    assert vis.last_annotated_frame is not None
