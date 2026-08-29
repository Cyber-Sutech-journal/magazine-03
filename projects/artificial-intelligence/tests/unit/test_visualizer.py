import numpy as np

from mot_counting.interfaces.visualizer import IVisualizer
from mot_counting.visualizers.opencv_visualizer import (
    OpenCvVisualizer,
    format_counters_overlay,
)


def test_visualizer_is_i_visualizer():
    assert issubclass(OpenCvVisualizer, IVisualizer)


def test_draw_does_not_mutate_input_frame():
    vis = OpenCvVisualizer()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    original = frame.copy()

    out = vis.draw(frame, tracks=[], lines=[], counters={})

    assert np.array_equal(frame, original)
    assert out is not frame


def test_format_counters_overlay_flat():
    counters = {"person": {"in": 2, "out": 1}}
    assert format_counters_overlay(counters) == ["Person IN: 2 OUT: 1"]


def test_format_counters_overlay_nested():
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
    counters = {"total": 7}
    assert format_counters_overlay(counters) == ["total: 7"]


def test_draw_with_empty_inputs_returns_frame():
    vis = OpenCvVisualizer()
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    out = vis.draw(frame, tracks=[], lines=[], counters={})

    assert out.shape == frame.shape
    assert out.dtype == frame.dtype
