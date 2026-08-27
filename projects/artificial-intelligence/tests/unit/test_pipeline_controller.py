"""Unit tests for PipelineController (§4.4, §7.3, T10)."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from mot_counting.config import AppConfig
from mot_counting.controllers.pipeline_controller import LineGeometryError, PipelineController
from mot_counting.interfaces.crossing import ICrossingLogic
from mot_counting.interfaces.detector import IDetector
from mot_counting.interfaces.frame_source import IFrameSource
from mot_counting.interfaces.repository import IEventRepository
from mot_counting.interfaces.tracker import ITracker
from mot_counting.interfaces.visualizer import IVisualizer
from mot_counting.observers.base import Subject
from mot_counting.types import CrossingEvent, Detection, Direction, Track

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_BLANK_FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
_DETECTION = Detection(xyxy=(0.0, 0.0, 10.0, 10.0), confidence=0.9, class_id=0, class_name="person")
_TRACK = Track(track_id=1, bbox=(0.0, 0.0, 10.0, 10.0), class_id=0, class_name="person", score=0.9)
_EVENT = CrossingEvent(
    frame_idx=0,
    timestamp_seconds=0.0,
    track_id=1,
    class_id=0,
    class_name="person",
    direction=Direction.IN,
    line_id="main_line",
)


def _make_config(
    line_id: str = "main_line",
    point_a: list[int] | None = None,
    point_b: list[int] | None = None,
) -> AppConfig:
    """Return a minimal valid AppConfig with one counting line."""
    return AppConfig.model_validate(
        {
            "video": {"path": "data/clip.mp4", "output_dir": "outputs/"},
            "detection": {
                "model_variant": "yolo26m",
                "imgsz": 640,
                "confidence_threshold": 0.4,
                "classes": ["person"],
            },
            "tracker": {
                "type": "bytetrack",
                "track_thresh": 0.5,
                "match_thresh": 0.8,
                "track_buffer": 30,
            },
            "lines": [
                {
                    "line_id": line_id,
                    "point_a": point_a or [10, 100],
                    "point_b": point_b or [600, 100],
                    "positive_direction": "A_to_B",
                }
            ],
            "crossing_logic": {
                "reference_point": "bottom_center",
                "history_length": 8,
                "confirmation_majority_threshold": 0.7,
                "cooldown_seconds": 1.5,
                "stale_track_timeout_seconds": 2.0,
                "min_displacement_px": None,
                "min_velocity_px_per_s": None,
            },
            "events": {"output_csv": "outputs/events.csv"},
            "evaluation": {"matching_tolerance_seconds": 1.0},
            "visualization": {"output_video": "outputs/annotated.mp4", "draw_trails": False},
        }
    )


def _make_controller(
    frames: list[tuple[bool, Any]],
    frame_size: tuple[int, int] = (640, 480),
    config: AppConfig | None = None,
    events_per_frame: list[CrossingEvent] | None = None,
) -> tuple[PipelineController, dict[str, MagicMock]]:
    """Build a PipelineController wired with controllable mocks.

    Args:
        frames: Sequence of ``(success, frame_or_None)`` returned by
            ``IFrameSource.read()`` in order.  After the sequence is
            exhausted the source returns ``(False, None)`` (end-of-video).
        frame_size: ``(width, height)`` reported by the mock frame source.
        config: AppConfig to use; defaults to ``_make_config()``.
        events_per_frame: Crossing events returned by crossing logic on each
            successful frame call.

    Returns:
        ``(controller, mocks)`` where ``mocks`` is a dict keyed by component
        name for later assertion.
    """
    if config is None:
        config = _make_config()
    if events_per_frame is None:
        events_per_frame = []

    # Append natural end-of-video sentinel after the provided frames.
    read_sequence = list(frames) + [(False, None)]

    frame_source = MagicMock(spec=IFrameSource)
    frame_source.get_frame_size.return_value = frame_size
    frame_source.get_fps.return_value = 30.0
    frame_source.read.side_effect = read_sequence

    detector = MagicMock(spec=IDetector)
    detector.predict.return_value = [_DETECTION]

    tracker = MagicMock(spec=ITracker)
    tracker.update.return_value = [_TRACK]

    crossing_logic = MagicMock(spec=ICrossingLogic)
    crossing_logic.process.return_value = list(events_per_frame)
    crossing_logic.get_counters.return_value = {}

    event_repository = MagicMock(spec=IEventRepository)
    visualizer = MagicMock(spec=IVisualizer)
    subject = MagicMock(spec=Subject)

    controller = PipelineController(
        config=config,
        frame_source=frame_source,
        detector=detector,
        tracker=tracker,
        crossing_logic=crossing_logic,
        event_repository=event_repository,
        visualizer=visualizer,
        subject=subject,
    )

    mocks = {
        "frame_source": frame_source,
        "detector": detector,
        "tracker": tracker,
        "crossing_logic": crossing_logic,
        "event_repository": event_repository,
        "visualizer": visualizer,
        "subject": subject,
    }
    return controller, mocks


# ---------------------------------------------------------------------------
# Main loop — call counts and order
# ---------------------------------------------------------------------------


def test_three_frame_run_calls_each_stage_three_times() -> None:
    good_frames = [(True, _BLANK_FRAME)] * 3
    controller, mocks = _make_controller(good_frames)

    controller.run()

    assert mocks["detector"].predict.call_count == 3
    assert mocks["tracker"].update.call_count == 3
    assert mocks["crossing_logic"].process.call_count == 3
    assert mocks["subject"].notify.call_count == 3


def test_frame_loop_sequence_is_detect_track_process_notify() -> None:
    """Verify the exact call order per frame using a shared call recorder."""
    call_order: list[str] = []

    frame_source = MagicMock(spec=IFrameSource)
    frame_source.get_frame_size.return_value = (640, 480)
    frame_source.get_fps.return_value = 30.0
    frame_source.read.side_effect = [(True, _BLANK_FRAME), (False, None)]

    detector = MagicMock(spec=IDetector)
    detector.predict.side_effect = lambda *a, **kw: call_order.append("detect") or [_DETECTION]

    tracker = MagicMock(spec=ITracker)
    tracker.update.side_effect = lambda *a, **kw: call_order.append("track") or [_TRACK]

    crossing_logic = MagicMock(spec=ICrossingLogic)
    crossing_logic.process.side_effect = lambda *a, **kw: call_order.append("process") or []
    crossing_logic.get_counters.return_value = {}

    subject = MagicMock(spec=Subject)
    subject.notify.side_effect = lambda *a, **kw: call_order.append("notify")

    controller = PipelineController(
        config=_make_config(),
        frame_source=frame_source,
        detector=detector,
        tracker=tracker,
        crossing_logic=crossing_logic,
        event_repository=MagicMock(spec=IEventRepository),
        visualizer=MagicMock(spec=IVisualizer),
        subject=subject,
    )

    controller.run()

    assert call_order == ["detect", "track", "process", "notify"]


# ---------------------------------------------------------------------------
# Line-geometry validation
# ---------------------------------------------------------------------------


def test_line_point_outside_frame_raises_before_first_frame() -> None:
    config = _make_config(point_a=[700, 100], point_b=[600, 100])  # x=700 >= width=640
    controller, mocks = _make_controller(
        frames=[(True, _BLANK_FRAME)],
        frame_size=(640, 480),
        config=config,
    )

    with pytest.raises(LineGeometryError, match="main_line"):
        controller.run()

    # No frames must have been processed.
    mocks["detector"].predict.assert_not_called()


def test_line_geometry_error_message_includes_line_id_and_dimensions() -> None:
    config = _make_config(point_b=[0, 600])  # y=600 >= height=480
    controller, _ = _make_controller(frames=[], frame_size=(640, 480), config=config)

    with pytest.raises(LineGeometryError) as exc_info:
        controller.run()

    msg = str(exc_info.value)
    assert "main_line" in msg
    assert "640" in msg
    assert "480" in msg


# ---------------------------------------------------------------------------
# Mid-stream decode failures
# ---------------------------------------------------------------------------


def test_mid_stream_decode_failure_skips_frame_and_continues(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """(False, non-None) mid-stream: 1 skipped, loop keeps going."""
    frames = [
        (True, _BLANK_FRAME),
        (False, _BLANK_FRAME),  # mid-stream failure (frame object present but success=False)
        (True, _BLANK_FRAME),
    ]
    controller, mocks = _make_controller(frames)

    with caplog.at_level(logging.WARNING):
        controller.run()

    assert controller.stats.frames_processed == 2
    assert controller.stats.frames_skipped == 1
    assert any("decode failure" in r.message.lower() for r in caplog.records)
    assert mocks["detector"].predict.call_count == 2


# ---------------------------------------------------------------------------
# cleanup() always called via finally
# ---------------------------------------------------------------------------


def test_cleanup_called_even_when_loop_raises_mid_run() -> None:
    """If the detector raises on the 2nd frame, cleanup() must still execute."""
    frames = [(True, _BLANK_FRAME), (True, _BLANK_FRAME)]
    controller, mocks = _make_controller(frames)

    call_count = {"n": 0}

    def _raise_on_second(*_a: object, **_kw: object) -> list[Detection]:
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated detector crash")
        return [_DETECTION]

    mocks["detector"].predict.side_effect = _raise_on_second

    with pytest.raises(RuntimeError, match="simulated detector crash"):
        controller.run()

    mocks["frame_source"].release.assert_called_once()
    mocks["event_repository"].flush.assert_called_once()
    mocks["event_repository"].close.assert_called_once()


# ---------------------------------------------------------------------------
# Crossing events are persisted via repository
# ---------------------------------------------------------------------------


def test_crossing_events_are_saved_via_repository() -> None:
    frames = [(True, _BLANK_FRAME)]
    controller, mocks = _make_controller(frames, events_per_frame=[_EVENT])

    controller.run()

    mocks["event_repository"].save.assert_called_once_with(_EVENT)


# ---------------------------------------------------------------------------
# Runtime statistics
# ---------------------------------------------------------------------------


def test_stats_frames_processed_equals_successful_frames() -> None:
    frames = [(True, _BLANK_FRAME)] * 5
    controller, _ = _make_controller(frames)

    controller.run()

    assert controller.stats.frames_processed == 5


def test_stats_average_fps_is_set_after_run() -> None:
    frames = [(True, _BLANK_FRAME)] * 2
    controller, _ = _make_controller(frames)

    controller.run()

    assert controller.stats.average_fps >= 0.0


def test_stats_final_counters_populated_after_run() -> None:
    frames = [(True, _BLANK_FRAME)]
    controller, mocks = _make_controller(frames)
    mocks["crossing_logic"].get_counters.return_value = {("person", "main_line", "IN"): 1}

    controller.run()

    assert controller.stats.final_counters == {("person", "main_line", "IN"): 1}


# ---------------------------------------------------------------------------
# stop() — graceful early termination
# ---------------------------------------------------------------------------


def test_stop_terminates_loop_after_current_frame() -> None:
    """Calling stop() mid-loop (e.g. from a side-effect on the 1st frame detect)
    causes the loop to exit after that frame completes — not immediately."""
    frames = [(True, _BLANK_FRAME)] * 10
    controller, mocks = _make_controller(frames)

    call_count = {"n": 0}

    def _stop_after_first(*_a: object, **_kw: object) -> list[Detection]:
        call_count["n"] += 1
        if call_count["n"] == 1:
            controller.stop()  # request stop during the first frame's detect step
        return [_DETECTION]

    mocks["detector"].predict.side_effect = _stop_after_first

    controller.run()

    # The first frame should have been fully processed; the loop must not
    # have started a second frame.
    assert controller.stats.frames_processed == 1
