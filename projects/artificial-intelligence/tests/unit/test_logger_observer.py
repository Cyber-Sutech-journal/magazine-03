"""Unit tests for LoggerObserver (§10.8, §12.2)."""

from __future__ import annotations

import logging

import pytest

from mot_counting.observers.base import Observer
from mot_counting.observers.logger_observer import LoggerObserver
from mot_counting.types import CrossingEvent, Direction, Track

_TRACK = Track(track_id=1, bbox=(0.0, 0.0, 10.0, 10.0), class_id=0, class_name="person", score=0.9)
_EVENT = CrossingEvent(
    frame_idx=7,
    timestamp_seconds=0.7,
    track_id=1,
    class_id=0,
    class_name="person",
    direction=Direction.IN,
    line_id="main_line",
    confidence=0.85,
)
_COUNTERS: dict = {("person", "main_line", "IN"): 1}


def _make_observer(level: int = logging.DEBUG) -> tuple[LoggerObserver, logging.Logger]:
    """Return a LoggerObserver wired to an isolated test logger."""
    test_logger = logging.getLogger("test.mot_counting.logger_observer")
    test_logger.handlers.clear()
    test_logger.setLevel(level)
    test_logger.propagate = False
    handler = logging.Handler()
    handler.setLevel(level)
    test_logger.addHandler(handler)
    return LoggerObserver(logger=test_logger), test_logger


def test_logger_observer_is_observer_subclass() -> None:
    observer, _ = _make_observer()
    assert isinstance(observer, Observer)


def test_update_emits_debug_record_for_crossing_decision(
    caplog: pytest.LogCaptureFixture,
) -> None:
    observer = LoggerObserver()
    caplog.set_level(logging.DEBUG, logger=observer._logger.name)  # noqa: SLF001

    observer.update(
        frame_idx=7,
        tracks=[_TRACK],
        events=[_EVENT],
        counters=_COUNTERS,
    )

    debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert debug_records
    assert any("Crossing decision" in r.message for r in debug_records)
    assert any("track_id=1" in r.message for r in debug_records)
    assert any("line_id=main_line" in r.message for r in debug_records)


def test_update_does_not_emit_debug_when_effective_level_is_info(
    caplog: pytest.LogCaptureFixture,
) -> None:
    observer = LoggerObserver()
    caplog.set_level(logging.INFO, logger=observer._logger.name)  # noqa: SLF001

    observer.update(
        frame_idx=7,
        tracks=[_TRACK],
        events=[_EVENT],
        counters=_COUNTERS,
    )

    debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert debug_records == []

    info_records = [r for r in caplog.records if r.levelno == logging.INFO]
    assert len(info_records) == 1
    assert "emitted 1 crossing event" in info_records[0].message


def test_update_with_no_events_emits_no_info_summary(
    caplog: pytest.LogCaptureFixture,
) -> None:
    observer = LoggerObserver()
    caplog.set_level(logging.DEBUG, logger=observer._logger.name)  # noqa: SLF001

    observer.update(frame_idx=0, tracks=[_TRACK], events=[], counters={})

    info_records = [r for r in caplog.records if r.levelno == logging.INFO]
    assert info_records == []


def test_log_run_start_emits_info(caplog: pytest.LogCaptureFixture) -> None:
    observer = LoggerObserver()
    caplog.set_level(logging.INFO, logger=observer._logger.name)  # noqa: SLF001

    observer.log_run_start(video_path="data/clip.mp4")

    assert any("Pipeline run started" in r.message for r in caplog.records)
    assert any("data/clip.mp4" in r.message for r in caplog.records)


def test_log_run_stop_emits_info(caplog: pytest.LogCaptureFixture) -> None:
    observer = LoggerObserver()
    caplog.set_level(logging.INFO, logger=observer._logger.name)  # noqa: SLF001

    observer.log_run_stop(frames_processed=100, elapsed_seconds=5.0, average_fps=20.0)

    assert any("Pipeline run stopped" in r.message for r in caplog.records)
    assert any("frames=100" in r.message for r in caplog.records)


def test_log_frame_warning_emits_warning(caplog: pytest.LogCaptureFixture) -> None:
    observer = LoggerObserver()
    caplog.set_level(logging.WARNING, logger=observer._logger.name)  # noqa: SLF001

    observer.log_frame_warning(frame_idx=3, message="decode failure — skipping frame")

    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warning_records) == 1
    assert "Frame 3" in warning_records[0].message
    assert "decode failure" in warning_records[0].message


def test_update_does_not_reconfigure_logging_handlers() -> None:
    """LoggerObserver must not add handlers on every update() call."""
    root_handler_count_before = len(logging.getLogger().handlers)

    observer, test_logger = _make_observer()
    # Replace the bare Handler with a NullHandler so emit() works in the test.
    test_logger.handlers.clear()
    test_logger.addHandler(logging.NullHandler())
    handler_count_before = len(test_logger.handlers)

    observer.update(frame_idx=0, tracks=[], events=[], counters={})
    observer.update(frame_idx=1, tracks=[], events=[], counters={})

    assert len(test_logger.handlers) == handler_count_before
    assert len(logging.getLogger().handlers) == root_handler_count_before
