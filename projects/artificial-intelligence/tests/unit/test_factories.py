"""Unit tests for DetectorFactory and TrackerFactory (§4.1).

T19 update
----------
Stub assertions (``NotImplementedError``) removed; tests now verify that
factories return real ``IDetector`` / ``ITracker`` instances when given
appropriate model objects (mock for detector, ``None`` for ByteTrack since
ByteTrackWrapper is self-contained).
"""

from __future__ import annotations

import pytest

from mot_counting.detectors.yolo26_detector import Yolo26Detector
from mot_counting.factories.detector_factory import DetectorFactory
from mot_counting.factories.tracker_factory import TrackerFactory
from mot_counting.interfaces.detector import IDetector
from mot_counting.interfaces.tracker import ITracker
from mot_counting.trackers.bytetrack_tracker import ByteTrackWrapper

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class _FakeModel:
    """Minimal stand-in for an already-loaded Ultralytics YOLO model.

    Only the attributes consumed by Yolo26Detector and DetectorFactory are
    provided so that no real model weights are required in unit tests.
    """

    names: dict[int, str] = {0: "person", 1: "car", 2: "truck"}


def _detector_factory() -> DetectorFactory:
    return DetectorFactory(confidence_threshold=0.4, classes=["person", "car"])


def _tracker_factory() -> TrackerFactory:
    return TrackerFactory(track_thresh=0.5, match_thresh=0.8, track_buffer=30, frame_rate=30)


# ---------------------------------------------------------------------------
# DetectorFactory — unknown variant raises ValueError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_variant",
    ["yolo11m", "yolov8n", "", "YOLO26M", "yolo26z"],
)
def test_detector_factory_unknown_variant_raises_value_error(bad_variant: str) -> None:
    factory = _detector_factory()
    with pytest.raises(ValueError, match="Unknown model_variant"):
        factory.create(bad_variant, _FakeModel())


# ---------------------------------------------------------------------------
# DetectorFactory — all known variants return a real IDetector (T19)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x"])
def test_detector_factory_known_variant_returns_idetector(variant: str) -> None:
    """Factory must return a concrete IDetector wrapping the provided model."""
    factory = _detector_factory()
    detector = factory.create(variant, _FakeModel())
    assert isinstance(detector, IDetector)
    assert isinstance(detector, Yolo26Detector)


def test_detector_factory_passes_config_to_yolo26_detector() -> None:
    """Confidence threshold and class list must be forwarded to the detector."""
    factory = DetectorFactory(confidence_threshold=0.7, classes=["truck"])
    detector = factory.create("yolo26m", _FakeModel())
    assert isinstance(detector, Yolo26Detector)
    assert detector.confidence_threshold == 0.7
    assert detector.allowed_classes == ["truck"]


# ---------------------------------------------------------------------------
# TrackerFactory — unknown type raises ValueError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_type",
    ["ByteTrack", "BYTETRACK", "sort", "deepsort", "", "botsort2"],
)
def test_tracker_factory_unknown_type_raises_value_error(bad_type: str) -> None:
    factory = _tracker_factory()
    with pytest.raises(ValueError, match="Unknown tracker type"):
        factory.create(bad_type, None)


# ---------------------------------------------------------------------------
# TrackerFactory — bytetrack returns real ITracker (T19)
# ---------------------------------------------------------------------------


def test_tracker_factory_bytetrack_returns_itracker() -> None:
    """Factory must return a ByteTrackWrapper satisfying ITracker."""
    factory = _tracker_factory()
    tracker = factory.create("bytetrack", None)
    assert isinstance(tracker, ITracker)
    assert isinstance(tracker, ByteTrackWrapper)


def test_tracker_factory_bytetrack_passes_hyperparams() -> None:
    """Tracker hyperparameters must be forwarded to the ByteTrackWrapper."""
    factory = TrackerFactory(track_thresh=0.6, match_thresh=0.75, track_buffer=20, frame_rate=25)
    tracker = factory.create("bytetrack", None)
    assert isinstance(tracker, ByteTrackWrapper)
    assert tracker.args.track_thresh == pytest.approx(0.6)
    assert tracker.args.match_thresh == pytest.approx(0.75)
    assert tracker.args.track_buffer == 20
    assert tracker.args.frame_rate == 25


# ---------------------------------------------------------------------------
# TrackerFactory — botsort still raises NotImplementedError (stretch goal)
# ---------------------------------------------------------------------------


def test_tracker_factory_botsort_raises_not_implemented() -> None:
    factory = _tracker_factory()
    with pytest.raises(NotImplementedError):
        factory.create("botsort", None)


# ---------------------------------------------------------------------------
# Factories never load models themselves (structural check)
# ---------------------------------------------------------------------------


def test_detector_factory_does_not_load_models_itself() -> None:
    """Constructing DetectorFactory must not raise."""
    factory = _detector_factory()
    assert factory is not None


def test_tracker_factory_does_not_initialise_trackers_itself() -> None:
    """Constructing TrackerFactory must not raise."""
    factory = _tracker_factory()
    assert factory is not None
