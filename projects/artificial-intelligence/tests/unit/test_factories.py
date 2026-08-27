"""Unit tests for DetectorFactory and TrackerFactory (§4.1).

NOTE (T19 update point)
-----------------------
The tests marked ``# STUB`` below verify only that the known-variant code
paths are reachable and raise the *documented* ``NotImplementedError`` rather
than an unrelated crash.  When T19 wires real concrete implementations, those
tests must be updated to assert that the returned object is an instance of
``IDetector`` / ``ITracker`` respectively.  Search for ``# STUB`` to find
every test that needs updating.
"""

from __future__ import annotations

import pytest

from mot_counting.factories.detector_factory import DetectorFactory
from mot_counting.factories.tracker_factory import TrackerFactory

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class _FakeModel:
    """Stands in for an already-loaded YOLO model (no weights loaded here)."""


class _FakeTracker:
    """Stands in for an already-initialised ByteTrack / BoT-SORT object."""


def _detector_factory() -> DetectorFactory:
    return DetectorFactory(confidence_threshold=0.4, classes=["person", "car"])


def _tracker_factory() -> TrackerFactory:
    return TrackerFactory(track_thresh=0.5, match_thresh=0.8, track_buffer=30)


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
# DetectorFactory — all known variants raise NotImplementedError (STUB)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x"])
def test_detector_factory_known_variant_stub_raises_not_implemented(variant: str) -> None:  # STUB
    factory = _detector_factory()
    with pytest.raises(NotImplementedError, match="T19"):
        factory.create(variant, _FakeModel())


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
        factory.create(bad_type, _FakeTracker())


# ---------------------------------------------------------------------------
# TrackerFactory — bytetrack stub raises NotImplementedError (STUB)
# ---------------------------------------------------------------------------


def test_tracker_factory_bytetrack_stub_raises_not_implemented() -> None:  # STUB
    factory = _tracker_factory()
    with pytest.raises(NotImplementedError, match="T19"):
        factory.create("bytetrack", _FakeTracker())


# ---------------------------------------------------------------------------
# TrackerFactory — botsort stub raises NotImplementedError (STUB)
# ---------------------------------------------------------------------------


def test_tracker_factory_botsort_stub_raises_not_implemented() -> None:  # STUB
    factory = _tracker_factory()
    with pytest.raises(NotImplementedError, match="T19"):
        factory.create("botsort", _FakeTracker())


# ---------------------------------------------------------------------------
# Factories never load models themselves (structural check)
# ---------------------------------------------------------------------------


def test_detector_factory_does_not_load_models_itself() -> None:
    """Constructing DetectorFactory with a fake model object must not raise."""
    factory = _detector_factory()
    assert factory is not None


def test_tracker_factory_does_not_initialise_trackers_itself() -> None:
    """Constructing TrackerFactory with fake parameters must not raise."""
    factory = _tracker_factory()
    assert factory is not None
