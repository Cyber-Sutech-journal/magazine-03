"""Unit tests for core ``abc.ABC`` interfaces in ``mot_counting.interfaces``."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from mot_counting.interfaces.crossing import ICrossingLogic
from mot_counting.interfaces.detector import IDetector
from mot_counting.interfaces.frame_source import IFrameSource
from mot_counting.interfaces.repository import IEventRepository
from mot_counting.interfaces.tracker import ITracker
from mot_counting.interfaces.visualizer import IVisualizer
from mot_counting.types import CrossingEvent, Detection, Direction, Track

# ---------------------------------------------------------------------------
# Shared dummy implementations
# ---------------------------------------------------------------------------

_DUMMY_FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
_DUMMY_DETECTION = Detection(
    xyxy=(0.0, 0.0, 1.0, 1.0), confidence=1.0, class_id=0, class_name="person"
)
_DUMMY_TRACK = Track(
    track_id=1, bbox=(0.0, 0.0, 1.0, 1.0), class_id=0, class_name="person", score=1.0
)
_DUMMY_EVENT = CrossingEvent(
    frame_idx=0,
    timestamp_seconds=0.0,
    track_id=1,
    class_id=0,
    class_name="person",
    direction=Direction.IN,
    line_id="main_line",
)


class _DummyDetector(IDetector):
    def predict(self, frame: np.ndarray) -> list[Detection]:
        return [_DUMMY_DETECTION]


class _DummyTracker(ITracker):
    def update(
        self,
        detections: list[Detection],
        frame_idx: int,
        frame: np.ndarray,
    ) -> list[Track]:
        return [_DUMMY_TRACK]


class _DummyCrossingLogic(ICrossingLogic):
    def process(
        self,
        tracks: list[Track],
        frame_idx: int,
        timestamp_seconds: float,
    ) -> list[CrossingEvent]:
        return []

    def get_counters(self) -> dict:
        return {}


class _DummyEventRepository(IEventRepository):
    def save(self, event: CrossingEvent) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class _DummyVisualizer(IVisualizer):
    def draw(
        self,
        frame: np.ndarray,
        tracks: list[Track],
        lines: list,
        counters: dict,
    ) -> np.ndarray:
        return frame.copy()

    def update(
        self,
        frame_idx: int,
        tracks: list[Track],
        events: list[CrossingEvent],
        counters: dict,
    ) -> None:
        pass


class _DummyFrameSource(IFrameSource):
    def read(self) -> tuple[bool, np.ndarray | None]:
        return False, None

    def get_fps(self) -> float:
        return 30.0

    def get_frame_size(self) -> tuple[int, int]:
        return (640, 480)

    def release(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Registry of interfaces and their complete dummy subclasses
# ---------------------------------------------------------------------------

INTERFACE_CASES: list[tuple[type, type, list[str]]] = [
    (IDetector, _DummyDetector, ["predict"]),
    (ITracker, _DummyTracker, ["update"]),
    (ICrossingLogic, _DummyCrossingLogic, ["process", "get_counters"]),
    (IEventRepository, _DummyEventRepository, ["save", "flush", "close"]),
    (IVisualizer, _DummyVisualizer, ["draw", "update"]),
    (IFrameSource, _DummyFrameSource, ["read", "get_fps", "get_frame_size", "release"]),
]


# ---------------------------------------------------------------------------
# Direct instantiation must fail
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("interface_cls", "_dummy_cls", "_methods"),
    INTERFACE_CASES,
    ids=[cls.__name__ for cls, _, _ in INTERFACE_CASES],
)
def test_direct_instantiation_raises_type_error(
    interface_cls: type,
    _dummy_cls: type,
    _methods: list[str],
) -> None:
    with pytest.raises(TypeError):
        interface_cls()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Complete dummy subclass can be instantiated
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("interface_cls", "dummy_cls", "_methods"),
    INTERFACE_CASES,
    ids=[cls.__name__ for cls, _, _ in INTERFACE_CASES],
)
def test_complete_dummy_subclass_can_be_instantiated(
    interface_cls: type,
    dummy_cls: type,
    _methods: list[str],
) -> None:
    instance = dummy_cls()
    assert isinstance(instance, interface_cls)


# ---------------------------------------------------------------------------
# Subclass missing one method cannot be instantiated
# ---------------------------------------------------------------------------


def _make_incomplete_subclass(
    interface: type,
    complete_dummy: type,
    method_to_omit: str,
) -> type:
    """Build a subclass of *interface* that implements all but one abstract method."""
    namespace: dict[str, Any] = {
        name: getattr(complete_dummy, name)
        for name in interface.__abstractmethods__
        if name != method_to_omit
    }
    return type(f"Incomplete{interface.__name__}", (interface,), namespace)


@pytest.mark.parametrize(
    ("interface_cls", "dummy_cls", "methods"),
    INTERFACE_CASES,
    ids=[cls.__name__ for cls, _, _ in INTERFACE_CASES],
)
def test_incomplete_subclass_raises_type_error(
    interface_cls: type,
    dummy_cls: type,
    methods: list[str],
) -> None:
    for method in methods:
        incomplete = _make_incomplete_subclass(interface_cls, dummy_cls, method)
        with pytest.raises(TypeError):
            incomplete()


# ---------------------------------------------------------------------------
# ITracker.update() frame parameter docstring mentions BoT-SORT
# ---------------------------------------------------------------------------


def test_itracker_update_frame_docstring_mentions_botsort() -> None:
    doc = ITracker.update.__doc__ or ""
    assert "BoT-SORT" in doc
    assert "ByteTrack" in doc
    assert "frame" in doc.lower()
