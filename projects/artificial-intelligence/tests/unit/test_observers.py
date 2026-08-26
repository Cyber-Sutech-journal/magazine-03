"""Unit tests for Observer / Subject base classes (§4.3)."""

from __future__ import annotations

import pytest

from mot_counting.observers.base import Observer, Subject
from mot_counting.types import CrossingEvent, Direction, Track

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_TRACK = Track(track_id=1, bbox=(0.0, 0.0, 10.0, 10.0), class_id=0, class_name="person", score=0.9)
_EVENT = CrossingEvent(
    frame_idx=5,
    timestamp_seconds=0.5,
    track_id=1,
    class_id=0,
    class_name="person",
    direction=Direction.IN,
    line_id="main_line",
)
_COUNTERS: dict = {("person", "main_line", "IN"): 1}


class _RecordingObserver(Observer):
    """Captures every update() call for later assertion."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def update(
        self,
        frame_idx: int,
        tracks: list[Track],
        events: list[CrossingEvent],
        counters: dict,
    ) -> None:
        self.calls.append(
            {
                "frame_idx": frame_idx,
                "tracks": tracks,
                "events": events,
                "counters": counters,
            }
        )


class _RaisingObserver(Observer):
    """Always raises RuntimeError from update()."""

    def update(
        self,
        frame_idx: int,
        tracks: list[Track],
        events: list[CrossingEvent],
        counters: dict,
    ) -> None:
        raise RuntimeError("observer intentionally failed")


# ---------------------------------------------------------------------------
# Observer ABC enforcement
# ---------------------------------------------------------------------------


def test_observer_direct_instantiation_raises_type_error() -> None:
    with pytest.raises(TypeError):
        Observer()  # type: ignore[abstract]


def test_observer_subclass_missing_update_raises_type_error() -> None:
    class _Incomplete(Observer):
        pass

    with pytest.raises(TypeError):
        _Incomplete()


def test_complete_observer_subclass_can_be_instantiated() -> None:
    obs = _RecordingObserver()
    assert isinstance(obs, Observer)


# ---------------------------------------------------------------------------
# Subject.subscribe + Subject.notify
# ---------------------------------------------------------------------------


def test_notify_calls_all_subscribed_observers() -> None:
    obs_a = _RecordingObserver()
    obs_b = _RecordingObserver()
    subject = Subject()
    subject.subscribe(obs_a)
    subject.subscribe(obs_b)

    subject.notify(
        frame_idx=7,
        tracks=[_TRACK],
        events=[_EVENT],
        counters=_COUNTERS,
    )

    for obs in (obs_a, obs_b):
        assert len(obs.calls) == 1
        call = obs.calls[0]
        assert call["frame_idx"] == 7
        assert call["tracks"] == [_TRACK]
        assert call["events"] == [_EVENT]
        assert call["counters"] == _COUNTERS


def test_notify_calls_observers_in_subscription_order() -> None:
    order: list[int] = []

    class _OrderedObserver(Observer):
        def __init__(self, tag: int) -> None:
            self._tag = tag

        def update(self, frame_idx: int, tracks: list, events: list, counters: dict) -> None:
            order.append(self._tag)

    subject = Subject()
    subject.subscribe(_OrderedObserver(1))
    subject.subscribe(_OrderedObserver(2))
    subject.subscribe(_OrderedObserver(3))

    subject.notify(frame_idx=0, tracks=[], events=[], counters={})

    assert order == [1, 2, 3]


def test_no_observers_subscribed_notify_is_a_no_op() -> None:
    subject = Subject()
    subject.notify(frame_idx=0, tracks=[], events=[], counters={})  # must not raise


# ---------------------------------------------------------------------------
# Duplicate subscriptions (documented: allowed, each fires once per sub)
# ---------------------------------------------------------------------------


def test_subscribing_same_observer_twice_calls_it_twice() -> None:
    """Duplicate subscriptions are intentionally allowed (see base.py docstring)."""
    obs = _RecordingObserver()
    subject = Subject()
    subject.subscribe(obs)
    subject.subscribe(obs)

    subject.notify(frame_idx=0, tracks=[], events=[], counters={})

    assert len(obs.calls) == 2


# ---------------------------------------------------------------------------
# Exception handling policy (documented: propagates; stops remaining observers)
# ---------------------------------------------------------------------------


def test_raising_observer_propagates_exception() -> None:
    """An observer that raises must propagate the exception from notify()."""
    subject = Subject()
    subject.subscribe(_RaisingObserver())

    with pytest.raises(RuntimeError, match="intentionally failed"):
        subject.notify(frame_idx=0, tracks=[], events=[], counters={})


def test_raising_observer_stops_subsequent_observers() -> None:
    """Once an observer raises, the remaining observers must not be called."""
    after = _RecordingObserver()
    subject = Subject()
    subject.subscribe(_RaisingObserver())
    subject.subscribe(after)

    with pytest.raises(RuntimeError):
        subject.notify(frame_idx=0, tracks=[], events=[], counters={})

    assert len(after.calls) == 0
