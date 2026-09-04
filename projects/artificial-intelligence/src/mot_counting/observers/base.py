"""Observer pattern base classes for the MOT counting pipeline (§4.3).

Design rationale — why the Observer pattern is used here and *not* for the
core pipeline sequence
----------------------------------------------------------------------
The core frame-processing sequence ``read → detect → track → update crossing
state`` is a **hard, ordered data dependency**: each stage's output is the
next stage's mandatory input.  Modelling that chain as independent observers
would not decouple anything — it would only hide the ordering requirement
behind a different mechanism while making the flow harder to read and debug.
That sequence therefore lives as a plain synchronous method call chain inside
``PipelineController``.

The Observer pattern is applied only to **side-effect consumers**: once the
crossing state for a frame has been updated, the Controller notifies all
subscribed observers (``Logger``, ``Visualizer``, and any future additions
such as a trajectory recorder).  These consumers have *no* ordering dependency
on each other and genuinely benefit from being pluggable — new consumers can
be added or removed without touching the Controller at all.

Exception handling policy
----------------------------------------------------------------------
``Subject.notify()`` propagates the **first exception** raised by any
observer and **stops notifying** the remaining observers in that call.

Rationale: in a synchronous, single-threaded pipeline, silently swallowing
exceptions from observers would hide bugs and allow silent data loss (e.g. a
broken ``Logger`` that never writes events).  A raised exception is the
fastest, most obvious signal that something is wrong.  It is the
*Controller's* responsibility to decide whether to abort the pipeline or
catch and log the error — not ``Subject``'s.  This policy is tested
explicitly in ``tests/unit/test_observers.py``.

Subscription duplicates
----------------------------------------------------------------------
``Subject.subscribe()`` allows the same observer instance to be subscribed
more than once.  Each subscription adds an independent entry to the internal
list, so a duplicated observer will receive ``notify()`` calls once per
subscription.  Callers are responsible for avoiding unintended duplicates.
This is simpler than a deduplication policy (which would require identity
checks and explicit error handling for attempted duplicates) and is tested
explicitly.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from mot_counting.types import CrossingEvent, Track

logger = logging.getLogger(__name__)


class Observer(ABC):
    """Abstract base class for all per-frame side-effect consumers.

    Concrete implementations (``Logger``, ``Visualizer``, …) must implement
    :meth:`update`.  Attempting to instantiate a subclass that omits
    :meth:`update` raises ``TypeError`` immediately, giving junior developers
    a clear, actionable error message (§4.2).
    """

    @abstractmethod
    def update(
        self,
        frame_idx: int,
        tracks: list[Track],
        events: list[CrossingEvent],
        counters: dict,
    ) -> None:
        """Receive a per-frame notification from the pipeline.

        Called by :class:`Subject` after the crossing state has been updated
        for the current frame.

        Args:
            frame_idx: Zero-based index of the current video frame.
            tracks: Active tracks returned by the tracker for this frame.
            events: Crossing events (possibly empty) emitted by the crossing
                logic for this frame.
            counters: Running count totals keyed by
                ``(class_name, line_id, direction)``, as returned by
                :meth:`~mot_counting.interfaces.crossing.ICrossingLogic.get_counters`.
        """


class Subject:
    """Concrete subject that maintains a list of observers and notifies them.

    The ``PipelineController`` owns a single ``Subject`` instance and calls
    :meth:`notify` once per frame after the crossing state has been updated.
    """

    def __init__(self) -> None:
        self._observers: list[Observer] = []

    def subscribe(self, observer: Observer) -> None:
        """Add *observer* to the notification list.

        Duplicate subscriptions are allowed; the same observer instance will
        receive :meth:`~Observer.update` once per subscription per
        :meth:`notify` call.  See the module docstring for the rationale.

        Args:
            observer: An :class:`Observer` instance to add.
        """
        self._observers.append(observer)

    def notify(
        self,
        frame_idx: int,
        tracks: list[Track],
        events: list[CrossingEvent],
        counters: dict,
    ) -> None:
        """Call :meth:`~Observer.update` on every subscribed observer.

        Observers are notified in subscription order.  If an observer raises
        an exception, that exception propagates immediately and subsequent
        observers in the list are **not** notified.  See the module docstring
        for the exception-handling policy rationale.

        Args:
            frame_idx: Zero-based index of the current video frame.
            tracks: Active tracks for this frame.
            events: Crossing events emitted this frame (possibly empty).
            counters: Running count totals keyed by
                ``(class_name, line_id, direction)``.
        """
        for observer in self._observers:
            observer.update(
                frame_idx=frame_idx,
                tracks=tracks,
                events=events,
                counters=counters,
            )
