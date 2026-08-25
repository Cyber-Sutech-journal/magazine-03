"""Crossing logic interface (§4.2, §6, §10.5).

Defines the ``ICrossingLogic`` contract for the custom line-crossing state
machine that determines IN/OUT direction and emits validated
:class:`~mot_counting.types.CrossingEvent` records.
"""

from abc import ABC, abstractmethod

from mot_counting.types import CrossingEvent, Track


class ICrossingLogic(ABC):
    """Abstract interface for virtual counting-line crossing detection."""

    @abstractmethod
    def process(
        self,
        tracks: list[Track],
        frame_idx: int,
        timestamp_seconds: float,
    ) -> list[CrossingEvent]:
        """Update crossing state for all active tracks and emit new events.

        Args:
            tracks: Active tracks returned by
                :class:`~mot_counting.interfaces.tracker.ITracker` for the
                current frame.
            frame_idx: Zero-based index of the current video frame.
            timestamp_seconds: Elapsed time from the start of the video in
                seconds.

        Returns:
            A (possibly empty) list of newly validated
            :class:`~mot_counting.types.CrossingEvent` objects emitted
            during this frame.
        """

    @abstractmethod
    def get_counters(self) -> dict:
        """Return current running count totals.

        Returns:
            A dictionary keyed by ``(class_name, line_id, direction)`` whose
            values are integer counts accumulated since the pipeline started.
        """
