"""Factory for creating :class:`~mot_counting.interfaces.tracker.ITracker` instances (§4.1).

Design contract
---------------
``TrackerFactory`` is intentionally trivial.  The composition root is
responsible for constructing (or lazily initialising) the underlying tracker
object.  The Factory's only job is to wrap that already-initialised object
into the correct concrete ``ITracker`` implementation and return it.

No tracker-initialisation code may ever live inside this Factory.
"""

from __future__ import annotations

from mot_counting.interfaces.tracker import ITracker

# Tracker type strings accepted in the configuration (§3, §7.7).
_SUPPORTED_TRACKER_TYPES: frozenset[str] = frozenset({"bytetrack", "botsort"})


class TrackerFactory:
    """Wraps an already-initialised tracker object into a concrete ``ITracker``.

    Usage (composition root)::

        raw_tracker = BYTETracker(args)       # initialised once, outside this Factory
        factory = TrackerFactory(config)
        tracker: ITracker = factory.create("bytetrack", raw_tracker)
    """

    def __init__(
        self,
        track_thresh: float,
        match_thresh: float,
        track_buffer: int,
        frame_rate: int = 30,
    ) -> None:
        """Initialise the factory with tracker hyperparameters.

        Args:
            track_thresh: Detection confidence threshold for new track
                initialisation.
            match_thresh: IoU threshold used for track–detection association.
            track_buffer: Number of frames a lost track is kept alive before
                removal.
            frame_rate: Video frame rate in FPS, passed to ByteTrackWrapper so
                that its internal track buffer is calibrated to the actual clip
                speed.  Defaults to 30.
        """
        self._track_thresh = track_thresh
        self._match_thresh = match_thresh
        self._track_buffer = track_buffer
        self._frame_rate = frame_rate

    def create(self, tracker_type: str, loaded_tracker: object) -> ITracker:
        """Wrap *loaded_tracker* into the ``ITracker`` matching *tracker_type*.

        Args:
            tracker_type: Tracker type string from the configuration (e.g.
                ``"bytetrack"``).  Must be one of the supported types;
                anything else raises ``ValueError`` for fail-fast startup
                (§12.1).
            loaded_tracker: An already-initialised tracker object provided by
                the composition root.  This Factory must never instantiate or
                configure a tracker itself (§4.1).

        Returns:
            A concrete ``ITracker`` implementation wrapping *loaded_tracker*.

        Raises:
            ValueError: If *tracker_type* is not a recognised tracker type.
            NotImplementedError: Temporary — raised until the concrete wrapper
                classes are wired in T19.
        """
        if tracker_type not in _SUPPORTED_TRACKER_TYPES:
            raise ValueError(
                f"Unknown tracker type {tracker_type!r}.  "
                f"Supported types: {sorted(_SUPPORTED_TRACKER_TYPES)}."
            )

        if tracker_type == "bytetrack":
            from mot_counting.trackers.bytetrack_tracker import ByteTrackWrapper

            return ByteTrackWrapper(
                frame_rate=self._frame_rate,
                track_thresh=self._track_thresh,
                match_thresh=self._match_thresh,
                track_buffer=self._track_buffer,
            )

        if tracker_type == "botsort":
            # TODO(T19): wire BoT-SORT wrapper here once/if it is implemented
            # as a stretch goal (§3, §14).
            raise NotImplementedError(
                "BoT-SORT tracker not yet wired — see T19 / §14.  "
                "This is a stretch goal; implement BoTSortWrapper if schedule allows."
            )

        # Unreachable given _SUPPORTED_TRACKER_TYPES above; guards future additions.
        raise ValueError(f"Unhandled tracker type {tracker_type!r}.")  # pragma: no cover
