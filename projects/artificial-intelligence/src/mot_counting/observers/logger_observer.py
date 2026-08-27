"""Concrete logging observer for the MOT counting pipeline (§4.3, §10.8, §12.2).

``LoggerObserver`` subscribes to the pipeline :class:`~mot_counting.observers.base.Subject`
and records lifecycle and per-frame information using the standard library
``logging`` module only.

Logging policy (§12.2)
-----------------------
- ``INFO``    — pipeline run start/stop and high-level per-frame summaries
                (e.g. crossing events emitted).
- ``WARNING`` — per-frame recoverable issues surfaced by the pipeline
                (via :meth:`log_frame_warning`).
- ``DEBUG``   — granular per-frame crossing-decision tracing; gated by the
                logger's effective level so production runs at ``INFO`` are
                not flooded.

Global logging configuration (handlers, formatters, root level) is performed
**once** at pipeline startup in the composition root (T19).  This class calls
``logging.getLogger(__name__)`` and never reconfigures global logging state.
"""

from __future__ import annotations

import logging

from mot_counting.observers.base import Observer
from mot_counting.types import CrossingEvent, Track


class LoggerObserver(Observer):
    """Observer that logs pipeline lifecycle and per-frame crossing activity."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Initialise the observer.

        Args:
            logger: Optional logger instance.  Defaults to
                ``logging.getLogger(__name__)``.  Inject a custom logger in
                tests; production wiring uses the module logger configured
                once at startup (T19).
        """
        self._logger = logger if logger is not None else logging.getLogger(__name__)

    def log_run_start(self, *, video_path: str) -> None:
        """Log pipeline run start at ``INFO`` level.

        Called once by the composition root or controller before the frame
        loop begins (not part of the per-frame ``Observer.update`` contract).

        Args:
            video_path: Path to the input video being processed.
        """
        self._logger.info("Pipeline run started — video: %s", video_path)

    def log_run_stop(
        self,
        *,
        frames_processed: int,
        elapsed_seconds: float,
        average_fps: float,
    ) -> None:
        """Log pipeline run completion at ``INFO`` level.

        Called once after the frame loop terminates.

        Args:
            frames_processed: Total frames successfully processed.
            elapsed_seconds: Wall-clock runtime in seconds.
            average_fps: Informational throughput metric (§1, §11).
        """
        self._logger.info(
            "Pipeline run stopped — frames=%d elapsed=%.2fs avg_fps=%.2f",
            frames_processed,
            elapsed_seconds,
            average_fps,
        )

    def log_frame_warning(self, frame_idx: int, message: str) -> None:
        """Log a per-frame recoverable issue at ``WARNING`` level (§12.2).

        Intended for issues such as decode failures that the controller
        surfaces without aborting the run.  The controller may call this
        directly or continue logging via its own module logger until T19
        centralises warning routing here.

        Args:
            frame_idx: Zero-based index of the affected frame.
            message: Human-readable description of the recoverable issue.
        """
        self._logger.warning("Frame %d: %s", frame_idx, message)

    def update(
        self,
        frame_idx: int,
        tracks: list[Track],
        events: list[CrossingEvent],
        counters: dict,
    ) -> None:
        """Log per-frame pipeline state after crossing logic has run.

        Emits ``DEBUG`` records for granular crossing-decision tracing and an
        ``INFO`` summary when one or more crossing events are emitted this
        frame.  ``DEBUG`` output is automatically suppressed when the logger's
        effective level is ``INFO`` or higher.

        Args:
            frame_idx: Zero-based index of the current video frame.
            tracks: Active tracks for this frame.
            events: Crossing events emitted this frame (possibly empty).
            counters: Running count totals keyed by
                ``(class_name, line_id, direction)``.
        """
        self._logger.debug(
            "frame=%d active_tracks=%d events=%d counters=%r",
            frame_idx,
            len(tracks),
            len(events),
            counters,
        )

        for event in events:
            self._logger.debug(
                "Crossing decision — frame=%d track_id=%d line_id=%s "
                "direction=%s class=%s confidence=%s",
                event.frame_idx,
                event.track_id,
                event.line_id,
                event.direction.value,
                event.class_name,
                event.confidence,
            )

        if events:
            self._logger.info(
                "Frame %d: emitted %d crossing event(s)",
                frame_idx,
                len(events),
            )
