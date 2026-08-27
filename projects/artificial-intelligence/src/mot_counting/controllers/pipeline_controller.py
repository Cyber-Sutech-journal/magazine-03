"""Pipeline controller (§4.4, §7.3, §8, §10.1).

Owns the high-level orchestration for the video analytics pipeline.  It
contains almost no business logic — only sequencing — and receives every
dependency via constructor injection.

Core sequence (synchronous, not implemented via Observers — see §4.3 and T07
for the rationale):

    read → detect → track → update crossing state → notify observers

The Observer pattern applies *only* to side-effect consumers (Logger,
Visualizer) that are notified once per frame after the crossing state has been
updated.  The core sequence itself is a plain method call chain.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from mot_counting.config import AppConfig
from mot_counting.interfaces.crossing import ICrossingLogic
from mot_counting.interfaces.detector import IDetector
from mot_counting.interfaces.frame_source import IFrameSource
from mot_counting.interfaces.repository import IEventRepository
from mot_counting.interfaces.tracker import ITracker
from mot_counting.interfaces.visualizer import IVisualizer
from mot_counting.observers.base import Subject

logger = logging.getLogger(__name__)


@dataclass
class RunStats:
    """Runtime statistics collected during a pipeline run.

    Attributes:
        frames_processed: Number of frames successfully decoded and processed.
        frames_skipped: Number of frames where ``IFrameSource.read()`` returned
            ``(False, ...)`` mid-stream (not counting natural end-of-video).
        elapsed_seconds: Wall-clock time from pipeline start to completion.
        average_fps: ``frames_processed / elapsed_seconds``, or ``0.0`` if
            ``elapsed_seconds`` is zero.  Informational only — never a
            pass/fail gate (§1).
        final_counters: Snapshot of ``ICrossingLogic.get_counters()`` taken
            immediately after the loop terminates.
    """

    frames_processed: int = 0
    frames_skipped: int = 0
    elapsed_seconds: float = 0.0
    average_fps: float = 0.0
    final_counters: dict = field(default_factory=dict)


class LineGeometryError(ValueError):
    """Raised when a configured line point falls outside the video frame (§7.3, §10.1)."""


class PipelineController:
    """Orchestrates the video analytics pipeline via constructor injection.

    All dependencies are interface-typed.  The controller must never import
    or depend on concrete detector/tracker implementations directly (§4.2).

    Lifecycle::

        controller.run()     # validates geometry, runs loop, calls cleanup()
        controller.stop()    # requests graceful termination from another thread
        stats = controller.stats  # available after run() returns
    """

    def __init__(
        self,
        config: AppConfig,
        frame_source: IFrameSource,
        detector: IDetector,
        tracker: ITracker,
        crossing_logic: ICrossingLogic,
        event_repository: IEventRepository,
        visualizer: IVisualizer,
        subject: Subject,
    ) -> None:
        """Initialise the controller with all pipeline dependencies.

        Args:
            config: Validated application configuration.
            frame_source: Video frame reader.
            detector: Object detector (``IDetector``).
            tracker: Multi-object tracker (``ITracker``).
            crossing_logic: Line-crossing state machine (``ICrossingLogic``).
            event_repository: Crossing-event persistence (``IEventRepository``).
            visualizer: Frame annotator (``IVisualizer``).
            subject: Observer subject for Logger/Visualizer side effects.
        """
        self._config = config
        self._frame_source = frame_source
        self._detector = detector
        self._tracker = tracker
        self._crossing_logic = crossing_logic
        self._event_repository = event_repository
        self._visualizer = visualizer
        self._subject = subject

        self._stop_requested: bool = False
        self._stats: RunStats = RunStats()

    # ------------------------------------------------------------------
    # Public read-only properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> AppConfig:
        """Return the validated application configuration."""
        return self._config

    @property
    def subject(self) -> Subject:
        """Return the observer subject for side-effect consumers."""
        return self._subject

    @property
    def stats(self) -> RunStats:
        """Return runtime statistics; populated only after :meth:`run` returns."""
        return self._stats

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Request graceful termination after the current frame completes."""
        self._stop_requested = True

    def cleanup(self) -> None:
        """Release all resources regardless of how the loop exited.

        Calls ``IFrameSource.release()`` and ``IEventRepository.flush()`` /
        ``IEventRepository.close()`` in a best-effort manner.
        """
        try:
            self._frame_source.release()
        except Exception:
            logger.exception("Error releasing frame source during cleanup.")
        try:
            self._event_repository.flush()
            self._event_repository.close()
        except Exception:
            logger.exception("Error flushing / closing event repository during cleanup.")

    def run(self) -> None:
        """Execute the full video analytics pipeline.

        Steps performed once before the loop:

        1. Query ``IFrameSource.get_frame_size()`` to obtain actual frame
           dimensions.
        2. Validate every configured line's ``point_a`` / ``point_b`` against
           those dimensions (§7.3, §10.1).  Raises :class:`LineGeometryError`
           immediately if any point falls outside ``[0, width) × [0, height)``.

        Then drives the synchronous frame loop:

            ``read → detect → track → update crossing state → notify observers``

        On a mid-stream decode failure (``read()`` returns ``(False, ...)``
        before natural end-of-video is detected), logs a warning and continues
        to the next frame rather than aborting the run.

        ``cleanup()`` is guaranteed to run via ``finally`` even if the loop
        raises an unexpected exception.

        Raises:
            LineGeometryError: If a configured line point falls outside the
                actual frame dimensions.
        """
        self._stop_requested = False
        self._stats = RunStats()
        start_time = time.monotonic()

        try:
            self._validate_line_geometry()
            self._run_loop()
        finally:
            self._stats.elapsed_seconds = time.monotonic() - start_time
            processed = self._stats.frames_processed
            elapsed = self._stats.elapsed_seconds
            self._stats.average_fps = processed / elapsed if elapsed > 0 else 0.0
            self._stats.final_counters = self._crossing_logic.get_counters()
            self.cleanup()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_line_geometry(self) -> None:
        """Validate all configured counting-line points against frame dimensions.

        Called once, immediately after the frame source is available, before
        any frame is processed (§7.3, §10.1).

        Raises:
            LineGeometryError: If any line endpoint falls outside the frame.
        """
        width, height = self._frame_source.get_frame_size()
        for line in self._config.lines:
            for label, point in (("point_a", line.point_a), ("point_b", line.point_b)):
                x, y = point[0], point[1]
                if not (0 <= x < width and 0 <= y < height):
                    raise LineGeometryError(
                        f"Line '{line.line_id}' {label} ({x}, {y}) is outside the "
                        f"video frame ({width}×{height} px).  "
                        f"Valid range: x in [0, {width}), y in [0, {height})."
                    )

    def _run_loop(self) -> None:
        """Drive the synchronous per-frame processing loop."""
        fps = self._frame_source.get_fps()

        frame_idx = 0
        while not self._stop_requested:
            success, frame = self._frame_source.read()

            if frame is None and not success:
                # Natural end-of-video: terminate normally.
                break

            if not success or frame is None:
                # Mid-stream decode failure: warn and skip this frame.
                logger.warning("Frame decode failure at frame_idx=%d — skipping.", frame_idx)
                self._stats.frames_skipped += 1
                frame_idx += 1
                continue

            timestamp_seconds = frame_idx / fps if fps > 0 else 0.0

            # Core sequence — plain synchronous calls, intentionally not
            # implemented via Observers (see §4.3 and T07 rationale).
            detections = self._detector.predict(frame)
            tracks = self._tracker.update(detections, frame_idx, frame)
            events = self._crossing_logic.process(tracks, frame_idx, timestamp_seconds)

            for event in events:
                self._event_repository.save(event)

            self._subject.notify(
                frame_idx=frame_idx,
                tracks=tracks,
                events=events,
                counters=self._crossing_logic.get_counters(),
            )

            self._stats.frames_processed += 1
            frame_idx += 1
