"""Crossing logic state machine (§7.4, §10.5).

This is the heart of the pipeline.  Every rule below is locked in §7.4 and
must not be changed without a full team discussion, since the evaluation
script (T18) and the published magazine article metrics depend on the exact
behavior documented here.

Known limitation — track ID switches (§7.4, §13)
-------------------------------------------------
If ByteTrack loses a track due to occlusion and later re-assigns a new
``track_id`` to the same physical object, the new ID begins with no crossing
state and a fresh ``confirmed_side``.  This can produce a duplicate count
(the new track "crosses" the line when first seen on the far side) or a
missed count (the crossing happens during the gap).  This failure mode is
documented as an explicit limitation — it is **not** patched silently here
and is expected to surface as a "track fragmentation / occlusion-induced
error" failure case during evaluation (§13).

Cooldown and confirmed_side flip policy (§7.4 step 5)
------------------------------------------------------
When a decisive opposite majority is observed while a ``(track_id, line_id)``
pair is in cooldown, no event is emitted **and ``confirmed_side`` is not
flipped**.  This is intentional: if the raw side were allowed to flip during
cooldown, rapid oscillation near the line could accumulate a queue of
suppressed-but-pending state changes.  ``confirmed_side`` only flips when a
fresh decisive majority is observed *after* cooldown has expired.

FPS and time-based quantities
------------------------------
Cooldown and stale-track timeout are expressed in the configuration in
seconds and converted to frame counts once at construction using the video
FPS supplied by the caller (``IFrameSource.get_fps()``).  Using frame counts
internally is consistent with the frame-indexed processing model and avoids
repeated floating-point conversions per frame.
"""

from __future__ import annotations

import logging
import math
from collections import Counter, deque
from dataclasses import dataclass, field

from mot_counting.config import CrossingLogicConfig, LineConfig
from mot_counting.interfaces.crossing import ICrossingLogic
from mot_counting.types import CrossingEvent, Direction, Track
from mot_counting.utils.geometry import (
    get_bbox_center,
    get_bottom_center,
    get_side,
    signed_distance,
)

logger = logging.getLogger(__name__)

# Type alias for a single entry in the raw-history sliding window.
# Stores side (1 or -1), class_name, and class_id together so one window
# serves both side-confirmation and class-attribution (§7.4).
_WindowEntry = tuple[int, str, int]  # (side, class_name, class_id)


@dataclass
class _PairState:
    """Per-(track_id, line_id) crossing state."""

    # Sliding window of (side, class_name, class_id) entries.
    window: deque[_WindowEntry]

    # Stably confirmed side — maintained separately from the raw window
    # so that oscillation near the line does not cause false re-confirmations.
    # None = this pair has never been observed (no event ever emitted).
    confirmed_side: int | None

    # Remaining frames before another event may be emitted on this pair.
    cooldown_frames_remaining: int

    # Frame index of the most recent observation — used for stale cleanup.
    last_seen_frame_idx: int

    # Reference-point position at the time of the last emitted event.
    # Used by the optional displacement/velocity safeguards.
    last_event_ref_point: tuple[float, float] | None = field(default=None)

    # Frame index of the last emitted event — used for velocity calculation.
    last_event_frame_idx: int | None = field(default=None)


class CrossingLogic(ICrossingLogic):
    """Concrete crossing-logic state machine implementing ``ICrossingLogic``.

    Args:
        lines: Configured counting lines from ``AppConfig.lines``.
        config: Crossing-logic parameters from ``AppConfig.crossing_logic``.
        fps: Video frame rate in frames per second, provided by
            ``IFrameSource.get_fps()`` after the video is opened.
    """

    def __init__(
        self,
        lines: list[LineConfig],
        config: CrossingLogicConfig,
        fps: float,
    ) -> None:
        self._lines = lines
        self._config = config
        self._fps = max(fps, 1.0)  # guard against 0-FPS to avoid division errors
        self._history_length = config.history_length
        self._threshold = config.confirmation_majority_threshold

        # Pre-compute frame-count equivalents of time-based config values.
        self._cooldown_frames: int = max(0, round(config.cooldown_seconds * self._fps))
        self._stale_timeout_frames: int = max(
            1, round(config.stale_track_timeout_seconds * self._fps)
        )

        # State keyed by (track_id, line_id).
        self._state: dict[tuple[int, str], _PairState] = {}

        # Running count totals — updated incrementally, never recomputed.
        self._counters: dict[tuple[str, str, Direction], int] = {}

    # ------------------------------------------------------------------
    # ICrossingLogic interface
    # ------------------------------------------------------------------

    def process(
        self,
        tracks: list[Track],
        frame_idx: int,
        timestamp_seconds: float,
    ) -> list[CrossingEvent]:
        """Update crossing state for all active tracks and emit new events.

        Performs stale-entry cleanup first, then processes every active
        track against every configured counting line.

        Args:
            tracks: Active tracks for the current frame.
            frame_idx: Zero-based index of the current video frame.
            timestamp_seconds: Elapsed time from video start in seconds.

        Returns:
            A (possibly empty) list of newly validated
            :class:`~mot_counting.types.CrossingEvent` objects.
        """
        self._cleanup_stale_entries(frame_idx)

        events: list[CrossingEvent] = []
        for track in tracks:
            for line in self._lines:
                event = self._process_pair(track, line, frame_idx, timestamp_seconds)
                if event is not None:
                    events.append(event)

        return events

    def get_counters(self) -> dict:
        """Return a snapshot of running count totals.

        Returns:
            A dict keyed by ``(class_name, line_id, direction)`` → int count.
        """
        return dict(self._counters)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cleanup_stale_entries(self, frame_idx: int) -> None:
        """Remove state entries not updated for longer than the stale timeout."""
        stale = [
            key
            for key, state in self._state.items()
            if (frame_idx - state.last_seen_frame_idx) > self._stale_timeout_frames
        ]
        for key in stale:
            logger.debug("Stale state removed: track_id=%d line_id=%s.", key[0], key[1])
            del self._state[key]

    def _get_reference_point(self, bbox: tuple[float, float, float, float]) -> tuple[float, float]:
        """Return the configured reference point for a bounding box."""
        if self._config.reference_point == "bottom_center":
            return get_bottom_center(bbox)
        return get_bbox_center(bbox)

    @staticmethod
    def _determine_direction(
        old_side: int,
        new_side: int,
        positive_direction: str,
    ) -> Direction:
        """Map a side flip onto ``Direction.IN`` or ``Direction.OUT``.

        Per §7.4, ``positive_direction = "A_to_B"`` means the crossing that
        takes the reference point from the negative side (signed_distance < 0,
        i.e., right of the directed A→B vector) to the positive side
        (signed_distance > 0, i.e., left of A→B) is labeled **IN**.
        ``"B_to_A"`` inverts this.

        Args:
            old_side: The ``confirmed_side`` before the flip (-1 or +1).
            new_side: The decisive majority side after the flip (-1 or +1).
            positive_direction: ``"A_to_B"`` or ``"B_to_A"`` from the line
                configuration.

        Returns:
            :class:`~mot_counting.types.Direction` value for the event.
        """
        if positive_direction == "A_to_B":
            # Moving to positive side (+1) = IN.
            return Direction.IN if new_side == 1 else Direction.OUT
        else:  # "B_to_A"
            # Moving to negative side (-1) = IN.
            return Direction.IN if new_side == -1 else Direction.OUT

    def _check_safeguards(
        self,
        state: _PairState,
        ref_point: tuple[float, float],
        line: LineConfig,
        frame_idx: int,
    ) -> bool:
        """Return True if optional displacement/velocity safeguards pass.

        Both safeguards are **disabled by default** (``None`` in config).
        When disabled they cost nothing at runtime.  When enabled they
        suppress crossings that are likely jitter artefacts near the line
        (§7.4).

        Args:
            state: Current ``_PairState`` for this ``(track_id, line_id)`` pair.
            ref_point: Current reference-point coordinates.
            line: The counting-line configuration.
            frame_idx: Current frame index.

        Returns:
            ``True`` if the crossing should proceed.
        """
        # --- Minimum displacement ---
        if self._config.min_displacement_px is not None and state.last_event_ref_point is not None:
            dx = ref_point[0] - state.last_event_ref_point[0]
            dy = ref_point[1] - state.last_event_ref_point[1]
            displacement = math.sqrt(dx * dx + dy * dy)
            if displacement < self._config.min_displacement_px:
                logger.debug(
                    "Crossing suppressed (displacement %.1f < min %.1f px) line_id=%s.",
                    displacement,
                    self._config.min_displacement_px,
                    line.line_id,
                )
                return False

        # --- Minimum velocity perpendicular to the line ---
        if (
            self._config.min_velocity_px_per_s is not None
            and state.last_event_ref_point is not None
            and state.last_event_frame_idx is not None
        ):
            frames_elapsed = frame_idx - state.last_event_frame_idx
            if frames_elapsed > 0:
                time_s = frames_elapsed / self._fps
                ax = float(line.point_a[0])
                ay = float(line.point_a[1])
                bx = float(line.point_b[0])
                by = float(line.point_b[1])
                line_len = math.sqrt((bx - ax) ** 2 + (by - ay) ** 2)

                if line_len > 0:
                    # Perpendicular distance from line (normalized to pixels).
                    d_now = signed_distance((ax, ay), (bx, by), ref_point) / line_len
                    d_prev = (
                        signed_distance((ax, ay), (bx, by), state.last_event_ref_point) / line_len
                    )
                    perp_velocity = abs(d_now - d_prev) / time_s
                    if perp_velocity < self._config.min_velocity_px_per_s:
                        logger.debug(
                            "Crossing suppressed (velocity %.1f < min %.1f px/s) line_id=%s.",
                            perp_velocity,
                            self._config.min_velocity_px_per_s,
                            line.line_id,
                        )
                        return False

        return True

    def _process_pair(
        self,
        track: Track,
        line: LineConfig,
        frame_idx: int,
        timestamp_seconds: float,
    ) -> CrossingEvent | None:
        """Execute the per-frame state-machine logic for one (track, line) pair.

        Implements §7.4 steps 1–5 exactly:

        1. Compute reference point and raw side.
        2. Append (side, class_name, class_id) to the sliding window.
        3. Initialize ``confirmed_side`` on first observation — never emit.
        4. Check for a decisive majority opposite to ``confirmed_side``.
        5. Emit event + update state, or suppress (cooldown / safeguards).

        Args:
            track: The active track for this frame.
            line: The counting line to test against.
            frame_idx: Zero-based current frame index.
            timestamp_seconds: Elapsed seconds from video start.

        Returns:
            A :class:`~mot_counting.types.CrossingEvent` if a crossing is
            confirmed this frame, otherwise ``None``.
        """
        key = (track.track_id, line.line_id)

        ref_point = self._get_reference_point(track.bbox)
        dist = signed_distance(
            (float(line.point_a[0]), float(line.point_a[1])),
            (float(line.point_b[0]), float(line.point_b[1])),
            ref_point,
        )
        raw_side = get_side(dist)

        # Ensure a state entry exists.
        if key not in self._state:
            self._state[key] = _PairState(
                window=deque(),
                confirmed_side=None,
                cooldown_frames_remaining=0,
                last_seen_frame_idx=frame_idx,
            )

        state = self._state[key]
        state.last_seen_frame_idx = frame_idx

        # Side == 0: reference point is exactly on the line — indeterminate.
        # Do not update window or crossing state, but mark as seen.
        if raw_side == 0:
            return None

        # Append to window; drop oldest entry when full.
        state.window.append((raw_side, track.class_name, track.class_id))
        if len(state.window) > self._history_length:
            state.window.popleft()

        # Decrement cooldown counter.
        if state.cooldown_frames_remaining > 0:
            state.cooldown_frames_remaining -= 1

        # Compute majority side and whether it is decisive.
        side_counts = Counter(e[0] for e in state.window)
        majority_side, majority_count = side_counts.most_common(1)[0]
        majority_fraction = majority_count / len(state.window)
        is_decisive = majority_fraction >= self._threshold

        # --- §7.4 step 3: initialize on first observation, never emit ---
        if state.confirmed_side is None:
            state.confirmed_side = majority_side
            return None

        # --- §7.4 step 4: crossing only on decisive opposite majority ---
        if not is_decisive or majority_side == state.confirmed_side:
            return None

        # --- §7.4 step 5: in cooldown → suppress, do NOT flip ---
        if state.cooldown_frames_remaining > 0:
            # confirmed_side intentionally stays unchanged (§7.4 step 5).
            return None

        # --- Optional safeguards (disabled by default) ---
        if not self._check_safeguards(state, ref_point, line, frame_idx):
            return None

        # --- Emit crossing event ---

        # Class attribution: majority class over the window (§7.4).
        class_pair_counts: Counter[tuple[str, int]] = Counter((e[1], e[2]) for e in state.window)
        majority_class_name, majority_class_id = class_pair_counts.most_common(1)[0][0]

        direction = self._determine_direction(
            state.confirmed_side, majority_side, line.positive_direction
        )

        event = CrossingEvent(
            frame_idx=frame_idx,
            timestamp_seconds=timestamp_seconds,
            track_id=track.track_id,
            class_id=majority_class_id,
            class_name=majority_class_name,
            direction=direction,
            line_id=line.line_id,
            confidence=track.score,  # from this frame's Track (§7.5)
            bbox=track.bbox,  # from this frame's Track (§7.5)
        )

        # Update state.
        state.confirmed_side = majority_side
        state.cooldown_frames_remaining = self._cooldown_frames
        state.last_event_ref_point = ref_point
        state.last_event_frame_idx = frame_idx

        # Update running counters incrementally.
        counter_key = (majority_class_name, line.line_id, direction)
        self._counters[counter_key] = self._counters.get(counter_key, 0) + 1

        logger.debug(
            "Crossing event: frame=%d track_id=%d line_id=%s direction=%s class=%s.",
            frame_idx,
            track.track_id,
            line.line_id,
            direction.value,
            majority_class_name,
        )

        return event
