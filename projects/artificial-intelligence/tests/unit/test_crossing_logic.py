"""Unit tests for CrossingLogic (§7.4, T11).

Test configuration
------------------
Most tests use ``history_length=3`` and ``threshold=0.7``.  With a 3-entry
window, a majority fraction is decisive only when ALL three entries agree
(3/3 = 1.0 ≥ 0.7).  Two agreeing entries (2/3 ≈ 0.667) are NOT decisive.
This makes the crossing trigger clean and predictable in test scenarios.

A simple crossing sequence therefore is:
    - 3 frames at side A  → confirmed_side initialized to A, no event.
    - 3 frames at side B  → window becomes [B,B,B], decisive → CROSSING.

Line geometry
-------------
All tests use a horizontal line from (0, 0) to (100, 0):
    signed_distance = (100-0)*(Py-0) - (0-0)*(Px-0) = 100*Py

    - ref_point y > 0  →  signed_distance > 0  →  side = +1  (below line)
    - ref_point y < 0  →  signed_distance < 0  →  side = -1  (above line)
    - positive_direction = "A_to_B" → crossing to side +1 = IN
"""

from __future__ import annotations

from mot_counting.config import CrossingLogicConfig, LineConfig
from mot_counting.crossing.crossing_logic import CrossingLogic
from mot_counting.types import Direction, Track

# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------

_FPS = 30.0
_HISTORY = 3  # all-agree threshold: 3/3 = 1.0 ≥ 0.7
_THRESHOLD = 0.7


def _make_config(
    *,
    history_length: int = _HISTORY,
    threshold: float = _THRESHOLD,
    cooldown_seconds: float = 0.0,
    stale_track_timeout_seconds: float = 2.0,
    min_displacement_px: float | None = None,
    min_velocity_px_per_s: float | None = None,
    reference_point: str = "bottom_center",
) -> CrossingLogicConfig:
    return CrossingLogicConfig(
        reference_point=reference_point,
        history_length=history_length,
        confirmation_majority_threshold=threshold,
        cooldown_seconds=cooldown_seconds,
        stale_track_timeout_seconds=stale_track_timeout_seconds,
        min_displacement_px=min_displacement_px,
        min_velocity_px_per_s=min_velocity_px_per_s,
    )


def _make_line(
    line_id: str = "line_a",
    point_a: list[int] | None = None,
    point_b: list[int] | None = None,
    positive_direction: str = "A_to_B",
) -> LineConfig:
    return LineConfig(
        line_id=line_id,
        point_a=point_a or [0, 0],
        point_b=point_b or [100, 0],
        positive_direction=positive_direction,
    )


def _make_logic(
    lines: list[LineConfig] | None = None,
    config: CrossingLogicConfig | None = None,
    fps: float = _FPS,
) -> CrossingLogic:
    return CrossingLogic(
        lines=lines or [_make_line()],
        config=config or _make_config(),
        fps=fps,
    )


def _make_track(
    *,
    track_id: int = 1,
    y: float,
    class_name: str = "person",
    class_id: int = 0,
    score: float = 0.9,
) -> Track:
    """Create a track whose bottom-center reference point is at (50, y).

    The horizontal test line is at y=0.
    y > 0 → side +1 (below line, IN for A_to_B positive_direction).
    y < 0 → side -1 (above line, OUT for A_to_B positive_direction).
    """
    # bbox: (x1, y1, x2, y2) — bottom-center = ((x1+x2)/2, y2) = (50, y)
    return Track(
        track_id=track_id,
        bbox=(0.0, float(y - 10), 100.0, float(y)),
        class_id=class_id,
        class_name=class_name,
        score=score,
    )


def _run_frames(
    logic: CrossingLogic,
    track_sequence: list[Track | None],
    *,
    fps: float = _FPS,
) -> list:
    """Run a sequence of frames through the logic.

    ``None`` in the sequence means "no track present this frame" (empty list).
    Returns a flat list of all CrossingEvent objects emitted.
    """
    all_events = []
    for frame_idx, track in enumerate(track_sequence):
        tracks = [] if track is None else [track]
        events = logic.process(
            tracks=tracks,
            frame_idx=frame_idx,
            timestamp_seconds=frame_idx / fps,
        )
        all_events.extend(events)
    return all_events


# ---------------------------------------------------------------------------
# 1. Jitter near the line — no false crossing when majority never decisive
# ---------------------------------------------------------------------------


def test_jitter_does_not_produce_false_crossing() -> None:
    """Rapid side oscillation must never trigger a crossing event."""
    logic = _make_logic()

    # Initialize confirmed_side = -1 with 3 solid frames above the line.
    init_sequence = [_make_track(y=-10)] * 3

    # Jitter: alternating sides for many frames — majority never reaches 3/3.
    jitter_sequence = [
        _make_track(y=+10),  # side +1
        _make_track(y=-10),  # side -1
        _make_track(y=+10),
        _make_track(y=-10),
        _make_track(y=+10),
        _make_track(y=-10),
        _make_track(y=+10),
    ]

    events = _run_frames(logic, init_sequence + jitter_sequence)
    assert events == [], f"Expected no events but got: {events}"


# ---------------------------------------------------------------------------
# 2. Clean sustained side change → exactly one CrossingEvent
# ---------------------------------------------------------------------------


def test_clean_sustained_crossing_emits_exactly_one_event() -> None:
    """A solid side change through the full window emits exactly one event."""
    logic = _make_logic()

    sequence = (
        [_make_track(y=-10)] * 3  # init: confirmed_side = -1
        + [_make_track(y=+10)] * 3  # fill window with +1 → decisive crossing
    )
    events = _run_frames(logic, sequence)

    assert len(events) == 1
    event = events[0]
    assert event.track_id == 1
    assert event.line_id == "line_a"
    assert event.direction == Direction.IN  # +1 = IN for A_to_B
    assert event.class_name == "person"
    assert event.frame_idx == 5  # 6th frame (0-indexed)


# ---------------------------------------------------------------------------
# 3. Cooldown suppresses immediate second crossing on the same line
# ---------------------------------------------------------------------------


def test_cooldown_suppresses_second_crossing_on_same_line() -> None:
    """A crossing within the cooldown window on the same line must be suppressed."""
    # cooldown = 2 s at 30 fps = 60 frames — well beyond the test sequence.
    config = _make_config(cooldown_seconds=2.0)
    logic = _make_logic(config=config)

    # First crossing: side -1 → +1 → event.
    # Then immediately try to cross back: +1 → -1 (but still in cooldown).
    sequence = (
        [_make_track(y=-10)] * 3  # init: confirmed_side = -1
        + [_make_track(y=+10)] * 3  # crossing: emit event, start cooldown
        + [_make_track(y=-10)] * 3  # try to cross back — in cooldown, suppressed
    )
    events = _run_frames(logic, sequence)

    assert len(events) == 1, f"Expected 1 event, got {len(events)}: {events}"


# ---------------------------------------------------------------------------
# 4. Cooldown on one line does NOT suppress a crossing on a different line
# ---------------------------------------------------------------------------


def test_cooldown_does_not_suppress_crossing_on_different_line() -> None:
    """Per-pair cooldown: a suppressed crossing on line_a must not affect line_b."""
    config = _make_config(cooldown_seconds=2.0)
    line_a = _make_line("line_a")
    # line_b is vertical: (50, -100) to (50, 100).
    # For a track at x=70, y=-10: the track is to the right of line_b.
    # signed_distance = (50-50)*(Py-(-100)) - (100-(-100))*(Px-50)
    #                 = 0 - 200*(Px-50) = -200*(70-50) = -4000 → side -1
    line_b = _make_line(
        "line_b",
        point_a=[50, -100],
        point_b=[50, 100],
        positive_direction="A_to_B",
    )
    logic = _make_logic(lines=[line_a, line_b], config=config)

    # Track moves below line_a (y>0) → side +1 on line_a from the start.
    # It also starts to the right of line_b (x=70) → side -1 on line_b.
    track_below_a_right_of_b = Track(
        track_id=1,
        bbox=(60.0, 0.0, 80.0, 20.0),  # bottom-center = (70, 20): y=20>0, x=70>50
        class_id=0,
        class_name="person",
        score=0.9,
    )

    # Track then moves above line_a (y<0) and to the left of line_b (x<50).
    track_above_a_left_of_b = Track(
        track_id=1,
        bbox=(10.0, -30.0, 40.0, -10.0),  # bottom-center = (25, -10): y=-10<0, x=25<50
        class_id=0,
        class_name="person",
        score=0.9,
    )

    # 3 frames init per line, then crossing per line.
    sequence = [track_below_a_right_of_b] * 3 + [track_above_a_left_of_b] * 3

    events = _run_frames(logic, sequence)

    line_a_events = [e for e in events if e.line_id == "line_a"]
    line_b_events = [e for e in events if e.line_id == "line_b"]

    assert len(line_a_events) == 1, f"line_a: expected 1, got {len(line_a_events)}"
    assert len(line_b_events) == 1, f"line_b: expected 1, got {len(line_b_events)}"


# ---------------------------------------------------------------------------
# 5. Multi-line independence: state on line_a unaffected by state on line_b
# ---------------------------------------------------------------------------


def test_multi_line_state_is_independent() -> None:
    """One track's crossing state on line_a must not bleed into line_b."""
    line_a = _make_line("line_a")
    line_b = _make_line("line_b")  # same geometry, different id
    logic = _make_logic(lines=[line_a, line_b])

    sequence = [_make_track(y=-10)] * 3 + [_make_track(y=+10)] * 3
    events = _run_frames(logic, sequence)

    # Both lines should detect the same crossing independently.
    assert len(events) == 2
    assert {e.line_id for e in events} == {"line_a", "line_b"}


# ---------------------------------------------------------------------------
# 6. Stale-track cleanup + re-initialization after timeout
# ---------------------------------------------------------------------------


def test_stale_entry_removed_and_reinitialised_as_fresh() -> None:
    """After the stale timeout, a re-appearing track must not emit on first observation."""
    # stale_timeout_seconds = 1.0 s at 30 fps = 30 frames.
    config = _make_config(stale_track_timeout_seconds=1.0)
    logic = _make_logic(config=config)

    # 3 frames to initialize confirmed_side = -1.
    init = [_make_track(y=-10)] * 3

    # 31 frames of no observations — stale cleanup fires.
    gap: list[Track | None] = [None] * 31

    # After re-appearing, 3 frames on side +1 — should NOT emit
    # (re-initializes confirmed_side = +1 on first observation, not a crossing).
    reappear = [_make_track(y=+10)] * 3

    all_events = []

    for frame_idx, track_or_none in enumerate(init + gap + reappear):
        tracks = [] if track_or_none is None else [track_or_none]
        events = logic.process(
            tracks=tracks,
            frame_idx=frame_idx,
            timestamp_seconds=frame_idx / _FPS,
        )
        all_events.extend(events)

    assert all_events == [], f"Expected no events after stale cleanup re-init: {all_events}"


# ---------------------------------------------------------------------------
# 7. Initial side never emits an event (brand-new pair)
# ---------------------------------------------------------------------------


def test_initial_side_never_emits_event() -> None:
    """A brand-new (track_id, line_id) pair initializes confirmed_side — no event."""
    logic = _make_logic()

    # 10 solid frames on side +1 — fills window with a decisive majority.
    # But confirmed_side is initialized on the very first frame and no flip
    # occurs, so no crossing should ever be emitted.
    sequence = [_make_track(y=+10)] * 10
    events = _run_frames(logic, sequence)

    assert events == [], f"Initial observation must never emit: {events}"


# ---------------------------------------------------------------------------
# 8. Class-majority-vote attribution at crossing time
# ---------------------------------------------------------------------------


def test_class_majority_vote_attribution() -> None:
    """Emitted event class must reflect window majority, not the current frame's label."""
    logic = _make_logic()

    # Establish confirmed_side = -1 with "person" tracks above the line.
    init = [_make_track(y=-10, class_name="person", class_id=0)] * 3

    # Crossing frames: first 2 are "person", last 1 (the trigger frame) is "car".
    # Window at trigger: [("person",-1 window), ("person", +1), ("car", +1)]
    # majority class = "person" (2/3), not "car".
    crossing = [
        _make_track(y=+10, class_name="person", class_id=0),
        _make_track(y=+10, class_name="person", class_id=0),
        _make_track(y=+10, class_name="car", class_id=2),  # flickered class
    ]

    events = _run_frames(logic, init + crossing)

    assert len(events) == 1
    event = events[0]
    assert event.class_name == "person", (
        f"Expected majority class 'person', got '{event.class_name}'"
    )
    assert event.class_id == 0


def test_confidence_and_bbox_from_current_frame() -> None:
    """confidence/bbox must come from the exact frame the event fires on (§7.5)."""
    logic = _make_logic()

    init = [_make_track(y=-10, score=0.5)] * 3

    # Each crossing frame has a different score; the trigger is the 3rd.
    trigger_track = _make_track(y=+10, score=0.99)
    crossing = [
        _make_track(y=+10, score=0.70),
        _make_track(y=+10, score=0.80),
        trigger_track,
    ]

    events = _run_frames(logic, init + crossing)

    assert len(events) == 1
    assert events[0].confidence == 0.99  # from the trigger frame
    assert events[0].bbox == trigger_track.bbox


# ---------------------------------------------------------------------------
# 9. get_counters() reflects correct totals across multiple events
# ---------------------------------------------------------------------------


def test_get_counters_correct_after_multiple_crossings() -> None:
    """Running counters must be accurate after a sequence of IN/OUT crossings."""
    config = _make_config(cooldown_seconds=0.0)  # no cooldown so we can cross back
    logic = _make_logic(config=config)

    # Crossing 1: side -1 → +1 → IN
    # Crossing 2: side +1 → -1 → OUT
    sequence = (
        [_make_track(y=-10)] * 3  # init: confirmed_side = -1
        + [_make_track(y=+10)] * 3  # IN  crossing
        + [_make_track(y=-10)] * 3  # OUT crossing
    )
    events = _run_frames(logic, sequence)

    assert len(events) == 2

    counters = logic.get_counters()
    assert counters.get(("person", "line_a", Direction.IN), 0) == 1
    assert counters.get(("person", "line_a", Direction.OUT), 0) == 1


def test_get_counters_independent_per_class() -> None:
    """Counters for 'person' and 'car' must be tracked independently."""
    config = _make_config(cooldown_seconds=0.0)
    logic = _make_logic(config=config)

    person_track_above = _make_track(y=-10, track_id=1, class_name="person", class_id=0)
    person_track_below = _make_track(y=+10, track_id=1, class_name="person", class_id=0)
    car_track_above = _make_track(y=-10, track_id=2, class_name="car", class_id=2)
    car_track_below = _make_track(y=+10, track_id=2, class_name="car", class_id=2)

    sequence: list[Track] = (
        # Initialize both tracks
        [person_track_above, car_track_above] * 3
        # Both cross IN
        + [person_track_below, car_track_below] * 3
    )

    for frame_idx, pair in enumerate(sequence):
        logic.process(
            tracks=[pair] if not isinstance(pair, list) else pair,
            frame_idx=frame_idx,
            timestamp_seconds=frame_idx / _FPS,
        )

    counters = logic.get_counters()
    assert counters.get(("person", "line_a", Direction.IN), 0) == 1
    assert counters.get(("car", "line_a", Direction.IN), 0) == 1


# ---------------------------------------------------------------------------
# 10. Cooldown: confirmed_side does NOT flip while in cooldown
# ---------------------------------------------------------------------------


def test_confirmed_side_does_not_flip_during_cooldown() -> None:
    """confirmed_side must not flip silently during cooldown.

    The expected sequence is:
    1. Frames 0-2:  side -1 → confirmed_side initialised to -1, no event.
    2. Frames 3-5:  side +1 → decisive crossing, emit IN event, start cooldown.
    3. Frames 6-14: side -1 (decisive) but cooldown active → no event.
    4. Frame 15:    cooldown finally expires; the decisive -1 majority is now
                    seen with confirmed_side STILL at +1 (not silently flipped),
                    so an OUT crossing fires legitimately.

    Two events total proves confirmed_side was correctly maintained at +1
    throughout the cooldown rather than being reset to -1 prematurely (which
    would have suppressed the OUT event because -1 would equal confirmed_side).
    """
    # cooldown = 10 frames at 30 fps.
    config = _make_config(cooldown_seconds=10.0 / _FPS)
    logic = _make_logic(config=config)

    sequence_init = [_make_track(y=-10)] * 3  # confirmed_side → -1
    sequence_cross = [_make_track(y=+10)] * 3  # IN event, cooldown starts
    sequence_back_during_cooldown = [_make_track(y=-10)] * 3  # in cooldown → no event
    sequence_drain = [_make_track(y=-10)] * 10  # drains cooldown; OUT fires

    all_events = []
    full_seq = sequence_init + sequence_cross + sequence_back_during_cooldown + sequence_drain
    for frame_idx, track in enumerate(full_seq):
        events = logic.process(
            tracks=[track],
            frame_idx=frame_idx,
            timestamp_seconds=frame_idx / _FPS,
        )
        all_events.extend(events)

    assert len(all_events) == 2, f"Expected IN then OUT, got {len(all_events)} events: {all_events}"
    assert all_events[0].direction == Direction.IN, "First event must be IN"
    assert all_events[1].direction == Direction.OUT, "Second event must be OUT after cooldown"


# ---------------------------------------------------------------------------
# 11. Optional displacement safeguard
# ---------------------------------------------------------------------------


def test_displacement_safeguard_suppresses_jitter_crossing() -> None:
    """When min_displacement_px is set, a crossing near the same point is suppressed."""
    config = _make_config(
        cooldown_seconds=0.0,
        min_displacement_px=100.0,  # very large threshold — no crossing should pass
    )
    logic = _make_logic(config=config)

    # Cross IN once (first crossing: last_event_ref_point is None → safeguard skipped).
    sequence = (
        [_make_track(y=-10)] * 3
        + [_make_track(y=+10)] * 3  # first crossing: displacement check skipped
        + [_make_track(y=-10)] * 3  # try OUT: displacement is small → suppressed
    )
    events = _run_frames(logic, sequence)

    # First crossing always fires (no previous event reference point).
    # Second crossing is suppressed because displacement < 100 px.
    assert len(events) == 1


def test_displacement_safeguard_disabled_by_default() -> None:
    """With min_displacement_px=None, two consecutive crossings both fire."""
    config = _make_config(cooldown_seconds=0.0, min_displacement_px=None)
    logic = _make_logic(config=config)

    sequence = [_make_track(y=-10)] * 3 + [_make_track(y=+10)] * 3 + [_make_track(y=-10)] * 3
    events = _run_frames(logic, sequence)

    assert len(events) == 2


# ---------------------------------------------------------------------------
# 12. Direction mapping
# ---------------------------------------------------------------------------


def test_direction_a_to_b_positive_direction() -> None:
    """positive_direction='A_to_B': crossing to +1 side = IN."""
    line = _make_line(positive_direction="A_to_B")
    logic = _make_logic(lines=[line])

    sequence = (
        [_make_track(y=-10)] * 3  # side -1, init
        + [_make_track(y=+10)] * 3  # side +1, crossing
    )
    events = _run_frames(logic, sequence)
    assert len(events) == 1
    assert events[0].direction == Direction.IN


def test_direction_b_to_a_positive_direction() -> None:
    """positive_direction='B_to_A': crossing to -1 side = IN."""
    line = _make_line(positive_direction="B_to_A")
    logic = _make_logic(lines=[line])

    sequence = (
        [_make_track(y=+10)] * 3  # side +1, init
        + [_make_track(y=-10)] * 3  # side -1, crossing
    )
    events = _run_frames(logic, sequence)
    assert len(events) == 1
    assert events[0].direction == Direction.IN
