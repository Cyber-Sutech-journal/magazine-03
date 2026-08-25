"""Unit tests for core domain types in ``mot_counting.types``."""

import dataclasses

import pytest

from mot_counting.types import CrossingEvent, Detection, Direction, Track


def test_detection_construction_and_field_access() -> None:
    detection = Detection(
        xyxy=(10.0, 20.0, 50.0, 80.0),
        confidence=0.92,
        class_id=0,
        class_name="person",
    )

    assert detection.xyxy == (10.0, 20.0, 50.0, 80.0)
    assert detection.confidence == 0.92
    assert detection.class_id == 0
    assert detection.class_name == "person"


def test_track_construction_and_field_access() -> None:
    track = Track(
        track_id=7,
        bbox=(100.0, 200.0, 150.0, 260.0),
        class_id=2,
        class_name="car",
        score=0.88,
    )

    assert track.track_id == 7
    assert track.bbox == (100.0, 200.0, 150.0, 260.0)
    assert track.class_id == 2
    assert track.class_name == "car"
    assert track.score == 0.88


def test_direction_string_enum_equality() -> None:
    assert Direction.IN == "IN"
    assert Direction.OUT == "OUT"
    assert Direction.IN.value == "IN"
    assert Direction.OUT.value == "OUT"


def test_crossing_event_required_fields_only() -> None:
    event = CrossingEvent(
        frame_idx=42,
        timestamp_seconds=1.4,
        track_id=3,
        class_id=0,
        class_name="person",
        direction=Direction.IN,
        line_id="main_entrance",
    )

    assert event.frame_idx == 42
    assert event.timestamp_seconds == 1.4
    assert event.track_id == 3
    assert event.class_id == 0
    assert event.class_name == "person"
    assert event.direction is Direction.IN
    assert event.line_id == "main_entrance"
    assert event.confidence is None
    assert event.bbox is None
    assert event.video_name is None


def test_crossing_event_with_optional_fields() -> None:
    event = CrossingEvent(
        frame_idx=99,
        timestamp_seconds=3.3,
        track_id=12,
        class_id=2,
        class_name="car",
        direction=Direction.OUT,
        line_id="lane_a",
        confidence=0.75,
        bbox=(1.0, 2.0, 3.0, 4.0),
        video_name="sample.mp4",
    )

    assert event.confidence == 0.75
    assert event.bbox == (1.0, 2.0, 3.0, 4.0)
    assert event.video_name == "sample.mp4"


@pytest.mark.parametrize(
    "instance",
    [
        Detection(xyxy=(0.0, 0.0, 1.0, 1.0), confidence=1.0, class_id=0, class_name="person"),
        Track(track_id=1, bbox=(0.0, 0.0, 1.0, 1.0), class_id=0, class_name="person", score=1.0),
        CrossingEvent(
            frame_idx=0,
            timestamp_seconds=0.0,
            track_id=1,
            class_id=0,
            class_name="person",
            direction=Direction.IN,
            line_id="line_1",
        ),
    ],
)
def test_frozen_dataclasses_reject_mutation(instance: Detection | Track | CrossingEvent) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        if isinstance(instance, Detection):
            instance.confidence = 0.5  # type: ignore[misc]
        elif isinstance(instance, Track):
            instance.track_id = 99  # type: ignore[misc]
        else:
            instance.frame_idx = 99  # type: ignore[misc]
