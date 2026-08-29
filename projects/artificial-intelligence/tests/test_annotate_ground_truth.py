from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.annotate_ground_truth import (
    CSV_FIELDNAMES,
    KEY_LEFT_CODES,
    KEY_RIGHT_CODES,
    AnnotationEvent,
    calculate_timestamp,
    normalize_key,
    save_events_to_csv,
)


def test_calculate_timestamp() -> None:
    assert calculate_timestamp(frame_idx=125, fps=25.0) == 5.0


def test_calculate_timestamp_with_invalid_fps() -> None:
    assert calculate_timestamp(frame_idx=10, fps=0.0) == 0.0
    assert calculate_timestamp(frame_idx=10, fps=-1.0) == 0.0


def test_annotation_event_stores_expected_values() -> None:
    event = AnnotationEvent(
        frame_idx=50,
        timestamp_seconds=2.0,
        class_name="car",
        direction="OUT",
        line_id="line_1",
        video_name="sample.mp4",
    )

    assert event.frame_idx == 50
    assert event.timestamp_seconds == 2.0
    assert event.class_name == "car"
    assert event.direction == "OUT"
    assert event.line_id == "line_1"
    assert event.video_name == "sample.mp4"


def test_save_events_to_csv_writes_exact_gt_header(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "ground_truth.csv"

    event = AnnotationEvent(
        frame_idx=25,
        timestamp_seconds=1.0,
        class_name="person",
        direction="IN",
        line_id="main_line",
        video_name="sample.mp4",
    )

    save_events_to_csv([event], output_path)

    assert output_path.is_file()

    with output_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        rows = list(reader)

    assert rows[0] == [
        "frame_idx",
        "timestamp_seconds",
        "class_name",
        "direction",
        "line_id",
        "video_name",
    ]
    assert rows[0] == CSV_FIELDNAMES
    assert rows[1] == [
        "25",
        "1.0",
        "person",
        "IN",
        "main_line",
        "sample.mp4",
    ]


def test_save_multiple_events_preserves_order(tmp_path: Path) -> None:
    output_path = tmp_path / "events.csv"

    events = [
        AnnotationEvent(
            frame_idx=10,
            timestamp_seconds=0.5,
            class_name="person",
            direction="IN",
            line_id="line_1",
            video_name="video.mp4",
        ),
        AnnotationEvent(
            frame_idx=20,
            timestamp_seconds=1.0,
            class_name="car",
            direction="OUT",
            line_id="line_1",
            video_name="video.mp4",
        ),
    ]

    save_events_to_csv(events, output_path)

    with output_path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 2
    assert rows[0]["frame_idx"] == "10"
    assert rows[0]["class_name"] == "person"
    assert rows[0]["direction"] == "IN"
    assert rows[1]["frame_idx"] == "20"
    assert rows[1]["class_name"] == "car"
    assert rows[1]["direction"] == "OUT"


def test_save_empty_events_writes_only_header(tmp_path: Path) -> None:
    output_path = tmp_path / "empty.csv"

    save_events_to_csv([], output_path)

    with output_path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))

    assert rows == [CSV_FIELDNAMES]


@pytest.mark.parametrize(
    ("frame_idx", "fps", "expected_timestamp"),
    [
        (0, 25.0, 0.0),
        (1, 25.0, 0.04),
        (24, 25.0, 0.96),
        (25, 25.0, 1.0),
        (125, 25.0, 5.0),
    ],
)
def test_calculate_timestamp_is_frame_accurate(
    frame_idx: int,
    fps: float,
    expected_timestamp: float,
) -> None:
    assert calculate_timestamp(frame_idx, fps) == expected_timestamp


def test_normalize_key_arrow_codes():
    for code in KEY_LEFT_CODES:
        assert normalize_key(code) == "left"
    for code in KEY_RIGHT_CODES:
        assert normalize_key(code) == "right"


def test_normalize_key_case_insensitivity():
    assert normalize_key(ord("m")) == "m"
    assert normalize_key(ord("M")) == "m"
    assert normalize_key(ord("u")) == "u"
    assert normalize_key(ord("U")) == "u"
    assert normalize_key(ord("q")) == "q"
    assert normalize_key(ord("Q")) == "q"


def test_normalize_key_unhandled_code():
    assert normalize_key(999999) == 999999
