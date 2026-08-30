import csv

from src.mot_counting.repositories.csv_event_repository import (
    CsvEventRepository,
    InMemoryEventRepository,
)
from src.mot_counting.types import CrossingEvent, Direction


def test_in_memory_repository_save():
    repo = InMemoryEventRepository()
    event = CrossingEvent(
        frame_idx=1,
        timestamp_seconds=0.033,
        track_id=42,
        class_id=1,
        class_name="person",
        direction=Direction.IN,
        line_id=0,
        confidence=0.95,
        bbox=(100, 200, 50, 100),
        video_name="test_video.mp4",
    )

    repo.save(event)

    assert len(repo.events) == 1
    assert repo.events[0] == event


def test_csv_repository_header(tmp_path):
    output_file = tmp_path / "test.csv"
    repo = CsvEventRepository(str(output_file))

    repo.close()
    with open(output_file, encoding="utf-8") as f:
        header = f.readline()

    expected_header = "frame_idx,timestamp_seconds,track_id,class_id,class_name,direction,line_id,confidence,bbox,video_name\n"
    assert header == expected_header


def test_csv_repository_save(tmp_path):
    output_file = tmp_path / "test.csv"
    repo = CsvEventRepository(str(output_file))

    events_test = [
        CrossingEvent(
            frame_idx=1,
            timestamp_seconds=0.033,
            track_id=42,
            class_id=1,
            class_name="person",
            direction=Direction.IN,
            line_id=0,
            confidence=0.95,
            bbox=(100, 200, 50, 100),
            video_name="test_video.mp4",
        ),
        CrossingEvent(
            frame_idx=2,
            timestamp_seconds=0.066,
            track_id=43,
            class_id=2,
            class_name="car",
            direction=Direction.OUT,
            line_id=1,
            confidence=None,
            bbox=None,
            video_name=None,
        ),
        CrossingEvent(
            frame_idx=3,
            timestamp_seconds=0.099,
            track_id=44,
            class_id=3,
            class_name="bicycle",
            direction=Direction.IN,
            line_id=2,
            confidence=0.85,
            bbox=(150, 250, 60, 120),
            video_name=None,
        ),
    ]

    for event in events_test:
        repo.save(event)
    repo.close()

    with open(output_file, encoding="utf-8") as f:
        lines = f.readlines()

    assert len(lines) == 4

    expected_line_1 = '1,0.033,42,1,person,IN,0,0.95,"100,200,50,100",test_video.mp4\n'
    expected_line_2 = "2,0.066,43,2,car,OUT,1,,,\n"
    expected_line_3 = '3,0.099,44,3,bicycle,IN,2,0.85,"150,250,60,120",\n'

    assert lines[1] == expected_line_1
    assert lines[2] == expected_line_2
    assert lines[3] == expected_line_3


def test_csv_repository_close_idempotent(tmp_path):
    output_file = tmp_path / "test.csv"
    repo = CsvEventRepository(str(output_file))

    event = CrossingEvent(
        frame_idx=1,
        timestamp_seconds=0.033,
        track_id=42,
        class_id=1,
        class_name="person",
        direction=Direction.IN,
        line_id=0,
        confidence=0.95,
        bbox=(100, 200, 50, 100),
        video_name="test_video.mp4",
    )

    repo.save(event)

    repo.close()
    assert repo.file.closed

    repo.close()
    assert repo.file.closed


def test_csv_repository_creates_parent_directory(tmp_path):
    nested_dir = tmp_path / "nested" / "dir"
    output_file = nested_dir / "test.csv"

    repo = CsvEventRepository(str(output_file))
    repo.close()

    assert nested_dir.exists()
    assert output_file.exists()


def test_csv_repository_bbox_roundtrip(tmp_path):
    output_file = tmp_path / "test_bbox.csv"
    repo = CsvEventRepository(str(output_file))

    test_bbox = (100.5, 200.5, 50.0, 100.0)
    event = CrossingEvent(
        frame_idx=1,
        timestamp_seconds=0.033,
        track_id=42,
        class_id=1,
        class_name="person",
        direction=Direction.IN,
        line_id=0,
        confidence=0.95,
        bbox=test_bbox,
        video_name="vid.mp4",
    )

    repo.save(event)
    repo.close()

    with open(output_file, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        row = next(reader)

    bbox_str = row[8]

    parsed_bbox = tuple(float(x) for x in bbox_str.split(","))

    assert parsed_bbox == test_bbox
