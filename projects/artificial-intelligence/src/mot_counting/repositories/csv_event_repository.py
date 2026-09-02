"""
CSV implementation of the event repository.
Handles persistent storage of CrossingEvents to disk.
"""

import csv
from dataclasses import fields
from pathlib import Path

from mot_counting.interfaces.repository import IEventRepository
from mot_counting.types import CrossingEvent


class CsvEventRepository(IEventRepository):
    """
    Repository for writing events to a CSV file.

    Write Strategy :
    This class keeps a file handle open during its lifecycle to avoid the overhead
    of opening/closing the file per event. Writes are buffered by the OS by default.
    Call `flush()` to force writing to disk.

    The controller managing this repository
    must ensure `close()` is always called even on errors.

    BBox Format :
    The `bbox` tuple (x1, y1, x2, y2) is serialized as a single comma-separated string.
    The csv.writer automatically encloses this in quotes (e.g., '"100,200,50,100"').
    """

    def __init__(self, output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.file = open(output_path, "w", encoding="utf-8", newline="")  # noqa: SIM115
        self.writer = csv.writer(self.file)

        self.header = [f.name for f in fields(CrossingEvent)]
        self.writer.writerow(self.header)

    def save(self, event: CrossingEvent) -> None:
        if event.bbox is not None:
            bbox_str = f"{event.bbox[0]},{event.bbox[1]},{event.bbox[2]},{event.bbox[3]}"
        else:
            bbox_str = ""

        row = [
            event.frame_idx,
            event.timestamp_seconds,
            event.track_id,
            event.class_id,
            event.class_name,
            event.direction.value,
            event.line_id,
            "" if event.confidence is None else event.confidence,
            bbox_str,
            "" if event.video_name is None else event.video_name,
        ]
        self.writer.writerow(row)

    def flush(self) -> None:
        self.file.flush()

    def close(self) -> None:
        if not self.file.closed:
            self.flush()
            self.file.close()
