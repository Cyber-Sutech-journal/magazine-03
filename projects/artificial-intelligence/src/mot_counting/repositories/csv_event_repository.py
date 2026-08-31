from pathlib import Path

from mot_counting.interfaces.repository import IEventRepository
from mot_counting.types import CrossingEvent


class CsvEventRepository(IEventRepository):
    def __init__(self, output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.file = open(output_path, "w", encoding="utf-8")  # noqa: SIM115
        self.file.write(
            "frame_idx,timestamp_seconds,track_id,class_id,class_name,direction,line_id,confidence,bbox,video_name\n"
        )

    def save(self, event: CrossingEvent) -> None:
        if event.bbox is None:
            bbox_str = ""
        else:
            bbox_str = f'"{event.bbox[0]},{event.bbox[1]},{event.bbox[2]},{event.bbox[3]}"'

        confidence_str = "" if event.confidence is None else str(event.confidence)
        video_name_str = "" if event.video_name is None else event.video_name

        self.file.write(
            f"{event.frame_idx},{event.timestamp_seconds},{event.track_id},{event.class_id},{event.class_name},{event.direction.value},{event.line_id},{confidence_str},{bbox_str},{video_name_str}\n"
        )

    def flush(self) -> None:
        self.file.flush()

    def close(self) -> None:
        if not self.file.closed:
            self.file.close()
