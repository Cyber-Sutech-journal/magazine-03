from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from ultralytics.trackers.byte_tracker import BYTETracker

from mot_counting.interfaces.tracker import ITracker
from mot_counting.types import Track


class UltralyticsResultsMock:
    """Lightweight mock of Ultralytics Results/Boxes expected by BYTETracker."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        num_dets = len(xyxy)
        if num_dets > 0:
            w = xyxy[:, 2] - xyxy[:, 0]
            h = xyxy[:, 3] - xyxy[:, 1]
            x_c = xyxy[:, 0] + w / 2.0
            y_c = xyxy[:, 1] + h / 2.0
            xywh = np.stack([x_c, y_c, w, h], axis=1)

            self.xyxy = torch.from_numpy(xyxy).float()
            self.xywh = torch.from_numpy(xywh).float()
            self.conf = torch.from_numpy(conf).float()
            self.cls = torch.from_numpy(cls).float()
        else:
            self.xyxy = torch.empty((0, 4), dtype=torch.float32)
            self.xywh = torch.empty((0, 4), dtype=torch.float32)
            self.conf = torch.empty((0,), dtype=torch.float32)
            self.cls = torch.empty((0,), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, idx: Any) -> "UltralyticsResultsMock":
        res = UltralyticsResultsMock.__new__(UltralyticsResultsMock)
        res.xyxy = self.xyxy[idx]
        res.xywh = self.xywh[idx]
        res.conf = self.conf[idx]
        res.cls = self.cls[idx]
        return res


class ByteTrackArgs(SimpleNamespace):
    """Dynamic configuration container for BYTETracker args."""

    def __getattr__(self, name: str) -> Any:
        defaults = {
            "track_high_thresh": getattr(self, "track_thresh", 0.25),
            "track_low_thresh": 0.1,
            "new_track_thresh": getattr(self, "track_thresh", 0.25) + 0.1,
        }
        return defaults.get(name, 0.0)


class ByteTrackWrapper(ITracker):
    """ByteTrack wrapper supporting detection objects as well as numpy arrays."""

    __slots__ = ("args", "tracker", "_class_names")

    def __init__(
        self,
        track_thresh: float = 0.25,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        frame_rate: int = 30,
        fuse_score: bool = True,
    ) -> None:
        self.args = ByteTrackArgs(
            track_thresh=track_thresh,
            track_high_thresh=track_thresh,
            track_low_thresh=0.1,
            new_track_thresh=track_thresh + 0.1,
            match_thresh=match_thresh,
            track_buffer=track_buffer,
            frame_rate=frame_rate,
            fuse_score=fuse_score,
        )
        self.tracker = BYTETracker(args=self.args)
        self._class_names: dict[int, str] = {}

    def update(
        self,
        detections: list[Any] | None = None,
        frame_idx: int = 0,
        frame: np.ndarray | None = None,
        boxes: np.ndarray | None = None,
        scores: np.ndarray | None = None,
        class_ids: np.ndarray | None = None,
    ) -> list[Track]:
        """Update tracker with detections from current frame."""
        if detections is not None:
            if len(detections) == 0:
                return []

            boxes_list, scores_list, cls_list = [], [], []
            for det in detections:
                xyxy = getattr(det, "xyxy", None)
                if xyxy is None and hasattr(det, "bbox"):
                    xyxy = det.bbox
                conf = getattr(det, "confidence", None)
                if conf is None and hasattr(det, "score"):
                    conf = det.score
                cid = getattr(det, "class_id", 0)
                cname = getattr(det, "class_name", str(cid))

                self._class_names[cid] = cname
                boxes_list.append(xyxy)
                scores_list.append(conf)
                cls_list.append(cid)

            boxes_arr = np.asarray(boxes_list, dtype=np.float32)
            scores_arr = np.asarray(scores_list, dtype=np.float32)
            class_ids_arr = np.asarray(cls_list, dtype=np.float32)

        elif boxes is not None and scores is not None and class_ids is not None:
            if len(boxes) == 0:
                return []
            boxes_arr = np.asarray(boxes, dtype=np.float32)
            scores_arr = np.asarray(scores, dtype=np.float32)
            class_ids_arr = np.asarray(class_ids, dtype=np.float32)
        else:
            return []

        results_mock = UltralyticsResultsMock(boxes_arr, scores_arr, class_ids_arr)

        tracks = self.tracker.update(results_mock, img=frame)
        if len(tracks) == 0:
            return []

        tracked_objects: list[Track] = []
        for track in tracks:
            cls_id = int(track[6])
            tracked_objects.append(
                Track(
                    track_id=int(track[4]),
                    bbox=[float(track[0]), float(track[1]), float(track[2]), float(track[3])],
                    score=float(track[5]),
                    class_id=cls_id,
                    class_name=self._class_names.get(cls_id, str(cls_id)),
                )
            )

        return tracked_objects

    def reset(self) -> None:
        """Reset internal tracker state and mappings."""
        self.tracker.reset()
        self._class_names.clear()
