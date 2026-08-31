from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from ultralytics.trackers.byte_tracker import BYTETracker

from mot_counting.interfaces.tracker import ITracker
from mot_counting.types import Detection, Track


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
            "track_high_thresh": getattr(self, "track_thresh", 0.5),
            "track_low_thresh": 0.1,
            "new_track_thresh": getattr(self, "track_thresh", 0.5) + 0.1,
        }
        return defaults.get(name, 0.0)


class ByteTrackWrapper(ITracker):
    """ByteTrack wrapper supporting detection objects."""

    __slots__ = ("args", "tracker", "_class_names")

    def __init__(
        self,
        track_thresh: float = 0.5,
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
        detections: list[Detection],
        frame_idx: int,
        frame: np.ndarray,
    ) -> list[Track]:
        """Update tracker with detections from current frame.

        Note:
            The `frame` argument is required by the `ITracker` interface for
            forward-compatibility with visual/appearance-based trackers (e.g., BoT-SORT).
            It is intentionally unused by ByteTrack, which is purely motion-based.
        """
        if detections:
            for det in detections:
                self._class_names[det.class_id] = det.class_name

            boxes_arr = np.array([det.xyxy for det in detections], dtype=np.float32)
            scores_arr = np.array([det.confidence for det in detections], dtype=np.float32)
            class_ids_arr = np.array([det.class_id for det in detections], dtype=np.int32)
        else:
            boxes_arr = np.empty((0, 4), dtype=np.float32)
            scores_arr = np.empty((0,), dtype=np.float32)
            class_ids_arr = np.empty((0,), dtype=np.int32)

        results_mock = UltralyticsResultsMock(boxes_arr, scores_arr, class_ids_arr)

        tracks = self.tracker.update(results_mock, img=None)
        if len(tracks) == 0:
            return []

        out_tracks: list[Track] = []
        for t in tracks:
            cls_id = int(t[5]) if len(t) > 5 else 0
            cls_name = self._class_names.get(cls_id, "")

            # Extract valid positive score (Ultralytics byte_tracker format fallback)
            score = 1.0
            if len(t) > 6 and float(t[6]) > 0.0:
                score = float(t[6])
            elif len(t) > 4 and 0.0 < float(t[4]) <= 1.0:
                score = float(t[4])

            out_tracks.append(
                Track(
                    track_id=int(t[4]),
                    bbox=tuple(t[:4]),
                    class_id=cls_id,
                    class_name=cls_name,
                    score=score,
                )
            )

        return out_tracks

    def reset(self) -> None:
        """Reset internal tracker state and mappings."""
        self.tracker.reset()
        self._class_names.clear()
