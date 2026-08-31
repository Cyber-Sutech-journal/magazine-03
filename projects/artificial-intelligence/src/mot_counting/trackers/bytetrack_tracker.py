from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from ultralytics.trackers.byte_tracker import BYTETracker

from mot_counting.interfaces.tracker import ITracker
from mot_counting.types import Detection, Track


# NOTE: torch is required because BYTETracker internally expects Results/Boxes
# to expose PyTorch tensor interfaces (.xyxy, .xywh, .conf, .cls).
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
        """Update tracker with detections from current frame."""
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

        # frame is part of ITracker for BoT-SORT forward-compat; ByteTrack is
        # motion-only and intentionally ignores it.
        tracks = self.tracker.update(results_mock, img=None)
        if len(tracks) == 0:
            return []

        out_tracks: list[Track] = []
        # BYTETracker output layout: [x1, y1, x2, y2, track_id, score, class_id, idx]
        for t in tracks:
            if len(t) < 7:
                continue

            bbox = (float(t[0]), float(t[1]), float(t[2]), float(t[3]))
            track_id = int(t[4])
            score_val = float(t[5])
            cls_id = int(t[6])

            # NOTE: ByteTrack does not expose association indices, so class_name
            # is resolved from the cumulative _class_names map.
            cls_name = self._class_names.get(cls_id, "")
            score = score_val if score_val > 0.0 else 1.0

            out_tracks.append(
                Track(
                    track_id=track_id,
                    bbox=bbox,
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
