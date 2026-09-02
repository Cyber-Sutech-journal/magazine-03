from typing import Any

import numpy as np

from mot_counting.interfaces.detector import IDetector
from mot_counting.types import Detection


class Yolo26Detector(IDetector):
    """
    YOLO26 Wrapper for object detection.

    This class adheres strictly to Dependency Injection. It DOES NOT load the model
    from disk or download weights. It expects a pre-loaded Ultralytics model object.

    Note: YOLO26 uses a native end-to-end, NMS-free detection head.
    Therefore, NO manual NMS or IoU post-processing is applied here.
    Validating if the loaded model actually supports the `allowed_classes` is
    also NOT the responsibility of this class (handled in composition root).
    """

    def __init__(
        self, model: Any, imgsz: int, confidence_threshold: float, allowed_classes: list[str]
    ):
        self.model = model
        self.imgsz = imgsz
        self.confidence_threshold = confidence_threshold
        self.allowed_classes = allowed_classes

    def predict(self, frame: np.ndarray) -> list[Detection]:
        results = self.model(
            frame,
            imgsz=self.imgsz,
            conf=self.confidence_threshold,
            verbose=False,
        )[0]
        detections = []

        for box in results.boxes:
            conf = box.conf[0].item()

            if conf < self.confidence_threshold:
                continue

            class_id = int(box.cls[0].item())
            class_name = self.model.names[class_id]

            if class_name not in self.allowed_classes:
                continue

            xyxy = tuple(box.xyxy[0].tolist())

            detections.append(
                Detection(xyxy=xyxy, confidence=conf, class_id=class_id, class_name=class_name)
            )

        return detections
