from unittest.mock import MagicMock

import numpy as np

from mot_counting.detectors.yolo26_detector import Yolo26Detector


def create_mock_model(conf_val: float, cls_val: int, xyxy_val: list[float], empty: bool = False):
    mock_model = MagicMock()
    mock_model.names = {0: "person", 1: "bicycle", 2: "car"}

    if empty:
        mock_results = MagicMock()
        mock_results.boxes = []
        mock_model.return_value = [mock_results]
        return mock_model

    mock_box = MagicMock()
    mock_box.conf = [MagicMock(item=MagicMock(return_value=conf_val))]
    mock_box.cls = [MagicMock(item=MagicMock(return_value=cls_val))]
    mock_box.xyxy = [MagicMock(tolist=MagicMock(return_value=xyxy_val))]

    mock_results = MagicMock()
    mock_results.boxes = [mock_box]
    mock_model.return_value = [mock_results]

    return mock_model


# Case 1: Verifies valid detection is returned and model is called with correct arguments.
def test_predict_returns_valid_detection():
    mock_model = create_mock_model(conf_val=0.9, cls_val=2, xyxy_val=[10.0, 20.0, 30.0, 40.0])

    detector = Yolo26Detector(
        model=mock_model, imgsz=640, confidence_threshold=0.5, allowed_classes=["car", "person"]
    )

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.predict(dummy_frame)

    mock_model.assert_called_once_with(dummy_frame, imgsz=640, conf=0.5, verbose=False)

    assert len(detections) == 1
    assert detections[0].class_id == 2
    assert detections[0].class_name == "car"
    assert detections[0].confidence == 0.9
    assert detections[0].xyxy == (10.0, 20.0, 30.0, 40.0)


# Case 2: Filters out detections belonging to classes not listed in allowed_classes.
def test_predict_filters_out_unallowed_class():
    mock_model = create_mock_model(conf_val=0.9, cls_val=1, xyxy_val=[10.0, 20.0, 30.0, 40.0])

    detector = Yolo26Detector(
        model=mock_model, imgsz=640, confidence_threshold=0.5, allowed_classes=["car", "person"]
    )

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.predict(dummy_frame)

    assert len(detections) == 0


# Case 3: Boundary check - detection with confidence exactly at threshold must be kept.
def test_predict_keeps_detection_at_confidence_threshold():
    mock_model = create_mock_model(conf_val=0.5, cls_val=2, xyxy_val=[10.0, 20.0, 30.0, 40.0])

    detector = Yolo26Detector(
        model=mock_model, imgsz=640, confidence_threshold=0.5, allowed_classes=["car", "person"]
    )

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.predict(dummy_frame)

    assert len(detections) == 1
    assert detections[0].confidence == 0.5


# Case 4: Boundary check - detection with confidence just below threshold must be dropped.
def test_predict_filters_out_detection_just_below_confidence_threshold():
    mock_model = create_mock_model(conf_val=0.49, cls_val=2, xyxy_val=[10.0, 20.0, 30.0, 40.0])

    detector = Yolo26Detector(
        model=mock_model, imgsz=640, confidence_threshold=0.5, allowed_classes=["car", "person"]
    )

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.predict(dummy_frame)

    assert len(detections) == 0


# Case 5: Handles empty frames where no detections are made.
def test_predict_handles_empty_frame():
    mock_model = create_mock_model(conf_val=0.0, cls_val=0, xyxy_val=[], empty=True)

    detector = Yolo26Detector(
        model=mock_model, imgsz=640, confidence_threshold=0.5, allowed_classes=["car", "person"]
    )

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.predict(dummy_frame)

    assert detections == []
