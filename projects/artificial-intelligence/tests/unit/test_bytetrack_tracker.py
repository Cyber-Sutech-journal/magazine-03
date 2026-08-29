"""Unit tests for ByteTrackWrapper implementation."""

from __future__ import annotations

import numpy as np
import pytest

from mot_counting.interfaces.tracker import ITracker
from mot_counting.trackers.bytetrack_tracker import ByteTrackWrapper
from mot_counting.types import Detection, Track


@pytest.fixture
def dummy_frame() -> np.ndarray:
    """Create a dummy black frame for tracker input."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def tracker() -> ByteTrackWrapper:
    """Initialize a ByteTrackWrapper instance for testing."""
    return ByteTrackWrapper(
        track_thresh=0.4,
        match_thresh=0.8,
        track_buffer=30,
        frame_rate=30,
    )


def test_implements_itracker_interface(tracker: ByteTrackWrapper) -> None:
    """Verify that ByteTrackWrapper correctly implements the ITracker interface."""
    assert isinstance(tracker, ITracker)


def test_empty_detections_returns_empty_tracks(
    tracker: ByteTrackWrapper, dummy_frame: np.ndarray
) -> None:
    """Verify that passing empty detection lists produces empty active tracks."""
    tracks = tracker.update(detections=[], frame_idx=0, frame=dummy_frame)
    assert isinstance(tracks, list)
    assert len(tracks) == 0


def test_tracker_generates_valid_tracks(tracker: ByteTrackWrapper, dummy_frame: np.ndarray) -> None:
    """Verify that consecutive detections initialize and update active tracks properly."""
    det_seq = [
        # Frame 0
        [
            Detection(
                xyxy=(100.0, 100.0, 200.0, 200.0),
                confidence=0.9,
                class_id=0,
                class_name="car",
            )
        ],
        # Frame 1 (slight movement)
        [
            Detection(
                xyxy=(102.0, 101.0, 202.0, 201.0),
                confidence=0.88,
                class_id=0,
                class_name="car",
            )
        ],
        # Frame 2 (continuous movement)
        [
            Detection(
                xyxy=(105.0, 103.0, 205.0, 203.0),
                confidence=0.91,
                class_id=0,
                class_name="car",
            )
        ],
    ]

    all_tracks: list[list[Track]] = []
    for frame_idx, detections in enumerate(det_seq):
        tracks = tracker.update(detections=detections, frame_idx=frame_idx, frame=dummy_frame)
        all_tracks.append(tracks)

    # ByteTrack typically activates tracks within 1-2 confirmed frames
    last_frame_tracks = all_tracks[-1]
    assert len(last_frame_tracks) >= 1

    tracked_obj = last_frame_tracks[0]
    assert isinstance(tracked_obj, Track)
    assert tracked_obj.class_id == 0
    assert tracked_obj.class_name == "car"
    assert len(tracked_obj.bbox) == 4
    assert tracked_obj.score > 0.0


def test_tracker_reset_clears_state(tracker: ByteTrackWrapper, dummy_frame: np.ndarray) -> None:
    """Verify that calling reset clears internal state and class mappings."""
    detections = [
        Detection(
            xyxy=(50.0, 50.0, 120.0, 120.0),
            confidence=0.95,
            class_id=1,
            class_name="person",
        )
    ]
    tracker.update(detections=detections, frame_idx=0, frame=dummy_frame)
    assert len(tracker._class_names) > 0

    tracker.reset()
    assert len(tracker._class_names) == 0
