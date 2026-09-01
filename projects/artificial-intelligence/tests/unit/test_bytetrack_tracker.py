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


def test_default_construction_and_hyperparameters() -> None:
    """Verify constructing ByteTrackWrapper with no arguments sets spec defaults per T15 §3."""
    default_tracker = ByteTrackWrapper()
    assert default_tracker.args.track_thresh == 0.5
    assert default_tracker.args.match_thresh == 0.8
    assert default_tracker.args.track_buffer == 30
    assert default_tracker.args.frame_rate == 30


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


def test_two_separated_detections_stability_and_class_preservation(
    dummy_frame: np.ndarray,
) -> None:
    """Verify that two well-separated detections produce distinct, stable IDs with preserved classes."""
    tracker = ByteTrackWrapper(track_thresh=0.5, match_thresh=0.8, track_buffer=30, frame_rate=30)

    # Frame 0: Two well-separated detections
    frame0_detections = [
        Detection(
            xyxy=(50.0, 50.0, 120.0, 120.0),
            confidence=0.95,
            class_id=0,
            class_name="car",
        ),
        Detection(
            xyxy=(400.0, 300.0, 480.0, 420.0),
            confidence=0.92,
            class_id=1,
            class_name="person",
        ),
    ]

    # Frame 1: Minor displacement for both objects
    frame1_detections = [
        Detection(
            xyxy=(53.0, 52.0, 123.0, 122.0),
            confidence=0.94,
            class_id=0,
            class_name="car",
        ),
        Detection(
            xyxy=(402.0, 303.0, 482.0, 423.0),
            confidence=0.91,
            class_id=1,
            class_name="person",
        ),
    ]

    tracks_f0 = tracker.update(detections=frame0_detections, frame_idx=0, frame=dummy_frame)
    tracks_f1 = tracker.update(detections=frame1_detections, frame_idx=1, frame=dummy_frame)

    assert len(tracks_f1) == 2

    # Map tracks by class_id to verify identity stability & metadata preservation
    tracks_by_class = {t.class_id: t for t in tracks_f1}
    assert 0 in tracks_by_class
    assert 1 in tracks_by_class

    car_track = tracks_by_class[0]
    person_track = tracks_by_class[1]

    # Ensure two distinct IDs were assigned
    assert car_track.track_id != person_track.track_id
    assert car_track.class_name == "car"
    assert person_track.class_name == "person"

    # If active in frame 0, verify ID stability across frames
    if len(tracks_f0) == 2:
        tracks_f0_by_class = {t.class_id: t for t in tracks_f0}
        assert car_track.track_id == tracks_f0_by_class[0].track_id
        assert person_track.track_id == tracks_f0_by_class[1].track_id


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
