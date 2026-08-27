from pathlib import Path

import cv2
import numpy as np
import pytest

from mot_counting.utils.video_io import OpenCvFrameSource


def create_synthetic_video(
    video_path: Path,
    *,
    frame_count: int = 3,
    width: int = 320,
    height: int = 240,
    fps: float = 10.0,
) -> None:
    """Create a small synthetic AVI video for unit tests."""
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(
        str(video_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        pytest.skip("MJPG video codec is not available")

    try:
        for index in range(frame_count):
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            # Each frame has a distinct color.
            frame[:, :] = (
                index * 50,
                100,
                200 - index * 50,
            )

            writer.write(frame)
    finally:
        writer.release()


def test_constructor_raises_for_missing_video() -> None:
    with pytest.raises(FileNotFoundError):
        OpenCvFrameSource("does-not-exist.avi")


def test_reads_frames_and_returns_none_at_end(tmp_path: Path) -> None:
    video_path = tmp_path / "synthetic.avi"
    create_synthetic_video(video_path, frame_count=3)

    source = OpenCvFrameSource(video_path)

    try:
        frames = []

        for _ in range(3):
            success, frame = source.read()

            assert success is True
            assert frame is not None
            assert isinstance(frame, np.ndarray)
            assert frame.shape == (240, 320, 3)
            assert frame.dtype == np.uint8

            frames.append(frame)

        success, frame = source.read()

        assert success is False
        assert frame is None
    finally:
        source.release()


def test_returns_video_metadata(tmp_path: Path) -> None:
    video_path = tmp_path / "metadata.avi"
    create_synthetic_video(
        video_path,
        frame_count=2,
        width=640,
        height=360,
        fps=15.0,
    )

    source = OpenCvFrameSource(video_path)

    try:
        fps = source.get_fps()
        width, height = source.get_frame_size()

        assert fps == pytest.approx(15.0, abs=1.0)
        assert (width, height) == (640, 360)
    finally:
        source.release()


def test_release_is_safe(tmp_path: Path) -> None:
    video_path = tmp_path / "release.avi"
    create_synthetic_video(video_path, frame_count=1)

    source = OpenCvFrameSource(video_path)

    source.release()
    source.release()
