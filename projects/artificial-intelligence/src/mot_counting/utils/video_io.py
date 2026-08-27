"""OpenCV-based implementation of the video frame source interface."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from mot_counting.interfaces.frame_source import IFrameSource


class OpenCvFrameSource(IFrameSource):
    """Read sequential frames and metadata from a video file using OpenCV."""

    _DEFAULT_FPS = 30.0

    def __init__(self, video_path: str | Path) -> None:
        """Open a video file and fail fast if it cannot be accessed.

        Args:
            video_path: Path to the input video file.

        Raises:
            FileNotFoundError: If the specified video file does not exist.
            RuntimeError: If OpenCV cannot open the video file.
        """
        self._video_path = Path(video_path)

        if not self._video_path.is_file():
            raise FileNotFoundError(f"Video file does not exist: {self._video_path}")

        self._capture = cv2.VideoCapture(str(self._video_path))

        if not self._capture.isOpened():
            self._capture.release()
            raise RuntimeError(f"Could not open video file: {self._video_path}")

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Read the next frame from the video source.

        Returns:
            A ``(success, frame)`` tuple. At end-of-video or upon read error,
            returns ``(False, None)``.
        """
        success, frame = self._capture.read()

        if not success or frame is None:
            return False, None

        return True, frame

    def get_fps(self) -> float:
        """Return the video frame rate, using a safe fallback if invalid."""
        fps = float(self._capture.get(cv2.CAP_PROP_FPS))

        if not np.isfinite(fps) or fps <= 0:
            return self._DEFAULT_FPS

        return fps

    def get_frame_size(self) -> tuple[int, int]:
        """Return the video dimensions as ``(width, height)`` in pixels."""
        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return width, height

    def release(self) -> None:
        """Release the underlying OpenCV video-capture resource."""
        self._capture.release()
