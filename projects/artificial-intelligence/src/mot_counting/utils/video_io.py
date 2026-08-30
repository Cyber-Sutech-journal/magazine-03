"""OpenCV-based implementation of the video frame source interface."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from mot_counting.interfaces.frame_source import IFrameSource


class OpenCvFrameSource(IFrameSource):
    """Read sequential frames and metadata from a video file using OpenCV."""

    # Fallback frame rate used when a video container does not report a valid,
    # finite FPS (e.g. 0, negative, NaN, or inf from cv2.CAP_PROP_FPS).
    # Rationale: Target surveillance and traffic footage in this project is predominantly
    # 25-30 FPS. Setting 30.0 FPS provides a standardized baseline and minimizes drift
    # when converting cooldown and stale-timeout thresholds from seconds to frames downstream.
    DEFAULT_FPS: float = 30.0

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
        """Return the video frame rate in frames per second.

        Returns the FPS reported by cv2.CAP_PROP_FPS. If the reported value
        is non-positive or non-finite (NaN, inf), falls back to the documented
        project baseline of ``_DEFAULT_FPS`` (30.0).

        Note:
            Downstream components rely on FPS to convert seconds-based timeouts
            (e.g., stale-timeout, cooldown) into frame counts.
        """
        fps = float(self._capture.get(cv2.CAP_PROP_FPS))

        if not np.isfinite(fps) or fps <= 0:
            return self.DEFAULT_FPS

        return fps

    def get_frame_size(self) -> tuple[int, int]:
        """Return the video dimensions as ``(width, height)`` in pixels."""
        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return width, height

    def release(self) -> None:
        """Release the underlying OpenCV video-capture resource."""
        self._capture.release()
