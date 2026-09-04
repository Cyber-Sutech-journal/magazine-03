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
    # Non-None sentinel returned with success=False so PipelineController can
    # skip a recoverable decode failure instead of treating it as EOF (§12.1).
    _DECODE_FAILURE_FRAME: np.ndarray = np.empty((0, 0, 3), dtype=np.uint8)

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

        Distinguishes end-of-video from a recoverable failure:

        - ``grab()`` succeeds and ``retrieve()`` fails → skippable decode error.
        - ``grab()`` fails while ``CAP_PROP_POS_FRAMES`` is still before a
          known ``CAP_PROP_FRAME_COUNT`` → skippable grab/decode error; the
          capture is advanced so the next ``read()`` can reach later frames.
        - ``grab()`` fails with no remaining frames (or an unknown/unreliable
          frame count) → ``(False, None)`` EOF.

        Returns:
            ``(True, frame)`` on a successful decode.
            ``(False, None)`` at natural end-of-video.
            ``(False, sentinel)`` when one frame could not be decoded.
        """
        pos_before = float(self._capture.get(cv2.CAP_PROP_POS_FRAMES))
        grabbed = self._capture.grab()
        if not grabbed:
            if self._is_eof_after_failed_grab(pos_before):
                return False, None
            self._advance_after_failed_grab(pos_before)
            return False, self._DECODE_FAILURE_FRAME

        retrieved, frame = self._capture.retrieve()
        if not retrieved or frame is None:
            return False, self._DECODE_FAILURE_FRAME

        return True, frame

    def _reported_frame_count(self) -> float:
        return float(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def _reported_position(self) -> float:
        return float(self._capture.get(cv2.CAP_PROP_POS_FRAMES))

    @staticmethod
    def _has_known_remaining_frames(pos: float, total: float) -> bool:
        """True when the container reports frames after *pos*."""
        return np.isfinite(total) and total > 0 and np.isfinite(pos) and pos < total

    def _is_eof_after_failed_grab(self, pos_before: float) -> bool:
        """Return True when a failed ``grab()`` should terminate the stream.

        ``grab() == False`` is definitive EOF only when the container does not
        report later frames.  A known ``FRAME_COUNT`` still ahead of the
        current position means this slot failed and later frames may exist.
        """
        total = self._reported_frame_count()
        pos_after = self._reported_position()
        pos = pos_after if np.isfinite(pos_after) else pos_before
        remaining = OpenCvFrameSource._has_known_remaining_frames(
            pos, total
        ) or OpenCvFrameSource._has_known_remaining_frames(pos_before, total)
        return not remaining

    def _advance_after_failed_grab(self, pos_before: float) -> None:
        """Move past a failed grab so the next read is not stuck on this slot."""
        pos_after = self._reported_position()
        if not np.isfinite(pos_before):
            return
        if np.isfinite(pos_after) and pos_after > pos_before:
            return
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, int(pos_before) + 1)

    def get_fps(self) -> float:
        """Return the video frame rate in frames per second.

        Returns the FPS reported by cv2.CAP_PROP_FPS. If the reported value
        is non-positive or non-finite (NaN, inf), falls back to the documented
        project baseline of ``DEFAULT_FPS`` (30.0).

        Note:
            Downstream components rely on FPS to convert seconds-based timeouts
            (e.g., stale-timeout, cooldown) into frame counts.
        """
        fps = float(self._capture.get(cv2.CAP_PROP_FPS))

        if not np.isfinite(fps) or fps <= 0:
            return self.DEFAULT_FPS

        return fps

    def get_frame_size(self) -> tuple[int, int]:
        """Return the video dimensions as ``(width, height)`` in pixels.

        Raises:
            RuntimeError: If the video reports invalid frame dimensions.
        """
        width = float(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = float(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if not np.isfinite(width) or not np.isfinite(height) or width <= 0 or height <= 0:
            raise RuntimeError(f"Invalid video frame dimensions: width={width}, height={height}")

        return int(width), int(height)

    def release(self) -> None:
        """Release the underlying OpenCV video-capture resource."""
        self._capture.release()
