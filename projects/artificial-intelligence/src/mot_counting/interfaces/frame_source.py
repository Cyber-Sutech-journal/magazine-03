"""Frame source interface (§4.2, §6, §10.8).

Defines the ``IFrameSource`` contract for reading video frames, exposing
frame-rate and resolution metadata, and releasing the underlying resource.
"""

from abc import ABC, abstractmethod

import numpy as np


class IFrameSource(ABC):
    """Abstract interface for sequential video frame access."""

    @abstractmethod
    def read(self) -> tuple[bool, np.ndarray | None]:
        """Read the next frame from the video source.

        Returns:
            A ``(success, frame)`` tuple.  When ``success`` is ``True``,
            ``frame`` is a BGR NumPy array with shape ``(H, W, 3)`` and dtype
            ``uint8``.  When ``success`` is ``False`` and ``frame`` is ``None``,
            the source has reached end-of-video.  When ``success`` is ``False``
            and ``frame`` is not ``None``, a single frame failed to decode and
            the caller should skip it and continue (§12.1).
        """

    @abstractmethod
    def get_fps(self) -> float:
        """Return the video frame rate in frames per second."""

    @abstractmethod
    def get_frame_size(self) -> tuple[int, int]:
        """Return the video frame dimensions.

        Returns:
            A ``(width, height)`` tuple in pixels.
        """

    @abstractmethod
    def release(self) -> None:
        """Release the underlying video capture resource."""
