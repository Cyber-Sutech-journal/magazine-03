"""Multi-object tracker interface (§4.2, §6, §10.4).

Defines the ``ITracker`` contract that every concrete tracker (ByteTrack
wrapper, optional BoT-SORT wrapper, mock tracker for tests, etc.) must
satisfy.
"""

from abc import ABC, abstractmethod

import numpy as np

from mot_counting.types import Detection, Track


class ITracker(ABC):
    """Abstract interface for multi-object tracking across video frames."""

    @abstractmethod
    def update(
        self,
        detections: list[Detection],
        frame_idx: int,
        frame: np.ndarray,
    ) -> list[Track]:
        """Associate detections with existing tracks and return active tracks.

        Args:
            detections: Detections produced by :class:`~mot_counting.interfaces.detector.IDetector`
                for the current frame.
            frame_idx: Zero-based index of the current video frame.
            frame: BGR image as a NumPy array with shape ``(H, W, 3)`` and
                dtype ``uint8``.  The primary v1 implementation
                (``ByteTrackWrapper``) does **not** use this argument because
                ByteTrack is motion-only and requires no appearance features.
                The parameter is part of the interface from v1 so that a
                future ``BoT-SORT`` implementation — which needs the raw image
                for appearance-embedding-based re-identification — can be
                added as a new concrete ``ITracker`` without changing this
                interface, the ``TrackerFactory``, or the
                ``PipelineController`` call site (§6, §10.4).

        Returns:
            A list of :class:`~mot_counting.types.Track` objects representing
            all currently active tracks after association.
        """
