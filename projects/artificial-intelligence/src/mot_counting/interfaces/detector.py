"""Object detector interface (§4.2, §6, §10.3).

Defines the ``IDetector`` contract that every concrete detector (YOLO26
wrapper, mock detector for tests, etc.) must satisfy.  The composition root
loads the underlying model once and injects it at construction time — the
model is never loaded inside ``predict()``.
"""

from abc import ABC, abstractmethod

import numpy as np

from mot_counting.types import Detection


class IDetector(ABC):
    """Abstract interface for frame-by-frame object detection."""

    @abstractmethod
    def predict(self, frame: np.ndarray) -> list[Detection]:
        """Run inference on a single video frame.

        Args:
            frame: BGR image as a NumPy array with shape ``(H, W, 3)`` and
                dtype ``uint8``.

        Returns:
            A list of :class:`~mot_counting.types.Detection` objects for
            every object that passes the configured confidence threshold and
            class filter.  Returns an empty list when nothing is detected.
        """
