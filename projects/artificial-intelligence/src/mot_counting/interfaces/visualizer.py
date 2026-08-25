"""Visualizer interface (§4.2, §6, §7.6, §10.7).

Defines the ``IVisualizer`` contract for drawing bounding boxes, counting
lines, and live counters onto annotated output frames.
"""

from abc import ABC, abstractmethod

import numpy as np

from mot_counting.types import Track


class IVisualizer(ABC):
    """Abstract interface for annotating video frames."""

    @abstractmethod
    def draw(
        self,
        frame: np.ndarray,
        tracks: list[Track],
        lines: list,
        counters: dict,
    ) -> np.ndarray:
        """Draw tracks, counting lines, and live counters onto a frame.

        Args:
            frame: BGR image as a NumPy array with shape ``(H, W, 3)`` and
                dtype ``uint8``.  Implementations must not mutate this array
                in place; return a new annotated copy.
            tracks: Active tracks to draw (bounding boxes, class labels, track
                IDs).
            lines: Configured counting lines to draw, typically
                :class:`~mot_counting.config.LineConfig` objects from the
                application configuration.
            counters: Running count totals as returned by
                :meth:`~mot_counting.interfaces.crossing.ICrossingLogic.get_counters`.

        Returns:
            A new BGR image array with all annotations drawn.
        """
