"""Visualizer interface (§4.2, §6, §7.6, §10.7).

Defines the ``IVisualizer`` contract for drawing bounding boxes, counting
lines, and live counters onto annotated output frames.

The visualizer is also an :class:`~mot_counting.observers.base.Observer`
(§4.3, §10.7): it is notified once per frame after crossing state is updated.
``draw()`` remains the pure rendering primitive; ``Observer.update()`` is the
pipeline callback.  The current raw frame is bound via :meth:`set_frame`
because the Observer contract does not include image data.
"""

from abc import abstractmethod

import numpy as np

from mot_counting.observers.base import Observer
from mot_counting.types import Track


class IVisualizer(Observer):
    """Abstract interface for annotating video frames as an Observer."""

    def set_frame(self, frame: np.ndarray) -> None:
        """Bind the current raw frame for the next :meth:`Observer.update` call.

        The controller supplies the frame immediately before notifying the
        ``Subject``.  Rendering itself happens inside ``update()``, not here.
        """
        self._current_frame = frame

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
