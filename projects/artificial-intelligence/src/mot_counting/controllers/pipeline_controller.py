"""Pipeline controller skeleton (§4.4, T10).

Owns the high-level orchestration: read → detect → track → update crossing
state → notify observers.  The full frame loop and lifecycle management are
implemented in T10; this skeleton exists so the composition root (T09) can
wire and return a callable object before every downstream module is ready.
"""

from __future__ import annotations

from mot_counting.config import AppConfig
from mot_counting.interfaces.crossing import ICrossingLogic
from mot_counting.interfaces.detector import IDetector
from mot_counting.interfaces.frame_source import IFrameSource
from mot_counting.interfaces.repository import IEventRepository
from mot_counting.interfaces.tracker import ITracker
from mot_counting.interfaces.visualizer import IVisualizer
from mot_counting.observers.base import Subject


class PipelineController:
    """Orchestrates the video analytics pipeline via constructor injection.

    All dependencies are interface-typed.  The controller must never import
    or depend on concrete detector/tracker implementations directly (§4.2).
    """

    def __init__(
        self,
        config: AppConfig,
        frame_source: IFrameSource,
        detector: IDetector,
        tracker: ITracker,
        crossing_logic: ICrossingLogic,
        event_repository: IEventRepository,
        visualizer: IVisualizer,
        subject: Subject,
    ) -> None:
        """Initialise the controller with all pipeline dependencies.

        Args:
            config: Validated application configuration.
            frame_source: Video frame reader.
            detector: Object detector (``IDetector``).
            tracker: Multi-object tracker (``ITracker``).
            crossing_logic: Line-crossing state machine (``ICrossingLogic``).
            event_repository: Crossing-event persistence (``IEventRepository``).
            visualizer: Frame annotator (``IVisualizer``).
            subject: Observer subject for Logger/Visualizer side effects.
        """
        self._config = config
        self._frame_source = frame_source
        self._detector = detector
        self._tracker = tracker
        self._crossing_logic = crossing_logic
        self._event_repository = event_repository
        self._visualizer = visualizer
        self._subject = subject

    @property
    def config(self) -> AppConfig:
        """Return the validated application configuration."""
        return self._config

    @property
    def subject(self) -> Subject:
        """Return the observer subject for side-effect consumers."""
        return self._subject

    def run(self) -> None:
        """Execute the full video analytics pipeline.

        TODO(T10): implement the synchronous frame loop:
        read → detect → track → update crossing state → notify observers.
        """
        raise NotImplementedError("Pipeline frame loop not yet implemented — see T10.")
