"""Manual composition root — single place for DI wiring (§4.6).

This module is the only location where concrete implementations are chosen
and assembled into a :class:`~mot_counting.controllers.pipeline_controller.PipelineController`.
Callers (e.g. ``scripts/run_pipeline.py``) must import only
:func:`build_pipeline` from here — never concrete detector/tracker classes.

Wiring status (T09 skeleton)
----------------------------
Several steps are intentionally stubbed until downstream tasks land:

- **Model loading** — placeholder objects stand in for Ultralytics YOLO26 and
  ByteTrack backends (T14/T15, wired in T19).
- **Class-list validation** — deferred to T19 (§4.1, §10.1).
- **Line-geometry validation** — deferred to T10/T19 inside
  ``PipelineController`` once the video is opened (§7.3).
- **Factory ``.create()``** — factories are constructed and called, but fall
  back to no-op interface stubs when concrete wrappers are not yet wired
  (T19).
- **Logger / Visualizer observers** — ``Subject`` is created and returned on
  the controller; T16/T17 will subscribe their concrete observers later.
"""

from __future__ import annotations

import numpy as np

from mot_counting.config import AppConfig, load_config
from mot_counting.controllers.pipeline_controller import PipelineController
from mot_counting.factories.detector_factory import DetectorFactory
from mot_counting.factories.tracker_factory import TrackerFactory
from mot_counting.interfaces.crossing import ICrossingLogic
from mot_counting.interfaces.detector import IDetector
from mot_counting.interfaces.frame_source import IFrameSource
from mot_counting.interfaces.repository import IEventRepository
from mot_counting.interfaces.tracker import ITracker
from mot_counting.interfaces.visualizer import IVisualizer
from mot_counting.observers.base import Subject
from mot_counting.types import CrossingEvent, Detection, Track

# ---------------------------------------------------------------------------
# Wiring stubs — replaced by real implementations in T19
# ---------------------------------------------------------------------------


class _StubDetector(IDetector):
    """No-op detector used until Yolo26Detector is wired in T19."""

    def predict(self, frame: np.ndarray) -> list[Detection]:
        return []


class _StubTracker(ITracker):
    """No-op tracker used until ByteTrackWrapper is wired in T19."""

    def update(
        self,
        detections: list[Detection],
        frame_idx: int,
        frame: np.ndarray,
    ) -> list[Track]:
        return []


class _StubCrossingLogic(ICrossingLogic):
    """No-op crossing logic used until CrossingLogic is implemented (T11)."""

    def process(
        self,
        tracks: list[Track],
        frame_idx: int,
        timestamp_seconds: float,
    ) -> list[CrossingEvent]:
        return []

    def get_counters(self) -> dict:
        return {}


class _StubEventRepository(IEventRepository):
    """In-memory no-op repository used until CsvEventRepository exists (T12)."""

    def save(self, event: CrossingEvent) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class _StubVisualizer(IVisualizer):
    """Pass-through visualizer used until the real visualizer exists (T17)."""

    def draw(
        self,
        frame: np.ndarray,
        tracks: list[Track],
        lines: list,
        counters: dict,
    ) -> np.ndarray:
        return frame.copy()


class _StubFrameSource(IFrameSource):
    """No-op frame source used until OpenCV video reader exists (T13)."""

    def read(self) -> tuple[bool, np.ndarray | None]:
        return False, None

    def get_fps(self) -> float:
        return 30.0

    def get_frame_size(self) -> tuple[int, int]:
        return (640, 480)

    def release(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Model-loading stubs (§4.1 boundary)
# ---------------------------------------------------------------------------


def _load_detector_model(config: AppConfig) -> object:
    """Load the underlying YOLO26 model object once per pipeline run.

    TODO(T19): replace with real Ultralytics model loading once T14 lands::

        from ultralytics import YOLO
        return YOLO(f"{config.detection.model_variant}.pt")

    Args:
        config: Validated application configuration.

    Returns:
        A placeholder object until real loading is wired in T19.
    """
    return object()


def _load_tracker_backend(config: AppConfig) -> object:
    """Initialise the underlying tracker backend once per pipeline run.

    TODO(T19): replace with real ByteTrack / BoT-SORT initialisation once T15
    lands.  The exact library and constructor depend on Farzad's chosen
    integration path.

    Args:
        config: Validated application configuration.

    Returns:
        A placeholder object until real initialisation is wired in T19.
    """
    return object()


def _validate_classes_against_model(config: AppConfig, loaded_model: object) -> None:
    """Fail-fast if configured class names are absent from the loaded model.

    TODO(T19): implement real validation against ``loaded_model.names``::

        model_names = set(loaded_model.names.values())
        unknown = [c for c in config.detection.classes if c not in model_names]
        if unknown:
            raise ValueError(
                f"Unknown class names in config: {unknown}.  "
                f"Valid names on loaded model: {sorted(model_names)}"
            )

    Args:
        config: Validated application configuration.
        loaded_model: Already-loaded YOLO model object from
            :func:`_load_detector_model`.
    """


def _create_detector(
    config: AppConfig,
    loaded_model: object,
) -> IDetector:
    """Construct an ``IDetector`` via the factory, falling back to a stub.

    Args:
        config: Validated application configuration.
        loaded_model: Already-loaded model object (never loaded inside the
            factory).

    Returns:
        An ``IDetector`` implementation.
    """
    factory = DetectorFactory(
        confidence_threshold=config.detection.confidence_threshold,
        classes=config.detection.classes,
    )
    try:
        return factory.create(config.detection.model_variant, loaded_model)
    except NotImplementedError:
        # TODO(T19): remove this fallback once DetectorFactory.create() wires
        # Yolo26Detector (T14).
        return _StubDetector()


def _create_tracker(
    config: AppConfig,
    loaded_tracker: object,
) -> ITracker:
    """Construct an ``ITracker`` via the factory, falling back to a stub.

    Args:
        config: Validated application configuration.
        loaded_tracker: Already-initialised tracker backend (never created
            inside the factory).

    Returns:
        An ``ITracker`` implementation.
    """
    factory = TrackerFactory(
        track_thresh=config.tracker.track_thresh,
        match_thresh=config.tracker.match_thresh,
        track_buffer=config.tracker.track_buffer,
    )
    try:
        return factory.create(config.tracker.type, loaded_tracker)
    except NotImplementedError:
        # TODO(T19): remove this fallback once TrackerFactory.create() wires
        # ByteTrackWrapper (T15).
        return _StubTracker()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_pipeline(config_path: str) -> PipelineController:
    """Load configuration and wire all pipeline dependencies.

    This is the single entry point for constructing a fully injected
    :class:`~mot_counting.controllers.pipeline_controller.PipelineController`.
    Callers must not instantiate concrete detector/tracker classes directly.

    Args:
        config_path: Path to a YAML configuration file (e.g.
            ``"configs/ci.yaml"``).

    Returns:
        A :class:`~mot_counting.controllers.pipeline_controller.PipelineController`
        with all dependencies injected via their ``abc.ABC`` interfaces.
    """
    config = load_config(config_path)

    loaded_detector_model = _load_detector_model(config)
    _validate_classes_against_model(config, loaded_detector_model)

    loaded_tracker_backend = _load_tracker_backend(config)

    detector = _create_detector(config, loaded_detector_model)
    tracker = _create_tracker(config, loaded_tracker_backend)

    # TODO(T11): replace _StubCrossingLogic with real CrossingLogic(config).
    crossing_logic: ICrossingLogic = _StubCrossingLogic()

    # TODO(T12): replace _StubEventRepository with CsvEventRepository(config.events.output_csv).
    event_repository: IEventRepository = _StubEventRepository()

    # TODO(T17): replace _StubVisualizer with concrete IVisualizer implementation.
    visualizer: IVisualizer = _StubVisualizer()

    # TODO(T13): replace _StubFrameSource with OpenCV-backed video reader
    # using config.video.path.
    frame_source: IFrameSource = _StubFrameSource()

    subject = Subject()
    # TODO(T16): subject.subscribe(Logger(...))
    # TODO(T17): subject.subscribe(VisualizerObserver(...))  — if distinct from IVisualizer

    return PipelineController(
        config=config,
        frame_source=frame_source,
        detector=detector,
        tracker=tracker,
        crossing_logic=crossing_logic,
        event_repository=event_repository,
        visualizer=visualizer,
        subject=subject,
    )
