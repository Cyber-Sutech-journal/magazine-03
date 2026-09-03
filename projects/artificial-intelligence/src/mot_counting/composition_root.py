"""Manual composition root — single place for DI wiring (§4.6).

This module is the **only** location where concrete implementations are chosen
and assembled into a :class:`~mot_counting.controllers.pipeline_controller.PipelineController`.
Callers (e.g. ``scripts/run_pipeline.py``) must import only
:func:`build_pipeline` from here — never concrete detector/tracker classes.

Wiring (T19)
------------
All stubs replaced by real implementations:

- YOLO26 model loaded once via Ultralytics.
- Class-list validated against ``model.names`` immediately after loading (§4.1, §7.1).
- ``Yolo26Detector`` and ``ByteTrackWrapper`` wired via their respective Factories.
- Real ``CrossingLogic``, ``CsvEventRepository``, ``OpenCvVisualizer``,
  ``OpenCvFrameSource`` injected into ``PipelineController``.
- ``LoggerObserver`` subscribed to the ``Subject``.
- ``cv2.VideoWriter`` created and passed to ``PipelineController`` for annotated output.
"""

from __future__ import annotations

import logging
import os

import cv2

from mot_counting.config import AppConfig, load_config
from mot_counting.controllers.pipeline_controller import PipelineController
from mot_counting.crossing.crossing_logic import CrossingLogic
from mot_counting.factories.detector_factory import DetectorFactory
from mot_counting.factories.tracker_factory import TrackerFactory
from mot_counting.interfaces.detector import IDetector
from mot_counting.interfaces.tracker import ITracker
from mot_counting.observers.base import Subject
from mot_counting.observers.logger_observer import LoggerObserver
from mot_counting.repositories.csv_event_repository import CsvEventRepository
from mot_counting.utils.video_io import OpenCvFrameSource
from mot_counting.visualizers.opencv_visualizer import OpenCvVisualizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loading (§4.1 boundary — happens in composition root, not in Factories)
# ---------------------------------------------------------------------------


def _load_detector_model(config: AppConfig) -> object:
    """Load the Ultralytics YOLO26 model object once per pipeline run.

    Args:
        config: Validated application configuration.

    Returns:
        An ``ultralytics.YOLO`` model object with weights loaded from a local
        ``.pt`` file (baked into the Docker image at ``/app``, or present in
        the working directory).  No runtime download is required.

    Raises:
        RuntimeError: If the model fails to load.
    """
    from pathlib import Path

    from ultralytics import YOLO  # local import keeps startup fast if YOLO is not needed

    model_variant = config.detection.model_variant
    weight_name = f"{model_variant}.pt"
    # Prefer an on-disk file (baked into the Docker image at /app, or present in cwd)
    # so Ultralytics never falls back to a runtime download (§11).
    candidates = [
        Path(weight_name),
        Path("/app") / weight_name,
        Path(__file__).resolve().parents[2] / weight_name,
    ]
    weight_path = next((p for p in candidates if p.is_file()), Path(weight_name))
    logger.info("Loading YOLO model from %s", weight_path)
    try:
        model = YOLO(str(weight_path))
    except Exception as exc:
        raise RuntimeError(f"Failed to load YOLO model '{weight_path}': {exc}") from exc
    logger.info("YOLO model loaded successfully.")
    return model


def _validate_classes_against_model(config: AppConfig, loaded_model: object) -> None:
    """Fail-fast if configured class names are absent from the loaded model (§4.1, §7.1).

    Args:
        config: Validated application configuration.
        loaded_model: Already-loaded YOLO model object from
            :func:`_load_detector_model`.

    Raises:
        ValueError: If any configured class name is not in ``loaded_model.names``.
    """
    names_attr = getattr(loaded_model, "names", None)
    if not isinstance(names_attr, dict):
        raise TypeError(
            "Loaded YOLO model has no usable .names mapping; cannot validate detection.classes."
        )
    model_names = {str(name) for name in names_attr.values()}
    unknown = [c for c in config.detection.classes if c not in model_names]
    if unknown:
        raise ValueError(
            f"Unknown class name(s) in config detection.classes: {unknown}.  "
            f"Valid class names available on the loaded model: {sorted(model_names)}"
        )


# ---------------------------------------------------------------------------
# Component construction helpers
# ---------------------------------------------------------------------------


def _create_detector(config: AppConfig, loaded_model: object) -> IDetector:
    """Construct an ``IDetector`` via ``DetectorFactory``.

    Args:
        config: Validated application configuration.
        loaded_model: Already-loaded YOLO model object.

    Returns:
        A ``Yolo26Detector`` instance satisfying ``IDetector``.
    """
    factory = DetectorFactory(
        confidence_threshold=config.detection.confidence_threshold,
        classes=config.detection.classes,
    )
    return factory.create(config.detection.model_variant, loaded_model)


def _create_tracker(config: AppConfig, fps: float) -> ITracker:
    """Construct an ``ITracker`` via ``TrackerFactory``.

    Args:
        config: Validated application configuration.
        fps: Video frame rate used to calibrate the tracker's internal buffer.

    Returns:
        A ``ByteTrackWrapper`` instance satisfying ``ITracker``.
    """
    factory = TrackerFactory(
        track_thresh=config.tracker.track_thresh,
        match_thresh=config.tracker.match_thresh,
        track_buffer=config.tracker.track_buffer,
        frame_rate=max(1, round(fps)),
    )
    # ByteTrackWrapper is self-contained (no pre-loaded external object needed).
    return factory.create(config.tracker.type, None)


def _create_video_writer(
    config: AppConfig,
    frame_source: OpenCvFrameSource,
) -> cv2.VideoWriter:
    """Create and return a configured ``cv2.VideoWriter``.

    Args:
        config: Validated application configuration.
        frame_source: Already-opened frame source, used to read FPS and frame size.

    Returns:
        An opened ``cv2.VideoWriter`` ready to receive annotated frames.

    Raises:
        RuntimeError: If the VideoWriter cannot be opened.
    """
    output_path = config.visualization.output_video
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fps = frame_source.get_fps()
    width, height = frame_source.get_frame_size()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(
            f"cv2.VideoWriter could not be opened for path '{output_path}'.  "
            "Check that the output directory exists and the codec is available."
        )
    return writer


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_pipeline(config_path: str) -> PipelineController:
    """Load configuration and wire all pipeline dependencies.

    This is the single entry point for constructing a fully injected
    :class:`~mot_counting.controllers.pipeline_controller.PipelineController`.
    Callers must not instantiate concrete detector/tracker classes directly.

    Wiring sequence (§4.6):

    1. Load and validate the YAML config.
    2. Load the Ultralytics YOLO26 model object (once).
    3. Validate configured class names against ``model.names`` — fail fast on mismatch.
    4. Open the video via ``OpenCvFrameSource`` (provides FPS and frame dimensions).
    5. Construct ``Yolo26Detector``, ``ByteTrackWrapper``, ``CrossingLogic``,
       ``CsvEventRepository``, ``OpenCvVisualizer`` via factories / constructors.
    6. Subscribe ``LoggerObserver`` to the ``Subject``.
    7. Create ``cv2.VideoWriter`` for annotated output.
    8. Construct and return ``PipelineController`` with all dependencies injected.

    Args:
        config_path: Path to a YAML configuration file (e.g. ``"configs/ci.yaml"``).

    Returns:
        A :class:`~mot_counting.controllers.pipeline_controller.PipelineController`
        with all dependencies injected via their ``abc.ABC`` interfaces.

    Raises:
        FileNotFoundError: If the config file or video file cannot be found.
        ValueError: If a configured class name is absent from the loaded model.
        RuntimeError: If the YOLO model or VideoWriter cannot be opened.
    """
    # -- 1. Load config -------------------------------------------------------
    config = load_config(config_path)

    # Configure root logging once at startup (§12.2).
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    # -- 2. Load YOLO26 model (§4.1) -----------------------------------------
    loaded_detector_model = _load_detector_model(config)

    # -- 3. Class-list fail-fast validation (§4.1, §7.1) ---------------------
    _validate_classes_against_model(config, loaded_detector_model)

    # -- 4. Open video frame source (needed for FPS + frame size) ------------
    frame_source = OpenCvFrameSource(config.video.path)
    fps = frame_source.get_fps()

    # -- 5. Construct pipeline components ------------------------------------
    detector = _create_detector(config, loaded_detector_model)
    tracker = _create_tracker(config, fps)
    crossing_logic = CrossingLogic(
        lines=config.lines,
        config=config.crossing_logic,
        fps=fps,
    )
    event_repository = CsvEventRepository(config.events.output_csv)
    visualizer = OpenCvVisualizer()

    # -- 6. Observer subscriptions (§4.3) ------------------------------------
    subject = Subject()
    log_observer = LoggerObserver()
    subject.subscribe(log_observer)

    # -- 7. VideoWriter for annotated output (§7.6) --------------------------
    video_writer = _create_video_writer(config, frame_source)

    # -- 8. Assemble controller ----------------------------------------------
    controller = PipelineController(
        config=config,
        frame_source=frame_source,
        detector=detector,
        tracker=tracker,
        crossing_logic=crossing_logic,
        event_repository=event_repository,
        visualizer=visualizer,
        subject=subject,
        video_writer=video_writer,
    )

    log_observer.log_run_start(video_path=config.video.path)
    return controller
