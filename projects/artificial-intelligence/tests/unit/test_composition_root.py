"""Unit tests for the composition root wiring skeleton (§4.6, T09).

Full behavioural testing of the pipeline frame loop happens in T10/T19.
These tests verify only that :func:`build_pipeline` assembles a
:class:`~mot_counting.controllers.pipeline_controller.PipelineController`
with the expected public shape using stubs/mocks.
"""

from __future__ import annotations

from pathlib import Path

from mot_counting.composition_root import build_pipeline
from mot_counting.controllers.pipeline_controller import PipelineController
from mot_counting.interfaces.crossing import ICrossingLogic
from mot_counting.interfaces.detector import IDetector
from mot_counting.interfaces.frame_source import IFrameSource
from mot_counting.interfaces.repository import IEventRepository
from mot_counting.interfaces.tracker import ITracker
from mot_counting.interfaces.visualizer import IVisualizer
from mot_counting.observers.base import Subject

PROJECT_ROOT = Path(__file__).parent.parent.parent


def test_build_pipeline_ci_config_returns_pipeline_controller() -> None:
    """build_pipeline must return a PipelineController wired from configs/ci.yaml."""
    controller = build_pipeline(str(PROJECT_ROOT / "configs" / "ci.yaml"))

    assert isinstance(controller, PipelineController)


def test_build_pipeline_ci_config_loads_expected_detection_settings() -> None:
    controller = build_pipeline(str(PROJECT_ROOT / "configs" / "ci.yaml"))

    assert controller.config.detection.model_variant == "yolo26n"
    assert controller.config.video.path == "data/ci_sample_clip.mp4"
    assert controller.config.detection.imgsz == 640
    assert controller._detector.imgsz == controller.config.detection.imgsz  # noqa: SLF001


def test_build_pipeline_exposes_interface_typed_dependencies() -> None:
    """Controller dependencies must be interface-typed, not concrete classes."""
    controller = build_pipeline(str(PROJECT_ROOT / "configs" / "ci.yaml"))

    # Access injected collaborators through the controller's public API.
    assert isinstance(controller.config, object)
    assert isinstance(controller.subject, Subject)

    # Private attributes are checked here only to confirm wiring shape before T10.
    assert isinstance(controller._detector, IDetector)  # noqa: SLF001
    assert isinstance(controller._tracker, ITracker)  # noqa: SLF001
    assert isinstance(controller._crossing_logic, ICrossingLogic)  # noqa: SLF001
    assert isinstance(controller._event_repository, IEventRepository)  # noqa: SLF001
    assert isinstance(controller._visualizer, IVisualizer)  # noqa: SLF001
    assert isinstance(controller._frame_source, IFrameSource)  # noqa: SLF001


def test_build_pipeline_default_config_also_succeeds() -> None:
    controller = build_pipeline(str(PROJECT_ROOT / "configs" / "default.yaml"))

    assert isinstance(controller, PipelineController)
    assert controller.config.detection.model_variant == "yolo26m"
