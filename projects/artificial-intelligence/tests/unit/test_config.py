"""Unit tests for mot_counting.config — schema validation and loader."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from mot_counting.config import AppConfig, load_config

# Absolute path to the project root (two levels above tests/unit/)
PROJECT_ROOT = Path(__file__).parent.parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_minimal_dict(**overrides: object) -> dict:
    """Return a minimal valid AppConfig dict, with optional overrides applied."""
    base: dict = {
        "video": {"path": "data/clip.mp4", "output_dir": "outputs/"},
        "detection": {
            "model_variant": "yolo26m",
            "imgsz": 640,
            "confidence_threshold": 0.4,
            "classes": ["person", "car"],
        },
        "tracker": {
            "type": "bytetrack",
            "track_thresh": 0.5,
            "match_thresh": 0.8,
            "track_buffer": 30,
        },
        "lines": [
            {
                "line_id": "main_line",
                "point_a": [100, 400],
                "point_b": [800, 400],
                "positive_direction": "A_to_B",
            }
        ],
        "crossing_logic": {
            "reference_point": "bottom_center",
            "history_length": 8,
            "confirmation_majority_threshold": 0.7,
            "cooldown_seconds": 1.5,
            "stale_track_timeout_seconds": 2.0,
            "min_displacement_px": None,
            "min_velocity_px_per_s": None,
        },
        "events": {"output_csv": "outputs/events.csv"},
        "evaluation": {"matching_tolerance_seconds": 1.0},
        "visualization": {"output_video": "outputs/annotated.mp4", "draw_trails": False},
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Load default.yaml
# ---------------------------------------------------------------------------


def test_load_default_yaml_succeeds() -> None:
    cfg = load_config(str(PROJECT_ROOT / "configs" / "default.yaml"))

    assert cfg.video.path == "data/ci_sample_clip.mp4"
    assert cfg.video.output_dir == "outputs/"

    assert cfg.detection.model_variant == "yolo26m"
    assert cfg.detection.imgsz == 640
    assert cfg.detection.confidence_threshold == pytest.approx(0.4)
    assert cfg.detection.classes == ["person", "car"]

    assert cfg.tracker.type == "bytetrack"
    assert cfg.tracker.track_thresh == pytest.approx(0.5)
    assert cfg.tracker.match_thresh == pytest.approx(0.8)
    assert cfg.tracker.track_buffer == 30

    assert len(cfg.lines) == 1
    line = cfg.lines[0]
    assert line.line_id == "main_line"
    assert line.point_a == [100, 400]
    assert line.point_b == [800, 400]
    assert line.positive_direction == "A_to_B"

    assert cfg.crossing_logic.reference_point == "bottom_center"
    assert cfg.crossing_logic.history_length == 8
    assert cfg.crossing_logic.confirmation_majority_threshold == pytest.approx(0.7)
    assert cfg.crossing_logic.cooldown_seconds == pytest.approx(1.5)
    assert cfg.crossing_logic.stale_track_timeout_seconds == pytest.approx(2.0)
    assert cfg.crossing_logic.min_displacement_px is None
    assert cfg.crossing_logic.min_velocity_px_per_s is None

    assert cfg.events.output_csv == "outputs/events.csv"
    assert cfg.evaluation.matching_tolerance_seconds == pytest.approx(1.0)
    assert cfg.visualization.output_video == "outputs/annotated.mp4"
    assert cfg.visualization.draw_trails is False


# ---------------------------------------------------------------------------
# Load ci.yaml
# ---------------------------------------------------------------------------


def test_load_ci_yaml_uses_yolo26n() -> None:
    cfg = load_config(str(PROJECT_ROOT / "configs" / "ci.yaml"))
    assert cfg.detection.model_variant == "yolo26n"
    assert cfg.video.path == "data/ci_sample_clip.mp4"


# ---------------------------------------------------------------------------
# Load configs/examples/multi_line.yaml
# ---------------------------------------------------------------------------


def test_load_multi_line_example_succeeds() -> None:
    cfg = load_config(str(PROJECT_ROOT / "configs" / "examples" / "multi_line.yaml"))
    assert len(cfg.lines) == 2
    ids = {line.line_id for line in cfg.lines}
    assert ids == {"entrance", "exit"}


# ---------------------------------------------------------------------------
# Field validators — invalid inputs raise ValidationError
# ---------------------------------------------------------------------------


def test_confidence_threshold_above_one_raises() -> None:
    data = _make_minimal_dict()
    data["detection"] = {**data["detection"], "confidence_threshold": 1.5}  # type: ignore[index]
    with pytest.raises(ValidationError, match="confidence_threshold"):
        AppConfig.model_validate(data)


def test_confidence_threshold_zero_raises() -> None:
    data = _make_minimal_dict()
    data["detection"] = {**data["detection"], "confidence_threshold": 0.0}  # type: ignore[index]
    with pytest.raises(ValidationError, match="confidence_threshold"):
        AppConfig.model_validate(data)


def test_imgsz_zero_raises() -> None:
    data = _make_minimal_dict()
    data["detection"] = {**data["detection"], "imgsz": 0}  # type: ignore[index]
    with pytest.raises(ValidationError, match="imgsz"):
        AppConfig.model_validate(data)


def test_history_length_zero_raises() -> None:
    data = _make_minimal_dict()
    data["crossing_logic"] = {**data["crossing_logic"], "history_length": 0}  # type: ignore[index]
    with pytest.raises(ValidationError, match="history_length"):
        AppConfig.model_validate(data)


def test_majority_threshold_at_half_raises() -> None:
    data = _make_minimal_dict()
    data["crossing_logic"] = {  # type: ignore[index]
        **data["crossing_logic"],
        "confirmation_majority_threshold": 0.5,
    }
    with pytest.raises(ValidationError, match="confirmation_majority_threshold"):
        AppConfig.model_validate(data)


def test_cooldown_negative_raises() -> None:
    data = _make_minimal_dict()
    data["crossing_logic"] = {**data["crossing_logic"], "cooldown_seconds": -1.0}  # type: ignore[index]
    with pytest.raises(ValidationError, match="cooldown_seconds"):
        AppConfig.model_validate(data)


def test_empty_lines_raises() -> None:
    data = _make_minimal_dict()
    data["lines"] = []
    with pytest.raises(ValidationError, match="lines"):
        AppConfig.model_validate(data)


def test_duplicate_line_ids_raise() -> None:
    data = _make_minimal_dict()
    data["lines"] = [
        {"line_id": "same", "point_a": [0, 0], "point_b": [100, 0], "positive_direction": "A_to_B"},
        {
            "line_id": "same",
            "point_a": [0, 200],
            "point_b": [100, 200],
            "positive_direction": "A_to_B",
        },
    ]
    with pytest.raises(ValidationError, match="[Dd]uplicate"):
        AppConfig.model_validate(data)


def test_invalid_positive_direction_raises() -> None:
    data = _make_minimal_dict()
    data["lines"] = [
        {"line_id": "l1", "point_a": [0, 0], "point_b": [100, 0], "positive_direction": "sideways"},
    ]
    with pytest.raises(ValidationError):
        AppConfig.model_validate(data)


# ---------------------------------------------------------------------------
# load_config — nonexistent path raises FileNotFoundError
# ---------------------------------------------------------------------------


def test_load_config_nonexistent_path_raises_file_not_found() -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        load_config("/nonexistent/path/config.yaml")


def test_load_config_malformed_yaml_raises_value_error(tmp_path: Path) -> None:
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(
        textwrap.dedent("""\
        video:
          path: [unclosed
    """),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="[Ff]ailed to parse"):
        load_config(str(bad_yaml))


# ---------------------------------------------------------------------------
# Optional crossing-logic fields can be None
# ---------------------------------------------------------------------------


def test_optional_crossing_fields_default_to_none() -> None:
    cfg = AppConfig.model_validate(_make_minimal_dict())
    assert cfg.crossing_logic.min_displacement_px is None
    assert cfg.crossing_logic.min_velocity_px_per_s is None
