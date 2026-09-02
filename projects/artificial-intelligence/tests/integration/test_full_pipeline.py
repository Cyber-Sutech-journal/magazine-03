"""End-to-end integration tests for the full pipeline (T19, §12.3).

These tests run against ``configs/ci.yaml`` and ``data/ci_sample_clip.mp4``
(a synthetic 2-second, 1280×720 @ 30 fps clip committed to the repo).

The purpose is to validate correct wiring of all independently-built modules
(composition root, controller loop, CSV schema, annotated-video output) rather
than detection quality.  ``yolo26n`` is used per ``configs/ci.yaml`` to keep CI
runtime short.

Running
-------
    pytest tests/integration/test_full_pipeline.py -v

Requirements
------------
- ``data/ci_sample_clip.mp4`` must exist (committed synthetic clip).
- YOLO26n weights are downloaded on first run and cached by Ultralytics.
- A writable ``outputs/`` directory (created automatically).
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
CI_CONFIG = PROJECT_ROOT / "configs" / "ci.yaml"
CI_CLIP = PROJECT_ROOT / "data" / "ci_sample_clip.mp4"
OUTPUT_VIDEO = PROJECT_ROOT / "outputs" / "annotated.mp4"
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "events.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_and_run_pipeline(config_path: str) -> object:
    """Build the pipeline and run it, returning the controller for assertions."""
    from mot_counting.composition_root import build_pipeline

    controller = build_pipeline(config_path)
    controller.run()
    return controller


# ---------------------------------------------------------------------------
# Full end-to-end run
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_full_pipeline_completes_without_raising() -> None:
    """Full pipeline run against ci.yaml must not raise any exception."""
    assert CI_CLIP.exists(), f"CI sample clip missing: {CI_CLIP}"
    _build_and_run_pipeline(str(CI_CONFIG))


@pytest.mark.integration
def test_full_pipeline_produces_annotated_video() -> None:
    """After a full run, outputs/annotated.mp4 must exist and be non-empty."""
    assert CI_CLIP.exists(), f"CI sample clip missing: {CI_CLIP}"

    # Ensure clean state
    OUTPUT_VIDEO.unlink(missing_ok=True)

    _build_and_run_pipeline(str(CI_CONFIG))

    assert OUTPUT_VIDEO.exists(), f"Annotated video not produced: {OUTPUT_VIDEO}"
    assert OUTPUT_VIDEO.stat().st_size > 0, "Annotated video is empty (0 bytes)."


@pytest.mark.integration
def test_full_pipeline_produces_events_csv_with_correct_header() -> None:
    """After a full run, outputs/events.csv must exist with the correct header row."""
    assert CI_CLIP.exists(), f"CI sample clip missing: {CI_CLIP}"

    OUTPUT_CSV.unlink(missing_ok=True)

    _build_and_run_pipeline(str(CI_CONFIG))

    assert OUTPUT_CSV.exists(), f"Events CSV not produced: {OUTPUT_CSV}"
    assert OUTPUT_CSV.stat().st_size > 0, "Events CSV is empty (0 bytes)."

    with OUTPUT_CSV.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames is not None, "Events CSV has no header row."
        expected_fields = {
            "frame_idx",
            "timestamp_seconds",
            "track_id",
            "class_id",
            "class_name",
            "direction",
            "line_id",
        }
        assert expected_fields.issubset(set(reader.fieldnames)), (
            f"CSV is missing expected columns. Got: {reader.fieldnames}"
        )


@pytest.mark.integration
def test_full_pipeline_counters_are_non_negative() -> None:
    """Final counters from get_counters() must all be non-negative integers."""
    assert CI_CLIP.exists(), f"CI sample clip missing: {CI_CLIP}"

    controller = _build_and_run_pipeline(str(CI_CONFIG))
    counters = controller.stats.final_counters

    for key, count in counters.items():
        assert isinstance(count, int), f"Counter {key!r} value is not int: {count!r}"
        assert count >= 0, f"Counter {key!r} is negative: {count}"


@pytest.mark.integration
def test_full_pipeline_stats_frames_processed_positive() -> None:
    """A 2-second clip at 30 fps should produce at least 1 processed frame."""
    assert CI_CLIP.exists(), f"CI sample clip missing: {CI_CLIP}"

    controller = _build_and_run_pipeline(str(CI_CONFIG))
    assert controller.stats.frames_processed > 0, "No frames were processed."


# ---------------------------------------------------------------------------
# Fail-fast: invalid class name
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_invalid_class_name_fails_fast(tmp_path: Path) -> None:
    """A config with an unrecognised class name must fail at startup with a
    clear error naming the offending class and listing valid names."""
    import yaml

    with CI_CONFIG.open() as f:
        raw = yaml.safe_load(f)
    raw["detection"]["classes"] = ["perso"]  # deliberate typo

    bad_config = tmp_path / "bad_classes.yaml"
    bad_config.write_text(yaml.dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="perso"):
        _build_and_run_pipeline(str(bad_config))


# ---------------------------------------------------------------------------
# Fail-fast: line geometry outside frame
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_line_outside_frame_fails_fast(tmp_path: Path) -> None:
    """A config with a line point far outside the real video frame dimensions
    must fail fast with a clear error naming the offending line."""
    import yaml

    from mot_counting.controllers.pipeline_controller import LineGeometryError

    with CI_CONFIG.open() as f:
        raw = yaml.safe_load(f)
    # 1280×720 video — set point far outside
    raw["lines"][0]["point_a"] = [9999, 9999]

    bad_config = tmp_path / "bad_geometry.yaml"
    bad_config.write_text(yaml.dump(raw), encoding="utf-8")

    with pytest.raises((LineGeometryError, ValueError)):
        _build_and_run_pipeline(str(bad_config))
