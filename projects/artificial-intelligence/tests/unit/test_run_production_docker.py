"""Regression: Docker argv composition for scripts/run_production.py."""

from __future__ import annotations

import sys

import pytest

from scripts.run_pipeline import _parse_args as parse_pipeline_args
from scripts.run_production import (
    DOCUMENTED_ENTRYPOINT_OVERRIDE,
    DOCUMENTED_PRODUCTION_COMMAND,
    IMAGE_PIPELINE_ENTRYPOINT,
    compose_container_argv,
)
from scripts.run_production import (
    _parse_args as parse_production_args,
)


def test_appending_production_tokens_to_image_entrypoint_is_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default Docker append produces run_pipeline.py python scripts/run_production.py …"""
    composed = compose_container_argv(
        ["python", "scripts/run_production.py", "--config", "configs/production.yaml"],
    )
    assert composed[:2] == list(IMAGE_PIPELINE_ENTRYPOINT)
    assert composed[2] == "python"

    monkeypatch.setattr(sys, "argv", composed[1:])
    with pytest.raises(SystemExit) as exc_info:
        parse_pipeline_args()
    assert exc_info.value.code == 2


def test_documented_command_invokes_production_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--entrypoint python`` + production script matches the image exec model."""
    composed = compose_container_argv(
        DOCUMENTED_PRODUCTION_COMMAND,
        entrypoint_override=DOCUMENTED_ENTRYPOINT_OVERRIDE,
    )
    assert composed[:2] == ["python", "scripts/run_production.py"]
    assert composed[1:] == list(DOCUMENTED_PRODUCTION_COMMAND)

    monkeypatch.setattr(sys, "argv", composed[1:])
    args = parse_production_args()
    assert args.config == "configs/production.yaml"
    assert args.clips == ["data/clip_a.mp4"]
    assert args.output_root == "outputs/production"
