"""Thin CLI entry point for the MOT counting pipeline (§4.6).

Usage::

    python scripts/run_pipeline.py --config configs/default.yaml
    python scripts/run_pipeline.py --config configs/ci.yaml --video data/clip.mp4

``--video`` optionally overrides the ``video.path`` value in the YAML config,
allowing quick one-off runs against a different clip without editing the file.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the project source tree is importable when the script is run directly
# (i.e. without ``pip install -e .``).
_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_pipeline",
        description="Run the MOT counting pipeline against a video file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to the YAML configuration file (e.g. configs/ci.yaml).",
    )
    parser.add_argument(
        "--video",
        default=None,
        metavar="PATH",
        help=(
            "Optional override for video.path in the config.  "
            "Use this for quick manual runs without editing YAML."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: build the pipeline, optionally patch video path, then run."""
    args = _parse_args()

    # Patch the YAML config when --video is supplied so the override is applied
    # before the composition root validates geometry and opens the file.
    config_path = args.config
    if args.video is not None:
        import tempfile

        import yaml

        raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        raw["video"]["path"] = args.video
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as tmp:
            yaml.dump(raw, tmp)
            config_path = tmp.name

    from mot_counting.composition_root import build_pipeline

    controller = build_pipeline(config_path)
    controller.run()

    stats = controller.stats
    logging.getLogger(__name__).info(
        "Run complete — frames=%d elapsed=%.2fs avg_fps=%.2f counters=%r",
        stats.frames_processed,
        stats.elapsed_seconds,
        stats.average_fps,
        stats.final_counters,
    )
    print("\n=== Final Counters ===")
    for key, count in sorted(stats.final_counters.items(), key=lambda kv: str(kv[0])):
        class_name, line_id, direction = key
        direction_label = direction.value if hasattr(direction, "value") else direction
        print(f"  {class_name:10s}  line={line_id}  {direction_label}: {count}")
    print("=====================\n")


if __name__ == "__main__":
    main()
