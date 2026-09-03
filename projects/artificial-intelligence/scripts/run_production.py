"""Production multi-clip pipeline runner for T23.

Runs the full MOT-counting pipeline against every evaluation clip and
produces a machine-readable manifest recording artifacts, counters, and
execution context.  Outputs for each clip are isolated under a per-clip
sub-directory so nothing can overwrite another clip's results.

Usage (native)::

    python scripts/run_production.py \\
        --config configs/production.yaml \\
        --clips data/clip_a.mp4 data/clip_b.mp4 \\
        --output-root outputs/production

Usage (Docker CPU)::

    docker run --rm \\
        -v "$(pwd)/configs:/app/configs:ro" \\
        -v "$(pwd)/data:/app/data:ro" \\
        -v "$(pwd)/outputs:/app/outputs" \\
        mot-counting:cpu \\
        python scripts/run_production.py \\
            --config configs/production.yaml \\
            --clips data/clip_a.mp4 \\
            --output-root outputs/production
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
)
_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_production",
        description=(
            "Run the MOT counting pipeline on one or more evaluation clips "
            "and produce a JSON manifest of all artifacts and metrics."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Base YAML configuration file (production.yaml or default.yaml).",
    )
    parser.add_argument(
        "--clips",
        nargs="+",
        required=True,
        metavar="PATH",
        help="One or more video clip paths to process.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/production",
        metavar="DIR",
        help="Root directory for all per-clip output sub-directories.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Execution context
# ---------------------------------------------------------------------------


def _collect_context() -> dict[str, object]:
    """Record hardware / runtime context for the manifest."""
    import importlib

    context: dict[str, object] = {
        "run_timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "machine": platform.machine(),
    }

    # Torch device info
    try:
        torch = importlib.import_module("torch")
        if torch.cuda.is_available():
            context["execution_mode"] = "gpu"
            context["cuda_device_name"] = torch.cuda.get_device_name(0)
            context["cuda_version"] = torch.version.cuda  # type: ignore[attr-defined]
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            context["execution_mode"] = "mps"
        else:
            context["execution_mode"] = "cpu"
        context["torch_version"] = torch.__version__
    except Exception:  # noqa: BLE001
        context["execution_mode"] = "unknown"

    return context


# ---------------------------------------------------------------------------
# Per-clip runner
# ---------------------------------------------------------------------------


def _run_clip(
    base_config_path: str,
    clip_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Process one clip; return a dict of per-clip result metadata."""
    from mot_counting.composition_root import build_pipeline

    output_dir.mkdir(parents=True, exist_ok=True)
    clip_stem = clip_path.stem

    csv_out = output_dir / f"{clip_stem}_events.csv"
    video_out = output_dir / f"{clip_stem}_annotated.mp4"

    # Patch the base config for this clip's paths
    raw = yaml.safe_load(Path(base_config_path).read_text(encoding="utf-8"))
    raw["video"]["path"] = str(clip_path)
    raw["video"]["output_dir"] = str(output_dir)
    raw["events"]["output_csv"] = str(csv_out)
    raw["visualization"]["output_video"] = str(video_out)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as tmp:
        yaml.dump(raw, tmp)
        patched_config = tmp.name

    _log.info("Processing clip: %s → output dir: %s", clip_path, output_dir)

    t0 = time.perf_counter()
    controller = build_pipeline(patched_config)
    controller.run()
    elapsed = time.perf_counter() - t0

    stats = controller.stats
    _log.info(
        "  frames=%d  elapsed=%.2fs  avg_fps=%.2f",
        stats.frames_processed,
        stats.elapsed_seconds,
        stats.average_fps,
    )

    # Counters: convert tuple keys to strings for JSON serialisation
    counters_serialisable: dict[str, int] = {}
    for key, count in stats.final_counters.items():
        class_name, line_id, direction = key
        direction_str = direction.value if hasattr(direction, "value") else str(direction)
        counters_serialisable[f"{class_name}|{line_id}|{direction_str}"] = count

    print(f"\n=== {clip_stem} — Final Counters ===")
    for k, v in sorted(counters_serialisable.items()):
        print(f"  {k}: {v}")
    print("=" * 40)

    return {
        "clip": str(clip_path),
        "clip_stem": clip_stem,
        "prediction_csv": str(csv_out),
        "annotated_video": str(video_out),
        "csv_exists": csv_out.exists(),
        "video_exists": video_out.exists(),
        "csv_bytes": csv_out.stat().st_size if csv_out.exists() else 0,
        "video_bytes": video_out.stat().st_size if video_out.exists() else 0,
        "frames_processed": stats.frames_processed,
        "elapsed_seconds": round(stats.elapsed_seconds, 3),
        "average_fps": round(stats.average_fps, 2),
        "final_counters": counters_serialisable,
        "total_wall_seconds": round(elapsed, 3),
    }


# ---------------------------------------------------------------------------
# T18 compatibility check
# ---------------------------------------------------------------------------


def _verify_t18_compatibility(csv_path: Path) -> dict[str, object]:
    """Load the CSV via evaluation.py and confirm it parses without errors."""
    from mot_counting.evaluation import load_prediction_events

    result: dict[str, object] = {
        "csv": str(csv_path),
        "compatible": False,
        "events_loaded": 0,
        "error": None,
    }
    try:
        events = load_prediction_events(csv_path)
        result["compatible"] = True
        result["events_loaded"] = len(events)
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: C901
    args = _parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    context = _collect_context()
    _log.info(
        "Execution context: mode=%s  platform=%s",
        context.get("execution_mode"),
        context.get("platform"),
    )

    # Read detector variant from base config for the manifest
    base_raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    detector_variant = base_raw.get("detection", {}).get("model_variant", "unknown")
    context["detector_model_variant"] = detector_variant

    clip_results: list[dict[str, object]] = []
    t18_checks: list[dict[str, object]] = []

    for clip_str in args.clips:
        clip_path = Path(clip_str)
        if not clip_path.exists():
            _log.error("Clip not found, skipping: %s", clip_path)
            clip_results.append({"clip": str(clip_path), "error": "file not found"})
            continue

        clip_stem = clip_path.stem
        out_dir = output_root / clip_stem

        try:
            result = _run_clip(args.config, clip_path, out_dir)
            clip_results.append(result)
        except Exception as exc:  # noqa: BLE001
            _log.exception("Pipeline failed for clip %s: %s", clip_path, exc)
            clip_results.append({"clip": str(clip_path), "error": str(exc)})
            continue

        # T18 compatibility check
        csv_path = Path(str(result["prediction_csv"]))
        if csv_path.exists():
            t18_result = _verify_t18_compatibility(csv_path)
            t18_checks.append(t18_result)
            if t18_result["compatible"]:
                _log.info("  T18 compatibility: ✓  (%d events parsed)", t18_result["events_loaded"])
            else:
                _log.error("  T18 compatibility: ✗  error=%s", t18_result["error"])

    # -----------------------------------------------------------------------
    # Write manifest
    # -----------------------------------------------------------------------
    manifest = {
        "t23_production_run": True,
        "execution_context": context,
        "clips": clip_results,
        "t18_compatibility": t18_checks,
        "summary": {
            "total_clips": len(args.clips),
            "successful_clips": sum(1 for r in clip_results if "error" not in r),
            "t18_all_compatible": all(c["compatible"] for c in t18_checks) if t18_checks else False,
        },
    }

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest_path = output_root / f"production_manifest_{ts}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _log.info("Manifest written to: %s", manifest_path)

    # -----------------------------------------------------------------------
    # Human-readable summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("T23 PRODUCTION RUN SUMMARY")
    print("=" * 60)
    print(f"  Detector model  : {detector_variant}")
    print(f"  Execution mode  : {context.get('execution_mode', 'unknown')}")
    print(f"  Platform        : {context.get('platform', 'unknown')}")
    print(
        f"  Clips processed : {manifest['summary']['successful_clips']} / {manifest['summary']['total_clips']}"
    )
    print(f"  T18 compatible  : {manifest['summary']['t18_all_compatible']}")
    print(f"  Manifest        : {manifest_path}")
    print("=" * 60)

    # Fail fast if any clip failed
    if manifest["summary"]["successful_clips"] < manifest["summary"]["total_clips"]:
        sys.exit(1)

    # Fail fast if T18 incompatible
    if t18_checks and not manifest["summary"]["t18_all_compatible"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
