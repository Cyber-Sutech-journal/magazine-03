"""Command-line evaluation of predicted crossing events against ground truth."""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Sequence
from pathlib import Path

from mot_counting.config import load_config
from mot_counting.evaluation import (
    EvaluationEvent,
    EvaluationResult,
    GroupKey,
    GroupMatchResult,
    GroupMetrics,
    MatchPair,
    compute_group_metrics,
    evaluate_events,
    event_from_crossing_fields,
    load_ground_truth_events,
    load_prediction_events,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
SUMMARY_FILENAME = "evaluation_summary.csv"
MATCHES_FILENAME = "evaluation_matches.csv"
UNDEFINED = "N/A"

SUMMARY_COLUMNS = (
    "scope",
    "line_id",
    "class_name",
    "direction",
    "tp",
    "fp",
    "fn",
    "predicted_count",
    "gt_count",
    "event_precision",
    "event_recall",
    "event_f1",
    "absolute_counting_error",
    "relative_counting_error",
    "percentage_counting_error",
)

MATCH_COLUMNS = (
    "status",
    "line_id",
    "class_name",
    "direction",
    "gt_frame_idx",
    "gt_timestamp_seconds",
    "prediction_frame_idx",
    "prediction_timestamp_seconds",
    "time_delta_seconds",
    "prediction_track_id",
)

__all__ = [
    "EvaluationEvent",
    "EvaluationResult",
    "GroupKey",
    "GroupMatchResult",
    "GroupMetrics",
    "MatchPair",
    "compute_group_metrics",
    "evaluate_events",
    "event_from_crossing_fields",
    "load_ground_truth_events",
    "load_prediction_events",
    "main",
    "write_matches_csv",
    "write_summary_csv",
]


def _non_negative_float(raw: str) -> float:
    value = float(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to zero")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate predicted crossing events against ground-truth events."
    )
    parser.add_argument("--predictions", required=True, type=Path, help="Prediction event CSV.")
    parser.add_argument("--ground-truth", required=True, type=Path, help="Ground-truth event CSV.")
    parser.add_argument(
        "--tolerance-seconds",
        type=_non_negative_float,
        help=(
            "Inclusive timestamp matching tolerance in seconds. Defaults to "
            "evaluation.matching_tolerance_seconds in configs/default.yaml."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Artifact directory. Defaults to the predictions CSV directory.",
    )
    return parser


def _resolve_tolerance(explicit_tolerance: float | None) -> float:
    if explicit_tolerance is not None:
        return explicit_tolerance
    return load_config(str(DEFAULT_CONFIG_PATH)).evaluation.matching_tolerance_seconds


def _sorted_group_keys(result: EvaluationResult) -> list[GroupKey]:
    return sorted(
        result.groups,
        key=lambda key: (key.line_id, key.class_name, key.direction.value),
    )


def _machine_number(value: float | None) -> str:
    if value is None:
        return UNDEFINED
    return format(value, ".12g")


def _summary_row(scope: str, metrics: GroupMetrics) -> dict[str, str | int]:
    key = metrics.key
    relative_error = metrics.relative_counting_error
    return {
        "scope": scope,
        "line_id": key.line_id if key is not None else "",
        "class_name": key.class_name if key is not None else "",
        "direction": key.direction.value if key is not None else "",
        "tp": metrics.tp,
        "fp": metrics.fp,
        "fn": metrics.fn,
        "predicted_count": metrics.predicted_count,
        "gt_count": metrics.gt_count,
        "event_precision": _machine_number(metrics.precision),
        "event_recall": _machine_number(metrics.recall),
        "event_f1": _machine_number(metrics.f1),
        "absolute_counting_error": metrics.absolute_counting_error,
        "relative_counting_error": _machine_number(relative_error),
        "percentage_counting_error": _machine_number(
            relative_error * 100 if relative_error is not None else None
        ),
    }


def write_summary_csv(result: EvaluationResult, path: str | Path) -> None:
    """Write overall and per-group metrics in deterministic order."""
    output_path = Path(path)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerow(_summary_row("overall", result.overall_metrics))
        for key in _sorted_group_keys(result):
            writer.writerow(_summary_row("group", result.per_group_metrics[key]))


def _optional(value: int | None) -> int | str:
    return "" if value is None else value


def _event_number(value: float | None) -> str:
    return "" if value is None else _machine_number(value)


def _match_row(
    status: str,
    key: GroupKey,
    *,
    prediction: EvaluationEvent | None = None,
    ground_truth: EvaluationEvent | None = None,
) -> dict[str, str | int]:
    delta = None
    if prediction is not None and ground_truth is not None:
        delta = prediction.timestamp_seconds - ground_truth.timestamp_seconds
    return {
        "status": status,
        "line_id": key.line_id,
        "class_name": key.class_name,
        "direction": key.direction.value,
        "gt_frame_idx": _optional(ground_truth.frame_idx) if ground_truth else "",
        "gt_timestamp_seconds": _event_number(ground_truth.timestamp_seconds)
        if ground_truth
        else "",
        "prediction_frame_idx": _optional(prediction.frame_idx) if prediction else "",
        "prediction_timestamp_seconds": _event_number(prediction.timestamp_seconds)
        if prediction
        else "",
        "time_delta_seconds": _event_number(delta),
        "prediction_track_id": _optional(prediction.track_id) if prediction else "",
    }


def write_matches_csv(result: EvaluationResult, path: str | Path) -> None:
    """Write one deterministic row for every TP, FP, and FN outcome."""
    output_path = Path(path)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MATCH_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for key in _sorted_group_keys(result):
            group = result.groups[key]
            for match in group.true_positives:
                writer.writerow(
                    _match_row(
                        "TP",
                        key,
                        prediction=match.prediction,
                        ground_truth=match.ground_truth,
                    )
                )
            for prediction in group.false_positives:
                writer.writerow(_match_row("FP", key, prediction=prediction))
            for ground_truth in group.false_negatives:
                writer.writerow(_match_row("FN", key, ground_truth=ground_truth))


def _display_metric(value: float | None) -> str:
    return UNDEFINED if value is None else f"{value:.4f}"


def _table_row(scope: str, group: str, metrics: GroupMetrics) -> tuple[str, ...]:
    return (
        scope,
        group,
        str(metrics.tp),
        str(metrics.fp),
        str(metrics.fn),
        _display_metric(metrics.precision),
        _display_metric(metrics.recall),
        _display_metric(metrics.f1),
        str(metrics.absolute_counting_error),
        _display_metric(metrics.relative_counting_error),
    )


def print_summary(result: EvaluationResult) -> None:
    """Print the main overall and per-group metrics as a fixed-width table."""
    headers = (
        "Scope",
        "Group",
        "TP",
        "FP",
        "FN",
        "Precision",
        "Recall",
        "F1",
        "Abs Count Error",
        "Rel Count Error",
    )
    rows = [_table_row("Overall", "All events", result.overall_metrics)]
    for key in _sorted_group_keys(result):
        group = f"{key.line_id} / {key.class_name} / {key.direction.value}"
        rows.append(_table_row("Group", group, result.per_group_metrics[key]))

    widths = [max(len(row[index]) for row in (headers, *rows)) for index in range(len(headers))]

    def format_row(row: tuple[str, ...]) -> str:
        cells = [row[0].ljust(widths[0]), row[1].ljust(widths[1])]
        cells.extend(row[index].rjust(widths[index]) for index in range(2, len(row)))
        return " | ".join(cells)

    print(f"Evaluation summary (tolerance <= {result.tolerance_seconds:g} seconds)")
    print(format_row(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(format_row(row))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the standalone evaluator."""
    args = _build_parser().parse_args(argv)
    tolerance_seconds = _resolve_tolerance(args.tolerance_seconds)
    try:
        predictions = load_prediction_events(args.predictions)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error loading prediction CSV: {exc}", file=sys.stderr)
        return 1

    try:
        ground_truths = load_ground_truth_events(args.ground_truth)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error loading ground-truth CSV: {exc}", file=sys.stderr)
        return 1

    result = evaluate_events(predictions, ground_truths, tolerance_seconds)

    output_dir = args.output_dir if args.output_dir is not None else args.predictions.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / SUMMARY_FILENAME
    matches_path = output_dir / MATCHES_FILENAME
    write_summary_csv(result, summary_path)
    write_matches_csv(result, matches_path)

    print_summary(result)
    print(f"Summary CSV: {summary_path}")
    print(f"Matches CSV: {matches_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
