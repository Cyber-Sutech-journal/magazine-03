"""T18 evaluation core — event matching and metric calculation (Stage 1).

Compares predicted crossing events against Ground Truth using deterministic
one-to-one matching within each ``(line_id, class_name, direction)`` group.

Assignment is computed by dynamic programming over sequences sorted by
``timestamp_seconds`` (then ``source_index``).  Because eligibility is a
temporal-window constraint on a line, an optimal matching is non-crossing
after that sort, so the DP is exact:

1. maximise the number of eligible matches;
2. among those assignments, minimise the total absolute timestamp error.

Further ties (equal cardinality and equal total error) are broken by the DP
recurrence order: skip-prediction, then skip-ground-truth, then match.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from mot_counting.types import Direction

_CSV_REQUIRED_COLUMNS = ("timestamp_seconds", "class_name", "direction", "line_id")


@dataclass(frozen=True)
class EvaluationEvent:
    """Minimal crossing-event view used by the evaluation matcher.

    Attributes:
        timestamp_seconds: Event time in seconds from video start.
        class_name: Object class label.
        direction: Crossing direction.
        line_id: Counting line identifier.
        source_index: Zero-based index in the originating input list.
        frame_idx: Optional frame index (present in CSV logs).
        track_id: Optional track id (predictions only).
        class_id: Optional detector class id.
    """

    timestamp_seconds: float
    class_name: str
    direction: Direction
    line_id: str
    source_index: int
    frame_idx: int | None = None
    track_id: int | None = None
    class_id: int | None = None


@dataclass(frozen=True)
class GroupKey:
    """Grouping key for per-stratum evaluation."""

    line_id: str
    class_name: str
    direction: Direction


@dataclass(frozen=True)
class MatchPair:
    """A true-positive match between one prediction and one ground-truth event."""

    prediction: EvaluationEvent
    ground_truth: EvaluationEvent
    temporal_error: float


@dataclass
class GroupMatchResult:
    """Classification outcome for a single ``(line_id, class_name, direction)`` group."""

    key: GroupKey
    true_positives: list[MatchPair]
    false_positives: list[EvaluationEvent]
    false_negatives: list[EvaluationEvent]


@dataclass
class GroupMetrics:
    """Event-level and counting metrics for one evaluation stratum.

    Precision, recall, and F1 are ``None`` when the corresponding denominator
    is zero (no predictions for precision; no ground-truth events for recall;
    either undefined for F1).  Relative counting error is ``None`` when
    ``gt_count == 0`` so Stage 2 can render it as ``N/A``.
    """

    key: GroupKey | None
    tp: int
    fp: int
    fn: int
    precision: float | None
    recall: float | None
    f1: float | None
    predicted_count: int
    gt_count: int
    absolute_counting_error: int
    relative_counting_error: float | None


@dataclass
class EvaluationResult:
    """Full evaluation output retained for Stage 2 artefact generation."""

    tolerance_seconds: float
    groups: dict[GroupKey, GroupMatchResult]
    per_group_metrics: dict[GroupKey, GroupMetrics]
    overall_metrics: GroupMetrics


def _parse_direction(raw: str) -> Direction:
    try:
        return Direction(raw.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid direction {raw!r}; expected 'IN' or 'OUT'.") from exc


def _row_to_event(row: dict[str, str], source_index: int) -> EvaluationEvent:
    missing = [col for col in _CSV_REQUIRED_COLUMNS if not row.get(col, "").strip()]
    if missing:
        cols = ", ".join(missing)
        raise ValueError(f"Row {source_index + 1} is missing required column(s): {cols}.")

    frame_idx_raw = row.get("frame_idx", "").strip()
    track_id_raw = row.get("track_id", "").strip()
    class_id_raw = row.get("class_id", "").strip()

    return EvaluationEvent(
        timestamp_seconds=float(row["timestamp_seconds"]),
        class_name=row["class_name"].strip(),
        direction=_parse_direction(row["direction"]),
        line_id=row["line_id"].strip(),
        source_index=source_index,
        frame_idx=int(frame_idx_raw) if frame_idx_raw else None,
        track_id=int(track_id_raw) if track_id_raw else None,
        class_id=int(class_id_raw) if class_id_raw else None,
    )


def _load_events(path: str | Path) -> list[EvaluationEvent]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Event CSV not found: {csv_path.resolve()!s}")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Event CSV has no header row: {csv_path!s}")
        return [_row_to_event(row, index) for index, row in enumerate(reader)]


def load_prediction_events(path: str | Path) -> list[EvaluationEvent]:
    """Load prediction crossing events from a CSV file."""
    return _load_events(path)


def load_ground_truth_events(path: str | Path) -> list[EvaluationEvent]:
    """Load ground-truth crossing events from a CSV file."""
    return _load_events(path)


def event_from_crossing_fields(
    *,
    timestamp_seconds: float,
    class_name: str,
    direction: Direction | str,
    line_id: str,
    source_index: int,
    frame_idx: int | None = None,
    track_id: int | None = None,
    class_id: int | None = None,
) -> EvaluationEvent:
    """Build an :class:`EvaluationEvent` for unit tests and programmatic callers."""
    if isinstance(direction, str):
        direction = _parse_direction(direction)
    return EvaluationEvent(
        timestamp_seconds=timestamp_seconds,
        class_name=class_name,
        direction=direction,
        line_id=line_id,
        source_index=source_index,
        frame_idx=frame_idx,
        track_id=track_id,
        class_id=class_id,
    )


def _group_key(event: EvaluationEvent) -> GroupKey:
    return GroupKey(line_id=event.line_id, class_name=event.class_name, direction=event.direction)


def _eligible(pred: EvaluationEvent, gt: EvaluationEvent, tolerance_seconds: float) -> bool:
    return (
        pred.class_name == gt.class_name
        and pred.direction == gt.direction
        and pred.line_id == gt.line_id
        and abs(pred.timestamp_seconds - gt.timestamp_seconds) <= tolerance_seconds
    )


def _temporal_error(pred: EvaluationEvent, gt: EvaluationEvent) -> float:
    return abs(pred.timestamp_seconds - gt.timestamp_seconds)


def _score_better(left: tuple[int, float], right: tuple[int, float]) -> bool:
    """Return True if ``left`` is lexicographically better than ``right``.

    Scores are ``(cardinality, -total_absolute_error)``.
    """
    return left > right


def _match_one_group(
    predictions: list[EvaluationEvent],
    ground_truths: list[EvaluationEvent],
    tolerance_seconds: float,
) -> list[MatchPair]:
    """Return TP matches for one ``(line_id, class_name, direction)`` group."""
    if not predictions or not ground_truths:
        return []

    preds = sorted(predictions, key=lambda event: (event.timestamp_seconds, event.source_index))
    gts = sorted(ground_truths, key=lambda event: (event.timestamp_seconds, event.source_index))
    n_pred = len(preds)
    n_gt = len(gts)

    # dp[i][j] = (cardinality, -total_error) using the first i preds and j GTs.
    dp: list[list[tuple[int, float]]] = [[(0, 0.0)] * (n_gt + 1) for _ in range(n_pred + 1)]
    # choice: 0 = skip prediction, 1 = skip GT, 2 = match.
    choice: list[list[int]] = [[0] * (n_gt + 1) for _ in range(n_pred + 1)]

    for i in range(1, n_pred + 1):
        for j in range(1, n_gt + 1):
            best = dp[i - 1][j]
            selected = 0
            if _score_better(dp[i][j - 1], best):
                best = dp[i][j - 1]
                selected = 1
            if _eligible(preds[i - 1], gts[j - 1], tolerance_seconds):
                prev_card, prev_neg_error = dp[i - 1][j - 1]
                candidate = (
                    prev_card + 1,
                    prev_neg_error - _temporal_error(preds[i - 1], gts[j - 1]),
                )
                if _score_better(candidate, best):
                    best = candidate
                    selected = 2
            dp[i][j] = best
            choice[i][j] = selected

    matches: list[MatchPair] = []
    i = n_pred
    j = n_gt
    while i > 0 and j > 0:
        selected = choice[i][j]
        if selected == 2:
            pred = preds[i - 1]
            gt = gts[j - 1]
            matches.append(
                MatchPair(
                    prediction=pred,
                    ground_truth=gt,
                    temporal_error=_temporal_error(pred, gt),
                )
            )
            i -= 1
            j -= 1
        elif selected == 1:
            j -= 1
        else:
            i -= 1

    matches.sort(
        key=lambda match: (
            match.prediction.timestamp_seconds,
            match.prediction.source_index,
            match.ground_truth.timestamp_seconds,
            match.ground_truth.source_index,
        )
    )
    return matches


def _classify_group(
    key: GroupKey,
    predictions: list[EvaluationEvent],
    ground_truths: list[EvaluationEvent],
    tolerance_seconds: float,
) -> GroupMatchResult:
    true_positives = _match_one_group(predictions, ground_truths, tolerance_seconds)
    matched_pred_indices = {match.prediction.source_index for match in true_positives}
    matched_gt_indices = {match.ground_truth.source_index for match in true_positives}

    false_positives = [
        event for event in predictions if event.source_index not in matched_pred_indices
    ]
    false_negatives = [
        event for event in ground_truths if event.source_index not in matched_gt_indices
    ]
    false_positives.sort(key=lambda event: (event.timestamp_seconds, event.source_index))
    false_negatives.sort(key=lambda event: (event.timestamp_seconds, event.source_index))

    return GroupMatchResult(
        key=key,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _f1_score(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_group_metrics(
    key: GroupKey | None,
    tp: int,
    fp: int,
    fn: int,
    predicted_count: int,
    gt_count: int,
) -> GroupMetrics:
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    relative_counting_error = abs(predicted_count - gt_count) / gt_count if gt_count > 0 else None
    return GroupMetrics(
        key=key,
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=_f1_score(precision, recall),
        predicted_count=predicted_count,
        gt_count=gt_count,
        absolute_counting_error=abs(predicted_count - gt_count),
        relative_counting_error=relative_counting_error,
    )


def evaluate_events(
    predictions: list[EvaluationEvent],
    ground_truths: list[EvaluationEvent],
    tolerance_seconds: float,
) -> EvaluationResult:
    """Match events and compute overall and per-group metrics."""
    if tolerance_seconds < 0:
        raise ValueError(f"tolerance_seconds must be >= 0, got {tolerance_seconds!r}.")

    grouped_predictions: dict[GroupKey, list[EvaluationEvent]] = defaultdict(list)
    grouped_ground_truths: dict[GroupKey, list[EvaluationEvent]] = defaultdict(list)

    for event in predictions:
        grouped_predictions[_group_key(event)].append(event)
    for event in ground_truths:
        grouped_ground_truths[_group_key(event)].append(event)

    all_keys = sorted(
        set(grouped_predictions) | set(grouped_ground_truths),
        key=lambda key: (key.line_id, key.class_name, key.direction.value),
    )

    groups: dict[GroupKey, GroupMatchResult] = {}
    per_group_metrics: dict[GroupKey, GroupMetrics] = {}
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_predicted = 0
    total_gt = 0

    for key in all_keys:
        preds = grouped_predictions.get(key, [])
        gts = grouped_ground_truths.get(key, [])
        group_result = _classify_group(key, preds, gts, tolerance_seconds)
        groups[key] = group_result

        tp = len(group_result.true_positives)
        fp = len(group_result.false_positives)
        fn = len(group_result.false_negatives)
        metrics = compute_group_metrics(key, tp, fp, fn, len(preds), len(gts))
        per_group_metrics[key] = metrics

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_predicted += len(preds)
        total_gt += len(gts)

    overall_metrics = compute_group_metrics(
        None,
        total_tp,
        total_fp,
        total_fn,
        total_predicted,
        total_gt,
    )
    return EvaluationResult(
        tolerance_seconds=tolerance_seconds,
        groups=groups,
        per_group_metrics=per_group_metrics,
        overall_metrics=overall_metrics,
    )
