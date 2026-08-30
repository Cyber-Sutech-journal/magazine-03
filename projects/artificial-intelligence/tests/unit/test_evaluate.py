"""Unit tests for T18 Stage 1 evaluation matching and metrics."""

from __future__ import annotations

from pathlib import Path

import pytest

from mot_counting.evaluation import (
    EvaluationEvent,
    evaluate_events,
    event_from_crossing_fields,
    load_ground_truth_events,
    load_prediction_events,
)
from mot_counting.types import Direction

LINE = "main_line"
CLASS = "person"
TOL = 1.0


def _pred(timestamp: float, source_index: int, **kwargs: object) -> EvaluationEvent:
    defaults: dict[str, object] = {
        "class_name": CLASS,
        "direction": Direction.IN,
        "line_id": LINE,
    }
    defaults.update(kwargs)
    return event_from_crossing_fields(
        timestamp_seconds=timestamp,
        source_index=source_index,
        **defaults,  # type: ignore[arg-type]
    )


def _gt(timestamp: float, source_index: int, **kwargs: object) -> EvaluationEvent:
    return _pred(timestamp, source_index, **kwargs)


def _single_group_result(
    predictions: list[EvaluationEvent],
    ground_truths: list[EvaluationEvent],
    tolerance_seconds: float = TOL,
):
    result = evaluate_events(predictions, ground_truths, tolerance_seconds)
    assert len(result.groups) == 1
    key = next(iter(result.groups))
    return result, result.groups[key], result.overall_metrics


def test_one_gt_one_valid_prediction_is_tp() -> None:
    _, group, metrics = _single_group_result([_pred(1.0, 0)], [_gt(1.2, 0)])

    assert len(group.true_positives) == 1
    assert group.false_positives == []
    assert group.false_negatives == []
    assert metrics.tp == 1
    assert metrics.fp == 0
    assert metrics.fn == 0


def test_one_gt_two_valid_predictions_yields_one_tp_one_fp() -> None:
    _, group, metrics = _single_group_result(
        [_pred(1.0, 0), _pred(1.1, 1)],
        [_gt(1.2, 0)],
    )

    assert len(group.true_positives) == 1
    assert len(group.false_positives) == 1
    assert group.false_negatives == []
    assert metrics.tp == 1
    assert metrics.fp == 1
    assert metrics.fn == 0


def test_prediction_without_eligible_gt_is_fp() -> None:
    _, group, metrics = _single_group_result([_pred(1.0, 0)], [])

    assert group.true_positives == []
    assert len(group.false_positives) == 1
    assert group.false_negatives == []
    assert metrics.tp == 0
    assert metrics.fp == 1
    assert metrics.fn == 0


def test_gt_without_eligible_prediction_is_fn() -> None:
    _, group, metrics = _single_group_result([], [_gt(1.0, 0)])

    assert group.true_positives == []
    assert group.false_positives == []
    assert len(group.false_negatives) == 1
    assert metrics.tp == 0
    assert metrics.fp == 0
    assert metrics.fn == 1


def test_wrong_class_name_prevents_match() -> None:
    result = evaluate_events(
        [_pred(1.0, 0, class_name="car")],
        [_gt(1.0, 0, class_name="person")],
        tolerance_seconds=TOL,
    )

    assert result.overall_metrics.tp == 0
    assert result.overall_metrics.fp == 1
    assert result.overall_metrics.fn == 1
    assert all(not group.true_positives for group in result.groups.values())


def test_wrong_direction_prevents_match() -> None:
    result = evaluate_events(
        [_pred(1.0, 0, direction=Direction.OUT)],
        [_gt(1.0, 0, direction=Direction.IN)],
        tolerance_seconds=TOL,
    )

    assert result.overall_metrics.tp == 0
    assert result.overall_metrics.fp == 1
    assert result.overall_metrics.fn == 1
    assert all(not group.true_positives for group in result.groups.values())


def test_wrong_line_id_prevents_match() -> None:
    result = evaluate_events(
        [_pred(1.0, 0, line_id="other_line")],
        [_gt(1.0, 0, line_id=LINE)],
        tolerance_seconds=TOL,
    )

    assert result.overall_metrics.tp == 0
    assert result.overall_metrics.fp == 1
    assert result.overall_metrics.fn == 1
    assert all(not group.true_positives for group in result.groups.values())


def test_prediction_on_tolerance_boundary_matches() -> None:
    _, group, metrics = _single_group_result(
        [_pred(2.0, 0)],
        [_gt(3.0, 0)],
        tolerance_seconds=1.0,
    )

    assert len(group.true_positives) == 1
    assert group.true_positives[0].temporal_error == pytest.approx(1.0)
    assert metrics.tp == 1
    assert metrics.fp == 0
    assert metrics.fn == 0


def test_zero_predictions_nonzero_gt_are_all_fn() -> None:
    _, group, metrics = _single_group_result([], [_gt(1.0, 0), _gt(2.0, 1)])

    assert group.true_positives == []
    assert group.false_positives == []
    assert len(group.false_negatives) == 2
    assert metrics.tp == 0
    assert metrics.fp == 0
    assert metrics.fn == 2
    assert metrics.gt_count == 2
    assert metrics.predicted_count == 0


def test_zero_gt_nonzero_predictions_are_fp_with_undefined_relative_error() -> None:
    _, group, metrics = _single_group_result([_pred(1.0, 0), _pred(2.0, 1)], [])

    assert group.true_positives == []
    assert len(group.false_positives) == 2
    assert group.false_negatives == []
    assert metrics.tp == 0
    assert metrics.fp == 2
    assert metrics.fn == 0
    assert metrics.gt_count == 0
    assert metrics.relative_counting_error is None
    assert metrics.precision == 0.0
    assert metrics.recall is None
    assert metrics.f1 is None


def test_precision_recall_f1_on_hand_verified_counts() -> None:
    predictions = [
        _pred(1.0, 0),
        _pred(2.0, 1),
        _pred(3.0, 2),
        _pred(9.0, 3),
    ]
    ground_truths = [
        _gt(1.1, 0),
        _gt(2.1, 1),
        _gt(6.0, 2),
    ]

    result = evaluate_events(predictions, ground_truths, tolerance_seconds=1.0)
    metrics = result.overall_metrics

    assert metrics.tp == 2
    assert metrics.fp == 2
    assert metrics.fn == 1
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(2 / 3)
    assert metrics.f1 == pytest.approx(2 * 0.5 * (2 / 3) / (0.5 + 2 / 3))


def test_greedy_trap_preserves_maximum_cardinality() -> None:
    """Nearest-neighbour greedy matching can consume the only GT a later prediction can use."""
    ground_truths = [_gt(0.0, 0), _gt(1.0, 1)]
    predictions = [_pred(0.9, 0), _pred(2.0, 1)]

    _, group, metrics = _single_group_result(predictions, ground_truths, tolerance_seconds=1.0)

    assert metrics.tp == 2
    assert metrics.fp == 0
    assert metrics.fn == 0
    matched_pairs = {
        (match.prediction.source_index, match.ground_truth.source_index)
        for match in group.true_positives
    }
    assert matched_pairs == {(0, 0), (1, 1)}


def test_minimum_total_temporal_error_among_max_cardinality_assignments() -> None:
    ground_truths = [_gt(5.0, 0), _gt(6.0, 1)]
    predictions = [_pred(5.0, 0), _pred(6.5, 1)]

    _, group, _metrics = _single_group_result(predictions, ground_truths, tolerance_seconds=1.5)

    assert len(group.true_positives) == 2
    total_error = sum(match.temporal_error for match in group.true_positives)
    assert total_error == pytest.approx(0.5)
    matched_pairs = {
        (match.prediction.source_index, match.ground_truth.source_index)
        for match in group.true_positives
    }
    assert matched_pairs == {(0, 0), (1, 1)}


def test_identical_inputs_produce_identical_results() -> None:
    predictions = [
        _pred(1.0, 0),
        _pred(4.0, 1),
        _pred(7.0, 2, direction=Direction.OUT),
    ]
    ground_truths = [
        _gt(1.2, 0),
        _gt(4.5, 1),
        _gt(7.1, 2, direction=Direction.OUT),
    ]

    first = evaluate_events(predictions, ground_truths, tolerance_seconds=1.0)
    second = evaluate_events(predictions, ground_truths, tolerance_seconds=1.0)

    assert first.overall_metrics == second.overall_metrics
    assert first.groups.keys() == second.groups.keys()
    for key in first.groups:
        first_group = first.groups[key]
        second_group = second.groups[key]
        assert [
            (match.prediction.source_index, match.ground_truth.source_index)
            for match in first_group.true_positives
        ] == [
            (match.prediction.source_index, match.ground_truth.source_index)
            for match in second_group.true_positives
        ]


def test_per_group_metrics_and_overall_aggregation() -> None:
    predictions = [
        _pred(1.0, 0, class_name="person"),
        _pred(1.0, 1, class_name="car"),
    ]
    ground_truths = [
        _gt(1.0, 0, class_name="person"),
        _gt(5.0, 1, class_name="car"),
    ]

    result = evaluate_events(predictions, ground_truths, tolerance_seconds=1.0)

    person_key = next(key for key in result.per_group_metrics if key.class_name == "person")
    car_key = next(key for key in result.per_group_metrics if key.class_name == "car")

    assert result.per_group_metrics[person_key].tp == 1
    assert result.per_group_metrics[person_key].absolute_counting_error == 0
    assert result.per_group_metrics[car_key].tp == 0
    assert result.per_group_metrics[car_key].fn == 1
    assert result.per_group_metrics[car_key].absolute_counting_error == 0
    assert result.per_group_metrics[car_key].relative_counting_error == pytest.approx(0.0)

    overall = result.overall_metrics
    assert overall.tp == 1
    assert overall.fp == 1
    assert overall.fn == 1
    assert overall.absolute_counting_error == 0


def test_prediction_csv_schema_round_trip(tmp_path: Path) -> None:
    csv_path = tmp_path / "events.csv"
    csv_path.write_text(
        "frame_idx,timestamp_seconds,track_id,class_id,class_name,direction,line_id\n"
        "10,1.25,3,0,person,IN,main_line\n",
        encoding="utf-8",
    )

    predictions = load_prediction_events(csv_path)
    ground_truths = load_ground_truth_events(csv_path)

    assert predictions[0].timestamp_seconds == pytest.approx(1.25)
    assert predictions[0].track_id == 3
    assert predictions[0].class_name == "person"
    assert predictions[0].direction is Direction.IN
    assert predictions[0].line_id == "main_line"
    assert ground_truths[0].frame_idx == 10
