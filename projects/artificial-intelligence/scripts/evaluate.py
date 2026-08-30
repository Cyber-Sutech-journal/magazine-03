"""T18 evaluation entry point.

Stage 1 exposes the importable evaluation core from :mod:`mot_counting.evaluation`.
Stage 2 will add argparse CLI, config fallback, and CSV artefact writers here.
"""

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
]
