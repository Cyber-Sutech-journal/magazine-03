# Evaluation Protocol

An evaluation event is one logged crossing of a configured counting line. Predictions and ground
truth are CSV files. Both require `timestamp_seconds`, `class_name`, `direction`, and `line_id`;
`frame_idx` is used in analysis output when present. Prediction rows may additionally provide
`track_id` and other detector, tracker, bounding-box, confidence, or video metadata. The evaluator
does not require those prediction-only fields from ground truth.

When CSV inputs are loaded, surrounding whitespace is removed, `class_name` is normalized to
lowercase to match the repository's detector-label convention, and `direction` is normalized to
uppercase before conversion to the `Direction` enum. Thus capitalization differences such as
`Person` versus `person`, or `in` versus `IN`, do not create separate evaluation groups.

## Matching

A prediction is eligible to match a ground-truth event only when `class_name`, `direction`, and
`line_id` are identical and:

```text
abs(prediction.timestamp_seconds - ground_truth.timestamp_seconds) <= tolerance_seconds
```

The comparison uses `timestamp_seconds` directly; the tolerance is not converted to frames. The
boundary is inclusive, and matching is one-to-one. Matching is performed independently within each
`(line_id, class_name, direction)` group. Within a group, predictions and ground-truth events are
deterministically sorted by timestamp and then by `source_index` (their original input order). An
exact dynamic-programming assignment maximizes match cardinality first and minimizes total absolute
temporal error second. If both objective values are exactly equal, the existing Stage 1 recurrence
precedence is used: skip prediction, then skip ground truth, then match. Identical ordered inputs
therefore produce identical results and artifacts.

Each matched pair is a true positive (TP). An unmatched prediction is a false positive (FP), and an
unmatched ground-truth event is a false negative (FN). Consequently, duplicate eligible predictions
cannot match the same ground-truth event: one may be a TP and the remaining duplicates are FP.

## Metrics and aggregation

The evaluator reports:

- Event Precision = `TP / (TP + FP)`
- Event Recall = `TP / (TP + FN)`
- Event F1 = harmonic mean of defined Precision and Recall
- Absolute Counting Error = `abs(predicted_count - gt_count)`
- Relative Counting Error = `Absolute Counting Error / gt_count`
- Percentage Counting Error = `Relative Counting Error * 100`

Metrics are calculated once over all events and separately for every
`(line_id, class_name, direction)` group appearing in either input. With zero predictions,
Precision is undefined; with zero GT events, Recall and relative/percentage counting error are
undefined. F1 is undefined if Precision or Recall is undefined. No division by zero is performed.
When Precision and Recall are both defined and equal to `0.0`, F1 is reported as `0.0`. Undefined
values are written and displayed as `N/A`.

## Outputs

`evaluation_summary.csv` contains one overall row followed by deterministic per-group rows with
event and counting metrics. `evaluation_matches.csv` contains exactly one row per TP, FP, or FN for
event-level analysis. Missing counterpart fields in FP and FN rows are blank. For TP rows,
`time_delta_seconds` is signed and defined as:

```text
prediction_timestamp_seconds - gt_timestamp_seconds
```

This signed analysis value does not change the matching objective, which minimizes absolute error.
Match rows are ordered deterministically by group, then TP, FP, and FN, with the stable ordering
provided by the matcher within each outcome.
