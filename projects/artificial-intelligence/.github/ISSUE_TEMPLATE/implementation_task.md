---
name: Implementation Task
about: Standard ticket for assigning a module implementation to a developer
title: "[TASK] "
labels: implementation
assignees: ''
---

## Context

<!-- Where this component fits in the overall architecture and why it is needed.
     Reference the relevant section(s) of SPEC.md (e.g., "§10.3 Detector").
     Explain which other components depend on this one and how they interact. -->

## Interface to implement

<!-- The exact `abc.ABC` interface (from §6 of SPEC.md / src/mot_counting/interfaces/)
     that this task must satisfy.  Paste the full abstract method signatures here so
     the implementer has everything in one place without needing to navigate the codebase. -->

```python
# Example — replace with the real interface for this task
from mot_counting.interfaces.detector import IDetector

class IDetector(ABC):
    @abstractmethod
    def predict(self, frame: np.ndarray) -> list[Detection]: ...
```

## Acceptance criteria

<!-- Concrete, checkable conditions that define "done".
     Each item must be independently verifiable by the reviewer.
     Examples:
     - [ ] `predict()` returns a `list[Detection]` with correct `xyxy`, `confidence`, `class_id`, `class_name` fields.
     - [ ] Only detections whose `class_name` appears in the configured `detection.classes` list are returned.
     - [ ] Confidence below `detection.confidence_threshold` are filtered out before returning.
     - [ ] The underlying model object is never loaded inside `predict()` — it is received at construction time. -->

- [ ]
- [ ]
- [ ]

## Test cases

<!-- The specific unit and/or integration test scenarios the implementer must cover
     before requesting review.  Reference the test layout in §12.3.
     Examples:
     - Unit: given a mock model that always returns one bounding box, `predict()` returns exactly one `Detection`.
     - Unit: detections with confidence < threshold are excluded.
     - Unit: class names not in the configured list are excluded.
     - Integration: full pipeline run against `data/ci_sample_clip.mp4` using `configs/ci.yaml` completes without error. -->

- [ ]
- [ ]
- [ ]
