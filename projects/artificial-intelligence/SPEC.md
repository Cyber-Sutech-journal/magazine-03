# Multi-Object Tracking and Directional Counting System for Fixed-Camera Video Analytics

## 1. Project Overview

**Title:** Multi-Object Tracking and Directional Counting System for Fixed-Camera Video Analytics

**Objective:** Build a complete, reproducible, production-oriented video analytics pipeline that:

- Detects objects of interest (primarily `person` and `car`, extensible to any COCO class) in a recorded video from a fixed camera using a pretrained YOLO model.
- Assigns persistent Track IDs via multi-object tracking (primary tracker: ByteTrack).
- Detects crossings of one or more user-defined virtual counting lines.
- Determines direction (IN / OUT) for each crossing.
- Counts objects per class and direction while preventing duplicate counting.
- Logs every crossing event with rich metadata.
- Produces an annotated output video and structured evaluation-ready outputs.

The system is designed as a clean, modular research codebase suitable for an academic magazine article and a public GitHub repository. Code quality, documentation, configuration management, reproducibility, and evaluation rigor must meet global university / research-lab standards.

**Input**

- Single recorded video file from a fixed (non-moving) camera.
- Configuration file controlling video path, classes, confidence, counting line geometry (one or more lines), tracker parameters, etc.

**Primary Outputs**

1. Annotated video (bounding boxes + class + Track ID + counting line(s) + live IN/OUT counters).
2. Event log (CSV) containing every validated crossing.
3. Final count summary (per-class IN / OUT totals).
4. Evaluation report materials (when ground truth is available).

**Non-Goals (Explicit Scope Boundaries)**

- No training or fine-tuning of the detector.
- No online / live camera streaming in the core pipeline (recorded video only).
- No multi-camera fusion or 3D tracking.
- No guaranteed re-identification across track ID switches — this is treated as a known, documented limitation (see §7.4 and §13), not a solved problem.
- Re-identification (ReID) and advanced trajectory analytics remain optional stretch goals only.
- No hard real-time performance guarantee: the pipeline processes recorded video offline, so throughput (FPS) is reported as a metric, not enforced as a constraint.

---

## 2. Project Roles & Responsibilities

| Role | Name | Primary Responsibilities |
|---|---|---|
| **Project Lead / Core Implementer** | **Mostafa** | Core pipeline architecture and implementation, Detection–Tracking integration, Crossing Logic implementation, configuration system, module integration, code quality, technical coordination of implementation, and repository structure |
| **AI Section Lead / Evaluation & Quality Lead** | **Armila** | Project scope and technical requirements definition, evaluation framework design and implementation, event matching and metrics implementation, results and failure analysis, final system validation, technical review of major design decisions, and PR review and merge approval |
| **Developer** | **Farzad** | Implementation and testing of assigned project modules |
| **Developer** | **Amirmohammad** | Implementation and testing of assigned project modules |

**Deadline:** the project must be complete by **September 1, 2026** (~10 days from project kickoff).

---

## 3. Core Technical Stack (Locked Decisions)

| Component | Choice | Notes |
|---|---|---|
| Object Detector | **YOLO26 (Ultralytics), `yolo26m` variant** | Default model for all development and evaluation. `yolo26s` or `yolo26n` may be swapped in via config as a CPU-friendly fallback for team members without a GPU. YOLO11 is used only for a late, optional, non-blocking comparison. |
| Tracker (Primary) | ByteTrack | Must be the default and fully evaluated path. Default hyperparameters: `track_thresh=0.5`, `match_thresh=0.8`, `track_buffer=30` (standard library defaults; may be tuned later only if real test-clip results show a concrete problem). |
| Tracker (Optional) | BoT-SORT | Stretch goal only if schedule allows. |
| Video I/O & Drawing | OpenCV | |
| Configuration | **YAML + Pydantic v2** | Locked choice (not plain dataclasses). Pydantic gives automatic validation and clear error messages, which matters for junior contributors. No hard-coded paths, classes, line coordinates, or thresholds. |
| Event Storage | CSV (primary) | JSONL optional later if genuinely needed. |
| Crossing Logic | Custom implementation | Do not rely on Supervision's `LineZone` as the production path. `LineZone` may be used only as an optional baseline comparison, always scored against the team's own manual Ground Truth. |
| Language / Runtime | Python 3.10+ | Strict typing, modern packaging. |
| Packaging | `pyproject.toml` + uv or poetry | Clean, reproducible environment. All dependencies are **exactly pinned** (`==`), not lower-bounded (`>=`) — the project's reproducibility claim depends on identical dependency resolution across machines, and Docker alone does not guarantee this if `pyproject.toml` allows a floating range. |
| Containerization | Docker + Docker Compose, with **separate CPU and GPU profiles** | Mandatory for full reproducibility across machines. CPU profile exists because not all team members have a GPU. |
| License | **MIT** (project code) | Public academic repository. See §3.1 for the required third-party licensing note regarding Ultralytics YOLO26. |
| Primary Dev Tooling | Cursor (AI-assisted IDE) | Used by Mostafa to accelerate implementation given the tight deadline. |
| Linting / Formatting | **Ruff** (`ruff check` + `ruff format`) | Enforced via a pre-commit hook — see §12.4. No other formatter/linter is used. |
| Logging | Python standard library `logging` module | No third-party logging library. Chosen for zero extra dependencies and universal familiarity, which matters for junior contributors. |
| Testing | `pytest` | See §12.3 for test layout. |
| CLI | Python standard library `argparse` | No `click`/`typer` dependency. Used only to point `run_pipeline.py` / `evaluate.py` at a config file and to optionally override the video path for quick manual runs. |

### 3.1 Third-Party Licensing Note (Locked)

The project's own source code is licensed under **MIT**. The `ultralytics` package (used to load and run YOLO26) is separately licensed under **AGPL-3.0** by Ultralytics.

- Because this entire project — code, configs, training-free inference pipeline, and all outputs — is published publicly and completely as an open-source academic repository, using `ultralytics` under AGPL-3.0 terms is fully permitted at no cost. No Ultralytics Enterprise License is required for this use case.
- An Enterprise License would only become necessary if any part of the project (or a downstream product built on it) were kept closed-source or deployed commercially without publishing the corresponding source code — which is explicitly out of scope for this academic project.
- The repository's `LICENSE` file contains the MIT license text for the project's own code. A short, clearly worded note is added to `README.md` stating: the project code is MIT-licensed; it depends on `ultralytics`, which is separately licensed under AGPL-3.0 by Ultralytics; users who wish to reuse the `ultralytics` dependency itself must comply with AGPL-3.0 (or obtain their own Ultralytics Enterprise License) independently of this project's MIT license.
- No `THIRD_PARTY_LICENSES.md` or repo-wide AGPL relicensing is needed — the project's own code stays MIT, and the dependency's license is disclosed transparently in the README.

---

## 4. Software Architecture & Design Patterns (Mandatory)

The system must follow a clean, layered architecture with explicit design patterns so that junior developers can work on isolated modules with minimal risk of interference. The following patterns are required, with their implementation style locked below.

### 4.1 Factory Pattern
- `DetectorFactory` creates detector instances (YOLO26, future YOLO11, or mock detectors for testing) based on configuration.
- `TrackerFactory` creates tracker instances (ByteTrack, optional BoT-SORT, or mock trackers).
- Factories receive configuration objects and return concrete implementations that satisfy the corresponding interfaces. This allows swapping implementations without touching the rest of the pipeline.
- **Model loading boundary:** the heavy lifting of loading the actual YOLO weights happens **outside** the Factory, inside the composition root (`composition_root.py`). The composition root loads the underlying model object once and passes it into the Factory, whose sole job is to wrap that already-loaded model into a class that satisfies `IDetector`/`ITracker`. This keeps the composition root as the single place where object lifetime and expensive initialization are decided, keeps the Factory itself trivial and easy to unit-test with a fake/mock model object, and guarantees the model is loaded exactly once per pipeline run.
- **Class-list validation boundary:** immediately after the composition root loads the underlying YOLO26 model object, it validates the configured `detection.classes` list against the model's own `model.names` mapping. Any class name in the config that does not exist in `model.names` causes an immediate fail-fast startup error naming the offending class and listing the valid class names available on the loaded model (see §10.1 and §12.1). This check lives in the composition root rather than in `DetectorFactory` itself, keeping the Factory trivial and keeping all "loaded-model-dependent" validation in one place.

### 4.2 Interface / Abstraction Layer + Dependency Injection
- All major components communicate exclusively through interfaces implemented as **`abc.ABC`** abstract base classes (locked choice — not `typing.Protocol`). Explicit inheritance means a developer who forgets to implement a required method gets an immediate `TypeError` at instantiation, which is easier to teach and debug for junior contributors, and maps cleanly onto the "Interface" pattern described in the magazine article.
- The central Controller must never import or depend directly on concrete YOLO or ByteTrack classes.
- Concrete implementations are injected at construction time via **pure constructor injection**, wired together in a manual **composition root** (`composition_root.py`). No external DI framework/library is used — for an academic codebase, explicit manual wiring is more transparent and readable than a DI container.
- Minimum required interfaces:
  - `IDetector`
  - `ITracker`
  - `ICrossingLogic`
  - `IEventRepository`
  - `IVisualizer`
  - `IFrameSource` / `IVideoReader`

This guarantees that modules remain loosely coupled and that unit tests can replace any component with a mock.

### 4.3 Observer Pattern (Scope Locked)
- Implemented **explicitly**, with real `Subject` / `Observer` base classes providing `subscribe()` and `notify()` — not a simplified callback list. This is a deliberate choice because the magazine article explicitly names the Observer pattern as one of the architecture's core patterns, so the code must have a genuine, referenceable implementation of it.
- **Scope of the Observer pattern is intentionally narrow:** the core sequence `read → detect → track → update crossing state` is a hard, ordered data dependency (each stage's output is the next stage's input), and is therefore implemented as a **plain synchronous method sequence inside `PipelineController`**, not as independent observers. Modeling a strict pipeline as decoupled asynchronous observers would only reintroduce an implicit ordering dependency through a different mechanism, without any real decoupling benefit.
- The Observer pattern applies to the **side-effect consumers only**: `Logger` and `Visualizer` (and any future addition, e.g. a trajectory recorder) subscribe as `Observer`s to a `Subject` that the Controller notifies once per frame, after crossing state has been updated. These consumers have no ordering dependency on each other and genuinely benefit from being pluggable without modifying the Controller.
- This gives the codebase a real, defensible Observer implementation (for the magazine article) while keeping the core detection/tracking/crossing sequence simple, debuggable, and free of hidden execution-order bugs.

### 4.4 Controller Pattern
- A single `PipelineController` (or `VideoAnalyticsController`) owns the high-level orchestration.
- It receives all dependencies via constructor injection.
- Responsibilities:
  - Drive the main frame loop.
  - Coordinate the sequence: read → detect → track → update crossing state → notify Observers (log, visualize).
  - Manage lifecycle (start, stop, cleanup).
  - Collect and expose final counters and runtime statistics.
  - Perform the runtime (lazy) validation of counting-line geometry against actual video frame dimensions once the video is opened (see §7.3).
- The Controller contains almost no business logic; it only sequences calls to the injected components.

### 4.5 Repository Pattern
- `EventRepository` (e.g. `CsvEventRepository`) is the only component allowed to write to the CSV event log.
- The Crossing Logic module emits domain events; the Repository persists them.
- This isolates file I/O, makes it trivial to switch to JSONL or a database later, and simplifies testing (an in-memory repository can be injected).

### 4.6 Composition Root & CLI
- `composition_root.py` is the single place where concrete classes are instantiated and wired together: it loads the validated config, loads the detector/tracker model objects, validates the configured class list against the loaded model's `model.names` (§4.1), constructs the Factories, builds the concrete `IDetector`/`ITracker`/`ICrossingLogic`/`IEventRepository`/`IVisualizer`/`IFrameSource` implementations, and constructs `PipelineController` with all of them injected via its constructor.
- `scripts/run_pipeline.py` is a thin entry point: it uses `argparse` to accept `--config <path>` (required) and an optional `--video <path>` override for quick manual runs without editing YAML, then calls into `composition_root.py`.
- `scripts/evaluate.py` follows the same pattern: `argparse` for `--predictions <csv>`, `--ground-truth <csv>`, `--tolerance-seconds <float>` (optional override of the config default), and produces the metrics described in §7.8.

These patterns together produce a strongly engineered, testable, and extensible system. Junior developers receive tasks that operate almost exclusively behind a single interface, dramatically reducing the chance of cross-module bugs.

---

## 5. Core Data Types (Locked Signatures)

These types are frozen before any implementation task is handed out, so that Farzad and Amirmohammad can implement against them without needing further clarification.

```python
from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixel coords
    confidence: float
    class_id: int
    class_name: str


@dataclass(frozen=True)
class Track:
    track_id: int
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixel coords
    class_id: int
    class_name: str
    score: float


class Direction(str, Enum):
    IN = "IN"
    OUT = "OUT"


@dataclass(frozen=True)
class CrossingEvent:
    frame_idx: int
    timestamp_seconds: float
    track_id: int
    class_id: int
    class_name: str
    direction: Direction
    line_id: str
    confidence: float | None = None
    bbox: tuple[float, float, float, float] | None = None
    video_name: str | None = None
```

---

## 6. Interfaces (Locked Signatures)

```python
from abc import ABC, abstractmethod
import numpy as np


class IDetector(ABC):
    @abstractmethod
    def predict(self, frame: np.ndarray) -> list[Detection]: ...


class ITracker(ABC):
    @abstractmethod
    def update(
        self, detections: list[Detection], frame_idx: int, frame: np.ndarray
    ) -> list[Track]: ...


class ICrossingLogic(ABC):
    @abstractmethod
    def process(
        self, tracks: list[Track], frame_idx: int, timestamp_seconds: float
    ) -> list[CrossingEvent]: ...

    @abstractmethod
    def get_counters(self) -> dict:
        """Returns current running totals, keyed by (class_name, line_id, direction)."""
        ...


class IEventRepository(ABC):
    @abstractmethod
    def save(self, event: CrossingEvent) -> None: ...

    @abstractmethod
    def flush(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


class IVisualizer(ABC):
    @abstractmethod
    def draw(
        self,
        frame: np.ndarray,
        tracks: list[Track],
        lines: list,
        counters: dict,
    ) -> np.ndarray: ...


class IFrameSource(ABC):
    @abstractmethod
    def read(self) -> tuple[bool, np.ndarray | None]: ...

    @abstractmethod
    def get_fps(self) -> float: ...

    @abstractmethod
    def get_frame_size(self) -> tuple[int, int]: ...

    @abstractmethod
    def release(self) -> None: ...
```

`IDetector` implementations receive an already-loaded model object at construction time (see §4.1) — the model itself is never loaded inside `predict()`.

`ITracker.update()` accepts the raw current `frame` in addition to `detections` and `frame_idx`. The primary implementation, `ByteTrackWrapper`, does not use the `frame` argument internally (ByteTrack is motion-only and requires no appearance features), but the parameter is part of the interface from v1 so that a future `BoT-SORT` implementation — which requires the raw image for appearance-embedding-based re-identification — can be added later purely as a new concrete `ITracker` implementation, without changing the interface, the `TrackerFactory`, or the `PipelineController`'s call site.

---

## 7. Functional Requirements (Must-Have)

### 7.1 Detection
- Frame-by-frame inference with YOLO26 (`yolo26m` default). Frames are processed **one at a time** (`batch_size=1`) in v1 — there is no hard real-time requirement (§1), so the added complexity of frame buffering and batched inference is not justified for the initial version. Batching may be revisited only as a late optimization if schedule allows.
- Inference image size (`imgsz`) is configurable, **default 640**.
- Filter by configurable class list and a single **global confidence threshold** (per-class thresholds are left as a documented future extension, not implemented in v1).
- **Class scope for v1:** `person` and `car` only. The class list is fully config-driven (never hard-coded), so extending to additional COCO classes (bicycle, motorcycle, bus, truck, etc.) later requires no code change — only a config edit. Limiting v1 to two classes keeps manual Ground-Truth creation fast given the 10-day deadline.
- **Class-name fail-fast validation:** every class name listed in `detection.classes` is validated against the loaded YOLO26 model's `model.names` mapping in the composition root, immediately after the model is loaded and before the pipeline starts processing any frame (see §4.1, §4.6, §10.1, §12.1). An unrecognized class name (e.g. a typo like `"perso"`) aborts startup immediately with an error message naming the invalid entry and listing the valid class names available on the loaded model, rather than silently producing zero detections/counts for that class.
- Note: YOLO26 uses a native end-to-end, NMS-free detection head by default, so there is no separate NMS/IoU threshold to configure for the standard inference path.
- Output: list of `Detection` objects.

### 7.2 Tracking
- ByteTrack assigns a unique, persistent `track_id` to each object across frames.
- Maintain track state; the previous position relative to the counting line(s) is mandatory and is owned by the Crossing Logic module (not the tracker itself — see §7.4).
- Each `Track` exposes at minimum: `track_id`, `bbox`, `class_id`, `class_name`, `score`.
- `ITracker.update()` receives the current frame image alongside detections (§6), even though `ByteTrackWrapper` ignores it in v1; this keeps the tracker interface stable for a future appearance-based tracker (BoT-SORT) without a breaking change.

### 7.3 Counting Line(s)
- Each line is defined by **two points in absolute pixel coordinates** in the configuration (`point_a`, `point_b`, each an `[x, y]` pixel pair matching the source video's resolution). Normalized (0–1) coordinates are explicitly **not** supported in v1: since the team works from a fixed, known set of test clips at fixed resolutions rather than deploying against arbitrary/unknown video sources, pixel coordinates are simpler to author, simpler to reason about, and avoid an unnecessary abstraction layer. This is not expected to block any evaluation clip in the current test set.
- **The configuration schema supports a list of one or more counting lines from the start**, even though the v1 pipeline may be evaluated with a single line. This avoids a later breaking refactor.
- Each line has a defined "positive" direction (from point A to point B), used to determine IN vs OUT.
- Each line has a unique `line_id` (string) used throughout Crossing Logic, event logging, and visualization.
- All configured lines are drawn on every annotated frame.
- **Runtime geometry validation:** line points are validated against the actual video frame dimensions at pipeline startup, inside `PipelineController` (not inside the Pydantic model, since frame dimensions are only known once the video file is opened). If any configured line point falls outside the frame, the pipeline fails fast with a clear error message naming the offending line and the frame dimensions (see §10.1).

### 7.4 Crossing Logic (Critical Custom Component)

This is the single most important custom module in the project and its behavior is fully locked below.

**Reference point on the bounding box:** the default reference point used for side/crossing computation is the **bottom-center of the bounding box** (the ground-contact point), not the geometric box center. This is configurable back to box-center if needed. Bottom-center is more stable for pedestrians and vehicles because it doesn't shift when a person bends or the box's apparent height changes.

**Side / direction formula:** for a line defined by points `A` and `B`, and a track's reference point `P`, the signed side is computed via the 2D cross product of the line vector `(B - A)` and the point vector `(P - A)`:

```
signed_distance = (B.x - A.x) * (P.y - A.y) - (B.y - A.y) * (P.x - A.x)
```

The sign of `signed_distance` gives the side of the line. The configured "positive direction" of the line determines whether a negative→positive sign change is labeled IN or OUT.

**State model — keyed by `(track_id, line_id)`, not by `track_id` alone.** A single object can independently cross multiple configured lines over its lifetime (e.g. a parking-lot entrance line and an internal counting line); each `(track_id, line_id)` pair has its own independent state, history window, and cooldown, so crossing one line never blocks or interferes with crossing another.

For each `(track_id, line_id)` pair, the state stored is:
- A sliding window of the last N raw sides (configurable length, default 5–10 frames) — the **raw history window**.
- A separately maintained **`confirmed_side`** — the side the track is currently considered to stably be on. This is distinct from the raw window.
- Remaining cooldown, tracked in seconds and converted internally to a frame count using the video's FPS.
- Timestamp of last observation (used for stale-track cleanup, see below).
- Alongside the raw side history, the **same sliding window also stores the track's `class_name` for each of those N frames** (paired with the side value at that frame). This reuses a single per-`(track_id, line_id)` window for two purposes — side confirmation and class attribution — rather than maintaining two separate histories.

**Sustained-change confirmation logic (locked):**
1. On every frame, compute the current raw side for the track/line pair and append it to the raw history window (drop the oldest entry once the window exceeds its configured length).
2. Compute the majority side of the current window.
3. If a track/line pair has no `confirmed_side` yet (i.e. this is the track's first observation for this line), initialize `confirmed_side` to the current majority side **without emitting any event.** A track's initial side is never treated as a crossing — a crossing is only ever a *change* from an established `confirmed_side`.
4. A crossing is only registered when the window majority is **decisive** (configurable threshold, default ≥ 70% of the window) for the side opposite to `confirmed_side`. This avoids two ambiguous, near-50/50 windows in a row from being misread as a change purely because both are "in flux."
5. When a decisive opposite majority is observed: if the track/line pair is not currently in cooldown, emit a `CrossingEvent`, update `confirmed_side` to the new side, and start the cooldown for that `(track_id, line_id)` pair. If the pair *is* in cooldown, no event is emitted, but note that `confirmed_side` intentionally still does not flip until cooldown clears and a fresh decisive majority is observed — this prevents rapid oscillation near the line from generating a queue of suppressed-but-pending state changes.

**Class attribution at crossing time (locked):** the `class_name`/`class_id` recorded on the emitted `CrossingEvent` is the **majority class over the same raw history window** used for side confirmation (i.e., the most frequent `class_name` among the window's last N frames at the moment the event is emitted), not simply the single frame's class label. This reuses the existing per-`(track_id, line_id)` window (see above) at effectively no extra implementation or runtime cost, and protects the per-class counters against single-frame classification flicker (e.g., a track briefly misclassified as `car` for one frame while otherwise consistently detected as `person`) landing exactly on the frame a crossing is confirmed.

**Cooldown Period:** after a track/line pair generates a valid crossing event it enters a cooldown. Cooldown is expressed in the configuration in **seconds** (canonical unit) and converted internally to a frame count using the video's FPS — this mirrors the same time-based approach agreed for evaluation tolerance (§7.8) and keeps behavior comparable across clips with different frame rates. During cooldown the track cannot generate another event **on that specific line** even if its reference point oscillates across it; crossings on other configured lines are unaffected.

**Minimum displacement/velocity safeguard:** implemented in v1 as an optional, additional check — a minimum Euclidean displacement of the reference point since the pair's last event, and/or a minimum velocity component perpendicular to the line. **Disabled by default** in the shipped config (`min_displacement_px: null` / `min_velocity_px_per_s: null`), so it costs nothing at runtime unless explicitly turned on. It exists so that if jitter problems surface on real test clips late in the timeline, the team can enable it via a one-line config change instead of writing new code under deadline pressure.

**Stale track cleanup:** a `(track_id, line_id)` state entry is removed if the track has not been observed for longer than `stale_track_timeout_seconds` (config value, default `2.0`), converted to frames using the video's FPS — consistent with how cooldown and matching tolerance are expressed. This bounds memory growth over long videos and prevents the state dictionary from accumulating entries for tracks that have permanently left the scene.

Counting is performed per class independently, and totals are tracked per `(class_name, line_id, direction)`.

**Known limitation — track ID switches:** if ByteTrack loses a track (e.g., due to occlusion) and reassigns a new `track_id` to the same physical object, this may cause a duplicate or missed count. The system does **not** attempt to solve this via re-identification in v1. Instead, this failure mode is explicitly documented as a limitation and is expected to surface as a qualitative "track fragmentation" / "occlusion-induced error" failure case during evaluation (§13), not something the pipeline guarantees to avoid.

### 7.5 Event Logging
Every validated crossing produces one `CrossingEvent` record (§5) written via the Repository, containing at minimum:
- `frame_idx` and `timestamp_seconds`
- `track_id`
- `class_name` / `class_id` — the majority-vote value from the raw history window, as defined in §7.4
- `direction` (`IN` or `OUT`)
- `line_id`
- Optional but recommended: `confidence`, `bbox`, `video_name`

**Source frame for `confidence`/`bbox` (locked):** these two optional fields are always populated from the specific `Track` observation at the exact frame on which the `CrossingEvent` is emitted (i.e., the frame that produced the decisive opposite-majority window in step 5 of §7.4) — not an average over the window and not the first frame where the raw side changed. This keeps the semantics simple and directly explainable in the magazine article ("this is the detector's confidence and box at the moment the system formally registered the crossing"), and requires no additional history beyond the `Track` object already available to `ICrossingLogic.process()` on that frame.

Storage format: CSV (UTF-8, header row). All persistence goes through the `IEventRepository` abstraction — the Crossing Logic module never touches the filesystem directly.

### 7.6 Visualization
Annotated video must show:
- Bounding boxes colored by class or track
- Class name + Track ID
- Counting line(s), labeled by `line_id`
- Live counters (e.g., `Person IN: X OUT: Y`, `Car IN: … OUT: …`), per line if multiple lines are configured
- Optional: short trajectory trails (stretch goal)

### 7.7 Configuration
Everything a user might want to change lives in a YAML file validated against a Pydantic v2 model: video path, output directory, class list, global confidence threshold, `imgsz`, one or more counting lines (points + positive direction + `line_id`), tracker hyperparameters, output paths, `cooldown_seconds`, `history_length`, `confirmation_majority_threshold`, `stale_track_timeout_seconds`, `min_displacement_px`, `min_velocity_px_per_s`, matching tolerance (seconds), etc. Hard-coding is forbidden.

**Reference `configs/default.yaml`:**

```yaml
video:
  path: "data/ci_sample_clip.mp4"
  output_dir: "outputs/"

detection:
  model_variant: "yolo26m"   # yolo26n | yolo26s | yolo26m | yolo26l | yolo26x
  imgsz: 640
  confidence_threshold: 0.4
  classes: ["person", "car"]   # validated against the loaded model's model.names at startup (§7.1)

tracker:
  type: "bytetrack"          # bytetrack | botsort
  track_thresh: 0.5
  match_thresh: 0.8
  track_buffer: 30

lines:
  - line_id: "main_line"
    point_a: [100, 400]       # absolute pixel coordinates, matching this clip's resolution
    point_b: [800, 400]       # absolute pixel coordinates, matching this clip's resolution
    positive_direction: "A_to_B"   # A_to_B crossing = IN

crossing_logic:
  reference_point: "bottom_center"   # bottom_center | box_center
  history_length: 8
  confirmation_majority_threshold: 0.7
  cooldown_seconds: 1.5
  stale_track_timeout_seconds: 2.0
  min_displacement_px: null
  min_velocity_px_per_s: null

events:
  output_csv: "outputs/events.csv"

evaluation:
  matching_tolerance_seconds: 1.0   # placeholder — to be finalized with Armila once real clip data exists

visualization:
  output_video: "outputs/annotated.mp4"
  draw_trails: false
```

**Reference `configs/ci.yaml` (CI/integration-test config):** a second, lightweight config is maintained specifically for the GitHub Actions integration test (§9, §12.3, §15.4). It mirrors `default.yaml` in structure but overrides `detection.model_variant` to `"yolo26n"` and points `video.path` at `data/ci_sample_clip.mp4`. The purpose of the integration test is to validate correct end-to-end wiring (composition root, controller loop, CSV schema, annotated-video output) rather than detection quality, so the smallest/fastest model variant is used to keep CI runtime low on GitHub-hosted runners. All published evaluation results in the magazine article are still produced using `configs/default.yaml` (`yolo26m`), never `configs/ci.yaml`.

### 7.8 Evaluation Support
- Event log format is designed so it can be matched against a simple Ground-Truth CSV (frame/time + class + direction).
- Matching rule (agreed):
  - Same class and same direction.
  - Temporal proximity within a configurable tolerance (time-based tolerance in seconds, converted to frames using the video's FPS).
  - One-to-one matching (each GT event matched to at most one prediction — bipartite matching).
- Metrics to compute (Armila's responsibility for final numbers, but the pipeline must emit the raw data needed):
  - Event Precision, Recall, F1
  - Counting Error — reported both as **absolute difference** (`|predicted_total − gt_total|`) and **relative/percentage error** per `(class_name, direction)`, since absolute error is easier to compare across clips of similar traffic volume while relative error is more informative for the magazine write-up and for comparison against other published work.
  - Runtime / FPS of the full pipeline (reported, not a hard requirement)

---

## 8. High-Level Architecture

```
Configuration (YAML, validated via Pydantic v2)
        ↓
Composition Root (manual constructor wiring, no DI framework;
                   loads detector/tracker model objects once;
                   validates configured class list against model.names)
        ↓
PipelineController  ←── injects ──→  IDetector (via DetectorFactory)
        │                              ITracker  (via TrackerFactory)
        │                              ICrossingLogic
        │                              IEventRepository
        │                              IVisualizer
        │                              IFrameSource
        ↓
synchronous loop: read → detect → track → update crossing state
        ↓
notify Observers (Subject) ──→ Logger, Visualizer  [Observer pattern scope]
        ↓
CrossingLogic (bottom-center reference, signed-distance formula,
               confirmed_side + raw history window (side + class),
               cooldown-in-seconds keyed by (track_id, line_id),
               multi-line aware)
        ↓
EventRepository → CSV
        ↓
Visualizer → Annotated Video
```

All components communicate exclusively through `abc.ABC` interfaces. The Controller never imports concrete detector or tracker classes.

---

## 9. Repository Structure (Recommended Production Layout)

```
artificial-intelligence/
├── README.md                  # English, primary documentation; includes third-party (AGPL-3.0) license note
├── README.fa.md               # Optional Persian summary for magazine readers
├── LICENSE                    # MIT (project code)
├── pyproject.toml             # exact-pinned dependencies
├── .pre-commit-config.yaml    # ruff check + ruff format hooks
├── Dockerfile                 # Base build; bakes YOLO26 model weights in at build time (§11)
├── docker-compose.cpu.yml     # CPU profile
├── docker-compose.gpu.yml     # GPU profile (CUDA base image)
├── .dockerignore
├── configs/
│   ├── default.yaml           # production/evaluation config (yolo26m)
│   ├── ci.yaml                # lightweight CI/integration-test config (yolo26n)
│   └── examples/
├── src/
│   └── mot_counting/
│       ├── __init__.py
│       ├── config.py                 # Pydantic v2 models
│       ├── composition_root.py       # Manual DI wiring, model loading, class-list validation
│       ├── types.py                  # Detection, Track, CrossingEvent, Direction
│       ├── interfaces/               # abc.ABC interfaces
│       │   ├── detector.py
│       │   ├── tracker.py
│       │   ├── crossing.py
│       │   ├── repository.py
│       │   ├── visualizer.py
│       │   └── frame_source.py
│       ├── factories/                # Factory Pattern
│       │   ├── detector_factory.py
│       │   └── tracker_factory.py
│       ├── detectors/                # Concrete detectors (YOLO26 wrapper)
│       ├── trackers/                 # Concrete trackers (ByteTrack)
│       ├── crossing/                 # CrossingLogic + geometry + state machine
│       ├── repositories/             # CsvEventRepository, InMemoryEventRepository
│       ├── visualizers/
│       ├── controllers/              # PipelineController
│       ├── observers/                # Subject / Observer base classes
│       └── utils/
│           ├── geometry.py           # side/signed-distance helpers
│           ├── video_io.py
│           └── metrics.py
├── scripts/
│   ├── run_pipeline.py
│   ├── evaluate.py
│   └── annotate_ground_truth.py      # manual GT capture tool, see §13
├── tests/
│   ├── unit/
│   └── integration/
├── data/
│   └── ci_sample_clip.mp4     # Synthetic or CC0 clip, for CI integration test only
├── outputs/
├── docs/
└── .github/
    ├── ISSUE_TEMPLATE/
    │   └── implementation_task.md    # see §15.2
    └── workflows/                     # Lint, type-check, unit tests, Docker integration test (uses configs/ci.yaml)
```

---

## 10. Detailed Module Specifications

### 10.1 Configuration (`config.py`)
- Load YAML → validated **Pydantic v2** model (locked; not a plain dataclass).
- Required fields: video path, output directory, class list, global confidence threshold, `imgsz`, counting line(s) (each a pair of points in absolute pixel coordinates + positive direction + `line_id`), tracker parameters, event CSV path, `history_length`, `cooldown_seconds`, `confirmation_majority_threshold`, `stale_track_timeout_seconds`, matching tolerance (seconds), etc.
- Support relative paths resolved from the project root.
- Line-point-vs-frame-dimension validation is **not** performed here (frame dimensions are unknown at config-parse time) — it is performed lazily by `PipelineController` once the video is opened (§7.3), and fails fast with a message naming the offending line and the actual frame dimensions.
- **Class-name-vs-model validation is likewise not performed here** (the set of valid class names is only known once the YOLO26 model object is loaded) — it is performed once, in the composition root, immediately after model loading (§4.1, §4.6, §7.1, §12.1).

### 10.2 Interfaces & Dependency Injection
- Every major component is defined behind an `abc.ABC` interface (§6).
- Concrete classes are created only inside Factories or the composition root.
- The `PipelineController` receives all dependencies via its constructor. No service-locator anti-pattern; pure constructor injection; no external DI library.

### 10.3 Detector (`detectors/` + Factory)
- Thin, clean wrapper around Ultralytics **YOLO26 (`yolo26m` default)** that implements `IDetector`.
- The composition root loads the underlying Ultralytics model object once; `DetectorFactory` wraps that already-loaded object into the concrete `IDetector` implementation (§4.1).
- Immediately after loading the model object and before constructing the Factory, the composition root validates `detection.classes` from config against the loaded model's `model.names`; any unknown class name fails startup immediately (§4.1, §7.1, §12.1).
- `Detection` objects (§5) contain: `xyxy`, `confidence`, `class_id`, `class_name`.
- Model variant (`yolo26n/s/m/l/x`) and `imgsz` are both read from configuration, so a CPU-only contributor can switch to a lighter variant without touching code.

### 10.4 Tracker (`trackers/` + Factory)
- Interface that accepts detections, the current frame image, and the frame index, and returns tracks with stable IDs (§6).
- Primary implementation: `ByteTrackWrapper`, with default hyperparameters `track_thresh=0.5`, `match_thresh=0.8`, `track_buffer=30` (§3). It receives the `frame` argument for interface compatibility but does not use it, since ByteTrack is a motion-only tracker with no appearance model.
- Each `Track` exposes at least: `track_id`, `bbox`, `class_id`, `class_name`, `score`. Position history relative to the counting line(s) is computed and owned entirely by the Crossing Logic module, not by the tracker.
- A future `BoTSortWrapper` (stretch goal, §14) can consume the same `frame` argument for its appearance-embedding-based association step without any change to `ITracker`, `TrackerFactory`, or `PipelineController`.

### 10.5 Crossing Logic (`crossing/`) – Highest Priority Custom Code
- Implements `ICrossingLogic` (§6).
- Maintains a dictionary keyed by `(track_id, line_id)` storing: the raw history window (side value **and** `class_name` per frame), `confirmed_side`, remaining cooldown (seconds → frames), and last-observed timestamp (for stale cleanup).
- On every frame, for each active track and each configured line, executes the sequence in §7.4 (compute reference point → compute side → append side and class to window → evaluate decisive majority against `confirmed_side` → check cooldown → emit event with majority-vote class / update state → determine direction from line geometry).
- When emitting a `CrossingEvent`, populates `confidence` and `bbox` from the current frame's `Track` observation (§7.5), and `class_name`/`class_id` from the majority vote over the window (§7.4) — not necessarily the same frame's individual class label.
- Also performs stale `(track_id, line_id)` cleanup based on `stale_track_timeout_seconds`.
- Robustness requirements are satisfied by the combination of raw-window majority confirmation + separately-maintained `confirmed_side` + per-`(track_id, line_id)` Cooldown Period (plus the optional, default-disabled minimum displacement/velocity safeguards).
- Track ID switches are **not** resolved here — see the documented limitation in §7.4.

### 10.6 Event Repository
- Implements `IEventRepository`.
- Only component that opens, writes, and closes the CSV file.
- Provides `save(event)`, `flush()`, `close()`.
- Crossing Logic never touches the filesystem directly.
- `InMemoryEventRepository` is also provided, used exclusively in unit tests to assert on emitted events without touching disk.

### 10.7 Visualizer
- Implements `IVisualizer`.
- Draws boxes, labels (class + track_id), counting line(s), and on-screen counters.
- Counters update live from the running totals maintained by `ICrossingLogic.get_counters()`.
- Subscribes as an `Observer` (§4.3), notified once per frame after crossing state is updated.

### 10.8 Logger (Observer)
- Subscribes as an `Observer` alongside the Visualizer.
- Uses the standard library `logging` module.
- Logs pipeline lifecycle events (start, stop, per-frame warnings on decode failure) and, at `DEBUG` level, per-frame crossing decisions for troubleshooting.

### 10.9 Pipeline Controller
- Owns the main loop and lifecycle.
- Runs the core `read → detect → track → update crossing state` sequence synchronously, then notifies the `Subject` so Observer consumers (Logger, Visualizer) run (§4.3, §8).
- Performs lazy line-geometry validation at startup (§7.3, §10.1).
- Reports average FPS and final counts at the end (FPS is informational, not a pass/fail requirement).

---

## 11. Docker & Reproducibility (Mandatory)

A production-grade academic system must guarantee identical results on any machine. Therefore:

- `Dockerfile` builds a self-contained image with the exact Python version, system libraries (OpenCV dependencies, etc.), and project dependencies (exact-pinned, §3).
- **YOLO26 model weights are downloaded and baked into the Docker image at `docker build` time**, not lazily downloaded at container runtime. This removes any dependency on network availability or Ultralytics Hub uptime when the container actually runs the pipeline, which is a direct requirement of the "identical results on any machine" reproducibility guarantee — a runtime download would introduce an external, non-reproducible variable into every pipeline run. The specific weight file(s) needed (matching `detection.model_variant` in `configs/default.yaml` and `configs/ci.yaml`) are fetched once during the image build and stored inside the image filesystem; the composition root loads them from local disk at runtime.
- **Two Compose profiles are provided:** `docker-compose.cpu.yml` and `docker-compose.gpu.yml` (CUDA base image), since not every team member has access to a GPU. Both must produce comparable results; the GPU profile exists purely for speed, not correctness.
- Compose configuration mounts the `configs/`, `data/`, and `outputs/` directories, and runs the pipeline with a single command; it can also optionally run tests or evaluation helpers.
- The README documents both native (uv/poetry) and Docker workflows, for both profiles.
- All published results in the magazine article must be reproducible via the Docker image.
- There is no real-time processing requirement — the pipeline may run slower than the source video's FPS; runtime/FPS is reported as an evaluation metric only.

This eliminates "works on my machine" problems that are especially common in computer-vision projects, and additionally eliminates "works only with network access" problems caused by on-demand model weight downloads.

---

## 12. Engineering Practices

### 12.1 Error Handling (Locked Policy)
- **Startup errors** (video file missing/corrupt, model fails to load, invalid/unreadable config, or a configured class name not present in the loaded model's `model.names` — see §7.1, §4.1) → **fail fast**: raise immediately with a clear, specific error message and exit. There is no recovery path for a fundamentally broken setup, and a silent partial start would be worse than a hard stop.
- **Per-frame errors** (a single frame fails to decode) → **log a warning and skip the frame**, then continue processing. A single bad frame must never abort a full video run.

### 12.2 Logging
- Standard library `logging`, configured once at pipeline startup (level configurable, default `INFO`).
- `WARNING` for per-frame recoverable issues, `ERROR`/exception for startup fail-fast conditions, `DEBUG` for granular crossing-decision tracing.

### 12.3 Testing
- Framework: `pytest`.
- `tests/unit/`: geometry/signed-distance helper functions, Crossing Logic edge cases (jitter near the line, cooldown behavior, multi-line independence, stale-track cleanup, initial-side-not-a-crossing behavior, class-majority-vote attribution at crossing time), Factories (with mock model objects, including a case that exercises the class-name-vs-`model.names` validation failure path), Repository (using `InMemoryEventRepository` plus a real-file test for `CsvEventRepository`).
- `tests/integration/`: full pipeline run against `data/ci_sample_clip.mp4` using `configs/ci.yaml` (`yolo26n`, for fast CI turnaround — see §7.7), runnable both natively and inside Docker (CPU profile in CI; GPU profile is a manual/local check only).
- All developers unit- and integration-test their own modules before requesting review (§2).

### 12.4 Pre-commit Hook (Locked)
- Tooling: **Ruff only** — `ruff check` (linting) and `ruff format` (formatting). No other linter/formatter is used in this project.
- Configured via `.pre-commit-config.yaml` at the repository root, installed with `pre-commit install` as part of the onboarding step for every contributor.
- **No commit is permitted unless both `ruff check` and `ruff format` pass.** This is enforced locally via the pre-commit hook and additionally re-checked in CI (§15.4) on every pushed commit/PR, so a contributor cannot bypass it by skipping local hook installation.

### 12.5 Documentation & Code Standards
- Strict type hints everywhere.
- Docstrings (Google or NumPy style) on all public functions and classes.
- Meaningful variable names; no magic numbers.
- Clean git history, conventional commits, meaningful PR descriptions.
- Comprehensive README (English, primary) with: problem statement; architecture diagram (text or Mermaid) including the design patterns; installation & quick-start (native + Docker, both CPU/GPU profiles); configuration reference; how to reproduce the published results; citation / MIT license information; and the third-party AGPL-3.0 licensing note for the `ultralytics` dependency (§3.1).
- Optional `README.fa.md` with a Persian summary for magazine readers.
- The CI/integration-test clip is synthetic or CC0/public-domain — never footage of real, identifiable people — to avoid any privacy or licensing issue.
- No hard-coded absolute paths or credentials.

---

## 13. Evaluation Protocol (Armila-Owned Final Assessment)

**Test Set:** 4–6 short clips covering:
- Normal traffic / pedestrian flow
- Higher density
- Partial occlusion
- Bi-directional movement

**Ground Truth creation:** built using a small dedicated capture script, `scripts/annotate_ground_truth.py`, rather than fully manual frame-by-frame CSV entry. The script plays the clip via OpenCV with the following controls:
- Standard real-time playback by default.
- **Pause / resume** on a dedicated key.
- **Single-frame step forward and single-frame step backward** on dedicated keys, usable at any time (including while paused), so Armila can land on the exact frame a crossing occurs rather than relying purely on reaction-time key presses during real-time playback. This directly improves Ground Truth precision, since real-time-only capture would introduce a systematic human-reaction-time offset between the true crossing frame and the recorded one.
- When Armila presses the "mark crossing" key at the current frame (whether during playback, while paused, or after frame-stepping to the precise moment), the script captures the current `frame_idx` and `timestamp_seconds` and prompts for `class_name` and `direction`, then appends the row to a GT CSV (same schema as the prediction event log, minus `track_id`).

This removes manual frame-index bookkeeping and the risk of timestamp transcription errors, gives Armila a precise way to pinpoint the exact crossing frame when needed, and remains fast enough to annotate all 4–6 clips within the project's tight timeline.

**Matching:**
- Identical class + direction.
- Temporal distance ≤ configurable tolerance (time-based, in seconds; converted to frames per clip using that clip's FPS). Default placeholder value: `1.0` second (§7.7, `evaluation.matching_tolerance_seconds`) — to be finalized with Armila once the team has annotated at least one real clip and can see how close true crossings land in practice.
- Bipartite one-to-one matching (each GT event matched to at most one prediction).

**Reported Metrics:**
- Event Precision / Recall / F1
- Counting Error — both absolute and relative/percentage, per `(class_name, direction)` (§7.8)
- Pipeline FPS / total runtime (informational)
- Qualitative Failure Cases (missed detection, class confusion, track fragmentation / ID switches, false crossing, occlusion-induced errors) with concrete examples from the test clips, identified by Armila's manual review of the annotated output video against the GT — no automated class-confusion detector is built for v1, since a spatial-temporal secondary matcher would add meaningful complexity for limited return given the timeline. Track ID switch cases identified in §7.4/§10.5 are expected to be documented here.

---

## 14. Stretch Goals (Only If Core Pipeline Is Solid and Schedule Allows)

1. Side-by-side comparison of ByteTrack vs BoT-SORT on the same videos and GT.
2. Optional ROI occupancy / time-in-area analytics.
3. Short trajectory visualization.
4. Limited ReID experiment (addressing the track ID switch limitation from §7.4).
5. Evaluation on a subset of MOT17 (for tracker quality only).
6. YOLO11 vs YOLO26 detection quality / speed comparison (late, non-blocking).
7. Frame-batching for detection throughput, if FPS proves to be a real bottleneck on team members' machines.

Given the 10-day timeline (deadline September 1, 2026), stretch goals are explicitly lower priority than a fully stable core pipeline and evaluation.

---

## 15. Development Workflow, Git Strategy & Task Allocation

### 15.1 Branch Strategy (Locked)
- `main` is the protected, stable branch.
- `ai-develop` is the team's integration branch, created off `main`. All feature work for this project happens off `ai-develop`, not off `main` directly.
- Every contributor (Mostafa, Farzad, Amirmohammad) creates their feature branches from `ai-develop` (naming convention: `feature/<short-task-name>`).
- All pull requests target `ai-develop`, not `main`. No one pushes directly to `ai-develop` or `main`.
- CI (§15.4) runs on every PR into `ai-develop`.

### 15.2 Task Ticket Format (Locked)
- Tasks for Farzad and Amirmohammad are issued as **GitHub Issues**, using a fixed template stored at `.github/ISSUE_TEMPLATE/implementation_task.md` with the following required sections:
  - **Context** — where this fits in the architecture and why it's needed.
  - **Interface to implement** — the exact `abc.ABC` interface and method signatures from §6 the task must satisfy.
  - **Acceptance criteria** — concrete, checkable conditions for "done."
  - **Test cases** — the specific unit/integration test scenarios the implementer must cover before requesting review.
- Using GitHub Issues (rather than a flat `TASKS.md`) keeps a linkable, citable history of task→PR→merge for both day-to-day tracking and the magazine article's process description.

### 15.3 Code Review & Merge Policy
- Every PR into `ai-develop` requires Mostafa's review and approval before merge — as project lead and primary architecture owner, he is the only one positioned to judge interface-contract compliance across modules.
- CI must be green (lint, type-check, unit tests, Docker CPU-profile integration test) before a PR can be merged.

### 15.4 Sequencing
1. Mostafa creates the repository skeleton, `ai-develop` branch, configuration system (Pydantic v2), all `abc.ABC` interfaces (§6), Factories, the manual composition root (including class-list validation against `model.names`), the `PipelineController` skeleton, the Observer implementation (Logger/Visualizer scope, §4.3), and the Crossing Logic with full confirmed-side/history-window (side + class) + Cooldown-in-seconds + multi-line support (§7.4). This is the bulk of Mostafa's ~60–70% share of the work, accelerated using Cursor given the 10-day deadline.
2. Docker (both CPU and GPU Compose profiles, with model weights baked in at build time per §11) and the pre-commit hook (§12.4) are added early so every subsequent developer works inside the same environment and tooling from their first commit.
3. Once the skeleton is stable and pushed to `ai-develop`, Farzad and Amirmohammad each receive highly detailed, independent task tickets (§15.2) — together covering roughly 20% of the project each — that operate almost exclusively behind a single interface (examples: implement a concrete `IDetector` wrapper, implement `IVisualizer`, implement `CsvEventRepository`, write unit tests for geometry helpers, build `annotate_ground_truth.py` with pause/frame-step controls, etc.).
4. Continuous integration (GitHub Actions) runs Ruff lint/format checks, type checking, unit tests, and a Docker-based integration test (CPU profile, using `configs/ci.yaml` with `yolo26n` for fast turnaround — §7.7, §12.3) on every PR into `ai-develop`.
5. Armila performs the official evaluation only after a complete, stable pipeline exists and the agreed event format is frozen.

---

## 16. Deliverables for the Magazine Article (Section 3 – Implementation)

The code repository itself is the primary artifact. The magazine text (≈ 4 pages) will:

- Clearly state the problem and input assumptions.
- Describe the overall architecture, explicitly mentioning the design patterns used (Factory, Observer, Controller, Repository) and the Dependency-Injection approach, and naming the concrete implementation choices (Pydantic v2, `abc.ABC`, explicit Observer classes scoped to side-effect consumers, manual composition root).
- Walk through Detection → Tracking → Crossing Logic, with emphasis on the bottom-center reference point, the signed-distance direction formula, the confirmed-side/history-window + Cooldown anti-jitter mechanisms, and the class-majority-vote attribution used to protect per-class counts against single-frame classification flicker.
- Explain per-class, per-line counting and event logging via the Repository.
- Present the evaluation protocol, metrics, and real failure cases observed on the team's test clips — including any track ID switch / fragmentation cases.
- Conclude with strengths, limitations (explicitly including the unresolved track ID switch issue), and possible extensions (the stretch goals in §14).

The repository README and this document together serve as the complete technical specification so that any competent developer can implement or extend the system without further clarification.
