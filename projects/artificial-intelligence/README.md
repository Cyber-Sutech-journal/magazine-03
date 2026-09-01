# Multi-Object Tracking and Directional Counting System

A production-oriented video analytics pipeline for fixed-camera object counting using YOLO26 + ByteTrack.
Built as part of the **Cyber Sutech Magazine 03** academic publication.

---

## Overview

This system detects objects (primarily `person` and `car`) in recorded video from a fixed camera,
assigns persistent track IDs via ByteTrack, detects crossings of user-defined virtual counting lines,
and determines the direction (IN / OUT) of each crossing. Every crossing event is logged to CSV;
an annotated output video is produced.

**Design patterns used:** Factory, Observer (scoped to side-effect consumers), Controller,
Repository, Dependency Injection (manual composition root, `abc.ABC` interfaces).

```
Configuration (YAML + Pydantic v2)
        ↓
Composition Root  →  IDetector · ITracker · ICrossingLogic · IEventRepository · IVisualizer · IFrameSource
        ↓
PipelineController: read → detect → track → update crossing state → notify Observers
        ↓
CrossingLogic (bottom-center ref, signed-distance, confirmed_side + history window, cooldown)
        ↓
EventRepository → CSV          Visualizer → Annotated Video
```

---

## Architecture

| Layer | Component | Pattern |
|---|---|---|
| Interfaces | `IDetector`, `ITracker`, `ICrossingLogic`, `IEventRepository`, `IVisualizer`, `IFrameSource` | Abstraction / DI |
| Factories | `DetectorFactory`, `TrackerFactory` | Factory |
| Orchestration | `PipelineController` | Controller |
| Side-effect consumers | `Logger`, `Visualizer` | Observer |
| Persistence | `CsvEventRepository`, `InMemoryEventRepository` | Repository |
| Wiring | `composition_root.py` | Composition Root |

All components communicate exclusively through `abc.ABC` interfaces.
The controller never imports concrete YOLO or ByteTrack classes.

---

## Installation

### Native (uv recommended)

```bash
# Clone the repository
git clone https://github.com/Cyber-Sutech-journal/magazine-03
cd projects/artificial-intelligence/src/mot_counting

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install the package with dev dependencies
cd ...
pip install -e ".[dev]"

# Install the pre-commit hooks (required for all contributors)
pre-commit install
```

### Docker — CPU profile

```bash
docker compose -f docker-compose.cpu.yml up
```

### Docker — GPU profile (CUDA required)

```bash
docker compose -f docker-compose.gpu.yml up
```

Both Docker profiles mount `configs/`, `data/`, and `outputs/` from the host.
YOLO26 model weights are baked into the image at build time for full reproducibility.

---

## Configuration Reference

All runtime parameters live in a YAML config file validated by Pydantic v2.
No hard-coded paths, thresholds, or class names exist anywhere in the source code.

| Key | Description | Default |
|---|---|---|
| `video.path` | Path to input video | `data/ci_sample_clip.mp4` |
| `video.output_dir` | Directory for pipeline outputs | `outputs/` |
| `detection.model_variant` | YOLO26 variant (`yolo26n/s/m/l/x`) | `yolo26m` |
| `detection.imgsz` | Inference image size | `640` |
| `detection.confidence_threshold` | Global detection confidence threshold | `0.4` |
| `detection.classes` | List of class names to detect | `["person", "car"]` |
| `tracker.type` | Tracker backend (`bytetrack`) | `bytetrack` |
| `tracker.track_thresh` | ByteTrack detection threshold | `0.5` |
| `tracker.match_thresh` | ByteTrack association threshold | `0.8` |
| `tracker.track_buffer` | Max frames to keep a lost track | `30` |
| `lines[].line_id` | Unique string ID for this counting line | — |
| `lines[].point_a` | `[x, y]` start point (absolute pixels) | — |
| `lines[].point_b` | `[x, y]` end point (absolute pixels) | — |
| `lines[].positive_direction` | `A_to_B` crossing = IN | `A_to_B` |
| `crossing_logic.reference_point` | `bottom_center` or `box_center` | `bottom_center` |
| `crossing_logic.history_length` | Raw side history window length (frames) | `8` |
| `crossing_logic.confirmation_majority_threshold` | Fraction of window required for decisive majority | `0.7` |
| `crossing_logic.cooldown_seconds` | Minimum seconds between events per (track, line) | `1.5` |
| `crossing_logic.stale_track_timeout_seconds` | Remove state for tracks unseen for this long | `2.0` |
| `crossing_logic.min_displacement_px` | Minimum displacement guard (disabled by default) | `null` |
| `crossing_logic.min_velocity_px_per_s` | Minimum velocity guard (disabled by default) | `null` |
| `events.output_csv` | Path for the crossing-event CSV | `outputs/events.csv` |
| `evaluation.matching_tolerance_seconds` | Temporal tolerance for GT matching | `1.0` |
| `visualization.output_video` | Path for the annotated output video | `outputs/annotated.mp4` |
| `visualization.draw_trails` | Draw short trajectory trails | `false` |

See `configs/default.yaml` for the full reference config.

---

## Reproducing Results

All published results from the magazine article use `configs/default.yaml` (`yolo26m`).

**Native:**

```bash
python scripts/run_pipeline.py --config configs/default.yaml

# Optional: override the video path without editing YAML
python scripts/run_pipeline.py --config configs/default.yaml --video path/to/video.mp4

# Evaluate against ground truth
python scripts/evaluate.py \
    --predictions outputs/events.csv \
    --ground-truth data/ground_truth.csv
```

`--tolerance-seconds` optionally overrides the inclusive matching tolerance. When omitted, the
evaluator loads `evaluation.matching_tolerance_seconds` from the project-local
`configs/default.yaml`, even when invoked from another working directory. Use `--output-dir` to
choose where `evaluation_summary.csv` and `evaluation_matches.csv` are written; without it, the
artifacts are written alongside the predictions CSV. See
[`docs/evaluation-protocol.md`](docs/evaluation-protocol.md) for matching and metric definitions.

**Docker (CPU):**

```bash
docker compose -f docker-compose.cpu.yml run mot-counting \
    python scripts/run_pipeline.py --config configs/default.yaml
```

---

## License

The source code of this project is licensed under the **MIT License** — see the [`LICENSE`](LICENSE) file for the full text.

### Third-Party Licensing Notice — Ultralytics YOLO26 (AGPL-3.0)

This project depends on the [`ultralytics`](https://github.com/ultralytics/ultralytics) package,
which is separately licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**
by Ultralytics. The MIT license above covers only this project's own source code, not the
`ultralytics` dependency.

Because this project — its code, configurations, and outputs — is published publicly and
completely as an open-source academic repository, using `ultralytics` under AGPL-3.0 is
fully permitted at no cost. No Ultralytics Enterprise License is required for this use case.

**If you wish to reuse the `ultralytics` dependency itself** in your own work, you must
independently comply with its AGPL-3.0 license (or obtain your own Ultralytics Enterprise
License). This is independent of this project's MIT license.

For the full AGPL-3.0 text, see: <https://www.gnu.org/licenses/agpl-3.0.html>
