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

**Default path for everyone (Windows, macOS, Linux):** native Python **or** the
**CPU Docker** profile. Both produce the same outputs. CUDA GPU Docker is
optional and only works where Docker can see an NVIDIA GPU.

| Machine | Recommended | GPU Docker (`--gpus all`) |
|---|---|---|
| Windows, macOS, Linux (no NVIDIA, or Apple Silicon) | Native install **or** `docker-compose.cpu.yml` | No — Docker has no NVIDIA device |
| Windows or Linux with NVIDIA GPU | Either profile; GPU is faster, not more correct | Yes — Docker Desktop (WSL2) or nvidia-container-toolkit |

All commands below are run from `projects/artificial-intelligence/`
(the folder that contains `pyproject.toml`, `configs/`, and `Dockerfile`).

---

### Native — Windows (PowerShell)

Requires [Python 3.10+](https://www.python.org/downloads/) (3.12 recommended).
In PowerShell, quote `".[dev]"` so `[dev]` is not treated as a wildcard.

```powershell
git clone https://github.com/Cyber-Sutech-journal/magazine-03
cd magazine-03\projects\artificial-intelligence

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
# If activation is blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

pip install -e ".[dev]"
pre-commit install

python -c "from ultralytics import YOLO; YOLO('yolo26n.pt'); YOLO('yolo26m.pt')"

python scripts\run_pipeline.py --config configs\ci.yaml
```

---

### Native — macOS / Linux (bash or zsh)

```bash
git clone https://github.com/Cyber-Sutech-journal/magazine-03
cd magazine-03/projects/artificial-intelligence

python3 -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
pre-commit install

python -c "from ultralytics import YOLO; YOLO('yolo26n.pt'); YOLO('yolo26m.pt')"

python scripts/run_pipeline.py --config configs/ci.yaml
```

Optional faster install with [uv](https://docs.astral.sh/uv/):  
`uv venv && uv pip install -e ".[dev]"` (same quote rule on Windows PowerShell).

---

### Docker — CPU profile (Windows, macOS, Linux)

This is the portable Docker path. No NVIDIA GPU is required.
Install [Docker Desktop](https://docs.docker.com/get-docker/) (Windows/macOS)
or Docker Engine (Linux). Compose V2 is included as `docker compose`.

**Before the first `docker compose build`:** `yolo26n.pt` and `yolo26m.pt` must
exist in this folder (they are copied into the image). If you already ran the
native install above, they are there. Otherwise:

```bash
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolo26n.pt'); YOLO('yolo26m.pt')"
```

```bash
docker compose -f docker-compose.cpu.yml build pipeline
```

Confirm weights are inside the image (this runs `ls` *inside* the Linux
container, so the same command works on Windows):

```bash
docker run --rm --entrypoint ls mot-counting:cpu -lh /app/yolo26n.pt /app/yolo26m.pt
```

Run the pipeline (one line — works in PowerShell, cmd, bash, and zsh):

```bash
docker compose -f docker-compose.cpu.yml run --rm pipeline --config configs/ci.yaml
```

Published evaluation config:

```bash
docker compose -f docker-compose.cpu.yml run --rm pipeline --config configs/default.yaml
```

Override the video file without editing YAML:

```bash
docker compose -f docker-compose.cpu.yml run --rm pipeline --config configs/default.yaml --video data/my_clip.mp4
```

Outputs land on the host at `outputs/annotated.mp4` and `outputs/events.csv`.

Unit tests in Docker:

```bash
docker compose -f docker-compose.cpu.yml run --rm tests
```

---

### Docker — GPU profile (optional, NVIDIA only)

CUDA is **speed only**; counts and events must match the CPU profile (§11).

**Do not use `--gpus all` on macOS or on any PC without an NVIDIA GPU.**
That flag fails with `could not select device driver ... capabilities: [[gpu]]`.
Use the CPU profile instead.

**Where `--gpus all` is valid**

- Linux: NVIDIA driver ≥ 545, plus
  [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- Windows: NVIDIA GPU + Docker Desktop with the WSL2 backend and GPU support enabled.

Check (Linux / Windows+WSL2 only):

```bash
docker run --gpus all --rm nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Build (CUDA base image is `linux/amd64`; first build is large and slow):

```bash
docker compose -f docker-compose.gpu.yml build pipeline
```

Run **with** GPU (NVIDIA host only):

```bash
docker compose -f docker-compose.gpu.yml run --rm --gpus all pipeline --config configs/ci.yaml
```

Run the CUDA image **without** a GPU (CPU fallback; useful as a smoke test).
On Apple Silicon this uses `linux/amd64` emulation and may print an NNPACK
warning — that is harmless if the run finishes:

```bash
docker compose -f docker-compose.gpu.yml run --rm pipeline --config configs/ci.yaml
```

---

### Evaluating results

```bash
python scripts/evaluate.py \
    --predictions outputs/events.csv \
    --ground-truth data/ground_truth.csv

# Optional: override temporal matching tolerance
python scripts/evaluate.py \
    --predictions outputs/events.csv \
    --ground-truth data/ground_truth.csv \
    --tolerance-seconds 0.5
```

See [`docs/evaluation-protocol.md`](docs/evaluation-protocol.md) for matching
rules, metric definitions, and how to create ground-truth annotations with
`scripts/annotate_ground_truth.py`.

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

**Docker (CPU — all platforms):**

```bash
docker compose -f docker-compose.cpu.yml build pipeline
docker compose -f docker-compose.cpu.yml run --rm pipeline --config configs/default.yaml
```

**Docker (GPU — NVIDIA Linux or Windows+WSL2 only):**

```bash
docker compose -f docker-compose.gpu.yml build pipeline
docker compose -f docker-compose.gpu.yml run --rm --gpus all pipeline --config configs/default.yaml
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

## Ground Truth Annotation Tool
Use `scripts/annotate_ground_truth.py` to create manual ground-truth
annotations for video line-crossing events.

### Frame Accuracy & Seeking Behavior
To prevent frame drift and codec decoding inaccuracies (especially on inter-frame compressed video like H.264), the tool relies on sequential reading (`capture.read()`) during normal playback and forward steps. Seeking (`CAP_PROP_POS_FRAMES`) is only performed when stepping backward or jumping.

### Usage

**Linux / macOS (Bash):**
```bash
python ./scripts/annotate_ground_truth.py \
 --video "./path/to/clip.mp4" \
 --output "./path/to/ground_truth.csv" 
```

**Windows (PowerShell):**
```powershell
python .\scripts\annotate_ground_truth.py `
--video ".\path\to\clip.mp4" `
--output ".\path\to\ground_truth.csv" 
```

### Keyboard controls
| Key | Action |
| --- | --- |
| `Space` | Pause or resume video playback |
| `Right Arrow` | Move one frame forward and pause |
| `Left Arrow` | Move one frame backward and pause |
| `M` | Mark a crossing on the currently displayed frame |
| `U` | Remove the most recently added annotation |
| `Q` or `Esc` | Save annotations and exit |

When marking a crossing, the tool prompts for:
- `class_name`
- `direction` (`IN` or `OUT`)
- `line_id`

### Output CSV schema
frame_idx,timestamp_seconds,class_name,direction,line_id,video_name

- `frame_idx` is zero-based.
- `timestamp_seconds` is calculated as `frame_idx / FPS`.
-  Each marked crossing creates exactly one CSV row.
