# syntax=docker/dockerfile:1.7
# =============================================================================
# Multi-Object Tracking and Directional Counting System — Docker image (§11)
#
# Build args
# ----------
# BASE_IMAGE   Python base image to use.  Default: python:3.12-slim (CPU).
#              GPU profile sets this to nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
#              via docker-compose.gpu.yml.
#
# Build examples
# --------------
# CPU:  docker build -t mot-counting:cpu .
# GPU:  docker build --build-arg BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 \
#                    -t mot-counting:gpu .
#
# Reproducibility guarantee (§11)
# --------------------------------
# YOLO26 model weights (yolo26n.pt, yolo26m.pt) are COPIED from the build
# context and baked into the image filesystem during 'docker build'.
# No network call to download weights happens at container runtime.
# =============================================================================

ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

# CPU profile uses the CPU-only PyTorch index so the image does not pull CUDA
# wheels.  GPU compose overrides this to https://download.pytorch.org/whl/cu124
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

# ---------------------------------------------------------------------------
# 1. Install Python when the base image is a CUDA/Ubuntu image that ships
#    without it (e.g. nvidia/cuda:*-ubuntu22.04).  Ubuntu 22.04 provides
#    Python 3.10, which satisfies requires-python >= 3.10 (§3).
#    For python:3.12-slim this block is a complete no-op.
# ---------------------------------------------------------------------------
RUN if ! command -v python3 > /dev/null 2>&1; then \
        apt-get update && \
        apt-get install -y --no-install-recommends \
            python3 \
            python3-pip \
            python3-venv && \
        ln -sf /usr/bin/python3 /usr/local/bin/python && \
        ln -sf /usr/bin/python3 /usr/local/bin/python3 && \
        ln -sf /usr/bin/pip3 /usr/local/bin/pip && \
        rm -rf /var/lib/apt/lists/*; \
    fi

# ---------------------------------------------------------------------------
# 2. System-level OpenCV runtime dependencies (required by opencv-python).
#    libgl1 provides libGL.so.1; libglib2.0-0 provides libgobject-2.0.so.0.
# ---------------------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 3. Install exact-pinned Python dependencies — split into two steps so that
#    the heavy dependency layer (ultralytics, torch, opencv, etc.) is cached
#    separately from the project source code.  Changing src/ only re-runs the
#    lightweight "install package itself" step, not the full pip solve.
# ---------------------------------------------------------------------------
WORKDIR /app

# 3a. Upgrade pip and install the exact-pinned runtime dependencies listed in
#     pyproject.toml.  These are installed directly (not via -e .) so that
#     this layer is cached whenever pyproject.toml's dep list is unchanged.
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url "${TORCH_INDEX_URL}" && \
    pip install --no-cache-dir --upgrade-strategy only-if-needed \
        "ultralytics==8.4.128" \
        "opencv-python==4.11.0.86" \
        "numpy==1.26.4" \
        "pydantic==2.13.4" \
        "pyyaml==6.0.3"

# 3b. Copy project files needed by setuptools for the editable install.
COPY README.md LICENSE pyproject.toml ./
COPY src/ src/

# 3c. Install the project package itself (no-deps: all deps already installed above).
#     pytest is included so the Compose `tests` service can run without a rebuild.
RUN pip install --no-cache-dir -e . --no-deps && \
    pip install --no-cache-dir "pytest==9.1.1" "lap==0.5.13"

# ---------------------------------------------------------------------------
# 4. Copy remaining project files.
# ---------------------------------------------------------------------------
COPY scripts/ scripts/
COPY configs/ configs/
COPY tests/    tests/
COPY data/     data/

# ---------------------------------------------------------------------------
# 5. Bake YOLO26 model weights into the image (§11).
#    Weights are COPIED from the build context into /app so that
#    YOLO("yolo26n.pt") / YOLO("yolo26m.pt") in composition_root.py resolves
#    to /app/yolo26n.pt and /app/yolo26m.pt — zero network call at runtime.
#
#    Both weight files must be present in the build context before running
#    'docker build'.  Download them once with:
#      python -c "from ultralytics import YOLO; YOLO('yolo26n.pt'); YOLO('yolo26m.pt')"
# ---------------------------------------------------------------------------
COPY yolo26n.pt yolo26m.pt /app/

# ---------------------------------------------------------------------------
# 6. Create the outputs directory and set environment.
# ---------------------------------------------------------------------------
RUN mkdir -p /app/outputs /app/.ultralytics

# Tell ultralytics where to look for cached settings so it never writes to
# an unexpected path inside the container.
ENV ULTRALYTICS_CONFIG_DIR=/app/.ultralytics \
    YOLO_CONFIG_DIR=/app/.ultralytics \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

# ---------------------------------------------------------------------------
# 7. Default command — runs the CI config; override at docker compose / run time.
# ---------------------------------------------------------------------------
ENTRYPOINT ["python", "scripts/run_pipeline.py"]
CMD ["--config", "configs/ci.yaml"]
