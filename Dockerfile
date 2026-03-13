# ============================================================================
# Multi-stage Dockerfile for SSL (Self-Supervised Learning) Pretraining
# Supports both GPU training and CPU-only lint/test stages
# ============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Base image with dependencies (used by both CI and training)
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS base

WORKDIR /app

# System dependencies for OpenCV and image processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ---------------------------------------------------------------------------
# Stage 2: Lint & validation (used in CI — no GPU required)
# ---------------------------------------------------------------------------
FROM base AS lint

RUN pip install --no-cache-dir ruff
RUN ruff check . && echo "Lint passed"

# ---------------------------------------------------------------------------
# Stage 3: GPU Training image (production)
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS train

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3-pip \
        libgl1 \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default data and output mount points
VOLUME ["/data", "/app/out"]

ENV SSL_DATA_PATH=/data
ENV SSL_OUTPUT_DIR=/app/out/my_experiment
ENV SSL_EPOCHS=100
ENV SSL_BATCH_SIZE=4

ENTRYPOINT ["python3", "pretrain.py"]
