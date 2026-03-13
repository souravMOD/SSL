# SSL — Self-Supervised Learning Pretraining Framework

[![CI/CD](https://github.com/souravMOD/SSL/actions/workflows/ci.yml/badge.svg)](https://github.com/souravMOD/SSL/actions/workflows/ci.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](./Dockerfile)

A production-ready framework for **self-supervised pretraining** of YOLO models using [LightlyTrain](https://github.com/lightly-ai/lightly-train) and DINO distillation. Pretrain on unlabelled images, then fine-tune with [Ultralytics](https://docs.ultralytics.com/) for detection, segmentation, or classification.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    SSL Pretraining Pipeline                   │
│                                                              │
│  ┌─────────────┐    ┌──────────────────┐    ┌────────────┐  │
│  │  Unlabelled  │───▶│  DINO Distill.   │───▶│  Pretrained │  │
│  │  Images      │    │  (LightlyTrain)  │    │  Backbone   │  │
│  └─────────────┘    └──────────────────┘    └─────┬──────┘  │
│                                                    │         │
│                                                    ▼         │
│                                            ┌──────────────┐  │
│                                            │  Fine-tune    │  │
│                                            │  (Ultralytics)│  │
│                                            └──────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

**Stage 1 — Pretrain (this repo):** Learn visual representations from unlabelled data using DINO self-distillation with a YOLO11-L backbone.

**Stage 2 — Fine-tune:** Load the pretrained checkpoint into Ultralytics and train on your labelled downstream task.

---

## Quick Start

### Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA 12.x (for training)
- Docker (optional)

### Local Setup

```bash
# Clone
git clone https://github.com/souravMOD/SSL.git
cd SSL

# Install dependencies
pip install -r requirements.txt

# Run pretraining
SSL_DATA_PATH=/path/to/images python pretrain.py
```

### Docker (GPU)

```bash
# Build the training image
docker build --target train -t ssl-train .

# Run with GPU access — mount your dataset and output directory
docker run --gpus all \
  -v /path/to/images:/data \
  -v $(pwd)/out:/app/out \
  ssl-train
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SSL_DATA_PATH` | `/data` | Path to directory containing training images |
| `SSL_OUTPUT_DIR` | `out/my_experiment` | Directory for checkpoints and logs |
| `SSL_EPOCHS` | `100` | Number of pretraining epochs |
| `SSL_BATCH_SIZE` | `4` | Batch size (adjust for your GPU VRAM) |

---

## Configuration

The pretraining script uses DINO distillation with these defaults:

| Parameter | Value | Notes |
|---|---|---|
| Model | `ultralytics/yolo11l.yaml` | YOLO11 Large backbone |
| Method | `distillation` | DINO self-distillation |
| Image size | 640 × 640 | Matches YOLO input resolution |
| Min crop scale | 0.1 | Aggressive random-resize augmentation |
| Color jitter | Disabled | Set to `None` explicitly |

See commented sections in `pretrain.py` for additional augmentation options (Gaussian blur, solarize, local views).

---

## CI/CD Pipeline

The GitHub Actions pipeline (`.github/workflows/ci.yml`) runs:

| Stage | Trigger | What it does |
|---|---|---|
| **Lint** | Every push & PR | Runs `ruff` for code quality |
| **Validate** | After lint passes | Verifies imports and dependency resolution |
| **Docker Build & Push** | Push to `main` only | Builds GPU image, pushes to `ghcr.io` |

---

## Project Structure

```
SSL/
├── .github/workflows/ci.yml   # CI/CD pipeline
├── Dockerfile                  # Multi-stage (lint + GPU train)
├── .dockerignore               # Docker build exclusions
├── requirements.txt            # Pinned Python dependencies
├── pretrain.py                 # Main SSL pretraining script
├── out/                        # Training outputs (git-ignored)
│   └── my_experiment/
│       ├── events.out.tfevents.*   # TensorBoard logs
│       └── metrics.jsonl           # Per-batch metrics
├── LICENSE                     # GPL v3
└── README.md                   # This file
```

---

## License

This project is licensed under the **GNU General Public License v3.0** — see [LICENSE](./LICENSE) for details.
