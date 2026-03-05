# Truth in a Blink

## Visual Sensor-Based Lie Detection Through Facial Expression Analysis

A research-quality deception detection system running locally on Apple Silicon with PyTorch MPS acceleration.

> **80.2% accuracy, 0.854 AUC-ROC** on the RLDD 2016 benchmark (121 real-life video clips)

---

## CP2+ Enhancements

| Update | What Changed |
|--------|-------------|
| **A. Leakage-Proof Splits** | Subject-wise 5-fold CV; no participant appears in both train & test |
| **B. MediaPipe Face Detection** | 3-tier fallback: MediaPipe → Haar cascade → centre-crop |
| **C. Precision-First Threshold** | Automated search for ≥ 90 % precision operating point + temperature calibration |
| **D. Normalised Optical Flow** | Zero-mean / unit-variance with 99th-percentile outlier clipping; optional TV-L1 |
| **E. Ablation Study Scripts** | Macro-only, micro-only, concat-fusion, no-pretrain — all across K folds |
| **F. Dashboard Upgrades** | High Precision Mode toggle, UNCERTAIN reason panel, thread-safe state |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    DUAL-STREAM PIPELINE                         │
│                                                                 │
│  ┌──────────────┐                    ┌──────────────┐          │
│  │ Face Image   │                    │ Video Frames │          │
│  │ (224×224 RGB)│                    │ (sequence)   │          │
│  └──────┬───────┘                    └──────┬───────┘          │
│         │                                   │                   │
│         ▼                                   ▼                   │
│  ┌──────────────┐                    ┌──────────────┐          │
│  │ MACRO STREAM │                    │ MICRO STREAM │          │
│  │ ViT-Small    │                    │ Optical Flow │          │
│  │ (FER2013     │                    │ + CNN Desc.  │          │
│  │  pretrained) │                    │ + Temporal   │          │
│  │              │                    │   Transformer│          │
│  └──────┬───────┘                    └──────┬───────┘          │
│         │ 256-dim                           │ 256-dim          │
│         └──────────────┬────────────────────┘                  │
│                        ▼                                        │
│               ┌────────────────┐                               │
│               │ ATTENTION      │                               │
│               │ FUSION MODULE  │──→ w_macro + w_micro = 1      │
│               └────────┬───────┘                               │
│                        │ 256-dim                                │
│                        ▼                                        │
│               ┌────────────────┐                               │
│               │  CLASSIFIER    │                               │
│               │  (MLP → σ)     │──→ P(deception) ∈ [0, 1]     │
│               └────────────────┘                               │
│                                                                 │
│  Decision: DECEPTIVE | UNCERTAIN | TRUTHFUL                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
TruthInABlink/
├── config.py                    # Central configuration
├── requirements.txt             # Python dependencies
├── setup.sh                     # One-click macOS setup
├── README.md                    # This file
│
├── models/                      # Neural network architectures
│   ├── macro_stream.py          # Vision Transformer (ViT)
│   ├── micro_stream.py          # Optical flow + temporal transformer
│   ├── fusion.py                # Attention-based stream fusion
│   ├── classifier.py            # Binary deception classifier
│   └── dual_stream.py           # Full end-to-end model
│
├── data/                        # Data loading & preprocessing
│   ├── fer_dataset.py           # FER2013 dataloader
│   ├── rldd_dataset.py          # RLDD 2016 dataloader
│   └── splits.json              # Subject-wise 5-fold CV splits [CP2+]
│
├── training/                    # Training pipeline
│   ├── train_fer.py             # Stage 1: FER2013 pretraining
│   ├── train_rldd.py            # Stage 2: RLDD dual-stream
│   └── train_custom.py          # Stage 3: Fine-tune on custom data
│
├── evaluation/                  # Evaluation tools
│   ├── evaluate.py              # Metrics, confusion matrix, threshold sweep
│   ├── evaluate_kfold.py        # K-fold cross-validation runner [CP2+]
│   ├── tune_threshold.py        # Precision-first threshold search [CP2+]
│   ├── calibrate.py             # Temperature-scaling calibration [CP2+]
│   └── ablate.py                # Component ablation study [CP2+]
│
├── inference/                   # Real-time inference
│   └── realtime_engine.py       # Webcam pipeline + CLI demo
│
├── ui/                          # User interface
│   └── dashboard.py             # Streamlit dashboard (v2.0 with HP mode) [CP2+]
│
├── utils/                       # Utilities
│   ├── helpers.py               # Device, seeding, EMA, decision logic
│   ├── face_crop.py             # MediaPipe → Haar → centre-crop [CP2+]
│   └── optical_flow.py          # Normalised flow + TV-L1 option [CP2+]
│
├── tools/                       # Pipeline tooling
│   └── build_splits.py          # Generate subject-wise K-fold splits [CP2+]
│
├── evaluation_report.ipynb      # Full evaluation notebook
├── checkpoints/                 # Saved model weights
├── recordings/                  # User-recorded videos
└── logs/                        # Training & session logs
```

---

## Quick Start

### 1. Environment Setup

```bash
cd TruthInABlink
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

### 2. Train the Model

**Stage 1 — Pretrain macro stream on FER2013:**
```bash
python -m training.train_fer
```

**Stage 2 — Train dual-stream on RLDD:**
```bash
python -m training.train_rldd
```

**Stage 2 (with macro fine-tuning):**
```bash
python -m training.train_rldd --unfreeze_macro
```

### 3. Evaluate

```bash
# Standard evaluation
python -m evaluation.evaluate --sweep

# K-fold cross-validation (leakage-proof) [CP2+]
python -m evaluation.evaluate_kfold --splits data/splits.json

# Precision-first threshold tuning [CP2+]
python -m evaluation.tune_threshold --splits data/splits.json --fold 0

# Temperature calibration [CP2+]
python -m evaluation.calibrate --splits data/splits.json --fold 0

# Ablation study [CP2+]
python -m evaluation.ablate --splits data/splits.json
```

### 4. Run Real-Time Detection

**Streamlit Dashboard (recommended):**
```bash
streamlit run ui/dashboard.py
```

**OpenCV CLI Demo:**
```bash
python -m inference.realtime_engine
```

---

## Training Pipeline

| Stage | Script | Dataset | Purpose |
|-------|--------|---------|---------|
| 1 | `training/train_fer.py` | FER2013 (35K images) | Pretrain ViT on facial expressions |
| 2 | `training/train_rldd.py` | RLDD 2016 (121 clips) | Train full dual-stream + fusion |
| 3 | `training/train_custom.py` | Custom recordings | Fine-tune on user data |

---

## Decision Strategy

Conservative 3-state output with adjustable thresholds:

| Condition | Output |
|-----------|--------|
| P(deception) ≥ 0.65 | **DECEPTIVE** 🔴 |
| P(deception) ≤ 0.35 | **TRUTHFUL** 🟢 |
| Otherwise | **UNCERTAIN** 🟡 |

Thresholds are adjustable via the Settings panel or `config.py`.

---

## Streamlit Dashboard Features

### 🎥 Live Detection
- Real-time webcam feed with face detection overlay
- Deception probability gauge
- Colour-coded status indicator (red/yellow/green)
- Macro vs Micro contribution display
- FPS counter and buffer status

### 📊 Visualisation
- Probability-over-time chart
- Fusion weight trends
- Session statistics (mean, std, min, max)
- Decision distribution histogram

### 🎬 Recording & Dataset Tools
- Record video clips with labels
- Save MP4 locally with auto-logging to CSV
- Capture face snapshots
- Export session logs as JSON

### 📈 Evaluation
- Run full RLDD evaluation from the UI
- Accuracy, Precision, Recall, F1, AUC
- Confusion matrix display
- Threshold sweep comparison chart
- Per-clip results table

### ⚙️ Settings
- Decision thresholds
- **🎯 High Precision Mode** — one-toggle to widen the UNCERTAIN zone (≥ 0.80 / ≤ 0.20) for fewer false positives [CP2+]
- EMA smoothing strength
- Motion buffer size
- Face crop padding
- **Camera selection** — auto-detects built-in, external, and iPhone (Continuity Camera) cameras with resolution display
- Model checkpoint selection

### 🔍 UNCERTAIN Reason Panel [CP2+]
When the verdict is UNCERTAIN, a contextual panel explains *why*:
- Distance to both thresholds and which direction the signal leans
- Whether face detection is failing
- Motion buffer fill status
- Whether High Precision Mode is active

---

## Hardware Requirements

| Component | Requirement |
|-----------|-------------|
| **CPU/GPU** | Apple Silicon (M1/M2/M3) or NVIDIA GPU |
| **RAM** | 16 GB minimum, 48 GB recommended |
| **OS** | macOS 13+ (Ventura or later) |
| **Python** | 3.10+ |
| **Acceleration** | MPS (Metal) on Apple Silicon |

The system runs **entirely offline** — no internet connection required after setup.

---

## Model Details

### Macro Stream — Vision Transformer
- **Architecture:** ViT-Small (6 blocks, 6 heads, dim 384)
- **Input:** 224×224 RGB face crop
- **Pretraining:** FER2013 (7-class emotion recognition)
- **Output:** 256-dimensional embedding

### Micro Stream — Motion Transformer
- **Flow:** Farneback dense optical flow
- **Descriptor:** Lightweight 3-layer CNN per flow field
- **Temporal:** 4-block transformer encoder
- **Output:** 256-dimensional embedding

### Fusion Module
- **Type:** Gated attention fusion with cross-stream interaction
- **Constraint:** w_macro + w_micro = 1 (softmax)
- **Output:** 256-dimensional fused representation

---

## Evaluation Results

| Metric | Value |
|--------|-------|
| **Accuracy** | 80.2% |
| **Precision** | 81.4% |
| **Recall** | 78.7% |
| **F1 Score** | 80.0% |
| **AUC-ROC** | 85.4% |
| **Specificity** | 81.7% |

Results above are from a single hold-out split. CP2+ adds subject-wise 5-fold
cross-validation (`evaluation/evaluate_kfold.py`), precision-first threshold
tuning (`evaluation/tune_threshold.py`), temperature calibration reducing
Expected Calibration Error (`evaluation/calibrate.py`), and a component
ablation study (`evaluation/ablate.py`). Run the k-fold pipeline and check
`logs/` for per-fold JSON reports.

See `evaluation_report.ipynb` for the full analysis with confusion matrices, ROC curves, training history, threshold sweeps, fusion weight analysis, and per-clip results.

---

## Configuration

All hyperparameters are centralised in `config.py`:

```python
from config import cfg

# Modify thresholds
cfg.decision.high_threshold = 0.70
cfg.decision.low_threshold  = 0.30

# Modify training parameters
cfg.training.fer_epochs     = 30
cfg.training.rldd_epochs    = 50
cfg.training.rldd_batch_size = 4
```

