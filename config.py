"""
Truth in a Blink — Configuration
=================================
Central configuration for all modules: paths, hyperparameters, thresholds.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# ─── Resolve workspace root (handles trailing-space directory name) ──────────
_WORKSPACE = Path(__file__).resolve().parent.parent          # …/AI /
PROJECT_ROOT = Path(__file__).resolve().parent                # …/TruthInABlink/

# ─── Dataset paths ───────────────────────────────────────────────────────────
FER2013_ROOT   = _WORKSPACE / "FER_2013"
RLDD_ROOT      = _WORKSPACE / "Real-life_Deception_Detection_2016"
RLDD_CLIPS     = RLDD_ROOT / "Clips"
RLDD_ANNOTATION = RLDD_ROOT / "Annotation" / "All_Gestures_Deceptive and Truthful.csv"

# ─── Checkpoint & output dirs ────────────────────────────────────────────────
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RECORDING_DIR  = PROJECT_ROOT / "recordings"
LOG_DIR        = PROJECT_ROOT / "logs"

for d in [CHECKPOINT_DIR, RECORDING_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class MacroStreamConfig:
    """Vision-Transformer-based macro (facial context) stream."""
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    embed_dim: int = 384           # ViT-Small dimension
    depth: int = 6                 # transformer blocks
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    num_fer_classes: int = 7       # FER2013: 7 emotion classes
    output_dim: int = 256          # final embedding size


@dataclass
class MicroStreamConfig:
    """Motion-analysis micro stream (optical flow + temporal transformer)."""
    flow_channels: int = 2         # (dx, dy) optical flow
    motion_descriptor_dim: int = 128
    seq_len: int = 16              # frames in the motion buffer
    embed_dim: int = 256
    depth: int = 4
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    output_dim: int = 256


@dataclass
class FusionConfig:
    """Attention-based dual-stream fusion."""
    macro_dim: int = 256
    micro_dim: int = 256
    hidden_dim: int = 256
    num_heads: int = 4
    dropout: float = 0.1


@dataclass
class ClassifierConfig:
    """Binary deception classifier head."""
    input_dim: int = 256
    hidden_dim: int = 128
    dropout: float = 0.3


@dataclass
class DecisionConfig:
    """Conservative 3-state decision thresholds."""
    high_threshold: float = 0.65   # p >= HIGH → DECEPTIVE
    low_threshold: float = 0.35    # p <= LOW  → TRUTHFUL
    # else → UNCERTAIN


@dataclass
class TrainingConfig:
    """Hyper-parameters shared across training stages."""
    # Stage 1 — FER2013 pretraining
    fer_epochs: int = 25
    fer_batch_size: int = 64
    fer_lr: float = 3e-4
    fer_weight_decay: float = 1e-4

    # Stage 2 — RLDD dual-stream training
    rldd_epochs: int = 40
    rldd_batch_size: int = 8
    rldd_lr: float = 1e-4
    rldd_weight_decay: float = 1e-4
    rldd_clip_frames: int = 64     # frames sampled per clip

    # General
    num_workers: int = 4
    seed: int = 42
    use_mps: bool = True           # Apple Metal acceleration


@dataclass
class InferenceConfig:
    """Real-time inference parameters."""
    motion_buffer_size: int = 16   # frames kept for micro-stream
    face_padding: float = 0.3      # fractional pad around detected face
    smoothing_alpha: float = 0.3   # EMA smoothing (0 = no smooth, 1 = instant)
    camera_index: int = 0
    target_fps: int = 30


@dataclass
class UIConfig:
    """Streamlit UI settings."""
    page_title: str = "Truth in a Blink"
    page_icon: str = "🔍"
    layout: str = "wide"


# ─── Convenience: single config namespace ────────────────────────────────────
@dataclass
class Config:
    macro: MacroStreamConfig = field(default_factory=MacroStreamConfig)
    micro: MicroStreamConfig = field(default_factory=MicroStreamConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    ui: UIConfig = field(default_factory=UIConfig)


# Default global config instance
cfg = Config()
