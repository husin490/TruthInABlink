"""
Truth in a Blink — Utility Helpers
====================================
Device selection, seeding, logging, and common functions.
"""

import os
import sys
import random
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import torch


# ─── Device Selection (MPS → CUDA → CPU) ─────────────────────────────────────

def get_device(prefer_mps: bool = True) -> torch.device:
    """
    Select the best available device.
    Priority: Apple Metal (MPS) → CUDA → CPU.

    Note: PyTorch 2.8 + some macOS versions may have MPS assertion failures.
    Set PYTORCH_ENABLE_MPS_FALLBACK=1 in your environment, or pass
    prefer_mps=False to force CPU.
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    if prefer_mps and torch.backends.mps.is_available():
        try:
            # Quick sanity check — some MPS builds crash on small tensors
            _t = torch.zeros(1, device="mps")
            del _t
            device = torch.device("mps")
            print(f"[Device] Apple Metal (MPS) — {device}")
        except Exception:
            device = torch.device("cpu")
            print("[Device] MPS test failed — falling back to CPU")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] CUDA — {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(f"[Device] CPU")
    return device


def seed_everything(seed: int = 42):
    """Reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # MPS doesn't support manual_seed_all yet, but torch.manual_seed covers it.
    os.environ["PYTHONHASHSEED"] = str(seed)


# ─── Logging ──────────────────────────────────────────────────────────────────

def setup_logger(name: str, log_dir: Path, level=logging.INFO) -> logging.Logger:
    """Create a file + console logger."""
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{name}_{timestamp}.log"

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
    ))

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def save_checkpoint(state: dict, path: Path, filename: str):
    """Save a training checkpoint."""
    path.mkdir(parents=True, exist_ok=True)
    filepath = path / filename
    torch.save(state, filepath)
    print(f"[✓] Checkpoint saved → {filepath}")


def load_checkpoint(path: Path, device: torch.device = torch.device("cpu")):
    """Load a training checkpoint."""
    state = torch.load(path, map_location=device, weights_only=False)
    print(f"[✓] Checkpoint loaded ← {path}")
    return state


# ─── EMA Smoothing ────────────────────────────────────────────────────────────

class EMASmooth:
    """Exponential moving average for real-time signal smoothing."""

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha       # 0 = no smoothing, 1 = instant update
        self.value: float | None = None

    def update(self, new_value: float) -> float:
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

    def reset(self):
        self.value = None


# ─── Decision logic ──────────────────────────────────────────────────────────

def classify_deception(prob: float, high: float = 0.65,
                       low: float = 0.35) -> str:
    """
    Three-state conservative decision.

    Returns
    -------
    "DECEPTIVE" | "TRUTHFUL" | "UNCERTAIN"
    """
    if prob >= high:
        return "DECEPTIVE"
    elif prob <= low:
        return "TRUTHFUL"
    else:
        return "UNCERTAIN"
