"""
Truth in a Blink — Temperature-Scaling Calibration
====================================================
Post-hoc calibration: learns a single temperature parameter T so that
    calibrated_prob = sigmoid(logit / T)
produces well-calibrated probabilities (reliability ≈ diagonal).

Also reports Expected Calibration Error (ECE) before and after calibration.

Usage
-----
    python -m evaluation.calibrate \
        --checkpoint checkpoints/dual_stream_best.pt \
        --splits data/splits.json --fold 0
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import cfg, RLDD_CLIPS, CHECKPOINT_DIR, LOG_DIR, PROJECT_ROOT
from models.dual_stream import DualStreamDeceptionDetector
from data.rldd_dataset import RLDDDataset
from utils.helpers import get_device, seed_everything


# ── Expected Calibration Error ────────────────────────────────────────────────

def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10,
) -> tuple[float, list[dict]]:
    """
    Compute ECE and per-bin statistics.

    Returns (ece, bins) where bins is a list of dicts with keys:
        bin_lower, bin_upper, avg_confidence, avg_accuracy, count, fraction
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    ece = 0.0
    n = len(probs)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs > lo) & (probs <= hi)
        count = int(mask.sum())
        if count == 0:
            bins.append({
                "bin_lower": round(float(lo), 2),
                "bin_upper": round(float(hi), 2),
                "avg_confidence": 0.0,
                "avg_accuracy": 0.0,
                "count": 0,
                "fraction": 0.0,
            })
            continue
        avg_conf = float(probs[mask].mean())
        avg_acc  = float(labels[mask].mean())
        ece += (count / n) * abs(avg_acc - avg_conf)
        bins.append({
            "bin_lower": round(float(lo), 2),
            "bin_upper": round(float(hi), 2),
            "avg_confidence": round(avg_conf, 4),
            "avg_accuracy": round(avg_acc, 4),
            "count": count,
            "fraction": round(count / n, 4),
        })
    return round(ece, 4), bins


# ── Temperature Scaling model ────────────────────────────────────────────────

class TemperatureScaler(nn.Module):
    """Learns a single temperature parameter."""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def calibrate(self, probs: torch.Tensor) -> torch.Tensor:
        """Convert raw probabilities → calibrated probabilities via logit → scale → sigmoid."""
        # Clamp to avoid log(0)
        eps = 1e-7
        probs = probs.clamp(eps, 1.0 - eps)
        logits = torch.log(probs / (1.0 - probs))  # inverse sigmoid
        scaled = self.forward(logits)
        return torch.sigmoid(scaled)


def _collect_probs_and_labels(model, dataset, device):
    """Run model on dataset, return (probs_tensor, labels_tensor)."""
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for idx in range(len(dataset)):
            face, flow, label = dataset[idx]
            face = face.unsqueeze(0).to(device)
            flow = flow.unsqueeze(0).to(device)
            prob, *_ = model(face, flow)
            probs.append(prob.item())
            labels.append(label)
    return torch.tensor(probs), torch.tensor(labels, dtype=torch.float32)


def _build_val_subset(full_ds: RLDDDataset, clip_ids: list[str]) -> Subset:
    stem_to_idx = {p.stem: i for i, (p, _) in enumerate(full_ds.samples)}
    indices = [stem_to_idx[c] for c in clip_ids if c in stem_to_idx]
    if not indices:
        raise ValueError("No matching clips in dataset for given split")
    return Subset(full_ds, indices)


# ── Fit temperature ──────────────────────────────────────────────────────────

def fit_temperature(
    raw_probs: torch.Tensor,
    labels: torch.Tensor,
    lr: float = 0.01,
    max_iter: int = 200,
) -> TemperatureScaler:
    """
    Optimise temperature T to minimise NLL on validation calibration set.
    """
    scaler = TemperatureScaler()
    criterion = nn.BCELoss()
    optimizer = optim.LBFGS([scaler.temperature], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        calibrated = scaler.calibrate(raw_probs)
        loss = criterion(calibrated, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return scaler


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Temperature-scaling calibration")
    parser.add_argument("--checkpoint", type=str,
                        default=str(CHECKPOINT_DIR / "dual_stream_best.pt"))
    parser.add_argument("--splits", type=str,
                        default=str(PROJECT_ROOT / "data" / "splits.json"))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output", type=str,
                        default=str(LOG_DIR / "calibration.json"))
    args = parser.parse_args()

    seed_everything(cfg.training.seed)
    device = get_device(cfg.training.use_mps)

    print("=" * 60)
    print("Temperature-Scaling Calibration")
    print("=" * 60)

    # ── Model ─────────────────────────────────────────────────────────
    model = DualStreamDeceptionDetector()
    model.macro_stream.to_feature_extractor()
    model = model.to(device)
    ckpt = Path(args.checkpoint)
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        print(f"[✓] Loaded checkpoint: {ckpt}")
    else:
        print(f"[!] No checkpoint. Using random weights.")

    # ── Validation set ────────────────────────────────────────────────
    full_ds = RLDDDataset(RLDD_CLIPS,
                          num_frames=cfg.training.rldd_clip_frames)
    with open(args.splits) as f:
        splits = json.load(f)
    fold_info = splits["folds"][args.fold]
    val_ds = _build_val_subset(full_ds, fold_info["val"])
    print(f"[fold {args.fold}] Calibration set: {len(val_ds)} clips")

    # ── Collect raw probabilities ─────────────────────────────────────
    raw_probs, labels = _collect_probs_and_labels(model, val_ds, device)

    # ── Pre-calibration ECE ───────────────────────────────────────────
    ece_before, bins_before = expected_calibration_error(
        raw_probs.numpy(), labels.numpy()
    )
    print(f"\nPre-calibration ECE:  {ece_before:.4f}")

    # ── Fit temperature ───────────────────────────────────────────────
    scaler = fit_temperature(raw_probs, labels)
    T = scaler.temperature.item()
    print(f"Learned temperature:  T = {T:.4f}")

    calibrated_probs = scaler.calibrate(raw_probs).detach().numpy()
    ece_after, bins_after = expected_calibration_error(
        calibrated_probs, labels.numpy()
    )
    print(f"Post-calibration ECE: {ece_after:.4f}")
    print(f"ECE improvement:      {ece_before - ece_after:+.4f}")

    # ── Save ──────────────────────────────────────────────────────────
    results = {
        "temperature": round(T, 4),
        "ece_before": ece_before,
        "ece_after": ece_after,
        "ece_improvement": round(ece_before - ece_after, 4),
        "bins_before": bins_before,
        "bins_after": bins_after,
        "n_samples": len(raw_probs),
        "fold": args.fold,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[✓] Calibration results saved → {out}")

    # ── Also save the scaler weights for inference use ────────────────
    scaler_path = CHECKPOINT_DIR / "temperature_scaler.pt"
    torch.save({"temperature": T}, scaler_path)
    print(f"[✓] Temperature scaler saved → {scaler_path}")


if __name__ == "__main__":
    main()
