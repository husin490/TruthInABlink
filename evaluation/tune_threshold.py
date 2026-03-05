"""
Truth in a Blink — Precision-First Threshold Tuning
=====================================================
Searches for the optimal decision threshold on validation data,
prioritising high-precision over recall.

Two modes:
  • best-f1       : threshold that maximises F1 (standard)
  • high-precision : threshold that keeps Precision ≥ 90% with best Recall

Usage
-----
    python -m evaluation.tune_threshold \
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
from torch.utils.data import Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import cfg, RLDD_CLIPS, CHECKPOINT_DIR, LOG_DIR, PROJECT_ROOT
from models.dual_stream import DualStreamDeceptionDetector
from data.rldd_dataset import RLDDDataset
from utils.helpers import get_device, seed_everything


# ── Helpers ───────────────────────────────────────────────────────────────────

def _collect_probabilities(
    model: DualStreamDeceptionDetector,
    dataset,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run model on every sample, return (probs, labels) arrays."""
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
    return np.array(probs), np.array(labels)


def _metrics_at(probs: np.ndarray, labels: np.ndarray, t: float):
    """Return (precision, recall, f1) at threshold t."""
    preds = (probs >= t).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def _build_val_subset(
    full_dataset: RLDDDataset,
    clip_ids: list[str],
) -> Subset:
    """Build a Subset from a list of clip_id strings (stems like 'trial_lie_001')."""
    stem_to_idx = {p.stem: i for i, (p, _) in enumerate(full_dataset.samples)}
    indices = [stem_to_idx[c] for c in clip_ids if c in stem_to_idx]
    if not indices:
        raise ValueError("No matching clips found in dataset for the given split")
    return Subset(full_dataset, indices)


# ── Main tuning logic ────────────────────────────────────────────────────────

def tune_threshold(
    model: DualStreamDeceptionDetector,
    val_dataset,
    device: torch.device,
    min_precision: float = 0.90,
    min_recall_floor: float = 0.40,
    steps: int = 200,
) -> dict:
    """
    Search thresholds and return:
      • best_f1_threshold
      • high_precision_threshold  (Prec ≥ min_precision, best Recall ≥ floor)
      • full sweep table
    """
    probs, labels = _collect_probabilities(model, val_dataset, device)

    thresholds = np.linspace(0.20, 0.85, steps)
    sweep = []
    best_f1, best_f1_t = 0.0, 0.5
    best_hp_rec, best_hp_t = 0.0, None  # high-precision pick

    for t in thresholds:
        prec, rec, f1 = _metrics_at(probs, labels, t)
        sweep.append({
            "threshold": round(float(t), 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        })
        if f1 > best_f1:
            best_f1, best_f1_t = f1, float(t)
        if prec >= min_precision and rec >= min_recall_floor:
            if rec > best_hp_rec:
                best_hp_rec, best_hp_t = rec, float(t)

    # If no threshold satisfies the precision constraint, pick the one
    # closest to min_precision from above
    if best_hp_t is None:
        candidates = [(s["threshold"], s["precision"])
                       for s in sweep if s["precision"] >= min_precision * 0.95]
        if candidates:
            best_hp_t = max(candidates, key=lambda x: x[1])[0]
        else:
            best_hp_t = best_f1_t  # fallback

    pf1 = _metrics_at(probs, labels, best_f1_t)
    php = _metrics_at(probs, labels, best_hp_t)

    return {
        "best_f1": {
            "threshold": round(best_f1_t, 4),
            "precision": round(pf1[0], 4),
            "recall": round(pf1[1], 4),
            "f1": round(pf1[2], 4),
        },
        "high_precision": {
            "threshold": round(best_hp_t, 4),
            "precision": round(php[0], 4),
            "recall": round(php[1], 4),
            "f1": round(php[2], 4),
            "target_precision": min_precision,
        },
        "sweep": sweep,
        "n_samples": len(probs),
        "n_positive": int(labels.sum()),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Precision-first threshold tuning")
    parser.add_argument("--checkpoint", type=str,
                        default=str(CHECKPOINT_DIR / "dual_stream_best.pt"))
    parser.add_argument("--splits", type=str,
                        default=str(PROJECT_ROOT / "data" / "splits.json"))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--min-precision", type=float, default=0.90,
                        help="Minimum precision for high-precision mode")
    parser.add_argument("--output", type=str,
                        default=str(LOG_DIR / "tune_threshold.json"))
    args = parser.parse_args()

    seed_everything(cfg.training.seed)
    device = get_device(cfg.training.use_mps)

    print("=" * 60)
    print("Precision-First Threshold Tuning")
    print("=" * 60)

    # ── Load model ────────────────────────────────────────────────────
    model = DualStreamDeceptionDetector()
    model.macro_stream.to_feature_extractor()
    model = model.to(device)
    ckpt = Path(args.checkpoint)
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        print(f"[✓] Loaded checkpoint: {ckpt}")
    else:
        print(f"[!] No checkpoint at {ckpt}. Using random weights.")

    # ── Load validation split ─────────────────────────────────────────
    full_ds = RLDDDataset(RLDD_CLIPS,
                          num_frames=cfg.training.rldd_clip_frames)

    with open(args.splits) as f:
        splits = json.load(f)
    fold_info = splits["folds"][args.fold]
    val_ids = fold_info["val"]
    val_ds = _build_val_subset(full_ds, val_ids)
    print(f"[fold {args.fold}] Validation: {len(val_ds)} clips")

    # ── Tune ──────────────────────────────────────────────────────────
    results = tune_threshold(
        model, val_ds, device,
        min_precision=args.min_precision,
    )

    bf1 = results["best_f1"]
    hp  = results["high_precision"]
    print(f"\n{'─' * 50}")
    print(f"Best-F1 threshold : {bf1['threshold']:.4f}  "
          f"(P={bf1['precision']:.3f}  R={bf1['recall']:.3f}  F1={bf1['f1']:.3f})")
    print(f"High-Prec threshold: {hp['threshold']:.4f}  "
          f"(P={hp['precision']:.3f}  R={hp['recall']:.3f}  F1={hp['f1']:.3f})")

    # ── Save ──────────────────────────────────────────────────────────
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[✓] Results saved → {out}")


if __name__ == "__main__":
    main()
