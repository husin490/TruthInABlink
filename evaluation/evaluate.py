"""
Truth in a Blink — Evaluation Tools
=====================================
Comprehensive evaluation on the RLDD dataset:
  • Accuracy, Precision, Recall, F1
  • Confusion matrix
  • Threshold sweep analysis
  • Per-clip results export

Usage
-----
    python -m evaluation.evaluate --checkpoint checkpoints/dual_stream_best.pt
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import cfg, RLDD_CLIPS, CHECKPOINT_DIR, LOG_DIR
from models.dual_stream import DualStreamDeceptionDetector
from data.rldd_dataset import RLDDDataset
from utils.helpers import get_device, seed_everything, classify_deception


@torch.no_grad()
def evaluate_model(
    model: DualStreamDeceptionDetector,
    dataset: RLDDDataset,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    """
    Run the model on full dataset and collect metrics.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, auc, confusion_matrix,
                    per_clip results, all_probs, all_labels
    """
    model.eval()
    all_probs = []
    all_labels = []
    all_w_macro = []
    all_w_micro = []
    per_clip = []

    for idx in range(len(dataset)):
        face, flow, label = dataset[idx]

        face = face.unsqueeze(0).to(device)           # (1, 3, 224, 224)
        flow = flow.unsqueeze(0).to(device)            # (1, T, 2, H, W)

        prob, w_macro, w_micro, _, _ = model(face, flow)

        p = prob.item()
        wm = w_macro.item()
        wc = w_micro.item()

        all_probs.append(p)
        all_labels.append(label)
        all_w_macro.append(wm)
        all_w_micro.append(wc)

        # Handle both full Dataset (has .samples) and Subset (has .dataset.samples + .indices)
        if hasattr(dataset, "samples"):
            clip_name = dataset.samples[idx][0].name
        elif hasattr(dataset, "dataset") and hasattr(dataset.dataset, "samples"):
            real_idx = dataset.indices[idx]
            clip_name = dataset.dataset.samples[real_idx][0].name
        else:
            clip_name = f"clip_{idx}"
        decision = classify_deception(p, cfg.decision.high_threshold,
                                      cfg.decision.low_threshold)
        per_clip.append({
            "clip": str(clip_name),
            "label": "deceptive" if label == 1 else "truthful",
            "prob": round(p, 4),
            "prediction": "deceptive" if p >= threshold else "truthful",
            "decision_3state": decision,
            "w_macro": round(wm, 4),
            "w_micro": round(wc, 4),
        })

    # ── Compute metrics ──────────────────────────────────────────────────
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs >= threshold).astype(int)

    acc  = accuracy_score(all_labels, preds)
    prec = precision_score(all_labels, preds, zero_division=0)
    rec  = recall_score(all_labels, preds, zero_division=0)
    f1   = f1_score(all_labels, preds, zero_division=0)
    cm   = confusion_matrix(all_labels, preds).tolist()

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    report = classification_report(all_labels, preds,
                                   target_names=["Truthful", "Deceptive"],
                                   zero_division=0)

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "auc": round(auc, 4),
        "confusion_matrix": cm,
        "classification_report": report,
        "per_clip": per_clip,
        "all_probs": all_probs.tolist(),
        "all_labels": all_labels.tolist(),
        "threshold": threshold,
        "mean_w_macro": round(np.mean(all_w_macro), 4),
        "mean_w_micro": round(np.mean(all_w_micro), 4),
    }


def threshold_sweep(
    model: DualStreamDeceptionDetector,
    dataset: RLDDDataset,
    device: torch.device,
    thresholds: list[float] | None = None,
) -> list[dict]:
    """
    Evaluate the model across different decision thresholds.
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.3, 0.75, 0.05)]

    results = []
    for t in thresholds:
        r = evaluate_model(model, dataset, device, threshold=t)
        results.append({
            "threshold": t,
            "accuracy": r["accuracy"],
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1"],
            "auc": r["auc"],
        })
        print(f"  Threshold {t:.2f} → Acc {r['accuracy']:.3f}  "
              f"P {r['precision']:.3f}  R {r['recall']:.3f}  "
              f"F1 {r['f1']:.3f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate dual-stream model on RLDD")
    parser.add_argument("--checkpoint", type=str,
                        default=str(CHECKPOINT_DIR / "dual_stream_best.pt"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sweep", action="store_true",
                        help="Run threshold sweep analysis")
    parser.add_argument("--output", type=str, default=str(LOG_DIR / "eval_results.json"))
    args = parser.parse_args()

    seed_everything(cfg.training.seed)
    device = get_device(cfg.training.use_mps)

    print("=" * 60)
    print("Evaluation — Dual-Stream Deception Detector")
    print("=" * 60)

    # ── Load model ───────────────────────────────────────────────────────
    model = DualStreamDeceptionDetector()
    model.macro_stream.to_feature_extractor()          # strip fer_head before loading
    model = model.to(device)
    ckpt_path = Path(args.checkpoint)

    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        print(f"[✓] Loaded checkpoint: {ckpt_path}")
        if "val_acc" in state:
            print(f"    Checkpoint val accuracy: {state['val_acc']:.1f}%")
    else:
        print(f"[!] No checkpoint at {ckpt_path}. Using random weights.")

    # ── Load dataset ─────────────────────────────────────────────────────
    dataset = RLDDDataset(
        RLDD_CLIPS,
        num_frames=cfg.training.rldd_clip_frames,
    )

    # ── Evaluate ─────────────────────────────────────────────────────────
    print(f"\nEvaluating with threshold = {args.threshold:.2f} …")
    results = evaluate_model(model, dataset, device, args.threshold)

    print(f"\n{'─' * 40}")
    print(f"Accuracy  : {results['accuracy']:.4f}")
    print(f"Precision : {results['precision']:.4f}")
    print(f"Recall    : {results['recall']:.4f}")
    print(f"F1 Score  : {results['f1']:.4f}")
    print(f"AUC       : {results['auc']:.4f}")
    print(f"Mean w_macro: {results['mean_w_macro']:.4f}")
    print(f"Mean w_micro: {results['mean_w_micro']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={results['confusion_matrix'][0][0]}  FP={results['confusion_matrix'][0][1]}")
    print(f"  FN={results['confusion_matrix'][1][0]}  TP={results['confusion_matrix'][1][1]}")
    print(f"\n{results['classification_report']}")

    # ── Threshold sweep ──────────────────────────────────────────────────
    if args.sweep:
        print("\n" + "=" * 60)
        print("Threshold Sweep Analysis")
        print("=" * 60)
        sweep_results = threshold_sweep(model, dataset, device)
        results["threshold_sweep"] = sweep_results

    # ── Save ─────────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[✓] Results saved → {output_path}")


if __name__ == "__main__":
    main()
