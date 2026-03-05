"""
Truth in a Blink — Ablation Study
===================================
Systematic ablation experiments to quantify each component's contribution:

  1. macro-only       – ViT macro-stream alone (no micro, no fusion)
  2. micro-only       – Micro-stream alone (no macro, no fusion)
  3. full-fusion      – Both streams + gated attention fusion (baseline)
  4. concat-fusion    – Replace gated attention with simple concatenation
  5. no-pretrain      – Full model, but macro NOT pre-trained on FER2013

Results are collected across all folds of the subject-wise split for statistical
robustness and written to a JSON report.

Usage
-----
    python -m evaluation.ablate \
        --checkpoint checkpoints/dual_stream_best.pt \
        --splits data/splits.json
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import json
import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import cfg, RLDD_CLIPS, CHECKPOINT_DIR, LOG_DIR, PROJECT_ROOT
from models.dual_stream import DualStreamDeceptionDetector
from models.macro_stream import MacroStreamViT
from models.micro_stream import MicroStreamTransformer
from models.classifier import DeceptionClassifier
from data.rldd_dataset import RLDDDataset
from utils.helpers import get_device, seed_everything


# ── Ablation model wrappers ──────────────────────────────────────────────────

class MacroOnlyModel(nn.Module):
    """Uses only the macro (ViT) stream → classifier. Ignores flow input."""

    def __init__(self, macro: MacroStreamViT, classifier_dim: int = 256):
        super().__init__()
        self.macro = macro
        self.classifier = DeceptionClassifier(
            input_dim=classifier_dim,
            hidden_dim=cfg.classifier.hidden_dim,
            dropout=cfg.classifier.dropout,
        )

    def forward(self, face, flow):
        emb = self.macro(face)                   # (B, 256)
        prob = self.classifier(emb)              # (B, 1)
        dummy = torch.tensor(1.0, device=face.device)
        return prob, dummy, torch.tensor(0.0, device=face.device), emb, emb


class MicroOnlyModel(nn.Module):
    """Uses only the micro (motion) stream → classifier. Ignores face input."""

    def __init__(self, micro: MicroStreamTransformer, classifier_dim: int = 256):
        super().__init__()
        self.micro = micro
        self.classifier = DeceptionClassifier(
            input_dim=classifier_dim,
            hidden_dim=cfg.classifier.hidden_dim,
            dropout=cfg.classifier.dropout,
        )

    def forward(self, face, flow):
        emb = self.micro(flow)                   # (B, 256)
        prob = self.classifier(emb)              # (B, 1)
        dummy = torch.tensor(0.0, device=face.device)
        return prob, dummy, torch.tensor(1.0, device=face.device), emb, emb


class ConcatFusionModel(nn.Module):
    """Replace gated attention fusion with simple concatenation + MLP."""

    def __init__(self, base_model: DualStreamDeceptionDetector):
        super().__init__()
        self.macro_stream = base_model.macro_stream
        self.micro_stream = base_model.micro_stream
        # Concat 256+256 → 256
        self.concat_proj = nn.Sequential(
            nn.Linear(cfg.fusion.macro_dim + cfg.fusion.micro_dim,
                      cfg.classifier.input_dim),
            nn.ReLU(),
            nn.Dropout(cfg.classifier.dropout),
        )
        self.classifier = DeceptionClassifier(
            input_dim=cfg.classifier.input_dim,
            hidden_dim=cfg.classifier.hidden_dim,
            dropout=cfg.classifier.dropout,
        )

    def forward(self, face, flow):
        macro_emb = self.macro_stream(face)
        micro_emb = self.micro_stream(flow)
        concat = torch.cat([macro_emb, micro_emb], dim=-1)
        fused = self.concat_proj(concat)
        prob = self.classifier(fused)
        return prob, torch.tensor(0.5), torch.tensor(0.5), macro_emb, micro_emb


# ── Evaluation helper ────────────────────────────────────────────────────────

@torch.no_grad()
def _evaluate_variant(model, dataset, device, threshold=0.5) -> dict:
    """Evaluate a model variant on a dataset, return metrics dict."""
    model.eval()
    probs, labels = [], []

    for idx in range(len(dataset)):
        face, flow, label = dataset[idx]
        face = face.unsqueeze(0).to(device)
        flow = flow.unsqueeze(0).to(device)
        prob, *_ = model(face, flow)
        probs.append(prob.item())
        labels.append(label)

    probs = np.array(probs)
    labels = np.array(labels)
    preds = (probs >= threshold).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    acc  = (tp + tn) / len(labels) if len(labels) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(labels, probs)
    except (ValueError, ImportError):
        auc = 0.0

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "auc": round(auc, 4),
        "n_samples": len(labels),
    }


# ── Build subset ─────────────────────────────────────────────────────────────

def _build_subset(full_ds: RLDDDataset, clip_ids: list[str]) -> Subset:
    stem_to_idx = {p.stem: i for i, (p, _) in enumerate(full_ds.samples)}
    indices = [stem_to_idx[c] for c in clip_ids if c in stem_to_idx]
    return Subset(full_ds, indices)


# ── Main ablation runner ─────────────────────────────────────────────────────

def run_ablation(
    checkpoint_path: Path,
    splits_path: Path,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    """
    Run all ablation variants across all folds.

    Returns
    -------
    dict with variant_name → {fold_results: [...], mean, std} for each metric
    """
    # Load splits
    with open(splits_path) as f:
        splits = json.load(f)

    # Full dataset (no augmentation)
    full_ds = RLDDDataset(RLDD_CLIPS,
                          num_frames=cfg.training.rldd_clip_frames)

    # Base model
    base_model = DualStreamDeceptionDetector()
    base_model.macro_stream.to_feature_extractor()
    base_model = base_model.to(device)

    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        base_model.load_state_dict(state["model_state_dict"])
        print(f"[✓] Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"[!] No checkpoint. Using random weights for ablation.")

    # Define variants
    variants = {}

    # 1. Full fusion (baseline)
    variants["full_fusion"] = base_model

    # 2. Macro-only
    macro_only = MacroOnlyModel(
        macro=copy.deepcopy(base_model.macro_stream),
    ).to(device)
    variants["macro_only"] = macro_only

    # 3. Micro-only
    micro_only = MicroOnlyModel(
        micro=copy.deepcopy(base_model.micro_stream),
    ).to(device)
    variants["micro_only"] = micro_only

    # 4. Concat fusion (re-initialised projection, keeps trained streams)
    concat_model = ConcatFusionModel(base_model).to(device)
    variants["concat_fusion"] = concat_model

    # 5. No pre-train (random macro)
    no_pretrain = DualStreamDeceptionDetector()
    no_pretrain.macro_stream.to_feature_extractor()
    no_pretrain = no_pretrain.to(device)
    # Keep micro weights from trained model, reset macro to random
    if checkpoint_path.exists():
        no_pretrain.micro_stream.load_state_dict(
            base_model.micro_stream.state_dict()
        )
        no_pretrain.fusion.load_state_dict(
            base_model.fusion.state_dict()
        )
        no_pretrain.classifier.load_state_dict(
            base_model.classifier.state_dict()
        )
    variants["no_pretrain"] = no_pretrain

    # Run across folds
    results = {v: {"fold_results": []} for v in variants}

    for fold_info in splits["folds"]:
        fold_idx = fold_info["fold_idx"]
        test_ds = _build_subset(full_ds, fold_info["test"])
        if len(test_ds) == 0:
            print(f"  [skip] Fold {fold_idx}: no test clips matched")
            continue

        print(f"\n── Fold {fold_idx} (test={len(test_ds)} clips) ──")

        for vname, vmodel in variants.items():
            metrics = _evaluate_variant(vmodel, test_ds, device, threshold)
            results[vname]["fold_results"].append(metrics)
            print(f"  {vname:20s}  Acc={metrics['accuracy']:.3f}  "
                  f"F1={metrics['f1']:.3f}  AUC={metrics['auc']:.3f}")

    # Aggregate statistics
    for vname in results:
        fold_r = results[vname]["fold_results"]
        if not fold_r:
            continue
        for metric in ["accuracy", "precision", "recall", "f1", "auc"]:
            vals = [r[metric] for r in fold_r]
            results[vname][f"mean_{metric}"] = round(float(np.mean(vals)), 4)
            results[vname][f"std_{metric}"] = round(float(np.std(vals)), 4)

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ablation study")
    parser.add_argument("--checkpoint", type=str,
                        default=str(CHECKPOINT_DIR / "dual_stream_best.pt"))
    parser.add_argument("--splits", type=str,
                        default=str(PROJECT_ROOT / "data" / "splits.json"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=str,
                        default=str(LOG_DIR / "ablation.json"))
    args = parser.parse_args()

    seed_everything(cfg.training.seed)
    device = get_device(cfg.training.use_mps)

    print("=" * 60)
    print("Ablation Study — Component Contribution Analysis")
    print("=" * 60)

    results = run_ablation(
        checkpoint_path=Path(args.checkpoint),
        splits_path=Path(args.splits),
        device=device,
        threshold=args.threshold,
    )

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Variant':20s}  {'Acc':>8s}  {'Prec':>8s}  {'Rec':>8s}  "
          f"{'F1':>8s}  {'AUC':>8s}")
    print("-" * 80)
    for vname, vdata in results.items():
        if f"mean_accuracy" in vdata:
            print(f"{vname:20s}  "
                  f"{vdata['mean_accuracy']:.3f}±{vdata['std_accuracy']:.3f}  "
                  f"{vdata['mean_precision']:.3f}±{vdata['std_precision']:.3f}  "
                  f"{vdata['mean_recall']:.3f}±{vdata['std_recall']:.3f}  "
                  f"{vdata['mean_f1']:.3f}±{vdata['std_f1']:.3f}  "
                  f"{vdata['mean_auc']:.3f}±{vdata['std_auc']:.3f}")
    print("=" * 80)

    # Save
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[✓] Ablation results saved → {out}")


if __name__ == "__main__":
    main()
