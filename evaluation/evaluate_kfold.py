"""
Truth in a Blink — K-Fold Cross-Validation Evaluation
=======================================================
Runs the dual-stream model on every fold defined in data/splits.json
and reports mean ± std across folds.  Saves per-fold results and an
aggregate eval_kfold_results.json.

Usage
-----
    python -m evaluation.evaluate_kfold
    python -m evaluation.evaluate_kfold --checkpoint checkpoints/dual_stream_best.pt
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import cfg, RLDD_CLIPS, CHECKPOINT_DIR, LOG_DIR, PROJECT_ROOT
from models.dual_stream import DualStreamDeceptionDetector
from data.rldd_dataset import RLDDDataset
from evaluation.evaluate import evaluate_model
from utils.helpers import get_device, seed_everything


def _clip_id_to_path(clip_id: str, clips_dir: Path) -> Path:
    """Convert a clip_id like 'trial_lie_001' to its full path."""
    if "lie" in clip_id:
        return clips_dir / "Deceptive" / f"{clip_id}.mp4"
    else:
        return clips_dir / "Truthful" / f"{clip_id}.mp4"


def _build_subset_dataset(
    clip_ids: list[str],
    clips_dir: Path,
    full_dataset: RLDDDataset,
) -> torch.utils.data.Subset:
    """
    Build a Subset of the full dataset containing only the given clip_ids.
    """
    # Build a map: clip_stem → index in full_dataset
    stem_to_idx = {}
    for idx, (path, _label) in enumerate(full_dataset.samples):
        stem_to_idx[path.stem] = idx

    indices = []
    for cid in clip_ids:
        if cid in stem_to_idx:
            indices.append(stem_to_idx[cid])
    return torch.utils.data.Subset(full_dataset, indices)


def evaluate_fold(
    model: DualStreamDeceptionDetector,
    fold: dict,
    full_dataset: RLDDDataset,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    """Evaluate model on a single fold's test set."""
    test_subset = _build_subset_dataset(fold["test"], RLDD_CLIPS, full_dataset)
    if len(test_subset) == 0:
        print(f"  [warn] Fold {fold['fold_idx']}: no test clips found on disk")
        return {}

    results = evaluate_model(model, test_subset, device, threshold)
    results["fold_idx"] = fold["fold_idx"]
    results["n_test"] = len(test_subset)
    return results


def main():
    parser = argparse.ArgumentParser(description="K-fold CV evaluation on RLDD")
    parser.add_argument("--checkpoint", type=str,
                        default=str(CHECKPOINT_DIR / "dual_stream_best.pt"))
    parser.add_argument("--splits", type=str,
                        default=str(PROJECT_ROOT / "data" / "splits.json"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=str,
                        default=str(LOG_DIR / "eval_kfold_results.json"))
    args = parser.parse_args()

    seed_everything(cfg.training.seed)
    device = get_device(cfg.training.use_mps)

    print("=" * 60)
    print("K-Fold Cross-Validation — Leakage-Proof Evaluation")
    print("=" * 60)

    # Load splits
    with open(args.splits) as f:
        splits = json.load(f)

    meta = splits["metadata"]
    print(f"Splits: {meta['n_folds']} folds, {meta['n_clips']} clips, "
          f"{meta['n_subjects']} subjects (seed={meta['seed']})")
    print(f"Method: {meta['split_method']}")

    # Load model
    model = DualStreamDeceptionDetector()
    model.macro_stream.to_feature_extractor()
    model = model.to(device)

    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        print(f"[OK] Loaded checkpoint: {ckpt_path.name}")
    else:
        print(f"[!] No checkpoint at {ckpt_path}. Using random weights.")

    # Load full dataset once
    full_dataset = RLDDDataset(
        RLDD_CLIPS,
        num_frames=cfg.training.rldd_clip_frames,
    )

    # Evaluate each fold
    fold_results = []
    for fold in splits["folds"]:
        fidx = fold["fold_idx"]
        print(f"\n--- Fold {fidx} ---")
        result = evaluate_fold(model, fold, full_dataset, device, args.threshold)
        if result:
            fold_results.append(result)
            print(f"  Acc={result['accuracy']:.3f}  P={result['precision']:.3f}  "
                  f"R={result['recall']:.3f}  F1={result['f1']:.3f}  "
                  f"AUC={result['auc']:.3f}  (n={result['n_test']})")

    if not fold_results:
        print("No folds evaluated. Check splits.json and clips on disk.")
        return

    # Aggregate
    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    agg = {}
    for m in metrics:
        vals = [r[m] for r in fold_results if m in r]
        agg[m] = {"mean": round(float(np.mean(vals)), 4),
                   "std": round(float(np.std(vals)), 4),
                   "per_fold": [round(v, 4) for v in vals]}

    print("\n" + "=" * 60)
    print("  K-FOLD AGGREGATE RESULTS (mean ± std)")
    print("=" * 60)
    for m in metrics:
        print(f"  {m:12s}: {agg[m]['mean']:.4f} ± {agg[m]['std']:.4f}")

    # Save
    output = {
        "metadata": meta,
        "threshold": args.threshold,
        "aggregate": agg,
        "per_fold": fold_results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n[OK] Results saved → {out_path}")


if __name__ == "__main__":
    main()
