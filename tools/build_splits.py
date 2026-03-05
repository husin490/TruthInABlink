"""
Truth in a Blink — Build Leakage-Proof Subject-Wise Splits
============================================================
Parses the RLDD 2016 annotation CSV to discover subject → clip mappings,
then generates K-fold (default 5) subject-wise stratified splits where
*no subject appears in both train and val/test*.

Outputs
-------
    data/splits.json  —  reproducible folds with clip-level train/val/test

Usage
-----
    python -m tools.build_splits
    python -m tools.build_splits --n_folds 5 --seed 42
"""

import sys
import json
import hashlib
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RLDD_ROOT, RLDD_CLIPS, RLDD_ANNOTATION, PROJECT_ROOT


def _parse_annotation(csv_path: Path) -> dict[str, list[dict]]:
    """
    Read the RLDD annotation CSV and build a subject → clip mapping.
    Returns  { "Subject Name": [ {"clip_id": "trial_lie_001", "label": "deceptive"}, … ] }
    """
    df = pd.read_csv(csv_path)
    # Standardise column names (the CSV has varied header styles)
    df.columns = [c.strip() for c in df.columns]

    # Detect key columns
    id_col = [c for c in df.columns if "id" in c.lower() or "sample" in c.lower()][0]
    subj_col = [c for c in df.columns if "name" in c.lower() or "subject" in c.lower()][0]

    # The label is encoded in the clip filename  (trial_lie_* → deceptive, trial_truth_* → truthful)
    subject_clips: dict[str, list[dict]] = defaultdict(list)
    seen = set()

    for _, row in df.iterrows():
        clip_id = str(row[id_col]).strip()
        subject = str(row[subj_col]).strip()
        if clip_id in seen:
            continue
        seen.add(clip_id)
        label = "deceptive" if "lie" in clip_id.lower() else "truthful"
        subject_clips[subject].append({"clip_id": clip_id, "label": label})

    return dict(subject_clips)


def _verify_clips_on_disk(subject_clips: dict, clips_dir: Path) -> int:
    """Verify that clips referenced in the annotation actually exist on disk."""
    all_ids = {c["clip_id"] for clips in subject_clips.values() for c in clips}
    on_disk = set()
    for sub in ["Deceptive", "Truthful"]:
        d = clips_dir / sub
        if d.exists():
            for f in d.glob("*.mp4"):
                on_disk.add(f.stem)
    missing = all_ids - on_disk
    if missing:
        print(f"[warn] {len(missing)} clips in annotation but not on disk: {sorted(missing)[:5]}…")
    return len(all_ids & on_disk)


def build_kfold_splits(
    subject_clips: dict[str, list[dict]],
    n_folds: int = 5,
    seed: int = 42,
    val_ratio: float = 0.25,
) -> list[dict]:
    """
    Build K-fold subject-wise splits.

    For each fold, subjects are partitioned into train+val vs test.
    Within train+val, a further split separates val subjects.
    Ensures NO subject leaks across train / val / test within any fold.
    """
    rng = np.random.RandomState(seed)
    subjects = sorted(subject_clips.keys())

    # Compute per-subject deceptive ratio for stratification
    subj_dec_ratio = {}
    for s in subjects:
        clips = subject_clips[s]
        n_dec = sum(1 for c in clips if c["label"] == "deceptive")
        subj_dec_ratio[s] = n_dec / max(len(clips), 1)

    # Sort subjects by deceptive ratio then shuffle with seed for determinism
    subjects_sorted = sorted(subjects, key=lambda s: subj_dec_ratio[s])
    idx = np.arange(len(subjects_sorted))
    rng.shuffle(idx)
    subjects_shuffled = [subjects_sorted[i] for i in idx]

    # Assign subjects to folds round-robin style (stratified)
    fold_subjects: list[list[str]] = [[] for _ in range(n_folds)]
    for i, s in enumerate(subjects_shuffled):
        fold_subjects[i % n_folds].append(s)

    folds = []
    for fold_idx in range(n_folds):
        test_subjects = set(fold_subjects[fold_idx])
        remaining = [s for s in subjects if s not in test_subjects]

        # Split remaining into train vs val
        rng2 = np.random.RandomState(seed + fold_idx)
        rng2.shuffle(remaining)
        n_val = max(1, int(len(remaining) * val_ratio))
        val_subjects = set(remaining[:n_val])
        train_subjects = set(remaining[n_val:])

        def _collect(subj_set):
            ids = []
            for s in sorted(subj_set):
                for c in subject_clips[s]:
                    ids.append(c["clip_id"])
            return sorted(ids)

        train_ids = _collect(train_subjects)
        val_ids = _collect(val_subjects)
        test_ids = _collect(test_subjects)

        def _stats(ids):
            n_dec = sum(1 for i in ids if "lie" in i)
            n_tru = len(ids) - n_dec
            return {"deceptive": n_dec, "truthful": n_tru, "total": len(ids)}

        folds.append({
            "fold_idx": fold_idx,
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
            "stats": {
                "train": _stats(train_ids),
                "val": _stats(val_ids),
                "test": _stats(test_ids),
            },
        })

    return folds


def main():
    parser = argparse.ArgumentParser(description="Build leakage-proof splits for RLDD")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str,
                        default=str(PROJECT_ROOT / "data" / "splits.json"))
    args = parser.parse_args()

    print("=" * 60)
    print("Building leakage-proof subject-wise splits")
    print("=" * 60)

    # Parse annotation
    subject_clips = _parse_annotation(RLDD_ANNOTATION)
    n_subjects = len(subject_clips)
    n_clips = sum(len(v) for v in subject_clips.values())
    print(f"Parsed {n_clips} clips across {n_subjects} subjects")

    n_on_disk = _verify_clips_on_disk(subject_clips, RLDD_CLIPS)
    print(f"Verified {n_on_disk} clips on disk")

    # Build folds
    folds = build_kfold_splits(subject_clips, n_folds=args.n_folds, seed=args.seed)

    # Summary
    for f in folds:
        s = f["stats"]
        print(f"  Fold {f['fold_idx']}: train={s['train']['total']} "
              f"val={s['val']['total']} test={s['test']['total']}")

    # Output
    content_hash = hashlib.md5(json.dumps(folds, sort_keys=True).encode()).hexdigest()[:8]
    output = {
        "metadata": {
            "dataset": "RLDD-2016",
            "n_folds": args.n_folds,
            "seed": args.seed,
            "n_subjects": n_subjects,
            "n_clips": n_clips,
            "leakage_proof": True,
            "split_method": "subject-wise stratified group k-fold",
            "generator": "build_splits.py",
            "hash": content_hash,
        },
        "subject_to_clips": subject_clips,
        "folds": folds,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fp:
        json.dump(output, fp, indent=2)

    print(f"\n[OK] Saved → {out_path}  (hash {content_hash})")


if __name__ == "__main__":
    main()
