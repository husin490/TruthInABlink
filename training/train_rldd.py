"""
Truth in a Blink — Stage 2: Train Dual-Stream Model on RLDD
=============================================================
Loads the pretrained macro-stream backbone, freezes it (optionally),
and trains the full dual-stream pipeline on the Real-Life Deception
Detection 2016 dataset.

Usage
-----
    python -m training.train_rldd
    python -m training.train_rldd --unfreeze_macro   # fine-tune macro too

Outputs
-------
    checkpoints/dual_stream_best.pt
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import cfg, RLDD_CLIPS, CHECKPOINT_DIR, LOG_DIR
from models.dual_stream import DualStreamDeceptionDetector
from data.rldd_dataset import get_rldd_dataloaders
from utils.helpers import get_device, seed_everything, save_checkpoint, setup_logger


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, logger):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for batch_idx, (faces, flows, labels) in enumerate(loader):
        faces  = faces.to(device)
        flows  = flows.to(device)
        labels = labels.to(device).unsqueeze(1)           # (B, 1)

        prob, w_macro, w_micro, _, _ = model(faces, flows)
        loss = criterion(prob, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * faces.size(0)
        preds = (prob > 0.5).float()
        correct += preds.eq(labels).sum().item()
        total += faces.size(0)

        if (batch_idx + 1) % 5 == 0:
            logger.info(
                f"  Epoch {epoch} | Batch {batch_idx + 1}/{len(loader)} | "
                f"Loss {loss.item():.4f} | Acc {100. * correct / total:.1f}% | "
                f"w_macro {w_macro.mean().item():.3f} w_micro {w_micro.mean().item():.3f}"
            )

    return running_loss / max(total, 1), 100. * correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []

    for faces, flows, labels in loader:
        faces  = faces.to(device)
        flows  = flows.to(device)
        labels = labels.to(device).unsqueeze(1)

        prob, w_macro, w_micro, _, _ = model(faces, flows)
        loss = criterion(prob, labels)

        running_loss += loss.item() * faces.size(0)
        preds = (prob > 0.5).float()
        correct += preds.eq(labels).sum().item()
        total += faces.size(0)

        all_probs.extend(prob.cpu().numpy().flatten().tolist())
        all_labels.extend(labels.cpu().numpy().flatten().tolist())

    acc = 100. * correct / max(total, 1)
    loss = running_loss / max(total, 1)
    return loss, acc, all_probs, all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unfreeze_macro", action="store_true",
                        help="Fully fine-tune the macro stream alongside.")
    parser.add_argument("--unfreeze_top_blocks", type=int, default=0,
                        help="Unfreeze only the top N transformer blocks "
                             "in the macro backbone (0 = freeze all blocks, "
                             "keeps projection trainable).")
    parser.add_argument("--macro_ckpt", type=str,
                        default=str(CHECKPOINT_DIR / "macro_fer_best.pt"),
                        help="Path to pretrained macro checkpoint.")
    args = parser.parse_args()

    # ── Setup ────────────────────────────────────────────────────────────
    seed_everything(cfg.training.seed)
    device = get_device(cfg.training.use_mps)
    logger = setup_logger("train_rldd", LOG_DIR)

    tc = cfg.training

    logger.info("=" * 60)
    logger.info("Stage 2 — Dual-Stream Training on RLDD")
    logger.info("=" * 60)

    # ── Data ─────────────────────────────────────────────────────────────
    train_loader, val_loader = get_rldd_dataloaders(
        RLDD_CLIPS,
        num_frames=tc.rldd_clip_frames,
        batch_size=tc.rldd_batch_size,
        val_split=0.2,
        num_workers=tc.num_workers,
        seed=tc.seed,
    )

    # ── Model ────────────────────────────────────────────────────────────
    model = DualStreamDeceptionDetector().to(device)

    # Load FER2013-pretrained macro stream (strips fer_head automatically)
    macro_ckpt = Path(args.macro_ckpt)
    if macro_ckpt.exists():
        model.load_macro_pretrained(str(macro_ckpt))
        if args.unfreeze_macro:
            # Full fine-tuning: every macro parameter is trainable
            model.unfreeze_macro()
            logger.info("Macro stream: FULLY UNFROZEN (all layers trainable)")
        elif args.unfreeze_top_blocks > 0:
            # Gradual unfreezing: backbone frozen except top N blocks + projection
            model.freeze_macro(keep_projection_trainable=True)
            model.unfreeze_macro_top(args.unfreeze_top_blocks)
            logger.info(f"Macro stream: top {args.unfreeze_top_blocks} blocks "
                        f"+ projection trainable")
        else:
            # Default: freeze backbone, keep projection trainable
            model.freeze_macro(keep_projection_trainable=True)
            logger.info("Macro stream: backbone FROZEN, projection trainable")
    else:
        logger.warning(f"Macro checkpoint not found at {macro_ckpt}. "
                       "Training macro from scratch (no FER2013 pretraining).")

    total_params = sum(p.numel() for p in model.parameters())
    trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_params:,} | Trainable: {trainable:,}")

    # ── Optimiser ────────────────────────────────────────────────────────
    criterion = nn.BCELoss()
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=tc.rldd_lr,
        weight_decay=tc.rldd_weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=tc.rldd_epochs, eta_min=1e-6)

    # ── Training loop ────────────────────────────────────────────────────
    best_acc = 0.0

    for epoch in range(1, tc.rldd_epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, logger
        )
        val_loss, val_acc, val_probs, val_labels = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch}/{tc.rldd_epochs} | "
            f"Train Loss {train_loss:.4f} Acc {train_acc:.1f}% | "
            f"Val Loss {val_loss:.4f} Acc {val_acc:.1f}% | "
            f"LR {scheduler.get_last_lr()[0]:.6f} | {elapsed:.1f}s"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                },
                CHECKPOINT_DIR,
                "dual_stream_best.pt",
            )
            logger.info(f"  ★ New best val accuracy: {val_acc:.1f}%")

    # Save final
    save_checkpoint(
        {
            "epoch": tc.rldd_epochs,
            "model_state_dict": model.state_dict(),
            "val_acc": val_acc,
        },
        CHECKPOINT_DIR,
        "dual_stream_final.pt",
    )

    logger.info(f"\nDone. Best validation accuracy: {best_acc:.1f}%")


if __name__ == "__main__":
    main()
