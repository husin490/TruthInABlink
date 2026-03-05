"""
Truth in a Blink — Stage 1: Pretrain Macro Stream on FER2013
=============================================================
Trains the Vision Transformer backbone on 7-class emotion recognition.
After training the `fer_head` is saved alongside the backbone so that
Stage 2 can load just the feature extractor.

Usage
-----
    python -m training.train_fer            # train from scratch
    python -m training.train_fer --resume   # resume from last checkpoint

Outputs
-------
    checkpoints/macro_fer_best.pt
"""

import argparse
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import cfg, FER2013_ROOT, CHECKPOINT_DIR, LOG_DIR
from models.macro_stream import MacroStreamViT
from data.fer_dataset import get_fer_dataloaders
from utils.helpers import get_device, seed_everything, save_checkpoint, setup_logger


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, logger):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        logits = model(images)                         # (B, 7)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = logits.max(dim=1)
        correct += preds.eq(labels).sum().item()
        total += images.size(0)

        if (batch_idx + 1) % 50 == 0:
            logger.info(
                f"  Epoch {epoch} | Batch {batch_idx + 1}/{len(loader)} | "
                f"Loss {loss.item():.4f} | Acc {100. * correct / total:.1f}%"
            )

    return running_loss / total, 100. * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = logits.max(dim=1)
        correct += preds.eq(labels).sum().item()
        total += images.size(0)

    return running_loss / total, 100. * correct / total


def main():
    # ── Setup ────────────────────────────────────────────────────────────
    seed_everything(cfg.training.seed)
    device = get_device(cfg.training.use_mps)
    logger = setup_logger("train_fer", LOG_DIR)

    mc = cfg.macro
    tc = cfg.training

    logger.info("=" * 60)
    logger.info("Stage 1 — FER2013 Macro Stream Pretraining")
    logger.info("=" * 60)

    # ── Data ─────────────────────────────────────────────────────────────
    train_loader, test_loader = get_fer_dataloaders(
        FER2013_ROOT,
        batch_size=tc.fer_batch_size,
        image_size=mc.image_size,
        num_workers=tc.num_workers,
    )

    # ── Model ────────────────────────────────────────────────────────────
    model = MacroStreamViT(
        image_size=mc.image_size,
        patch_size=mc.patch_size,
        in_channels=mc.in_channels,
        embed_dim=mc.embed_dim,
        depth=mc.depth,
        num_heads=mc.num_heads,
        mlp_ratio=mc.mlp_ratio,
        dropout=mc.dropout,
        output_dim=mc.output_dim,
        num_fer_classes=mc.num_fer_classes,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # ── Optimiser & Scheduler ────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=tc.fer_lr,
                      weight_decay=tc.fer_weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=tc.fer_epochs,
                                  eta_min=1e-6)

    # ── Resume from checkpoint ───────────────────────────────────────────
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from macro_fer_best.pt")
    args = parser.parse_args()

    start_epoch = 1
    best_acc = 0.0

    if args.resume:
        ckpt_path = CHECKPOINT_DIR / "macro_fer_best.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_acc = ckpt.get("val_acc", 0.0)
            # Advance scheduler to the correct position
            for _ in range(start_epoch - 1):
                scheduler.step()
            logger.info(f"Resumed from epoch {start_epoch - 1} (best val acc: {best_acc:.1f}%)")
        else:
            logger.warning(f"No checkpoint found at {ckpt_path}, starting from scratch.")

    # ── Training loop ────────────────────────────────────────────────────

    for epoch in range(start_epoch, tc.fer_epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, logger
        )
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch}/{tc.fer_epochs} | "
            f"Train Loss {train_loss:.4f} Acc {train_acc:.1f}% | "
            f"Val Loss {val_loss:.4f} Acc {val_acc:.1f}% | "
            f"LR {scheduler.get_last_lr()[0]:.6f} | {elapsed:.1f}s"
        )

        # Save best
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
                "macro_fer_best.pt",
            )
            logger.info(f"  ★ New best val accuracy: {val_acc:.1f}%")

    # Save final
    save_checkpoint(
        {
            "epoch": tc.fer_epochs,
            "model_state_dict": model.state_dict(),
            "val_acc": val_acc,
        },
        CHECKPOINT_DIR,
        "macro_fer_final.pt",
    )

    logger.info(f"\nDone. Best validation accuracy: {best_acc:.1f}%")


if __name__ == "__main__":
    main()
