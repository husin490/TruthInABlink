"""
Truth in a Blink — Stage 3: Fine-tune on User-Collected Data (Optional)
=========================================================================
Fine-tunes the frozen dual-stream model on custom video recordings
collected through the Streamlit dashboard.

Expects recorded MP4 files and a CSV log at:
    recordings/recordings_log.csv

Usage
-----
    python -m training.train_custom
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
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import cfg, RECORDING_DIR, CHECKPOINT_DIR, LOG_DIR
from models.dual_stream import DualStreamDeceptionDetector
from models.micro_stream import compute_flow_sequence
from data.rldd_dataset import detect_face, sample_frames, _face_transform
from utils.helpers import get_device, seed_everything, save_checkpoint, setup_logger


class CustomVideoDataset(Dataset):
    """Loads user-recorded videos from the recordings directory."""

    def __init__(self, recordings_dir: Path, num_frames: int = 64,
                 flow_size: tuple[int, int] = (56, 56)):
        super().__init__()
        self.num_frames = num_frames
        self.flow_size = flow_size
        self.samples: list[tuple[Path, int]] = []

        csv_path = recordings_dir / "recordings_log.csv"
        if not csv_path.exists():
            print("[Custom] No recordings_log.csv found.")
            return

        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            video_path = recordings_dir / row["filename"]
            label = 1 if row["label"] == "deceptive" else 0
            if video_path.exists():
                self.samples.append((video_path, label))

        print(f"[Custom] Loaded {len(self.samples)} user recordings.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]

        cap = cv2.VideoCapture(str(video_path))
        frames = sample_frames(cap, self.num_frames)
        cap.release()

        if len(frames) < 2:
            face_tensor = torch.zeros(3, 224, 224)
            flow_tensor = torch.zeros(max(1, self.num_frames - 1), 2, *self.flow_size)
            return face_tensor, flow_tensor, label

        mid = len(frames) // 2
        face_crop = detect_face(frames[mid])
        if face_crop is None:
            face_crop = cv2.cvtColor(frames[mid], cv2.COLOR_BGR2RGB)
        face_tensor = _face_transform(face_crop)

        grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        flow_np = compute_flow_sequence(grays, self.flow_size)
        flow_tensor = torch.from_numpy(flow_np).float()

        return face_tensor, flow_tensor, label


def collate_fn(batch):
    faces, flows, labels = zip(*batch)
    faces = torch.stack(faces)
    labels = torch.tensor(labels, dtype=torch.float32)

    max_t = max(f.shape[0] for f in flows)
    padded = []
    for f in flows:
        if f.shape[0] < max_t:
            pad = torch.zeros(max_t - f.shape[0], *f.shape[1:])
            f = torch.cat([f, pad], dim=0)
        padded.append(f)
    flows = torch.stack(padded)
    return faces, flows, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--checkpoint", type=str,
                        default=str(CHECKPOINT_DIR / "dual_stream_best.pt"))
    args = parser.parse_args()

    seed_everything(cfg.training.seed)
    device = get_device(cfg.training.use_mps)
    logger = setup_logger("train_custom", LOG_DIR)

    logger.info("=" * 60)
    logger.info("Stage 3 — Fine-tune on User-Collected Data")
    logger.info("=" * 60)

    dataset = CustomVideoDataset(RECORDING_DIR, num_frames=cfg.training.rldd_clip_frames)
    if len(dataset) == 0:
        logger.error("No user recordings found. Record some videos first.")
        return

    loader = DataLoader(dataset, batch_size=4, shuffle=True,
                        num_workers=2, collate_fn=collate_fn)

    model = DualStreamDeceptionDetector().to(device)
    ckpt = Path(args.checkpoint)
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        logger.info(f"Loaded base model from {ckpt}")

    # Freeze macro, train micro + fusion + classifier
    model.freeze_macro()

    criterion = nn.BCELoss()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=args.lr, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for faces, flows, labels in loader:
            faces  = faces.to(device)
            flows  = flows.to(device)
            labels = labels.to(device).unsqueeze(1)

            prob, _, _, _, _ = model(faces, flows)
            loss = criterion(prob, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * faces.size(0)
            preds = (prob > 0.5).float()
            correct += preds.eq(labels).sum().item()
            total += faces.size(0)

        acc = 100. * correct / max(total, 1)
        logger.info(f"Epoch {epoch}/{args.epochs} | "
                    f"Loss {total_loss / max(total, 1):.4f} | Acc {acc:.1f}%")

    save_checkpoint(
        {"model_state_dict": model.state_dict(), "epoch": args.epochs},
        CHECKPOINT_DIR, "dual_stream_finetuned.pt",
    )
    logger.info("Done. Fine-tuned model saved.")


if __name__ == "__main__":
    main()
