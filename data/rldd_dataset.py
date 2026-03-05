"""
Truth in a Blink — RLDD Dataset Loader
========================================
Loads the Real-Life Deception Detection 2016 dataset.

Each sample is a video clip (MP4) labelled as truthful (0) or deceptive (1).
The loader extracts frames, computes optical-flow, and returns:
  • face_image  : centre-frame face crop (3, 224, 224)
  • flow_seq    : optical-flow sequence  (T, 2, 56, 56)
  • label       : 0 or 1

The face detection uses OpenCV's Haar cascade (no internet required).
"""

from pathlib import Path
from typing import Optional
import random
import csv

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models.micro_stream import compute_flow_sequence


# ── Helpers ───────────────────────────────────────────────────────────────────

# OpenCV ships its own Haar cascade XML
_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(_CASCADE_PATH)


def detect_face(frame_bgr: np.ndarray, padding: float = 0.3
                ) -> Optional[np.ndarray]:
    """
    Detect the largest face and return a padded crop (RGB, uint8).
    Returns None if no face is found.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None

    # Largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # Pad
    H, W = frame_bgr.shape[:2]
    pad_w, pad_h = int(w * padding), int(h * padding)
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(W, x + w + pad_w)
    y2 = min(H, y + h + pad_h)

    crop = frame_bgr[y1:y2, x1:x2]
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def sample_frames(cap: cv2.VideoCapture, num_frames: int = 64
                  ) -> list[np.ndarray]:
    """Uniformly sample `num_frames` from a video capture, returns BGR frames."""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        total = len(frames)
        if total == 0:
            return []
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        return [frames[i] for i in indices]

    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    return frames


# ── Transforms for face crops ────────────────────────────────────────────────

_face_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

_face_transform_aug = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Dataset ───────────────────────────────────────────────────────────────────

class RLDDDataset(Dataset):
    """
    Real-Life Deception Detection 2016 dataset.

    Parameters
    ----------
    clips_dir      : Path to Clips/ directory.
    num_frames     : Frames to sample from each video.
    flow_size      : Spatial size for optical-flow fields.
    face_padding   : Padding ratio for face crop.
    augment        : Apply data augmentation to face crops.
    """

    def __init__(
        self,
        clips_dir: Path,
        num_frames: int = 64,
        flow_size: tuple[int, int] = (56, 56),
        face_padding: float = 0.3,
        augment: bool = False,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.flow_size = flow_size
        self.face_padding = face_padding
        self.augment = augment
        self.transform = _face_transform_aug if augment else _face_transform

        # Ensure Path object (handles str input)
        if isinstance(clips_dir, str):
            clips_dir = Path(clips_dir)

        # Auto-detect Clips/ subdirectory
        if not (clips_dir / "Deceptive").exists() and (clips_dir / "Clips" / "Deceptive").exists():
            clips_dir = clips_dir / "Clips"

        # Collect all video paths + labels
        self.samples: list[tuple[Path, int]] = []

        deceptive_dir = clips_dir / "Deceptive"
        truthful_dir  = clips_dir / "Truthful"

        if deceptive_dir.exists():
            for f in sorted(deceptive_dir.glob("*.mp4")):
                self.samples.append((f, 1))
        if truthful_dir.exists():
            for f in sorted(truthful_dir.glob("*.mp4")):
                self.samples.append((f, 0))

        print(f"[RLDD] Loaded {len(self.samples)} clips "
              f"({sum(s[1] for s in self.samples)} deceptive, "
              f"{sum(1 - s[1] for s in self.samples)} truthful)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        video_path, label = self.samples[idx]

        cap = cv2.VideoCapture(str(video_path))
        frames = sample_frames(cap, self.num_frames)
        cap.release()

        if len(frames) < 2:
            # Fallback: black frames
            face_tensor = torch.zeros(3, 224, 224)
            flow_tensor = torch.zeros(max(1, self.num_frames - 1),
                                      2, *self.flow_size)
            return face_tensor, flow_tensor, label

        # ── Face crop from centre frame ──────────────────────────────────
        mid = len(frames) // 2
        face_crop = detect_face(frames[mid], self.face_padding)
        if face_crop is None:
            # Try other frames
            for offset in [1, -1, 2, -2, 3, -3]:
                fidx = mid + offset
                if 0 <= fidx < len(frames):
                    face_crop = detect_face(frames[fidx], self.face_padding)
                    if face_crop is not None:
                        break
        if face_crop is None:
            # Last resort: use full centre frame as RGB
            face_crop = cv2.cvtColor(frames[mid], cv2.COLOR_BGR2RGB)

        face_tensor = self.transform(face_crop)            # (3, 224, 224)

        # ── Optical flow sequence ────────────────────────────────────────
        grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        flow_np = compute_flow_sequence(grays, self.flow_size)  # (T-1, 2, H, W)
        flow_tensor = torch.from_numpy(flow_np).float()

        return face_tensor, flow_tensor, label


# ── Collate function (module-level for pickling in multiprocessing) ───────────

def rldd_collate_fn(batch):
    """Collate RLDD samples, padding flow sequences to same length."""
    faces, flows, labels = zip(*batch)
    faces = torch.stack(faces)
    labels = torch.tensor(labels, dtype=torch.float32)

    # Pad flow sequences to same length
    max_t = max(f.shape[0] for f in flows)
    padded = []
    for f in flows:
        if f.shape[0] < max_t:
            pad = torch.zeros(max_t - f.shape[0], *f.shape[1:])
            f = torch.cat([f, pad], dim=0)
        padded.append(f)
    flows = torch.stack(padded)

    return faces, flows, labels


# ── DataLoader factory ────────────────────────────────────────────────────────

def get_rldd_dataloaders(
    clips_dir: Path,
    num_frames: int = 64,
    batch_size: int = 8,
    val_split: float = 0.2,
    num_workers: int = 2,
    seed: int = 42,
    flow_size: tuple[int, int] = (56, 56),
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders for RLDD.
    Splits clips stratified by label.
    """
    full_ds = RLDDDataset(clips_dir, num_frames=num_frames,
                          flow_size=flow_size, augment=False)

    # Stratified split
    deceptive = [i for i, (_, l) in enumerate(full_ds.samples) if l == 1]
    truthful  = [i for i, (_, l) in enumerate(full_ds.samples) if l == 0]

    rng = random.Random(seed)
    rng.shuffle(deceptive)
    rng.shuffle(truthful)

    n_val_d = max(1, int(len(deceptive) * val_split))
    n_val_t = max(1, int(len(truthful)  * val_split))

    val_indices  = deceptive[:n_val_d] + truthful[:n_val_t]
    train_indices = deceptive[n_val_d:] + truthful[n_val_t:]

    train_ds = torch.utils.data.Subset(full_ds, train_indices)
    val_ds   = torch.utils.data.Subset(full_ds, val_indices)

    # Enable augmentation on the training subset's underlying dataset view
    # (We create a copy with augment=True for training)
    train_ds_aug = RLDDDataset(clips_dir, num_frames=num_frames,
                               flow_size=flow_size, augment=True)
    train_ds = torch.utils.data.Subset(train_ds_aug, train_indices)

    print(f"[RLDD] Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=rldd_collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=rldd_collate_fn,
    )
    return train_loader, val_loader
