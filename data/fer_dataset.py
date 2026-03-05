"""
Truth in a Blink — FER2013 DataLoader
======================================
Loads FER2013 images organised in subdirectories:
    FER_2013/train/{angry,disgust,fear,happy,neutral,sad,surprise}/*.png
    FER_2013/test/{angry,disgust,fear,happy,neutral,sad,surprise}/*.png

Returns 224×224 RGB tensors with standard ImageNet-style normalisation.
"""

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Emotion label mapping (alphabetical directory order matches this)
FER_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# ── Transforms ────────────────────────────────────────────────────────────────

def get_fer_train_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_fer_val_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ── Dataset factory ──────────────────────────────────────────────────────────

def get_fer_dataloaders(
    data_root: Path,
    batch_size: int = 64,
    image_size: int = 224,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and test DataLoaders for FER2013.

    Parameters
    ----------
    data_root   : Path to FER_2013/ root (contains train/ and test/).
    batch_size  : Mini-batch size.
    image_size  : Target image size (square).
    num_workers : Dataloader worker processes.
    pin_memory  : Pin memory for faster GPU transfer.

    Returns
    -------
    train_loader, test_loader
    """
    train_dir = data_root / "train"
    test_dir  = data_root / "test"

    train_ds = datasets.ImageFolder(str(train_dir),
                                    transform=get_fer_train_transform(image_size))
    test_ds  = datasets.ImageFolder(str(test_dir),
                                    transform=get_fer_val_transform(image_size))

    # Verify class-to-index mapping matches expected order
    print(f"[FER2013] Classes: {train_ds.classes}")
    print(f"[FER2013] Train samples: {len(train_ds)}  |  Test samples: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader
