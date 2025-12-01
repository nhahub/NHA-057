"""Training script for the Inception I3D sign recognition model.

This script is a structured version of the training loops used in the
`01_sign_to_text.ipynb` notebook. It is designed to be:

- Self-contained inside the `CV` package.
- Safe for the existing inference code (no behavior changes).
- Flexible enough to train on WLASL, ASL Citizen, or custom datasets
  via CSV manifests.

Usage (example):

```bash
python -m CV.training.train_i3d \
  --train-manifest /path/to/train_manifest.csv \
  --val-manifest /path/to/val_manifest.csv \
  --base-dir /path/to/videos_root \
  --label-map CV/assets/label_mapping.json \
  --epochs 50 \
  --batch-size 8
```

The manifest CSVs must contain at least:
- `video_path`: relative or absolute path to each video.
- `label`: integer class id.

The label map JSON must follow the same format as used by the
inference pipeline (see `CV/models/loader.py`).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from CV import config
from CV.models.loader import create_model, load_label_mapping
from CV.training.datasets import VideoDatasetConfig, VideoClassificationDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Inception I3D for sign recognition.")

    parser.add_argument("--train-manifest", type=str, required=True, help="Path to training manifest CSV.")
    parser.add_argument("--val-manifest", type=str, required=True, help="Path to validation manifest CSV.")
    parser.add_argument("--base-dir", type=str, default=None, help="Optional base directory for video files.")

    parser.add_argument("--label-map", type=str, default=None, help="Path to label_mapping.json (defaults to CV.config.LABEL_MAP_PATH).")

    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay (L2 regularization).")

    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader workers.")

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(config.CHECKPOINTS_DIR)),
        help="Directory to save checkpoints.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Optional checkpoint .pth to resume fine-tuning from.",
    )

    return parser.parse_args()


def build_dataloaders(
    train_manifest: str,
    val_manifest: str,
    base_dir: str | None,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation DataLoaders."""

    train_cfg = VideoDatasetConfig(
        manifest_path=train_manifest,
        base_dir=base_dir,
        num_frames=getattr(config, "NUM_FRAMES", 32),
        image_size=getattr(config, "IMAGE_SIZE", 224),
        augment=True,
    )
    val_cfg = VideoDatasetConfig(
        manifest_path=val_manifest,
        base_dir=base_dir,
        num_frames=getattr(config, "NUM_FRAMES", 32),
        image_size=getattr(config, "IMAGE_SIZE", 224),
        augment=False,
    )

    train_dataset = VideoClassificationDataset(train_cfg)
    val_dataset = VideoClassificationDataset(val_cfg)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_acc: float,
    output_dir: Path,
    filename: str,
    extra: Dict | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / filename

    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_acc": best_val_acc,
    }
    if extra:
        payload.update(extra)

    torch.save(payload, ckpt_path)
    print(f"[save_checkpoint] Saved checkpoint to: {ckpt_path}")


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", ncols=100)
    for clips, labels in pbar:
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(clips)  # (B, num_classes)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * clips.size(0)
        _, preds = torch.max(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        avg_loss = running_loss / max(total, 1)
        acc = correct / max(total, 1)
        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{acc*100:.2f}%"})

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [val]  ", ncols=100)
    for clips, labels in pbar:
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(clips)
        loss = criterion(logits, labels)

        running_loss += loss.item() * clips.size(0)
        _, preds = torch.max(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        avg_loss = running_loss / max(total, 1)
        acc = correct / max(total, 1)
        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{acc*100:.2f}%"})

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


def main() -> None:
    args = parse_args()

    device = torch.device(config.DEVICE)
    print(f"[train_i3d] Using device: {device}")

    # Load label mappings to determine num_classes
    gloss_to_label, label_to_gloss, num_classes = load_label_mapping(args.label_map)
    print(f"[train_i3d] Loaded label mapping with {num_classes} classes.")

    # Build model
    model = create_model(num_classes=num_classes, device=str(device))

    # Optionally resume from checkpoint
    if args.resume_from is not None:
        ckpt_path = Path(args.resume_from)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found at: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        print(f"[train_i3d] Resumed model weights from: {ckpt_path}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    train_loader, val_loader = build_dataloaders(
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        base_dir=args.base_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    output_dir = Path(args.output_dir)
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, epoch)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f}, train_acc={train_acc*100:.2f}% | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc*100:.2f}%"
        )

        # Save last checkpoint
        save_checkpoint(
            model,
            optimizer,
            epoch,
            best_val_acc,
            output_dir,
            filename="last_checkpoint.pth",
            extra={"gloss_to_label": gloss_to_label, "label_to_gloss": label_to_gloss},
        )

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_val_acc,
                output_dir,
                filename="best_checkpoint.pth",
                extra={"gloss_to_label": gloss_to_label, "label_to_gloss": label_to_gloss},
            )
            print(
                f"[train_i3d] New best val_acc={best_val_acc*100:.2f}%. "
                f"Saved best_checkpoint.pth."
            )

    print("[train_i3d] Training complete.")


if __name__ == "__main__":
    main()
