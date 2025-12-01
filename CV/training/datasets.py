"""Datasets for training the I3D sign recognition model.

This module provides a simple, manifest-driven video classification
Dataset that reuses the existing I3D preprocessing utilities from
`CV.data.transforms`.

The design is intentionally generic so that it can be used with
WLASL, ASL Citizen, or custom datasets, as long as you prepare a
CSV manifest with at least the following columns:

- `video_path`: path to the video file (relative to a base directory
  or an absolute path).
- `label`: integer class id in the range [0, num_classes-1].

Optional columns like `gloss` can also be present but are not
required for training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from CV.data.transforms import preprocess_frames


@dataclass
class VideoDatasetConfig:
    """Configuration for the VideoClassificationDataset."""

    manifest_path: str
    base_dir: Optional[str] = None
    num_frames: Optional[int] = None
    image_size: Optional[int] = None
    augment: bool = False


class VideoClassificationDataset(Dataset):
    """Generic video classification dataset backed by a CSV manifest.

    Args:
        config: VideoDatasetConfig instance with manifest and options.

    Expected manifest columns:
        - `video_path`: path to the video file (relative or absolute).
        - `label`: integer class id.

    Any extra columns (e.g. `gloss`, `split`) are ignored by the dataset.
    """

    def __init__(self, config: VideoDatasetConfig) -> None:
        super().__init__()
        self.config = config
        self.manifest_path = Path(config.manifest_path)
        self.base_dir = Path(config.base_dir) if config.base_dir is not None else None
        self.num_frames = config.num_frames
        self.image_size = config.image_size
        self.augment = config.augment

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest CSV not found at: {self.manifest_path}")

        self.df = pd.read_csv(self.manifest_path)

        if "video_path" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError(
                "Manifest must contain at least 'video_path' and 'label' columns. "
                f"Columns found: {list(self.df.columns)}"
            )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.df)

    def _resolve_video_path(self, idx: int) -> Path:
        rel_or_abs = Path(self.df.loc[idx, "video_path"])
        if self.base_dir is not None and not rel_or_abs.is_absolute():
            return self.base_dir / rel_or_abs
        return rel_or_abs

    def _read_video_frames(self, path: Path) -> list[np.ndarray]:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {path}")

        frames: list[np.ndarray] = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
        finally:
            cap.release()

        if not frames:
            raise RuntimeError(f"No frames read from video file: {path}")

        return frames

    def _maybe_augment(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Apply video-level augmentation when `augment` is enabled.

        This mirrors the richer augmentation pipeline used in the
        training notebook and operates directly on uint8 frames:

        1. Horizontal flip (30% chance).
        2. Temporal cropping to 75–100% of frames, then resample to
           the original length.
        3. Brightness adjustment (0.85–1.15×).
        4. Contrast adjustment (0.85–1.15×).
        5. Small rotation (±3°).
        6. Spatial crop (85–100% area) + resize back to original size.
        7. Gaussian noise (σ=5, 20% chance).
        """

        if not self.augment:
            return frames

        # Stack to (T, H, W, C)
        arr = np.stack(frames, axis=0)
        T, H, W, C = arr.shape
        original_T = T

        # 1. Horizontal flip (30% chance)
        if np.random.rand() < 0.3:
            arr = np.flip(arr, axis=2).copy()

        # 2. Temporal cropping (75–100% frames) + resample back to original_T
        crop_ratio = np.random.uniform(0.75, 1.0)
        num_frames = max(int(T * crop_ratio), 16)

        if num_frames < T:
            start_idx = np.random.randint(0, T - num_frames + 1)
            cropped = arr[start_idx : start_idx + num_frames]
            indices = np.linspace(0, num_frames - 1, original_T, dtype=int)
            arr = cropped[indices]

        # Refresh shape after possible temporal resampling
        T, H, W, C = arr.shape

        # 3. Brightness adjustment (0.85–1.15×)
        brightness_factor = np.random.uniform(0.85, 1.15)
        arr = np.clip(arr.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)

        # 4. Contrast adjustment (0.85–1.15×)
        contrast_factor = np.random.uniform(0.85, 1.15)
        mean = arr.mean()
        arr = np.clip((arr.astype(np.float32) - mean) * contrast_factor + mean, 0, 255).astype(
            np.uint8
        )

        # 5. Rotation (±3°)
        angle = np.random.uniform(-3, 3)
        center = (W // 2, H // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated = np.zeros_like(arr)
        for t in range(T):
            rotated[t] = cv2.warpAffine(arr[t], M, (W, H), borderMode=cv2.BORDER_REPLICATE)
        arr = rotated

        # 6. Spatial crop (85–100% area) + resize
        crop_ratio_spatial = np.random.uniform(0.85, 1.0)
        crop_h = int(H * crop_ratio_spatial)
        crop_w = int(W * crop_ratio_spatial)

        top = np.random.randint(0, H - crop_h + 1) if crop_h < H else 0
        left = np.random.randint(0, W - crop_w + 1) if crop_w < W else 0

        cropped_frames = np.zeros_like(arr)
        for t in range(T):
            cropped = arr[t, top : top + crop_h, left : left + crop_w]
            cropped_frames[t] = cv2.resize(cropped, (W, H), interpolation=cv2.INTER_LINEAR)
        arr = cropped_frames

        # 7. Gaussian noise (σ=5, 20% chance)
        if np.random.rand() < 0.2:
            noise = np.random.normal(0, 5, arr.shape).astype(np.float32)
            arr = np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Final safety: ensure temporal length matches original_T
        if arr.shape[0] != original_T:
            indices = np.linspace(0, arr.shape[0] - 1, original_T, dtype=int)
            arr = arr[indices]

        # Convert back to list of frames
        frames_aug = [arr[t] for t in range(arr.shape[0])]
        return frames_aug

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:  # type: ignore[override]
        row = self.df.iloc[index]
        video_path = self._resolve_video_path(index)
        label = int(row["label"])

        frames = self._read_video_frames(video_path)
        frames = self._maybe_augment(frames)

        clip_tensor = preprocess_frames(
            frames,
            num_frames=self.num_frames,
            image_size=self.image_size,
        )
        return clip_tensor, label
