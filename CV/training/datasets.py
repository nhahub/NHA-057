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
        """Apply simple spatial augmentation if enabled.

        This is intentionally conservative compared to the full Kaggle
        notebook pipeline but can be extended as needed.
        """

        if not self.augment:
            return frames

        # Example: random horizontal flip with 50% probability
        if np.random.rand() < 0.5:
            frames = [cv2.flip(f, 1) for f in frames]

        return frames

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
