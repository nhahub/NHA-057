"""Preprocessing utilities to convert raw frames into I3D input tensors."""

from __future__ import annotations

from typing import List

import cv2
import numpy as np
import torch

from CV import config


def _pad_or_sample_frames(frames: List[np.ndarray], target_num_frames: int) -> List[np.ndarray]:
    """Pad or uniformly sample a list of frames to a fixed length.

    - If len(frames) > target_num_frames: uniformly sample target_num_frames indices.
    - If len(frames) < target_num_frames: repeat the last frame until the length matches.
    - If len(frames) == 0: raise ValueError.
    """
    num_frames = len(frames)
    if num_frames == 0:
        raise ValueError("No frames provided to _pad_or_sample_frames")

    if num_frames == target_num_frames:
        return frames

    if num_frames > target_num_frames:
        # Uniform sampling over the sequence
        indices = np.linspace(0, num_frames - 1, target_num_frames).astype(int)
        return [frames[i] for i in indices]

    # num_frames < target_num_frames: pad with last frame
    last = frames[-1]
    padded = frames + [last] * (target_num_frames - num_frames)
    return padded


def preprocess_frames(
    frames: List[np.ndarray],
    num_frames: int | None = None,
    image_size: int | None = None,
) -> torch.Tensor:
    """Convert a list of raw BGR frames into a (1, 3, T, H, W) tensor for I3D.

    Args:
        frames: list of OpenCV images in BGR format.
        num_frames: desired temporal length T; defaults to config.NUM_FRAMES.
        image_size: desired H = W; defaults to config.IMAGE_SIZE.

    Returns:
        torch.Tensor of shape (1, 3, T, H, W), dtype float32 in [0, 1].
    """
    if num_frames is None:
        num_frames = getattr(config, "NUM_FRAMES", 32)
    if image_size is None:
        image_size = getattr(config, "IMAGE_SIZE", 224)

    # Ensure fixed T
    frames = _pad_or_sample_frames(frames, num_frames)

    processed = []
    for frame in frames:
        # OpenCV gives BGR; convert to RGB to match typical training convention
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
        frame = frame.astype("float32") / 255.0  # scale to [0, 1]
        processed.append(frame)

    # Shape: (T, H, W, C)
    clip = np.stack(processed, axis=0)
    # Reorder to (C, T, H, W)
    clip = np.transpose(clip, (3, 0, 1, 2))

    # Add batch dimension -> (1, C, T, H, W)
    tensor = torch.from_numpy(clip).unsqueeze(0)
    return tensor
