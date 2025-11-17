"""
Model and label-mapping loading utilities for the CV module.

This module provides high-level helpers to:
- Load the 100-class gloss/label mapping from JSON
- Construct the InceptionI3d model
- Load trained weights from a checkpoint
"""

from __future__ import annotations

import json
import os
from typing import Dict, Tuple

import torch

from CV import config
from .i3d import InceptionI3d


def load_label_mapping(label_map_path: str | None = None) -> Tuple[Dict[str, int], Dict[int, str], int]:
    """Load gloss/label mappings from a JSON file.

    Returns:
        gloss_to_label: mapping from gloss string to integer label
        label_to_gloss: mapping from integer label to gloss string
        num_classes: number of classes in the mapping
    """
    path = label_map_path or config.LABEL_MAP_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(f"Label mapping not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gloss_to_label = data.get("gloss_to_label")
    label_to_gloss_raw = data.get("label_to_gloss")
    num_classes = int(data.get("num_classes", 0))

    if gloss_to_label is None or label_to_gloss_raw is None or num_classes <= 0:
        raise ValueError(f"Invalid label mapping format in: {path}")

    # Ensure label_to_gloss has integer keys
    label_to_gloss: Dict[int, str] = {int(k): v for k, v in label_to_gloss_raw.items()}

    if len(gloss_to_label) != num_classes or len(label_to_gloss) != num_classes:
        # Not fatal, but warn via exception so it can be fixed explicitly
        raise ValueError(
            f"num_classes={num_classes} but mapping has "
            f"{len(gloss_to_label)} gloss_to_label and {len(label_to_gloss)} label_to_gloss entries"
        )

    return gloss_to_label, label_to_gloss, num_classes


def create_model(num_classes: int | None = None, device: str | None = None) -> InceptionI3d:
    """Create an InceptionI3d model with the given number of classes.

    Args:
        num_classes: number of output classes. If None, uses config.NUM_CLASSES.
        device: 'cpu' or 'cuda'. If None, uses config.DEVICE.

    Returns:
        InceptionI3d model on the requested device in eval mode.
    """
    if num_classes is None:
        num_classes = config.NUM_CLASSES

    if device is None:
        device = config.DEVICE

    model = InceptionI3d(num_classes=num_classes, in_channels=3)

    # Respect actual device availability
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = model.to(device)
    model.eval()
    return model


def load_model_from_checkpoint(
    checkpoint_path: str | None = None,
    label_map_path: str | None = None,
    device: str | None = None,
) -> Tuple[InceptionI3d, Dict[str, int], Dict[int, str]]:
    """Load a trained InceptionI3d model and label mappings from disk.

    This is the main entry point you will typically use.

    Args:
        checkpoint_path: path to .pth checkpoint. If None, uses config.DEFAULT_CHECKPOINT_PATH.
        label_map_path: path to label_mapping.json. If None, uses config.LABEL_MAP_PATH.
        device: 'cpu' or 'cuda'. If None, uses config.DEVICE.

    Returns:
        model: InceptionI3d with weights loaded and set to eval().
        gloss_to_label: mapping gloss -> int label.
        label_to_gloss: mapping int label -> gloss.
    """
    ckpt_path = checkpoint_path or config.DEFAULT_CHECKPOINT_PATH

    if device is None:
        device = config.DEVICE

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    # Load label mappings first to get num_classes
    gloss_to_label, label_to_gloss, num_classes = load_label_mapping(label_map_path)

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # Determine how weights are stored
    if "model" in checkpoint and isinstance(checkpoint["model"], torch.nn.Module):
        # Full model object saved; still recreate fresh model for safety
        state_dict = checkpoint["model"].state_dict()
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        raise KeyError(
            "Checkpoint does not contain 'model', 'model_state_dict', or 'state_dict' keys: "
            f"available keys = {list(checkpoint.keys())}"
        )

    # Handle common DataParallel-style "module." prefix so keys match our model
    keys = list(state_dict.keys())
    if keys and all(k.startswith("module.") for k in keys):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    # Create model and ensure classifier head matches the mapping
    model = create_model(num_classes=num_classes, device=device)

    # First try a strict load. If shapes/names don't match perfectly, fall back.
    try:
        incompatible = model.load_state_dict(state_dict, strict=True)
        missing = incompatible.missing_keys
        unexpected = incompatible.unexpected_keys
    except RuntimeError as e:
        print("[load_model_from_checkpoint] Strict load failed, falling back to strict=False")
        print("  Error:", e)
        incompatible = model.load_state_dict(state_dict, strict=False)
        missing = incompatible.missing_keys
        unexpected = incompatible.unexpected_keys

    if missing or unexpected:
        # For now just print; in production you might want logging instead
        print("[load_model_from_checkpoint] Warning: key mismatch while loading state_dict")
        print(f"  Missing keys: {len(missing)}")
        print(f"  Unexpected keys: {len(unexpected)}")

    model.eval()
    return model, gloss_to_label, label_to_gloss
