"""Central configuration for the CV module.

This file defines paths and basic settings used across
model loading, preprocessing and inference.
"""

import os
import torch

# Base directory for the repository
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# CV root directory
CV_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths for assets and checkpoints
ASSETS_DIR = os.path.join(CV_DIR, "assets")
CHECKPOINTS_DIR = os.path.join(CV_DIR, "checkpoints")

# Default files (can be overridden by environment variables)
LABEL_MAP_PATH = os.environ.get(
    "SIGNBRIDGE_LABEL_MAP",
    os.path.join(ASSETS_DIR, "label_mapping.json"),
)

# Final 100-class Citizen+WLASL model (best accuracy: 87.68%)
DEFAULT_CHECKPOINT_PATH = os.environ.get(
    "SIGNBRIDGE_CHECKPOINT",
    os.path.join(CHECKPOINTS_DIR, "best_model_citizen100_87pct.pth"),
)

# Device selection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Input configuration
NUM_FRAMES = 32
IMAGE_SIZE = 224

# Number of classes for the final model
NUM_CLASSES = 100
