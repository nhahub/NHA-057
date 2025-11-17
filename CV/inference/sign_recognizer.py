"""High-level I3D inference wrapper for SignBridge."""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from CV.models.loader import load_model_from_checkpoint
from CV.data.transforms import preprocess_frames
from .types import PredictionResult


class SignRecognizer:
    """Wrapper around the trained I3D model for easy prediction on clips."""

    def __init__(self, device: str | None = None) -> None:
        # Load model + mappings using our loader
        self.model, self.gloss_to_label, self.label_to_gloss = load_model_from_checkpoint(
            device=device
        )
        self.device = next(self.model.parameters()).device

    @torch.no_grad()
    def predict_clip(self, frames: List[np.ndarray], topk: int = 5) -> PredictionResult:
        """Predict the sign for a list of raw frames.

        Args:
            frames: list of OpenCV frames in BGR format.
            topk: number of top predictions to return.

        Returns:
            PredictionResult with top-1 prediction and top-k lists.
        """
        if not frames:
            raise ValueError("predict_clip received an empty frame list")

        clip = preprocess_frames(frames)
        clip = clip.to(self.device)

        logits = self.model(clip)  # (1, num_classes)
        probs = F.softmax(logits, dim=1)[0]

        topk = min(topk, probs.shape[0])
        values, indices = torch.topk(probs, k=topk)

        topk_labels = indices.cpu().tolist()
        topk_probs = values.cpu().tolist()
        topk_glosses = [self.label_to_gloss[int(i)] for i in topk_labels]

        best_label = int(topk_labels[0])
        best_gloss = topk_glosses[0]
        best_prob = float(topk_probs[0])

        return PredictionResult(
            gloss=best_gloss,
            label=best_label,
            probability=best_prob,
            topk_glosses=topk_glosses,
            topk_probabilities=[float(p) for p in topk_probs],
        )
