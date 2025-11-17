"""Typed structures for I3D prediction outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class PredictionResult:
    """Container for a single prediction and its top-k alternatives."""

    gloss: str
    label: int
    probability: float
    topk_glosses: List[str]
    topk_probabilities: List[float]
