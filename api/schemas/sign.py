from typing import List

from pydantic import BaseModel


class TopKPrediction(BaseModel):
    gloss: str
    label: int
    probability: float


class SignPredictionResponse(BaseModel):
    gloss: str
    label: int
    probability: float
    topk: List[TopKPrediction]
    num_frames_used: int
    processing_ms: float
