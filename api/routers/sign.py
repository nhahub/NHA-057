from __future__ import annotations

import os
import tempfile
import time

from fastapi import APIRouter, File, HTTPException, UploadFile

from api.dependencies.models import get_sign_recognizer
from api.schemas.sign import SignPredictionResponse, TopKPrediction
from api.utils.video_io import read_video_frames


router = APIRouter(prefix="/sign", tags=["sign"])


@router.post("/predict", response_model=SignPredictionResponse)
async def predict_sign(
    video: UploadFile = File(..., description="Video file containing a single sign."),
    top_k: int = 5,
) -> SignPredictionResponse:
    """Predict the sign in an uploaded video clip.

    The video is expected to contain a single isolated sign. The backend
    will decode the video into frames, run the I3D model and return the
    top prediction along with top-k alternatives.
    """
    if top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be positive")

    # Save the uploaded file to a temporary path so OpenCV can read it.
    try:
        suffix = os.path.splitext(video.filename or "uploaded")[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            raw = await video.read()
            tmp.write(raw)
            tmp_path = tmp.name
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=f"Failed to store uploaded video: {exc}") from exc

    start = time.perf_counter()
    try:
        frames = read_video_frames(tmp_path)
    except Exception as exc:  # pylint: disable=broad-except
        os.unlink(tmp_path)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        recognizer = get_sign_recognizer()
        result = recognizer.predict_clip(frames, topk=top_k)
    except Exception as exc:  # pylint: disable=broad-except
        os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}") from exc

    os.unlink(tmp_path)

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    topk = [
        TopKPrediction(gloss=g, label=-1, probability=float(p))
        for g, p in zip(result.topk_glosses, result.topk_probabilities)
    ]

    return SignPredictionResponse(
        gloss=result.gloss,
        label=int(result.label),
        probability=float(result.probability),
        topk=topk,
        num_frames_used=len(frames),
        processing_ms=elapsed_ms,
    )
