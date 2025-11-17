from __future__ import annotations

from typing import List

import cv2
import numpy as np


def read_video_frames(path: str, max_frames: int | None = None) -> List[np.ndarray]:
    """Read frames from a video file using OpenCV.

    Args:
        path: Path to the video file.
        max_frames: Optional maximum number of frames to read.

    Returns:
        List of frames as BGR numpy arrays.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {path}")

    frames: List[np.ndarray] = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            if max_frames is not None and len(frames) >= max_frames:
                break
    finally:
        cap.release()

    if not frames:
        raise ValueError("No frames could be read from the uploaded video.")

    return frames
