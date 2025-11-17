"""Webcam and video reading utilities for SignBridge I3D inference."""

from __future__ import annotations

from typing import List

import cv2


def open_webcam(camera_index: int = 0) -> cv2.VideoCapture:
    """Open a webcam device for capturing frames.

    Raises:
        RuntimeError: if the camera cannot be opened.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam with index {camera_index}")
    return cap


def release_webcam(cap: cv2.VideoCapture | None) -> None:
    """Release an opened webcam safely."""
    if cap is not None:
        cap.release()
