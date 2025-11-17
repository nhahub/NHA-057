"""
Local webcam demo for the I3D sign recognition model.

Run this script to test the trained 100-class model on your webcam.
"""

from __future__ import annotations

import cv2
import threading
from queue import Queue

from CV import config
from CV.data.video_reader import open_webcam, release_webcam
from CV.inference.sign_recognizer import SignRecognizer


def prediction_worker(recognizer, frame_queue: Queue) -> None:
    """Worker thread that runs predictions on collected frames."""
    while True:
        frames = frame_queue.get()
        if frames is None:  # Signal to stop
            break
        try:
            result = recognizer.predict_clip(frames, topk=5)
            print(
                f"Prediction: {result.gloss} (label={result.label}, prob={result.probability:.2f})"
            )
            print(
                "  Top-5: "
                + ", ".join(
                    f"{g} ({p:.2f})" for g, p in zip(result.topk_glosses, result.topk_probabilities)
                )
            )
        except Exception as e:  # pylint: disable=broad-except
            print(f"Prediction error: {e}")


def main() -> None:
    recognizer = SignRecognizer()
    num_frames_per_clip = getattr(config, "NUM_FRAMES", 32)

    print("==============================")
    print(" SignBridge I3D Webcam Demo")
    print("==============================")
    print("Press 'q' in the video window to quit.")
    print(f"Collecting {num_frames_per_clip} frames per clip before each prediction.\n")

    cap = None
    try:
        cap = open_webcam(0)
    except Exception as e:  # pylint: disable=broad-except
        print(f"Error opening webcam: {e}")
        return

    # Start prediction worker thread
    frame_queue = Queue(maxsize=1)
    worker_thread = threading.Thread(target=prediction_worker, args=(recognizer, frame_queue), daemon=True)
    worker_thread.start()

    frames = []
    window_name = "SignBridge - Webcam"

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam. Exiting.")
            break

        # Show live preview
        cv2.imshow(window_name, frame)

        # Accumulate frames for the current clip
        frames.append(frame)

        if len(frames) >= num_frames_per_clip:
            # Send frames to worker thread (non-blocking)
            try:
                frame_queue.put_nowait(frames.copy())
            except:  # pylint: disable=bare-except
                pass  # Skip if queue is full
            frames = []

        # Handle keypresses
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    frame_queue.put(None)  # Signal worker to stop
    worker_thread.join(timeout=5)
    release_webcam(cap)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()