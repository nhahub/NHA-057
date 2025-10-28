from google.colab import drive
drive.mount('/content/drive')

DATASET_PATH = "/content/drive/MyDrive/Signbridge/MS-ASL/"

!mkdir -p SignBridge/data/raw/MS-ASL
!cp {DATASET_PATH}/*.json SignBridge/data/raw/MS-ASL/




import os
import re
import json
import subprocess
import tempfile
import warnings
from moviepy.video.io.VideoFileClip import VideoFileClip
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from time import time
import shutil

# Suppress TensorFlow and MediaPipe warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('mediapipe').setLevel(logging.ERROR)

# ----------------- CONFIG -----------------
RAW_JSON = "SignBridge/data/raw/MS-ASL/MSASL_train.json"
SUBSET_JSON = "SignBridge/data/subsets/msasl200_subset.json"
OUT_DIR = "SignBridge/data/processed"
SAMPLES_DIR = os.path.join(OUT_DIR, "samples")
PROCESSED_IDX = os.path.join(OUT_DIR, "processed_index.txt")
FAILED_IDX = os.path.join(OUT_DIR, "failed_index.txt")

LABEL_THRESHOLD = 100
MAX_SAMPLES = None  # set to 50 for testing
MIN_FRAMES = 5  # minimum frames required per sample
MAX_VIDEO_SIZE_MB = 500  # skip videos larger than this

# ensure dirs
os.makedirs(SAMPLES_DIR, exist_ok=True)
TEMP_DIR = tempfile.mkdtemp(prefix="msasl_")

# mediapipe
mp_holistic = mp.solutions.holistic

# ----------------- helpers -----------------
def sanitize_filename(s):
    s = re.sub(r'[^\w\d_. -]', '_', s)
    return s[:120]

def ensure_yt_dlp():
    try:
        result = subprocess.run(
            ["yt-dlp", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        print("Installing yt-dlp...")
        subprocess.run(["pip", "install", "-q", "yt-dlp"], check=True)
        return True

def download_video_segment(url, save_path, start_time, end_time):
    """Download only the required segment to save bandwidth and storage."""
    try:
        if url.startswith("www."):
            url = "https://" + url
        elif not url.startswith("http"):
            url = "https://" + url

        # Download only the segment needed
        duration = end_time - start_time
        cmd = [
            "yt-dlp",
            "-f", "best[height<=480][ext=mp4]/best[ext=mp4]/best",  # lower quality for speed
            "--download-sections", f"*{start_time}-{end_time}",
            "-o", save_path,
            "--quiet",
            "--no-warnings",
            url
        ]

        result = subprocess.run(
            cmd,
            timeout=60,  # 60 second timeout
            capture_output=True,
            text=True
        )

        # Verify file exists and is valid
        if os.path.exists(save_path):
            size_mb = os.path.getsize(save_path) / (1024 * 1024)
            if size_mb > MAX_VIDEO_SIZE_MB:
                os.remove(save_path)
                print(f"  ⚠️ video too large ({size_mb:.1f}MB), skipping")
                return False

            # Quick validation
            cap = cv2.VideoCapture(save_path)
            is_valid = cap.isOpened()
            cap.release()
            return is_valid

        return False

    except subprocess.TimeoutExpired:
        print("  ⚠️ download timeout")
        return False
    except Exception as e:
        print(f"  ⚠️ download failed: {e}")
        return False

def extract_frame_features(results):
    """Extract 225-dim feature vector: pose(33*3) + left_hand(21*3) + right_hand(21*3)"""
    pose_vec = np.zeros((33, 3), dtype=np.float32)
    lhand_vec = np.zeros((21, 3), dtype=np.float32)
    rhand_vec = np.zeros((21, 3), dtype=np.float32)

    if results.pose_landmarks:
        arr = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark], dtype=np.float32)
        pose_vec[:min(33, arr.shape[0]), :] = arr[:33, :]

    if results.left_hand_landmarks:
        arr = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark], dtype=np.float32)
        lhand_vec[:min(21, arr.shape[0]), :] = arr[:21, :]

    if results.right_hand_landmarks:
        arr = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark], dtype=np.float32)
        rhand_vec[:min(21, arr.shape[0]), :] = arr[:21, :]

    return np.concatenate([pose_vec.flatten(), lhand_vec.flatten(), rhand_vec.flatten()])

def extract_landmarks_from_file(video_path):
    """Extract landmarks from entire video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    features = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        vec = extract_frame_features(results)
        features.append(vec)

    cap.release()
    holistic.close()

    if len(features) < MIN_FRAMES:
        return None

    return np.vstack(features)

def load_processed_sets():
    """Load both processed and failed indices."""
    processed = set()
    failed = set()

    if os.path.exists(PROCESSED_IDX):
        with open(PROCESSED_IDX) as f:
            processed = set(line.strip() for line in f)

    if os.path.exists(FAILED_IDX):
        with open(FAILED_IDX) as f:
            failed = set(line.strip() for line in f)

    return processed, failed

def mark_processed(unique_id):
    with open(PROCESSED_IDX, "a") as f:
        f.write(unique_id + "\n")

def mark_failed(unique_id, reason):
    with open(FAILED_IDX, "a") as f:
        f.write(f"{unique_id}\t{reason}\n")

# ----------------- main -----------------
def main():
    ensure_yt_dlp()

    # Load dataset
    if os.path.exists(SUBSET_JSON):
        with open(SUBSET_JSON) as f:
            data_all = json.load(f)
        print(f"Using subset: {SUBSET_JSON} -> {len(data_all)} samples")
    else:
        with open(RAW_JSON) as f:
            raw = json.load(f)
        data_all = [d for d in raw if d.get("label", 9999) < LABEL_THRESHOLD]
        print(f"Filtered RAW -> {len(data_all)} samples with label < {LABEL_THRESHOLD}")

    processed_set, failed_set = load_processed_sets()
    print(f"Already processed: {len(processed_set)}, Failed: {len(failed_set)}")

    count = 0
    failed_count = 0
    start_time_overall = time()

    try:
        for idx, sample in enumerate(data_all):
            if MAX_SAMPLES and count >= MAX_SAMPLES:
                break

            unique_id = f"{idx}_{sample.get('label', 'NA')}"
            if unique_id in processed_set or unique_id in failed_set:
                continue

            url = sample.get("url", "")
            label = sample.get("label", -1)
            word = sample.get("text", "").strip() or f"label_{label}"
            start_t = float(sample.get("start_time", sample.get('start', 0)))
            end_t = float(sample.get("end_time", sample.get('end', start_t + 2)))

            # Progress indicator
            elapsed = int(time() - start_time_overall)
            print(f"[{idx}/{len(data_all)}] {word} (label={label}) | ✅ {count} processed | ❌ {failed_count} failed | ⏱️ {elapsed}s")

            # Download segment
            fname = sanitize_filename(f"{word}_{idx}.mp4")
            video_path = os.path.join(TEMP_DIR, fname)

            ok = download_video_segment(url, video_path, start_t, end_t)
            if not ok:
                mark_failed(unique_id, "download_failed")
                failed_count += 1
                continue

            try:
                # Extract landmarks
                frames_feats = extract_landmarks_from_file(video_path)

                # Cleanup temp video immediately
                try:
                    os.remove(video_path)
                except:
                    pass

                if frames_feats is None:
                    print(f"  ⚠️ insufficient frames (< {MIN_FRAMES})")
                    mark_failed(unique_id, "insufficient_frames")
                    failed_count += 1
                    continue

                # Validate features
                if np.any(np.isnan(frames_feats)) or np.any(np.isinf(frames_feats)):
                    print("  ⚠️ invalid features (NaN/Inf)")
                    mark_failed(unique_id, "invalid_features")
                    failed_count += 1
                    continue

                # Save sample
                sample_out = os.path.join(SAMPLES_DIR, f"{idx}_{label}.npz")
                np.savez_compressed(sample_out, X=frames_feats, y=np.array(label), word=word)
                mark_processed(unique_id)

                count += 1
                if count % 10 == 0:
                    elapsed = int(time() - start_time_overall)
                    rate = count / elapsed if elapsed > 0 else 0
                    eta_remaining = int((len(data_all) - idx) / rate) if rate > 0 else 0
                    print(f"  ✅ Checkpoint: {count} samples | {rate:.2f} samples/sec | ETA: {eta_remaining//60}min")

            except Exception as e:
                print(f"  ⚠️ error: {e}")
                mark_failed(unique_id, str(e)[:50])
                failed_count += 1
                try:
                    os.remove(video_path)
                except:
                    pass
                continue

    finally:
        # Cleanup temp directory
        try:
            shutil.rmtree(TEMP_DIR)
            print(f"\n🧹 Cleaned up temp directory: {TEMP_DIR}")
        except:
            pass

    print(f"\n✅ Processing complete!")
    print(f"   Successfully processed: {count} samples")
    print(f"   Failed: {failed_count} samples")
    print(f"   Total time: {int(time() - start_time_overall)}s")
    print(f"   Success rate: {count/(count+failed_count)*100:.1f}%")

    # Create final dataset with memory-efficient loading
    print("\n📦 Creating final dataset...")
    all_files = sorted(Path(SAMPLES_DIR).glob("*.npz"))

    # Use memory mapping for large datasets
    Xs, Ys, words = [], [], []
    for fpath in all_files:
        d = np.load(str(fpath), allow_pickle=True)
        Xs.append(d["X"])
        Ys.append(int(d["y"]))
        words.append(str(d.get("word", "")))

    final_path = os.path.join(OUT_DIR, f"msasl_train_landmarks_{LABEL_THRESHOLD}.npz")
    np.savez_compressed(
        final_path,
        X=np.array(Xs, dtype=object),
        y=np.array(Ys),
        words=np.array(words)
    )
    print(f"✅ Final dataset: {final_path}")
    print(f"   Samples: {len(Ys)}, Labels: {len(set(Ys))}")


for fpath in sorted(Path(SAMPLES_DIR).glob("*.npz")):
    try:
        d = np.load(str(fpath), allow_pickle=True)
        Xs.append(d["X"])
        Ys.append(int(d["y"]))
        words.append(str(d.get("word", "")))
    except Exception as e:
        print(f"⚠️ Skipping corrupt file: {fpath.name}")
        corrupt += 1

print(f"\n✅ Loaded {len(Xs)} valid samples ({corrupt} corrupt skipped)")
np.savez_compressed(OUT_PATH, X=np.array(Xs, dtype=object), y=np.array(Ys), words=np.array(words))
print(f"✅ Final dataset saved to: {OUT_PATH}")

