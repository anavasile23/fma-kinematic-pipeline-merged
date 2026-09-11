"""
Extract 2D pixel coordinates for relevant body landmarks from each camera's
trimmed movement clips, using MediaPipe PoseLandmarker.

Output: pose2d/<camera>_<movement>_pose2d.csv with columns:
    frame, time_sec, landmark, x_px, y_px, visibility

Run:
    python 03_extract_pose2d.py
"""

import csv
import os
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

import config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(BASE_DIR, config.DIR_VIDEOS_TRIMMED)
OUT_DIR = os.path.join(BASE_DIR, config.DIR_POSE2D)
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(BASE_DIR, config.POSE_MODEL_PATH)
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)

# Subset of the 33 MediaPipe landmarks relevant to FMA-UE/LE movements.
# Index reference: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
LANDMARKS_OF_INTEREST = {
    0: "NOSE",
    11: "LEFT_SHOULDER",
    12: "RIGHT_SHOULDER",
    13: "LEFT_ELBOW",
    14: "RIGHT_ELBOW",
    15: "LEFT_WRIST",
    16: "RIGHT_WRIST",
    17: "LEFT_PINKY",
    18: "RIGHT_PINKY",
    23: "LEFT_HIP",
    24: "RIGHT_HIP",
    25: "LEFT_KNEE",
    26: "RIGHT_KNEE",
    27: "LEFT_ANKLE",
    28: "RIGHT_ANKLE",
    29: "LEFT_HEEL",
    30: "RIGHT_HEEL",
}


def download_model():
    """fetch the mediapipe pose model on first run, skip it if already there."""
    if os.path.exists(MODEL_PATH):
        print("  Pose model found locally.")
        return
    print("  Downloading PoseLandmarker model (~30MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("  Done.")


def make_detector():
    """set up the mediapipe pose landmarker for single-person detection."""
    base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        num_poses=1,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def process_video(video_path, out_csv_path, detector):
    """run pose detection frame by frame and write the landmark pixel coordinates to csv."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    ERROR: cannot open {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"    {width}x{height} @ {fps:.2f}fps, {total} frames")

    rows = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        time_sec = frame_idx / fps
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            for idx, name in LANDMARKS_OF_INTEREST.items():
                lm = landmarks[idx]
                vis = lm.visibility if lm.visibility is not None else 0.0
                rows.append({
                    "frame": frame_idx,
                    "time_sec": round(time_sec, 4),
                    "landmark": name,
                    "x_px": round(lm.x * width, 2),
                    "y_px": round(lm.y * height, 2),
                    "visibility": round(vis, 3),
                })
        else:
            for idx, name in LANDMARKS_OF_INTEREST.items():
                rows.append({
                    "frame": frame_idx,
                    "time_sec": round(time_sec, 4),
                    "landmark": name,
                    "x_px": None,
                    "y_px": None,
                    "visibility": 0.0,
                })

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"      ... {frame_idx}/{total}")

    cap.release()

    fieldnames = ["frame", "time_sec", "landmark", "x_px", "y_px", "visibility"]
    with open(out_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_detected = sum(1 for r in rows if r["x_px"] is not None)
    n_possible = frame_idx * len(LANDMARKS_OF_INTEREST)
    pct = 100 * n_detected / n_possible if n_possible else 0
    print(f"    Saved {frame_idx} frames -> {out_csv_path}")
    print(f"    Detection rate: {n_detected}/{n_possible} ({pct:.1f}%)")

    return True


def main():
    """extract 2d landmarks for every movement clip of every camera."""
    download_model()
    detector = make_detector()

    movement_names = [m["name"] for m in config.MOVEMENTS.values()]

    for movement in movement_names:
        print(f"-- {movement} --")
        for cam in config.CAMERAS:
            video_name = f"{cam}_{movement}.mp4"
            video_path = os.path.join(VIDEOS_DIR, video_name)

            if not os.path.exists(video_path):
                print(f"  [{cam}] missing: {video_name}, skipping")
                continue

            out_path = os.path.join(OUT_DIR, f"{cam}_{movement}_pose2d.csv")
            print(f"  [{cam}] processing {video_name}...")
            process_video(video_path, out_path, detector)

    detector.close()
    print(f"\nDone. CSV files in: {OUT_DIR}")


if __name__ == "__main__":
    main()
