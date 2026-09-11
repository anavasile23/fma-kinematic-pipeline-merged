"""
Intrinsic camera calibration from a checkerboard pattern.

For each camera, extracts frames from its calibration clip, detects the
checkerboard, and computes the intrinsic camera matrix and distortion
coefficients via cv2.calibrateCamera.

Output: calib/<camera>_params.npz, each containing:
    camera_matrix : 3x3 intrinsic matrix
    dist_coeffs   : distortion coefficients
    rms_error     : calibration RMS reprojection error (pixels)

Run:
    python 02_calibrate_cameras.py
"""

import os

import cv2
import numpy as np

import config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(BASE_DIR, config.DIR_VIDEOS_TRIMMED)
CALIB_DIR = os.path.join(BASE_DIR, config.DIR_CALIB)
os.makedirs(CALIB_DIR, exist_ok=True)

FRAME_STEP = 10
MIN_VALID_FRAMES = 15
TARGET_RMS = 1.5


def extract_frames(video_path, step=FRAME_STEP):
    """grab every nth frame from the calibration clip, so we don't run detection on all of them."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {video_path}")

    frames, idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frames.append(frame)
        idx += 1
    cap.release()

    print(f"  Extracted {len(frames)} frames out of {idx} total")
    return frames


def calibrate_single(frames, cam_name):
    """find the checkerboard in each frame and solve for the camera intrinsics and distortion."""
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    cols, rows = config.CHECKERBOARD_SIZE
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= config.CHECKERBOARD_SQUARE_M

    obj_points, img_points = [], []
    img_shape = None

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(
            gray, config.CHECKERBOARD_SIZE,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if found:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners2)

    print(f"  Checkerboard detected in {len(img_points)}/{len(frames)} frames")

    if len(img_points) < MIN_VALID_FRAMES:
        raise RuntimeError(
            f"Too few valid frames for calibration: {len(img_points)} < {MIN_VALID_FRAMES}"
        )

    rms, K, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, img_shape, None, None)

    status = "good" if rms < TARGET_RMS else ("acceptable" if rms < 2.0 else "high")
    print(f"  RMS reprojection error: {rms:.4f} px ({status})")

    return K, dist, rms


def save_intrinsics(cam_name, K, dist, rms):
    """save one camera's calibration to an npz file."""
    path = os.path.join(CALIB_DIR, f"{cam_name}_params.npz")
    np.savez(path, camera_matrix=K, dist_coeffs=dist, rms_error=rms)
    print(f"  Saved: {path}")


def main():
    """calibrate every camera that has a calibration clip."""
    for cam in config.CAMERAS:
        video_path = os.path.join(VIDEOS_DIR, f"{cam}_calib.mp4")
        if not os.path.exists(video_path):
            print(f"[{cam}] missing calibration clip: {video_path}")
            continue

        print(f"-- {cam} --")
        frames = extract_frames(video_path)
        K, dist, rms = calibrate_single(frames, cam)
        save_intrinsics(cam, K, dist, rms)

    print(f"\nDone. Calibration files in: {CALIB_DIR}")


if __name__ == "__main__":
    main()
