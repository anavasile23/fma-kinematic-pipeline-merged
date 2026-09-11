"""
Triangulate 2D landmark detections from a camera pair into 3D coordinates.

For each movement, uses the camera pair defined in config.MOVEMENTS to:
  1. Match landmark detections between the two cameras by frame.
  2. Estimate relative camera pose (essential matrix + recoverPose).
  3. Triangulate matched points into 3D.
  4. Filter outliers based on depth consistency.

Coordinates are in an arbitrary scale at this stage; absolute scale is
recovered later in 05_normalize_smooth.py using a known body segment
length (shoulder width).

Output: joints3d_raw/<movement>_3d.csv with columns:
    frame, time_sec, landmark, X, Y, Z, vis_a, vis_b

Run:
    python 04_triangulate.py
"""

import os

import cv2
import numpy as np
import pandas as pd

import config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_DIR = os.path.join(BASE_DIR, config.DIR_CALIB)
POSE2D_DIR = os.path.join(BASE_DIR, config.DIR_POSE2D)
OUT_DIR = os.path.join(BASE_DIR, config.DIR_JOINTS3D_RAW)
os.makedirs(OUT_DIR, exist_ok=True)

MIN_VISIBILITY = config.POSE_MIN_VISIBILITY


def load_calibration(cam):
    """load one camera's intrinsic matrix and distortion."""
    data = np.load(os.path.join(CALIB_DIR, f"{cam}_params.npz"))
    return data["camera_matrix"], data["dist_coeffs"]


def load_pose2d(movement, cam):
    """load the 2d landmark csv for one camera and movement."""
    return pd.read_csv(os.path.join(POSE2D_DIR, f"{cam}_{movement}_pose2d.csv"))


def triangulate_pair(movement, cam_a, cam_b, K_a, dist_a, K_b, dist_b):
    """match landmarks between the two cameras, recover their relative pose and triangulate to 3d. the scale is arbitrary here, it gets fixed in step 05."""
    df_a = load_pose2d(movement, cam_a)
    df_b = load_pose2d(movement, cam_b)
    df_a = df_a[df_a["visibility"] >= MIN_VISIBILITY]
    df_b = df_b[df_b["visibility"] >= MIN_VISIBILITY]

    merged = df_a.merge(df_b, on=["frame", "landmark"], suffixes=("_a", "_b"))
    merged = merged.dropna(subset=["x_px_a", "y_px_a", "x_px_b", "y_px_b"])
    print(f"  Matched points across both cameras: {len(merged)}")

    pts_a = merged[["x_px_a", "y_px_a"]].values.astype(np.float32)
    pts_b = merged[["x_px_b", "y_px_b"]].values.astype(np.float32)

    pts_a_n = cv2.undistortPoints(pts_a.reshape(-1, 1, 2), K_a, dist_a).reshape(-1, 2)
    pts_b_n = cv2.undistortPoints(pts_b.reshape(-1, 1, 2), K_b, dist_b).reshape(-1, 2)

    E, inliers = cv2.findEssentialMat(
        pts_a_n, pts_b_n, focal=1.0, pp=(0.0, 0.0),
        method=cv2.RANSAC, prob=0.999, threshold=1e-3,
    )
    mask = inliers.ravel().astype(bool)
    print(f"  Pose estimation inliers: {mask.sum()}/{len(mask)} ({100 * mask.mean():.1f}%)")
    _, R, t, _ = cv2.recoverPose(E, pts_a_n[mask], pts_b_n[mask])

    P_a = np.hstack([np.eye(3), np.zeros((3, 1))])
    P_b = np.hstack([R, t])
    pts4d = cv2.triangulatePoints(P_a, P_b, pts_a_n.T, pts_b_n.T)
    pts3d = (pts4d[:3] / pts4d[3]).T

    z = pts3d[:, 2]
    z_med, z_std = np.median(z), np.std(z)
    valid = (z > 0) & (np.abs(z - z_med) < 5 * z_std)
    print(f"  Valid 3D points after depth filtering: {valid.sum()}/{len(valid)}")

    merged = merged.reset_index(drop=True)
    merged_valid = merged[valid].reset_index(drop=True)
    pts3d_valid = pts3d[valid]

    rows = []
    for i, row in merged_valid.iterrows():
        X, Y, Z = pts3d_valid[i]
        rows.append({
            "frame": int(row["frame"]),
            "time_sec": round(float(row["time_sec_a"]), 4),
            "landmark": row["landmark"],
            "X": round(float(X), 6),
            "Y": round(float(Y), 6),
            "Z": round(float(Z), 6),
            "vis_a": round(float(row["visibility_a"]), 3),
            "vis_b": round(float(row["visibility_b"]), 3),
        })
    return pd.DataFrame(rows)


def report_coverage(df3d):
    """print how many frames each landmark was reconstructed in."""
    n_frames = df3d["frame"].nunique()
    avg_landmarks = df3d.groupby("frame")["landmark"].count().mean()
    print(f"  {len(df3d)} points | {n_frames} frames | avg {avg_landmarks:.1f} landmarks/frame")
    for lm in sorted(df3d["landmark"].unique()):
        count = df3d[df3d["landmark"] == lm]["frame"].nunique()
        pct = 100 * count / n_frames
        flag = "OK" if pct > 80 else ("LOW" if pct > 50 else "MISSING")
        print(f"    {flag:7s} {lm:<20} {pct:.0f}%")


def main():
    """triangulate every movement using its configured camera pair."""
    calib_cache = {}
    for cam in config.CAMERAS:
        try:
            calib_cache[cam] = load_calibration(cam)
            print(f"  [{cam}] calibration loaded")
        except FileNotFoundError:
            print(f"  [{cam}] calibration not found")

    for movement_cfg in config.MOVEMENTS.values():
        movement = movement_cfg["name"]
        cam_a, cam_b = movement_cfg["camera_pair"]
        print(f"\n-- {movement} ({cam_a}+{cam_b}) --")

        if cam_a not in calib_cache or cam_b not in calib_cache:
            print("  SKIP: missing calibration for one or both cameras")
            continue

        K_a, dist_a = calib_cache[cam_a]
        K_b, dist_b = calib_cache[cam_b]

        df3d = triangulate_pair(movement, cam_a, cam_b, K_a, dist_a, K_b, dist_b)
        if df3d.empty:
            print("  SKIP: no valid 3D points")
            continue

        report_coverage(df3d)

        out_path = os.path.join(OUT_DIR, f"{movement}_3d.csv")
        df3d.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")

    print(f"\nDone. CSV files in: {OUT_DIR}")


if __name__ == "__main__":
    main()
