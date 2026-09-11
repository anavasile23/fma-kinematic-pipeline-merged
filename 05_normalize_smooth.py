"""
Rescale triangulated 3D coordinates to metric units and smooth trajectories.

Triangulation produces coordinates in an arbitrary scale (determined by the
essential-matrix decomposition, which is scale-ambiguous). This script:

  1. Computes a scale factor from the known shoulder width
     (LEFT_SHOULDER to RIGHT_SHOULDER distance), set in config.SHOULDER_WIDTH_M.
  2. Applies that scale to all coordinates.
  3. Smooths each landmark's trajectory with a Savitzky-Golay filter to
     reduce frame-to-frame jitter from pose estimation noise.
  4. Reports sanity checks against expected anthropometric proportions.

Output: joints3d_ready/<movement>_3d.csv (same schema as input, rescaled).

Run:
    python 05_normalize_smooth.py
"""

import os

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

import config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IN_DIR = os.path.join(BASE_DIR, config.DIR_JOINTS3D_RAW)
OUT_DIR = os.path.join(BASE_DIR, config.DIR_JOINTS3D)
os.makedirs(OUT_DIR, exist_ok=True)

SMOOTH_WINDOW = 11
SMOOTH_POLYORDER = 3

# Approximate expected segment lengths (metres) for sanity-checking the
# rescaled output. These are rough population averages, not subject-specific
# measurements; large deviations may indicate a triangulation or scale issue.
EXPECTED_SEGMENTS = [
    ("Shoulder width", "LEFT_SHOULDER", "RIGHT_SHOULDER", config.SHOULDER_WIDTH_M),
    ("Left forearm", "LEFT_ELBOW", "LEFT_WRIST", 0.27),
    ("Right forearm", "RIGHT_ELBOW", "RIGHT_WRIST", 0.27),
    ("Left shank", "LEFT_KNEE", "LEFT_ANKLE", 0.40),
    ("Right shank", "RIGHT_KNEE", "RIGHT_ANKLE", 0.40),
]


def compute_scale(df):
    """find the factor that turns the arbitrary triangulation units into metres, using the known shoulder width."""
    distances = []
    for frame in df["frame"].unique():
        frame_df = df[df["frame"] == frame]
        left = frame_df[frame_df["landmark"] == "LEFT_SHOULDER"][["X", "Y", "Z"]].values
        right = frame_df[frame_df["landmark"] == "RIGHT_SHOULDER"][["X", "Y", "Z"]].values
        if len(left) and len(right):
            distances.append(np.linalg.norm(left[0] - right[0]))

    if not distances:
        print("  WARNING: no shoulder landmarks found, scale not applied")
        return 1.0

    scale = config.SHOULDER_WIDTH_M / np.median(distances)
    print(f"  Scale factor: {scale:.4f} (raw shoulder width: {np.median(distances):.4f})")
    return scale


def apply_scale(df, scale):
    """multiply all coordinates by the scale factor."""
    df = df.copy()
    df["X"] *= scale
    df["Y"] *= scale
    df["Z"] *= scale
    return df


def smooth_trajectories(df):
    """smooth each landmark's x/y/z over time to take out the frame-to-frame jitter."""
    smoothed = []
    for landmark in df["landmark"].unique():
        sub = df[df["landmark"] == landmark].sort_values("frame").copy()
        if len(sub) >= SMOOTH_WINDOW:
            for axis in ["X", "Y", "Z"]:
                sub[axis] = savgol_filter(sub[axis].values, SMOOTH_WINDOW, SMOOTH_POLYORDER)
        smoothed.append(sub)
    return pd.concat(smoothed).sort_values(["frame", "landmark"]).reset_index(drop=True)


def sanity_check(df):
    """compare a few reconstructed segment lengths against rough expected values, to catch scale or triangulation problems."""
    for label, lm_a, lm_b, expected in EXPECTED_SEGMENTS:
        distances = []
        for frame in list(df["frame"].unique())[:100]:
            frame_df = df[df["frame"] == frame]
            a = frame_df[frame_df["landmark"] == lm_a][["X", "Y", "Z"]].values
            b = frame_df[frame_df["landmark"] == lm_b][["X", "Y", "Z"]].values
            if len(a) and len(b):
                distances.append(np.linalg.norm(a[0] - b[0]))

        if not distances:
            continue

        mean_dist = np.mean(distances)
        error_pct = abs(mean_dist - expected) / expected * 100
        flag = "OK" if error_pct < 20 else ("CHECK" if error_pct < 35 else "WARN")
        print(f"    {flag:5s} {label:<16} {mean_dist:.3f}m (expected ~{expected}m, {error_pct:.0f}% off)")


def process_movement(movement_name):
    """rescale and smooth one movement's 3d data and save it."""
    in_path = os.path.join(IN_DIR, f"{movement_name}_3d.csv")
    if not os.path.exists(in_path):
        print(f"  Missing: {in_path}")
        return

    df = pd.read_csv(in_path)
    print(f"  {len(df)} points, {df['frame'].nunique()} frames")

    scale = compute_scale(df)
    df = apply_scale(df, scale)
    df = smooth_trajectories(df)
    sanity_check(df)

    out_path = os.path.join(OUT_DIR, f"{movement_name}_3d.csv")
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")


def main():
    """rescale and smooth every movement."""
    print(f"Shoulder width reference: {config.SHOULDER_WIDTH_M} m\n")
    for movement_cfg in config.MOVEMENTS.values():
        print(f"-- {movement_cfg['name']} --")
        process_movement(movement_cfg["name"])

    print(f"\nDone. Output in: {OUT_DIR}")


if __name__ == "__main__":
    main()
