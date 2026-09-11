"""
Compute joint angles, compensation metrics and movement velocity for each
FMA movement, on the affected side defined in config.AFFECTED_SIDE.

For each movement this script:
  1. Computes the relevant joint angle(s) from 3D landmark positions.
  2. Detects repetition boundaries (peaks or valleys in the primary signal).
  3. Computes raw and smoothed movement velocity.
  4. Flags compensatory patterns according to movement-specific rules
     (see FLAG NOTES below).
  5. Saves a per-frame CSV and a multi-panel summary figure per movement.

FLAG NOTES
----------
Two flag types are used throughout:

  - Rep-level (peak-based): evaluated once per repetition, at its peak
    value. Used for ROM criteria (e.g. "did this repetition reach 90 degrees
    at its peak?"). A single bad repetition should not be diluted by
    averaging across the whole movement.

  - Live (instantaneous): evaluated every frame. Used for compensation
    markers that are only meaningful during active movement (e.g. shoulder
    elevation only counts as compensation while the arm is actively
    abducting, not throughout the whole recording).

Output: kinematics/<movement>_kinematics.csv and kinematics/<movement>_analysis.png

Run:
    python 06_compute_kinematics.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, savgol_filter

import config

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 13,
})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JOINTS_DIR = os.path.join(BASE_DIR, config.DIR_JOINTS3D)
OUT_DIR = os.path.join(BASE_DIR, config.DIR_KINEMATICS)
os.makedirs(OUT_DIR, exist_ok=True)

SIDE = config.AFFECTED_SIDE.upper()
assert SIDE in ("LEFT", "RIGHT"), "config.AFFECTED_SIDE must be 'LEFT' or 'RIGHT'"

REP_COLORS = ["#2980B9", "#27AE60", "#E74C3C"]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def get_point(frame_df, landmark):
    """pull one landmark's xyz for a single frame, or none if it's missing."""
    row = frame_df[frame_df["landmark"] == landmark]
    return row[["X", "Y", "Z"]].values[0] if len(row) else None


def angle_3pts(a, b, c):
    """angle at vertex b, formed by points a-b-c, in degrees."""
    ba, bc = a - b, c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))


def vector_angle(v1, v2):
    """angle between two vectors, in degrees."""
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))


def filter_artifacts(series, max_delta=25):
    """drop sudden jumps and out-of-range values, then fill the gaps by interpolation."""
    series = series.astype(float)
    delta = np.abs(np.diff(series, prepend=series[0]))
    series[delta > max_delta] = np.nan
    series[series < -5] = np.nan
    series[series > 185] = np.nan

    nan_mask = np.isnan(series)
    if nan_mask.any() and (~nan_mask).sum() > 3:
        idx = np.arange(len(series))
        interp = interp1d(idx[~nan_mask], series[~nan_mask], kind="linear",
                           fill_value="extrapolate", bounds_error=False)
        series[nan_mask] = interp(idx[nan_mask])
    return series


def smooth(series, window=11, polyorder=3):
    """savitzky-golay smoothing that interpolates over nans first so they don't break the filter."""
    series = np.array(series, dtype=float)
    if len(series) < window:
        return series

    nan_mask = np.isnan(series)
    if nan_mask.all():
        return series
    if nan_mask.any():
        idx = np.arange(len(series))
        series[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], series[~nan_mask])

    return savgol_filter(series, window, polyorder)


def detect_repetitions(curve, n_reps=3, kind="peaks"):
    """find the peaks (or valleys) that mark each repetition, keeping the most prominent ones."""
    source = -curve if kind == "valleys" else curve
    prominence = np.ptp(curve) * 0.20
    distance = max(len(curve) // (n_reps * 3), 10)
    peaks, props = find_peaks(source, prominence=prominence, distance=distance)

    if len(peaks) > n_reps:
        order = np.argsort(props["prominences"])[::-1]
        peaks = np.sort(peaks[order[:n_reps]])
    return peaks


def assign_rep_ids(n_frames, peaks):
    """label every frame with the repetition it belongs to, split at the midpoints between peaks."""
    ids = np.zeros(n_frames, dtype=int)
    if len(peaks) == 0:
        return ids

    bounds = [(peaks[i] + peaks[i + 1]) // 2 for i in range(len(peaks) - 1)]
    for i in range(n_frames):
        for rep_idx, peak in enumerate(peaks):
            start = bounds[rep_idx - 1] if rep_idx > 0 else 0
            end = bounds[rep_idx] if rep_idx < len(bounds) else n_frames
            if start <= i < end:
                ids[i] = rep_idx + 1
                break
        else:
            if i >= (bounds[-1] if bounds else peaks[-1]):
                ids[i] = len(peaks)
    return ids


def velocity_raw(values, times):
    """frame-to-frame rate of change, as an absolute value."""
    return np.abs(np.gradient(values, times))


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def style_axis(ax):
    """shared background and grid styling for the plots."""
    ax.set_facecolor("#F8F9FA")
    ax.grid(True, alpha=0.35, color="#BDC3C7")
    for spine in ax.spines.values():
        spine.set_edgecolor("#BDC3C7")


def mark_repetitions(ax, times, values, rep_peaks, label_offset=6):
    """draw a marker and a label at each repetition peak."""
    for i, peak in enumerate(rep_peaks):
        color = REP_COLORS[i % len(REP_COLORS)]
        ax.axvline(times[peak], color=color, lw=1.5, ls="--", alpha=0.7)
        ax.scatter([times[peak]], [values[peak]], color=color, s=80, zorder=5)
        ax.annotate(f"Rep {i + 1}\n{values[peak]:.1f}",
                     xy=(times[peak], values[peak]), xytext=(4, label_offset),
                     textcoords="offset points", color=color, fontsize=7.5,
                     fontweight="bold")


def make_table(ax, rows, col_labels, title):
    """draw the per-repetition summary table."""
    ax.axis("off")
    if not rows:
        ax.text(0.5, 0.5, "Repetitions not detected", ha="center", va="center",
                 color="#E74C3C", fontsize=10, transform=ax.transAxes)
        return

    table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)
    for (r, c), cell in table.get_celld().items():
        cell.set_facecolor("#D6EAF8" if r == 0 else ("#EBF5FB" if r % 2 == 0 else "white"))
        cell.set_text_props(color="#2C3E50", fontweight="bold" if r == 0 else "normal")
        cell.set_edgecolor("#AED6F1")
    ax.set_title(title, fontsize=10, pad=12, color="#2C3E50", fontweight="bold")


# ---------------------------------------------------------------------------
# M1 - Volitional Synergy UE
# ---------------------------------------------------------------------------

def compute_m1(movement_name):
    """m1: shoulder abduction, elbow flexion and shoulder elevation (compensation), plus velocity and repetitions."""
    print(f"  Computing M1 ({SIDE} side)...")
    df = pd.read_csv(os.path.join(JOINTS_DIR, f"{movement_name}_3d.csv"))
    frames = sorted(df["frame"].unique())

    shoulder_lm = f"{SIDE}_SHOULDER"
    elbow_lm = f"{SIDE}_ELBOW"
    wrist_lm = f"{SIDE}_WRIST"
    other_shoulder_lm = "RIGHT_SHOULDER" if SIDE == "LEFT" else "LEFT_SHOULDER"
    hip_lm = f"{SIDE}_HIP"
    other_hip_lm = "RIGHT_HIP" if SIDE == "LEFT" else "LEFT_HIP"

    ref_y_values = []
    for frame in frames[:10]:
        frame_df = df[df["frame"] == frame]
        s = get_point(frame_df, shoulder_lm)
        s_other = get_point(frame_df, other_shoulder_lm)
        if s is not None and s_other is not None:
            ref_y_values.append((s[1] + s_other[1]) / 2)
    ref_y = np.mean(ref_y_values) if ref_y_values else 0

    times, abduction, elbow_flexion, elevation = [], [], [], []

    for frame in frames:
        frame_df = df[df["frame"] == frame]
        times.append(frame_df["time_sec"].iloc[0])

        s = get_point(frame_df, shoulder_lm)
        e = get_point(frame_df, elbow_lm)
        w = get_point(frame_df, wrist_lm)
        s_other = get_point(frame_df, other_shoulder_lm)
        h = get_point(frame_df, hip_lm)
        h_other = get_point(frame_df, other_hip_lm)

        if s is not None and e is not None and s_other is not None and h is not None and h_other is not None:
            trunk_up = (s + s_other) / 2 - (h + h_other) / 2
            abduction.append(180 - vector_angle(e - s, trunk_up))
        else:
            abduction.append(np.nan)

        if s is not None and e is not None and w is not None:
            elbow_flexion.append(180 - angle_3pts(s, e, w))
        else:
            elbow_flexion.append(np.nan)

        if s is not None and s_other is not None:
            mid_y = (s[1] + s_other[1]) / 2
            elev = (ref_y - mid_y) * 100
            elevation.append(elev if abs(elev) < 15 else np.nan)
        else:
            elevation.append(np.nan)

    times = np.array(times)
    abduction = smooth(filter_artifacts(np.array(abduction), max_delta=20))
    elbow_flexion = smooth(filter_artifacts(np.array(elbow_flexion), max_delta=20))
    elevation = smooth(filter_artifacts(np.array(elevation), max_delta=8))

    vel_raw = velocity_raw(abduction, times)
    vel_smooth = smooth(vel_raw)

    reps = detect_repetitions(abduction, n_reps=3, kind="peaks")
    rep_ids = assign_rep_ids(len(frames), reps)

    out_df = pd.DataFrame({
        "frame": frames,
        "time_sec": times,
        "shoulder_abduction": abduction,
        "elbow_flexion": elbow_flexion,
        "shoulder_elevation": elevation,
        "velocity_raw": vel_raw,
        "velocity": vel_smooth,
        "rep_id": rep_ids,
    })
    out_df.to_csv(os.path.join(OUT_DIR, "m1_kinematics.csv"), index=False)
    print(f"    Repetitions at frames: {[frames[p] for p in reps]}")

    _plot_m1(times, abduction, elbow_flexion, elevation, vel_raw, vel_smooth, reps, frames)


def _plot_m1(times, abduction, elbow_flexion, elevation, vel_raw, vel_smooth, reps, frames):
    """build the m1 summary figure."""
    fig = plt.figure(figsize=(18, 9), facecolor="white")
    fig.suptitle(f"M1 - Volitional Synergy UE ({SIDE} side) | Kinematic Analysis",
                 fontsize=13, fontweight="bold", color="#2C3E50", y=0.98)
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.50, wspace=0.38,
                            top=0.92, bottom=0.08, left=0.05, right=0.97)

    ax1 = fig.add_subplot(gs[0, 0])
    style_axis(ax1)
    ax1.plot(times, abduction, color="#2980B9", lw=1.8)
    ax1.fill_between(times, 0, abduction, alpha=0.12, color="#2980B9")
    ax1.axhline(90, color="#F39C12", lw=1.3, ls="--", label="90 deg target")
    ax1.fill_between(times, 0, abduction, where=np.array(abduction) < 90,
                      alpha=0.18, color="#F39C12", label="Below target")
    mark_repetitions(ax1, times, abduction, reps)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Degrees")
    ax1.set_title(f"Shoulder Abduction ({SIDE})")
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    style_axis(ax2)
    ax2.plot(times, elbow_flexion, color="#27AE60", lw=1.8)
    ax2.fill_between(times, 0, elbow_flexion, alpha=0.12, color="#27AE60")
    elbow_peaks = detect_repetitions(elbow_flexion, n_reps=3, kind="peaks")
    mark_repetitions(ax2, times, elbow_flexion, elbow_peaks)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Degrees")
    ax2.set_title(f"Elbow Flexion ({SIDE}, 0 deg = extended)")

    ax3 = fig.add_subplot(gs[0, 2])
    style_axis(ax3)
    ax3.plot(times, elevation, color="#E74C3C", lw=1.8)
    ax3.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax3.axhline(3, color="#E74C3C", lw=1.3, ls="--", label="3cm threshold")
    ax3.fill_between(times, 3, elevation, where=np.array(elevation) > 3,
                      alpha=0.25, color="#E74C3C", label="Compensation")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("cm")
    ax3.set_title(f"Shoulder Elevation ({SIDE}, compensation)")
    ax3.legend()

    ax4 = fig.add_subplot(gs[0, 3])
    style_axis(ax4)
    ax4.plot(times, vel_raw, color="#BDC3C7", lw=1.0, alpha=0.7, label="Raw")
    ax4.plot(times, vel_smooth, color="#8E44AD", lw=2.0, label="Smoothed")
    ax4.fill_between(times, 0, vel_smooth, alpha=0.12, color="#8E44AD")
    cv = np.nanstd(vel_raw) / (np.nanmean(vel_raw) + 1e-8) * 100
    smoothness = "Smooth" if cv < 80 else "Irregular"
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("deg/s")
    ax4.set_title(f"Movement Velocity\n{smoothness} (CV={cv:.0f}%)")
    ax4.legend()

    ax5 = fig.add_subplot(gs[1, 1:])
    rows = []
    for i, peak in enumerate(reps):
        start = reps[i - 1] if i > 0 else 0
        end = reps[i + 1] if i < len(reps) - 1 else len(frames) - 1
        peak_abd = np.nanmax(abduction[start:end + 1])
        peak_elb = np.nanmax(elbow_flexion[start:end + 1])
        peak_elev = np.nanmax(elevation[start:end + 1])
        peak_vel = np.nanmax(vel_smooth[start:end + 1])
        rows.append([
            f"Rep {i + 1}",
            f"{peak_abd:.1f} deg {'(below)' if peak_abd < 90 else '(ok)'}",
            f"{peak_elb:.1f} deg",
            f"{peak_elev:.1f} cm {'(comp)' if peak_elev > 3 else '(ok)'}",
            f"{peak_vel:.1f} deg/s",
        ])

    make_table(ax5, rows,
               ["Rep", "Peak Shoulder\nAbduction", "Peak Elbow\nFlexion",
                "Max Shoulder\nElevation", "Peak\nVelocity"],
               "ROM per Repetition - M1")

    fig.add_subplot(gs[1, 0]).axis("off")
    plt.savefig(os.path.join(OUT_DIR, "m1_analysis.png"), dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close()


# ---------------------------------------------------------------------------
# M2 - Hand to Lumbar Spine
# ---------------------------------------------------------------------------

def compute_m2(movement_name):
    """m2: wrist-to-lumbar distance, shoulder extension and trunk flexion (compensation), plus velocity and repetitions."""
    print(f"  Computing M2 ({SIDE} side)...")
    df = pd.read_csv(os.path.join(JOINTS_DIR, f"{movement_name}_3d.csv"))
    frames = sorted(df["frame"].unique())

    shoulder_lm = f"{SIDE}_SHOULDER"
    elbow_lm = f"{SIDE}_ELBOW"
    wrist_lm = f"{SIDE}_WRIST"
    other_shoulder_lm = "RIGHT_SHOULDER" if SIDE == "LEFT" else "LEFT_SHOULDER"
    hip_lm = f"{SIDE}_HIP"
    other_hip_lm = "RIGHT_HIP" if SIDE == "LEFT" else "LEFT_HIP"

    times, wrist_dist, shoulder_ext, trunk_flex = [], [], [], []

    for frame in frames:
        frame_df = df[df["frame"] == frame]
        times.append(frame_df["time_sec"].iloc[0])

        s = get_point(frame_df, shoulder_lm)
        e = get_point(frame_df, elbow_lm)
        w = get_point(frame_df, wrist_lm)
        s_other = get_point(frame_df, other_shoulder_lm)
        h = get_point(frame_df, hip_lm)
        h_other = get_point(frame_df, other_hip_lm)

        if w is not None and h is not None and h_other is not None:
            wrist_dist.append(np.linalg.norm(w - (h + h_other) / 2) * 100)
        else:
            wrist_dist.append(np.nan)

        if s is not None and e is not None and s_other is not None and h is not None and h_other is not None:
            trunk_up = (s + s_other) / 2 - (h + h_other) / 2
            shoulder_ext.append(180 - vector_angle(e - s, trunk_up))
        else:
            shoulder_ext.append(np.nan)

        if s is not None and s_other is not None and h is not None and h_other is not None:
            trunk_vec = (s + s_other) / 2 - (h + h_other) / 2
            trunk_flex.append(vector_angle(trunk_vec, np.array([0, -1, 0])))
        else:
            trunk_flex.append(np.nan)

    times = np.array(times)
    wrist_dist = smooth(filter_artifacts(np.array(wrist_dist), max_delta=15))
    shoulder_ext = smooth(filter_artifacts(np.array(shoulder_ext), max_delta=20))
    trunk_flex = smooth(filter_artifacts(np.array(trunk_flex), max_delta=10))

    vel_raw = velocity_raw(wrist_dist, times)
    vel_smooth = smooth(vel_raw)

    reps = detect_repetitions(wrist_dist, n_reps=3, kind="valleys")
    rep_ids = assign_rep_ids(len(frames), reps)

    out_df = pd.DataFrame({
        "frame": frames,
        "time_sec": times,
        "wrist_lumbar_dist": wrist_dist,
        "shoulder_extension": shoulder_ext,
        "trunk_flexion": trunk_flex,
        "velocity_raw": vel_raw,
        "velocity": vel_smooth,
        "rep_id": rep_ids,
    })
    out_df.to_csv(os.path.join(OUT_DIR, "m2_kinematics.csv"), index=False)
    print(f"    Repetitions at frames: {[frames[p] for p in reps]}")

    _plot_m2(times, wrist_dist, shoulder_ext, trunk_flex, vel_raw, vel_smooth, reps, frames)


def _plot_m2(times, wrist_dist, shoulder_ext, trunk_flex, vel_raw, vel_smooth, reps, frames):
    """build the m2 summary figure."""
    fig = plt.figure(figsize=(18, 9), facecolor="white")
    fig.suptitle(f"M2 - Hand to Lumbar Spine ({SIDE} side) | Kinematic Analysis",
                 fontsize=13, fontweight="bold", color="#2C3E50", y=0.98)
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.50, wspace=0.38,
                            top=0.92, bottom=0.08, left=0.05, right=0.97)

    ax1 = fig.add_subplot(gs[0, 0])
    style_axis(ax1)
    ax1.plot(times, wrist_dist, color="#2980B9", lw=1.8)
    ax1.fill_between(times, 0, wrist_dist, alpha=0.12, color="#2980B9")
    ax1.axhline(30, color="#F39C12", lw=1.3, ls="--", label="30cm target")
    ax1.fill_between(times, 30, wrist_dist, where=np.array(wrist_dist) > 30,
                      alpha=0.18, color="#F39C12", label="Incomplete reach")
    mark_repetitions(ax1, times, wrist_dist, reps, label_offset=-20)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("cm")
    ax1.set_title(f"Wrist to Lumbar Distance ({SIDE})")
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    style_axis(ax2)
    ax2.plot(times, shoulder_ext, color="#27AE60", lw=1.8)
    ax2.fill_between(times, 0, shoulder_ext, alpha=0.12, color="#27AE60")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Degrees")
    ax2.set_title(f"Shoulder Extension ({SIDE})")

    ax3 = fig.add_subplot(gs[0, 2])
    style_axis(ax3)
    ax3.plot(times, trunk_flex, color="#E74C3C", lw=1.8)
    ax3.axhline(10, color="#E74C3C", lw=1.3, ls="--", label="10 deg threshold")
    ax3.fill_between(times, 10, trunk_flex, where=np.array(trunk_flex) > 10,
                      alpha=0.25, color="#E74C3C", label="Trunk compensation")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Degrees")
    ax3.set_title("Trunk Flexion (flagged when > 10 deg)")
    ax3.legend()

    ax4 = fig.add_subplot(gs[0, 3])
    style_axis(ax4)
    ax4.plot(times, vel_raw, color="#BDC3C7", lw=1.0, alpha=0.7, label="Raw")
    ax4.plot(times, vel_smooth, color="#8E44AD", lw=2.0, label="Smoothed")
    ax4.fill_between(times, 0, vel_smooth, alpha=0.12, color="#8E44AD")
    cv = np.nanstd(vel_raw) / (np.nanmean(vel_raw) + 1e-8) * 100
    smoothness = "Smooth" if cv < 80 else "Irregular"
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("cm/s")
    ax4.set_title(f"Hand Velocity\n{smoothness} (CV={cv:.0f}%)")
    ax4.legend()

    ax5 = fig.add_subplot(gs[1, 1:])
    rows = []
    for i, peak in enumerate(reps):
        start = reps[i - 1] if i > 0 else 0
        end = reps[i + 1] if i < len(reps) - 1 else len(frames) - 1
        min_dist = np.nanmin(wrist_dist[start:end + 1])
        max_ext = np.nanmax(shoulder_ext[start:end + 1])
        max_trunk = np.nanmax(trunk_flex[start:end + 1])
        peak_vel = np.nanmax(vel_smooth[start:end + 1])
        rows.append([
            f"Rep {i + 1}",
            f"{min_dist:.1f} cm {'(incomplete)' if min_dist > 30 else '(ok)'}",
            f"{max_ext:.1f} deg",
            f"{max_trunk:.1f} deg {'(comp)' if max_trunk > 10 else '(ok)'}",
            f"{peak_vel:.1f} cm/s",
        ])

    make_table(ax5, rows,
               ["Rep", "Min Wrist-Lumbar\nDistance", "Peak Shoulder\nExtension",
                "Max Trunk\nFlexion", "Peak\nVelocity"],
               "ROM per Repetition - M2")

    fig.add_subplot(gs[1, 0]).axis("off")
    plt.savefig(os.path.join(OUT_DIR, "m2_analysis.png"), dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close()


# ---------------------------------------------------------------------------
# M4 - Knee Flexion in Standing
# ---------------------------------------------------------------------------

def compute_m4(movement_name):
    """m4: knee flexion, hip flexion and pelvic tilt (both compensation), plus velocity and repetitions."""
    print(f"  Computing M4 ({SIDE} side)...")
    df = pd.read_csv(os.path.join(JOINTS_DIR, f"{movement_name}_3d.csv"))
    frames = sorted(df["frame"].unique())

    hip_lm = f"{SIDE}_HIP"
    knee_lm = f"{SIDE}_KNEE"
    ankle_lm = f"{SIDE}_ANKLE"
    shoulder_lm = f"{SIDE}_SHOULDER"
    other_shoulder_lm = "RIGHT_SHOULDER" if SIDE == "LEFT" else "LEFT_SHOULDER"
    other_hip_lm = "RIGHT_HIP" if SIDE == "LEFT" else "LEFT_HIP"

    times, knee_flexion, hip_flexion, pelvis_tilt = [], [], [], []

    for frame in frames:
        frame_df = df[df["frame"] == frame]
        times.append(frame_df["time_sec"].iloc[0])

        h = get_point(frame_df, hip_lm)
        k = get_point(frame_df, knee_lm)
        a = get_point(frame_df, ankle_lm)
        s = get_point(frame_df, shoulder_lm)
        s_other = get_point(frame_df, other_shoulder_lm)
        h_other = get_point(frame_df, other_hip_lm)

        if h is not None and k is not None and a is not None:
            knee_flexion.append(180 - angle_3pts(h, k, a))
        else:
            knee_flexion.append(np.nan)

        if h is not None and k is not None and s is not None and s_other is not None and h_other is not None:
            trunk_up = (s + s_other) / 2 - (h + h_other) / 2
            thigh_vec = k - h
            hip_flexion.append(180 - vector_angle(trunk_up, thigh_vec))
        else:
            hip_flexion.append(np.nan)

        if h is not None and h_other is not None:
            pelvis_tilt.append(abs(h[1] - h_other[1]) * 100)
        else:
            pelvis_tilt.append(np.nan)

    times = np.array(times)
    knee_flexion = smooth(filter_artifacts(np.array(knee_flexion), max_delta=20))
    hip_flexion = smooth(filter_artifacts(np.array(hip_flexion), max_delta=15))
    pelvis_tilt = smooth(filter_artifacts(np.array(pelvis_tilt), max_delta=5))

    vel_raw = velocity_raw(knee_flexion, times)
    vel_smooth = smooth(vel_raw)

    reps = detect_repetitions(knee_flexion, n_reps=3, kind="peaks")
    rep_ids = assign_rep_ids(len(frames), reps)

    out_df = pd.DataFrame({
        "frame": frames,
        "time_sec": times,
        "knee_flexion": knee_flexion,
        "hip_flexion": hip_flexion,
        "pelvis_tilt": pelvis_tilt,
        "velocity_raw": vel_raw,
        "velocity": vel_smooth,
        "rep_id": rep_ids,
    })
    out_df.to_csv(os.path.join(OUT_DIR, "m4_kinematics.csv"), index=False)
    print(f"    Repetitions at frames: {[frames[p] for p in reps]}")

    _plot_m4(times, knee_flexion, hip_flexion, pelvis_tilt, vel_raw, vel_smooth, reps, frames)


def _plot_m4(times, knee_flexion, hip_flexion, pelvis_tilt, vel_raw, vel_smooth, reps, frames):
    """build the m4 summary figure."""
    fig = plt.figure(figsize=(18, 9), facecolor="white")
    fig.suptitle(f"M4 - Volitional Movement, No Synergy, Standing ({SIDE} side) | Kinematic Analysis",
                 fontsize=13, fontweight="bold", color="#2C3E50", y=0.98)
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.50, wspace=0.38,
                            top=0.92, bottom=0.08, left=0.05, right=0.97)

    ax1 = fig.add_subplot(gs[0, 0])
    style_axis(ax1)
    ax1.plot(times, knee_flexion, color="#2980B9", lw=1.8)
    ax1.fill_between(times, 0, knee_flexion, alpha=0.12, color="#2980B9")
    ax1.axhline(90, color="#F39C12", lw=1.5, ls="--", label="Target 90 deg")
    ax1.fill_between(times, 0, knee_flexion, where=np.array(knee_flexion) < 90,
                      alpha=0.18, color="#F39C12", label="Below 90 deg")
    mark_repetitions(ax1, times, knee_flexion, reps)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Degrees")
    ax1.set_title(f"Knee Flexion ({SIDE})")
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    style_axis(ax2)
    ax2.plot(times, hip_flexion, color="#E74C3C", lw=1.8)
    ax2.axhline(15, color="#E74C3C", lw=1.5, ls="--", label="FMA threshold (15 deg)")
    active_mask = np.array(knee_flexion) > 20
    ax2.fill_between(times, 15, hip_flexion, where=(np.array(hip_flexion) > 15) & active_mask,
                      alpha=0.25, color="#E74C3C", label="Hip compensation")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Degrees")
    ax2.set_title(f"Hip Flexion ({SIDE}, flagged only during knee movement)")
    ax2.legend()

    ax3 = fig.add_subplot(gs[0, 2])
    style_axis(ax3)
    ax3.plot(times, pelvis_tilt, color="#F39C12", lw=1.8)
    ax3.axhline(2, color="#E74C3C", lw=1.3, ls="--", label="2cm threshold")
    ax3.fill_between(times, 2, pelvis_tilt, where=np.array(pelvis_tilt) > 2,
                      alpha=0.25, color="#E74C3C", label="Pelvic instability")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("cm")
    ax3.set_title("Pelvic Tilt (flagged when > 2cm)")
    ax3.legend()

    ax4 = fig.add_subplot(gs[0, 3])
    style_axis(ax4)
    ax4.plot(times, vel_raw, color="#BDC3C7", lw=1.0, alpha=0.7, label="Raw")
    ax4.plot(times, vel_smooth, color="#8E44AD", lw=2.0, label="Smoothed")
    ax4.fill_between(times, 0, vel_smooth, alpha=0.12, color="#8E44AD")
    cv = np.nanstd(vel_raw) / (np.nanmean(vel_raw) + 1e-8) * 100
    smoothness = "Smooth" if cv < 80 else "Irregular"
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("deg/s")
    ax4.set_title(f"Knee Velocity\n{smoothness} (CV={cv:.0f}%)")
    ax4.legend()

    ax5 = fig.add_subplot(gs[1, 1:])
    rows = []
    for i, peak in enumerate(reps):
        start = reps[i - 1] if i > 0 else 0
        end = reps[i + 1] if i < len(reps) - 1 else len(frames) - 1
        peak_knee = np.nanmax(knee_flexion[start:end + 1])
        active_seg = np.array(knee_flexion[start:end + 1]) > 20
        hip_seg = np.array(hip_flexion[start:end + 1])
        peak_hip = np.nanmax(hip_seg[active_seg]) if active_seg.any() else 0
        peak_tilt = np.nanmax(pelvis_tilt[start:end + 1])
        peak_vel = np.nanmax(vel_smooth[start:end + 1])
        rows.append([
            f"Rep {i + 1}",
            f"{peak_knee:.1f} deg {'(below)' if peak_knee < 90 else '(ok)'}",
            f"{peak_hip:.1f} deg {'(comp)' if peak_hip > 15 else '(ok)'}",
            f"{peak_tilt:.1f} cm {'(unstable)' if peak_tilt > 2 else '(ok)'}",
            f"{peak_vel:.1f} deg/s",
        ])

    make_table(ax5, rows,
               ["Rep", "Peak Knee\nFlexion", "Max Hip Flexion\n(during movement)",
                "Max Pelvic\nTilt", "Peak\nVelocity"],
               "ROM per Repetition - M4")

    fig.add_subplot(gs[1, 0]).axis("off")
    plt.savefig(os.path.join(OUT_DIR, "m4_analysis.png"), dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

COMPUTE_FUNCTIONS = {
    "M1": compute_m1,
    "M2": compute_m2,
    "M4": compute_m4,
}


def main():
    """run the kinematics for every configured movement that has a compute function."""
    print(f"Affected side: {SIDE}\n")
    for key, movement_cfg in config.MOVEMENTS.items():
        if key in COMPUTE_FUNCTIONS:
            COMPUTE_FUNCTIONS[key](movement_cfg["name"])

    print(f"\nDone. Files in: {OUT_DIR}")


if __name__ == "__main__":
    main()
