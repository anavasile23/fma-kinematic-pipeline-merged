"""
Interactive viewer for FMA kinematic analysis.

Displays, synchronised frame-by-frame:
  - Two camera feeds (with faces anonymised via a black rectangle over the
    NOSE landmark)
  - A 3D stickman reconstruction
  - Per-movement kinematic graphs with a moving time cursor
  - A live/repetition compensation flag panel

Controls:
    Left / Right arrow  - step one frame
    Space                - play / pause
    1, 2, 4              - switch to movement M1, M2, M4

Run:
    python 07_viewer.py
"""

import os

import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import numpy as np
import pandas as pd

import config

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
    "axes.titlesize": 8.5,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "figure.titlesize": 11,
})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(BASE_DIR, config.DIR_VIDEOS_TRIMMED)
JOINTS_DIR = os.path.join(BASE_DIR, config.DIR_JOINTS3D)
KIN_DIR = os.path.join(BASE_DIR, config.DIR_KINEMATICS)
POSE2D_DIR = os.path.join(BASE_DIR, config.DIR_POSE2D)

PLAYBACK_INTERVAL_MS = 16   # ~60fps timer
PLAYBACK_FRAME_STEP = 2     # frames advanced per timer tick


def build_movement_views():
    """turn config.MOVEMENTS into the per-movement display settings the viewer needs."""
    views = {}
    for key, movement_cfg in config.MOVEMENTS.items():
        name = movement_cfg["name"]
        cam_a, cam_b = movement_cfg["camera_pair"]
        views[key] = {
            "title": f"{key} - {movement_cfg['label']}",
            "csv_3d": f"{name}_3d.csv",
            "csv_kin": f"{key.lower()}_kinematics.csv",
            "cam_a": f"{cam_a}_{name}.mp4",
            "cam_b": f"{cam_b}_{name}.mp4",
            "pose2d_a": f"{cam_a}_{name}_pose2d.csv",
            "pose2d_b": f"{cam_b}_{name}_pose2d.csv",
            "cam_a_label": cam_a.upper(),
            "cam_b_label": cam_b.upper(),
            "graphs": config.VIEWER_GRAPHS[key],
            "velocity_label": config.VIEWER_VELOCITY_LABEL[key],
            "flags_rep": config.VIEWER_FLAGS_REP[key],
            "flags_live": config.VIEWER_FLAGS_LIVE[key],
        }
    return views


MOVEMENTS = build_movement_views()
ACCENT = config.MOVEMENT_ACCENT_COLOR
CONNECTIONS = config.SKELETON_CONNECTIONS

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

state = {
    "key": list(MOVEMENTS.keys())[0],
    "df3d": None, "df_kin": None,
    "frames": [], "frame_idx": 0,
    "cap_a": None, "cap_b": None,
    "nose_a": {}, "nose_b": {},
    "skeleton_artists": [], "midpoint": np.zeros(3), "range_": 0.5,
    "times": None, "kin_values": {}, "rep_peak_values": {},
    "orig_w": 1920, "orig_h": 1080,
    "playing": False,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_nose_positions(pose2d_csv):
    """load the nose pixel position per frame, filling gaps so every frame has one (used to cover the face)."""
    path = os.path.join(POSE2D_DIR, pose2d_csv)
    if not os.path.exists(path):
        return {}

    df = pd.read_csv(path)
    nose = df[(df["landmark"] == "NOSE") & (df["visibility"] >= 0.3)]
    nose = nose.dropna(subset=["x_px", "y_px"])
    if nose.empty:
        return {}

    all_frames = sorted(df["frame"].unique())
    detected = {int(r["frame"]): (float(r["x_px"]), float(r["y_px"]))
                for _, r in nose.iterrows()}

    result = {}
    last = None
    for f in all_frames:
        if f in detected:
            last = detected[f]
        if last:
            result[f] = last

    if detected:
        first_value = detected[min(detected.keys())]
        for f in all_frames:
            result.setdefault(f, first_value)

    if result:
        xs = [v[0] for v in result.values()]
        ys = [v[1] for v in result.values()]
        median_pos = (float(np.median(xs)), float(np.median(ys)))
        for f in all_frames:
            result.setdefault(f, median_pos)

    return result


def load_movement(key):
    """load the 3d data, kinematics, videos and per-repetition peaks for one movement into the shared state."""
    movement_cfg = MOVEMENTS[key]

    df3d = pd.read_csv(os.path.join(JOINTS_DIR, movement_cfg["csv_3d"]))
    frames = sorted(df3d["frame"].unique())
    coords = df3d[["X", "Y", "Z"]].values
    midpoint = coords.mean(axis=0)
    range_ = max(np.ptp(coords, axis=0).max() / 2, 0.1) * 1.2

    kin_path = os.path.join(KIN_DIR, movement_cfg["csv_kin"])
    rep_peak_values = {}
    if os.path.exists(kin_path):
        df_kin = pd.read_csv(kin_path)
        times = df_kin["time_sec"].values
        kin_values = {}
        for g in movement_cfg["graphs"]:
            if g["col"] in df_kin.columns:
                kin_values[g["col"]] = df_kin[g["col"]].values
        for col in ["velocity", "velocity_raw"]:
            if col in df_kin.columns:
                kin_values[col] = df_kin[col].values

        if "rep_id" in df_kin.columns:
            for rep_id in sorted(df_kin["rep_id"].unique()):
                if rep_id == 0:
                    continue
                mask = df_kin["rep_id"] == rep_id
                rep_peak_values[rep_id] = {}
                for col in kin_values:
                    if col in ("velocity", "velocity_raw"):
                        continue
                    segment = df_kin.loc[mask, col].values
                    if len(segment):
                        rep_peak_values[rep_id][col] = (
                            np.nanmin(segment) if "dist" in col else np.nanmax(segment)
                        )
    else:
        df_kin = pd.DataFrame()
        times = np.linspace(0, len(frames) / 30, len(frames))
        kin_values = {}

    if state["cap_a"]:
        state["cap_a"].release()
    if state["cap_b"]:
        state["cap_b"].release()

    cap_a = cv2.VideoCapture(os.path.join(VIDEOS_DIR, movement_cfg["cam_a"]))
    cap_b = cv2.VideoCapture(os.path.join(VIDEOS_DIR, movement_cfg["cam_b"]))
    orig_w = int(cap_a.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap_a.get(cv2.CAP_PROP_FRAME_HEIGHT))

    state.update({
        "key": key, "df3d": df3d, "df_kin": df_kin,
        "frames": frames, "frame_idx": 0,
        "cap_a": cap_a, "cap_b": cap_b,
        "nose_a": load_nose_positions(movement_cfg["pose2d_a"]),
        "nose_b": load_nose_positions(movement_cfg["pose2d_b"]),
        "midpoint": midpoint, "range_": range_,
        "times": times, "kin_values": kin_values,
        "rep_peak_values": rep_peak_values,
        "orig_w": orig_w, "orig_h": orig_h,
    })


# ---------------------------------------------------------------------------
# Video frame + face anonymisation
# ---------------------------------------------------------------------------

def cover_face(frame, nose_x, nose_y, orig_w, orig_h):
    """black out a rectangle over the face, scaled to the displayed frame size."""
    h, w = frame.shape[:2]
    if orig_w == 0 or orig_h == 0:
        return frame

    scale_x, scale_y = w / orig_w, h / orig_h
    cx, cy = int(nose_x * scale_x), int(nose_y * scale_y)
    half_w = int(config.FACE_COVER_HALF_WIDTH_PX * scale_x)
    half_h = int(config.FACE_COVER_HALF_HEIGHT_PX * scale_y)
    cy -= int(half_h * config.FACE_COVER_Y_SHIFT)

    frame[max(0, cy - half_h):min(h, cy + half_h),
          max(0, cx - half_w):min(w, cx + half_w)] = 0
    return frame


def get_video_frame(cap, frame_idx, nose_positions, orig_w, orig_h):
    """read one frame, resize it for display and cover the face."""
    blank = np.ones((360, 640, 3), dtype=np.uint8) * 230
    if cap is None or not cap.isOpened():
        return blank

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        return blank

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]
    scale = min(640 / w, 360 / h)
    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    nose_pos = nose_positions.get(frame_idx)
    if nose_pos is None and nose_positions:
        closest = min(nose_positions.keys(), key=lambda k: abs(k - frame_idx))
        nose_pos = nose_positions[closest]
    if nose_pos:
        frame = cover_face(frame, nose_pos[0], nose_pos[1], orig_w, orig_h)
    return frame


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def clear_skeleton():
    """remove the previous frame's skeleton lines from the 3d axis."""
    for artist in state["skeleton_artists"]:
        try:
            artist.remove()
        except Exception:
            pass
    state["skeleton_artists"] = []


def draw_skeleton():
    """draw the 3d skeleton for the current frame."""
    clear_skeleton()
    if not state["frames"]:
        return

    frame_num = state["frames"][state["frame_idx"]]
    frame_df = state["df3d"][state["df3d"]["frame"] == frame_num]
    points = {row["landmark"]: np.array([row["X"], row["Y"], row["Z"]])
              for _, row in frame_df.iterrows()}

    for a, b, color, lw in CONNECTIONS:
        if a in points and b in points:
            line, = ax_3d.plot(
                [points[a][0], points[b][0]],
                [points[a][1], points[b][1]],
                [points[a][2], points[b][2]],
                color=color, lw=lw, alpha=0.92,
            )
            state["skeleton_artists"].append(line)

    if points:
        coords = np.array(list(points.values()))
        scatter = ax_3d.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                                 color="#2C3E50", s=28, zorder=5, depthshade=False)
        state["skeleton_artists"].append(scatter)

    m, r = state["midpoint"], state["range_"]
    ax_3d.set_xlim(m[0] - r, m[0] + r)
    ax_3d.set_ylim(m[1] - r, m[1] + r)
    ax_3d.set_zlim(m[2] - r, m[2] + r)


def draw_cameras():
    """update both camera panels for the current frame."""
    idx = state["frame_idx"]
    frame_a = get_video_frame(state["cap_a"], idx, state["nose_a"],
                               state["orig_w"], state["orig_h"])
    frame_b = get_video_frame(state["cap_b"], idx, state["nose_b"],
                               state["orig_w"], state["orig_h"])
    im_a.set_data(frame_a)
    im_b.set_data(frame_b)
    ax_cam_a.set_xlim(0, frame_a.shape[1])
    ax_cam_a.set_ylim(frame_a.shape[0], 0)
    ax_cam_b.set_xlim(0, frame_b.shape[1])
    ax_cam_b.set_ylim(frame_b.shape[0], 0)


def draw_graphs():
    """move the time cursor on each graph to the current frame and update the readout."""
    if state["times"] is None:
        return

    t = state["times"][min(state["frame_idx"], len(state["times"]) - 1)]
    for i, vline in enumerate(cursor_lines):
        if vline is None:
            continue
        vline.set_xdata([t, t])
        col = MOVEMENTS[state["key"]]["graphs"][i]["col"]
        if col in state["kin_values"]:
            val = state["kin_values"][col][min(state["frame_idx"],
                                                len(state["kin_values"][col]) - 1)]
            if not np.isnan(val):
                cursor_texts[i].set_text(f"{val:.1f}")
                cursor_texts[i].set_x(t)

    if velocity_cursor_line is not None:
        velocity_cursor_line.set_xdata([t, t])


def draw_flags():
    """evaluate and show the repetition-level and live compensation flags for the current frame."""
    movement_cfg = MOVEMENTS[state["key"]]
    df_kin = state["df_kin"]
    if df_kin is None or df_kin.empty:
        return

    row_idx = min(state["frame_idx"], len(df_kin) - 1)
    rep_id = int(df_kin["rep_id"].iloc[row_idx]) if "rep_id" in df_kin.columns else 0
    active_flags = []

    if rep_id > 0 and rep_id in state["rep_peak_values"]:
        for flag in movement_cfg["flags_rep"]:
            col = flag["col"]
            if col not in state["rep_peak_values"][rep_id]:
                continue
            peak_val = state["rep_peak_values"][rep_id][col]
            triggered = ((peak_val < flag["thresh"]) if flag["dir"] == "below"
                         else (peak_val > flag["thresh"]))
            if triggered:
                active_flags.append((f"Rep {rep_id}: {flag['label']} ({peak_val:.1f})",
                                      flag["color"]))

    for flag in movement_cfg["flags_live"]:
        col = flag["col"]
        if col not in df_kin.columns:
            continue
        val = df_kin[col].iloc[row_idx]
        if np.isnan(val):
            continue

        context_col = flag.get("context_col")
        if context_col and context_col in df_kin.columns:
            if df_kin[context_col].iloc[row_idx] < flag.get("context_thresh", 0):
                continue

        triggered = ((val > flag["thresh"]) if flag["dir"] == "above"
                     else (val < flag["thresh"]))
        if triggered:
            active_flags.append((f"LIVE: {flag['label']} ({val:.1f})", flag["color"]))

    rep_badge_text.set_text(f"Rep {rep_id}" if rep_id > 0 else "Transition")
    rep_badge_text.get_bbox_patch().set_facecolor(
        ACCENT[state["key"]] if rep_id > 0 else "#95A5A6"
    )

    for i in range(len(flag_badge_texts)):
        if i < len(active_flags):
            message, color = active_flags[i]
            flag_badge_texts[i].set_text(message)
            flag_badge_texts[i].get_bbox_patch().set_facecolor(color)
            flag_badge_texts[i].get_bbox_patch().set_alpha(0.92)
        else:
            flag_badge_texts[i].set_text("")
            flag_badge_texts[i].get_bbox_patch().set_alpha(0.0)

    if active_flags:
        status_text.set_text("Compensation / flags detected")
        status_text.set_color("#C0392B")
        ax_flags.set_facecolor("#FEF5E7")
    else:
        status_text.set_text("No compensation detected")
        status_text.set_color("#1E8449")
        ax_flags.set_facecolor("#EAFAF1")


def draw_all():
    """redraw everything for the current frame."""
    if not state["frames"]:
        return

    idx, frames = state["frame_idx"], state["frames"]
    t = state["times"][min(idx, len(state["times"]) - 1)] if state["times"] is not None else 0
    movement_cfg = MOVEMENTS[state["key"]]

    accent_bar.set_facecolor(ACCENT[state["key"]])
    header_text.set_text(movement_cfg["title"])
    frame_text.set_text(f"Frame {frames[idx]}   ({idx + 1}/{len(frames)})   t = {t:.2f}s")

    draw_skeleton()
    draw_cameras()
    draw_graphs()
    draw_flags()
    fig.canvas.draw_idle()


def goto_frame(idx):
    """jump to a frame and refresh."""
    if not state["frames"]:
        return
    state["frame_idx"] = max(0, min(idx, len(state["frames"]) - 1))
    slider.eventson = False
    slider.set_val(state["frame_idx"])
    slider.eventson = True
    draw_all()


def play_step():
    """advance one playback step, stop at the end."""
    if not state["playing"]:
        return
    if state["frame_idx"] >= len(state["frames"]) - 1:
        toggle_play()
        return
    state["frame_idx"] = min(state["frame_idx"] + PLAYBACK_FRAME_STEP,
                              len(state["frames"]) - 1)
    slider.eventson = False
    slider.set_val(state["frame_idx"])
    slider.eventson = True
    draw_all()


def toggle_play(event=None):
    """start or stop playback."""
    state["playing"] = not state["playing"]
    if state["playing"]:
        play_button.label.set_text("Pause")
        timer.start()
    else:
        play_button.label.set_text("Play")
        timer.stop()
    fig.canvas.draw_idle()


def switch_movement(key):
    """load a different movement and reset the viewer to its first frame."""
    if state["playing"]:
        toggle_play()
    load_movement(key)
    rebuild_graphs()
    slider.eventson = False
    slider.valmax = max(len(state["frames"]) - 1, 1)
    slider.ax.set_xlim(0, slider.valmax)
    slider.set_val(0)
    slider.eventson = True
    draw_all()


def rebuild_graphs():
    """redraw the graph panels for the current movement, since each movement plots different signals."""
    global velocity_cursor_line
    movement_cfg = MOVEMENTS[state["key"]]

    for i, (ax, graph_cfg) in enumerate(zip(graph_axes, movement_cfg["graphs"])):
        ax.cla()
        for spine in ax.spines.values():
            spine.set_edgecolor("#D5D8DC")
        ax.set_facecolor("#FDFEFE")
        ax.grid(True, alpha=0.3, color="#D5D8DC", lw=0.7)
        ax.set_xlabel("Time (s)", fontsize=7)
        ax.set_ylabel(graph_cfg["label"], fontsize=7)
        ax.set_title(graph_cfg["label"], fontsize=8, fontweight="bold")

        col = graph_cfg["col"]
        if col in state["kin_values"] and state["times"] is not None:
            values, times = state["kin_values"][col], state["times"]
            ax.plot(times, values, color=graph_cfg["color"], lw=1.6, alpha=0.9)
            ax.fill_between(times, np.nanmin(values), values, alpha=0.09, color=graph_cfg["color"])

            if graph_cfg.get("thresh") is not None:
                thresh = graph_cfg["thresh"]
                ax.axhline(thresh, color="#E74C3C", lw=1.2, ls="--", alpha=0.85,
                           label=graph_cfg.get("thresh_label", ""))
                if graph_cfg["thresh_dir"] == "above":
                    ax.fill_between(times, thresh, values, where=np.array(values) > thresh,
                                     alpha=0.14, color="#FDECEA")
                else:
                    ax.fill_between(times, values, thresh, where=np.array(values) < thresh,
                                     alpha=0.14, color="#FEF9E7")
                ax.legend(fontsize=6, loc="upper right", framealpha=0.8)

            cursor_lines[i] = ax.axvline(times[0], color="#2C3E50", lw=1.8, alpha=0.85, zorder=10)
            ylim = ax.get_ylim()
            cursor_texts[i] = ax.text(
                times[0], ylim[0] + (ylim[1] - ylim[0]) * 0.88, "",
                fontsize=8, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="#AEB6BF", alpha=0.9),
            )
        else:
            cursor_lines[i] = ax.axvline(0, color="#2C3E50", lw=1.8)
            cursor_texts[i] = ax.text(0, 0, "", fontsize=8)

    ax_velocity.cla()
    for spine in ax_velocity.spines.values():
        spine.set_edgecolor("#D5D8DC")
    ax_velocity.set_facecolor("#FDFEFE")
    ax_velocity.grid(True, alpha=0.3, color="#D5D8DC", lw=0.7)
    ax_velocity.set_xlabel("Time (s)", fontsize=7)
    ax_velocity.set_ylabel(movement_cfg["velocity_label"], fontsize=7)
    ax_velocity.set_title("Movement Velocity", fontsize=8, fontweight="bold")

    if "velocity_raw" in state["kin_values"] and "velocity" in state["kin_values"]:
        v_raw = state["kin_values"]["velocity_raw"]
        v_smooth = state["kin_values"]["velocity"]
        times = state["times"]
        ax_velocity.plot(times, v_raw, color="#D5D8DC", lw=0.9, alpha=0.85, label="Raw")
        ax_velocity.plot(times, v_smooth, color="#8E44AD", lw=2.0, label="Smoothed")
        ax_velocity.fill_between(times, 0, v_smooth, alpha=0.10, color="#8E44AD")

        cv = np.nanstd(v_raw) / (np.nanmean(v_raw) + 1e-8) * 100
        label = "Smooth" if cv < 80 else "Irregular"
        color = "#1E8449" if cv < 80 else "#C0392B"
        ax_velocity.text(0.97, 0.93, f"{label}\nCV = {cv:.0f}%", transform=ax_velocity.transAxes,
                          fontsize=7, color=color, ha="right", va="top", fontweight="bold",
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                    edgecolor="#D5D8DC", alpha=0.9))
        ax_velocity.legend(fontsize=6, loc="upper left", framealpha=0.8)
        velocity_cursor_line = ax_velocity.axvline(times[0], color="#2C3E50", lw=1.8, alpha=0.85)
    else:
        velocity_cursor_line = ax_velocity.axvline(0, color="#2C3E50", lw=1.8)


def on_key_press(event):
    """keyboard shortcuts: arrows step frames, space plays, number keys switch movement."""
    key_map = {key[1:]: key for key in MOVEMENTS}  # "1" -> "M1", "2" -> "M2", ...
    if event.key == "right":
        goto_frame(state["frame_idx"] + 1)
    elif event.key == "left":
        goto_frame(state["frame_idx"] - 1)
    elif event.key == " ":
        toggle_play()
    elif event.key in key_map:
        switch_movement(key_map[event.key])


# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------

load_movement(state["key"])

fig = plt.figure(figsize=(19.2, 10.8), facecolor="white", dpi=100)
fig.canvas.manager.set_window_title("FMA Kinematic Viewer")

first_key = state["key"]

accent_bar = fig.add_axes([0, 0.962, 1, 0.038])
accent_bar.set_facecolor(ACCENT[first_key])
accent_bar.axis("off")
header_text = accent_bar.text(0.015, 0.5, MOVEMENTS[first_key]["title"],
                               transform=accent_bar.transAxes, fontsize=12,
                               fontweight="bold", color="white", va="center")
frame_text = accent_bar.text(0.72, 0.5, "", transform=accent_bar.transAxes,
                              fontsize=9, color="white", va="center", alpha=0.9)

ax_cam_a = fig.add_axes([0.010, 0.545, 0.265, 0.405])
ax_cam_a.set_facecolor("#1A1A1A")
ax_cam_a.axis("off")
im_a = ax_cam_a.imshow(np.zeros((360, 640, 3), dtype=np.uint8), aspect="auto")
ax_cam_a.text(0.02, 0.96, MOVEMENTS[first_key]["cam_a_label"], transform=ax_cam_a.transAxes,
              fontsize=9, fontweight="bold", color="white", va="top",
              bbox=dict(boxstyle="round,pad=0.35", facecolor="#2C3E50", edgecolor="none", alpha=0.85))

ax_3d = fig.add_axes([0.285, 0.530, 0.420, 0.425], projection="3d")
ax_3d.set_facecolor("white")
ax_3d.xaxis.pane.fill = ax_3d.yaxis.pane.fill = ax_3d.zaxis.pane.fill = False
ax_3d.xaxis.pane.set_edgecolor("none")
ax_3d.yaxis.pane.set_edgecolor("none")
ax_3d.zaxis.pane.set_edgecolor("none")
ax_3d.set_xticks([])
ax_3d.set_yticks([])
ax_3d.set_zticks([])
ax_3d.grid(False)
fig.text(0.493, 0.536, "3D Skeleton", fontsize=7.5, color="#95A5A6", ha="center", style="italic")

ax_cam_b = fig.add_axes([0.718, 0.545, 0.270, 0.405])
ax_cam_b.set_facecolor("#1A1A1A")
ax_cam_b.axis("off")
im_b = ax_cam_b.imshow(np.zeros((360, 640, 3), dtype=np.uint8), aspect="auto")
ax_cam_b.text(0.02, 0.96, MOVEMENTS[first_key]["cam_b_label"], transform=ax_cam_b.transAxes,
              fontsize=9, fontweight="bold", color="white", va="top",
              bbox=dict(boxstyle="round,pad=0.35", facecolor="#2C3E50", edgecolor="none", alpha=0.85))

graph_w, graph_h, graph_y = 0.228, 0.285, 0.225
graph_axes = [fig.add_axes([0.010 + i * (graph_w + 0.012), graph_y, graph_w, graph_h])
              for i in range(3)]
ax_velocity = fig.add_axes([0.010 + 3 * (graph_w + 0.012), graph_y, graph_w, graph_h])
cursor_lines = [None, None, None]
cursor_texts = [None, None, None]
velocity_cursor_line = None
rebuild_graphs()

ax_flags = fig.add_axes([0.010, 0.105, 0.978, 0.105])
ax_flags.set_facecolor("#EAFAF1")
ax_flags.set_xticks([])
ax_flags.set_yticks([])
for spine in ax_flags.spines.values():
    spine.set_edgecolor("#D5D8DC")

status_text = ax_flags.text(0.008, 0.72, "No compensation detected",
                             transform=ax_flags.transAxes, fontsize=10,
                             fontweight="bold", color="#1E8449")
rep_badge_text = ax_flags.text(0.83, 0.72, "Rep 1", transform=ax_flags.transAxes,
                                fontsize=9, fontweight="bold", color="white", ha="center",
                                bbox=dict(boxstyle="round,pad=0.5",
                                          facecolor=ACCENT[first_key], edgecolor="none"))
flag_badge_texts = [
    ax_flags.text(x, 0.20, "", transform=ax_flags.transAxes, fontsize=8.5,
                  fontweight="bold", color="white",
                  bbox=dict(boxstyle="round,pad=0.45", facecolor="#E74C3C",
                            edgecolor="none", alpha=0.0))
    for x in [0.008, 0.26, 0.51, 0.75]
]

ax_slider = fig.add_axes([0.010, 0.048, 0.565, 0.022], facecolor="#EBF5FB")
slider = Slider(ax_slider, "", 0, max(len(state["frames"]) - 1, 1), valinit=0, valstep=1,
                color="#2980B9")
slider.label.set_visible(False)
slider.on_changed(lambda v: (state.update({"frame_idx": int(v)}), draw_all()))

ax_prev = fig.add_axes([0.590, 0.032, 0.065, 0.038], facecolor="#F2F3F4")
button_prev = Button(ax_prev, "Prev", color="#F2F3F4", hovercolor="#D6EAF8")
button_prev.on_clicked(lambda e: goto_frame(state["frame_idx"] - 1))

ax_play = fig.add_axes([0.530, 0.032, 0.056, 0.038], facecolor="#2C3E50")
play_button = Button(ax_play, "Play", color="#2C3E50", hovercolor="#1A252F")
play_button.label.set_color("white")
play_button.on_clicked(toggle_play)

ax_next = fig.add_axes([0.660, 0.032, 0.065, 0.038], facecolor="#F2F3F4")
button_next = Button(ax_next, "Next", color="#F2F3F4", hovercolor="#D6EAF8")
button_next.on_clicked(lambda e: goto_frame(state["frame_idx"] + 1))

movement_buttons = []
for i, key in enumerate(MOVEMENTS):
    ax = fig.add_axes([0.738 + i * 0.086, 0.032, 0.082, 0.038], facecolor=ACCENT[key])
    button = Button(ax, key, color=ACCENT[key], hovercolor="#AED6F1")
    button.label.set_color("white")
    button.label.set_fontweight("bold")
    button.on_clicked(lambda e, k=key: switch_movement(k))
    movement_buttons.append(button)

movement_keys_hint = ", ".join(f"{k[1:]}={k}" for k in MOVEMENTS)
fig.text(0.010, 0.014, f"Keyboard: Left/Right = frame, Space = play/pause, {movement_keys_hint}",
         fontsize=7, color="#AAB7B8")

timer = fig.canvas.new_timer(interval=PLAYBACK_INTERVAL_MS)
timer.add_callback(play_step)

fig.canvas.mpl_connect("key_press_event", on_key_press)

draw_all()
plt.show()
