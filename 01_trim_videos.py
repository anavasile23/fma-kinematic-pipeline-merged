"""
Trim raw multi-camera recordings into per-movement clips.

Each camera records a single continuous video covering all movements.
This script cuts out the relevant time segments (calibration + each
movement) and removes audio, producing one clip per camera per segment.

Edit SEGMENTS below with the start/end timestamps (in seconds) for each
camera, then run:

    python 01_trim_videos.py
"""

import os
import subprocess

import config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IN_DIR = os.path.join(BASE_DIR, config.DIR_VIDEOS_RAW)
OUT_DIR = os.path.join(BASE_DIR, config.DIR_VIDEOS_TRIMMED)
os.makedirs(OUT_DIR, exist_ok=True)

# (start_seconds, end_seconds, segment_label) per camera.
# "calib" should cover the period where the checkerboard is visible.
SEGMENTS = {
    "cam1": [
        (0, 0, "calib"),
        (0, 0, "m1_volitional_synergy_UE"),
        (0, 0, "m2_hand_lumbar"),
        (0, 0, "m4_volitional_no_synergy_standing"),
    ],
    "cam2": [
        (0, 0, "calib"),
        (0, 0, "m1_volitional_synergy_UE"),
        (0, 0, "m2_hand_lumbar"),
        (0, 0, "m4_volitional_no_synergy_standing"),
    ],
    "cam3": [
        (0, 0, "calib"),
        (0, 0, "m1_volitional_synergy_UE"),
        (0, 0, "m2_hand_lumbar"),
        (0, 0, "m4_volitional_no_synergy_standing"),
    ],
}


def cut_mute(in_path, out_path, start, end):
    """cut one segment out of a video and drop the audio."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", in_path,
        "-t", str(end - start),
        "-an",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        out_path,
    ]
    print(f"  {os.path.basename(in_path)} [{start}s-{end}s] -> {os.path.basename(out_path)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-300:]}")


def main():
    """trim every camera and segment listed in SEGMENTS."""
    for cam, segments in SEGMENTS.items():
        in_path = os.path.join(IN_DIR, f"{cam}.mp4")
        if not os.path.exists(in_path):
            print(f"[{cam}] missing source file: {in_path}")
            continue
        print(f"-- {cam} --")
        for start, end, label in segments:
            out_path = os.path.join(OUT_DIR, f"{cam}_{label}.mp4")
            cut_mute(in_path, out_path, start, end)

    print(f"\nDone. Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
