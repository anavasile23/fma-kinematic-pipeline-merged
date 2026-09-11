# FMA Kinematic Analysis Pipeline

Markerless 3D motion capture pipeline for quantitative kinematic analysis
of Fugl-Meyer Assessment (FMA) movements in post-stroke patients.

The Fugl-Meyer Assessment is a validated, widely used clinical scale for
post-stroke motor function, scored observationally on an ordinal scale.
This pipeline adds a quantitative kinematic layer on top of the same
assessment movements — joint range of motion, compensatory strategies,
movement velocity and smoothness, and repetition-by-repetition detail —
using only consumer action cameras and open-source pose estimation.

## What it does

1. Trims raw multi-camera footage into per-movement clips.
2. Calibrates camera intrinsics from a checkerboard pattern.
3. Extracts 2D body landmarks per camera with MediaPipe Pose.
4. Triangulates matched landmarks across a camera pair into 3D.
5. Rescales coordinates to metric units and smooths trajectories.
6. Computes joint angles, repetition boundaries, compensation flags
   and movement velocity for each FMA movement.
7. Provides an interactive viewer synchronising camera footage, the 3D
   skeleton reconstruction, and the kinematic graphs frame-by-frame.

Patient faces are automatically anonymised in the viewer by drawing an
opaque rectangle over the detected nose position in each camera feed.

## Pipeline

| Script | Purpose | Output |
|---|---|---|
| `01_trim_videos.py` | Cut raw recordings into movement clips | `videos_trimmed/` |
| `02_calibrate_cameras.py` | Intrinsic camera calibration | `calib/*.npz` |
| `03_extract_pose2d.py` | 2D landmark extraction (MediaPipe) | `pose2d/*.csv` |
| `04_triangulate.py` | Stereo triangulation to 3D | `joints3d_raw/*.csv` |
| `05_normalize_smooth.py` | Metric rescaling + smoothing | `joints3d_ready/*.csv` |
| `06_compute_kinematics.py` | Joint angles, flags, velocity | `kinematics/*.csv`, `*.png` |
| `07_viewer.py` | Interactive synchronised viewer | — |

Run them in order from the project root:

```bash
python 01_trim_videos.py
python 02_calibrate_cameras.py
python 03_extract_pose2d.py
python 04_triangulate.py
python 05_normalize_smooth.py
python 06_compute_kinematics.py
python 07_viewer.py
```

## Configuration

All settings live in `config.py`, and every script reads from it, so a
single edit propagates through the whole pipeline.

### Selecting a subject

The two subjects analysed in this study differ in only three respects:
the affected side, the estimated shoulder width, and which camera pair is
triangulated per movement. These are collected in the `SUBJECTS`
dictionary, and the active subject is chosen with a single line:

```python
SUBJECT = "p1"   # "p1" or "p2"
```

Switching `SUBJECT` sets `AFFECTED_SIDE`, `SHOULDER_WIDTH_M` and the
per-movement `camera_pair` for that subject. Nothing subject-specific is
hard-coded in the scripts.

- `AFFECTED_SIDE` — `"LEFT"` or `"RIGHT"`, determines which limb's
  landmarks are used for the affected-side kinematics.
- `SHOULDER_WIDTH_M` — shoulder width used to rescale triangulated
  coordinates to metres. For both subjects this was estimated from
  patient height, not measured directly (see Limitations).
- `MOVEMENTS` — per-movement camera pair, FMA section reference, and
  number of repetitions. The camera pair was chosen as the one with the
  best visibility of the affected limb for that movement, and differs
  between movements and between subjects.
- `VIEWER_GRAPHS`, `VIEWER_FLAGS_REP`, `VIEWER_FLAGS_LIVE` — which
  kinematic signals are plotted and which compensation rules are
  evaluated in the viewer.

To add a further subject, add an entry to `SUBJECTS` and point `SUBJECT`
at it.

### Camera setup

The pipeline expects three fixed cameras (`cam1`, `cam2`, `cam3`) and a
patient who may face a different camera depending on the movement being
performed. Two cameras are triangulated per movement; which pair is most
appropriate depends on which cameras have an unobstructed, sufficiently
oblique view of the limb being assessed for that specific movement.

## Movements currently supported

| Key | Movement | FMA Section |
|---|---|---|
| M1 | Volitional Movement Within Synergies (Upper Extremity) | FMA-UE Section II |
| M2 | Hand to Lumbar Spine | FMA-UE Section III |
| M4 | Volitional Movement with Little/No Synergy, Standing | FMA-LE Section IV |

Movement M3 (Lower Extremity, supine) is not supported: MediaPipe Pose
performs poorly on supine subjects due to occlusion and unusual limb
orientation relative to training data.

Additional movements can be added by extending `config.MOVEMENTS` and
implementing a corresponding `compute_*` function in
`06_compute_kinematics.py`.

## Compensation flag logic

Two flag types are used:

- **Repetition-level (peak-based)** — evaluated once per repetition at
  its peak value. Used for ROM criteria, e.g. "did this repetition reach
  90 degrees of abduction at its peak?" A single failed repetition is
  not diluted by averaging across the whole recording.
- **Live (instantaneous)** — evaluated every frame, optionally only
  while another signal is above a threshold (e.g. shoulder elevation is
  only flagged as compensation while the arm is actively abducting, not
  during rest between repetitions).

## Requirements

- Python 3.10+
- ffmpeg (for video trimming)
- A 9x5 (or similar) checkerboard pattern for camera calibration

Install Python dependencies:

```bash
pip install -r requirements.txt
```

On some systems, `pip install mediapipe` may require
`pip install mediapipe --break-system-packages` or a virtual environment.

## Directory layout

```
project/
├── config.py
├── videos_raw/            cam1.mp4, cam2.mp4, cam3.mp4
├── videos_trimmed/        generated by 01_trim_videos.py
├── calib/                 generated by 02_calibrate_cameras.py
├── pose2d/                generated by 03_extract_pose2d.py
├── joints3d_raw/          generated by 04_triangulate.py
├── joints3d_ready/        generated by 05_normalize_smooth.py
└── kinematics/            generated by 06_compute_kinematics.py
```

## Limitations

- Single-pair stereo triangulation: combining triangulations from
  multiple camera pairs into one merged skeleton was attempted and
  abandoned — without full multi-camera stereo calibration, the relative
  scale between pairs is not reliably consistent, which corrupts the
  merged 3D positions. A single, well-chosen camera pair per movement
  produces clean and reliable kinematic signals, at the cost of some
  landmarks (typically on the side facing away from both cameras) not
  being reconstructed. This does not affect the affected-side kinematics
  computed in `06_compute_kinematics.py`, only the completeness of the
  3D skeleton visualisation.
- Metric scale rests on an estimated shoulder width. For both subjects the
  biacromial width was estimated from patient height rather than measured
  directly, so absolute distance-based metrics (shoulder elevation, wrist-to-
  lumbar distance, pelvic tilt, and their centimetre thresholds) carry this
  uncertainty. Angular parameters are scale-invariant and unaffected.
- MediaPipe Pose is unreliable for supine positions; movements requiring
  a supine posture are out of scope for this pipeline.
- This is a research pipeline, not a validated clinical instrument. It
  is intended to complement, not replace, the standard FMA protocol.

## License

Non-commercial use with attribution. Commercial use requires prior written
permission from the author. See `LICENSE` for full terms.
