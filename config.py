"""
Configuration for the FMA kinematic analysis pipeline.

Every pipeline script imports its settings from here, so a single edit
propagates through calibration, triangulation, normalisation, kinematics
and the viewer.

To process a subject, set SUBJECT below to the right key and run the
pipeline. Only the three things that actually differ between subjects
live in SUBJECTS -- affected side, shoulder width, and the camera pair
used per movement. Everything else is shared.
"""

# ---------------------------------------------------------------------------
# Subject selection  --  change this one line to switch subject
# ---------------------------------------------------------------------------

SUBJECT = "p1"   # "p1" or "p2"

# Per-subject settings.
#
# shoulder_width_m: neither subject had a direct biacromial measurement;
#   both widths were estimated from patient height. Absolute distance-based
#   metrics (elevation, wrist-to-lumbar, pelvic tilt, in cm) scale with this
#   estimate; angular parameters are scale-invariant and unaffected.
#
# camera_pairs: cameras were fixed and the patient rotated between movements,
#   so the pair triangulated per movement was chosen as the one with the best
#   visibility of the affected limb for that movement. A single well-chosen
#   pair per movement is used (not a multi-pair merge), because merging pairs
#   with different relative orientations corrupts the reconstructed coordinates.

SUBJECTS = {
    "p1": {
        "affected_side": "RIGHT",
        "shoulder_width_m": 0.492,   # estimated from patient height (~1.85 m)
        "camera_pairs": {
            "M1": ("cam1", "cam2"),
            "M2": ("cam1", "cam2"),
            "M4": ("cam1", "cam3"),
        },
    },
    "p2": {
        "affected_side": "LEFT",
        "shoulder_width_m": 0.44,    # estimated from patient height (~1.60 m)
        "camera_pairs": {
            "M1": ("cam1", "cam3"),
            "M2": ("cam1", "cam3"),
            "M4": ("cam1", "cam2"),
        },
    },
}

_subject = SUBJECTS[SUBJECT]

# Side of the body with the motor deficit. Determines which landmarks
# (LEFT_* or RIGHT_*) are used for the affected-limb kinematics.
AFFECTED_SIDE = _subject["affected_side"]

# Shoulder width used to rescale triangulated coordinates from arbitrary
# units to metres (see note above on estimation).
SHOULDER_WIDTH_M = _subject["shoulder_width_m"]

# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------

CAMERAS = ["cam1", "cam2", "cam3"]

# Checkerboard used for intrinsic calibration: (columns, rows) of inner
# corners, and the physical size of one square in metres.
CHECKERBOARD_SIZE = (9, 5)
CHECKERBOARD_SQUARE_M = 0.026

# ---------------------------------------------------------------------------
# Movements
# ---------------------------------------------------------------------------
# `camera_pair` is filled in per subject from SUBJECTS above.

MOVEMENTS = {
    "M1": {
        "name": "m1_volitional_synergy_UE",
        "label": "Volitional Movement Within Synergies (Upper Extremity)",
        "fma_section": "FMA-UE Section II",
        "camera_pair": _subject["camera_pairs"]["M1"],
        "n_repetitions": 3,
    },
    "M2": {
        "name": "m2_hand_lumbar",
        "label": "Hand to Lumbar Spine",
        "fma_section": "FMA-UE Section III",
        "camera_pair": _subject["camera_pairs"]["M2"],
        "n_repetitions": 3,
    },
    "M4": {
        "name": "m4_volitional_no_synergy_standing",
        "label": "Volitional Movement with Little/No Synergy (Standing)",
        "fma_section": "FMA-LE Section IV",
        "camera_pair": _subject["camera_pairs"]["M4"],
        "n_repetitions": 3,
    },
}

# ---------------------------------------------------------------------------
# Directory layout (all relative to the project root)
# ---------------------------------------------------------------------------

DIR_VIDEOS_RAW    = "videos_raw"
DIR_VIDEOS_TRIMMED = "videos_trimmed"
DIR_CALIB         = "calib"
DIR_POSE2D        = "pose2d"
DIR_JOINTS3D_RAW  = "joints3d_raw"
DIR_JOINTS3D      = "joints3d_ready"
DIR_KINEMATICS    = "kinematics"

# ---------------------------------------------------------------------------
# Pose estimation
# ---------------------------------------------------------------------------

POSE_MODEL_PATH = "pose_landmarker_full.task"
POSE_MIN_VISIBILITY = 0.4

# ---------------------------------------------------------------------------
# Viewer display settings
# ---------------------------------------------------------------------------
# Defines which kinematic columns are plotted for each movement, their
# thresholds, and which compensation flags to evaluate. Column names must
# match the output of 06_compute_kinematics.py.
#
# Flag types:
#   "rep"  - evaluated once per repetition at its peak value (ROM criteria).
#   "live" - evaluated every frame (instantaneous compensation markers).
#            An optional "context_col"/"context_thresh" restricts the flag
#            to frames where another signal is above a threshold, so a
#            compensation marker only fires during active movement.

VIEWER_GRAPHS = {
    "M1": [
        {"col": "shoulder_abduction", "label": "Shoulder Abduction (deg)",
         "color": "#2980B9", "thresh": 90, "thresh_dir": "below", "thresh_label": "90 deg target"},
        {"col": "elbow_flexion", "label": "Elbow Flexion (deg)",
         "color": "#27AE60", "thresh": None},
        {"col": "shoulder_elevation", "label": "Shoulder Elevation (cm)",
         "color": "#E74C3C", "thresh": 3, "thresh_dir": "above", "thresh_label": "3 cm"},
    ],
    "M2": [
        {"col": "wrist_lumbar_dist", "label": "Wrist to Lumbar (cm)",
         "color": "#2980B9", "thresh": 30, "thresh_dir": "above", "thresh_label": "30 cm"},
        {"col": "shoulder_extension", "label": "Shoulder Extension (deg)",
         "color": "#27AE60", "thresh": None},
        {"col": "trunk_flexion", "label": "Trunk Flexion (deg)",
         "color": "#E74C3C", "thresh": 10, "thresh_dir": "above", "thresh_label": "10 deg"},
    ],
    "M4": [
        {"col": "knee_flexion", "label": "Knee Flexion (deg)",
         "color": "#2980B9", "thresh": 90, "thresh_dir": "below", "thresh_label": "90 deg target"},
        {"col": "hip_flexion", "label": "Hip Flexion / Comp. (deg)",
         "color": "#E74C3C", "thresh": 15, "thresh_dir": "above", "thresh_label": "15 deg FMA"},
        {"col": "pelvis_tilt", "label": "Pelvic Tilt (cm)",
         "color": "#E67E22", "thresh": 2, "thresh_dir": "above", "thresh_label": "2 cm"},
    ],
}

VIEWER_VELOCITY_LABEL = {
    "M1": "Shoulder Velocity (deg/s)",
    "M2": "Hand Velocity (cm/s)",
    "M4": "Knee Velocity (deg/s)",
}

VIEWER_FLAGS_REP = {
    "M1": [{"col": "shoulder_abduction", "thresh": 90, "dir": "below",
            "label": "Peak abduction < 90 deg", "color": "#E67E22"}],
    "M2": [{"col": "wrist_lumbar_dist", "thresh": 30, "dir": "above",
            "label": "Hand did not reach lumbar", "color": "#E67E22", "use_min": True}],
    "M4": [{"col": "knee_flexion", "thresh": 90, "dir": "below",
            "label": "Peak knee flexion < 90 deg", "color": "#E67E22"}],
}

VIEWER_FLAGS_LIVE = {
    "M1": [{"col": "shoulder_elevation", "thresh": 3, "dir": "above",
            "label": "Shoulder compensation", "color": "#C0392B",
            "context_col": "shoulder_abduction", "context_thresh": 30}],
    "M2": [{"col": "trunk_flexion", "thresh": 10, "dir": "above",
            "label": "Trunk compensation", "color": "#C0392B"}],
    "M4": [
        {"col": "hip_flexion", "thresh": 15, "dir": "above",
         "label": "Hip compensation (FMA)", "color": "#C0392B",
         "context_col": "knee_flexion", "context_thresh": 20},
        {"col": "pelvis_tilt", "thresh": 2, "dir": "above",
         "label": "Pelvic instability", "color": "#C0392B"},
    ],
}

MOVEMENT_ACCENT_COLOR = {"M1": "#2980B9", "M2": "#27AE60", "M4": "#C0392B"}

# Connections drawn for the 3D stickman: (point_a, point_b, color, linewidth)
SKELETON_CONNECTIONS = [
    ("LEFT_SHOULDER", "RIGHT_SHOULDER", "#34495E", 3.0),
    ("LEFT_SHOULDER", "LEFT_HIP", "#34495E", 3.0),
    ("RIGHT_SHOULDER", "RIGHT_HIP", "#34495E", 3.0),
    ("LEFT_HIP", "RIGHT_HIP", "#34495E", 3.0),
    ("NOSE", "LEFT_SHOULDER", "#E67E22", 2.0),
    ("NOSE", "RIGHT_SHOULDER", "#E67E22", 2.0),
    ("LEFT_SHOULDER", "LEFT_ELBOW", "#27AE60", 2.5),
    ("LEFT_ELBOW", "LEFT_WRIST", "#27AE60", 2.5),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW", "#2980B9", 2.5),
    ("RIGHT_ELBOW", "RIGHT_WRIST", "#2980B9", 2.5),
    ("LEFT_HIP", "LEFT_KNEE", "#27AE60", 2.5),
    ("LEFT_KNEE", "LEFT_ANKLE", "#27AE60", 2.5),
    ("RIGHT_HIP", "RIGHT_KNEE", "#2980B9", 2.5),
    ("RIGHT_KNEE", "RIGHT_ANKLE", "#2980B9", 2.5),
]

# Face anonymisation: size of the black rectangle drawn over the NOSE
# landmark in each camera feed, in pixels at the original video resolution.
FACE_COVER_HALF_WIDTH_PX = 85
FACE_COVER_HALF_HEIGHT_PX = 100
FACE_COVER_Y_SHIFT = 0.20  # fraction of half-height, shifts box upward
