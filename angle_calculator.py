"""
angle_calculator.py
-------------------
Coordinate-geometry functions for calculating joint angles from
MediaPipe pose key-points.

Two core formulas are used:

1. three_point_angle(A, B, C)
   Returns the angle at joint B formed by vectors B→A and B→C.
   Used for:  Knee  (hip  → knee  → ankle)
              Hip   (shoulder → hip → knee)
              Elbow (shoulder → elbow → wrist)

2. vertical_angle(A, B)
   Returns the angle between the line A→B and the vertical axis.
   Used for:  Spine inclination (mid-shoulder → mid-hip vs. vertical)

Maths recap
-----------
    cos(θ) = (u · v) / (|u| × |v|)

where u = A - B  and  v = C - B  (vectors from joint B).
arccos gives the angle in radians; np.degrees converts to °.
"""

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Core geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def three_point_angle(point_a: dict, point_b: dict, point_c: dict) -> float:
    """
    Calculate the interior angle at point_b (in degrees).

    Each point must be a dict with 'x' and 'y' keys (pixel coordinates).

    Example
    -------
    knee_angle = three_point_angle(hip, knee, ankle)
    # Returns ~90° when knee is at parallel squat depth.
    """
    a = np.array([point_a["x"], point_a["y"]], dtype=float)
    b = np.array([point_b["x"], point_b["y"]], dtype=float)
    c = np.array([point_c["x"], point_c["y"]], dtype=float)

    # Vectors from B to A and B to C
    ba = a - b
    bc = c - b

    # Cosine of the angle using dot product formula
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    # Guard against zero-length vectors (joint collapsed on another)
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0

    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)

    # Clamp to [-1, 1] to avoid arccos domain errors from floating point
    cosine = float(np.clip(cosine, -1.0, 1.0))
    angle  = np.degrees(np.arccos(cosine))

    return round(float(angle), 1)


def vertical_angle(point_top: dict, point_bottom: dict) -> float:
    """
    Calculate the angle (degrees) between the line top→bottom
    and the upward vertical axis [0, -1] (negative y = up in image coords).

    Used to measure spine/torso forward lean:
        0°  = perfectly upright
        90° = horizontal

    Example
    -------
    spine_lean = vertical_angle(mid_shoulder, mid_hip)
    """
    top    = np.array([point_top["x"],    point_top["y"]],    dtype=float)
    bottom = np.array([point_bottom["x"], point_bottom["y"]], dtype=float)

    # Vector from bottom to top (pointing upward along spine)
    vec = top - bottom
    norm = np.linalg.norm(vec)

    if norm < 1e-6:
        return 0.0

    # Reference: upward direction in image coordinates = (0, -1)
    vertical_up = np.array([0.0, -1.0])

    cosine = np.dot(vec / norm, vertical_up)
    cosine = float(np.clip(cosine, -1.0, 1.0))
    angle  = np.degrees(np.arccos(cosine))

    return round(float(angle), 1)


# ──────────────────────────────────────────────────────────────────────────────
# Mid-point helper
# ──────────────────────────────────────────────────────────────────────────────

def midpoint(p1: dict, p2: dict) -> dict:
    """
    Return the pixel midpoint between two landmark dicts.
    Result is a minimal dict with 'x', 'y', 'x_norm', 'y_norm'.
    """
    return {
        "x":      (p1["x"] + p2["x"]) / 2.0,
        "y":      (p1["y"] + p2["y"]) / 2.0,
        "x_norm": (p1.get("x_norm", 0) + p2.get("x_norm", 0)) / 2.0,
        "y_norm": (p1.get("y_norm", 0) + p2.get("y_norm", 0)) / 2.0,
    }


# ──────────────────────────────────────────────────────────────────────────────
# High-level angle computation
# ──────────────────────────────────────────────────────────────────────────────

def calculate_all_angles(key_points: dict) -> dict:
    """
    Compute all biomechanical angles from a set of named key-points.

    Returns a dict with these keys (when landmarks are available):
        left_knee, right_knee, knee       (averaged)
        left_hip,  right_hip,  hip        (averaged)
        left_elbow                        (if wrist visible)
        spine                             (torso inclination from vertical)
        mid_shoulder, mid_hip             (midpoint dicts for annotation)

    All angle values are floats in degrees.
    """
    angles = {}

    try:
        # ── KNEE ANGLES  (hip → knee → ankle) ─────────────────────────────
        if all(k in key_points for k in ["left_hip", "left_knee", "left_ankle"]):
            angles["left_knee"] = three_point_angle(
                key_points["left_hip"],
                key_points["left_knee"],
                key_points["left_ankle"],
            )

        if all(k in key_points for k in ["right_hip", "right_knee", "right_ankle"]):
            angles["right_knee"] = three_point_angle(
                key_points["right_hip"],
                key_points["right_knee"],
                key_points["right_ankle"],
            )

        # Average both sides for a single "knee" reading
        if "left_knee" in angles and "right_knee" in angles:
            angles["knee"] = round((angles["left_knee"] + angles["right_knee"]) / 2.0, 1)
        elif "left_knee" in angles:
            angles["knee"] = angles["left_knee"]
        elif "right_knee" in angles:
            angles["knee"] = angles["right_knee"]

        # ── HIP ANGLES  (shoulder → hip → knee) ───────────────────────────
        if all(k in key_points for k in ["left_shoulder", "left_hip", "left_knee"]):
            angles["left_hip"] = three_point_angle(
                key_points["left_shoulder"],
                key_points["left_hip"],
                key_points["left_knee"],
            )

        if all(k in key_points for k in ["right_shoulder", "right_hip", "right_knee"]):
            angles["right_hip"] = three_point_angle(
                key_points["right_shoulder"],
                key_points["right_hip"],
                key_points["right_knee"],
            )

        if "left_hip" in angles and "right_hip" in angles:
            angles["hip"] = round((angles["left_hip"] + angles["right_hip"]) / 2.0, 1)
        elif "left_hip" in angles:
            angles["hip"] = angles["left_hip"]
        elif "right_hip" in angles:
            angles["hip"] = angles["right_hip"]

        # ── SPINE / TORSO INCLINATION ─────────────────────────────────────
        # Uses the midpoint of both shoulders and both hips for stability
        if all(k in key_points for k in [
            "left_shoulder", "right_shoulder", "left_hip", "right_hip"
        ]):
            mid_sh  = midpoint(key_points["left_shoulder"], key_points["right_shoulder"])
            mid_hip = midpoint(key_points["left_hip"],      key_points["right_hip"])

            angles["spine"]       = vertical_angle(mid_sh, mid_hip)
            # Store midpoints for use in annotation (app.py)
            angles["mid_shoulder"] = mid_sh
            angles["mid_hip"]      = mid_hip

        # ── ELBOW ANGLES  (shoulder → elbow → wrist) ─────────────────────
        if all(k in key_points for k in ["left_shoulder", "left_elbow", "left_wrist"]):
            angles["left_elbow"] = three_point_angle(
                key_points["left_shoulder"],
                key_points["left_elbow"],
                key_points["left_wrist"],
            )

        if all(k in key_points for k in ["right_shoulder", "right_elbow", "right_wrist"]):
            angles["right_elbow"] = three_point_angle(
                key_points["right_shoulder"],
                key_points["right_elbow"],
                key_points["right_wrist"],
            )

    except Exception:
        # Never let a calculation error crash the video pipeline
        pass

    return angles
