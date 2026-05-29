"""
utils.py
--------
Pure helper functions used by app.py.  Nothing here has side-effects
outside of drawing on frames or writing files.

Functions
---------
create_output_dirs()
resize_frame(frame, max_width)
draw_angle_label(frame, point, angle, color)
draw_info_overlay(frame, rep_count, phase, exercise_type, angles)
draw_warnings_on_frame(frame, warnings)
plot_angle_history(angle_history, exercise_type) → matplotlib Figure
save_processed_video(frames, path, fps, size)
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — must come before pyplot
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────────
# Directory setup
# ──────────────────────────────────────────────────────────────────────────────

def create_output_dirs():
    """Create outputs/ and sample_videos/ directories if they don't exist."""
    os.makedirs("outputs",       exist_ok=True)
    os.makedirs("sample_videos", exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Frame resize
# ──────────────────────────────────────────────────────────────────────────────

def resize_frame(frame: np.ndarray, max_width: int = 640) -> np.ndarray:
    """
    Shrink a frame so its width ≤ max_width, preserving aspect ratio.
    Returns the original frame unchanged if already small enough.
    """
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale  = max_width / w
    new_w  = max_width
    new_h  = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ──────────────────────────────────────────────────────────────────────────────
# Angle label drawing
# ──────────────────────────────────────────────────────────────────────────────

def draw_angle_label(
    frame: np.ndarray,
    point: dict,
    angle: float,
    color: tuple = (255, 255, 0),
) -> np.ndarray:
    """
    Draw the angle value (e.g. "92°") near a joint on the frame.

    Parameters
    ----------
    point : dict   landmark dict with 'x' and 'y' pixel keys.
    angle : float  angle in degrees.
    color : BGR tuple.
    """
    x = int(point["x"]) + 10
    y = int(point["y"]) - 10

    # Small dark shadow for legibility on bright backgrounds
    cv2.putText(frame, f"{angle:.0f}", (x + 1, y + 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    cv2.putText(frame, f"{angle:.0f}\u00b0", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return frame


# ──────────────────────────────────────────────────────────────────────────────
# Info overlay (top-left panel)
# ──────────────────────────────────────────────────────────────────────────────

def draw_info_overlay(
    frame: np.ndarray,
    rep_count: int,
    phase: str,
    exercise_type: str,
    angles: dict,
) -> np.ndarray:
    """
    Render a semi-transparent info panel in the top-left corner showing:
      - Exercise type
      - Rep count
      - Movement phase
      - Key angles

    Uses addWeighted blending for the translucent background.
    """
    h, w = frame.shape[:2]
    panel_w, panel_h = 260, 170

    # Draw semi-transparent dark rectangle
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (15, 15, 35), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Thin border line for polish
    cv2.rectangle(frame, (0, 0), (panel_w, panel_h), (0, 200, 150), 1)

    # ── Text lines ────────────────────────────────────────────────────
    font  = cv2.FONT_HERSHEY_SIMPLEX
    WHITE = (255, 255, 255)
    CYAN  = (0, 255, 220)
    GOLD  = (30, 200, 255)

    # Title
    cv2.putText(frame, exercise_type.upper(), (10, 28), font, 0.75, CYAN, 2)

    # Rep count
    cv2.putText(frame, f"Reps: {rep_count}", (10, 58), font, 0.65, WHITE, 2)

    # Movement phase
    cv2.putText(frame, f"Phase: {phase}", (10, 88), font, 0.50, GOLD, 1)

    # Angles (up to 3 numeric ones)
    display_keys = ["knee", "hip", "spine"]
    y_cur = 115
    for key in display_keys:
        val = angles.get(key)
        if val is not None and isinstance(val, (int, float)):
            label = key.capitalize()
            cv2.putText(frame, f"{label}: {val:.0f}", (10, y_cur),
                        font, 0.48, (200, 255, 200), 1)
            y_cur += 22

    return frame


# ──────────────────────────────────────────────────────────────────────────────
# Warnings banner (bottom of frame)
# ──────────────────────────────────────────────────────────────────────────────

def draw_warnings_on_frame(frame: np.ndarray, warnings: list) -> np.ndarray:
    """
    Draw up to 3 warning messages along the bottom of the frame.
    Each message gets a translucent red background band.
    """
    if not warnings:
        return frame

    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Limit to 3 warnings to avoid covering the entire frame
    display_warnings = warnings[:3]
    line_h = 28

    for i, msg in enumerate(display_warnings):
        # Strip emoji (OpenCV doesn't render Unicode)
        clean = (
            msg.replace("⚠️", "!")
               .replace("✅", "+")
               .replace("°", " deg")
        )
        # Truncate if too long
        clean = clean[:70]

        y_base = h - (len(display_warnings) - i) * line_h - 5

        # Translucent red band
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y_base - 20), (w, y_base + 6), (20, 0, 100), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Warning text
        cv2.putText(frame, clean, (8, y_base),
                    font, 0.48, (80, 160, 255), 1, cv2.LINE_AA)

    return frame


# ──────────────────────────────────────────────────────────────────────────────
# Angle graph
# ──────────────────────────────────────────────────────────────────────────────

def plot_angle_history(angle_history: list, exercise_type: str):
    """
    Generate a dark-themed 3-panel Matplotlib figure showing
    knee, hip, and spine angles over the video frames.

    Parameters
    ----------
    angle_history : list[dict]  One dict per frame from calculate_all_angles().
    exercise_type : str         'squat' or 'deadlift' — used in the title.

    Returns
    -------
    fig : matplotlib.figure.Figure   (caller must call plt.close(fig))
    """
    BG_DARK  = "#0e1117"
    BG_PANEL = "#1a1f2e"
    GRID     = "#2a2f3e"

    fig, axes = plt.subplots(3, 1, figsize=(11, 7), dpi=100)
    fig.patch.set_facecolor(BG_DARK)
    fig.suptitle(
        f"{exercise_type.capitalize()} — Joint Angle Analysis Over Time",
        fontsize=13, fontweight="bold", color="white", y=0.98,
    )

    frames = list(range(len(angle_history)))

    panel_cfg = [
        ("knee",  "Knee Angle (hip→knee→ankle)",           "#00e676", [60, 90, 120, 150, 180]),
        ("hip",   "Hip Angle (shoulder→hip→knee)",         "#ff6b6b", [45, 90, 135, 180]),
        ("spine", "Spine Inclination (° from vertical)",   "#4fc3f7", [0, 15, 30, 45, 60]),
    ]

    for ax, (key, title, color, ref_lines) in zip(axes, panel_cfg):
        ax.set_facecolor(BG_PANEL)
        ax.grid(color=GRID, linewidth=0.6, linestyle="--")

        # Extract values, skip frames where angle was not computed
        values = [a.get(key) for a in angle_history]
        valid_f = [f for f, v in zip(frames, values) if v is not None]
        valid_v = [v for v in values if v is not None]

        if valid_v:
            ax.plot(valid_f, valid_v, color=color, linewidth=1.8, label=f"{key.capitalize()} angle")
            ax.fill_between(valid_f, valid_v, alpha=0.15, color=color)

            # Min/Max annotations
            max_v = max(valid_v)
            min_v = min(valid_v)
            ax.annotate(f"Max: {max_v:.0f}°", xy=(valid_f[valid_v.index(max_v)], max_v),
                        xytext=(10, 6), textcoords="offset points",
                        fontsize=7, color=color, arrowprops=dict(arrowstyle="-", color=color, lw=0.8))

        # Reference lines
        for ref in ref_lines:
            ax.axhline(y=ref, color="#555", linestyle=":", linewidth=0.9, alpha=0.7)
            ax.text(len(frames) * 0.01, ref + 1, f"{ref}°", fontsize=6, color="#888")

        ax.set_title(title, color="white", fontsize=9.5, pad=4)
        ax.set_ylabel("Degrees (°)", color="#aaa", fontsize=8)
        ax.tick_params(colors="#aaa", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.legend(loc="upper right", fontsize=7.5,
                  facecolor=BG_PANEL, edgecolor=GRID, labelcolor="white")

    axes[-1].set_xlabel("Frame Number", color="#aaa", fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Video saving
# ──────────────────────────────────────────────────────────────────────────────

def save_processed_video(
    frames: list,
    output_path: str,
    fps: float = 30.0,
    size: tuple | None = None,
) -> bool:
    """
    Write a list of BGR numpy frames to an MP4 file.

    Parameters
    ----------
    frames      : list[np.ndarray]
    output_path : str   destination .mp4 path
    fps         : float frames per second
    size        : (w, h) optional resize before writing

    Returns True on success, False if frames list is empty.
    """
    if not frames:
        return False

    h, w = frames[0].shape[:2]
    if size:
        w, h = size

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, float(fps), (w, h))

    for frame in frames:
        out_frame = cv2.resize(frame, (w, h)) if size else frame
        writer.write(out_frame)

    writer.release()
    return True
