"""
feedback.py
-----------
Biomechanics rule engine.

Two public functions:
    analyze_squat(angles, key_points)    → (warnings, good_form)
    analyze_deadlift(angles, key_points) → (warnings, good_form)

Each returns:
    warnings  : list[str]  – issues to correct
    good_form : list[str]  – things being done well

Helper:
    get_form_score(warnings, good_form) → int   (0-100 %)

All threshold constants are defined at the top so they are easy
to tune without digging into logic.
"""

# ──────────────────────────────────────────────────────────────────────────────
# THRESHOLD CONSTANTS  (degrees unless noted)
# ──────────────────────────────────────────────────────────────────────────────

# ── Squat ─────────────────────────────────────────────────────────────────────
SQUAT_DEPTH_GOOD        = 100   # knee angle ≤ this → good depth
SQUAT_DEPTH_EXCELLENT   = 90    # knee angle ≤ this → excellent (below parallel)
SQUAT_DEPTH_BAD         = 120   # knee angle > this → too shallow → warn

SQUAT_LEAN_OK           = 50    # hip angle  ≥ this → acceptable forward lean
SQUAT_LEAN_BAD          = 40    # hip angle  < this → excessive lean

SQUAT_SPINE_OK          = 30    # spine angle ≤ this → neutral
SQUAT_SPINE_WARNING     = 45    # spine angle > this → rounded back

SQUAT_VALGUS_THRESHOLD  = 20    # pixels – knee inside ankle by this much = cave

# ── Deadlift ──────────────────────────────────────────────────────────────────
DL_SPINE_NEUTRAL        = 35    # spine angle ≤ this → neutral
DL_SPINE_SLIGHT         = 50    # spine angle > this → rounded
DL_SPINE_BAD            = 60    # spine angle > this → severe rounding

DL_HIP_KNEE_SYNC        = 35    # if hip angle > knee angle + this → hips shooting
DL_LOCKOUT_THRESHOLD    = 160   # joint angle ≥ this → fully locked out

DL_BAR_PATH_THRESHOLD   = 60    # pixels – wrist–ankle lateral drift


# ──────────────────────────────────────────────────────────────────────────────
# SQUAT ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def analyze_squat(angles: dict, key_points: dict):
    """
    Evaluate squat form using joint angles and key-point positions.

    Checks performed
    ----------------
    1. Squat depth        – via knee angle
    2. Knee valgus (cave) – knee x vs. ankle x alignment
    3. Forward lean       – via hip angle (shoulder-hip-knee)
    4. Spine neutrality   – via spine vertical angle

    Returns
    -------
    warnings  : list[str]  – corrective cues
    good_form : list[str]  – positive feedback
    """
    warnings  = []
    good_form = []

    # ── 1. SQUAT DEPTH ────────────────────────────────────────────────────────
    if "knee" in angles:
        knee = angles["knee"]

        if knee <= SQUAT_DEPTH_EXCELLENT:
            good_form.append(f"✅ Excellent depth — below parallel ({knee:.0f}°)")
        elif knee <= SQUAT_DEPTH_GOOD:
            good_form.append(f"✅ Good squat depth — at parallel ({knee:.0f}°)")
        elif knee <= SQUAT_DEPTH_BAD:
            warnings.append(f"⚠️ Go deeper — squat above parallel ({knee:.0f}°)")
        else:
            warnings.append(f"⚠️ Much deeper required — barely bending knees ({knee:.0f}°)")

    # ── 2. KNEE VALGUS (cave-in) ──────────────────────────────────────────────
    # Works best from a front-facing camera view.
    # In image coordinates: left side has smaller x (left of screen).
    # Valgus: left knee drifts RIGHT (x increases) past left ankle.
    #         right knee drifts LEFT  (x decreases) past right ankle.
    left_caved  = False
    right_caved = False

    if "left_knee" in key_points and "left_ankle" in key_points:
        lk_x = key_points["left_knee"]["x"]
        la_x = key_points["left_ankle"]["x"]
        # Left knee drifting toward centre = x increasing (past ankle)
        left_caved = (lk_x - la_x) > SQUAT_VALGUS_THRESHOLD

    if "right_knee" in key_points and "right_ankle" in key_points:
        rk_x = key_points["right_knee"]["x"]
        ra_x = key_points["right_ankle"]["x"]
        # Right knee drifting toward centre = x decreasing (past ankle)
        right_caved = (ra_x - rk_x) > SQUAT_VALGUS_THRESHOLD

    if left_caved or right_caved:
        side = "both knees" if (left_caved and right_caved) else ("left knee" if left_caved else "right knee")
        warnings.append(f"⚠️ Knee valgus detected ({side}) — push knees out over toes")
    else:
        good_form.append("✅ Good knee-ankle alignment — no valgus")

    # ── 3. FORWARD LEAN ───────────────────────────────────────────────────────
    if "hip" in angles:
        hip = angles["hip"]

        if hip >= SQUAT_LEAN_OK:
            good_form.append(f"✅ Controlled torso lean ({hip:.0f}°) — chest up")
        elif hip >= SQUAT_LEAN_BAD:
            warnings.append(f"⚠️ Moderate forward lean ({hip:.0f}°) — keep chest up")
        else:
            warnings.append(f"⚠️ Excessive forward lean ({hip:.0f}°) — improve ankle mobility or widen stance")

    # ── 4. SPINE NEUTRALITY ───────────────────────────────────────────────────
    if "spine" in angles:
        spine = angles["spine"]

        if spine <= SQUAT_SPINE_OK:
            good_form.append(f"✅ Neutral spine maintained ({spine:.0f}° from vertical)")
        elif spine <= SQUAT_SPINE_WARNING:
            warnings.append(f"⚠️ Slight back rounding ({spine:.0f}°) — brace your core")
        else:
            warnings.append(f"⚠️ Rounded back detected ({spine:.0f}°) — maintain neutral spine")

    return warnings, good_form


# ──────────────────────────────────────────────────────────────────────────────
# DEADLIFT ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def analyze_deadlift(angles: dict, key_points: dict):
    """
    Evaluate deadlift form using joint angles and key-point positions.

    Checks performed
    ----------------
    1. Spine neutrality         – torso inclination angle
    2. Hip-knee synchronisation – hips not shooting up faster than knees extend
    3. Lockout completion       – full hip & knee extension at top
    4. Bar path proximity       – wrist stays close to ankle (vertical bar path)

    Returns
    -------
    warnings  : list[str]
    good_form : list[str]
    """
    warnings  = []
    good_form = []

    # ── 1. SPINE NEUTRALITY ───────────────────────────────────────────────────
    if "spine" in angles:
        spine = angles["spine"]

        if spine <= DL_SPINE_NEUTRAL:
            good_form.append(f"✅ Neutral spine maintained ({spine:.0f}°)")
        elif spine <= DL_SPINE_SLIGHT:
            warnings.append(f"⚠️ Slight back rounding ({spine:.0f}°) — lat activation needed")
        else:
            warnings.append(f"⚠️ Rounded spine ({spine:.0f}°) — maintain neutral back, risk of injury")

    # ── 2. HIP-KNEE SYNCHRONISATION ───────────────────────────────────────────
    # At setup both hip & knee angles are acute (~90°).
    # As the lift progresses, BOTH should extend together.
    # If hip angle greatly exceeds knee angle, hips are shooting up early.
    if "hip" in angles and "knee" in angles:
        hip   = angles["hip"]
        knee  = angles["knee"]
        delta = hip - knee

        if delta > DL_HIP_KNEE_SYNC:
            warnings.append(
                f"⚠️ Hips shooting up early (hip {hip:.0f}° vs knee {knee:.0f}°) — "
                f"drive through heels and keep bar over mid-foot"
            )
        else:
            good_form.append(
                f"✅ Hip-knee synchronisation good (hip {hip:.0f}°, knee {knee:.0f}°)"
            )

    # ── 3. LOCKOUT COMPLETION ─────────────────────────────────────────────────
    if "knee" in angles and "hip" in angles:
        knee = angles["knee"]
        hip  = angles["hip"]

        knee_locked = knee >= DL_LOCKOUT_THRESHOLD
        hip_locked  = hip  >= DL_LOCKOUT_THRESHOLD

        if knee_locked and hip_locked:
            good_form.append(
                f"✅ Full lockout achieved (knee {knee:.0f}°, hip {hip:.0f}°)"
            )
        elif not knee_locked and not hip_locked:
            # Only show this cue when near the top of the lift (mid-phase hips are fine)
            if hip > 120:
                warnings.append(
                    f"⚠️ Incomplete lockout — fully extend hips and knees at the top"
                )
        elif hip_locked and not knee_locked:
            warnings.append(f"⚠️ Hyperextending hips — straighten knees fully too")

    # ── 4. BAR PATH (wrist proximity to ankle) ───────────────────────────────
    # In an optimal deadlift the bar travels vertically close to the shins.
    # Wrists follow the bar → wrist x ≈ ankle x throughout the lift.
    bar_checked = False
    for side in [("left_wrist", "left_ankle"), ("right_wrist", "right_ankle")]:
        wrist_key, ankle_key = side
        if wrist_key in key_points and ankle_key in key_points:
            drift = abs(
                key_points[wrist_key]["x"] - key_points[ankle_key]["x"]
            )
            if drift > DL_BAR_PATH_THRESHOLD:
                warnings.append(
                    f"⚠️ Bar drifting away from body — keep bar close to shins"
                )
            else:
                good_form.append("✅ Good vertical bar path")
            bar_checked = True
            break   # Only check one side to avoid duplicate messages

    return warnings, good_form


# ──────────────────────────────────────────────────────────────────────────────
# FORM SCORE
# ──────────────────────────────────────────────────────────────────────────────

def get_form_score(warnings: list, good_form: list) -> int:
    """
    Calculate a simple form quality score (0–100 %).

    Formula:  good_checks / total_checks  × 100

    Returns 0 if no checks were triggered (no landmarks detected).
    """
    total = len(warnings) + len(good_form)
    if total == 0:
        return 0
    score = (len(good_form) / total) * 100
    return round(score)
