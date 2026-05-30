"""
rep_counter.py
--------------
Knee-angle-driven repetition counter using a two-state FSM
with hysteresis.

WHY KNEE ANGLE — NOT HIP Y-POSITION
-------------------------------------
Hip vertical displacement during a lift is only ~4 % of frame height
(normalised y ≈ 0.04 units). MediaPipe jitter alone can be 1–2 %,
giving a true signal-to-noise ratio of ~2:1 — far too low.

Knee angle, by contrast, travels 80–90 ° between standing and depth,
providing a 40–50× larger signal window that noise cannot bridge.

FSM DESIGN
----------
Two stable states with a hysteresis gap:

    EXTENDED  ← knee angle ≥ UP_THRESH   (standing / lockout)
    FLEXED    ← knee angle ≤ DOWN_THRESH  (at depth / setup)
    <buffer>     DOWN_THRESH < angle < UP_THRESH  (transition zone — hold state)

The buffer zone is the hysteresis gap. Noise cannot flip the state
because it would need to travel the full width of the gap.

Counting rule (exercise-specific):
    Squat    : FLEXED → EXTENDED  transition  = 1 completed rep
                (athlete stood back up from depth)
    Deadlift : EXTENDED → FLEXED  transition  = 1 completed rep
                (athlete lowered bar back to floor after lockout)

SMOOTHING
---------
A 7-frame rolling average (collections.deque, O(1) append/pop) removes
per-frame MediaPipe jitter without introducing the lag of a longer window.

DEBOUNCE
--------
A minimum-frame guard (_MIN_REP_FRAMES = 12) prevents a single
smooth transition from being counted twice if the angle lingers
exactly on the threshold boundary.
"""

from collections import deque


# ──────────────────────────────────────────────────────────────────────────────
# Threshold constants  (all in degrees)
# ──────────────────────────────────────────────────────────────────────────────

# Knee angle at which we consider the lifter "at the top / extended"
_UP_THRESH: float = 155.0

# Knee angle at which we consider the lifter "at the bottom / flexed"
_DOWN_THRESH: float = 115.0

# Hysteresis gap = UP_THRESH − DOWN_THRESH = 40 °
# Noise must span this entire gap to cause a false state flip.

_SMOOTH_WINDOW:  int = 7   # rolling average frame count
_WARMUP_FRAMES:  int = 15  # discard first N frames (MediaPipe not yet stable)
_MIN_REP_FRAMES: int = 12  # minimum frames between counted reps (debounce)


# ──────────────────────────────────────────────────────────────────────────────
# State labels  (public so app.py can display them)
# ──────────────────────────────────────────────────────────────────────────────

STATE_EXTENDED = "extended"   # knee straight  (standing / lockout)
STATE_FLEXED   = "flexed"     # knee bent      (at depth / setup)
STATE_BUFFER   = "buffer"     # transition zone — state held


class RepCounter:
    """
    Counts repetitions for squats and deadlifts.

    Parameters
    ----------
    exercise_type : str
        'squat' or 'deadlift'
        Controls which state-transition direction increments the rep count.
    """

    def __init__(self, exercise_type: str = "squat"):
        if exercise_type not in ("squat", "deadlift"):
            raise ValueError(f"exercise_type must be 'squat' or 'deadlift', got '{exercise_type}'")

        self.exercise_type = exercise_type

        # ── Public ────────────────────────────────────────────────────
        self.rep_count: int = 0
        self.state:     str = (
            STATE_EXTENDED if exercise_type == "squat" else STATE_FLEXED
        )
        # Squats start standing (extended); deadlifts start at setup (flexed).

        # ── Internal ──────────────────────────────────────────────────
        self._angle_buf:      deque  = deque(maxlen=_SMOOTH_WINDOW)
        self._warmup_count:   int    = 0
        self._frames_since_rep: int  = _MIN_REP_FRAMES  # start ready to count

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def update(self, knee_angle: float | None) -> tuple[int, str]:
        """
        Advance the FSM with the current frame's (smoothed) knee angle.

        Parameters
        ----------
        knee_angle : float | None
            Averaged knee angle from calculate_all_angles().
            Pass None if landmarks were not detected this frame.

        Returns
        -------
        (rep_count, state) : (int, str)
        """
        # ── Missing landmark — hold state, advance debounce clock ──────
        if knee_angle is None:
            self._frames_since_rep += 1
            return self.rep_count, self.state

        # ── Warmup: accumulate initial frames, don't count yet ─────────
        self._warmup_count += 1
        self._angle_buf.append(float(knee_angle))

        if self._warmup_count <= _WARMUP_FRAMES:
            # During warmup, set the initial state from actual angles
            # so we don't miscount if the video starts mid-rep.
            if len(self._angle_buf) == _SMOOTH_WINDOW:
                avg = self._smooth()
                if avg >= _UP_THRESH:
                    self.state = STATE_EXTENDED
                elif avg <= _DOWN_THRESH:
                    self.state = STATE_FLEXED
            return self.rep_count, self.state

        smooth = self._smooth()
        self._frames_since_rep += 1

        # ── FSM ────────────────────────────────────────────────────────
        if smooth >= _UP_THRESH:
            new_state = STATE_EXTENDED
        elif smooth <= _DOWN_THRESH:
            new_state = STATE_FLEXED
        else:
            # Inside the hysteresis buffer zone — hold the previous state.
            return self.rep_count, self.state

        # ── Count on the qualifying state transition ───────────────────
        if new_state != self.state and self._frames_since_rep >= _MIN_REP_FRAMES:
            rep_transition = (
                (STATE_FLEXED,   STATE_EXTENDED)  # squat:    rose from depth
                if self.exercise_type == "squat"
                else
                (STATE_EXTENDED, STATE_FLEXED)    # deadlift: lowered from lockout
            )
            if (self.state, new_state) == rep_transition:
                self.rep_count     += 1
                self._frames_since_rep = 0

        self.state = new_state
        return self.rep_count, self.state

    def reset(self) -> None:
        """Reset all counters and state for a new video."""
        self.rep_count         = 0
        self.state             = (
            STATE_EXTENDED if self.exercise_type == "squat" else STATE_FLEXED
        )
        self._angle_buf        = deque(maxlen=_SMOOTH_WINDOW)
        self._warmup_count     = 0
        self._frames_since_rep = _MIN_REP_FRAMES

    @property
    def phase_label(self) -> str:
        """Human-readable movement phase for the on-frame overlay."""
        return {
            STATE_EXTENDED: "Standing / Lockout",
            STATE_FLEXED:   "At Depth / Setup",
            STATE_BUFFER:   "Transitioning",
        }.get(self.state, self.state.capitalize())

    # ──────────────────────────────────────────────────────────────────
    # Private
    # ──────────────────────────────────────────────────────────────────

    def _smooth(self) -> float:
        """Rolling mean of the current angle buffer."""
        return sum(self._angle_buf) / len(self._angle_buf)
