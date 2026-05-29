"""
rep_counter.py
--------------
Counts repetitions for squats and deadlifts using a
finite state machine (FSM) driven by hip vertical position.

How it works
------------
MediaPipe gives us normalised y-coordinates (0 = top, 1 = bottom).
When an athlete squats down, the hips move DOWN in the image
→ y_norm INCREASES.  When they stand back up, y_norm DECREASES.

The FSM has four states:
    STANDING    → initial / top position
    GOING_DOWN  → hips descending
    BOTTOM      → hips reached lowest point and reversed
    GOING_UP    → hips ascending back toward start

A completed rep is recorded when the lifter returns to STANDING
after having been in GOING_UP.

A 5-frame rolling average smooths out MediaPipe jitter so that
small wobbles don't trigger false state transitions.
"""


class RepCounter:
    """
    Hip-based repetition counter.

    Parameters
    ----------
    exercise_type : str
        'squat' or 'deadlift' — kept for potential future
        exercise-specific tuning.
    descent_threshold : float
        How many normalised-y units the hip must drop below the
        standing baseline before we consider the rep started.
        Default 0.04 ≈ 4 % of frame height.
    ascent_threshold : float
        How close to the baseline the hip must return (above min_depth)
        before we trigger the state change back toward STANDING.
    """

    # State labels
    STANDING   = "standing"
    GOING_DOWN = "going_down"
    BOTTOM     = "bottom"
    GOING_UP   = "going_up"

    def __init__(
        self,
        exercise_type: str = "squat",
        descent_threshold: float = 0.04,
        ascent_threshold: float  = 0.04,
    ):
        self.exercise_type     = exercise_type
        self.descent_threshold = descent_threshold
        self.ascent_threshold  = ascent_threshold

        # Public state
        self.rep_count = 0
        self.state     = self.STANDING

        # Internal tracking
        self._hip_y_history: list[float] = []   # raw y_norm values
        self._start_y: float | None      = None  # baseline (standing hip y)
        self._min_depth: float | None    = None  # deepest y during descent
        self._warmup_frames: int         = 0     # wait for stable baseline
        self._WARMUP: int                = 10    # frames to collect before tracking
        self._SMOOTH_WINDOW: int         = 5     # rolling average window

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, key_points: dict) -> tuple[int, str]:
        """
        Feed the latest key-points and advance the FSM.

        Call once per processed frame.

        Parameters
        ----------
        key_points : dict  Named key-point dict from PoseDetector.

        Returns
        -------
        (rep_count, state) : (int, str)
        """
        hip_y = self._get_hip_y_norm(key_points)

        if hip_y is None:
            # No hip detected — hold current state
            return self.rep_count, self.state

        # Accumulate raw history for smoothing
        self._hip_y_history.append(hip_y)
        if len(self._hip_y_history) > 60:   # cap memory
            self._hip_y_history.pop(0)

        # Wait for enough frames to establish a stable baseline
        self._warmup_frames += 1
        if self._warmup_frames <= self._WARMUP:
            # Use these frames to set the initial standing baseline
            self._start_y = self._smooth_y()
            return self.rep_count, self.state

        smooth_y = self._smooth_y()

        # ── FSM transitions ───────────────────────────────────────────
        if self.state == self.STANDING:
            # Hips have dropped below baseline → rep is starting
            if smooth_y > self._start_y + self.descent_threshold:
                self.state     = self.GOING_DOWN
                self._min_depth = smooth_y

        elif self.state == self.GOING_DOWN:
            # Track the deepest point reached
            if smooth_y > self._min_depth:
                self._min_depth = smooth_y
            # If hips start rising again → transition to GOING_UP
            if smooth_y < self._min_depth - self.ascent_threshold:
                self.state = self.GOING_UP

        elif self.state == self.GOING_UP:
            # Hip has returned to within ascent_threshold of the baseline
            if smooth_y <= self._start_y + self.descent_threshold * 0.5:
                self.rep_count += 1
                self.state      = self.STANDING
                self._min_depth = None
                # Update baseline dynamically (handles camera drift)
                self._start_y   = smooth_y

        return self.rep_count, self.state

    def reset(self):
        """Reset all counters and state (e.g. for a new video)."""
        self.rep_count       = 0
        self.state           = self.STANDING
        self._hip_y_history  = []
        self._start_y        = None
        self._min_depth      = None
        self._warmup_frames  = 0

    @property
    def phase_label(self) -> str:
        """Human-readable movement phase for overlay display."""
        labels = {
            self.STANDING:   "Standing",
            self.GOING_DOWN: "Descending",
            self.BOTTOM:     "Bottom",
            self.GOING_UP:   "Ascending",
        }
        return labels.get(self.state, self.state.capitalize())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_hip_y_norm(self, key_points: dict) -> float | None:
        """Average normalised y of left and right hips (whichever visible)."""
        left_y  = key_points.get("left_hip",  {}).get("y_norm")
        right_y = key_points.get("right_hip", {}).get("y_norm")

        if left_y is not None and right_y is not None:
            return (left_y + right_y) / 2.0
        return left_y if left_y is not None else right_y

    def _smooth_y(self) -> float:
        """Rolling mean of the last SMOOTH_WINDOW y values."""
        window = self._hip_y_history[-self._SMOOTH_WINDOW:]
        return sum(window) / len(window) if window else 0.0
