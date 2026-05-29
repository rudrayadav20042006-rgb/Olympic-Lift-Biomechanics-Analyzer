"""
pose_detector.py
----------------
Wraps Google MediaPipe Pose to:
  - Detect 33 body landmarks in each video frame
  - Draw the skeleton overlay (bones + joints)
  - Return a clean dictionary of named key-points
    (shoulders, hips, knees, ankles, elbows, wrists)

MediaPipe Pose landmark indices used in this project:
  0  – nose
  11 – left shoulder   12 – right shoulder
  13 – left elbow      14 – right elbow
  15 – left wrist      16 – right wrist
  23 – left hip        24 – right hip
  25 – left knee       26 – right knee
  27 – left ankle      28 – right ankle
"""

import cv2
import mediapipe as mp


class PoseDetector:
    """
    Handles all MediaPipe Pose operations.

    Parameters
    ----------
    static_image_mode : bool
        True for images, False for video (enables temporal tracking).
    model_complexity : int
        0 = fastest/least accurate, 1 = balanced, 2 = most accurate.
    min_detection_confidence : float
        Threshold for initial person detection (0.0 – 1.0).
    min_tracking_confidence : float
        Threshold for landmark tracking across frames (0.0 – 1.0).
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------

    def detect_pose(self, frame):
        """
        Run MediaPipe Pose on a single BGR frame.

        Parameters
        ----------
        frame : np.ndarray  BGR image from OpenCV.

        Returns
        -------
        results : MediaPipe Pose result object containing pose_landmarks.
        """
        # MediaPipe expects RGB; OpenCV reads BGR → convert first
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Mark as not writeable to improve performance (avoids a copy)
        rgb_frame.flags.writeable = False
        results = self.pose.process(rgb_frame)
        rgb_frame.flags.writeable = True

        return results

    # ------------------------------------------------------------------
    # Landmark extraction
    # ------------------------------------------------------------------

    def extract_landmarks(self, results, frame_shape: tuple) -> dict:
        """
        Convert MediaPipe landmark objects into a plain dictionary.

        Key = landmark index (int)
        Value = dict with:
            x, y       – pixel coordinates (scaled by frame dimensions)
            x_norm, y_norm – original normalised coordinates [0, 1]
            z          – depth estimate (relative, not in metres)
            visibility – confidence score [0, 1]

        Returns an empty dict if no pose was detected.
        """
        landmarks = {}

        if results.pose_landmarks is None:
            return landmarks

        h, w = frame_shape[:2]

        for idx, lm in enumerate(results.pose_landmarks.landmark):
            landmarks[idx] = {
                "x": lm.x * w,          # pixel x
                "y": lm.y * h,          # pixel y
                "z": lm.z,              # relative depth
                "visibility": lm.visibility,
                "x_norm": lm.x,         # normalised [0,1]
                "y_norm": lm.y,         # normalised [0,1]
            }

        return landmarks

    # ------------------------------------------------------------------
    # Skeleton drawing
    # ------------------------------------------------------------------

    def draw_landmarks(self, frame, results):
        """
        Draw pose skeleton (joints + connections) on the frame in-place.

        Uses MediaPipe's default drawing styles which colour-code
        left/right body sides and vary joint sizes by confidence.

        Returns the annotated frame (same object, modified in-place).
        """
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
            )
        return frame

    # ------------------------------------------------------------------
    # Named key-point extraction
    # ------------------------------------------------------------------

    def get_key_points(self, landmarks: dict) -> dict:
        """
        Extract the biomechanically relevant joints by name.

        Returns a dict like:
            {
              'left_shoulder': {'x': 320, 'y': 180, ...},
              'right_knee':    {'x': 280, 'y': 480, ...},
              ...
            }

        Only includes a joint if its landmark was detected.
        """
        # Map human-readable names → MediaPipe landmark indices
        key_index_map = {
            "nose":            0,
            "left_shoulder":   11,
            "right_shoulder":  12,
            "left_elbow":      13,
            "right_elbow":     14,
            "left_wrist":      15,
            "right_wrist":     16,
            "left_hip":        23,
            "right_hip":       24,
            "left_knee":       25,
            "right_knee":      26,
            "left_ankle":      27,
            "right_ankle":     28,
        }

        key_points = {}
        for name, idx in key_index_map.items():
            if idx in landmarks:
                # Only include joints with reasonable visibility
                if landmarks[idx]["visibility"] > 0.3:
                    key_points[name] = landmarks[idx]

        return key_points

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def release(self):
        """Close the MediaPipe Pose model and free resources."""
        self.pose.close()
