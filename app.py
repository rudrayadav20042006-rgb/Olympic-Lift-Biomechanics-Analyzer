"""
app.py
------
Entry point for the Olympic Lift Biomechanics Analyzer.

Run with:
    streamlit run app.py

Architecture
------------
  1. Sidebar        – exercise selector, tips, about panel
  2. Tab 1: Upload  – file uploader, video preview, Analyze button
  3. Tab 2: Results – metrics, form score, feedback, angle graph
  4. Tab 3: About   – tech stack, biomechanics reference, future work

Processing pipeline (process_video):
  Upload → OpenCV frame loop → MediaPipe pose detection
  → angle calculation → feedback rules → rep counting
  → frame annotation → save processed video
"""

import os
import tempfile

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from pose_detector    import PoseDetector
from angle_calculator import calculate_all_angles
from feedback         import analyze_squat, analyze_deadlift, get_form_score
from rep_counter      import RepCounter
from utils            import (
    create_output_dirs,
    resize_frame,
    draw_angle_label,
    draw_info_overlay,
    draw_warnings_on_frame,
    plot_angle_history,
    save_processed_video,
)

# ──────────────────────────────────────────────────────────────────────────────
# Page configuration (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "Olympic Lift Biomechanics Analyzer",
    page_icon  = "🏋️",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS – dark sporty theme
# ──────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* ── Global ── */
    html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

    /* ── Gradient header ── */
    .main-header {
        font-size: 2.6rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #00e676, #00b4d8, #9b5de5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        text-align: center;
        color: #8892a4;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    /* ── Feedback boxes ── */
    .warn-box {
        background-color: #1f0a0a;
        border-left: 4px solid #ff4444;
        border-radius: 0 8px 8px 0;
        padding: 0.65rem 1rem;
        margin: 0.3rem 0;
        font-size: 0.9rem;
        color: #ffaaaa;
    }
    .good-box {
        background-color: #0a1f0a;
        border-left: 4px solid #44ff88;
        border-radius: 0 8px 8px 0;
        padding: 0.65rem 1rem;
        margin: 0.3rem 0;
        font-size: 0.9rem;
        color: #aaffcc;
    }

    /* ── Progress bar colour ── */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00e676, #00b4d8);
    }

    /* ── Metric delta colour fix ── */
    [data-testid="metric-container"] { border-radius: 10px; }

    /* ── Divider ── */
    hr { border-color: #2a2f3e; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# Core processing pipeline
# ──────────────────────────────────────────────────────────────────────────────

def process_video(
    video_path:    str,
    exercise_type: str,
    progress_bar,
    status_text,
) -> dict:
    """
    Read every frame from video_path, run the full biomechanics pipeline,
    and return a results dictionary.

    Parameters
    ----------
    video_path    : str   path to the uploaded (temp) video file
    exercise_type : str   'squat' or 'deadlift'
    progress_bar  : Streamlit progress widget
    status_text   : Streamlit empty widget for status messages

    Returns
    -------
    dict with keys:
        frames        – list of annotated BGR numpy frames
        angle_history – list of angle dicts (one per frame)
        rep_count     – int
        form_score    – int (0–100)
        warnings      – list[str]  unique warning messages
        good_form     – list[str]  unique positive messages
        fps           – float
    """

    # ── Initialise components ──────────────────────────────────────────
    detector    = PoseDetector(model_complexity=1)
    rep_counter = RepCounter(exercise_type)

    cap          = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0

    processed_frames: list   = []
    angle_history:    list   = []
    all_warnings:     list   = []
    all_good_form:    list   = []
    frame_idx:        int    = 0

    # ── Frame loop ────────────────────────────────────────────────────
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Progress update
        if total_frames > 0:
            progress_bar.progress(min(frame_idx / total_frames, 1.0))
        status_text.text(f"⚙️  Analysing frame {frame_idx} / {total_frames} …")

        # Keep resolution manageable for speed
        frame = resize_frame(frame, max_width=640)

        # ── Pose detection ─────────────────────────────────────────────
        results   = detector.detect_pose(frame)
        landmarks = detector.extract_landmarks(results, frame.shape)

        # Draw skeleton regardless of angle availability
        frame = detector.draw_landmarks(frame, results)

        if landmarks:
            key_points = detector.get_key_points(landmarks)

            # ── Angle calculation ──────────────────────────────────────
            angles = calculate_all_angles(key_points)
            angle_history.append(angles)

            # ── Biomechanics feedback ──────────────────────────────────
            if exercise_type == "squat":
                warnings, good_form = analyze_squat(angles, key_points)
            else:
                warnings, good_form = analyze_deadlift(angles, key_points)

            all_warnings.extend(warnings)
            all_good_form.extend(good_form)

            # ── Rep counting ───────────────────────────────────────────
            rep_count, phase = rep_counter.update(key_points)

            # ── Angle labels on frame ──────────────────────────────────
            if "knee" in angles and "left_knee" in key_points:
                frame = draw_angle_label(frame, key_points["left_knee"],
                                         angles["knee"],  (0, 255, 150))

            if "hip" in angles and "left_hip" in key_points:
                frame = draw_angle_label(frame, key_points["left_hip"],
                                         angles["hip"],   (0, 220, 255))

            if "spine" in angles and "mid_hip" in angles:
                # mid_hip is stored inside angles by calculate_all_angles
                frame = draw_angle_label(frame, angles["mid_hip"],
                                         angles["spine"], (255, 140, 0))

            # ── Info panel ─────────────────────────────────────────────
            display_angles = {
                k: v for k, v in angles.items()
                if k in ("knee", "hip", "spine") and isinstance(v, (int, float))
            }
            frame = draw_info_overlay(frame, rep_count, rep_counter.phase_label,
                                      exercise_type, display_angles)

            # ── Warnings banner ────────────────────────────────────────
            if warnings:
                frame = draw_warnings_on_frame(frame, warnings)

        else:
            # No pose detected — record empty angles so history stays aligned
            angle_history.append({})

        processed_frames.append(frame.copy())

    cap.release()
    detector.release()

    # ── Aggregate results ──────────────────────────────────────────────
    unique_warnings  = list(dict.fromkeys(all_warnings))   # deduplicate, keep order
    unique_good_form = list(dict.fromkeys(all_good_form))
    form_score       = get_form_score(unique_warnings, unique_good_form)

    return {
        "frames":        processed_frames,
        "angle_history": angle_history,
        "rep_count":     rep_counter.rep_count,
        "form_score":    form_score,
        "warnings":      unique_warnings,
        "good_form":     unique_good_form,
        "fps":           fps,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Helper – save frames → temp MP4, return bytes
# ──────────────────────────────────────────────────────────────────────────────

def frames_to_mp4_bytes(frames: list, fps: float) -> tuple[bytes, str] | tuple[None, None]:
    """
    Write annotated frames to outputs/processed_output.mp4 and
    return the raw bytes alongside the file path.
    """
    if not frames:
        return None, None

    out_path = os.path.join("outputs", "processed_output.mp4")
    success  = save_processed_video(frames, out_path, fps)

    if not success:
        return None, None

    with open(out_path, "rb") as f:
        return f.read(), out_path


# ──────────────────────────────────────────────────────────────────────────────
# Results display
# ──────────────────────────────────────────────────────────────────────────────

def display_results(results: dict, exercise_type: str):
    """Render all result sections in Tab 2."""

    # ── Metric cards ──────────────────────────────────────────────────
    st.markdown("### 📊 Performance Summary")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("🔄 Total Reps",       results["rep_count"])
    with c2:
        st.metric("📐 Form Score",       f"{results['form_score']} %")
    with c3:
        st.metric("⚠️ Issues Detected",  len(results["warnings"]))
    with c4:
        st.metric("🎞️ Frames Processed", len(results["frames"]))

    # ── Form quality progress bar ──────────────────────────────────────
    st.markdown("**Form Quality Score**")
    score = results["form_score"]
    st.progress(score / 100)

    if score >= 80:
        st.success(f"🏆 Excellent technique!  Score: **{score} %**")
    elif score >= 55:
        st.warning(f"👍 Good form with some corrections needed.  Score: **{score} %**")
    else:
        st.error(f"⚠️ Significant form issues detected.  Score: **{score} %**  — review warnings below.")

    st.divider()

    # ── Feedback columns ───────────────────────────────────────────────
    col_w, col_g = st.columns(2)

    with col_w:
        st.markdown("### ⚠️ Issues & Corrections")
        if results["warnings"]:
            for msg in results["warnings"]:
                st.markdown(f'<div class="warn-box">{msg}</div>', unsafe_allow_html=True)
        else:
            st.success("🎉 No major form issues detected — great work!")

    with col_g:
        st.markdown("### ✅ Good Form Points")
        if results["good_form"]:
            for msg in results["good_form"]:
                st.markdown(f'<div class="good-box">{msg}</div>', unsafe_allow_html=True)
        else:
            st.info("Upload a video to receive positive feedback.")

    st.divider()

    # ── Joint angle graph ──────────────────────────────────────────────
    if results["angle_history"]:
        st.markdown("### 📈 Joint Angle Time-Series")
        st.caption(
            "Each panel shows angle (°) vs. frame number. "
            "Dashed lines are biomechanical reference thresholds."
        )
        fig = plot_angle_history(results["angle_history"], exercise_type)
        st.pyplot(fig)
        plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────

def render_sidebar() -> str:
    """Render sidebar controls and return the selected exercise type."""

    with st.sidebar:
        st.markdown(
            "<h2 style='color:#00e676;text-align:center;'>🏋️ Lift Analyzer</h2>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        exercise_type = st.selectbox(
            "🏋️ **Select Exercise**",
            options=["squat", "deadlift"],
            format_func=lambda x: "🦵 Back Squat" if x == "squat" else "💪 Deadlift",
        )

        st.markdown("---")
        st.markdown("### 📋 Analysis Checklist")

        if exercise_type == "squat":
            st.markdown(
                """
                - 🔵 Squat depth (knee angle)
                - 🔵 Knee valgus detection
                - 🔵 Forward lean check
                - 🔵 Spine neutrality
                - 🔵 Rep counting (hip tracking)
                """
            )
        else:
            st.markdown(
                """
                - 🔵 Spine neutrality
                - 🔵 Hip-knee synchronisation
                - 🔵 Lockout completion
                - 🔵 Bar path proximity
                - 🔵 Rep counting (hip tracking)
                """
            )

        st.markdown("---")
        st.markdown("### 📸 Camera Tips")
        st.info(
            "**Best results with:**\n"
            "- Side-view (90° to lifter)\n"
            "- Full body in frame\n"
            "- Good lighting\n"
            "- Stable camera\n"
            "- Minimal background clutter"
        )

        st.markdown("---")
        st.markdown(
            "<small>Built with 🐍 Python · 👁️ MediaPipe · 🎥 OpenCV · 📊 Matplotlib · "
            "🚀 Streamlit</small>",
            unsafe_allow_html=True,
        )

    return exercise_type


# ──────────────────────────────────────────────────────────────────────────────
# ABOUT TAB
# ──────────────────────────────────────────────────────────────────────────────

def render_about_tab():
    st.markdown("### ℹ️ About This Project")

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown(
            """
            #### 🎯 Project Goal
            Automate the qualitative feedback that a strength coach
            gives by watching a lift — but using computer vision,
            coordinate geometry, and rule-based biomechanics logic.

            #### 🔬 Technology Stack

            | Component | Library |
            |-----------|---------|
            | Pose estimation | MediaPipe Pose |
            | Video I/O | OpenCV |
            | Maths | NumPy |
            | Dashboard | Streamlit |
            | Graphs | Matplotlib |

            #### 📐 Angle Definitions

            | Name | Landmarks used |
            |------|---------------|
            | Knee angle | Hip → Knee → Ankle |
            | Hip angle | Shoulder → Hip → Knee |
            | Spine angle | Mid-shoulder → Mid-hip vs. vertical |
            """
        )

    with col_r:
        st.markdown(
            """
            #### 🦵 Squat Thresholds

            | Check | Threshold |
            |-------|-----------|
            | Excellent depth | Knee ≤ 90° |
            | Acceptable depth | Knee ≤ 100° |
            | Too shallow | Knee > 120° |
            | Neutral spine | Spine ≤ 30° |
            | Excessive lean | Hip < 40° |

            #### 💪 Deadlift Thresholds

            | Check | Threshold |
            |-------|-----------|
            | Neutral spine | Spine ≤ 35° |
            | Slight rounding | Spine 35–50° |
            | Hip shooting | Hip > Knee + 35° |
            | Full lockout | Both joints ≥ 160° |

            #### 🔁 Rep Counting Logic
            5-frame smoothed hip y-coordinate drives a
            **4-state FSM**: Standing → Descending →
            Ascending → Standing. A rep is logged on
            each Standing re-entry.
            """
        )

    st.divider()
    st.markdown(
        """
        #### 🔮 Future Improvements

        - **3-D pose**: Use MediaPipe Holistic or BlazePose 3D for true depth angles
        - **Real-time webcam**: Stream live from `st.camera_input`
        - **Barbell detection**: YOLOv8 to track bar position independently
        - **Session history**: SQLite storage for rep-by-rep progression tracking
        - **AI cue generation**: Pass angle data to an LLM for natural-language coaching
        - **Multi-angle fusion**: Combine front + side cameras for full valgus detection
        - **Mobile app**: Wrap in a React Native or Flutter shell using the same backend
        """
    )

    st.caption("Olympic Lift Biomechanics Analyzer — Mechanical Engineering × AI × Sports Science")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    create_output_dirs()

    # ── Header ──────────────────────────────────────────────────────────
    st.markdown(
        '<h1 class="main-header">🏋️ Olympic Lift Biomechanics Analyzer</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">'
        "AI-powered squat &amp; deadlift form analysis using MediaPipe + OpenCV"
        "</p>",
        unsafe_allow_html=True,
    )

    exercise_type = render_sidebar()

    # ── Tabs ─────────────────────────────────────────────────────────────
    tab_upload, tab_results, tab_about = st.tabs(
        ["📤 Upload & Analyse", "📊 Results", "ℹ️ About"]
    )

    # ═════════════════════════════════════════════════════════════════════
    # TAB 1 – UPLOAD
    # ═════════════════════════════════════════════════════════════════════
    with tab_upload:
        st.markdown(f"### 📤 Upload a **{exercise_type.capitalize()}** Video")

        col_up, col_hint = st.columns([2, 1])

        with col_up:
            uploaded = st.file_uploader(
                "Drag & drop or browse for your video",
                type=["mp4", "avi", "mov", "mkv"],
                help="Side-view recordings work best for angle accuracy.",
            )

        with col_hint:
            st.markdown(
                """
                **Supported formats:** MP4, AVI, MOV, MKV

                **Recommended:**
                - Resolution ≥ 480p
                - Frame rate ≥ 24 fps
                - Duration 10 – 120 s
                - Side-on or front-facing
                """
            )

        if uploaded is not None:
            # ── Preview uploaded video ───────────────────────────────────
            st.markdown("#### 🎬 Uploaded Video Preview")
            st.video(uploaded)

            st.markdown(
                f"**File:** `{uploaded.name}`  &nbsp;|&nbsp;"
                f"**Size:** `{uploaded.size / 1024:.1f} KB`  &nbsp;|&nbsp;"
                f"**Exercise:** `{exercise_type}`"
            )

            st.divider()

            col_btn, col_note = st.columns([1, 2])
            with col_btn:
                analyse_btn = st.button(
                    "🔍 Analyse Video",
                    type="primary",
                    use_container_width=True,
                )
            with col_note:
                st.info(
                    "Processing time ≈ 1–3 × real-time depending on resolution. "
                    "The annotated video will appear below once done."
                )

            if analyse_btn:
                # Save uploaded bytes to a temp file OpenCV can open
                uploaded.seek(0)
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".mp4"
                ) as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name

                st.markdown("---")
                st.markdown("### ⚙️ Processing …")
                progress_bar = st.progress(0)
                status_text  = st.empty()

                try:
                    with st.spinner("Running biomechanics analysis …"):
                        results = process_video(
                            tmp_path,
                            exercise_type,
                            progress_bar,
                            status_text,
                        )

                    progress_bar.progress(1.0)
                    status_text.text("✅ Analysis complete!")

                    # Persist results in session state for Tab 2
                    st.session_state["results"]       = results
                    st.session_state["exercise_type"] = exercise_type

                    st.success(
                        "✅ Video analysed!  "
                        "Switch to the **📊 Results** tab for full details."
                    )

                    # ── Processed video preview ───────────────────────────
                    st.markdown("### 🎬 Annotated Output Video")
                    video_bytes, out_path = frames_to_mp4_bytes(
                        results["frames"], results["fps"]
                    )

                    if video_bytes:
                        st.video(video_bytes)
                        st.download_button(
                            label     = "⬇️ Download Annotated Video",
                            data      = video_bytes,
                            file_name = f"{exercise_type}_analysed.mp4",
                            mime      = "video/mp4",
                        )
                    else:
                        st.warning(
                            "Could not render processed video — "
                            "check that OpenCV has write permission to outputs/."
                        )

                except Exception as exc:
                    st.error(f"❌ An error occurred during processing: {exc}")
                    st.exception(exc)

                finally:
                    # Always clean up the temp file
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

    # ═════════════════════════════════════════════════════════════════════
    # TAB 2 – RESULTS
    # ═════════════════════════════════════════════════════════════════════
    with tab_results:
        if "results" in st.session_state:
            display_results(
                st.session_state["results"],
                st.session_state["exercise_type"],
            )
        else:
            st.info(
                "📤 Upload and analyse a video first (Tab 1) to see "
                "your biomechanics report here."
            )
            # Placeholder empty metric cards
            st.markdown("#### Dashboard Preview")
            pc1, pc2, pc3, pc4 = st.columns(4)
            pc1.metric("🔄 Total Reps",       "—")
            pc2.metric("📐 Form Score",        "— %")
            pc3.metric("⚠️ Issues Detected",   "—")
            pc4.metric("🎞️ Frames Processed",  "—")

    # ═════════════════════════════════════════════════════════════════════
    # TAB 3 – ABOUT
    # ═════════════════════════════════════════════════════════════════════
    with tab_about:
        render_about_tab()


if __name__ == "__main__":
    main()
