# Lift Biomechanics Analyzer

> **AI-powered squat & deadlift form analysis using Computer Vision, MediaPipe, and Streamlit.**
>
> A Mechanical Engineering × Artificial Intelligence × Sports Biomechanics project.

---

## 📌 Table of Contents

1. [Project Overview](#project-overview)
2. [Live Demo Screenshots](#live-demo-screenshots)
3. [Tech Stack](#tech-stack)
4. [Project Structure](#project-structure)
5. [Core Features](#core-features)
6. [Biomechanics Logic](#biomechanics-logic)
7. [Installation](#installation)
8. [Running the App](#running-the-app)
9. [How It Works (Pipeline)](#how-it-works-pipeline)
10. [File-by-File Explanation](#file-by-file-explanation)
11. [Angle Reference Table](#angle-reference-table)
12. [Rep Counting FSM](#rep-counting-fsm)
13. [Troubleshooting](#troubleshooting)
14. [Future Improvements](#future-improvements)
15. [License](#license)

---

## Project Overview

This tool automatically analyses barbell squat and deadlift videos to provide:

- **Real-time skeleton overlay** using MediaPipe Pose (33 body landmarks)
- **Biomechanical joint angle calculation** (knee, hip, spine) using coordinate geometry
- **Form feedback** — both warnings (corrections) and positive cues
- **Repetition counting** via a hip-position state machine
- **Annotated output video** with angle labels, rep count, and warning banners
- **Joint angle time-series graphs** using Matplotlib
- **Form quality score** (0–100 %) based on rule checks

The project requires **no deep learning training**, no GPUs, and no paid APIs. It runs entirely on a standard laptop using open-source libraries.

---

## Live Demo Screenshots

> *(Add your own screenshots after running the app.)*

| Tab | Description |
|-----|-------------|
| `📤 Upload & Analyse` | File uploader, video preview, Analyse button |
| `📊 Results` | Metrics, form score, feedback cards, angle graph |
| `ℹ️ About` | Architecture, thresholds, future work |

**Example metric cards (after analysis):**

```
🔄 Total Reps: 5     📐 Form Score: 78%     ⚠️ Issues: 2     🎞️ Frames: 312
```

**Example warning output:**
```
⚠️ Go deeper — squat above parallel (125°)
⚠️ Knee valgus detected (left knee) — push knees out over toes
✅ Neutral spine maintained (22° from vertical)
✅ Good knee-ankle alignment — no valgus
```

---

## Tech Stack

| Purpose | Library | Version |
|---------|---------|---------|
| Dashboard UI | Streamlit | ≥ 1.28 |
| Pose detection | MediaPipe | ≥ 0.10 |
| Video I/O & drawing | OpenCV | ≥ 4.8 |
| Maths / arrays | NumPy | ≥ 1.24 |
| Graphs | Matplotlib | ≥ 3.7 |
| Image utilities | Pillow | ≥ 10.0 |

**No TensorFlow. No PyTorch. No GPU required.**

---

## Project Structure

```
Olympic-Lift-Analyzer/
│
├── app.py                  # Streamlit dashboard + processing pipeline
├── pose_detector.py        # MediaPipe Pose wrapper
├── angle_calculator.py     # Joint angle geometry (dot product formula)
├── feedback.py             # Biomechanics rule engine (squat + deadlift)
├── rep_counter.py          # FSM-based repetition counter
├── utils.py                # Drawing helpers, graph generator, video saver
│
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
├── sample_videos/          # Place test videos here
└── outputs/                # Processed annotated videos saved here
```

---

## Core Features

### 1. Video Upload
- Supports MP4, AVI, MOV, MKV formats
- Preview uploaded video directly in Streamlit
- Displays file name and size

### 2. Pose Detection (MediaPipe)
MediaPipe Pose detects **33 body landmarks** per frame. This project uses:

| Landmark | Index |
|----------|-------|
| Left / Right Shoulder | 11, 12 |
| Left / Right Elbow | 13, 14 |
| Left / Right Wrist | 15, 16 |
| Left / Right Hip | 23, 24 |
| Left / Right Knee | 25, 26 |
| Left / Right Ankle | 27, 28 |

### 3. Joint Angle Calculation
Two geometry formulas:

**Three-point angle** (for knee, hip, elbow):
```
cos(θ) = (BA · BC) / (|BA| × |BC|)
θ = arccos(cos(θ)) × (180/π)
```

**Vertical inclination angle** (for spine):
```
cos(θ) = (spine_vector · [0, -1]) / |spine_vector|
```

### 4. Biomechanics Feedback
Rule-based engine with clear threshold constants. No black-box model.

### 5. Rep Counting
5-frame smoothed hip-y FSM. Works for both squat and deadlift.

### 6. Annotated Video Output
- Skeleton overlay (MediaPipe default style)
- Angle labels at joints (colour-coded)
- Semi-transparent info panel (top-left)
- Warning banners at bottom of frame
- Downloadable as MP4

### 7. Angle Time-Series Graph
3-panel dark-themed Matplotlib chart (knee, hip, spine vs. frame number).

---

## Biomechanics Logic

### Squat Rules

| Check | Logic | Thresholds |
|-------|-------|------------|
| **Squat depth** | Knee angle | ≤90° excellent, ≤100° good, >120° warn |
| **Knee valgus** | Knee-x vs Ankle-x offset | >20 px inward = cave |
| **Forward lean** | Hip angle (shoulder-hip-knee) | <40° = excessive, <50° = moderate |
| **Rounded back** | Spine vertical angle | >45° = warning, >30° = note |

### Deadlift Rules

| Check | Logic | Thresholds |
|-------|-------|------------|
| **Spine neutrality** | Spine vertical angle | ≤35° neutral, 35-50° slight, >50° warn |
| **Hip shooting** | Hip angle − Knee angle | Delta >35° = hips rising too fast |
| **Lockout** | Both knee & hip angles | ≥160° = full lockout |
| **Bar path** | Wrist-x vs Ankle-x drift | >60 px = bar drifting away from body |

---

## Installation

### Prerequisites
- Python 3.9 or newer
- pip

### Step 1 — Clone or download the project

```bash
git clone https://github.com/yourusername/Olympic-Lift-Analyzer.git
cd Olympic-Lift-Analyzer
```

Or simply unzip the downloaded folder.

### Step 2 — Create a virtual environment (recommended)

```bash
# Create
python -m venv venv

# Activate — Windows
venv\Scripts\activate

# Activate — macOS / Linux
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **Note:** MediaPipe may take a minute to download its pose model (~25 MB) on first run.

---

## Running the App

```bash
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**

**Steps:**
1. Select exercise (**Squat** or **Deadlift**) in the sidebar
2. Click **📤 Upload & Analyse** tab
3. Drag & drop a video file
4. Click **🔍 Analyse Video**
5. Wait for processing (progress bar shown)
6. View annotated video + download it
7. Switch to **📊 Results** for full biomechanics report

---

## How It Works (Pipeline)

```
┌─────────────────┐
│  Uploaded Video  │
└────────┬────────┘
         │ OpenCV VideoCapture
         ▼
┌─────────────────────────────────┐
│  Frame Loop (frame by frame)    │
│                                 │
│  1. resize_frame()              │ ← utils.py
│  2. detect_pose()               │ ← pose_detector.py (MediaPipe)
│  3. extract_landmarks()         │ ← pose_detector.py
│  4. get_key_points()            │ ← pose_detector.py
│  5. calculate_all_angles()      │ ← angle_calculator.py
│  6. analyze_squat/deadlift()    │ ← feedback.py
│  7. rep_counter.update()        │ ← rep_counter.py
│  8. draw_landmarks()            │ ← pose_detector.py
│  9. draw_angle_label()          │ ← utils.py
│ 10. draw_info_overlay()         │ ← utils.py
│ 11. draw_warnings_on_frame()    │ ← utils.py
└────────┬────────────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Processed Frame List     │
└────────┬─────────────────┘
         │
         ├──► save_processed_video()  → outputs/processed_output.mp4
         ├──► Streamlit st.video()    → inline preview
         ├──► display_results()       → metrics, feedback cards
         └──► plot_angle_history()    → Matplotlib time-series chart
```

---

## File-by-File Explanation

### `pose_detector.py`
Thin wrapper around `mp.solutions.pose.Pose`.
- `detect_pose(frame)` — converts BGR→RGB and calls `pose.process()`
- `extract_landmarks(results, shape)` — converts normalised coords to pixel coords
- `get_key_points(landmarks)` — filters to the 13 named joints we care about
- `draw_landmarks(frame, results)` — draws the default MediaPipe skeleton
- `release()` — closes the model cleanly

### `angle_calculator.py`
Pure maths, no dependencies on any other module.
- `three_point_angle(A, B, C)` — dot-product formula, returns angle at B
- `vertical_angle(top, bottom)` — angle between a segment and vertical axis
- `midpoint(p1, p2)` — average of two landmark dicts
- `calculate_all_angles(key_points)` — calls the above for all joints, returns dict

### `feedback.py`
Rule engine with all thresholds as named constants at the top.
- `analyze_squat(angles, key_points)` — 4 checks, returns (warnings, good_form)
- `analyze_deadlift(angles, key_points)` — 4 checks, returns (warnings, good_form)
- `get_form_score(warnings, good_form)` — `good / total × 100`

### `rep_counter.py`
`RepCounter` class with a 4-state FSM.
- State: STANDING → GOING_DOWN → GOING_UP → STANDING
- Triggered by smoothed hip y-coordinate crossing thresholds
- `update(key_points)` → `(rep_count, state)`
- `reset()` — clears all state for a new video

### `utils.py`
Drawing and I/O helpers.
- `resize_frame()` — keeps processing fast on large videos
- `draw_angle_label()` — puts angle text near a joint with shadow
- `draw_info_overlay()` — semi-transparent top-left info panel
- `draw_warnings_on_frame()` — translucent red warning banners at bottom
- `plot_angle_history()` — dark-themed 3-panel Matplotlib figure
- `save_processed_video()` — OpenCV VideoWriter wrapper

### `app.py`
Streamlit entry point.
- `process_video()` — main frame loop tying all modules together
- `display_results()` — renders metrics, feedback cards, and graph in Tab 2
- `render_sidebar()` — exercise selector, tips, tech stack credits
- `render_about_tab()` — full project documentation inside the app
- `main()` — page config, header, tabs, session state management

---

## Angle Reference Table

```
STANDING SQUAT (side view):
  Knee angle  ≈ 170°   Hip angle  ≈ 175°   Spine  ≈  5°

PARALLEL SQUAT:
  Knee angle  ≈  90°   Hip angle  ≈  65°   Spine  ≈ 35°

BELOW PARALLEL:
  Knee angle  ≈  80°   Hip angle  ≈  55°   Spine  ≈ 40°

DEADLIFT SETUP:
  Knee angle  ≈  90°   Hip angle  ≈  75°   Spine  ≈ 30°

DEADLIFT LOCKOUT:
  Knee angle  ≈ 175°   Hip angle  ≈ 175°   Spine  ≈  8°
```

---

## Rep Counting FSM

```
         ┌─────────────────────────────────────────┐
         │                                         │
         ▼                                         │
    ┌──────────┐   hip drops > threshold    ┌──────────────┐
    │ STANDING │ ──────────────────────────► │ GOING_DOWN   │
    └──────────┘                             └──────┬───────┘
         ▲                                          │
         │                                          │ hip reverses (rises)
         │                                          ▼
         │  hip returns near baseline        ┌──────────────┐
         └───────────────────────────────── │  GOING_UP    │
              rep_count += 1                └──────────────┘

5-frame rolling average smooths out MediaPipe jitter.
Baseline y is updated dynamically after each rep to handle camera drift.
```

---

## Troubleshooting

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| `ModuleNotFoundError: mediapipe` | Not installed | `pip install mediapipe` |
| `ModuleNotFoundError: cv2` | Not installed | `pip install opencv-python` |
| No landmarks detected | Poor lighting / person not fully in frame | Use a well-lit side-view video |
| Rep count = 0 | Hips barely move / video too short | Check warmup (first 10 frames) and that full body is visible |
| Processed video not saving | No write permission to `outputs/` | Run `mkdir outputs` manually |
| App crashes on upload | Video codec unsupported | Convert to H.264 MP4 with VLC or FFmpeg: `ffmpeg -i input.avi output.mp4` |
| Very slow processing | High resolution video | Resize to 480p before uploading, or reduce `max_width` in `utils.py` |
| Mediapipe pose model download fails | No internet on first run | Manually download mediapipe models or use a machine with internet |

---

## Future Improvements

- **3-D pose angles** — MediaPipe 3D landmarks for true out-of-plane analysis
- **Live webcam mode** — `st.camera_input` or WebRTC integration
- **Barbell tracking** — YOLOv8 nano for independent bar-path detection
- **Session database** — SQLite log of reps, dates, and form scores
- **LLM coaching cues** — Pass angle data to Claude/GPT for natural-language advice
- **Multi-camera fusion** — Combine front + side view for full 3-D knee valgus
- **Mobile app** — React Native wrapper calling this as a FastAPI backend
- **Tempo analysis** — Measure eccentric/concentric phase duration
- **Force estimation** — Combine with GRF data from a force plate
- **Athlete profiles** — Store body dimensions for anthropometry-adjusted thresholds

---

## License

MIT License — free to use, modify, and distribute with attribution.

---

*Built for educational purposes as a Mechanical Engineering + AI capstone project.*
*Ideal for college portfolios, internship demos, and biomechanics coursework.*
