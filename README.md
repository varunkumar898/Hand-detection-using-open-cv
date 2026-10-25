# Real-Time Hand Detection using OpenCV & MediaPipe

A lightweight real-time computer vision application that captures webcam video to detect human hands, track 21 3D landmarks, and render dynamic bounding boxes with live on-screen metrics.

---


## Features

- **Multi-Hand Detection**: Concurrently detects and tracks up to 2 hands in the camera feed using MediaPipe Hands.
- **21 3D Landmark Tracking**: Locates key skeletal joint coordinates (wrist, knuckles, fingertips) and visualizes connections.
- **Dynamic Padded Bounding Boxes**: Computes tight bounding rectangles around detected hands with 20px padding.
- **Mirrored Display**: Flips webcam input horizontally for an intuitive selfie-view experience.
- **Live Smoothed FPS Counter**: Measures frame rates via `time.perf_counter` with exponential smoothing to prevent visual flicker.
- **On-Screen Hand Count**: Displays the real-time count of currently detected hands.
- **Camera Validation**: Verifies webcam availability and displays clear error feedback if the camera cannot be opened.

---

## Tech Stack

- **Python 3.10**
- **OpenCV** (`opencv-python`): Video capture, image transformations, drawing overlays, and GUI window handling.
- **MediaPipe** (`mediapipe`): Machine learning pipeline for real-time hand and 21-landmark detection.
- **Conda**: Isolated Python environment management.

---

## Setup and Usage

For full prerequisites and environment troubleshooting, see the [SETUP.md](SETUP.md) guide.

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/varunkumar898/Hand-detection-using-open-cv.git
   cd Hand-detection-using-open-cv
   ```

2. **Create and activate the Conda environment:**
   ```bash
   conda create -n cv python=3.10 -y
   conda activate cv
   ```

3. **Install dependencies:**
   ```bash
   conda install -c conda-forge opencv -y
   pip install mediapipe
   ```
   *(Or install via pip: `pip install -r requirements.txt`)*

4. **Run the script:**
   ```bash
   python hand_detection.py
   ```

5. **Controls:**
   - Press **`q`** while focused on the video window to quit.

---

## How It Works

The processing pipeline runs frame-by-frame through five stages:

1. **Webcam Capture**: Frames are grabbed sequentially from the primary camera using `cv2.VideoCapture(0)`.
2. **Mirror & Color Space Conversion**: The raw BGR frame is flipped horizontally (`cv2.flip(frame, 1)`) and converted to RGB (`cv2.COLOR_BGR2RGB`), matching MediaPipe's required input format.
3. **MediaPipe Hands Inference**: The RGB frame is processed by MediaPipe Hands (`min_detection_confidence=0.6`, `min_tracking_confidence=0.6`), outputting normalized $(x, y, z)$ coordinates for 21 landmarks per detected hand (up to 2 hands).
4. **Bounding Box from 21 Landmarks**: Landmark coordinates are mapped to pixel coordinates $(x \cdot \text{width}, y \cdot \text{height})$. The minimum and maximum coordinates determine the hand extent, padded by 20 pixels and clamped to frame boundaries.
5. **HUD & Display**: Skeletal connections and landmarks are drawn using MediaPipe utilities. A smoothed FPS reading and detected hand count are overlaid on the frame before presenting it via `cv2.imshow`.

---

## Possible Next Steps

The following are potential enhancements for future iterations of this project (not yet implemented):

- **Gesture Recognition**: Classify static hand gestures (e.g., thumbs-up, peace sign, open palm, fist).
- **Finger Counting**: Determine raised fingers by evaluating landmark tip positions relative to PIP joints.
- **Edge Deployment**: Optimize the inference loop for lightweight single-board computers (such as Raspberry Pi).
