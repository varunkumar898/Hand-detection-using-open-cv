# Environment Setup Guide

This guide walks you through setting up an isolated Python 3.10 environment using Conda, installing OpenCV via `conda-forge`, and installing MediaPipe via `pip`.

---

## Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) installed on your system.
- A working webcam.

---

## Step-by-Step Installation

### 1. Create a Conda Environment

Create a clean, isolated environment named `cv` with Python 3.10:

```bash
conda create -n cv python=3.10 -y
```

### 2. Activate the Environment

Activate your newly created environment:

```bash
conda activate cv
```

> **Note:** Ensure that your terminal prompt shows `(cv)` before proceeding with the package installations.

### 3. Install OpenCV via conda-forge

Install OpenCV and its core video and GUI dependencies (such as FFmpeg, NumPy, and Qt) from the `conda-forge` channel:

```bash
conda install -c conda-forge opencv -y
```

### 4. Install MediaPipe via pip

MediaPipe is **not available on conda-forge for Windows** (attempting `conda install -c conda-forge mediapipe` results in a `PackagesNotFoundError`).

To install MediaPipe, use `pip` inside your active Conda environment:

```bash
pip install mediapipe
```

If you encounter package resolution or index issues on your network, use the official PyPI index URL:

```bash
pip install mediapipe --extra-index-url https://pypi.org/simple
```

---

## Verification

Confirm that both OpenCV and MediaPipe are installed properly and can be imported:

```bash
python -c "import cv2, mediapipe as mp; print('OpenCV version:', cv2.__version__); print('MediaPipe version:', mp.__version__)"
```

You should see the installed version numbers printed without errors.

---

## Alternative: pip & requirements.txt

If you prefer using standard Python virtual environments or direct `pip` installation:

```bash
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Running the Application

Once setup is complete, start the hand detection script:

```bash
python hand_detection.py
```

Press **`q`** in the application window to stop the stream and exit.
