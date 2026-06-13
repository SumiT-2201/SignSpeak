# SignSpeak 🖐️

SignSpeak is a machine learning-powered application that translates sign language gestures into text and speech in real-time. This repository contains the data collection and landmark extraction pipeline, which serves as the foundation for training our sign language recognition model.

## Features (Phase 1)
- **Real-Time Hand Tracking**: Uses MediaPipe Hands to track 21 landmarks on a user's hand in real-time.
- **Visual Overlay**: Real-time OpenCV visualization with hand skeletons and landmark tracking.
- **One-Key Capture**: Instantly maps keys `0-9` to custom labels, logging 63-dimensional coordinate vectors `(x, y, z)` directly into a CSV file.
- **Robust Environment**: Fully packaged configuration using Python 3.12.

## Label Mapping
The current data collector uses the following hotkey mapping:
* `0` ➡️ `hello`
* `1` ➡️ `yes`
* `2` ➡️ `no`
* `3` ➡️ `thank_you`
* `4` ➡️ `i_love_you`
* `5` ➡️ `please`
* `6` ➡️ `sorry`
* `7` ➡️ `good`
* `8` ➡️ `help`
* `9` ➡️ `stop`

---

## Setup & Running Guide

### 1. Prerequisites
Ensure you have Python 3.12 installed on your machine.

### 2. Setup Virtual Environment
Run the following to initialize the environment and install the required dependencies (`mediapipe`, `opencv-python`, `pandas`, `numpy`):
```powershell
# Create venv (if not already created)
python -m venv venv

# Activate venv
.\venv\Scripts\Activate.ps1

# Install requirements
pip install mediapipe==0.10.14 opencv-python pandas numpy
```

### 3. Run the Data Collector
Run the collection script:
```powershell
python data/collect_data.py
```
- Position your hand in the webcam viewport.
- Press keys `0` to `9` to log data rows.
- Press `q` to quit the collector safely. 
- The recorded dataset is generated and saved at `data/landmarks.csv`.
