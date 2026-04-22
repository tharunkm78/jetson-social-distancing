# Real-Time Social Distancing Monitoring System (Jetson Orin Nano)

## Overview

This project implements a real-time social distancing and human activity monitoring system using pose estimation.

It is designed to run on NVIDIA Jetson devices and also supports execution on non-Jetson environments (e.g., Windows/Linux).  
The application automatically detects the runtime environment and switches accordingly.

If Jetson libraries are unavailable, the system falls back to a YOLOv8-based pose estimation pipeline.

The system processes live video, detects human poses, computes inter-person distances, identifies violations, classifies basic actions, and streams results through a web interface.

---

## Platform Support

### Jetson (Primary)

- Device: Jetson Orin Nano
- Inference Engine: jetson-inference (poseNet)
- Model: resnet18-body
- Camera: USB (/dev/video0)
- Acceleration: CUDA (GPU)

---

### Fallback Mode (Windows / Non-Jetson)

- Framework: OpenCV + Ultralytics YOLOv8
- Model: yolov8n-pose.pt
- Purpose: Development and testing without Jetson hardware

---

## Approach

### Pose Detection

- Jetson: poseNet (GPU accelerated)
- Fallback: YOLOv8 pose model

### Distance Estimation

- Uses hip midpoint for each person
- Computes pairwise distances
- Applies dynamic scaling based on body proportions

### Violation Logic

- Each person has a dynamic “safe radius”
- Violation occurs when radii overlap
- Temporal smoothing prevents flickering detections

### Action Detection

Basic posture/activity classification:
- standing
- walking
- sitting
- fallen
- arms raised

---

## System Pipeline

Camera → Pose Detection → Tracking → Distance Logic → Action Detection → Logging → Web UI

---

## Features

- Real-time pose estimation
- Dynamic, scale-aware social distancing detection
- Persistent ID tracking across frames
- Action classification per individual
- Live MJPEG video streaming via Flask
- Interactive web dashboard:
  - live video feed
  - skeleton overlays
  - violation highlighting
  - activity predictions
- CSV logging with throttling
- Automatic annotated video recording

---

## Results

### Social Distancing Detection

![Detection](screenshots/detection.png)

### Web Interface

![UI](screenshots/ui.png)

### Logs

![Logs](screenshots/logs.png)

---

## Output

Generated files are stored in:

results/

### Files

- violations_log.csv
- action_log.csv
- output*.mp4

### Example (Violations Log)

Timestamp, Person1_ID, Person2_ID  
2026-04-21 18:32:10, 0, 2

---

## How to Run

### On Jetson

Run inside jetson-inference Docker container:
```
sudo docker run -it --rm \
--runtime nvidia \
--network host \
--volume /home/<username>/Desktop/project:/workspace/project \
dustynv/jetson-inference:r35.4.1
```
```
cd /workspace/project  
python3 posenet_socialdistancing.py
```
Open in browser:  
http://<jetson-ip>:5000

---

### On Windows / Linux (Fallback)

Install dependencies:

pip install -r requirements.txt

Run:

python posenet_socialdistancing.py

Open:  
http://localhost:5000

---

## Arguments

| Argument      | Description                        | Default          |
|--------------|----------------------------------|------------------|
| --input      | Camera or video source           | /dev/video0      |
| --network    | PoseNet model (Jetson only)      | resnet18-body    |

---

## Project Structure
```
project/
│
├── posenet_socialdistancing.py
├── yolov8n-pose.pt
│
├── engine/
│   ├── pose_engine.py
│   ├── social_logic.py
│   ├── logger.py
│
├── templates/
│   └── index.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js
│
├── results/
├── test/
│
└── README.md
```
---

## Key Highlights

- Automatic Jetson vs fallback detection
- GPU-accelerated inference on Jetson
- Pose-based distance estimation (not bounding-box based)
- Perspective-aware scaling using body proportions
- Integrated backend + frontend system
- Real-time visualization and logging
