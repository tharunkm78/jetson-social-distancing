# Eagle Vision: Social Distancing & Anomaly Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jetson Orin Nano](https://img.shields.io/badge/Hardware-Jetson%20Orin%20Nano-green.svg)](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-orange.svg)]()

**Eagle Vision** is a professional-grade, real-time social distancing and anomaly detection system. Inspired by the "Animus" tactical interface, it provides high-fidelity human pose extraction and automated violation logging. Optimized for the **NVIDIA Jetson Orin Nano**, it also features a seamless fallback mode for **Windows/CPU** environments using YOLOv8.

![Eagle Vision UI](./screenshots/ui.png)

## 🚀 Key Features

- **Dual-Engine Support**: Automatically switches between `jetson-inference` (TensorRT accelerated) and `YOLOv8-pose` (Windows/CPU).
- **Animus-Themed Dashboard**: A high-end, responsive web interface with cinematic "Eagle Eye" shutter effects and "Desync" flicker animations.
- **Social Distancing Logic**: Real-time calculation of inter-person distances with visual alerts and automated CSV logging.
- **Sequential Recording**: Automatically saves test results into a `results/` folder as `output1.mp4`, `output2.mp4`, etc., ensuring no data is overwritten.
- **Fail-Safe Cleanup**: Graceful terminal interrupt handling to ensure video files are always finalized and playable.

## 🛠 System Architecture

```mermaid
graph TD
    A[Camera / Video Source] --> B{Hardware Detect}
    B -- Jetson --> C[jetson-inference / ResNet-18]
    B -- Windows --> D[Ultralytics YOLOv8-Pose]
    C --> E[Pose Extraction]
    D --> E
    E --> F[Social Distancing Engine]
    F --> G[Rule Validator]
    G --> H[Web Dashboard / MJPEG Stream]
    G --> I[CSV Event Logger]
    G --> J[Sequential Video Recorder]
```

## 📸 Prediction Showcase

![Prediction Preview](./screenshots/prediction.png)

## 📥 Installation

### On Jetson (Docker)
Run inside the `jetson-inference` container for maximum performance:
```bash
sudo docker run -it --rm --runtime nvidia --network host \
  --volume ~/path/to/project:/workspace \
  dustynv/jetson-inference:r35.4.1
cd /workspace
python3 posenet_socialdistancing.py --input /dev/video0
```

### On Windows
```bash
pip install flask ultralytics opencv-python numpy
python posenet_socialdistancing.py --input 0
```

## 🎮 Usage

| Argument | Description | Default |
|----------|-------------|---------|
| `--input` | Video source (camera ID or file path) | `/dev/video0` |
| `--network` | PoseNet model to use | `resnet18-body` |
| `--distance` | Violation threshold in pixels | `150` |

### Terminal Shortcuts
- `Ctrl + C`: Safe shutdown (closes video writer and saves results).

## 📂 Project Structure

```text
.
├── engine/                 # Core logic and hardware abstraction
│   ├── pose_engine.py      # Dual-engine pose extraction
│   ├── social_logic.py     # Distancing & Violation math
│   └── logger.py           # CSV Data logging
├── results/                # Saved sequential video outputs
├── screenshots/            # UI and Preview images
├── static/                 # CSS/JS and UI assets
├── templates/              # HTML Dashboard
└── posenet_socialdistancing.py  # Main Entry Point
```

---
*Developed for advanced human monitoring and brotherhood protocols.*
