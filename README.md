# YOLOv8 & YOLO11 Fire Detection System

This repository contains Python scripts for real-time fire and object detection using the Ultralytics YOLO framework. It supports both local webcams and IP camera streams.

## 🚀 Features
- **Real-time Detection**: Uses YOLOv8/YOLO11 for high-speed interference.
- **Fire Alarm System**: Integrated audio alarm (using `pygame`) that triggers when fire or flames are detected.
- **IP Camera Support**: Easily connect to mobile IP Webcam apps or network cameras.
- **Multithreading**: Optimized camera streaming to prevent UI lag.
- **GPU Acceleration**: Automatically uses NVIDIA CUDA if available.

---

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/zihad2003/yolov8.git
cd yolov8
```

### 2. Install Dependencies
Ensure you have Python 3.8+ installed. Install the required packages via pip:

```bash
pip install ultralytics opencv-python pygame torch torchvision torchaudio
```

---

## 📂 Project Structure
- `run_yolo.py`: The main fire detection script with alarm logic and IP camera support.
- `yolo_webcam.py`: A simple script for basic object detection using a local webcam.
- `fire.mp3`: (Required) The alarm sound file triggered during fire detection.
- `yolov8n.pt` / `yolo11n.pt`: Weight files for the detection models.

---

## 🚦 How to Run

### Option 1: Main Fire Detection System
This script is configured for fire detection with an audible alarm.

1. **Configure Camera**: Open `run_yolo.py` and set `CAMERA_SOURCE`.
   - For local webcam: `CAMERA_SOURCE = 0`
   - For IP Webcam: `CAMERA_SOURCE = "http://192.168.x.x:8080/video"`
2. **Add Alarm Sound**: Ensure a file named `fire.mp3` exists in the project folder.
3. **Run**:
   ```bash
   python run_yolo.py
   ```

### Option 2: Simple Object Detection
To quickly test general object detection:
```bash
python yolo_webcam.py
```

---

## ⚙️ Configuration Hints
- **Confidence Threshold**: Adjust `CONF_THRESHOLD` in `run_yolo.py` to make the detection more or less sensitive.
- **Model Check**: The script uses `fire_v11.pt` by default. If not found, it falls back to `yolo11n.pt`. For best results, use a model specifically trained on fire datasets.

## 📄 License
This project is for educational purposes as part of the SAD Lab.