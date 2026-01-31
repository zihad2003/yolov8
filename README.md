# YOLOv8 & YOLO11 Fire Detection System

This repository contains Python scripts for real-time fire and object detection using the Ultralytics YOLO framework. It supports both local webcams and IP camera streams.

## 🚀 Features
- **Real-time Detection**: Uses YOLOv8/YOLO11 for high-speed interference.
- **Fire Alarm System**: Integrated audio alarm (using `pygame`) that triggers when fire or flames are detected.
- **IP Camera Support**: Easily connect to mobile IP Webcam apps or network cameras.
- **Multithreading**: Optimized camera streaming to prevent UI lag.
- **GPU Acceleration**: Automatically uses NVIDIA CUDA if available.

---

## 🛠️ Installation & Setup

### 1. Install Dependencies
Ensure you have Python 3.8+ installed. Use the provided requirements file:
```bash
pip install -r requirements.txt
```

### 2. Specialized Fire Model (Recommended)
For the best performance, download a model specifically trained on fire data:
- [Download best.pt (Fire Model)](https://github.com/godrock44/YOLO-V8-FIRE-DETECTION/raw/main/best.pt)
- Rename it to `fire_v11.pt` and place it in the project root.

---

## 🚦 How to Run

### Main Fire Detection System
The primary script `run_yolo.py` supports several command-line flags:

**Using Local Webcam:**
```bash
python run_yolo.py --source 0
```

**Using IP Camera URL:**
```bash
python run_yolo.py --source "http://192.168.0.101:8080/video"
```

**Custom Settings:**
```bash
python run_yolo.py --model my_model.pt --conf 0.5 --alarm sound.mp3
```

---

## ⚙️ Configuration Flags
| Flag | Default | Description |
|------|---------|-------------|
| `--source` | `0` | Camera index or IP stream URL |
| `--model` | `fire_v11.pt` | YOLO model weights path |
| `--alarm` | `fire.mp3` | Alarm audio file path |
| `--conf` | `0.35` | Confidence threshold (0-1) |
| `--imgsz` | `160` | Inference size (lower = faster) |

---

## 📂 Project Structure
- `run_yolo.py`: Primary production script with multithreading and alarm.
- `yolo_webcam.py`: Simplified test script.
- `requirements.txt`: List of required Python packages.
- `.gitignore`: Git exclusion rules.

## 📄 License
This project is for educational purposes as part of the SAD Lab.