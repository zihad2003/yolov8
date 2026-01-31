from ultralytics import YOLO
import cv2
import time
import threading
try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False
    print("Warning: pygame not found. Alarm audio will be disabled.")

import torch
import os

# ------------------ CONFIGURATION ------------------
# SOURCE: 
# - Use 0 for default laptop/USB webcam
# - Use a string URL for IP Webcam (e.g., "http://192.168.0.101:8080/video")
CAMERA_SOURCE = "http://10.15.0.133:8080/video" # <--- Fixed with quotes and /video path

# CRITICAL: You must change this to a model trained on fire!
# Download one from Roboflow Universe or train your own.
# "yolov8n.pt" WILL NOT WORK as it is not trained to see fire.
MODEL_PATH = "fire_v11.pt" 
FALLBACK_MODEL_PATH = "yolo11n.pt" # Upgraded to YOLO11 (Latest)

if not os.path.exists(MODEL_PATH):
    if os.path.exists(FALLBACK_MODEL_PATH):
        print(f"Warning: '{MODEL_PATH}' not found. Falling back to '{FALLBACK_MODEL_PATH}' for demonstration.")
        MODEL_PATH = FALLBACK_MODEL_PATH
    else:
        print(f"FATAL: Neither '{MODEL_PATH}' nor '{FALLBACK_MODEL_PATH}' were found.")
        exit()

# Make sure this file is in the same folder as the script
ALARM_SOUND_PATH = "fire.mp3"

# Detection settings
CONF_THRESHOLD = 0.35  # Slightly lower for faster detection
DETECT_EVERY_N_FRAMES = 1  
RESIZE_DIM = (240, 180)  # Lowered resolution for faster processing
FIRE_CLASS_NAMES = ["fire", "flame", "person", "cell phone"]
IMG_SIZE = 160 # YOLO11 can run at 160 for extreme speed on CPU

# ------------------ DEVICE SETUP ------------------
# Auto-detect GPU (NVIDIA CUDA) or fall back to CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ------------------ TRACKING HISTORY (Disabled for Speed) ------------------
track_history = {} 

# ------------------ AUDIO SYSTEM SETUP ------------------
if HAS_PYGAME:
    try:
        pygame.mixer.init()
        if os.path.exists(ALARM_SOUND_PATH):
            pygame.mixer.music.load(ALARM_SOUND_PATH)
            print(f"Audio loaded: {ALARM_SOUND_PATH}")
        else:
            print(f"Warning: Alarm audio file '{ALARM_SOUND_PATH}' not found.")
            HAS_PYGAME = False
    except Exception as e:
        print(f"Error loading audio: {e}")
        HAS_PYGAME = False
        print("WARNING: Alarm audio will not play.")

def trigger_alarm():
    """Plays the alarm sound if it's not already playing."""
    if HAS_PYGAME and not pygame.mixer.music.get_busy():
        print("...ALARM SOUNDING...")
        pygame.mixer.music.play()

# ------------------ CAMERA STREAM THREAD ------------------
class CameraStream:
    """
    Manages reading frames from the webcam in a separate thread
    to prevent blocking the main detection loop.
    """
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret, self.frame = ret, frame
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.ret else (False, None)

    def release(self):
        self.running = False
        self.cap.release()

# ------------------ LOAD YOLO MODEL ------------------
try:
    model = YOLO(MODEL_PATH)
    print(f"Successfully loaded model: {MODEL_PATH}")
    # This is a key check: See what your model can actually detect!
    print(f"Model classes: {model.names}")
except Exception as e:
    print(f"FATAL: Failed to load model '{MODEL_PATH}'.")
    print(f"Error: {e}")
    exit()

# ------------------ START WEBCAM ------------------
cam = CameraStream(CAMERA_SOURCE)
time.sleep(1.0)  # Give camera time to initialize
prev_time = 0
frame_count = 0
last_boxes = []  # To keep boxes between skips

print("🔥 Fire detection system running... Press Q to quit.")

# ------------------ MAIN LOOP ------------------
try:
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame, check webcam.")
            time.sleep(0.5)
            continue

        frame_count += 1
        
        # Get original frame dimensions
        frame_height, frame_width = frame.shape[:2]
        x_scale = frame_width / RESIZE_DIM[0]
        y_scale = frame_height / RESIZE_DIM[1]

        # YOLO TRACKING (Optimized)
        results = model.track(
            frame,
            persist=True,
            device=DEVICE,
            conf=CONF_THRESHOLD,
            verbose=False,
            imgsz=IMG_SIZE, # High-performance setting
            tracker="botsort.yaml"
        )

        is_fire_present_in_frame = False

        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            class_ids = results[0].boxes.cls.int().cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

            for box, track_id, cls, conf in zip(boxes, track_ids, class_ids, confs):
                label_name = model.names[int(cls)].lower()
                
                if label_name in FIRE_CLASS_NAMES:
                    if label_name in ["fire", "flame"]:
                        is_fire_present_in_frame = True
                    
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Optimized Drawing
                    color = (0, 255, 0) if label_name == "person" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1) # Thinner lines
                    cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # ------------ ALARM TRIGGER LOGIC ------------
        if is_fire_present_in_frame:
            # Start alarm in a new thread to avoid blocking
            threading.Thread(target=trigger_alarm, daemon=True).start()

        # FPS Display
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show window
        cv2.imshow("Fire Detection System", frame)

        # Exit on Q
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("Stopped manually.")

finally:
    cam.release()
    cv2.destroyAllWindows()
    if HAS_PYGAME:
        try:
            pygame.mixer.music.stop()
            pygame.mixer.quit()
        except:
            pass
    print("System closed.")