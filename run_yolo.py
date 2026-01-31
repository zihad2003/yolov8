from ultralytics import YOLO
import cv2
import time
import threading
import argparse
import os
import sys

# Try to import pygame for audio
try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False

import torch

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Fire Detection System")
    parser.add_argument("--source", type=str, default="0", help="Camera source (0 for webcam or URL for IP cam)")
    parser.add_argument("--model", type=str, default="fire_v11.pt", help="Path to YOLO model file")
    parser.add_argument("--alarm", type=str, default="fire.mp3", help="Path to alarm sound file")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=160, help="Inference image size")
    return parser.parse_args()

def setup_audio(alarm_path):
    if not HAS_PYGAME:
        print(" [!] Warning: pygame not found. Alarm audio disabled.")
        return False
    
    try:
        pygame.mixer.init()
        if os.path.exists(alarm_path):
            pygame.mixer.music.load(alarm_path)
            print(f" [+] Audio loaded: {alarm_path}")
            return True
        else:
            print(f" [!] Warning: Alarm file '{alarm_path}' not found.")
            return False
    except Exception as e:
        print(f" [!] Error loading audio: {e}")
        return False

def trigger_alarm():
    if HAS_PYGAME and not pygame.mixer.music.get_busy():
        try:
            pygame.mixer.music.play()
        except:
            pass

class CameraStream:
    def __init__(self, src):
        # Handle string '0' from argparse
        if src.isdigit():
            src = int(src)
        
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            print(f" [!] FATAL: Could not open video source: {src}")
            sys.exit(1)
            
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

def main():
    args = parse_args()
    
    print("\n" + "="*40)
    print(" 🔥 YOLO FIRE DETECTION SYSTEM 🔥")
    print("="*40)
    
    # Model Selection
    model_path = args.model
    if not os.path.exists(model_path):
        fallback = "yolo11n.pt"
        if os.path.exists(fallback):
            print(f" [!] '{model_path}' not found. Using fallback: {fallback}")
            model_path = fallback
        else:
            print(f" [!] FATAL: No model found at {model_path} or {fallback}")
            return

    # Load Model
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = YOLO(model_path)
        print(f" [+] Model loaded: {model_path} ({device})")
    except Exception as e:
        print(f" [!] Error loading model: {e}")
        return

    # Audio Setup
    audio_enabled = setup_audio(args.alarm)
    
    # Camera Setup
    cam = CameraStream(args.source)
    print(f" [+] Camera initialized: {args.source}")
    print(" [i] Press 'Q' to quit.\n")

    prev_time = time.time()
    fire_classes = ["fire", "flame"]

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                continue

            # Inference
            results = model.track(
                frame,
                persist=True,
                device=device,
                conf=args.conf,
                verbose=False,
                imgsz=args.imgsz,
                tracker="botsort.yaml"
            )

            is_fire = False
            
            if results and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                class_ids = results[0].boxes.cls.int().cpu().numpy()
                
                for box, cls in zip(boxes, class_ids):
                    label = model.names[int(cls)].lower()
                    
                    if label in fire_classes:
                        is_fire = True
                    
                    # Draw
                    x1, y1, x2, y2 = map(int, box)
                    color = (0, 0, 255) if label in fire_classes else (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if is_fire and audio_enabled:
                threading.Thread(target=trigger_alarm, daemon=True).start()

            # FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Detection Stream", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        print("\n [i] Shutting down...")
        cam.release()
        cv2.destroyAllWindows()
        if HAS_PYGAME:
            pygame.mixer.quit()
        print(" [✓] System closed.")

if __name__ == "__main__":
    main()
