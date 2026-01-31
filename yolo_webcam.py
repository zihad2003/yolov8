from ultralytics import YOLO
import cv2

# CONFIGURATION
# Set CAMERA_SOURCE to 0 for local webcam, or the URL for IP Webcam
# e.g., "http://192.168.0.101:8080/video"
CAMERA_SOURCE = 0 

# Load YOLOv8n model
model = YOLO("yolov8n.pt")

# Run prediction
# Using stream=True for generator processing
results = model.predict(source=CAMERA_SOURCE, device="cpu", show=True, stream=True)

print("Running... Press Q in the window to stop.")

for r in results:
    # Just iterate through results to keep the window alive
    pass

cv2.destroyAllWindows()
