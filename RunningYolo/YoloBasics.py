from ultralytics import YOLO
import cv2

model = YOLO('../YOLO-Weights/yolov8l.pt')
results = model("RunningYolo/images/y2.jpeg", show=True)
cv2.waitKey(0)