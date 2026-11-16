from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *


cap = cv2.VideoCapture("Videos/C1.mp4") #for video


model = YOLO('../YOLO-Weights/yolov8n.pt')

classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

mask = cv2.imread("Project 1 - Car Counter/mask.png")

#tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", 960, 540)

while True:
  success, img = cap.read()

  if not success:
    break

  img = cv2.resize(img, (960, 540))
  imgRegion = cv2.bitwise_and(img, mask)

  results = model(imgRegion, stream = True)

  detections = np.empty((0,5))

  for r in results: #extracting coordinates
    boxes = r.boxes
    for box in boxes:
      x1,y1,x2,y2 = box.xyxy[0]
      x1,y1,x2,y2 = int(x1), int(y1) ,int(x2), int(y2)
      #cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255), 3)
      w,h = x2-x1, y2-y1

      conf = math.ceil((box.conf[0]*100))/100

      cls = int(box.cls[0])

      currentClass = classNames[cls]
      if (currentClass == "car" or currentClass == "bus" or currentClass == "truck" or currentClass == "motorbike") and conf>0.3:
        #cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0,x1), max(35,y1)), offset=3)
        cvzone.cornerRect(img,(x1,y1,w,h), l=12, rt=5)
        currentArray = np.array([x1, y1, x2, y2, conf])
        detections = np.vstack((detections, currentArray))
  
  resultsTracker = tracker.update(detections)

  for result in resultsTracker:
    x1,y1,x2,y2,id = result
    print(result)
    x1,y1,x2,y2 = int(x1), int(y1) ,int(x2), int(y2)
    w,h = x2-x1, y2-y1
    cvzone.cornerRect(img,(x1,y1,w,h), l=12, rt=2, colorR=(255,0,0))
    cvzone.putTextRect(img, f' {int(id)}', (max(0,x1), max(35,y1)), offset=3)


  cv2.imshow("Image", img)
  cv2.imshow("ImageRegion", imgRegion)
  cv2.waitKey(1)