from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import numpy as np

cap = cv2.VideoCapture("../Videos/people.mp4")


model = YOLO("../Yolo-Weights/yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
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

mask = cv2.imread("mask2.png")

tracker = Sort(max_age=20 , min_hits=3 ,iou_threshold=0.3)

limitsUp = [103,161,296,161]
limitsDown = [527,489,735,489]

totalCountUp = set()
totalCountDown = set()

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img,mask)
    results = model(imgRegion, stream=True)

    detections = np.empty((0,5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = math.ceil((box.conf[0] * 100)) / 100
            currentClass = classNames[cls]
            if currentClass == "person" and conf > 0.3 :
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections,currentArray))

    resultsTracker = tracker.update(detections)
    cv2.line(img,(limitsUp[0],limitsUp[1]),(limitsUp[2],limitsUp[3]),(0,0,255),5)
    cv2.line(img,(limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (255, 0, 255), 5)

    for result in resultsTracker:
        x1,y1,x2,y2,Id = result
        print(result)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=10, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(img, f'{int(Id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3,
                           offset=10)

        cx,cy = x1+w//2 , y1+h//2

        if limitsUp[0] <cx < limitsUp[2] and abs(cy - limitsUp[1]) < 20 :
            totalCountUp.add(Id)
        elif limitsDown[0] <cx < limitsDown[2] and abs(cy - limitsDown[1]) < 20 :
            totalCountDown.add(Id)

    cvzone.putTextRect(img, f'Up-{len(totalCountUp)}', (50,50))

    cvzone.putTextRect(img, f'Down-{len(totalCountDown)}', (50, 100))
    cv2.imshow("Image", img)
    cv2.waitKey(1)
