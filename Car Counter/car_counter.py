from ultralytics import YOLO
import cv2
import tracker
import numpy as np

cap = cv2.VideoCapture("cars.mp4")
model = YOLO("yolov8l.pt")

colors = {"car":(255, 0, 255), "truck": (255, 0, 0), "bus": (0, 255, 0), "motorcycle": (0, 0, 255)}

mask = cv2.imread("mask1.png")
tracker = tracker.Tracker()

limits = [300, 340, 670, 340]

total_count = {"car": 0, "truck": 0, "bus": 0, "motorcycle": 0}
frame_counter = 0
counted_ids = []
while True:
    detections = np.empty((0, 6))
    frame_counter += 1
    if frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        frame_counter = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    success, img = cap.read()
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 3)
    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream=True)
    for r in results:
        global labels
        labels = r.names
        for box in r.boxes:
            cls = int(box.cls[0])
            if labels[cls] in colors.keys() and box.conf[0].item() >= 0.25:
                xyxy = [int(x) for x in box.xyxy[0]] 
                detections = np.concatenate((detections, np.array([[xyxy[0], xyxy[1], xyxy[2], xyxy[3], -1, cls]])), axis=0)
                cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), colors[labels[cls]], 1)
                cv2.rectangle(img, (xyxy[0] + len(labels[cls]) * 12 + 36, max(0, xyxy[1] - 12)), (xyxy[0], max(12, xyxy[1])), colors[labels[cls]], -1)
                cv2.putText(img, str(round(box.conf[0].item(), 2)), (xyxy[0], max(10, xyxy[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
                cv2.putText(img, labels[cls], (xyxy[0] + 36, max(10, xyxy[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

    ids, last_bboxs = tracker.update(detections[:, :-1])
    for i, bbox in enumerate(detections):
        cx, cy = bbox[0] + (bbox[2] - bbox[0]) // 2, bbox[1] + (bbox[3] - bbox[1]) // 2
        cv2.putText(img, str(int(ids[i])), (int(bbox[0] - 15), int(bbox[1] - 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        if limits[0] < cx < limits[2] and abs(limits[1] - cy) < 40 and ids[i] not in counted_ids:
            total_count[labels[bbox[5]]] += 1
            counted_ids.append(ids[i])

    cv2.putText(img, f"cars: {total_count['car']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.putText(img, f"trucks: {total_count['truck']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.putText(img, f"buses: {total_count['bus']}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.putText(img, f"motorcycles: {total_count['motorcycle']}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)