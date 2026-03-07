import cv2
import cvzone
import math
import numpy as np
import os
from datetime import datetime
from ultralytics import YOLO
from sort import Sort

video_path = "C:/Users/Abduvaliy.DESKTOP-PR9DVS1/Documents/video.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise SystemExit(f"Failed to open video: {video_path}")
#use yolov8n model with 90 confidence threshold and 0.5 iou threshold

model = YOLO("yolov8n.pt")

with open("classes.txt", "r") as f:
    classnames = f.read().splitlines()

allowed_classes = {"car", "truck", "bike", "bicycle", "motorbike"}
track_start_frame = {}
track_class_name = {}
car_track_ids = set()
truck_track_ids = set()
frame_index = 0

# Prepare output saving
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_video_path = os.path.join(output_dir, f"annotated_{timestamp}.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
try:
    fps = float(fps)
except Exception:
    fps = 30.0
if not fps or math.isnan(fps) or fps <= 0:
    fps = 30.0

# Keep a track alive for short detector dropouts so timer does not reset.
tracker = Sort(max_age=int(fps * 2), min_hits=1, iou_threshold=0.2)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_video_path, fourcc, fps, (1920, 1080))


def iou(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    ix1 = max(xa1, xb1)
    iy1 = max(ya1, yb1)
    ix2 = min(xa2, xb2)
    iy2 = min(ya2, yb2)
    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter = inter_w * inter_h
    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    union = area_a + area_b - inter
    return (inter / union) if union > 0 else 0.0


def normalize_vehicle_class(name):
    if name in {"bicycle", "motorbike"}:
        return "bike"
    if name in {"bus", "train"}:
        return "truck"
    return name


while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    frame_index += 1
    frame = cv2.resize(frame, (1920, 1080))

    results = model(frame, verbose=False, conf=0.45)
    current_detections = np.empty((0, 5), dtype=np.float32)
    detection_records = []

    for info in results:
        for box in info.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            score = float(box.conf[0])          # 0.0 - 1.0
            cls_id = int(box.cls[0])
            cls_name = normalize_vehicle_class(classnames[cls_id])
            min_score = 0.30 if cls_name == "truck" else 0.45
            if (cls_name in allowed_classes) and (score > min_score):
                det = np.array([x1, y1, x2, y2, score], dtype=np.float32)
                current_detections = np.vstack((current_detections, det))
                detection_records.append({
                    "bbox": (x1, y1, x2, y2),
                    "score": score,
                    "class_name": cls_name,
                })

    track_results = tracker.update(current_detections)
    for x1, y1, x2, y2, tid in track_results:
        x1, y1, x2, y2, tid = int(x1), int(y1), int(x2), int(y2), int(tid)
        if tid not in track_start_frame:
            track_start_frame[tid] = frame_index

        best_class = track_class_name.get(tid, "tracker")
        best_score = 0.0
        track_box = (x1, y1, x2, y2)
        for det in detection_records:
            overlap = iou(track_box, det["bbox"])
            if overlap > best_score:
                best_score = overlap
                best_class = det["class_name"]
        track_class_name[tid] = best_class
        if best_class == "car":
            car_track_ids.add(tid)
        if best_class == "truck":
            truck_track_ids.add(tid)

        stay_seconds = (frame_index - track_start_frame[tid]) / fps
        label = f"{best_class} tracker:{tid} time:{stay_seconds:.1f}s"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cvzone.putTextRect(frame, label, (x1 + 6, max(35, y1 - 10)), thickness=2, scale=1)

    cvzone.putTextRect(
        frame,
        f"Cars: {len(car_track_ids)}",
        (20, 45),
        thickness=3,
        scale=2,
        border=2,
    )
    cvzone.putTextRect(
        frame,
        f"Trucks: {len(truck_track_ids)}",
        (20, 95),
        thickness=3,
        scale=2,
        border=2,
    )

    cv2.imshow("frame", frame)
    # write annotated frame to output video
    if writer is not None and writer.isOpened():
        writer.write(frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord("q"):
        break

cap.release()
if writer is not None:
    writer.release()
    print(f"Saved annotated video to: {output_video_path}")
cv2.destroyAllWindows()
