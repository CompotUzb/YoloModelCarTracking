import cv2
import cvzone
import math
import os
from datetime import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

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
tracker = DeepSort(
    max_age=int(fps * 2),
    n_init=2,
    nms_max_overlap=1.0,
    max_cosine_distance=0.35,
)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_video_path, fourcc, fps, (1920, 1080))


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
    current_detections = []

    for info in results:
        for box in info.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            score = float(box.conf[0])          # 0.0 - 1.0
            cls_id = int(box.cls[0])
            cls_name = normalize_vehicle_class(classnames[cls_id])
            min_score = 0.30 if cls_name == "truck" else 0.45
            if (cls_name in allowed_classes) and (score > min_score):
                current_detections.append(([x1, y1, x2 - x1, y2 - y1], score, cls_name))

    track_results = tracker.update_tracks(current_detections, frame=frame)
    for track in track_results:
        if not track.is_confirmed():
            continue

        x1, y1, x2, y2 = track.to_ltrb()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        tid = int(track.track_id)
        if tid not in track_start_frame:
            track_start_frame[tid] = frame_index

        det_class = track.get_det_class()
        if det_class is None:
            det_class = track_class_name.get(tid, "tracker")
        track_class_name[tid] = det_class

        if det_class == "car":
            car_track_ids.add(tid)
        if det_class == "truck":
            truck_track_ids.add(tid)

        stay_seconds = (frame_index - track_start_frame[tid]) / fps
        label = f"{det_class} tracker:{tid} time:{stay_seconds:.1f}s"

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
