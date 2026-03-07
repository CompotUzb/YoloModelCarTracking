import cv2
import numpy as np

polygon_points = []

video_path = "C:/Users/Abduvaliy.DESKTOP-PR9DVS1/Documents/video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise SystemExit(f"Failed to open video: {video_path}")

ret, frame = cap.read()
if not ret or frame is None:
    raise SystemExit("Failed to read the first frame (bad file or codec).")

frame = cv2.resize(frame, (1920, 1080))

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_points.append((x, y))
        print(f"Point Added: (X: {x}, Y: {y})")

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", mouse_callback)

while True:
    display = frame.copy()

    if len(polygon_points) > 1:
        cv2.polylines(
            display,
            [np.array(polygon_points, dtype=np.int32)],
            isClosed=False,
            color=(0, 255, 0),
            thickness=2,
        )

    for (x, y) in polygon_points:
        cv2.circle(display, (x, y), 4, (0, 0, 255), -1)

    cv2.imshow("Frame", display)

    key = cv2.waitKey(10) & 0xFF
    if key == 27:  # Esc
        break
    elif key == ord("u"):  # undo last point
        if polygon_points:
            polygon_points.pop()
    elif key == ord("c"):  # clear all points
        polygon_points.clear()

cv2.destroyAllWindows()
cap.release()

print("Polygon Points:")
for x, y in polygon_points:
    print(f"X: {x}, Y: {y}")