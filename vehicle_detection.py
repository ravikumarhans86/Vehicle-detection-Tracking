import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
from collections import defaultdict

# ---------------- CONFIG ----------------
VIDEO_PATH = "vehicle.mp4"
MODEL_PATH = "yolov8n.pt"
CONFIDENCE = 0.4
# ---------------------------------------

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

# ---------------- ROIs ----------------
IN_ROI = np.array([
    [100, 250],
    [600, 250],
    [600, 500],
    [100, 500]
])

OUT_ROI = np.array([
    [650, 250],
    [1150, 250],
    [1150, 500],
    [650, 500]
])

in_zone = sv.PolygonZone(IN_ROI)
out_zone = sv.PolygonZone(OUT_ROI)

# ---------------- Annotators ----------------
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=0.6)

in_annotator = sv.PolygonZoneAnnotator(
    zone=in_zone, color=sv.Color.GREEN, thickness=2
)
out_annotator = sv.PolygonZoneAnnotator(
    zone=out_zone, color=sv.Color.RED, thickness=2
)

tracker = sv.ByteTrack()

# COCO vehicle classes
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, bike, bus, truck

# ---------------- Counters ----------------
in_ids = set()
out_ids = set()

in_counts = defaultdict(int)
out_counts = defaultdict(int)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=CONFIDENCE, classes=VEHICLE_CLASSES)[0]
    detections = sv.Detections.from_ultralytics(results)

    detections = tracker.update_with_detections(detections)

    if detections.tracker_id is None:
        continue

    in_mask = in_zone.trigger(detections)
    out_mask = out_zone.trigger(detections)

    for i, tid in enumerate(detections.tracker_id):
        class_name = model.names[int(detections.class_id[i])]

        # IN ROI counting
        if in_mask[i] and tid not in in_ids:
            in_ids.add(tid)
            in_counts[class_name] += 1

        # OUT ROI counting
        if out_mask[i] and tid in in_ids and tid not in out_ids:
            out_ids.add(tid)
            out_counts[class_name] += 1

    # ---------- Labels ----------
    labels = [
        model.names[int(class_id)]
        for class_id in detections.class_id
    ]

    frame = box_annotator.annotate(frame, detections)
    frame = label_annotator.annotate(frame, detections, labels)
    frame = in_annotator.annotate(frame)
    frame = out_annotator.annotate(frame)

    # ---------- Display IN data ----------
    y = 40
    cv2.putText(frame, "IN ROI", (30, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    for cls, cnt in in_counts.items():
        y += 30
        cv2.putText(frame, f"{cls}: {cnt}", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # ---------- Display OUT data ----------
    y = 40
    cv2.putText(frame, "OUT ROI", (900, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    for cls, cnt in out_counts.items():
        y += 30
        cv2.putText(frame, f"{cls}: {cnt}", (900, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Vehicle IN / OUT ROI Counting", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
