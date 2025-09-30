import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import matplotlib.pyplot as plt


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")


model = YOLO("yolov8n.pt")
model.to(device)

#  باز کردن ویدئو
video_path = "video.webm"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {video_path}")


#  ساخت tracker DeepSORT
tracker = DeepSort(max_age=40)


# ذخیره آمار شمارش افراد در هر 4 فریم
counts_per_frame = []


#  حلقه پردازش فریم‌ها
frame_count = 0
process_every_n = 2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % process_every_n != 0:
        continue

    scale_width = 800
    h, w = frame.shape[:2]
    new_h = int(h * scale_width / w)
    frame_resized = cv2.resize(frame, (scale_width, new_h))

    results = model.predict(frame_resized, iou=0.3, conf=0.1, classes=[0], verbose=False, device=device)

    detections_for_deepsort = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            detections_for_deepsort.append(([x1, y1, x2, y2], conf, "person"))

    tracked_objects = tracker.update_tracks(detections_for_deepsort, frame=frame_resized)

    person_count = 0
    for track in tracked_objects:
        if not track.is_confirmed():
            continue
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        track_id = track.track_id
        person_count += 1

        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(frame_resized, f"ID {track_id}", (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    counts_per_frame.append(person_count)

    cv2.putText(frame_resized, f"Count: {person_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("YOLO + DeepSORT Tracking", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# آزادسازی منابع
cap.release()
cv2.destroyAllWindows()

# گزارش آماری
if counts_per_frame:
    avg_count = sum(counts_per_frame) / len(counts_per_frame)
    max_count = max(counts_per_frame)
    min_count = min(counts_per_frame)

    print("\n--- counting person report ---")
    print(f"avg person in every 4 frame: {avg_count:.2f}")
    print(f"max person in every 4 frame: {max_count}")
    print(f"min person in every 4 frame: {min_count}")

    # رسم نمودار
    plt.figure(figsize=(12, 5))
    plt.plot(counts_per_frame, label="Person Count per Frame")
    plt.xlabel("Frame")
    plt.ylabel("Person Count")
    plt.title("Person Count Trend in Video")
    plt.legend()
    plt.grid(True)
    plt.show()



