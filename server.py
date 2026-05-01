from flask import Flask, request, jsonify
import cv2
import time
from ultralytics import YOLO

app = Flask(__name__)

model = YOLO("yolov8n.pt")

@app.route("/", methods=["GET"])
def home():
    return "CCTV Counter Server Running"

@app.route("/count", methods=["POST"])
def count_cars():
    data = request.json

    video_url = data.get("video_url")
    duration = int(data.get("duration", 30))

    if not video_url:
        return jsonify({"success": False, "error": "video_url 없음"})

    cap = cv2.VideoCapture(video_url)

    if not cap.isOpened():
        return jsonify({"success": False, "error": "영상 열기 실패"})

    counted_ids = set()
    vehicle_count = 0

    start_time = time.time()

    while time.time() - start_time < duration:
        ret, frame = cap.read()

        if not ret:
            time.sleep(0.2)
            continue

        h, w = frame.shape[:2]
        line_y = int(h * 0.55)

        results = model.track(frame, persist=True, verbose=False)

        if not results or results[0].boxes is None or results[0].boxes.id is None:
            continue

        boxes = results[0].boxes
        ids = boxes.id.cpu().numpy().astype(int)
        classes = boxes.cls.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()

        for box, track_id, cls in zip(xyxy, ids, classes):
            if cls not in [2, 3, 5, 7]:
                continue

            x1, y1, x2, y2 = box
            center_y = int((y1 + y2) / 2)

            if abs(center_y - line_y) < 12 and track_id not in counted_ids:
                counted_ids.add(track_id)
                vehicle_count += 1

    cap.release()

    return jsonify({
        "success": True,
        "vehicle_count": vehicle_count,
        "duration": duration
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
