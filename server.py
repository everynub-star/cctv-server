from flask import Flask, request, jsonify
import cv2
import time

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "CCTV Counter Server Running"

@app.route("/count", methods=["POST"])
def count_cars():
    data = request.json or {}

    video_url = data.get("video_url")
    duration = int(data.get("duration", 30))

    if not video_url:
        return jsonify({
            "success": False,
            "error": "video_url 없음"
        })

    cap = cv2.VideoCapture(video_url)

    if not cap.isOpened():
        return jsonify({
            "success": False,
            "error": "영상 열기 실패"
        })

    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=300,
        varThreshold=40,
        detectShadows=True
    )

    vehicle_count = 0
    last_count_time = 0
    start_time = time.time()

    counted_cooldown = 1.2

    while time.time() - start_time < duration:
        ret, frame = cap.read()

        if not ret:
            time.sleep(0.2)
            continue

        frame = cv2.resize(frame, (640, 360))
        height, width = frame.shape[:2]

        line_y = int(height * 0.55)

        fgmask = subtractor.apply(frame)
        _, thresh = cv2.threshold(fgmask, 220, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        now = time.time()

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < 900:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            if w < 25 or h < 20:
                continue

            center_y = y + h // 2

            if abs(center_y - line_y) < 12:
                if now - last_count_time >= counted_cooldown:
                    vehicle_count += 1
                    last_count_time = now

    cap.release()

    return jsonify({
        "success": True,
        "vehicle_count": vehicle_count,
        "duration": duration
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
