from flask import Flask, request, jsonify
import cv2
import time
import subprocess
import numpy as np
import imageio_ffmpeg

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "CCTV Counter Server Running"

@app.route("/count", methods=["POST"])
def count_cars():
    try:
        data = request.json or {}

        video_url = data.get("video_url")
        duration = int(data.get("duration", 20))

        if not video_url:
            return jsonify({
                "success": False,
                "error": "video_url 없음"
            })

        width = 640
        height = 360
        fps = 5
        frame_size = width * height * 3

        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

        cmd = [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel", "error",
            "-headers", "User-Agent: Mozilla/5.0\r\n",
            "-i", video_url,
            "-vf", f"fps={fps},scale={width}:{height}",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "pipe:1"
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8
        )

        subtractor = cv2.createBackgroundSubtractorMOG2(
            history=120,
            varThreshold=45,
            detectShadows=True
        )

        vehicle_count = 0
        last_count_time = 0
        counted_cooldown = 1.0

        start_time = time.time()

        while time.time() - start_time < duration:
            raw_frame = process.stdout.read(frame_size)

            if len(raw_frame) != frame_size:
                break

            frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))

            line_y = int(height * 0.55)

            fgmask = subtractor.apply(frame)
            _, thresh = cv2.threshold(fgmask, 220, 255, cv2.THRESH_BINARY)

            kernel = np.ones((5, 5), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            thresh = cv2.dilate(thresh, kernel, iterations=2)

            contours, _ = cv2.findContours(
                thresh,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            now = time.time()

            for contour in contours:
                area = cv2.contourArea(contour)

                if area < 1200:
                    continue

                x, y, w, h = cv2.boundingRect(contour)

                if w < 35 or h < 25:
                    continue

                center_y = y + h // 2

                if abs(center_y - line_y) < 15:
                    if now - last_count_time >= counted_cooldown:
                        vehicle_count += 1
                        last_count_time = now

        try:
            process.kill()
        except:
            pass

        return jsonify({
            "success": True,
            "vehicle_count": vehicle_count,
            "duration": duration
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
