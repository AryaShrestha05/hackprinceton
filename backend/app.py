from __future__ import annotations

from typing import Generator

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response
from flask_cors import CORS


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}})

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/video_feed")
    def video_feed() -> Response:
        return Response(
            stream_with_visualization(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    return app


def stream_with_visualization() -> Generator[bytes, None, None]:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam")

    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    try:
        while True:
            success, frame = cap.read()
            if not success:
                continue

            # Process with MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            annotated = frame.copy()
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    annotated,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(245, 255, 255), thickness=2, circle_radius=2),
                )

                lm = results.pose_landmarks.landmark
                h, w, _ = frame.shape

                nose = np.array([lm[0].x * w, lm[0].y * h])
                left_shoulder = np.array([lm[11].x * w, lm[11].y * h])
                right_shoulder = np.array([lm[12].x * w, lm[12].y * h])

                left_angle = calculate_angle(nose, left_shoulder, right_shoulder)
                right_angle = calculate_angle(nose, right_shoulder, left_shoulder)
                nose_angle = calculate_angle(left_shoulder, nose, right_shoulder)

                cv2.putText(
                    annotated,
                    f"{int(left_angle)}°",
                    tuple(left_shoulder.astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    f"{int(right_angle)}°",
                    tuple(right_shoulder.astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    f"{int(nose_angle)}°",
                    tuple(nose.astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            # Mirror the output
            mirrored = cv2.flip(annotated, 1)

            ret, buffer = cv2.imencode(".jpg", mirrored)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
    finally:
        cap.release()
        pose.close()


def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return float(angle)


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)

