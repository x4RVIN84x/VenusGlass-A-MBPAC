# hmi_app/io/camera.py
from __future__ import annotations
import cv2

class OpenCVCamera:
    def __init__(
        self,
        index: int = 0,
        width: int = 1920,   # C920 PRO native 1080p (or 1280 for 720p)
        height: int = 1080,
        fps: int = 30,
        use_dshow: bool = True,
        disable_autofocus: bool = True
    ):
        api = cv2.CAP_DSHOW if use_dshow else 0
        self.cap = cv2.VideoCapture(index, api)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index={index}")

        # Use MJPEG for high-resolution 30 FPS without USB 2.0 bus bottleneck
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        self.cap.set(cv2.CAP_PROP_FPS, int(fps))

        if disable_autofocus:
            # 0 = Manual focus (locks current physical distance)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    def read(self):
        ok, frame = self.cap.read()
        return ok, frame

    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass