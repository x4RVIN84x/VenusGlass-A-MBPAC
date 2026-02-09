from __future__ import annotations
import cv2

class OpenCVCamera:
    def __init__(self, index=0, width=1280, height=720, fps=30, use_dshow=True):
        api = cv2.CAP_DSHOW if use_dshow else 0
        self.cap = cv2.VideoCapture(index, api)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index={index}")

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        self.cap.set(cv2.CAP_PROP_FPS, int(fps))

    def read(self):
        ok, frame = self.cap.read()
        return ok, frame

    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass
