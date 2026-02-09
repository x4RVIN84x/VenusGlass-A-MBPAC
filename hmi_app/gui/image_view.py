from __future__ import annotations
import numpy as np
import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel
from PySide6.QtWidgets import QLabel, QSizePolicy

def bgr_to_qimage(bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)

class ImageView(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background: #111; border: 1px solid #2a2a2a; border-radius: 10px;")
        self._last = None

        # ✅ stops “gets bigger every frame”
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setMinimumSize(320, 240)

    def set_bgr(self, frame_bgr: np.ndarray):
        self._last = frame_bgr
        qimg = bgr_to_qimage(frame_bgr)
        pix = QPixmap.fromImage(qimg)
        self.setPixmap(pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._last is not None:
            self.set_bgr(self._last)
