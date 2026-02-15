from __future__ import annotations

import cv2
import numpy as np

from PySide6.QtCore import Qt, QSize, QRect
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QWidget, QSizePolicy


def bgr_to_qimage(bgr: np.ndarray) -> QImage:
    """
    Convert a BGR OpenCV frame to a QImage (RGB888).
    NOTE: We .copy() so the QImage owns its memory and doesn't reference
    a numpy buffer that will be overwritten on the next frame.
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()


class ImageView(QWidget):
    """
    Stable live image view for HMI.

    Key design choice:
    - We DO NOT use QLabel.setPixmap() because it can cause layout / sizeHint
      feedback at high frame rates ("breathing" between 2 sizes).
    - We paint the current QImage in paintEvent instead.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setStyleSheet("background: #111; border: 1px solid #2a2a2a; border-radius: 10px;")

        # Stable in layouts, doesn't fight geometry
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(320, 240)

        self._img: QImage | None = None

    def sizeHint(self) -> QSize:
        # Give the layout a stable hint so it doesn't oscillate by a pixel or two.
        # Adjust if you want a different default preview footprint.
        return QSize(960, 540)

    def set_bgr(self, frame_bgr: np.ndarray) -> None:
        self._img = bgr_to_qimage(frame_bgr)
        self.update()  # schedule repaint

    def set_qimage(self, img: QImage) -> None:
        # Optional convenience if you already have a QImage elsewhere
        self._img = img if img.isNull() else img.copy()
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        if self._img is None or self._img.isNull():
            return

        target = self.rect()

        # Fit image to widget while keeping aspect ratio
        img_size = self._img.size()
        img_size.scale(target.size(), Qt.KeepAspectRatio)

        x = (target.width() - img_size.width()) // 2
        y = (target.height() - img_size.height()) // 2
        fitted = QRect(x, y, img_size.width(), img_size.height())

        painter.drawImage(fitted, self._img)
