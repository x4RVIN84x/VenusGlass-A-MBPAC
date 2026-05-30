from __future__ import annotations

from typing import Any, Dict, Optional

import cv2
import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout,
    QPushButton,
    QToolButton,
    QFrame,
    QSizePolicy,
    QScrollArea,
)


# ----------------------------
# Image helpers
# ----------------------------
def _is_bgr_image(img) -> bool:
    return isinstance(img, np.ndarray) and img.size > 0 and img.ndim in (2, 3)


def _to_bgr(img) -> Optional[np.ndarray]:
    if not _is_bgr_image(img):
        return None

    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.ndim == 3 and img.shape[2] == 3:
        return img.copy()

    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    return None


def _bgr_to_pixmap(img: Optional[np.ndarray], target_w: int, target_h: int) -> QPixmap:
    bgr = _to_bgr(img)

    if bgr is None:
        bgr = np.zeros((max(1, target_h), max(1, target_w), 3), dtype=np.uint8)
        cv2.putText(
            bgr,
            "NO SIGNAL",
            (20, max(35, target_h // 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    qimg = QImage(
        rgb.data,
        w,
        h,
        int(rgb.strides[0]),
        QImage.Format_RGB888,
    ).copy()

    pix = QPixmap.fromImage(qimg)

    if target_w > 0 and target_h > 0:
        pix = pix.scaled(
            target_w,
            target_h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

    return pix


# ----------------------------
# BGR QLabel
# ----------------------------
class BgrImageLabel(QLabel):
    def __init__(self, *, min_h: int = 120, parent=None):
        super().__init__(parent)

        self._bgr = None

        self.setMinimumHeight(min_h)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("""
            QLabel {
                background: #06070a;
                border: 1px solid #303747;
                border-radius: 10px;
                color: #8990a3;
            }
        """)

    def set_bgr(self, img):
        self._bgr = None if img is None else _to_bgr(img)
        self._refresh_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_pixmap()

    def _refresh_pixmap(self):
        if self.width() <= 2 or self.height() <= 2:
            return

        pix = _bgr_to_pixmap(self._bgr, self.width() - 10, self.height() - 10)
        self.setPixmap(pix)


# ----------------------------
# Feed card
# ----------------------------
class ProcFeedCard(QFrame):
    focus_requested = Signal(str)

    def __init__(self, key: str, title: str, help_text: str = "", parent=None):
        super().__init__(parent)

        self.key = key
        self.title = title
        self.help_text = help_text

        self._collapsed = False
        self._focused = False

        self.setObjectName("ProcFeedCard")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(6)

        self.lbl_title = QLabel(title)
        self.lbl_title.setStyleSheet("font-weight: 900; font-size: 12px; color: #f2f5ff;")

        self.btn_help = QToolButton()
        self.btn_help.setText("?")
        self.btn_help.setToolTip(help_text)
        self.btn_help.setFixedSize(22, 22)
        self.btn_help.setCursor(Qt.WhatsThisCursor)
        self.btn_help.setStyleSheet("""
            QToolButton {
                border: 1px solid #4a5166;
                border-radius: 11px;
                color: #dfe6ff;
                background: #1c2030;
                font-weight: 900;
            }
            QToolButton:hover {
                background: #2e385a;
                border: 1px solid #7d8dff;
            }
        """)

        self.btn_focus = QToolButton()
        self.btn_focus.setText("⛶")
        self.btn_focus.setToolTip("Focus this feed in the large viewer")
        self.btn_focus.setFixedSize(26, 22)
        self.btn_focus.clicked.connect(lambda: self.focus_requested.emit(self.key))

        self.btn_collapse = QToolButton()
        self.btn_collapse.setText("−")
        self.btn_collapse.setToolTip("Collapse / expand")
        self.btn_collapse.setFixedSize(26, 22)
        self.btn_collapse.clicked.connect(self.toggle_collapsed)

        for b in (self.btn_focus, self.btn_collapse):
            b.setStyleSheet("""
                QToolButton {
                    border: 1px solid #3c4355;
                    border-radius: 6px;
                    color: #eef2ff;
                    background: #161923;
                    font-weight: 900;
                }
                QToolButton:hover {
                    background: #26314c;
                    border: 1px solid #7285ff;
                }
            """)

        title_row.addWidget(self.lbl_title, 1)
        title_row.addWidget(self.btn_help, 0)
        title_row.addWidget(self.btn_focus, 0)
        title_row.addWidget(self.btn_collapse, 0)

        self.image = BgrImageLabel(min_h=118)
        self.image.setMinimumHeight(118)
        self.image.setMaximumHeight(170)

        root.addLayout(title_row)
        root.addWidget(self.image)

        self._apply_style()

    def set_image(self, img):
        self.image.set_bgr(img)

    def set_help(self, help_text: str):
        self.help_text = help_text or ""
        self.btn_help.setToolTip(self.help_text)

    def set_title(self, title: str):
        self.title = title or self.key
        self.lbl_title.setText(self.title)

    def set_focused(self, focused: bool):
        self._focused = bool(focused)
        self._apply_style()

    def toggle_collapsed(self):
        self._collapsed = not self._collapsed
        self.image.setVisible(not self._collapsed)
        self.btn_collapse.setText("+" if self._collapsed else "−")
        self._apply_style()

    def mouseDoubleClickEvent(self, event):
        self.focus_requested.emit(self.key)
        super().mouseDoubleClickEvent(event)

    def _apply_style(self):
        if self._focused:
            border = "#4a78ff"
            bg = "#101827"
        else:
            border = "#2c2f3a"
            bg = "#0d0f15"

        self.setStyleSheet(f"""
            QFrame#ProcFeedCard {{
                background: {bg};
                border: 1px solid {border};
                border-radius: 14px;
            }}
        """)


# ----------------------------
# Dashboard
# ----------------------------
class ProcDiagnosticsDashboard(QWidget):
    FEED_ORDER = [
        "base_gray",
        "base_edges",
        "base_mask",
        "frame_debug",
    ]

    DEFAULT_META = {
        "base_gray": {
            "title": "BASE GRAY / CONTRAST",
            "help": (
                "This shows the cleaned grayscale/contrast image before baseplate edge detection. "
                "Use it to check lighting, shadows, glare, and whether the baseplate is visible enough."
            ),
        },
        "base_edges": {
            "title": "BASE EDGES",
            "help": (
                "This shows the edges the app finds around the baseplate. "
                "If there are too many random edges, tune Canny/blur/contrast. "
                "If the baseplate edge disappears, thresholds are probably too strict."
            ),
        },
        "base_mask": {
            "title": "BASE CLOSED / MASK",
            "help": (
                "This shows the cleaned mask after gap-filling and morphology. "
                "Use it to see if the baseplate shape is solid, broken, or merged with noise."
            ),
        },
        "frame_debug": {
            "title": "FRAME / ANCHOR DEBUG",
            "help": (
                "This shows the geometry that places the live ROI: notch side lines, bottom line, "
                "bottom-left, bottom-mid, bottom-right, current center, and rotated ROI."
            ),
        },
    }

    def __init__(self, parent=None):
        super().__init__(parent)

        self._payload: Dict[str, Any] = {}
        self._focus_key: Optional[str] = None
        self._cards: Dict[str, ProcFeedCard] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        # Top toolbar
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(8)

        self.lbl_mode = QLabel("PROC DASHBOARD")
        self.lbl_mode.setStyleSheet("font-size: 15px; font-weight: 1000; color: #f4f6ff;")

        self.btn_main = QPushButton("MAIN")
        self.btn_main.setToolTip("Show the full diagnostic canvas in the large viewer")
        self.btn_main.clicked.connect(self.clear_focus)

        self.btn_reset = QPushButton("RESET PANELS")
        self.btn_reset.setToolTip("Expand every feed and clear focus")
        self.btn_reset.clicked.connect(self.reset_panels)

        for b in (self.btn_main, self.btn_reset):
            b.setStyleSheet("""
                QPushButton {
                    background: #151821;
                    color: #f2f5ff;
                    border: 1px solid #2f3545;
                    border-radius: 8px;
                    padding: 7px 12px;
                    font-weight: 900;
                }
                QPushButton:hover {
                    background: #22304c;
                    border: 1px solid #6880ff;
                }
            """)

        toolbar.addWidget(self.lbl_mode, 1)
        toolbar.addWidget(self.btn_main, 0)
        toolbar.addWidget(self.btn_reset, 0)

        root.addLayout(toolbar)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(10)

        self.main_view = BgrImageLabel(min_h=420)
        body.addWidget(self.main_view, 1)

        self.side_scroll = QScrollArea()
        self.side_scroll.setWidgetResizable(True)
        self.side_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.side_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.side_scroll.setMinimumWidth(300)
        self.side_scroll.setMaximumWidth(390)

        self.side_inner = QWidget()
        self.side_layout = QVBoxLayout(self.side_inner)
        self.side_layout.setContentsMargins(4, 4, 4, 4)
        self.side_layout.setSpacing(8)

        for key in self.FEED_ORDER:
            meta = self.DEFAULT_META[key]
            card = ProcFeedCard(key, meta["title"], meta["help"])
            card.focus_requested.connect(self.focus_feed)
            self._cards[key] = card
            self.side_layout.addWidget(card)

        self.side_layout.addStretch(1)
        self.side_scroll.setWidget(self.side_inner)

        body.addWidget(self.side_scroll, 0)
        root.addLayout(body, 1)

        self.setStyleSheet("""
            QWidget {
                background: #0b0d12;
            }
            QScrollArea {
                border: none;
                background: transparent;
            }
            QToolTip {
                color: #f4f6ff;
                background-color: #202433;
                border: 1px solid #6570a6;
                padding: 10px;
                font-size: 12px;
            }
        """)

    def set_payload(self, payload: Optional[Dict[str, Any]]):
        self._payload = payload or {}

        feeds = self._payload.get("feeds")
        if not isinstance(feeds, dict):
            feeds = {}

        for key, card in self._cards.items():
            feed = feeds.get(key)

            if isinstance(feed, dict):
                title = feed.get("title") or self.DEFAULT_META[key]["title"]
                help_text = feed.get("help") or self.DEFAULT_META[key]["help"]
                image = feed.get("image")
            else:
                title = self.DEFAULT_META[key]["title"]
                help_text = self.DEFAULT_META[key]["help"]
                image = feed

            card.set_title(title)
            card.set_help(help_text)
            card.set_image(image)

        self._refresh_main()

    def focus_feed(self, key: str):
        if key not in self._cards:
            return

        if self._focus_key == key:
            self._focus_key = None
        else:
            self._focus_key = key

        self._refresh_main()

    def clear_focus(self):
        self._focus_key = None
        self._refresh_main()

    def reset_panels(self):
        self._focus_key = None

        for card in self._cards.values():
            if card._collapsed:
                card.toggle_collapsed()

        self._refresh_main()

    def _refresh_main(self):
        feeds = self._payload.get("feeds")
        if not isinstance(feeds, dict):
            feeds = {}

        for key, card in self._cards.items():
            card.set_focused(key == self._focus_key)

        if self._focus_key is not None:
            feed = feeds.get(self._focus_key)
            if isinstance(feed, dict):
                img = feed.get("image")
                title = feed.get("title") or self.DEFAULT_META[self._focus_key]["title"]
            else:
                img = feed
                title = self.DEFAULT_META[self._focus_key]["title"]

            self.lbl_mode.setText(f"PROC DASHBOARD  /  FOCUS: {title}")
            self.main_view.set_bgr(img)
            return

        main = self._payload.get("main")
        if main is None:
            main = self._payload.get("main_bgr")

        self.lbl_mode.setText("PROC DASHBOARD")
        self.main_view.set_bgr(main)