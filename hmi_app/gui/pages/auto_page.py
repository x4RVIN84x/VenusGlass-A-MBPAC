from __future__ import annotations

import re
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QCheckBox,
    QSlider,
    QSizePolicy,
    QApplication,
)

from hmi_app.gui.image_view import ImageView
from hmi_app.io.camera import OpenCVCamera
from hmi_app.core.engine import QCPreviewEngine


class AutoPage(QWidget):
    CONTROLS_FIXED_W = 430

    def __init__(self, *, engine: QCPreviewEngine, cam: OpenCVCamera, parent=None):
        super().__init__(parent)

        self.engine = engine
        self.cam = cam

        self._running = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.on_tick)

        # Lost-track alarm state
        self._track_alarm_after_s = 5.0
        self._track_alarm_since: Optional[float] = None
        self._track_alarm_active = False
        self._track_alarm_last_beep = 0.0
        self._track_alarm_flash_i = 0
        self._track_alarm_manual_hold = False
        self._track_alarm_last_reset_at = 0.0

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        self.view = ImageView()
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self.view, 1)

        self.controls_panel = QWidget()
        self.controls_panel.setFixedWidth(self.CONTROLS_FIXED_W)
        self.controls_panel.setMinimumWidth(self.CONTROLS_FIXED_W)
        self.controls_panel.setMaximumWidth(self.CONTROLS_FIXED_W)
        self.controls_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        root.addWidget(self.controls_panel, 0)

        panel = QVBoxLayout(self.controls_panel)
        panel.setContentsMargins(0, 0, 0, 0)
        panel.setSpacing(10)

        gb = QGroupBox("Auto Controls")
        vb = QVBoxLayout(gb)
        vb.setSpacing(8)

        self.chk_show_stab = QCheckBox("Show stabilizer overlay")
        self.chk_show_stab.setChecked(True)
        vb.addWidget(self.chk_show_stab)

        self.chk_show_search_roi = QCheckBox("Search ROI box")
        self.chk_show_search_roi.setChecked(True)
        vb.addWidget(self.chk_show_search_roi)

        self.chk_show_notch_contour = QCheckBox("Actual notch contour edge")
        self.chk_show_notch_contour.setChecked(True)
        vb.addWidget(self.chk_show_notch_contour)

        self.chk_show_fitted_lines = QCheckBox("Fitted notch lines")
        self.chk_show_fitted_lines.setChecked(True)
        vb.addWidget(self.chk_show_fitted_lines)

        self.chk_show_new_anchors = QCheckBox("New anchors / skeleton")
        self.chk_show_new_anchors.setChecked(True)
        vb.addWidget(self.chk_show_new_anchors)

        self.chk_show_raw_points = QCheckBox("Raw feature points")
        self.chk_show_raw_points.setChecked(False)
        vb.addWidget(self.chk_show_raw_points)

        self.chk_show_legacy_debug = QCheckBox("Legacy dot/line fallback debug")
        self.chk_show_legacy_debug.setChecked(False)
        vb.addWidget(self.chk_show_legacy_debug)

        self.chk_show_stab_text = QCheckBox("Stabilizer text")
        self.chk_show_stab_text.setChecked(True)
        vb.addWidget(self.chk_show_stab_text)

        self.chk_show_bp = QCheckBox("Show baseplate contour + center")
        self.chk_show_bp.setChecked(True)
        vb.addWidget(self.chk_show_bp)

        self.chk_alarm = QCheckBox("DRAMATIC lost-track alarm after 5s")
        self.chk_alarm.setChecked(True)
        self.chk_alarm.setStyleSheet("""
            QCheckBox {
                color: #ffdddd;
                font-weight: 900;
            }
        """)
        vb.addWidget(self.chk_alarm)

        self.lbl_alarm = QLabel("Alarm: armed")
        self.lbl_alarm.setStyleSheet("font-size: 12px; font-weight: 800; color: #d8d8d8;")
        self.lbl_alarm.setWordWrap(False)
        vb.addWidget(self.lbl_alarm)

        self.btn_reset_alarm = QPushButton("RESET / ACK ALARM")
        self.btn_reset_alarm.setEnabled(False)
        self.btn_reset_alarm.clicked.connect(self.reset_alarm)
        vb.addWidget(self.btn_reset_alarm)

        # Display zoom
        self.chk_zoom = QCheckBox("Enable display zoom around baseplate")
        self.chk_zoom.setChecked(False)
        vb.addWidget(self.chk_zoom)

        self.lbl_zoom = QLabel("Display zoom: 1.00x")
        self.lbl_zoom.setStyleSheet("font-size: 12px; font-weight: 800;")
        vb.addWidget(self.lbl_zoom)

        self.sld_zoom = QSlider(Qt.Horizontal)
        self.sld_zoom.setRange(100, 300)
        self.sld_zoom.setValue(100)
        self.sld_zoom.valueChanged.connect(self._update_zoom_label)
        vb.addWidget(self.sld_zoom)

        vb.addWidget(QLabel("Stabilize cadence (every N frames)"))
        self.sld_stab_n = QSlider(Qt.Horizontal)
        self.sld_stab_n.setRange(1, 20)
        self.sld_stab_n.setValue(6)
        vb.addWidget(self.sld_stab_n)

        vb.addWidget(QLabel("Search padding (px)"))
        self.sld_pad = QSlider(Qt.Horizontal)
        self.sld_pad.setRange(0, 250)
        self.sld_pad.setValue(120)
        vb.addWidget(self.sld_pad)

        self.lbl_status = QLabel("Status: IDLE")
        self.lbl_status.setStyleSheet("font-size: 14px; font-weight: 800;")
        self.lbl_status.setWordWrap(False)
        self.lbl_status.setFixedHeight(32)
        vb.addWidget(self.lbl_status)

        self.btn_start = QPushButton("START")
        self.btn_stop = QPushButton("STOP")
        self.btn_stop.setEnabled(False)

        vb.addWidget(self.btn_start)
        vb.addWidget(self.btn_stop)

        panel.addWidget(gb)
        panel.addStretch(1)

        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)

    def _set_status(self, text: str):
        fm = QFontMetrics(self.lbl_status.font())
        self.lbl_status.setText(fm.elidedText(str(text), Qt.ElideRight, self.lbl_status.width()))

    def _update_zoom_label(self):
        z = float(self.sld_zoom.value()) / 100.0
        self.lbl_zoom.setText(f"Display zoom: {z:.2f}x")

    def _set_engine_settings_from_ui(self):
        self.engine.settings.stab_every_n = int(self.sld_stab_n.value())
        self.engine.settings.search_padding_px = int(self.sld_pad.value())

        self.engine.settings.show_stab = bool(self.chk_show_stab.isChecked())
        self.engine.settings.show_baseplate = bool(self.chk_show_bp.isChecked())

        setattr(self.engine.settings, "show_search_roi", bool(self.chk_show_search_roi.isChecked()))
        setattr(self.engine.settings, "show_notch_contour", bool(self.chk_show_notch_contour.isChecked()))
        setattr(self.engine.settings, "show_fitted_lines", bool(self.chk_show_fitted_lines.isChecked()))
        setattr(self.engine.settings, "show_new_anchors", bool(self.chk_show_new_anchors.isChecked()))
        setattr(self.engine.settings, "show_raw_points", bool(self.chk_show_raw_points.isChecked()))
        setattr(self.engine.settings, "show_legacy_debug", bool(self.chk_show_legacy_debug.isChecked()))
        setattr(self.engine.settings, "show_stabilizer_text", bool(self.chk_show_stab_text.isChecked()))

    def _reset_alarm_state(self):
        self._track_alarm_since = None
        self._track_alarm_active = False
        self._track_alarm_last_beep = 0.0
        self._track_alarm_flash_i = 0
        self._track_alarm_manual_hold = False
        self._track_alarm_last_reset_at = 0.0
        self.lbl_alarm.setText("Alarm: armed")
        self.lbl_alarm.setStyleSheet("font-size: 12px; font-weight: 800; color: #d8d8d8;")
        self.btn_reset_alarm.setEnabled(False)

    def reset_alarm(self):
        now = time.monotonic()
        self._track_alarm_since = now
        self._track_alarm_active = False
        self._track_alarm_last_beep = 0.0
        self._track_alarm_flash_i = 0
        self._track_alarm_manual_hold = True
        self._track_alarm_last_reset_at = now

        self.lbl_alarm.setText("Alarm: manually reset / monitoring again")
        self.lbl_alarm.setStyleSheet("font-size: 12px; font-weight: 900; color: #66d9ff;")
        self.btn_reset_alarm.setEnabled(False)

    def _track_progress_from_out(self, out) -> int:
        try:
            state = str(getattr(out, "state", "") or "").upper()
        except Exception:
            state = ""

        if state == "PASS":
            return 999

        try:
            text = str(getattr(out, "status_text", "") or "")
        except Exception:
            text = ""

        m = re.search(r"TRACK\s+(\d+)\s*/\s*(\d+)", text, flags=re.IGNORECASE)
        if not m:
            return 0

        try:
            return int(m.group(1))
        except Exception:
            return 0

    def _play_track_alarm_sound(self):
        try:
            import winsound
            winsound.PlaySound("SystemHand", winsound.SND_ALIAS | winsound.SND_ASYNC)
            return
        except Exception:
            pass

        try:
            QApplication.beep()
        except Exception:
            pass

    def _update_track_alarm(self, out) -> Tuple[bool, float]:
        if not self.chk_alarm.isChecked():
            self._reset_alarm_state()
            self.lbl_alarm.setText("Alarm: disabled")
            self.lbl_alarm.setStyleSheet("font-size: 12px; font-weight: 800; color: #9a9a9a;")
            return False, 0.0

        now = time.monotonic()
        progress = self._track_progress_from_out(out)
        good_track = progress > 0

        if good_track:
            self._reset_alarm_state()
            return False, 0.0

        if self._track_alarm_since is None:
            self._track_alarm_since = now

        elapsed = now - self._track_alarm_since
        active = elapsed >= float(self._track_alarm_after_s)
        self._track_alarm_active = active

        if active:
            self.lbl_alarm.setText(f"ALARM: lost track for {elapsed:.1f}s")
            self.lbl_alarm.setStyleSheet("font-size: 12px; font-weight: 1000; color: #ff4444;")
            self.btn_reset_alarm.setEnabled(True)

            if now - self._track_alarm_last_beep >= 0.80:
                self._track_alarm_last_beep = now
                self._play_track_alarm_sound()
        else:
            remaining = max(0.0, float(self._track_alarm_after_s) - elapsed)

            if self._track_alarm_manual_hold:
                self.lbl_alarm.setText(f"Alarm: reset acknowledged, re-arming in {remaining:.1f}s")
                self.lbl_alarm.setStyleSheet("font-size: 12px; font-weight: 900; color: #66d9ff;")
            else:
                self.lbl_alarm.setText(f"Alarm: warning in {remaining:.1f}s")
                self.lbl_alarm.setStyleSheet("font-size: 12px; font-weight: 800; color: #ffcc66;")

            self.btn_reset_alarm.setEnabled(self._track_alarm_since is not None)

        return active, elapsed

    def _focus_point_from_out(self, out, W: int, H: int):
        stab = getattr(out, "stab_info", None)
        if not isinstance(stab, dict):
            return W * 0.5, H * 0.5

        pts = []

        for key in ("expected_baseplate_center_abs", "current_baseplate_center_abs"):
            pt = stab.get(key)
            try:
                arr = np.asarray(pt, dtype=np.float64).reshape(-1)
                if arr.size >= 2 and np.isfinite(arr[0]) and np.isfinite(arr[1]):
                    pts.append((float(arr[0]), float(arr[1])))
            except Exception:
                pass

        if not pts:
            return W * 0.5, H * 0.5

        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)

        return cx, cy

    def _zoom_display_image(self, img, out):
        if img is None:
            return img

        if not self.chk_zoom.isChecked():
            return img

        zoom = float(self.sld_zoom.value()) / 100.0
        if zoom <= 1.01:
            return img

        H, W = img.shape[:2]

        focus_x, focus_y = self._focus_point_from_out(out, W, H)

        crop_w = max(50, int(round(W / zoom)))
        crop_h = max(50, int(round(H / zoom)))

        x0 = int(round(focus_x - crop_w * 0.5))
        y0 = int(round(focus_y - crop_h * 0.5))

        x0 = max(0, min(x0, W - crop_w))
        y0 = max(0, min(y0, H - crop_h))

        crop = img[y0:y0 + crop_h, x0:x0 + crop_w].copy()

        out_img = cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR)

        cv2.rectangle(out_img, (14, 14), (255, 54), (0, 0, 0), -1)
        cv2.rectangle(out_img, (14, 14), (255, 54), (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(
            out_img,
            f"DISPLAY ZOOM {zoom:.2f}x",
            (26, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        return out_img

    def _draw_track_alarm_overlay(self, img, elapsed_s: float):
        if img is None:
            return img

        vis = img.copy()
        H, W = vis.shape[:2]

        self._track_alarm_flash_i += 1
        flash = (self._track_alarm_flash_i // 6) % 2 == 0

        red = np.zeros_like(vis)
        red[:, :, 2] = 255
        alpha = 0.42 if flash else 0.24
        vis = cv2.addWeighted(red, alpha, vis, 1.0 - alpha, 0)

        scan_color = (0, 0, 80) if flash else (0, 0, 45)
        for y in range(0, H, 18):
            cv2.line(vis, (0, y), (W, y), scan_color, 1, lineType=cv2.LINE_AA)

        bar_color = (0, 0, 255) if flash else (0, 130, 255)
        cv2.rectangle(vis, (0, 0), (26, H), bar_color, -1)
        cv2.rectangle(vis, (W - 26, 0), (W, H), bar_color, -1)

        box_w = min(W - 120, 1020)
        box_h = 220
        x0 = max(36, (W - box_w) // 2)
        y0 = max(28, int(H * 0.08))
        x1 = x0 + box_w
        y1 = y0 + box_h

        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 0), -1)
        border_color = (0, 0, 255) if flash else (0, 220, 255)
        cv2.rectangle(vis, (x0, y0), (x1, y1), border_color, 6, lineType=cv2.LINE_AA)

        stripe_y0 = y1 - 36
        stripe_h = 26
        cv2.rectangle(vis, (x0 + 6, stripe_y0), (x1 - 6, stripe_y0 + stripe_h), (0, 0, 0), -1)

        stripe_w = 34
        for x in range(x0 + 8, x1 - 8, stripe_w):
            pts = np.array(
                [
                    [x, stripe_y0 + stripe_h],
                    [x + stripe_w // 2, stripe_y0],
                    [x + stripe_w, stripe_y0],
                    [x + stripe_w // 2, stripe_y0 + stripe_h],
                ],
                dtype=np.int32,
            )
            cv2.fillConvexPoly(vis, pts, (0, 170, 255) if flash else (0, 100, 200))

        icon_pad = 22
        icon_w = 120
        icon_x0 = x1 - icon_w - icon_pad
        icon_x1 = x1 - icon_pad
        icon_y0 = y0 + 24
        icon_y1 = y1 - 56

        tri = np.array(
            [
                [(icon_x0 + icon_x1) // 2, icon_y0 + 8],
                [icon_x0 + 10, icon_y1 - 8],
                [icon_x1 - 10, icon_y1 - 8],
            ],
            dtype=np.int32,
        )

        tri_fill = (0, 0, 255) if flash else (0, 130, 255)
        cv2.fillConvexPoly(vis, tri, tri_fill)
        cv2.polylines(vis, [tri], True, (0, 0, 0), 3, lineType=cv2.LINE_AA)

        ex_text = "!"
        ex_font = cv2.FONT_HERSHEY_SIMPLEX
        ex_scale = 2.3
        ex_thick = 5
        (tw, th), _base = cv2.getTextSize(ex_text, ex_font, ex_scale, ex_thick)

        tri_center_x = int((tri[0][0] + tri[1][0] + tri[2][0]) / 3.0)
        tri_center_y = int((tri[0][1] + tri[1][1] + tri[2][1]) / 3.0) + 10

        tx = int(round(tri_center_x - tw / 2))
        ty = int(round(tri_center_y + th / 2))

        cv2.putText(vis, ex_text, (tx, ty), ex_font, ex_scale, (0, 0, 0), ex_thick + 4, cv2.LINE_AA)
        cv2.putText(vis, ex_text, (tx, ty), ex_font, ex_scale, (255, 255, 255), ex_thick, cv2.LINE_AA)

        text_left = x0 + 30
        text_right = icon_x0 - 18
        text_cx = (text_left + text_right) // 2

        def put_centered(text, y, scale, color, thickness):
            (tw2, _th2), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
            tx2 = int(round(text_cx - tw2 / 2))
            ty2 = int(round(y))
            cv2.putText(vis, text, (tx2, ty2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 5, cv2.LINE_AA)
            cv2.putText(vis, text, (tx2, ty2), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

        put_centered("!!! TRACK LOST !!!", y0 + 74, 1.45, (0, 0, 255), 4)
        put_centered(f"NO GOOD TRACK FOR {elapsed_s:.1f}s", y0 + 128, 0.92, (255, 255, 255), 2)
        put_centered("CHECK GLASS POSITION / LIGHTING / BASEPLATE", y0 + 176, 0.72, (0, 230, 255), 2)

        return vis

    def start(self):
        if self._running:
            return

        self._running = True
        self._reset_alarm_state()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self._timer.start(33)
        self._set_status("Status: RUNNING")

    def stop(self):
        if not self._running:
            return

        self._running = False
        self._timer.stop()

        self._reset_alarm_state()

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

        self._set_status("Status: STOPPED")

    def on_tick(self):
        ok, frame = self.cam.read()

        if not ok or frame is None:
            self._set_status("Status: CAMERA READ FAIL")
            return

        if self.engine.recipe is None:
            self.view.set_bgr(frame)
            self._set_status("Status: NO PRODUCT LOADED (showing raw feed)")
            self._reset_alarm_state()
            return

        self._set_engine_settings_from_ui()

        try:
            out = self.engine.process_frame(frame)
        except Exception as e:
            self._set_status(f"Status: ENGINE ERROR: {e}")
            self.view.set_bgr(frame)
            return

        display = out.overlay_bgr if out.overlay_bgr is not None else frame

        # Zoom first so the alarm HUD stays full-screen on top of the zoomed feed.
        display = self._zoom_display_image(display, out)

        alarm_on, alarm_elapsed = self._update_track_alarm(out)
        if alarm_on:
            display = self._draw_track_alarm_overlay(display, alarm_elapsed)

        self.view.set_bgr(display)
        self._set_status(out.status_text)

    def close(self):
        try:
            self.stop()
        except Exception:
            pass