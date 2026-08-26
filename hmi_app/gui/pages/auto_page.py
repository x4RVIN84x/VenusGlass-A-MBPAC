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
from hmi_app.core.engine import QCPreviewEngine, get_configured_tolerances
from hmi_app.core.overlay import (
    draw_adaptive_text,
    draw_cctv_footer,
    draw_operator_measurement_hud,
)


class AutoPage(QWidget):
    CONTROLS_FIXED_W = 430

    def __init__(self, *, engine: QCPreviewEngine, cam: OpenCVCamera, parent=None):
        super().__init__(parent)

        self.engine = engine
        self.cam = cam

        self._running = False
        self._last_evidence_bgr = None
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

        self.chk_show_stab_text = QCheckBox("Measurement HUD")
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
        self.chk_zoom.setChecked(True)
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

        # Kept directly above the changing status so the operator can always
        # see the active PASS limits without digging into Calibration.
        self.lbl_tolerance = QLabel("Limits: —")
        self.lbl_tolerance.setStyleSheet(
            "font-size: 11px; font-weight: 900; color: #cbd9ff; "
            "background: #171b25; border: 1px solid #2f3a54; "
            "border-radius: 5px; padding: 3px 6px;"
        )
        self.lbl_tolerance.setWordWrap(False)
        self.lbl_tolerance.setFixedHeight(26)
        self.lbl_tolerance.setToolTip("Acceptance limits saved for the active product.")
        vb.addWidget(self.lbl_tolerance)

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

        self._last_zoom_crop = None
        self._update_zoom_label()
        self._update_tolerance_label()

    def _set_status(self, text: str):
        fm = QFontMetrics(self.lbl_status.font())
        self.lbl_status.setText(fm.elidedText(str(text), Qt.ElideRight, self.lbl_status.width()))

    def _update_zoom_label(self):
        z = float(self.sld_zoom.value()) / 100.0
        self.lbl_zoom.setText(f"Display zoom: {z:.2f}x")

    def _update_tolerance_label(self):
        recipe = getattr(self.engine, "recipe", None)
        if recipe is None:
            self.lbl_tolerance.setText("Limits: no product loaded")
            return

        tolerance = get_configured_tolerances(getattr(recipe, "cfg", {}))
        x_mm = tolerance.get("x_mm")
        y_mm = tolerance.get("y_mm")
        angle_deg = tolerance.get("angle_deg")

        if x_mm is not None and y_mm is not None:
            self.lbl_tolerance.setText(
                f"LIMITS  X +/- {float(x_mm):.3f} mm | "
                f"Y +/- {float(y_mm):.3f} mm | "
                f"ang +/- {float(angle_deg):.2f} deg"
            )
            return

        # Only uncalibrated, pre-v1.5.4.2 recipes can reach this path.
        self.lbl_tolerance.setText(
            f"LIMITS  X/Y need calibration scale | ang +/- {float(angle_deg):.2f} deg"
        )

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
        # Guidance and measurement text are redrawn below, after display zoom.
        self.engine.settings.defer_operator_hud = True

    def get_evidence_frame(self):
        """Return the last fully annotated Auto frame for a future Reports page."""
        if self._last_evidence_bgr is None:
            return None
        return self._last_evidence_bgr.copy()

    @staticmethod
    def _timestamp_for_footer(out) -> str:
        stab = getattr(out, "stab_info", None)
        if isinstance(stab, dict):
            stamp = stab.get("frame_timestamp")
            if isinstance(stamp, dict) and stamp.get("local"):
                return str(stamp["local"])

        raw = str(getattr(out, "captured_at", "") or "")
        return raw.replace("T", " ")[:19] or "---- -- -- --:--:--"

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

        self._last_zoom_crop = None

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

        self._last_zoom_crop = (int(x0), int(y0), int(crop_w), int(crop_h), int(W), int(H))

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

    def _operator_lines_from_stab(self, stab: dict):
        if not isinstance(stab, dict):
            return []

        corr = stab.get("baseplate_correction_vector")

        if not isinstance(corr, dict):
            offset = stab.get("current_offset_display")
            if isinstance(offset, dict):
                try:
                    corr = {
                        "unit": str(offset.get("unit", "px")),
                        "screen_dx": -float(offset.get("dx", 0.0)),
                        "screen_dy": -float(offset.get("dy", 0.0)),
                    }
                except Exception:
                    corr = None

        if not isinstance(corr, dict):
            measure = stab.get("current_notch_measure_mm")
            if isinstance(measure, dict):
                try:
                    corr = {
                        "unit": "mm",
                        "screen_dx": -float(measure.get("dx", 0.0)),
                        "screen_dy": -float(measure.get("dy", 0.0)),
                    }
                except Exception:
                    corr = None

        if not isinstance(corr, dict):
            return []

        try:
            unit = str(corr.get("unit", "px"))
            dx = float(corr.get("screen_dx", 0.0))
            dy = float(corr.get("screen_dy", 0.0))
        except Exception:
            return []

        lines = []

        if abs(dx) >= 0.01:
            lines.append(f"{abs(dx):.2f}{unit} {'RIGHT' if dx > 0 else 'LEFT'}")

        if abs(dy) >= 0.01:
            lines.append(f"{abs(dy):.2f}{unit} {'DOWN' if dy > 0 else 'UP'}")

        if not lines:
            return ["CENTERED"]

        if len(lines) == 1:
            return [f"MOVE {lines[0]}"]

        return [f"MOVE {lines[0]}", f"AND {lines[1]}"]

    def _map_abs_pt_to_display(self, pt, W: int, H: int):
        try:
            x = float(pt[0])
            y = float(pt[1])
        except Exception:
            return None

        if not np.isfinite(x) or not np.isfinite(y):
            return None

        zinfo = getattr(self, "_last_zoom_crop", None)

        if zinfo is None:
            return int(round(x)), int(round(y))

        try:
            x0, y0, crop_w, crop_h, out_w, out_h = zinfo
            dx = (x - float(x0)) * float(out_w) / max(1.0, float(crop_w))
            dy = (y - float(y0)) * float(out_h) / max(1.0, float(crop_h))
        except Exception:
            return int(round(x)), int(round(y))

        return int(round(dx)), int(round(dy))

    def _expected_pt_from_stab_for_display(self, stab: dict):
        if not isinstance(stab, dict):
            return None

        exp = stab.get("expected_baseplate_center_abs")
        try:
            arr = np.asarray(exp, dtype=np.float64).reshape(-1)
            if arr.size >= 2 and np.isfinite(arr[0]) and np.isfinite(arr[1]):
                return float(arr[0]), float(arr[1])
        except Exception:
            pass

        cur = stab.get("current_baseplate_center_abs")
        try:
            cur = np.asarray(cur, dtype=np.float64).reshape(-1)
            if cur.size < 2 or not np.isfinite(cur[0]) or not np.isfinite(cur[1]):
                return None
        except Exception:
            return None

        corr = stab.get("baseplate_correction_vector")
        if isinstance(corr, dict):
            try:
                if "screen_dx_px" in corr and "screen_dy_px" in corr:
                    return (
                        float(cur[0]) + float(corr.get("screen_dx_px", 0.0)),
                        float(cur[1]) + float(corr.get("screen_dy_px", 0.0)),
                    )

                dx = float(corr.get("screen_dx", 0.0))
                dy = float(corr.get("screen_dy", 0.0))
                unit = str(corr.get("unit", "px")).lower()
                px_per_mm = corr.get("px_per_mm", None)

                if unit == "mm" and px_per_mm is not None:
                    dx *= float(px_per_mm)
                    dy *= float(px_per_mm)

                return float(cur[0]) + dx, float(cur[1]) + dy
            except Exception:
                pass

        return None

    def _draw_operator_guidance_hud(self, img, out):
        """
        Draw target cross + bottom-right movement badge AFTER display zoom.

        This guarantees the operator guidance remains visible even when the
        zoomed view crops the original overlay HUD.
        """
        if img is None or out is None:
            return img

        stab = getattr(out, "stab_info", None)
        if not isinstance(stab, dict):
            return img

        vis = img.copy()
        H, W = vis.shape[:2]

        cur = stab.get("current_baseplate_center_abs")
        exp = self._expected_pt_from_stab_for_display(stab)

        cur_d = self._map_abs_pt_to_display(cur, W, H) if cur is not None else None
        exp_d = self._map_abs_pt_to_display(exp, W, H) if exp is not None else None

        if exp_d is not None and (-80 <= exp_d[0] <= W + 80) and (-80 <= exp_d[1] <= H + 80):
            ex = max(0, min(W - 1, exp_d[0]))
            ey = max(0, min(H - 1, exp_d[1]))

            cv2.circle(vis, (ex, ey), 22, (255, 0, 255), 3, lineType=cv2.LINE_AA)
            cv2.circle(vis, (ex, ey), 13, (255, 255, 0), 2, lineType=cv2.LINE_AA)
            cv2.drawMarker(
                vis,
                (ex, ey),
                (255, 0, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=44,
                thickness=3,
                line_type=cv2.LINE_AA,
            )

            draw_adaptive_text(vis, "TARGET", (ex + 16, ey - 18), scale=0.50, thickness=1)

        if cur_d is not None and exp_d is not None:
            cx = max(0, min(W - 1, cur_d[0]))
            cy = max(0, min(H - 1, cur_d[1]))
            ex = max(0, min(W - 1, exp_d[0]))
            ey = max(0, min(H - 1, exp_d[1]))

            cv2.arrowedLine(vis, (cx, cy), (ex, ey), (255, 0, 255), 4, cv2.LINE_AA, tipLength=0.25)
            cv2.circle(vis, (cx, cy), 15, (0, 0, 255), 2, lineType=cv2.LINE_AA)

        lines = self._operator_lines_from_stab(stab)
        if lines:
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.58
            thick = 2
            sizes = [cv2.getTextSize(str(t), font, scale, thick)[0] for t in lines]
            text_w = max((s[0] for s in sizes), default=180)
            box_w = int(min(max(text_w + 48, 260), max(260, W - 48)))
            box_h = 58 if len(lines) == 1 else 82

            x0 = max(18, W - box_w - 24)
            y0 = max(130, H - box_h - 48)
            x1 = min(W - 18, x0 + box_w)
            y1 = min(H - 18, y0 + box_h)

            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 0), -1, lineType=cv2.LINE_AA)
            cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 0, 255), 2, lineType=cv2.LINE_AA)
            cv2.rectangle(vis, (x0, y0), (x0 + 8, y1), (255, 0, 255), -1, lineType=cv2.LINE_AA)

            start_y = int(round((y0 + y1) * 0.5 - (len(lines) - 1) * 13 + 8))
            for i, line in enumerate(lines):
                (tw, _th), _base = cv2.getTextSize(str(line), font, scale, thick)
                tx = int(round((x0 + x1) * 0.5 - tw * 0.5))
                ty = int(round(start_y + i * 26))

                cv2.putText(vis, str(line), (tx, ty), font, scale, (0, 0, 0), thick + 4, cv2.LINE_AA)
                cv2.putText(vis, str(line), (tx, ty), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

        return vis


    def start(self):
        if self._running:
            return

        self._running = True
        self._reset_alarm_state()
        self._update_tolerance_label()

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
        self._update_tolerance_label()
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

        # Draw operator-facing annotations AFTER zoom so the readable HUD stays
        # on screen and exists exactly once.
        if self.chk_show_stab.isChecked() and self.chk_show_stab_text.isChecked():
            draw_operator_measurement_hud(display, getattr(out, "stab_info", None))

        display = self._draw_operator_guidance_hud(display, out)

        alarm_on, alarm_elapsed = self._update_track_alarm(out)
        if alarm_on:
            display = self._draw_track_alarm_overlay(display, alarm_elapsed)

        # Burn report evidence metadata into the final displayed frame.  It is
        # deliberately last so zoom/alarm rendering cannot cover or crop it.
        fps_value = getattr(self.engine, "_fps", None)
        if fps_value is not None and float(fps_value) <= 0:
            fps_value = None
        draw_cctv_footer(
            display,
            timestamp=self._timestamp_for_footer(out),
            product_name=getattr(self.engine.recipe, "name", "NO PRODUCT"),
            fps=fps_value,
        )
        self._last_evidence_bgr = display.copy()

        self.view.set_bgr(display)
        self._set_status(out.status_text)

    def close(self):
        try:
            self.stop()
        except Exception:
            pass
