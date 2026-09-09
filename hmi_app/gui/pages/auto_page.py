from __future__ import annotations

import re
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from PySide6.QtCore import QTimer, Qt, QPointF
from PySide6.QtGui import QColor, QBrush, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QGroupBox,
    QFrame,
    QScrollArea,
    QLabel,
    QPushButton,
    QCheckBox,
    QSlider,
    QProgressBar,
    QSizePolicy,
    QApplication,
)

from hmi_app.gui.image_view import ImageView
from hmi_app.io.camera import OpenCVCamera
from hmi_app.core.engine import QCPreviewEngine
from hmi_app.core.report_store import ReportStore
from hmi_app.plc import (
    InspectionOutcome,
    InspectionPlcService,
    InspectionTelemetry,
    LiveInspectionState,
    PendingDecisionError,
    build_decision_from_output,
    build_live_telemetry,
)


class ConfidenceTriangle(QWidget):
    """A three-outcome evidence map for PASS, FAIL, and a missing baseplate.

    The centre is neutral amber.  The live dot moves smoothly from the centre
    toward the evidence it has accumulated: top is PASS, lower-right is FAIL,
    and lower-left is BASEPLATE NOT FOUND.  A small cached triangular colour
    mesh makes that relationship legible without using a heavy chart library.
    """

    PASS = QColor("#38d27b")
    FAIL = QColor("#d2042d")
    BASEPLATE = QColor("#80461b")
    NEUTRAL = QColor("#f2a23a")

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scores = {"PASS": 0.0, "FAIL": 0.0, "BASEPLATE": 0.0}
        self._cache = QPixmap()
        self._cache_size = None
        self.setMinimumSize(118, 82)
        self.setMaximumHeight(88)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setToolTip(
            "Amber centre: tracking. Top: PASS. Lower right: FAIL. Lower left: baseplate not found."
        )

    @staticmethod
    def _mix_colours(weights) -> QColor:
        total = sum(max(0.0, float(weight)) for _colour, weight in weights)
        if total <= 0.0:
            return QColor(ConfidenceTriangle.NEUTRAL)
        red = sum(colour.red() * max(0.0, float(weight)) for colour, weight in weights) / total
        green = sum(colour.green() * max(0.0, float(weight)) for colour, weight in weights) / total
        blue = sum(colour.blue() * max(0.0, float(weight)) for colour, weight in weights) / total
        return QColor(round(red), round(green), round(blue))

    def set_scores(self, scores: dict):
        self._scores = {
            "PASS": max(0.0, min(1.0, float(scores.get("PASS", 0.0)))),
            "FAIL": max(0.0, min(1.0, float(scores.get("FAIL", 0.0)))),
            "BASEPLATE": max(0.0, min(1.0, float(scores.get("BASEPLATE", 0.0)))),
        }
        self.update()

    def visual_color(self) -> QColor:
        weights = self._normalised_weights()
        return self._mix_colours(
            (
                (self.NEUTRAL, weights["NEUTRAL"]),
                (self.PASS, weights["PASS"]),
                (self.FAIL, weights["FAIL"]),
                (self.BASEPLATE, weights["BASEPLATE"]),
            )
        )

    def dominant_outcome(self) -> tuple[str, float]:
        outcome = max(self._scores, key=self._scores.get)
        return outcome, float(self._scores[outcome])

    def _normalised_weights(self) -> dict:
        total = sum(self._scores.values())
        if total > 1.0:
            return {
                "PASS": self._scores["PASS"] / total,
                "FAIL": self._scores["FAIL"] / total,
                "BASEPLATE": self._scores["BASEPLATE"] / total,
                "NEUTRAL": 0.0,
            }
        return {**self._scores, "NEUTRAL": 1.0 - total}

    @staticmethod
    def _point(a: float, b: float, c: float, top: QPointF, right: QPointF, left: QPointF) -> QPointF:
        return QPointF(
            a * top.x() + b * right.x() + c * left.x(),
            a * top.y() + b * right.y() + c * left.y(),
        )

    @classmethod
    def _mesh_colour(cls, a: float, b: float, c: float) -> QColor:
        # At the centroid all three barycentric values are equal, so amber is
        # dominant.  It fades naturally toward the three outcome vertices.
        centre_weight = min(1.0, 3.0 * min(a, b, c))
        edge_mix = cls._mix_colours(((cls.PASS, a), (cls.FAIL, b), (cls.BASEPLATE, c)))
        return cls._mix_colours(((cls.NEUTRAL, centre_weight), (edge_mix, 1.0 - centre_weight)))

    def _triangle_points(self):
        margin = 10.0
        top = QPointF(self.width() * 0.5, margin)
        left = QPointF(margin, self.height() - margin)
        right = QPointF(self.width() - margin, self.height() - margin)
        return top, right, left

    def _background(self):
        size = self.size()
        if self._cache_size == size and not self._cache.isNull():
            return self._cache

        self._cache_size = size
        self._cache = QPixmap(size)
        self._cache.fill(Qt.transparent)
        painter = QPainter(self._cache)
        painter.setRenderHint(QPainter.Antialiasing)
        top, right, left = self._triangle_points()
        resolution = 18

        # The small mesh is rendered only after resize, not for every camera
        # frame.  It gives a smooth three-way colour blend inside the triangle.
        for i in range(resolution):
            for j in range(resolution - i):
                def pt(ai, bj):
                    a = ai / resolution
                    b = bj / resolution
                    c = 1.0 - a - b
                    return self._point(a, b, c, top, right, left)

                def colour(ai, bj):
                    a = ai / resolution
                    b = bj / resolution
                    return self._mesh_colour(a, b, 1.0 - a - b)

                p0, p1, p2 = pt(i, j), pt(i + 1, j), pt(i, j + 1)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(self._mix_colours(((colour(i, j), 1), (colour(i + 1, j), 1), (colour(i, j + 1), 1)))))
                painter.drawPolygon(QPolygonF((p0, p1, p2)))

                if i + j < resolution - 1:
                    p3 = pt(i + 1, j + 1)
                    painter.setBrush(QBrush(self._mix_colours(((colour(i + 1, j), 1), (colour(i + 1, j + 1), 1), (colour(i, j + 1), 1)))))
                    painter.drawPolygon(QPolygonF((p1, p3, p2)))

        outline = QPainterPath()
        outline.moveTo(top)
        outline.lineTo(right)
        outline.lineTo(left)
        outline.closeSubpath()
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor("#dfe6ef"), 1.3))
        painter.drawPath(outline)
        painter.end()
        return self._cache

    def paintEvent(self, _event):
        if self.width() < 24 or self.height() < 24:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.drawPixmap(0, 0, self._background())

        top, right, left = self._triangle_points()
        weights = self._normalised_weights()
        dot = self._point(weights["PASS"] + weights["NEUTRAL"] / 3.0,
                          weights["FAIL"] + weights["NEUTRAL"] / 3.0,
                          weights["BASEPLATE"] + weights["NEUTRAL"] / 3.0,
                          top, right, left)
        colour = self.visual_color()
        painter.setPen(QPen(QColor("#11151b"), 2.8))
        painter.setBrush(QBrush(colour))
        painter.drawEllipse(dot, 6.2, 6.2)
        painter.setPen(QPen(QColor("#ffffff"), 1.1))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(dot, 3.2, 3.2)
        painter.end()


class AutoPage(QWidget):
    CONTROLS_FIXED_W = 430

    def __init__(
        self,
        *,
        engine: QCPreviewEngine,
        cam: OpenCVCamera,
        report_store: Optional[ReportStore] = None,
        plc_service: Optional[InspectionPlcService] = None,
        parent=None,
    ):
        super().__init__(parent)

        self.engine = engine
        self.cam = cam
        self.report_store = report_store
        self.plc_service = plc_service

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
        self._alarm_sound_enabled = True
        self._control_size_percent = 100
        self._reduce_alarm_motion = False
        # Candidate PASS/FAIL states must hold continuously for this duration
        # before they become an operator result or a report event.
        self._decision_confirmation_s = 1.0
        self._decision_candidate_state: Optional[str] = None
        self._decision_candidate_since: Optional[float] = None
        self._decision_confidence_percent = 0
        self._decision_confirmed = False
        # Outcome evidence fades gradually when tracking flickers instead of
        # being thrown away on a single dropped frame.
        self._confidence_scores = {"PASS": 0.0, "FAIL": 0.0, "BASEPLATE": 0.0}
        self._confidence_last_updated: Optional[float] = None
        # A finished inspection is not ready for another glass until the
        # camera has seen a genuinely clear station for this long.  That
        # matches the minimum expected conveyor spacing, so a brief track dropout
        # cannot create a second report row for the same glass.
        self._inspection_rearm_after_s = 5.0
        self._inspection_clear_since: Optional[float] = None
        self._part_seen_since_rearm = False
        self._report_terminal_recorded = False
        self._inspection_cycle_mode = "STOPPED"
        self._last_saved_result: Optional[str] = None
        self._last_saved_cause = ""
        self._last_saved_time = ""
        self._baseplate_missing_since: Optional[float] = None
        self._baseplate_missing_recorded = False
        # This is deliberately separate from the audible/visual alarm.  It
        # lets the operator see the detector evidence even when the alarm is
        # muted or visual alarms are disabled.
        self._baseplate_missing_candidate = False
        self._baseplate_missing_elapsed = 0.0
        self._baseplate_detection_mode = "WAITING"
        self._run_button_pulse_on = False
        self._run_button_timer = QTimer(self)
        self._run_button_timer.setInterval(480)
        self._run_button_timer.timeout.connect(self._toggle_run_button_pulse)
        self._summary_state = "READY"
        self._last_status_detail = "Select a product and press START"
        self._last_plc_publish_error = ""

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        content = QHBoxLayout()
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(12)

        self.view = ImageView()
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content.addWidget(self.view, 1)

        self.controls_panel = QWidget()
        self.controls_panel.setFixedWidth(self.CONTROLS_FIXED_W)
        self.controls_panel.setMinimumWidth(self.CONTROLS_FIXED_W)
        self.controls_panel.setMaximumWidth(self.CONTROLS_FIXED_W)
        self.controls_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        content.addWidget(self.controls_panel, 0)

        panel = QVBoxLayout(self.controls_panel)
        panel.setContentsMargins(0, 0, 0, 0)
        panel.setSpacing(10)

        self.status_summary = QFrame()
        self.status_summary.setObjectName("autoStatusSummary")
        # Status text changes every live frame.  Reserve its complete height
        # so a wrap/state change never resizes the camera feed underneath it.
        self.status_summary.setFixedHeight(184)
        summary_lay = QHBoxLayout(self.status_summary)
        summary_lay.setContentsMargins(24, 14, 24, 14)
        summary_lay.setSpacing(22)

        state_column = QVBoxLayout()
        state_title = QLabel("INSPECTION STATUS")
        state_title.setAlignment(Qt.AlignCenter)
        state_title.setStyleSheet("font-size: 12px; font-weight: 900; color: #c7ccd8;")
        state_column.addWidget(state_title)

        self.lbl_state = QLabel("READY")
        self.lbl_state.setAlignment(Qt.AlignCenter)
        state_column.addWidget(self.lbl_state)
        summary_lay.addLayout(state_column, 2)

        result_column = QVBoxLayout()
        result_title = QLabel("CURRENT RESULT")
        result_title.setAlignment(Qt.AlignCenter)
        result_title.setStyleSheet("font-size: 12px; font-weight: 900; color: #c7ccd8;")
        result_column.addWidget(result_title)

        self.lbl_status = QLabel("Select a product and press START")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setFixedHeight(48)
        result_column.addWidget(self.lbl_status)
        summary_lay.addLayout(result_column, 5)

        limits_column = QVBoxLayout()
        limits_title = QLabel("ACCEPTANCE LIMITS")
        limits_title.setAlignment(Qt.AlignCenter)
        limits_title.setStyleSheet("font-size: 12px; font-weight: 900; color: #c7ccd8;")
        limits_column.addWidget(limits_title)

        self.lbl_tolerance = QLabel("No product loaded")
        self.lbl_tolerance.setAlignment(Qt.AlignCenter)
        self.lbl_tolerance.setWordWrap(True)
        self.lbl_tolerance.setFixedHeight(48)
        self.lbl_tolerance.setStyleSheet(
            "padding: 5px; font-size: 15px; font-weight: 800; color: #dbe4f2;"
        )
        limits_column.addWidget(self.lbl_tolerance)
        summary_lay.addLayout(limits_column, 4)

        confidence_column = QVBoxLayout()
        confidence_title = QLabel("CONFIDENCE")
        confidence_title.setAlignment(Qt.AlignCenter)
        confidence_title.setStyleSheet("font-size: 12px; font-weight: 900; color: #c7ccd8;")
        confidence_column.addWidget(confidence_title)

        self.confidence_triangle = ConfidenceTriangle()
        confidence_column.addWidget(self.confidence_triangle, 1, Qt.AlignHCenter)

        self.lbl_confidence = QLabel("—")
        self.lbl_confidence.setAlignment(Qt.AlignCenter)
        self.lbl_confidence.setWordWrap(True)
        self.lbl_confidence.setFixedHeight(32)
        self.lbl_confidence.setStyleSheet("font-size: 13px; font-weight: 1000; color: #aeb8c8;")
        confidence_column.addWidget(self.lbl_confidence)
        summary_lay.addLayout(confidence_column, 2)

        root.addWidget(self.status_summary)
        root.addLayout(content, 1)

        self.controls_scroll = QScrollArea()
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.controls_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.controls_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

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

        self.chk_show_stab_text = QCheckBox("Show diagnostic text (advanced)")
        self.chk_show_stab_text.setChecked(False)
        vb.addWidget(self.chk_show_stab_text)

        self.chk_show_bp = QCheckBox("Show baseplate contour + center")
        self.chk_show_bp.setChecked(True)
        vb.addWidget(self.chk_show_bp)

        self.chk_alarm = QCheckBox("Enable visual lost-track alarm after 5s")
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

        self.baseplate_detection_panel = QFrame()
        self.baseplate_detection_panel.setObjectName("baseplateDetectionPanel")
        self.baseplate_detection_panel.setMinimumHeight(94)
        detection_lay = QVBoxLayout(self.baseplate_detection_panel)
        detection_lay.setContentsMargins(12, 8, 12, 8)
        detection_lay.setSpacing(3)

        detection_title = QLabel("DETECTION CHECK")
        detection_title.setObjectName("baseplateDetectionTitle")
        detection_lay.addWidget(detection_title)

        self.lbl_notch_detection = QLabel("Glass notch: waiting for fitted lines")
        self.lbl_notch_detection.setObjectName("notchDetectionStatus")
        detection_lay.addWidget(self.lbl_notch_detection)

        self.lbl_baseplate_detection = QLabel("Baseplate: monitoring")
        self.lbl_baseplate_detection.setObjectName("baseplateDetectionStatus")
        detection_lay.addWidget(self.lbl_baseplate_detection)
        vb.addWidget(self.baseplate_detection_panel)

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

        self.btn_start = QPushButton("START")
        self.btn_start.setObjectName("autoStartButton")
        self.btn_stop = QPushButton("STOP")
        self.btn_stop.setObjectName("autoStopButton")
        self.btn_stop.setEnabled(False)

        self.controls_scroll.setWidget(gb)
        panel.addWidget(self.controls_scroll, 1)

        # Keep the production controls outside the optional-overlay scroll
        # area.  Operators can always start/stop and see whether the current
        # glass has already been committed to Reports.
        self.inspection_action_panel = QFrame()
        self.inspection_action_panel.setObjectName("inspectionActionPanel")
        action_lay = QVBoxLayout(self.inspection_action_panel)
        action_lay.setContentsMargins(12, 10, 12, 12)
        action_lay.setSpacing(7)

        action_title = QLabel("INSPECTION CYCLE")
        action_title.setObjectName("inspectionCycleTitle")
        action_lay.addWidget(action_title)

        self.lbl_inspection_cycle = QLabel("STOPPED")
        self.lbl_inspection_cycle.setObjectName("inspectionCycleStatus")
        self.lbl_inspection_cycle.setWordWrap(True)
        action_lay.addWidget(self.lbl_inspection_cycle)

        self.lbl_last_saved_result = QLabel()
        self.lbl_last_saved_result.setObjectName("inspectionCycleSaved")
        self.lbl_last_saved_result.setWordWrap(True)
        self.lbl_last_saved_result.setMinimumHeight(28)
        action_lay.addWidget(self.lbl_last_saved_result)

        self.pb_inspection_rearm = QProgressBar()
        self.pb_inspection_rearm.setRange(0, 100)
        self.pb_inspection_rearm.setValue(0)
        self.pb_inspection_rearm.setTextVisible(False)
        self.pb_inspection_rearm.setFixedHeight(8)
        action_lay.addWidget(self.pb_inspection_rearm)

        self.lbl_inspection_rule = QLabel()
        self.lbl_inspection_rule.setObjectName("inspectionCycleRule")
        self.lbl_inspection_rule.setWordWrap(True)
        action_lay.addWidget(self.lbl_inspection_rule)

        action_buttons = QHBoxLayout()
        action_buttons.setSpacing(10)
        action_buttons.addWidget(self.btn_start)
        action_buttons.addWidget(self.btn_stop)
        action_lay.addLayout(action_buttons)
        panel.addWidget(self.inspection_action_panel, 0)

        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)

        self._last_zoom_crop = None
        self._update_zoom_label()
        self._update_tolerance_label()
        self._set_decision_confidence_display(None, 0, False)
        self._set_summary_state("READY")
        self._set_baseplate_detection_state("WAITING")
        self._set_inspection_cycle("STOPPED")
        self._refresh_run_button_appearance()
        self._apply_control_size()

    def _set_summary_state(self, state: str):
        state = str(state or "READY").upper()
        self._summary_state = state
        styles = {
            "PASS": ("PASS", "#123c27", "#38d27b"),
            "TRACK": ("TRACK", "#5a3414", "#f2a23a"),
            "SEARCH": ("SEARCHING", "#303744", "#aeb8c8"),
            "SETUP": ("SETUP REQUIRED", "#5a4214", "#f2c14e"),
            "FAIL": ("FAIL", "#541c25", "#ff6673"),
            "BASEPLATE_WARNING": ("BASEPLATE CHECK", "#573514", "#f2a23a"),
            "BASEPLATE_MISSING": ("BASEPLATE MISSING", "#551923", "#ff6673"),
            "RUNNING": ("RUNNING", "#173a58", "#66c6ff"),
            "READY": ("READY", "#1f3d2b", "#78d89c"),
            "IDLE": ("IDLE", "#2a2a2e", "#c7ccd8"),
        }
        label, background, border = styles.get(state, styles["READY"])
        # During a live inspection, the ribbon follows the same blended
        # evidence colour as the triangle rather than jumping sharply between
        # fixed PASS/TRACK/FAIL colours.
        if self._running and state in {
            "PASS", "FAIL", "TRACK", "SEARCH", "BASEPLATE_WARNING", "BASEPLATE_MISSING",
        }:
            evidence = self.confidence_triangle.visual_color()
            background_colour = self._blend_colour(QColor("#121a20"), evidence, 0.31)
            background = background_colour.name()
            border = evidence.name()
        self.lbl_state.setText(self._tr(label))
        self.lbl_state.setStyleSheet(f"font-size: 30px; font-weight: 1000; color: {border};")
        self.lbl_status.setStyleSheet("font-size: 16px; font-weight: 800; color: #f3f5f8;")
        self.status_summary.setStyleSheet(
            f"QFrame#autoStatusSummary {{ background: {background}; border: 2px solid {border}; border-radius: 12px; }}"
        )

    @staticmethod
    def _blend_colour(start: QColor, end: QColor, amount: float) -> QColor:
        amount = max(0.0, min(1.0, float(amount)))
        return QColor(
            round(start.red() + (end.red() - start.red()) * amount),
            round(start.green() + (end.green() - start.green()) * amount),
            round(start.blue() + (end.blue() - start.blue()) * amount),
        )

    def _state_from_text(self, text: str) -> str:
        upper = str(text or "").upper()
        if "PASS" in upper:
            return "PASS"
        if "BASEPLATE" in upper and ("MISSING" in upper or "NOT DETECTED" in upper):
            return "BASEPLATE_MISSING"
        if "TRACK" in upper or "VERIFY" in upper:
            return "TRACK"
        if "FAIL" in upper or "ERROR" in upper or "CAMERA" in upper:
            return "FAIL"
        if "SEARCH" in upper:
            return "SEARCH"
        if "SETUP" in upper or "NO PRODUCT" in upper:
            return "SETUP"
        if "RUNNING" in upper:
            return "RUNNING"
        if "IDLE" in upper or "STOPPED" in upper:
            return "IDLE"
        return "READY"

    def _set_status(self, text: str, state: Optional[str] = None):
        raw = str(text or "Status: READY").strip()
        detail = re.sub(r"^status\s*:\s*", "", raw, flags=re.IGNORECASE).strip()
        self._set_summary_state(state or self._state_from_text(detail))
        self._last_status_detail = detail or "Ready for inspection"
        self.lbl_status.setText(self._tr(self._last_status_detail))

    def _display_seconds(self, value: float) -> str:
        formatted = f"{max(0.0, float(value)):.1f}s"
        localizer = getattr(self, "localizer", None)
        return localizer.digits(formatted) if localizer is not None else formatted

    def _set_baseplate_detection_state(self, mode: str, elapsed: float = 0.0):
        """Show the evidence behind the baseplate alarm in the control pane."""
        mode = str(mode or "WAITING").upper()
        elapsed = max(0.0, float(elapsed))
        self._baseplate_detection_mode = mode
        self._baseplate_missing_candidate = mode in {"VERIFYING", "ALARM"}
        self._baseplate_missing_elapsed = elapsed if self._baseplate_missing_candidate else 0.0

        styles = {
            "WAITING": ("#232a34", "#596577", "#aeb8c8"),
            "DETECTED": ("#123c27", "#38d27b", "#d9fbe7"),
            "VERIFYING": ("#4b2c13", "#f2a23a", "#fff0d0"),
            "ALARM": ("#521923", "#ff6673", "#ffe2e6"),
        }
        background, border, text_color = styles.get(mode, styles["WAITING"])
        self.baseplate_detection_panel.setStyleSheet(
            "QFrame#baseplateDetectionPanel { "
            f"background: {background}; border: 1px solid {border}; border-radius: 8px; }} "
            "QLabel#baseplateDetectionTitle { color: #c7ccd8; font-size: 10px; font-weight: 900; } "
            f"QLabel#notchDetectionStatus, QLabel#baseplateDetectionStatus {{ color: {text_color}; font-size: 12px; font-weight: 800; }}"
        )

        if mode == "DETECTED":
            notch = self._tr("Glass notch: CONFIRMED")
            baseplate = self._tr("Baseplate: DETECTED")
        elif mode == "VERIFYING":
            notch = self._tr("Glass notch: CONFIRMED")
            baseplate = (
                f"{self._tr('Baseplate: NOT DETECTED')} · "
                f"{self._tr('Verifying absence')} {self._display_seconds(elapsed)} / "
                f"{self._display_seconds(self._track_alarm_after_s)}"
            )
        elif mode == "ALARM":
            notch = self._tr("Glass notch: CONFIRMED")
            baseplate = (
                f"{self._tr('Baseplate: NOT DETECTED')} · "
                f"{self._tr('ALARM ACTIVE')} {self._display_seconds(elapsed)}"
            )
        else:
            notch = self._tr("Glass notch: waiting for fitted lines")
            baseplate = self._tr("Baseplate: monitoring")

        self.lbl_notch_detection.setText(notch)
        self.lbl_baseplate_detection.setText(baseplate)

    def _set_inspection_cycle(self, mode: str, clear_elapsed: float = 0.0):
        """Make report-save/rearm protection visible to the operator."""
        mode = str(mode or "STOPPED").upper()
        clear_elapsed = max(0.0, float(clear_elapsed))
        self._inspection_cycle_mode = mode

        styles = {
            "STOPPED": ("#252b35", "#667386", "#c7d1df"),
            "WAITING": ("#253344", "#6aa5d8", "#d9efff"),
            "INSPECTING": ("#3a2e16", "#f2a23a", "#fff0d0"),
            "SAVED": ("#123c27", "#38d27b", "#dcffea"),
            "CLEARING": ("#4a3015", "#f2a23a", "#fff0d0"),
            "READY": ("#123c27", "#38d27b", "#dcffea"),
        }
        background, border, text_color = styles.get(mode, styles["STOPPED"])

        # Once a result is committed, the panel's colour follows that saved
        # result until the station has cleared.  This is intentionally
        # separate from the live ribbon, which may already be seeing a new
        # camera state while the previous glass is still latched.
        saved_result = str(self._last_saved_result or "").upper()
        saved_cause = str(self._last_saved_cause or "").upper()
        if mode in {"SAVED", "CLEARING"} and saved_result == "PASS":
            background, border, text_color = "#123c27", "#38d27b", "#dcffea"
        elif mode in {"SAVED", "CLEARING"} and "BASEPLATE" in saved_cause:
            background, border, text_color = "#402b1d", "#a96738", "#ffe0c9"
        elif mode in {"SAVED", "CLEARING"} and saved_result == "FAIL":
            background, border, text_color = "#541c25", "#ff6673", "#fff0f2"

        if mode == "INSPECTING":
            text = self._tr("INSPECTING CURRENT GLASS")
            progress = 0
        elif mode == "SAVED":
            text = self._tr("GLASS COMPLETE · HOLDING CURRENT GLASS")
            progress = 100
        elif mode == "CLEARING":
            text = (
                f"{self._tr('CLEARING STATION')} · "
                f"{self._display_seconds(clear_elapsed)} / {self._display_seconds(self._inspection_rearm_after_s)}"
            )
            progress = int(min(99.0, 100.0 * clear_elapsed / max(0.1, self._inspection_rearm_after_s)))
        elif mode == "READY":
            text = self._tr("READY FOR NEXT GLASS")
            progress = 100
        elif mode == "WAITING":
            text = self._tr("WAITING FOR CAMERA / RECIPE")
            progress = 0
        else:
            text = self._tr("STOPPED")
            progress = 0

        self.inspection_action_panel.setStyleSheet(
            "QFrame#inspectionActionPanel { "
            f"background: {background}; border: 2px solid {border}; border-radius: 10px; }} "
            "QLabel#inspectionCycleTitle { color: #c7d1df; font-size: 10px; font-weight: 1000; } "
            f"QLabel#inspectionCycleStatus {{ color: {text_color}; font-size: 14px; font-weight: 1000; }} "
            f"QLabel#inspectionCycleSaved {{ color: {text_color}; font-size: 12px; font-weight: 900; }} "
            "QLabel#inspectionCycleRule { color: #c7d1df; font-size: 10px; font-weight: 700; }"
        )
        self.lbl_inspection_cycle.setText(text)
        self._update_last_saved_result_label()
        self.lbl_inspection_rule.setText(
            f"{self._tr('Clear-station delay:')} {self._display_seconds(self._inspection_rearm_after_s)} · "
            f"{self._tr('Reports re-arm only when clear')}"
        )
        self.pb_inspection_rearm.setValue(progress)
        self.pb_inspection_rearm.setStyleSheet(
            "QProgressBar { background: #1d222a; border: 1px solid #465263; border-radius: 3px; } "
            f"QProgressBar::chunk {{ background: {border}; border-radius: 3px; }}"
        )

    def _update_last_saved_result_label(self):
        """Show exactly what was committed, not merely that a row was saved."""
        result = str(self._last_saved_result or "").upper()
        if result not in {"PASS", "FAIL"}:
            self.lbl_last_saved_result.setText(self._tr("LAST SAVED: —"))
            return

        details = [self._tr(result)]
        cause = str(self._last_saved_cause or "").strip()
        if cause:
            details.append(self._tr(cause))
        if self._last_saved_time:
            localizer = getattr(self, "localizer", None)
            timestamp = localizer.digits(self._last_saved_time) if localizer is not None else self._last_saved_time
            details.append(timestamp)
        self.lbl_last_saved_result.setText(
            f"{self._tr('LAST SAVED:')} " + " · ".join(details)
        )

    def _mark_result_saved(self, result: str, cause: str = ""):
        self._last_saved_result = str(result or "").upper()
        self._last_saved_cause = str(cause or "").strip()
        self._last_saved_time = time.strftime("%H:%M:%S")

    def _refresh_run_button_appearance(self):
        """Give Start/Stop an unambiguous, high-contrast machine-control state."""
        if self._running:
            green = "#20c46b" if self._run_button_pulse_on else "#0f7b43"
            border = "#79f0aa" if self._run_button_pulse_on else "#29b866"
            self.btn_start.setText(self._tr("RUNNING"))
            self.btn_start.setStyleSheet(
                "QPushButton { "
                f"background: {green}; border: 2px solid {border}; color: #f5fff8; "
                "border-radius: 10px; padding: 10px 12px; font-weight: 1000; } "
                "QPushButton:disabled { color: #f5fff8; }"
            )
        else:
            self.btn_start.setText(self._tr("START"))
            self.btn_start.setStyleSheet(
                "QPushButton { background: #126c3d; border: 2px solid #2dc977; color: #f5fff8; "
                "border-radius: 10px; padding: 10px 12px; font-weight: 1000; } "
                "QPushButton:hover { background: #188a50; } "
                "QPushButton:pressed { background: #0b4e2c; }"
            )

        stop_background = "#9f2835" if self.btn_stop.isEnabled() else "#4f2229"
        stop_border = "#ff7782" if self.btn_stop.isEnabled() else "#7d3a43"
        stop_text = "#fff5f6" if self.btn_stop.isEnabled() else "#c7a6aa"
        self.btn_stop.setText(self._tr("STOP"))
        self.btn_stop.setStyleSheet(
            "QPushButton { "
            f"background: {stop_background}; border: 2px solid {stop_border}; color: {stop_text}; "
            "border-radius: 10px; padding: 10px 12px; font-weight: 1000; } "
            "QPushButton:hover:enabled { background: #c93645; } "
            "QPushButton:pressed:enabled { background: #761c27; }"
        )

    def _toggle_run_button_pulse(self):
        if not self._running:
            self._run_button_timer.stop()
            return
        self._run_button_pulse_on = not self._run_button_pulse_on
        self._refresh_run_button_appearance()

    def _tr(self, text: str) -> str:
        localizer = getattr(self, "localizer", None)
        return localizer.tr(text) if localizer is not None else str(text)

    def _measurement_summary(self, stab_info) -> str:
        if not isinstance(stab_info, dict):
            return ""

        metrics = stab_info.get("inspection_metrics")
        if not isinstance(metrics, (list, tuple)):
            return ""

        values = []
        for metric in metrics[:3]:
            if not isinstance(metric, dict):
                continue
            label = str(metric.get("label", "")).replace(" OFFSET", "")
            value = str(metric.get("value", ""))
            if label and value:
                values.append(f"{label}: {value}")

        return "  |  ".join(values)

    def _set_result_status(self, out, *, confidence_percent: int = 0, decision_confirmed: bool = False):
        state = str(getattr(out, "state", "") or "TRACK").upper()
        stab_info = getattr(out, "stab_info", None)
        summary = self._measurement_summary(stab_info)
        messages = {
            "PASS": "Within recipe limits",
            "TRACK": "Verifying position",
            "SEARCH": "Searching for the baseplate",
            "SETUP": "Enter recipe tolerances in Calibration",
            "FAIL": "Outside acceptance limits",
        }
        display_state = state
        if state == "SEARCH" and self._baseplate_missing_candidate:
            confirmed = self._tr("Glass notch geometry confirmed")
            if self._track_alarm_active:
                detail = (
                    f"{confirmed}\n"
                    f"{self._tr('Baseplate not detected')} · {self._tr('ALARM ACTIVE')} "
                    f"{self._display_seconds(self._baseplate_missing_elapsed)}"
                )
                display_state = "BASEPLATE_MISSING"
            else:
                detail = (
                    f"{confirmed}\n"
                    f"{self._tr('Baseplate not detected')} · {self._tr('Verifying absence')} "
                    f"{self._display_seconds(self._baseplate_missing_elapsed)} / "
                    f"{self._display_seconds(self._track_alarm_after_s)}"
                )
                display_state = "BASEPLATE_WARNING"
        elif state in {"PASS", "FAIL"}:
            percent_text = self._confidence_percent_text(confidence_percent)
            verdict = self._tr(state)
            if decision_confirmed:
                detail = f"{verdict} {self._tr('confirmed')} · {percent_text} {self._tr('confidence')}"
            else:
                detail = f"{verdict} {self._tr('candidate')} · {percent_text} {self._tr('confidence')}"
                display_state = "TRACK"
        else:
            detail = messages.get(state, str(getattr(out, "status_text", "Inspection active") or "Inspection active"))
        if summary:
            detail = f"{detail}\n{summary}"
        self._set_status(detail, state=display_state)

    def _confidence_percent_text(self, percent: int) -> str:
        value = max(0, min(100, int(percent)))
        localizer = getattr(self, "localizer", None)
        return localizer.digits(f"{value}%") if localizer is not None else f"{value}%"

    def _set_decision_confidence_display(self, state: Optional[str], percent: int, confirmed: bool):
        state = str(state or "").upper()
        percent = max(0, min(100, int(percent)))
        self.confidence_triangle.set_scores(self._confidence_scores)
        colour = self.confidence_triangle.visual_color().name()

        labels = {
            "PASS": self._tr("PASS"),
            "FAIL": self._tr("FAIL"),
            "BASEPLATE": self._tr("BASEPLATE NOT FOUND"),
        }
        if state in labels:
            suffix = self._tr("confirmed") if confirmed else self._tr("candidate")
            self.lbl_confidence.setText(
                f"{labels[state]} · {self._confidence_percent_text(percent)}\n{suffix}"
            )
            self.lbl_confidence.setStyleSheet(
                f"font-size: 12px; font-weight: 1000; color: {colour};"
            )
            return

        dominant, retained = self.confidence_triangle.dominant_outcome()
        retained_percent = self._confidence_percent_text(int(round(retained * 100.0)))
        if state == "TRACK":
            self.lbl_confidence.setText(f"{self._tr('TRACK')} · {retained_percent}")
            self.lbl_confidence.setStyleSheet(
                f"font-size: 13px; font-weight: 1000; color: {colour};"
            )
        else:
            self.lbl_confidence.setText("—")
            self.lbl_confidence.setStyleSheet("font-size: 18px; font-weight: 1000; color: #aeb8c8;")

    def _reset_decision_confidence(self, *, clear_evidence: bool = True):
        self._decision_candidate_state = None
        self._decision_candidate_since = None
        self._decision_confidence_percent = 0
        self._decision_confirmed = False
        if clear_evidence:
            self._confidence_scores = {"PASS": 0.0, "FAIL": 0.0, "BASEPLATE": 0.0}
            self._confidence_last_updated = None
        self._set_decision_confidence_display(None, 0, False)

    def _evidence_outcome_from_out(self, out) -> Optional[str]:
        """Classify only live evidence; a brief absence simply decays it."""
        state = str(getattr(out, "state", "") or "").upper()
        if state in {"PASS", "FAIL"}:
            return state
        if state == "SEARCH" and self._glass_present_from_out(out):
            return "BASEPLATE"
        return None

    def _update_decision_confidence(self, out) -> tuple[int, bool]:
        """Update temporal evidence without erasing it on a track flicker."""
        state = str(getattr(out, "state", "") or "").upper()
        now = time.monotonic()
        if self._confidence_last_updated is None:
            elapsed = 0.0
        else:
            # Never apply a long wall-clock pause as a sudden evidence loss.
            elapsed = min(0.25, max(0.0, now - self._confidence_last_updated))
        self._confidence_last_updated = now

        outcome = self._evidence_outcome_from_out(out)
        required = max(0.2, float(self._decision_confirmation_s))
        gain_per_second = 1.0 / required
        decay_per_second = 0.18 if outcome is None else 0.46
        for key, score in self._confidence_scores.items():
            if key == outcome:
                score += gain_per_second * elapsed
            else:
                score -= decay_per_second * elapsed
            self._confidence_scores[key] = max(0.0, min(1.0, score))

        confidence = float(self._confidence_scores.get(outcome, 0.0)) if outcome else 0.0
        confirmed = bool(outcome in {"PASS", "FAIL"} and confidence >= 0.999)
        percent = int(min(100.0, confidence * 100.0))
        if outcome in {"PASS", "FAIL"} and not confirmed:
            percent = min(99, percent)
        self._decision_confidence_percent = percent
        self._decision_confirmed = confirmed
        display_state = outcome or ("TRACK" if state == "TRACK" else None)
        self._set_decision_confidence_display(display_state, percent, confirmed)
        return percent, confirmed

    def _update_tolerance_label(self):
        recipe = getattr(self.engine, "recipe", None)
        cfg = getattr(recipe, "cfg", None)
        if not isinstance(cfg, dict):
            self.lbl_tolerance.setText(self._tr("Enter X, Y, and angle limits in Calibration"))
            return

        tolerance = cfg.get("tolerance_mm")
        tolerance = tolerance if isinstance(tolerance, dict) else None

        # Older recipe files may only have pixel limits. Present them in the
        # operator-facing unit whenever a calibration scale is available.
        if tolerance is None:
            legacy_px = cfg.get("tolerance_px")
            try:
                scale = None
                for key in ("px_per_mm", "baseplate_px_per_mm", "pixels_per_mm", "scale_px_per_mm"):
                    value = float(cfg.get(key))
                    if value > 0:
                        scale = value
                        break
                if scale is None:
                    for key in ("mm_per_px", "baseplate_mm_per_px", "scale_mm_per_px"):
                        value = float(cfg.get(key))
                        if value > 0:
                            scale = 1.0 / value
                            break
                if scale is None and isinstance(cfg.get("baseplate_scale"), dict):
                    scale = float(cfg["baseplate_scale"].get("px_per_mm"))
                if scale is None or scale <= 0:
                    raise ValueError
                tolerance = {
                    "x": float(legacy_px["x"]) / scale,
                    "y": float(legacy_px["y"]) / scale,
                    "angle": float(legacy_px["angle"]),
                }
            except (TypeError, ValueError, KeyError, ZeroDivisionError):
                tolerance = None

        if not isinstance(tolerance, dict):
            self.lbl_tolerance.setText(self._tr("Enter X/Y limits in mm and angle limit in Calibration"))
            return

        try:
            x = float(tolerance["x"])
            y = float(tolerance["y"])
            angle = float(tolerance["angle"])
            if min(x, y, angle) <= 0:
                raise ValueError
            self.lbl_tolerance.setText(
                f"X <= {x:.3f}mm     Y <= {y:.3f}mm\n"
                f"ANGLE <= {angle:.2f}deg"
            )
        except Exception:
            self.lbl_tolerance.setText(self._tr("Enter X/Y limits in mm and angle limit in Calibration"))

    def refresh_recipe_summary(self):
        self._update_tolerance_label()
        if not self._running:
            self._set_status("Status: READY")

    def set_control_size_percent(self, percent: int):
        self._control_size_percent = max(80, min(140, int(percent)))
        self._apply_control_size()

    def control_size_percent(self) -> int:
        return int(self._control_size_percent)

    def set_alarm_sound_enabled(self, enabled: bool):
        self._alarm_sound_enabled = bool(enabled)

    def alarm_sound_enabled(self) -> bool:
        return bool(self._alarm_sound_enabled)

    def set_reduce_alarm_motion(self, enabled: bool):
        self._reduce_alarm_motion = bool(enabled)

    def reduce_alarm_motion(self) -> bool:
        return bool(self._reduce_alarm_motion)

    def set_decision_confirmation_seconds(self, seconds: float):
        self._decision_confirmation_s = max(0.2, min(5.0, float(seconds)))
        self._reset_decision_confidence()

    def decision_confirmation_seconds(self) -> float:
        return float(self._decision_confirmation_s)

    def set_inspection_rearm_seconds(self, seconds: float):
        self._inspection_rearm_after_s = max(1.0, min(15.0, float(seconds)))
        if self._inspection_cycle_mode == "CLEARING":
            elapsed = 0.0 if self._inspection_clear_since is None else time.monotonic() - self._inspection_clear_since
            self._set_inspection_cycle("CLEARING", elapsed)
        else:
            self._set_inspection_cycle(self._inspection_cycle_mode)

    def inspection_rearm_seconds(self) -> float:
        return float(self._inspection_rearm_after_s)

    def _apply_control_size(self):
        percent = int(self._control_size_percent)
        scale = float(percent) / 100.0
        width = max(350, int(round(self.CONTROLS_FIXED_W * scale)))
        button_h = max(34, int(round(36 * scale)))
        font_px = max(12, int(round(12 * scale)))

        self.controls_panel.setFixedWidth(width)
        self.controls_panel.setStyleSheet(
            f"QCheckBox, QLabel {{ font-size: {font_px}px; }} "
            f"QGroupBox {{ font-size: {font_px}px; }} "
            f"QPushButton {{ font-size: {font_px}px; }}"
        )

        for button in (self.btn_start, self.btn_stop, self.btn_reset_alarm):
            button.setMinimumHeight(button_h)

        self.lbl_state.setMinimumHeight(max(38, int(round(42 * scale))))

    def _update_zoom_label(self):
        z = float(self.sld_zoom.value()) / 100.0
        self.lbl_zoom.setText(f"{self._tr('Display zoom:')} {z:.2f}x")

    def retranslate_ui(self):
        """Refresh live Auto labels after a language preference change."""
        localizer = getattr(self, "localizer", None)
        if localizer is None:
            return
        localizer.apply_widget_text(self)
        self._set_summary_state(self._summary_state)
        self.lbl_status.setText(self._tr(self._last_status_detail))
        self._update_zoom_label()
        self._update_tolerance_label()
        self._set_baseplate_detection_state(
            self._baseplate_detection_mode,
            self._baseplate_missing_elapsed,
        )
        self._set_inspection_cycle(self._inspection_cycle_mode)
        self._refresh_run_button_appearance()

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
        self.lbl_alarm.setText(self._tr("Alarm: armed"))
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

        self.lbl_alarm.setText(self._tr("Alarm: manually reset / monitoring again"))
        self.lbl_alarm.setStyleSheet("font-size: 12px; font-weight: 900; color: #66d9ff;")
        self.btn_reset_alarm.setEnabled(False)

    def _track_progress_from_out(self, out) -> int:
        try:
            state = str(getattr(out, "state", "") or "").upper()
        except Exception:
            state = ""

        if state == "PASS":
            return 999

        # FAIL means the part was measured but is outside its recipe limit;
        # it is not a lost camera track. SETUP similarly needs operator setup,
        # not a flashing lost-track warning.
        if state in {"FAIL", "SETUP"}:
            return 999

        if state == "TRACK":
            stab_info = getattr(out, "stab_info", None)
            if isinstance(stab_info, dict):
                stability = stab_info.get("inspection_stability")
                if isinstance(stability, (tuple, list)) and stability:
                    try:
                        return max(1, int(stability[0]))
                    except Exception:
                        pass
            return 1

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
        now = time.monotonic()
        progress = self._track_progress_from_out(out)
        good_track = progress > 0

        if good_track:
            self._reset_alarm_state()
            self._set_baseplate_detection_state("DETECTED")
            return False, 0.0

        # A clear camera view (or an unrelated object such as a sheet of
        # paper) is not a missing-baseplate fault.  Only alarm when the
        # engine has just confirmed the expected fitted notch geometry.
        if not self._glass_present_from_out(out):
            self._reset_alarm_state()
            self._set_baseplate_detection_state("WAITING")
            self.lbl_alarm.setText(self._tr("Alarm: waiting for fitted glass notch"))
            self.lbl_alarm.setStyleSheet("font-size: 12px; font-weight: 800; color: #9a9a9a;")
            return False, 0.0

        if self._track_alarm_since is None:
            self._track_alarm_since = now

        elapsed = now - self._track_alarm_since
        active = self.chk_alarm.isChecked() and elapsed >= float(self._track_alarm_after_s)
        self._track_alarm_active = active

        # The fitted glass notch is a positive glass-presence signal.  Show
        # that evidence before the timer expires, not only once the alarm is
        # already sounding.  This keeps "nothing in view" distinct from a
        # real glass whose baseplate is absent.
        self._set_baseplate_detection_state("ALARM" if active else "VERIFYING", elapsed)

        if not self.chk_alarm.isChecked():
            self.lbl_alarm.setText(self._tr("Alarm: visual alarm disabled · baseplate check visible"))
            self.lbl_alarm.setStyleSheet("font-size: 12px; font-weight: 800; color: #aeb8c8;")
            self.btn_reset_alarm.setEnabled(False)
            return False, elapsed

        if active:
            self.lbl_alarm.setText(
                f"{self._tr('ALARM: baseplate not found')} · {self._display_seconds(elapsed)}"
            )
            self.lbl_alarm.setStyleSheet(
                "background: #5a1d26; border: 1px solid #ff6673; border-radius: 6px; "
                "padding: 7px 8px; font-size: 12px; font-weight: 1000; color: #fff0f2;"
            )
            self.btn_reset_alarm.setEnabled(True)

            if self._alarm_sound_enabled and now - self._track_alarm_last_beep >= 0.80:
                self._track_alarm_last_beep = now
                self._play_track_alarm_sound()
        else:
            remaining = max(0.0, float(self._track_alarm_after_s) - elapsed)

            if self._track_alarm_manual_hold:
                self.lbl_alarm.setText(
                    f"{self._tr('Alarm: reset acknowledged, re-arming in')} {self._display_seconds(remaining)}"
                )
                self.lbl_alarm.setStyleSheet("font-size: 12px; font-weight: 900; color: #66d9ff;")
            else:
                self.lbl_alarm.setText(
                    f"{self._tr('Alarm: baseplate warning in')} {self._display_seconds(remaining)}"
                )
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

        return out_img

    def _draw_track_alarm_overlay(self, img, elapsed_s: float):
        """Draw a concise alarm card after a fitted glass notch loses its baseplate.

        This intentionally says *what was proven* (the notch fit) and *what
        was absent* (the baseplate), instead of showing a generic track-lost
        error that can send an operator to the wrong troubleshooting step.
        """
        if img is None:
            return img

        vis = img.copy()
        H, W = vis.shape[:2]

        self._track_alarm_flash_i += 1
        flash = True if self._reduce_alarm_motion else (self._track_alarm_flash_i // 6) % 2 == 0

        red = np.zeros_like(vis)
        red[:, :, 2] = 255
        alpha = 0.16 if self._reduce_alarm_motion else (0.24 if flash else 0.13)
        vis = cv2.addWeighted(red, alpha, vis, 1.0 - alpha, 0)

        accent = (98, 102, 255) if flash else (72, 82, 210)
        cv2.rectangle(vis, (0, 0), (12, H), accent, -1)
        cv2.rectangle(vis, (W - 12, 0), (W, H), accent, -1)

        box_w = min(max(280, W - 56), 920)
        box_h = 232
        x0 = max(28, (W - box_w) // 2)
        y0 = max(28, int(H * 0.075))
        x1 = x0 + box_w
        y1 = y0 + box_h

        cv2.rectangle(vis, (x0, y0), (x1, y1), (24, 20, 27), -1)
        cv2.rectangle(vis, (x0, y0), (x1, y1), accent, 3, lineType=cv2.LINE_AA)
        cv2.rectangle(vis, (x0 + 3, y0 + 3), (x1 - 3, y0 + 50), (45, 35, 82), -1)

        icon_center = (x0 + 46, y0 + 27)
        cv2.circle(vis, icon_center, 17, (79, 83, 235), -1, lineType=cv2.LINE_AA)
        cv2.putText(vis, "!", (icon_center[0] - 5, icon_center[1] + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        def put_text(text, x, y, scale, color, thickness=2):
            cv2.putText(vis, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
            cv2.putText(vis, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

        put_text("INSPECTION ALARM", x0 + 78, y0 + 34, 0.70, (235, 235, 255), 2)
        put_text("BASEPLATE NOT DETECTED", x0 + 30, y0 + 96, 1.08, (102, 102, 255), 3)
        put_text("NOTCH FIT: CONFIRMED", x0 + 32, y0 + 140, 0.70, (123, 221, 166), 2)
        put_text(f"NO BASEPLATE DETECTION FOR {elapsed_s:.1f} s", x0 + 32, y0 + 174, 0.62, (242, 242, 242), 2)

        footer_y0 = y1 - 42
        cv2.rectangle(vis, (x0 + 3, footer_y0), (x1 - 3, y1 - 3), (33, 29, 43), -1)
        put_text("ACKNOWLEDGE ALARM  •  CHECK CLIP / FIXTURE", x0 + 30, footer_y0 + 27, 0.52, (170, 215, 255), 1)

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

            cv2.putText(vis, "TARGET", (ex + 16, ey - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(vis, "TARGET", (ex + 16, ey - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 0, 255), 1, cv2.LINE_AA)

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
        self._reset_decision_confidence()
        self._report_terminal_recorded = False
        self._inspection_clear_since = None
        self._part_seen_since_rearm = False
        self._last_saved_result = None
        self._last_saved_cause = ""
        self._last_saved_time = ""
        self._baseplate_missing_since = None
        self._baseplate_missing_recorded = False

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._run_button_pulse_on = True
        self._run_button_timer.start()
        self._refresh_run_button_appearance()
        self._set_inspection_cycle("READY")

        self._timer.start(33)
        self._set_status("Status: RUNNING")

    def stop(self):
        if not self._running:
            return

        self._running = False
        self._timer.stop()
        self._run_button_timer.stop()

        self._reset_alarm_state()
        self._reset_decision_confidence()
        self._report_terminal_recorded = False
        self._inspection_clear_since = None
        self._part_seen_since_rearm = False
        self._baseplate_missing_since = None
        self._baseplate_missing_recorded = False

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._run_button_pulse_on = False
        self._refresh_run_button_appearance()
        self._set_inspection_cycle("STOPPED")

        self._set_status("Status: STOPPED")

    def on_tick(self):
        ok, frame = self.cam.read()

        if not ok or frame is None:
            self._reset_decision_confidence()
            self._set_status("Status: CAMERA READ FAIL")
            self._set_inspection_cycle("WAITING")
            self._publish_plc_state(LiveInspectionState.CAMERA_FAULT, "CAMERA READ FAIL")
            return

        if self.engine.recipe is None:
            self._reset_decision_confidence()
            self.view.set_bgr(frame)
            self._set_status("Status: NO PRODUCT LOADED (showing raw feed)")
            self._reset_alarm_state()
            self._set_inspection_cycle("WAITING")
            self._publish_plc_state(LiveInspectionState.SETUP_REQUIRED, "NO PRODUCT LOADED")
            return

        self._set_engine_settings_from_ui()

        try:
            out = self.engine.process_frame(frame)
        except Exception as e:
            self._set_status(f"Status: ENGINE ERROR: {e}")
            self.view.set_bgr(frame)
            self._publish_plc_state(LiveInspectionState.CAMERA_FAULT, "ENGINE ERROR")
            return

        display = out.overlay_bgr if out.overlay_bgr is not None else frame

        # Zoom first so the alarm HUD stays full-screen on top of the zoomed feed.
        display = self._zoom_display_image(display, out)

        confidence_percent, decision_confirmed = self._update_decision_confidence(out)
        alarm_on, alarm_elapsed = self._update_track_alarm(out)
        self._publish_plc_telemetry(out, confidence_percent=confidence_percent)
        # Update the one-glass reporting latch first.  The missing-baseplate
        # monitor needs to know whether this is a new glass or a normal clear
        # interval after a completed inspection.
        self._record_report_result(out, decision_confirmed=decision_confirmed)
        self._record_baseplate_missing(out)
        if alarm_on:
            display = self._draw_track_alarm_overlay(display, alarm_elapsed)

        self.view.set_bgr(display)
        self._set_result_status(
            out,
            confidence_percent=confidence_percent,
            decision_confirmed=decision_confirmed,
        )

    def _publish_plc_state(self, state: LiveInspectionState, text: str) -> None:
        """Replace stale HMI telemetry when the camera/recipe is unavailable."""
        if self.plc_service is None:
            return
        try:
            self.plc_service.update_live_telemetry(InspectionTelemetry(state=state, state_text=text))
            self._last_plc_publish_error = ""
        except Exception as exc:
            self._log_plc_publish_error("telemetry", exc)

    def _publish_plc_telemetry(self, out, *, confidence_percent: int) -> None:
        """Publish display telemetry without blocking the Qt camera timer."""
        if self.plc_service is None:
            return
        try:
            self.plc_service.update_live_telemetry(
                build_live_telemetry(
                    out,
                    confidence_percent=confidence_percent,
                    glass_present=self._glass_present_from_out(out),
                    # A muted visual alarm must not make the PLC/HMI forget
                    # that the fitted-notch gate confirmed a real, persistent
                    # no-baseplate condition.
                    baseplate_alarm_active=(
                        self._baseplate_missing_candidate
                        and self._baseplate_missing_elapsed >= float(self._track_alarm_after_s)
                    ),
                )
            )
            self._last_plc_publish_error = ""
        except Exception as exc:
            self._log_plc_publish_error("telemetry", exc)

    def _publish_terminal_result(
        self,
        out,
        *,
        outcome: InspectionOutcome,
        recipe: str,
        report_event_id: int,
        failure_cause: str = "",
    ) -> None:
        """Offer one durable result to the PLC acknowledgement handshake.

        This never writes a conveyor command.  The PLC must acknowledge this
        exact sequence and enforce its own safety and station-release logic.
        """
        if self.plc_service is None:
            return
        try:
            decision = build_decision_from_output(
                out,
                outcome=outcome,
                recipe=recipe,
                report_event_id=report_event_id,
                failure_cause=failure_cause,
                confidence_percent=self._decision_confidence_percent,
            )
            self.plc_service.submit_terminal_result(decision)
            self._last_plc_publish_error = ""
        except PendingDecisionError as exc:
            self._log_plc_publish_error("result pending", exc)
        except Exception as exc:
            self._log_plc_publish_error("result", exc)

    def _log_plc_publish_error(self, context: str, exc: Exception) -> None:
        """Avoid flooding the terminal if an optional transport is unhealthy."""
        message = f"{context}: {exc}"
        if message != self._last_plc_publish_error:
            print(f"[HMI] PLC {message}")
            self._last_plc_publish_error = message

    def _plc_acknowledgement_pending(self) -> bool:
        """Keep a completed station held until its PLC result is acknowledged."""
        if self.plc_service is None:
            return False
        try:
            return bool(self.plc_service.status().awaiting_acknowledgement)
        except Exception as exc:
            # An enabled handoff whose status cannot be read must fail closed:
            # keep the one-glass latch armed instead of accepting another part.
            self._log_plc_publish_error("status", exc)
            return bool(getattr(getattr(self.plc_service, "config", None), "enabled", False))

    def _glass_present_from_out(self, out) -> bool:
        """Return whether the live notch geometry confirms a glass is present.

        Baseplate detection alone cannot distinguish an empty conveyor from a
        glass whose baseplate is absent.  The registered notch frame is a
        separate signal generated from the glass contour and fitted notch
        lines, so it provides that distinction.
        """
        state = str(getattr(out, "state", "") or "").upper()
        if state in {"TRACK", "PASS", "FAIL"}:
            return True

        stab_info = getattr(out, "stab_info", None)
        if not isinstance(stab_info, dict):
            return False

        if not bool(stab_info.get("glass_presence_confirmed")):
            return False

        try:
            checked_frame = int(stab_info["glass_presence_checked_frame"])
            engine_frame = int(stab_info["engine_frame_index"])
            cadence = max(1, int(getattr(self.engine.settings, "stab_every_n", 6)))
        except Exception:
            return False

        # A stabilizer fit is normally renewed every ``cadence`` frames.  Give
        # it one extra frame, then fail closed instead of using stale geometry.
        return 0 <= engine_frame - checked_frame <= cadence + 1

    def _record_baseplate_missing(self, out):
        """Record one missing-baseplate failure for a glass that is present."""
        state = str(getattr(out, "state", "") or "").upper()
        if (
            state != "SEARCH"
            or not self._glass_present_from_out(out)
            or self._report_terminal_recorded
        ):
            self._baseplate_missing_since = None
            self._baseplate_missing_recorded = False
            return

        now = time.monotonic()
        if self._baseplate_missing_since is None:
            self._baseplate_missing_since = now

        if (
            self.report_store is None
            or self._baseplate_missing_recorded
            or now - self._baseplate_missing_since < float(self._track_alarm_after_s)
        ):
            return

        recipe = getattr(getattr(self.engine, "recipe", None), "name", "")
        try:
            report_event_id = self.report_store.record_result(
                result="FAIL",
                recipe=recipe,
                cause="BASEPLATE NOT FOUND",
                metrics=[{"label": "BASEPLATE NOT FOUND", "passed": False}],
            )
            self._baseplate_missing_recorded = True
            self._report_terminal_recorded = True
            self._mark_result_saved("FAIL", "BASEPLATE NOT FOUND")
            self._set_inspection_cycle("SAVED")
            self._publish_terminal_result(
                out,
                outcome=InspectionOutcome.BASEPLATE_NOT_FOUND,
                recipe=recipe,
                report_event_id=report_event_id,
                failure_cause="BASEPLATE NOT FOUND",
            )
        except Exception:
            # Reporting must never interrupt the live inspection loop.
            pass

    def _record_report_result(self, out, *, decision_confirmed: bool = False):
        """Persist one event per glass, re-arming only after a clear station."""
        state = str(getattr(out, "state", "") or "").upper()
        if state == "SEARCH":
            if self._glass_present_from_out(out):
                # Glass geometry is still visible, so SEARCH means the
                # baseplate detector dropped out rather than that the glass
                # has left the conveyor.
                self._part_seen_since_rearm = True
                self._inspection_clear_since = None
                self._set_inspection_cycle("SAVED" if self._report_terminal_recorded else "INSPECTING")
                return

            now = time.monotonic()
            if self._inspection_clear_since is None:
                self._inspection_clear_since = now

            if self._report_terminal_recorded and self._plc_acknowledgement_pending():
                # The camera may already be clear, but a physical station must
                # not be released/re-armed until the PLC has accepted the same
                # terminal result sequence.
                self._set_inspection_cycle("SAVED")
                return

            # Do not re-arm on a one- or two-frame SEARCH flicker.  A new
            # glass cannot arrive within the configured clear-station gap.
            if (
                self._report_terminal_recorded
                and now - self._inspection_clear_since >= self._inspection_rearm_after_s
            ):
                self._report_terminal_recorded = False
                self._part_seen_since_rearm = False
                self._set_inspection_cycle("READY")
            elif self._report_terminal_recorded:
                self._set_inspection_cycle("CLEARING", now - self._inspection_clear_since)
            else:
                self._set_inspection_cycle("READY")
            return

        self._inspection_clear_since = None
        if state in {"TRACK", "PASS", "FAIL"}:
            self._part_seen_since_rearm = True

        if not self._report_terminal_recorded and state in {"TRACK", "PASS", "FAIL"}:
            self._set_inspection_cycle("INSPECTING")

        if state not in {"PASS", "FAIL"}:
            # Keep the latch during TRACK so one glass does not create multiple
            # rows when its result briefly jitters between PASS and TRACK.
            return

        if not decision_confirmed:
            return

        if self._report_terminal_recorded:
            self._set_inspection_cycle("SAVED")
            return

        if self.report_store is None:
            return

        stab_info = getattr(out, "stab_info", None)
        metrics = stab_info.get("inspection_metrics") if isinstance(stab_info, dict) else None
        failed_labels = []
        if isinstance(metrics, (list, tuple)):
            for metric in metrics:
                if isinstance(metric, dict) and metric.get("passed") is False:
                    label = str(metric.get("label", "")).replace(" OFFSET", "").strip()
                    if label:
                        failed_labels.append(label)

        if state == "FAIL":
            cause = " + ".join(dict.fromkeys(failed_labels)) or "Outside recipe limits"
        else:
            cause = ""

        recipe = getattr(getattr(self.engine, "recipe", None), "name", "")
        try:
            report_event_id = self.report_store.record_result(
                result=state,
                recipe=recipe,
                cause=cause,
                metrics=metrics or [],
            )
            self._report_terminal_recorded = True
            self._mark_result_saved(state, cause)
            self._set_inspection_cycle("SAVED")
            self._publish_terminal_result(
                out,
                outcome=InspectionOutcome.PASS if state == "PASS" else InspectionOutcome.FAIL,
                recipe=recipe,
                report_event_id=report_event_id,
                failure_cause=cause,
            )
        except Exception:
            # Reporting must never interrupt the live inspection loop.
            pass

    def close(self):
        try:
            self.stop()
        except Exception:
            pass
