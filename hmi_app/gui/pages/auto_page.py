from __future__ import annotations

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
)

from hmi_app.gui.image_view import ImageView
from hmi_app.io.camera import OpenCVCamera
from hmi_app.core.engine import QCPreviewEngine


class AutoPage(QWidget):
    CONTROLS_FIXED_W = 420

    def __init__(self, *, engine: QCPreviewEngine, cam: OpenCVCamera, parent=None):
        super().__init__(parent)

        self.engine = engine
        self.cam = cam  # shared camera from MainWindow

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        # Video
        self.view = ImageView()
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self.view, 1)

        # Controls
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
        vb.setSpacing(7)

        # ----------------------------
        # Overlay toggles
        # ----------------------------
        self.chk_show_stab = QCheckBox("Show stabilizer overlay")
        self.chk_show_stab.setChecked(True)
        vb.addWidget(self.chk_show_stab)

        self.chk_stab_search = QCheckBox("  Search ROI box")
        self.chk_stab_search.setChecked(True)
        vb.addWidget(self.chk_stab_search)

        self.chk_stab_points = QCheckBox("  Notch / solid-edge points")
        self.chk_stab_points.setChecked(True)
        vb.addWidget(self.chk_stab_points)

        self.chk_stab_anchors = QCheckBox("  Anchor model")
        self.chk_stab_anchors.setChecked(True)
        vb.addWidget(self.chk_stab_anchors)

        self.chk_stab_legacy = QCheckBox("  Legacy dot/line fallback debug")
        self.chk_stab_legacy.setChecked(False)
        vb.addWidget(self.chk_stab_legacy)

        self.chk_stab_text = QCheckBox("  Stabilizer text")
        self.chk_stab_text.setChecked(True)
        vb.addWidget(self.chk_stab_text)

        self.chk_show_bp = QCheckBox("Show baseplate contour + center")
        self.chk_show_bp.setChecked(True)
        vb.addWidget(self.chk_show_bp)

        # ----------------------------
        # Sliders
        # ----------------------------
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

        # ----------------------------
        # Status/buttons
        # ----------------------------
        self.lbl_status = QLabel("Status: IDLE")
        self.lbl_status.setStyleSheet("font-size: 14px; font-weight: 800;")
        self.lbl_status.setWordWrap(False)
        self.lbl_status.setFixedHeight(28)
        vb.addWidget(self.lbl_status)

        self.btn_start = QPushButton("START")
        self.btn_stop = QPushButton("STOP")
        self.btn_stop.setEnabled(False)

        vb.addWidget(self.btn_start)
        vb.addWidget(self.btn_stop)

        panel.addWidget(gb)
        panel.addStretch(1)

        self._running = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.on_tick)

        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)

        # Keep children enabled/disabled with master stabilizer checkbox.
        self.chk_show_stab.toggled.connect(self._sync_stab_layer_enabled)
        self._sync_stab_layer_enabled(self.chk_show_stab.isChecked())

    def _sync_stab_layer_enabled(self, enabled: bool):
        for w in (
            self.chk_stab_search,
            self.chk_stab_points,
            self.chk_stab_anchors,
            self.chk_stab_legacy,
            self.chk_stab_text,
        ):
            w.setEnabled(bool(enabled))

    def _set_status(self, text: str):
        fm = QFontMetrics(self.lbl_status.font())
        self.lbl_status.setText(fm.elidedText(text, Qt.ElideRight, self.lbl_status.width()))

    def start(self):
        if self._running:
            return

        self._running = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._timer.start(33)
        self._set_status("Status: RUNNING")

    def stop(self):
        if not self._running:
            return

        self._running = False
        self._timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._set_status("Status: STOPPED")

    def _push_settings_to_engine(self):
        s = self.engine.settings

        s.stab_every_n = int(self.sld_stab_n.value())
        s.search_padding_px = int(self.sld_pad.value())

        s.show_stab = bool(self.chk_show_stab.isChecked())
        s.show_baseplate = bool(self.chk_show_bp.isChecked())

        s.show_stab_search_roi = bool(self.chk_stab_search.isChecked())
        s.show_stab_feature_points = bool(self.chk_stab_points.isChecked())
        s.show_stab_anchors = bool(self.chk_stab_anchors.isChecked())
        s.show_stab_legacy = bool(self.chk_stab_legacy.isChecked())
        s.show_stab_text = bool(self.chk_stab_text.isChecked())

    def on_tick(self):
        ok, frame = self.cam.read()

        if not ok or frame is None:
            self._set_status("Status: CAMERA READ FAIL")
            return

        if self.engine.recipe is None:
            self.view.set_bgr(frame)
            self._set_status("Status: NO RECIPE LOADED (showing raw feed)")
            return

        self._push_settings_to_engine()

        out = self.engine.process_frame(frame)

        self.view.set_bgr(out.overlay_bgr)
        self._set_status(out.status_text)

    def close(self):
        # camera owned by MainWindow; do not release here
        try:
            self.stop()
        except Exception:
            pass