from __future__ import annotations

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QLabel, QPushButton, QCheckBox, QSlider

from hmi_app.gui.image_view import ImageView
from hmi_app.io.camera import OpenCVCamera
from hmi_app.core.engine import QCPreviewEngine


class AutoPage(QWidget):
    def __init__(self, engine: QCPreviewEngine, cam: OpenCVCamera | None = None, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.cam = cam
        self._owns_cam = cam is None

        if self.cam is None:
            self.cam = OpenCVCamera(index=0, width=1280, height=720, fps=30, use_dshow=True)

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        self.view = ImageView()
        root.addWidget(self.view, 1)

        panel = QVBoxLayout()
        panel.setSpacing(10)
        root.addLayout(panel)

        gb = QGroupBox("Auto Controls")
        vb = QVBoxLayout(gb)

        self.chk_show_stab = QCheckBox("Show ROI-stabilizer debug (lines/points/anchors)")
        self.chk_show_stab.setChecked(True)
        vb.addWidget(self.chk_show_stab)

        self.chk_show_bp = QCheckBox("Show baseplate contour + center")
        self.chk_show_bp.setChecked(True)
        vb.addWidget(self.chk_show_bp)

        vb.addWidget(QLabel("Stabilize cadence (every N frames)"))
        self.sld_stab_n = QSlider(Qt.Horizontal)
        self.sld_stab_n.setMinimum(1)
        self.sld_stab_n.setMaximum(20)
        self.sld_stab_n.setValue(6)
        vb.addWidget(self.sld_stab_n)

        vb.addWidget(QLabel("Search padding (px)"))
        self.sld_pad = QSlider(Qt.Horizontal)
        self.sld_pad.setMinimum(0)
        self.sld_pad.setMaximum(250)
        self.sld_pad.setValue(120)
        vb.addWidget(self.sld_pad)

        self.lbl_status = QLabel("Status: IDLE")
        self.lbl_status.setStyleSheet("font-size: 14px; font-weight: 800;")
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

    def start(self):
        if self._running:
            return
        self._running = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._timer.start(33)
        self.lbl_status.setText("Status: RUNNING")

    def stop(self):
        if not self._running:
            return
        self._running = False
        self._timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_status.setText("Status: STOPPED")

    def on_tick(self):
        ok, frame = self.cam.read()
        if not ok or frame is None:
            self.lbl_status.setText("Status: CAMERA READ FAIL")
            return

        # If no recipe loaded, just show raw feed (fast) + don’t run pipeline
        if self.engine.recipe is None:
            self.view.set_bgr(frame)
            self.lbl_status.setText("Status: NO RECIPE LOADED (showing raw feed)")
            return

        self.engine.settings.stab_every_n = int(self.sld_stab_n.value())
        self.engine.settings.search_padding_px = int(self.sld_pad.value())
        self.engine.settings.show_stab = bool(self.chk_show_stab.isChecked())
        self.engine.settings.show_baseplate = bool(self.chk_show_bp.isChecked())

        out = self.engine.process_frame(frame)
        self.view.set_bgr(out.overlay_bgr)
        self.lbl_status.setText(out.status_text)

    def close(self):
        try:
            if self._owns_cam and self.cam is not None:
                self.cam.release()
        except Exception:
            pass
