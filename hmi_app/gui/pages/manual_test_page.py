from __future__ import annotations

from typing import Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QRadioButton, QButtonGroup,
    QSizePolicy
)

from hmi_app.gui.image_view import ImageView
from hmi_app.io.camera import OpenCVCamera
from hmi_app.core.engine import QCPreviewEngine
from hmi_app.core.recipe_manager import RecipeManager


class ManualTestPage(QWidget):
    """
    One-shot test page:
      - Choose config for current recipe
      - Grab frame
      - Run pipeline once (same engine + detector)
      - Toggle RAW / PROC / OVERLAY display
    """

    CONTROLS_FIXED_W = 420

    def __init__(
        self,
        *,
        engine: QCPreviewEngine,
        recipe_manager: RecipeManager,
        cam: Optional[OpenCVCamera] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName("ManualTestPage")
        self.engine = engine
        self.recipe_manager = recipe_manager
        self.cam = cam
        self._owns_cam = cam is None

        if self.cam is None:
            self.cam = OpenCVCamera(index=0, width=1280, height=720, fps=30, use_dshow=True)

        self._current_recipe_name: str = ""
        self._current_config_name: Optional[str] = None
        self._last_frame_bgr = None
        self._last_out = None

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        # Left: image
        self.view = ImageView()
        root.addWidget(self.view, 1)

        # Right: controls (fixed width to avoid breathing)
        panel = QWidget()
        panel.setFixedWidth(int(self.CONTROLS_FIXED_W))
        panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        panel_lay = QVBoxLayout(panel)
        panel_lay.setContentsMargins(0, 0, 0, 0)
        panel_lay.setSpacing(10)
        root.addWidget(panel, 0)

        gb = QGroupBox("Manual Test")
        vb = QVBoxLayout(gb)
        vb.setSpacing(10)

        # Config select
        vb.addWidget(QLabel("Config:"))
        self.cmb_config = QComboBox()
        vb.addWidget(self.cmb_config)

        # View mode
        vb.addWidget(QLabel("View:"))
        self.rb_raw = QRadioButton("RAW")
        self.rb_proc = QRadioButton("PROC")
        self.rb_ovl = QRadioButton("OVERLAY")
        self.rb_ovl.setChecked(True)

        self.grp_view = QButtonGroup(self)
        for rb in (self.rb_raw, self.rb_proc, self.rb_ovl):
            self.grp_view.addButton(rb)
            vb.addWidget(rb)

        # Actions
        self.btn_grab = QPushButton("GRAB FRAME")
        self.btn_run = QPushButton("RUN ONCE")
        self.btn_run.setMinimumHeight(52)
        vb.addWidget(self.btn_grab)
        vb.addWidget(self.btn_run)

        self.lbl_status = QLabel("Status: IDLE")
        self.lbl_status.setStyleSheet("font-size: 12pt; font-weight: 800;")
        vb.addWidget(self.lbl_status)

        self.lbl_metrics = QLabel("dx/dy/dθ: —")
        self.lbl_metrics.setStyleSheet("font-size: 11pt; font-weight: 650;")
        vb.addWidget(self.lbl_metrics)

        panel_lay.addWidget(gb)
        panel_lay.addStretch(1)

        # wiring
        self.cmb_config.currentTextChanged.connect(self._on_config_changed)
        self.grp_view.buttonClicked.connect(self._apply_view_mode)
        self.btn_grab.clicked.connect(self.grab_frame)
        self.btn_run.clicked.connect(self.run_once)

        self._apply_view_mode()

    # Called by MainWindow when top recipe changes
    def set_recipe_name(self, recipe_name: str):
        self._current_recipe_name = (recipe_name or "").strip()
        self._reload_configs()

    def _reload_configs(self):
        self.cmb_config.blockSignals(True)
        try:
            self.cmb_config.clear()

            if not self._current_recipe_name:
                self.lbl_status.setText("Status: NO RECIPE")
                return

            configs = self.recipe_manager.list_configs(self._current_recipe_name)
            if configs:
                for c in configs:
                    self.cmb_config.addItem(c)

                # default to active if exists
                active = self.recipe_manager.get_active_config_name(self._current_recipe_name)
                if active and active in configs:
                    self.cmb_config.setCurrentText(active)
                else:
                    self.cmb_config.setCurrentIndex(0)
            else:
                # legacy mode
                self.cmb_config.addItem("legacy")
                self.cmb_config.setCurrentIndex(0)

        finally:
            self.cmb_config.blockSignals(False)

        self._current_config_name = self.cmb_config.currentText()
        self.lbl_status.setText("Status: READY")

    def _on_config_changed(self, name: str):
        self._current_config_name = (name or "").strip() or None
        if self._current_recipe_name and self._current_config_name and self._current_config_name != "legacy":
            # persist active config selection
            self.recipe_manager.set_active_config_name(self._current_recipe_name, self._current_config_name)

        # Try to load recipe+config into engine immediately
        try:
            if not self._current_recipe_name:
                return
            if self._current_config_name and self._current_config_name != "legacy":
                recipe = self.recipe_manager.load(self._current_recipe_name, config_name=self._current_config_name)
            else:
                recipe = self.recipe_manager.load(self._current_recipe_name)
            self.engine.set_recipe(recipe)
            self.lbl_status.setText("Status: READY")
        except Exception as e:
            self.lbl_status.setText(f"Status: CONFIG LOAD FAIL: {e}")

    def _apply_view_mode(self):
        if self.rb_raw.isChecked():
            self.engine.settings.view_mode = "RAW"
            self.engine.settings.compute_proc = False
        elif self.rb_proc.isChecked():
            self.engine.settings.view_mode = "PROC"
            self.engine.settings.compute_proc = True
        else:
            self.engine.settings.view_mode = "OVERLAY"
            self.engine.settings.compute_proc = False

        self._refresh_view_from_last()

    def _refresh_view_from_last(self):
        if self._last_out is None:
            return
        if self.engine.settings.view_mode == "RAW" and self._last_out.raw_bgr is not None:
            self.view.set_bgr(self._last_out.raw_bgr)
        elif self.engine.settings.view_mode == "PROC" and self._last_out.proc_bgr is not None:
            self.view.set_bgr(self._last_out.proc_bgr)
        else:
            self.view.set_bgr(self._last_out.overlay_bgr)

    def grab_frame(self):
        ok, frame = self.cam.read()
        if not ok or frame is None:
            self.lbl_status.setText("Status: CAMERA READ FAIL")
            return
        self._last_frame_bgr = frame
        self.lbl_status.setText("Status: FRAME GRABBED")
        # show grabbed raw immediately
        self.view.set_bgr(frame)

    def run_once(self):
        if self._last_frame_bgr is None:
            self.grab_frame()
            if self._last_frame_bgr is None:
                return

        if self.engine.recipe is None:
            self.lbl_status.setText("Status: NO RECIPE LOADED")
            return

        out = self.engine.process_frame(self._last_frame_bgr)
        self._last_out = out
        self.lbl_status.setText(out.status_text)

        # metrics
        if out.center_rel is not None and self.engine.recipe is not None:
            cfg = self.engine.recipe.cfg
            gc = tuple(cfg.get("expected_center", (0.0, 0.0)))
            ga = float(cfg.get("expected_angle", 0.0))
            dx = abs(float(out.center_rel[0]) - float(gc[0]))
            dy = abs(float(out.center_rel[1]) - float(gc[1]))
            dth = abs(float(out.angle or 0.0) - ga)
            self.lbl_metrics.setText(f"dx/dy/dθ: {dx:.2f} / {dy:.2f} / {dth:.2f}")
        else:
            self.lbl_metrics.setText("dx/dy/dθ: —")

        self._refresh_view_from_last()

    def close(self):
        try:
            if self._owns_cam and self.cam is not None:
                self.cam.release()
        except Exception:
            pass
