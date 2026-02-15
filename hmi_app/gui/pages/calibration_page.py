from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Optional

import cv2

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox,
    QRadioButton, QButtonGroup, QSizePolicy, QScrollArea
)

from hmi_app.gui.image_view import ImageView
from hmi_app.io.camera import OpenCVCamera
from hmi_app.core.engine import QCPreviewEngine
from hmi_app.core.recipe_manager import RecipeManager


def _safe_write_json(path: str, data: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    os.replace(tmp, path)


def _lab(txt: str) -> QLabel:
    l = QLabel(txt)
    l.setStyleSheet("font-size: 12px; font-weight: 800;")
    return l


class CalibrationPage(QWidget):
    """
    Calibration:
      - config dropdown + create/duplicate/active
      - capture golden into config folder
      - tuning saved into golden_config.json
      - live preview RAW/PROC/OVERLAY (PROC uses engine.proc_bgr)
    """

    # Right panel sizing (responsive but clamped)
    RIGHT_W_FRAC = 0.30
    RIGHT_W_MIN = 420
    RIGHT_W_MAX = 620

    # Input safety (prevents eliding "...")
    INPUT_MIN_W = 180

    # Panel background color (match dark theme)
    PANEL_BG = "#0f1014"
    INPUT_BG = "#121318"
    BORDER = "#2c2f3a"
    FOCUS = "#4a78ff"
    TEXT = "#f3f3f3"
    LABEL = "#d7d7db"

    def __init__(
        self,
        *,
        engine: QCPreviewEngine,
        recipe_manager: RecipeManager,
        cam: Optional[OpenCVCamera] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.engine = engine
        self.recipe_manager = recipe_manager
        self.cam = cam
        self._owns_cam = cam is None

        if self.cam is None:
            self.cam = OpenCVCamera(index=0, width=1280, height=720, fps=30, use_dshow=True)

        self._current_recipe_name: str = ""
        self._last_frame_bgr = None

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick_preview)

        # Root layout
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        # Left: video/image view
        self.view = ImageView()
        self.view.setMinimumWidth(240)
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self.view, 1)

        # Right: scroll area
        self.controls_container = QWidget()
        self.controls_layout = QVBoxLayout(self.controls_container)
        self.controls_layout.setContentsMargins(10, 10, 10, 10)
        self.controls_layout.setSpacing(10)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll.setWidget(self.controls_container)
        self.scroll.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        root.addWidget(self.scroll, 0)

        # Apply styling: (A) dark scroll viewport + container background
        self.scroll.setStyleSheet(f"""
            QScrollArea {{
                background: transparent;
                border: none;
            }}
            QScrollArea > QWidget > QWidget {{
                background: {self.PANEL_BG};
            }}
        """)
        self.controls_container.setStyleSheet(f"background: {self.PANEL_BG};")

        # Build UI
        self._build_config_group()
        self._build_preview_group()
        self._build_tuning_group()

        self.controls_layout.addStretch(1)

        # Apply input/label styling ONLY inside controls container
        # (Do NOT change generic QWidget background here.)
        self.controls_container.setStyleSheet(self.controls_container.styleSheet() + f"""
            QLabel {{
                color: {self.LABEL};
            }}

            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
                background: {self.INPUT_BG};
                color: {self.TEXT};
                border: 1px solid {self.BORDER};
                border-radius: 6px;
                padding: 6px 8px;
                font-size: 13px;
                selection-background-color: {self.FOCUS};
                selection-color: #ffffff;
            }}
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
                border: 1px solid {self.FOCUS};
            }}
            QComboBox QAbstractItemView {{
                background: {self.INPUT_BG};
                color: {self.TEXT};
                selection-background-color: {self.FOCUS};
                selection-color: #ffffff;
            }}
        """)

        # Initial sizing
        self._update_right_width()

        # Default view mode
        self._apply_view_mode()

    # -------------------------
    # Build UI sections
    # -------------------------
    def _build_config_group(self):
        gb = QGroupBox("Config Management")
        gb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        vb = QVBoxLayout(gb)
        vb.setSpacing(8)

        vb.addWidget(_lab("Config:"))
        self.cmb_config = QComboBox()
        self.cmb_config.setMinimumWidth(self.INPUT_MIN_W)
        vb.addWidget(self.cmb_config)

        row = QHBoxLayout()
        self.ed_new_name = QLineEdit()
        self.ed_new_name.setPlaceholderText("new_config_name")
        self.ed_new_name.setMinimumWidth(self.INPUT_MIN_W)
        self.btn_create = QPushButton("CREATE")
        row.addWidget(self.ed_new_name, 1)
        row.addWidget(self.btn_create, 0)
        vb.addLayout(row)

        self.btn_dup = QPushButton("DUPLICATE FROM SELECTED")
        vb.addWidget(self.btn_dup)

        self.btn_set_active = QPushButton("SET AS ACTIVE")
        vb.addWidget(self.btn_set_active)

        self.btn_capture = QPushButton("CAPTURE GOLDEN (save)")
        self.btn_capture.setMinimumHeight(52)
        vb.addWidget(self.btn_capture)

        self.lbl_cfg_status = QLabel("Status: IDLE")
        self.lbl_cfg_status.setStyleSheet("font-size: 13px; font-weight: 900;")
        vb.addWidget(self.lbl_cfg_status)

        self.controls_layout.addWidget(gb)

        # wiring
        self.cmb_config.currentTextChanged.connect(self._on_config_changed)
        self.btn_create.clicked.connect(self.create_new_config)
        self.btn_dup.clicked.connect(self.duplicate_selected_config)
        self.btn_set_active.clicked.connect(self.set_active_config)
        self.btn_capture.clicked.connect(self.capture_golden)

    def _build_preview_group(self):
        gb = QGroupBox("Preview")
        gb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        vb = QVBoxLayout(gb)
        vb.setSpacing(6)

        self.rb_raw = QRadioButton("RAW")
        self.rb_proc = QRadioButton("PROC")
        self.rb_ovl = QRadioButton("OVERLAY")
        self.rb_ovl.setChecked(True)

        self.grp_view = QButtonGroup(self)
        for rb in (self.rb_raw, self.rb_proc, self.rb_ovl):
            self.grp_view.addButton(rb)
            vb.addWidget(rb)

        self.btn_preview_start = QPushButton("START LIVE PREVIEW")
        self.btn_preview_stop = QPushButton("STOP PREVIEW")
        self.btn_preview_stop.setEnabled(False)

        vb.addWidget(self.btn_preview_start)
        vb.addWidget(self.btn_preview_stop)

        self.controls_layout.addWidget(gb)

        # wiring
        self.grp_view.buttonClicked.connect(self._apply_view_mode)
        self.btn_preview_start.clicked.connect(self.start_preview)
        self.btn_preview_stop.clicked.connect(self.stop_preview)

    def _build_tuning_group(self):
        gb = QGroupBox("Preprocess Tuning (saved to golden_config.json)")
        gb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        vb = QVBoxLayout(gb)
        vb.setSpacing(8)

        vb.addWidget(_lab("canny_low"))
        self.sp_canny_low = QSpinBox()
        self.sp_canny_low.setRange(0, 255)
        self.sp_canny_low.setMinimumWidth(self.INPUT_MIN_W)
        vb.addWidget(self.sp_canny_low)

        vb.addWidget(_lab("canny_high"))
        self.sp_canny_high = QSpinBox()
        self.sp_canny_high.setRange(0, 255)
        self.sp_canny_high.setMinimumWidth(self.INPUT_MIN_W)
        vb.addWidget(self.sp_canny_high)

        vb.addWidget(_lab("blur_ksize (odd)"))
        self.sp_blur = QSpinBox()
        self.sp_blur.setRange(3, 31)
        self.sp_blur.setSingleStep(2)
        self.sp_blur.setMinimumWidth(self.INPUT_MIN_W)
        vb.addWidget(self.sp_blur)

        vb.addWidget(_lab("clahe_clip"))
        self.sp_clahe = QDoubleSpinBox()
        self.sp_clahe.setRange(0.1, 10.0)
        self.sp_clahe.setSingleStep(0.1)
        self.sp_clahe.setDecimals(2)
        self.sp_clahe.setMinimumWidth(self.INPUT_MIN_W)
        vb.addWidget(self.sp_clahe)

        vb.addWidget(_lab("dilate_iter"))
        self.sp_dilate = QSpinBox()
        self.sp_dilate.setRange(0, 10)
        self.sp_dilate.setMinimumWidth(self.INPUT_MIN_W)
        vb.addWidget(self.sp_dilate)

        vb.addWidget(_lab("close_iter"))
        self.sp_close = QSpinBox()
        self.sp_close.setRange(0, 10)
        self.sp_close.setMinimumWidth(self.INPUT_MIN_W)
        vb.addWidget(self.sp_close)

        vb.addWidget(_lab("contrast_min"))
        self.sp_contrast = QDoubleSpinBox()
        self.sp_contrast.setRange(0.0, 100.0)
        self.sp_contrast.setSingleStep(0.5)
        self.sp_contrast.setDecimals(2)
        self.sp_contrast.setMinimumWidth(self.INPUT_MIN_W)
        vb.addWidget(self.sp_contrast)

        self.btn_save = QPushButton("SAVE CONFIG JSON")
        vb.addWidget(self.btn_save)

        self.lbl_save = QLabel("Saved: —")
        vb.addWidget(self.lbl_save)

        self.controls_layout.addWidget(gb)

        # wiring
        self.btn_save.clicked.connect(self.save_config_json)

    # -------------------------
    # Responsive sizing
    # -------------------------
    def _update_right_width(self):
        W = max(1, self.width())
        target = int(W * float(self.RIGHT_W_FRAC))
        target = max(int(self.RIGHT_W_MIN), min(int(self.RIGHT_W_MAX), target))
        self.scroll.setMinimumWidth(target)
        self.scroll.setMaximumWidth(target)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_right_width()

    # -------------------------
    # External wiring
    # -------------------------
    def set_recipe_name(self, recipe_name: str):
        self._current_recipe_name = (recipe_name or "").strip()
        self._reload_configs()

    def _reload_configs(self):
        self.cmb_config.blockSignals(True)
        try:
            self.cmb_config.clear()
            if not self._current_recipe_name:
                self.lbl_cfg_status.setText("Status: NO RECIPE")
                return

            configs = self.recipe_manager.list_configs(self._current_recipe_name)
            if configs:
                for c in configs:
                    self.cmb_config.addItem(c)
                active = self.recipe_manager.get_active_config_name(self._current_recipe_name)
                if active and active in configs:
                    self.cmb_config.setCurrentText(active)
                else:
                    self.cmb_config.setCurrentIndex(0)
            else:
                self.cmb_config.addItem("legacy")
                self.cmb_config.setCurrentIndex(0)
        finally:
            self.cmb_config.blockSignals(False)

        self._on_config_changed(self.cmb_config.currentText())

    def _on_config_changed(self, name: str):
        name = (name or "").strip()
        if not self._current_recipe_name:
            return

        try:
            if name and name != "legacy":
                recipe = self.recipe_manager.load(self._current_recipe_name, config_name=name)
            else:
                recipe = self.recipe_manager.load(self._current_recipe_name)

            self.engine.set_recipe(recipe)
            self._load_tuning_from_cfg()
            self.lbl_cfg_status.setText("Status: READY")
        except Exception as e:
            self.lbl_cfg_status.setText(f"Status: LOAD FAIL: {e}")

    # -------------------------
    # View mode
    # -------------------------
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

    # -------------------------
    # Preview
    # -------------------------
    def start_preview(self):
        if self._timer.isActive():
            return
        self._timer.start(33)
        self.btn_preview_start.setEnabled(False)
        self.btn_preview_stop.setEnabled(True)
        self.lbl_cfg_status.setText("Status: PREVIEW RUNNING")

    def stop_preview(self):
        if not self._timer.isActive():
            return
        self._timer.stop()
        self.btn_preview_start.setEnabled(True)
        self.btn_preview_stop.setEnabled(False)
        self.lbl_cfg_status.setText("Status: PREVIEW STOPPED")

    def _tick_preview(self):
        ok, frame = self.cam.read()
        if not ok or frame is None:
            self.lbl_cfg_status.setText("Status: CAMERA READ FAIL")
            return

        self._last_frame_bgr = frame

        if self.engine.recipe is None:
            self.view.set_bgr(frame)
            return

        out = self.engine.process_frame(frame)
        mode = (self.engine.settings.view_mode or "OVERLAY").upper()

        if mode == "RAW" and out.raw_bgr is not None:
            self.view.set_bgr(out.raw_bgr)
        elif mode == "PROC" and out.proc_bgr is not None:
            self.view.set_bgr(out.proc_bgr)
        else:
            self.view.set_bgr(out.overlay_bgr)

    # -------------------------
    # Config folder ops
    # -------------------------
    def _configs_dir(self) -> Optional[Path]:
        if not self._current_recipe_name:
            return None
        return Path(self.recipe_manager.recipes_root) / self._current_recipe_name / "configs"

    def create_new_config(self):
        if not self._current_recipe_name:
            return

        new_name = (self.ed_new_name.text() or "").strip()
        if not new_name:
            self.lbl_cfg_status.setText("Status: ENTER CONFIG NAME")
            return

        cfgs_dir = self._configs_dir()
        if cfgs_dir is None:
            return

        cfgs_dir.mkdir(parents=True, exist_ok=True)
        dst = cfgs_dir / new_name
        if dst.exists():
            self.lbl_cfg_status.setText("Status: CONFIG EXISTS")
            return

        dst.mkdir(parents=True, exist_ok=True)

        base_cfg = dict(self.engine.recipe.cfg) if self.engine.recipe is not None else {}
        base_cfg["golden_image_path"] = "golden.png"
        _safe_write_json(str(dst / "golden_config.json"), base_cfg)

        if self._last_frame_bgr is not None:
            cv2.imwrite(str(dst / "golden.png"), self._last_frame_bgr)

        self.ed_new_name.setText("")
        self._reload_configs()
        self.cmb_config.setCurrentText(new_name)
        self.recipe_manager.set_active_config_name(self._current_recipe_name, new_name)
        self.lbl_cfg_status.setText("Status: CREATED")

    def duplicate_selected_config(self):
        if not self._current_recipe_name:
            return

        src_name = (self.cmb_config.currentText() or "").strip()
        new_name = (self.ed_new_name.text() or "").strip()

        if not new_name:
            self.lbl_cfg_status.setText("Status: ENTER NEW NAME")
            return
        if not src_name or src_name == "legacy":
            self.lbl_cfg_status.setText("Status: SELECT NON-LEGACY CONFIG")
            return

        cfgs_dir = self._configs_dir()
        if cfgs_dir is None:
            return

        src = cfgs_dir / src_name
        dst = cfgs_dir / new_name
        if dst.exists():
            self.lbl_cfg_status.setText("Status: TARGET EXISTS")
            return

        shutil.copytree(src, dst)

        self.ed_new_name.setText("")
        self._reload_configs()
        self.cmb_config.setCurrentText(new_name)
        self.recipe_manager.set_active_config_name(self._current_recipe_name, new_name)
        self.lbl_cfg_status.setText("Status: DUPLICATED")

    def set_active_config(self):
        if not self._current_recipe_name:
            return
        name = (self.cmb_config.currentText() or "").strip()
        if not name or name == "legacy":
            self.lbl_cfg_status.setText("Status: SELECT CONFIG")
            return
        self.recipe_manager.set_active_config_name(self._current_recipe_name, name)
        self.lbl_cfg_status.setText("Status: ACTIVE SET")

    def capture_golden(self):
        if self.engine.recipe is None:
            self.lbl_cfg_status.setText("Status: NO RECIPE LOADED")
            return
        if self._last_frame_bgr is None:
            self.lbl_cfg_status.setText("Status: NO FRAME (start preview)")
            return

        recipe = self.engine.recipe
        cfg = dict(recipe.cfg)

        cfg_dir = Path(recipe.config_dir) if getattr(recipe, "config_dir", "") else Path(recipe.recipe_dir)
        cfg_dir.mkdir(parents=True, exist_ok=True)

        golden_path = cfg_dir / "golden.png"
        cv2.imwrite(str(golden_path), self._last_frame_bgr)

        cfg["golden_image_path"] = "golden.png"
        _safe_write_json(str(Path(recipe.config_path)), cfg)

        # reload to pick up new golden immediately
        try:
            if getattr(recipe, "config_name", "") and recipe.config_name != "legacy":
                reloaded = self.recipe_manager.load(recipe.name, config_name=recipe.config_name)
            else:
                reloaded = self.recipe_manager.load(recipe.name)
            self.engine.set_recipe(reloaded)
        except Exception:
            pass

        self.lbl_cfg_status.setText("Status: GOLDEN CAPTURED")

    # -------------------------
    # Tuning save/load
    # -------------------------
    def _load_tuning_from_cfg(self):
        if self.engine.recipe is None:
            return
        cfg = self.engine.recipe.cfg

        self.sp_canny_low.setValue(int(cfg.get("canny_low", 50)))
        self.sp_canny_high.setValue(int(cfg.get("canny_high", 120)))
        self.sp_blur.setValue(int(cfg.get("blur_ksize", 5)))
        self.sp_clahe.setValue(float(cfg.get("clahe_clip", 1.5)))
        self.sp_dilate.setValue(int(cfg.get("dilate_iter", 1)))
        self.sp_close.setValue(int(cfg.get("close_iter", 1)))
        self.sp_contrast.setValue(float(cfg.get("contrast_min", 6.0)))

    def save_config_json(self):
        if self.engine.recipe is None:
            self.lbl_save.setText("Saved: NO RECIPE")
            return

        recipe = self.engine.recipe
        cfg = dict(recipe.cfg)

        cfg["canny_low"] = int(self.sp_canny_low.value())
        cfg["canny_high"] = int(self.sp_canny_high.value())

        blur = int(self.sp_blur.value())
        if blur % 2 == 0:
            blur += 1
        cfg["blur_ksize"] = blur

        cfg["clahe_clip"] = float(self.sp_clahe.value())
        cfg["dilate_iter"] = int(self.sp_dilate.value())
        cfg["close_iter"] = int(self.sp_close.value())
        cfg["contrast_min"] = float(self.sp_contrast.value())

        if cfg.get("golden_image_path") in (None, "", "golden.png", "golden.jpg", "golden.jpeg"):
            cfg["golden_image_path"] = cfg.get("golden_image_path") or "golden.png"

        try:
            _safe_write_json(str(Path(recipe.config_path)), cfg)
            self.lbl_save.setText(f"Saved: {Path(recipe.config_path).name}")
            recipe.cfg.clear()
            recipe.cfg.update(cfg)
        except Exception as e:
            self.lbl_save.setText(f"Saved: FAIL ({e})")

    def close(self):
        try:
            self.stop_preview()
            if self._owns_cam and self.cam is not None:
                self.cam.release()
        except Exception:
            pass
