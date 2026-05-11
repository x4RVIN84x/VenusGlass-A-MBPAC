from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Optional, Tuple, Callable

import cv2

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QRadioButton,
    QButtonGroup,
    QSizePolicy,
    QScrollArea,
)

from detector import detect_baseplate

from hmi_app.gui.image_view import ImageView
from hmi_app.io.camera import OpenCVCamera
from hmi_app.core.engine import QCPreviewEngine
from hmi_app.core.recipe_manager import RecipeManager


def _safe_write_json(path: str, data: dict):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    tmp = str(p) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    os.replace(tmp, str(p))


def _safe_load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _lab(txt: str) -> QLabel:
    l = QLabel(txt)
    l.setStyleSheet("font-size: 12px; font-weight: 800;")
    return l


def _sanitize_product_name(name: str) -> str:
    name = (name or "").strip()
    name = name.replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    name = name.strip("._-")
    return name


def _copy_first_existing_golden(src_dir: Path, dst_dir: Path) -> bool:
    for n in ("golden.png", "golden.jpg", "golden.jpeg"):
        src = src_dir / n
        if src.is_file():
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dst_dir / "golden.png"))
            return True
    return False


def _clamp_roi(roi, W: int, H: int):
    x, y, w, h = map(int, roi)

    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))

    return x, y, w, h


class CalibrationPage(QWidget):
    """
    Product calibration page.

    Flat product workflow:

      recipes/<PRODUCT_NAME>/
        golden_config.json
        golden.png

    CAPTURE GOLDEN + SET EXPECTED:
      - saves golden.png
      - detects baseplate in the current frame
      - saves expected_center
      - saves expected_angle
      - reloads product
    """

    RIGHT_W_FRAC = 0.30
    RIGHT_W_MIN = 420
    RIGHT_W_MAX = 620

    INPUT_MIN_W = 180

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
        on_products_changed: Optional[Callable[..., None]] = None,
        parent=None,
    ):
        super().__init__(parent)

        self.engine = engine
        self.recipe_manager = recipe_manager
        self.cam = cam
        self.on_products_changed = on_products_changed
        self._owns_cam = cam is None

        if self.cam is None:
            self.cam = OpenCVCamera(index=0, width=1280, height=720, fps=30, use_dshow=True)

        self._current_product_name: str = ""
        self._last_frame_bgr = None

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick_preview)

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        self.view = ImageView()
        self.view.setMinimumWidth(240)
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self.view, 1)

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

        self._build_product_group()
        self._build_preview_group()
        self._build_tuning_group()

        self.controls_layout.addStretch(1)

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
        """)

        self._update_right_width()
        self._apply_view_mode()

    # -------------------------
    # Build UI sections
    # -------------------------
    def _build_product_group(self):
        gb = QGroupBox("Product Management")
        gb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        vb = QVBoxLayout(gb)
        vb.setSpacing(8)

        vb.addWidget(_lab("Current product:"))
        self.lbl_product = QLabel("—")
        self.lbl_product.setStyleSheet("font-size: 13px; font-weight: 900;")
        vb.addWidget(self.lbl_product)

        row = QHBoxLayout()
        self.ed_new_name = QLineEdit()
        self.ed_new_name.setPlaceholderText("new_product_name")
        self.ed_new_name.setMinimumWidth(self.INPUT_MIN_W)
        self.btn_create = QPushButton("CREATE PRODUCT")
        row.addWidget(self.ed_new_name, 1)
        row.addWidget(self.btn_create, 0)
        vb.addLayout(row)

        self.btn_dup = QPushButton("DUPLICATE SELECTED PRODUCT")
        vb.addWidget(self.btn_dup)

        self.btn_capture = QPushButton("CAPTURE GOLDEN + SET EXPECTED")
        self.btn_capture.setMinimumHeight(58)
        vb.addWidget(self.btn_capture)

        self.lbl_expected = QLabel("Expected: —")
        self.lbl_expected.setStyleSheet("font-size: 12px; font-weight: 800;")
        vb.addWidget(self.lbl_expected)

        self.lbl_cfg_status = QLabel("Status: IDLE")
        self.lbl_cfg_status.setStyleSheet("font-size: 13px; font-weight: 900;")
        vb.addWidget(self.lbl_cfg_status)

        self.controls_layout.addWidget(gb)

        self.btn_create.clicked.connect(self.create_new_product)
        self.btn_dup.clicked.connect(self.duplicate_selected_product)
        self.btn_capture.clicked.connect(self.capture_golden_and_expected)

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

        self.grp_view.buttonClicked.connect(self._apply_view_mode)
        self.btn_preview_start.clicked.connect(self.start_preview)
        self.btn_preview_stop.clicked.connect(self.stop_preview)

    def _build_tuning_group(self):
        gb = QGroupBox("Product Tuning (saved to golden_config.json)")
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

        self.btn_save = QPushButton("SAVE PRODUCT JSON")
        vb.addWidget(self.btn_save)

        self.lbl_save = QLabel("Saved: —")
        vb.addWidget(self.lbl_save)

        self.controls_layout.addWidget(gb)

        self.btn_save.clicked.connect(self.save_product_json)

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
    # External wiring from MainWindow
    # -------------------------
    def set_recipe_name(self, recipe_name: str):
        self._current_product_name = (recipe_name or "").strip()
        self.lbl_product.setText(self._current_product_name or "—")
        self._reload_current_product()

    def _reload_current_product(self):
        if not self._current_product_name:
            self.lbl_cfg_status.setText("Status: NO PRODUCT")
            return

        try:
            recipe = self.recipe_manager.load(self._current_product_name)
            self.engine.set_recipe(recipe)
            self._load_tuning_from_cfg()
            self._update_expected_label()
            self.lbl_cfg_status.setText(f"Status: READY ({recipe.name})")
        except Exception as e:
            self.lbl_cfg_status.setText(f"Status: LOAD FAIL: {e}")

    def _notify_products_changed(self, select_name: Optional[str] = None):
        if callable(self.on_products_changed):
            try:
                self.on_products_changed(select_name=select_name)
                return
            except TypeError:
                try:
                    self.on_products_changed(select_name)
                    return
                except Exception:
                    pass
            except Exception:
                pass

        w = self.window()
        if w is None:
            return

        for fn_name in ("reload_products", "refresh_recipe_list", "refresh_config_selectors"):
            fn = getattr(w, fn_name, None)
            if callable(fn):
                try:
                    fn(select_name=select_name)
                except TypeError:
                    try:
                        fn()
                    except Exception:
                        pass
                except Exception:
                    pass
                return

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
    # Product filesystem helpers
    # -------------------------
    def _products_root(self) -> Path:
        return Path(self.recipe_manager.recipes_root)

    def _base_cfg_for_new_product(self) -> dict:
        if self.engine.recipe is not None:
            cfg = dict(self.engine.recipe.cfg)
        else:
            cfg = {}

        cfg["golden_image_path"] = "golden.png"
        cfg.setdefault("prefer_solid_edge", True)
        cfg.setdefault("hysteresis_enabled", True)
        cfg.setdefault("roi", [0, 0, 100, 100])
        cfg.setdefault("registration_roi", cfg.get("roi", [0, 0, 100, 100]))
        cfg.setdefault("expected_center", [50.0, 50.0])
        cfg.setdefault("expected_angle", 0.0)

        return cfg

    def _write_golden_for_new_product(self, dst_dir: Path) -> bool:
        dst_dir.mkdir(parents=True, exist_ok=True)
        golden_dst = dst_dir / "golden.png"

        if self._last_frame_bgr is not None:
            return bool(cv2.imwrite(str(golden_dst), self._last_frame_bgr))

        if self.engine.recipe is not None and self.engine.recipe.golden_bgr is not None:
            return bool(cv2.imwrite(str(golden_dst), self.engine.recipe.golden_bgr))

        if self.engine.recipe is not None:
            src_dir = Path(self.engine.recipe.recipe_dir)
            if _copy_first_existing_golden(src_dir, dst_dir):
                return True

        return False

    # -------------------------
    # Product ops
    # -------------------------
    def create_new_product(self):
        new_name = _sanitize_product_name(self.ed_new_name.text())

        if not new_name:
            self.lbl_cfg_status.setText("Status: ENTER VALID PRODUCT NAME")
            return

        root = self._products_root()
        root.mkdir(parents=True, exist_ok=True)

        dst = root / new_name

        if dst.exists():
            self.lbl_cfg_status.setText("Status: PRODUCT EXISTS")
            return

        try:
            dst.mkdir(parents=True, exist_ok=False)

            cfg = self._base_cfg_for_new_product()
            _safe_write_json(str(dst / "golden_config.json"), cfg)

            wrote_golden = self._write_golden_for_new_product(dst)

            if not wrote_golden:
                shutil.rmtree(dst, ignore_errors=True)
                self.lbl_cfg_status.setText("Status: CREATE FAIL: no golden frame/image")
                return

            self.ed_new_name.setText("")
            self._current_product_name = new_name
            self.lbl_product.setText(new_name)

            self._notify_products_changed(select_name=new_name)
            self._reload_current_product()

            self.lbl_cfg_status.setText(f"Status: CREATED PRODUCT {new_name}")

        except Exception as e:
            shutil.rmtree(dst, ignore_errors=True)
            self.lbl_cfg_status.setText(f"Status: CREATE FAIL: {e}")

    def duplicate_selected_product(self):
        if not self._current_product_name:
            self.lbl_cfg_status.setText("Status: NO SELECTED PRODUCT")
            return

        new_name = _sanitize_product_name(self.ed_new_name.text())

        if not new_name:
            self.lbl_cfg_status.setText("Status: ENTER VALID NEW NAME")
            return

        root = self._products_root()
        src = root / self._current_product_name
        dst = root / new_name

        if not src.is_dir():
            self.lbl_cfg_status.setText("Status: SOURCE PRODUCT MISSING")
            return

        if dst.exists():
            self.lbl_cfg_status.setText("Status: TARGET PRODUCT EXISTS")
            return

        try:
            shutil.copytree(src, dst)

            cfg_path = dst / "golden_config.json"

            if cfg_path.is_file():
                cfg = _safe_load_json(str(cfg_path))
            else:
                cfg = self._base_cfg_for_new_product()

            cfg["golden_image_path"] = "golden.png"
            cfg.setdefault("prefer_solid_edge", True)
            cfg.setdefault("hysteresis_enabled", True)

            _safe_write_json(str(cfg_path), cfg)

            if not (dst / "golden.png").is_file():
                _copy_first_existing_golden(src, dst)

            if not (dst / "golden.png").is_file():
                wrote_golden = self._write_golden_for_new_product(dst)
                if not wrote_golden:
                    shutil.rmtree(dst, ignore_errors=True)
                    self.lbl_cfg_status.setText("Status: DUP FAIL: no golden image")
                    return

            self.ed_new_name.setText("")
            self._current_product_name = new_name
            self.lbl_product.setText(new_name)

            self._notify_products_changed(select_name=new_name)
            self._reload_current_product()

            self.lbl_cfg_status.setText(f"Status: DUPLICATED PRODUCT {new_name}")

        except Exception as e:
            shutil.rmtree(dst, ignore_errors=True)
            self.lbl_cfg_status.setText(f"Status: DUP FAIL: {e}")

    # -------------------------
    # Golden capture + expected target
    # -------------------------
    def _detector_kwargs_from_cfg(self, cfg: dict) -> dict:
        return dict(
            shrink_border_px=int(cfg.get("shrink_border_px", 10)),
            border_margin=int(cfg.get("border_margin", 12)),
            use_clahe=bool(cfg.get("use_clahe", True)),
            clahe_clip=float(cfg.get("clahe_clip", 1.5)),
            clahe_grid=int(cfg.get("clahe_grid", 8)),
            blur_ksize=int(cfg.get("blur_ksize", 5)),
            canny_low=int(cfg.get("canny_low", 50)),
            canny_high=int(cfg.get("canny_high", 120)),
            dilate_iter=int(cfg.get("dilate_iter", 1)),
            close_iter=int(cfg.get("close_iter", 1)),
            area_min_frac=float(cfg.get("area_min_frac", 0.005)),
            area_max_frac=float(cfg.get("area_max_frac", 0.60)),
            aspect_min=float(cfg.get("aspect_min", 0.5)),
            aspect_max=float(cfg.get("aspect_max", 2.2)),
            solidity_min=float(cfg.get("solidity_min", 0.7)),
            extent_min=float(cfg.get("extent_min", 0.25)),
            contrast_min=float(cfg.get("contrast_min", 6.0)),
        )

    def _detect_expected_from_frame(
        self,
        frame_bgr,
        cfg: dict,
    ) -> Tuple[Optional[Tuple[float, float]], Optional[float], Optional[str]]:
        if frame_bgr is None:
            return None, None, "no_frame"

        if "roi" not in cfg:
            return None, None, "missing_roi"

        H, W = frame_bgr.shape[:2]
        roi = _clamp_roi(cfg["roi"], W, H)
        x, y, w, h = roi

        crop = frame_bgr[y:y + h, x:x + w].copy()

        center_rel, angle, contour_rel, dbg = detect_baseplate(
            crop,
            full_image_bgr=frame_bgr,
            roi_xywh_abs=roi,
            padding=int(cfg.get("padding", 150)),
            return_debug=True,
            **self._detector_kwargs_from_cfg(cfg),
        )

        if center_rel is None or angle is None:
            reason = "baseplate_not_found"
            if isinstance(dbg, dict):
                reason = str(dbg.get("reason", reason))
            return None, None, reason

        return (float(center_rel[0]), float(center_rel[1])), float(angle), None

    def capture_golden_and_expected(self):
        if self.engine.recipe is None:
            self.lbl_cfg_status.setText("Status: NO PRODUCT LOADED")
            return

        if self._last_frame_bgr is None:
            self.lbl_cfg_status.setText("Status: NO FRAME (start preview)")
            return

        recipe = self.engine.recipe
        product_dir = Path(recipe.recipe_dir)
        product_dir.mkdir(parents=True, exist_ok=True)

        cfg = dict(recipe.cfg)
        cfg["golden_image_path"] = "golden.png"
        cfg.setdefault("prefer_solid_edge", True)
        cfg.setdefault("hysteresis_enabled", True)

        cfg = self._cfg_with_current_tuning(cfg)

        expected_center, expected_angle, err = self._detect_expected_from_frame(self._last_frame_bgr, cfg)

        if err is not None:
            self.lbl_cfg_status.setText(f"Status: CAPTURE FAIL: {err}")
            return

        cfg["expected_center"] = [float(expected_center[0]), float(expected_center[1])]
        cfg["expected_angle"] = float(expected_angle)

        golden_path = product_dir / "golden.png"
        ok = cv2.imwrite(str(golden_path), self._last_frame_bgr)

        if not ok:
            self.lbl_cfg_status.setText("Status: GOLDEN SAVE FAIL")
            return

        try:
            _safe_write_json(str(product_dir / "golden_config.json"), cfg)

            reloaded = self.recipe_manager.load(recipe.name)
            self.engine.set_recipe(reloaded)
            self._load_tuning_from_cfg()
            self._update_expected_label()

            self.lbl_cfg_status.setText(
                f"Status: GOLDEN + EXPECTED SAVED  "
                f"cx={expected_center[0]:.1f} cy={expected_center[1]:.1f} a={expected_angle:.1f}"
            )

        except Exception as e:
            self.lbl_cfg_status.setText(f"Status: SAVED IMAGE, JSON FAIL: {e}")

    # -------------------------
    # Tuning save/load
    # -------------------------
    def _cfg_with_current_tuning(self, cfg: dict) -> dict:
        cfg = dict(cfg)

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

        cfg["golden_image_path"] = "golden.png"
        cfg.setdefault("prefer_solid_edge", True)
        cfg.setdefault("hysteresis_enabled", True)

        return cfg

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

        self._update_expected_label()

    def _update_expected_label(self):
        if self.engine.recipe is None:
            self.lbl_expected.setText("Expected: —")
            return

        cfg = self.engine.recipe.cfg
        c = cfg.get("expected_center")
        a = cfg.get("expected_angle")

        if c is None or a is None:
            self.lbl_expected.setText("Expected: not set")
            return

        try:
            self.lbl_expected.setText(
                f"Expected: cx={float(c[0]):.1f}, cy={float(c[1]):.1f}, angle={float(a):.1f}"
            )
        except Exception:
            self.lbl_expected.setText("Expected: invalid")

    def save_product_json(self):
        if self.engine.recipe is None:
            self.lbl_save.setText("Saved: NO PRODUCT")
            return

        recipe = self.engine.recipe
        cfg = self._cfg_with_current_tuning(recipe.cfg)

        product_dir = Path(recipe.recipe_dir)
        product_dir.mkdir(parents=True, exist_ok=True)

        try:
            _safe_write_json(str(product_dir / "golden_config.json"), cfg)

            recipe.cfg.clear()
            recipe.cfg.update(cfg)

            self._update_expected_label()
            self.lbl_save.setText("Saved: golden_config.json")

        except Exception as e:
            self.lbl_save.setText(f"Saved: FAIL ({e})")

    def close(self):
        try:
            self.stop_preview()
            if self._owns_cam and self.cam is not None:
                self.cam.release()
        except Exception:
            pass