from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Optional, Callable

import cv2
import numpy as np

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
    QToolButton,
)

from hmi_app.gui.image_view import ImageView
from hmi_app.io.camera import OpenCVCamera
from hmi_app.core.engine import QCPreviewEngine, get_configured_tolerances
from hmi_app.core.recipe_manager import RecipeManager


# ----------------------------
# File helpers
# ----------------------------
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


# ----------------------------
# UI helpers
# ----------------------------
def _lab(txt: str) -> QLabel:
    l = QLabel(txt)
    l.setStyleSheet("font-size: 12px; font-weight: 900;")
    return l


def _tip_html(
    *,
    title: str,
    what: str,
    increase: str,
    decrease: str,
    typical: str,
    step: str,
    warning: str = "",
) -> str:
    warn_html = ""
    if warning:
        warn_html = f"""
        <p style="margin-top:8px;">
            <b>Careful:</b> {warning}
        </p>
        """

    return f"""
    <div style="width: 390px;">
        <h3 style="margin:0 0 8px 0;">{title}</h3>

        <p><b>What it does:</b><br>{what}</p>

        <p><b>If you increase it:</b><br>{increase}</p>

        <p><b>If you decrease it:</b><br>{decrease}</p>

        <p><b>Usual safe range:</b><br>{typical}</p>

        <p><b>Change amount:</b><br>{step}</p>

        {warn_html}
    </div>
    """


def _help_button(tip: str) -> QToolButton:
    b = QToolButton()
    b.setText("?")
    b.setToolTip(tip)
    b.setCursor(Qt.WhatsThisCursor)
    b.setAutoRaise(False)
    b.setFixedSize(22, 22)
    b.setStyleSheet("""
        QToolButton {
            border: 1px solid #4a5166;
            border-radius: 11px;
            color: #dfe6ff;
            background: #1c2030;
            font-weight: 900;
            font-size: 12px;
        }
        QToolButton:hover {
            background: #2e385a;
            border: 1px solid #7d8dff;
        }
    """)
    return b


def _add_param(vb: QVBoxLayout, name: str, widget, tip: str):
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(6)

    row.addWidget(_lab(name), 1)
    row.addWidget(_help_button(tip), 0)

    vb.addLayout(row)

    widget.setMinimumWidth(CalibrationPage.INPUT_MIN_W)
    vb.addWidget(widget)


# ----------------------------
# Calibration Page
# ----------------------------
class CalibrationPage(QWidget):
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
        self._last_engine_out = None

        self._loading_ui = False
        self._live_dirty = False

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick_preview)

        self._live_apply_timer = QTimer(self)
        self._live_apply_timer.setSingleShot(True)
        self._live_apply_timer.timeout.connect(self._apply_tuning_to_live_recipe)

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        # Normal preview image. No proc dashboard / preview_stack on this branch.
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
        self._build_baseplate_tuning_group()
        self._build_tolerance_group()
        self._build_notch_tuning_group()
        self._build_save_group()

        self.controls_layout.addStretch(1)

        self.controls_container.setStyleSheet(self.controls_container.styleSheet() + f"""
            QLabel {{
                color: {self.LABEL};
            }}

            QLineEdit, QSpinBox, QDoubleSpinBox {{
                background: {self.INPUT_BG};
                color: {self.TEXT};
                border: 1px solid {self.BORDER};
                border-radius: 6px;
                padding: 6px 8px;
                font-size: 13px;
                selection-background-color: {self.FOCUS};
                selection-color: #ffffff;
            }}

            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
                border: 1px solid {self.FOCUS};
            }}

            QPushButton {{
                min-height: 34px;
                font-weight: 800;
            }}

            QToolTip {{
                color: #f4f6ff;
                background-color: #202433;
                border: 1px solid #6570a6;
                padding: 10px;
                font-size: 12px;
            }}
        """)

        self._connect_tuning_signals()
        self._update_right_width()
        self._apply_view_mode()

    # -------------------------
    # Build UI
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
        self.lbl_expected.setWordWrap(True)
        vb.addWidget(self.lbl_expected)

        self.lbl_cfg_status = QLabel("Status: IDLE")
        self.lbl_cfg_status.setStyleSheet("font-size: 13px; font-weight: 900;")
        self.lbl_cfg_status.setWordWrap(True)
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

    def _build_baseplate_tuning_group(self):
        gb = QGroupBox("Baseplate Tuning + Scale")
        gb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        vb = QVBoxLayout(gb)
        vb.setSpacing(6)

        self.sp_baseplate_w_mm = QDoubleSpinBox()
        self.sp_baseplate_w_mm.setRange(1.0, 200.0)
        self.sp_baseplate_w_mm.setSingleStep(0.1)
        self.sp_baseplate_w_mm.setDecimals(2)
        _add_param(
            vb,
            "baseplate_width_mm",
            self.sp_baseplate_w_mm,
            _tip_html(
                title="Baseplate width in millimeters",
                what="This is the real physical width of the metal baseplate/clip. The app uses it to convert camera pixels into millimeters.",
                increase="The app will think each pixel is worth fewer millimeters. The displayed mm offset becomes smaller.",
                decrease="The app will think each pixel is worth more millimeters. The displayed mm offset becomes larger.",
                typical="Use the real measured width. For this baseplate: 20.4 mm.",
                step="Do not tune by guessing. Measure the part, then enter the real value. Change by 0.1 mm only if your measurement was wrong.",
                warning="Wrong physical dimensions make all mm readings wrong, even if the tracking looks good.",
            ),
        )

        self.sp_baseplate_h_mm = QDoubleSpinBox()
        self.sp_baseplate_h_mm.setRange(1.0, 200.0)
        self.sp_baseplate_h_mm.setSingleStep(0.1)
        self.sp_baseplate_h_mm.setDecimals(2)
        _add_param(
            vb,
            "baseplate_height_mm",
            self.sp_baseplate_h_mm,
            _tip_html(
                title="Baseplate height in millimeters",
                what="This is the real physical height of the metal baseplate/clip. It is used together with width to calculate pixel-to-mm scale.",
                increase="The saved px/mm scale usually becomes smaller. Displayed mm movement becomes smaller.",
                decrease="The saved px/mm scale usually becomes larger. Displayed mm movement becomes larger.",
                typical="Use the real measured height. For this baseplate: 26.5 mm.",
                step="Measure the part and enter the real value. Change by 0.1 mm only for measurement correction.",
                warning="This should describe the metal baseplate, not the ROI box and not the glass notch.",
            ),
        )

        self.lbl_scale = QLabel("Scale: —")
        self.lbl_scale.setStyleSheet("font-size: 12px; font-weight: 800;")
        self.lbl_scale.setWordWrap(True)
        vb.addWidget(self.lbl_scale)

        self.sp_canny_low = QSpinBox()
        self.sp_canny_low.setRange(0, 255)
        _add_param(
            vb,
            "canny_low",
            self.sp_canny_low,
            _tip_html(
                title="Baseplate edge sensitivity - low threshold",
                what="This controls how easily the app starts seeing an edge on the metal baseplate.",
                increase="Weak edges disappear. This can remove noise, but may also make the baseplate vanish.",
                decrease="More weak edges appear. This can help find a dim baseplate, but may also detect dirt, glare, or texture.",
                typical="Usually 40 to 90.",
                step="Change by 5 at a time. After changing, watch whether the green baseplate outline becomes cleaner or worse.",
                warning="If this is too low, the app may chase random scratches. If too high, it may fail to detect the baseplate.",
            ),
        )

        self.sp_canny_high = QSpinBox()
        self.sp_canny_high.setRange(0, 255)
        _add_param(
            vb,
            "canny_high",
            self.sp_canny_high,
            _tip_html(
                title="Baseplate edge confirmation - high threshold",
                what="This controls how strong an edge must be before the app trusts it as real.",
                increase="Only strong edges remain. Good for noisy images, bad for low contrast lighting.",
                decrease="More edges are accepted. Good for dim footage, bad if reflections or dots confuse detection.",
                typical="Usually 100 to 180.",
                step="Change by 10 at a time. Keep it higher than canny_low.",
                warning="A good starting pair is low=50 to 70 and high=120 to 160.",
            ),
        )

        self.sp_blur = QSpinBox()
        self.sp_blur.setRange(3, 31)
        self.sp_blur.setSingleStep(2)
        _add_param(
            vb,
            "blur_ksize",
            self.sp_blur,
            _tip_html(
                title="Baseplate blur amount",
                what="This slightly smooths the image before detecting the baseplate. It helps ignore tiny texture and camera noise.",
                increase="The image becomes smoother. Small noise disappears, but sharp baseplate corners may get softer.",
                decrease="The image stays sharper. Good for crisp edges, but can make detection flicker in noisy lighting.",
                typical="Usually 3, 5, or 7.",
                step="Change by 2 only. This number must stay odd: 3, 5, 7, 9...",
                warning="Too much blur can make the baseplate outline fat or inaccurate.",
            ),
        )

        self.sp_clahe = QDoubleSpinBox()
        self.sp_clahe.setRange(0.1, 10.0)
        self.sp_clahe.setSingleStep(0.1)
        self.sp_clahe.setDecimals(2)
        _add_param(
            vb,
            "clahe_clip",
            self.sp_clahe,
            _tip_html(
                title="Local contrast boost",
                what="This boosts contrast in small areas so the metal baseplate stands out more from the dark glass.",
                increase="Edges and surface marks become stronger. Helpful in flat lighting, risky with glare or dirty glass.",
                decrease="The image looks more natural and less noisy, but weak baseplate edges may be harder to see.",
                typical="Usually 1.0 to 2.5.",
                step="Change by 0.1 or 0.2 at a time.",
                warning="Too high can make stains, dust, or reflections look important.",
            ),
        )

        self.sp_dilate = QSpinBox()
        self.sp_dilate.setRange(0, 10)
        _add_param(
            vb,
            "dilate_iter",
            self.sp_dilate,
            _tip_html(
                title="Thicken detected baseplate edges",
                what="This expands detected edge pixels so broken baseplate outlines can connect.",
                increase="Gaps may close and the baseplate may be found more often. But nearby noise can also merge into the baseplate.",
                decrease="Edges stay thinner and more precise, but broken outlines may fail.",
                typical="Usually 0 to 2.",
                step="Change by 1 at a time.",
                warning="Too high can make the detected shape bigger than the real baseplate.",
            ),
        )

        self.sp_close = QSpinBox()
        self.sp_close.setRange(0, 10)
        _add_param(
            vb,
            "close_iter",
            self.sp_close,
            _tip_html(
                title="Fill small gaps in baseplate shape",
                what="This tries to close tiny holes or breaks in the baseplate outline.",
                increase="The green baseplate contour becomes more solid and stable if the edge is broken.",
                decrease="The contour becomes less artificially connected and may be more precise.",
                typical="Usually 1 to 3.",
                step="Change by 1 at a time.",
                warning="Too high can connect the baseplate to shadows or glare nearby.",
            ),
        )

        self.sp_contrast = QDoubleSpinBox()
        self.sp_contrast.setRange(0.0, 100.0)
        self.sp_contrast.setSingleStep(0.5)
        self.sp_contrast.setDecimals(2)
        _add_param(
            vb,
            "contrast_min",
            self.sp_contrast,
            _tip_html(
                title="Minimum baseplate contrast",
                what="This rejects detections that do not look different enough from their background.",
                increase="The app becomes pickier. It may ignore false detections, but may also lose the real baseplate in bad lighting.",
                decrease="The app accepts weaker detections. Helpful if the baseplate is dim, risky if noise gets detected.",
                typical="Usually 3.0 to 8.0.",
                step="Change by 0.5 at a time.",
                warning="If the baseplate flickers between found and not found, adjust this together with canny_low/high.",
            ),
        )

        self.controls_layout.addWidget(gb)

    def _build_tolerance_group(self):
        gb = QGroupBox("Acceptance Tolerances")
        gb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        vb = QVBoxLayout(gb)
        vb.setSpacing(6)

        hint = QLabel("PASS limits for this product. They are saved in mm and degrees.")
        hint.setWordWrap(True)
        hint.setStyleSheet("font-size: 12px; font-weight: 800; color: #aeb3c2;")
        vb.addWidget(hint)

        self.sp_tol_x_mm = QDoubleSpinBox()
        self.sp_tol_x_mm.setRange(0.01, 50.0)
        self.sp_tol_x_mm.setSingleStep(0.05)
        self.sp_tol_x_mm.setDecimals(3)
        self.sp_tol_x_mm.setSuffix(" mm")
        _add_param(
            vb,
            "X tolerance",
            self.sp_tol_x_mm,
            _tip_html(
                title="Allowed X movement in millimeters",
                what="The product still passes when its X offset from the captured golden position is within this distance.",
                increase="Allows more left/right movement before the part fails.",
                decrease="Makes X placement stricter.",
                typical="Start with the engineering tolerance for this product, for example 0.25 to 1.00 mm.",
                step="Use the actual allowed manufacturing tolerance; do not tune this from noisy camera pixels.",
                warning="This is an acceptance limit, not a detector tuning control.",
            ),
        )

        self.sp_tol_y_mm = QDoubleSpinBox()
        self.sp_tol_y_mm.setRange(0.01, 50.0)
        self.sp_tol_y_mm.setSingleStep(0.05)
        self.sp_tol_y_mm.setDecimals(3)
        self.sp_tol_y_mm.setSuffix(" mm")
        _add_param(
            vb,
            "Y tolerance",
            self.sp_tol_y_mm,
            _tip_html(
                title="Allowed Y movement in millimeters",
                what="The product still passes when its Y offset from the captured golden position is within this distance.",
                increase="Allows more up/down movement before the part fails.",
                decrease="Makes Y placement stricter.",
                typical="Start with the engineering tolerance for this product, for example 0.25 to 1.00 mm.",
                step="Use the actual allowed manufacturing tolerance; do not tune this from noisy camera pixels.",
                warning="This is an acceptance limit, not a detector tuning control.",
            ),
        )

        self.sp_tol_angle_deg = QDoubleSpinBox()
        self.sp_tol_angle_deg.setRange(0.1, 45.0)
        self.sp_tol_angle_deg.setSingleStep(0.1)
        self.sp_tol_angle_deg.setDecimals(2)
        self.sp_tol_angle_deg.setSuffix(" degrees")
        _add_param(
            vb,
            "Angle tolerance",
            self.sp_tol_angle_deg,
            _tip_html(
                title="Allowed rotation in degrees",
                what="The product still passes when its measured rotation from the captured golden position is within this angle.",
                increase="Allows more angular rotation before the part fails.",
                decrease="Makes angular alignment stricter.",
                typical="Use the product drawing's angular tolerance; a common starting point is 1 to 5 degrees.",
                step="Change by 0.1 degree when the engineering requirement calls for it.",
                warning="This is evaluated independently from the X and Y millimeter limits.",
            ),
        )

        self.controls_layout.addWidget(gb)

    def _build_notch_tuning_group(self):
        gb = QGroupBox("Notch / Glass Edge Tuning")
        gb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        vb = QVBoxLayout(gb)
        vb.setSpacing(6)

        self.sp_notch_blur = QSpinBox()
        self.sp_notch_blur.setRange(3, 81)
        self.sp_notch_blur.setSingleStep(2)
        _add_param(
            vb,
            "notch_blur_ksize",
            self.sp_notch_blur,
            _tip_html(
                title="Notch blur amount",
                what="This smooths the glass notch area before finding the solid dark edge. It helps kill the printed dot pattern.",
                increase="Dots and tiny texture disappear more. The notch edge can become smoother, but exact corners may shift.",
                decrease="The notch stays sharper, but the app may start seeing the dot pattern again.",
                typical="Usually 15 to 31.",
                step="Change by 2 or 4 at a time. Keep it odd: 15, 17, 19, 21...",
                warning="Too low means dots can confuse the notch edge. Too high can move the edge away from the real boundary.",
            ),
        )

        self.sp_notch_close = QSpinBox()
        self.sp_notch_close.setRange(1, 81)
        self.sp_notch_close.setSingleStep(2)
        _add_param(
            vb,
            "notch_close_ksize",
            self.sp_notch_close,
            _tip_html(
                title="Connect the notch edge",
                what="This closes small gaps in the detected dark notch region so the app sees one cleaner shape.",
                increase="Broken notch edges connect better. Good if the yellow contour has gaps.",
                decrease="The app trusts the raw edge more. Good if the contour is already clean.",
                typical="Usually 7 to 17.",
                step="Change by 2 at a time.",
                warning="Too high can smear the notch into nearby dark areas and shift fitted lines.",
            ),
        )

        self.sp_notch_open = QSpinBox()
        self.sp_notch_open.setRange(1, 81)
        self.sp_notch_open.setSingleStep(2)
        _add_param(
            vb,
            "notch_open_ksize",
            self.sp_notch_open,
            _tip_html(
                title="Remove tiny notch noise",
                what="This removes small isolated blobs before fitting the notch lines.",
                increase="More tiny noise disappears. Helpful for dots, dust, and speckles.",
                decrease="More detail remains. Helpful if the real notch edge is getting erased.",
                typical="Usually 3 to 11.",
                step="Change by 2 at a time.",
                warning="Too high can delete real thin parts of the notch edge.",
            ),
        )

        self.sp_notch_bias = QDoubleSpinBox()
        self.sp_notch_bias.setRange(-80.0, 80.0)
        self.sp_notch_bias.setSingleStep(1.0)
        self.sp_notch_bias.setDecimals(1)
        _add_param(
            vb,
            "notch_threshold_bias",
            self.sp_notch_bias,
            _tip_html(
                title="Dark edge threshold bias",
                what="This nudges the brightness cutoff used to decide what counts as the dark notch area.",
                increase="The app usually becomes stricter about what is dark enough. The detected dark region may shrink.",
                decrease="The app accepts more area as dark. The detected dark region may grow.",
                typical="Usually -10 to +10.",
                step="Change by 1 or 2 at a time.",
                warning="If the yellow contour is outside the real edge, move this slowly. Big jumps can completely change the detected notch.",
            ),
        )

        self.sp_notch_bottom_band = QDoubleSpinBox()
        self.sp_notch_bottom_band.setRange(0.05, 0.80)
        self.sp_notch_bottom_band.setSingleStep(0.05)
        self.sp_notch_bottom_band.setDecimals(2)
        _add_param(
            vb,
            "notch_bottom_band_frac",
            self.sp_notch_bottom_band,
            _tip_html(
                title="How much of the lower notch is used",
                what="This controls how much of the lower part of the notch is used when fitting the bottom reference line.",
                increase="The app looks at a taller lower region. More data can be stable, but may include curved side areas.",
                decrease="The app focuses closer to the bottom edge. More precise if clean, less stable if noisy.",
                typical="Usually 0.25 to 0.45.",
                step="Change by 0.05 at a time.",
                warning="If the orange bottom line is pulled into the curved corners, reduce this.",
            ),
        )

        self.sp_notch_side_band = QDoubleSpinBox()
        self.sp_notch_side_band.setRange(0.05, 0.80)
        self.sp_notch_side_band.setSingleStep(0.05)
        self.sp_notch_side_band.setDecimals(2)
        _add_param(
            vb,
            "notch_side_band_frac",
            self.sp_notch_side_band,
            _tip_html(
                title="How much of the side notch walls are used",
                what="This controls how much of the left and right notch edges are used to fit the green side lines.",
                increase="More side edge points are used. This can stabilize the line, but curved parts may pull it away.",
                decrease="Fewer side edge points are used. This can focus on the straight wall, but may become noisy.",
                typical="Usually 0.25 to 0.45.",
                step="Change by 0.05 at a time.",
                warning="If the green side lines look perfect, do not touch this. If they lean into the curve, reduce it.",
            ),
        )

        self.sp_roi_extra_pad = QSpinBox()
        self.sp_roi_extra_pad.setRange(0, 200)
        _add_param(
            vb,
            "notch_frame_roi_extra_pad",
            self.sp_roi_extra_pad,
            _tip_html(
                title="Extra safety room around live ROI",
                what="This adds extra pixels around the predicted baseplate ROI so the detector has room to find the baseplate.",
                increase="The cyan ROI becomes bigger. Safer if the ROI is slightly off, but more likely to include noise.",
                decrease="The cyan ROI becomes tighter. Faster and cleaner if placement is accurate.",
                typical="Usually 5 to 40.",
                step="Change by 2 to 5 pixels at a time.",
                warning="Too small can cut off the baseplate. Too large can make the detector pick up wrong edges.",
            ),
        )

        self.controls_layout.addWidget(gb)

    def _build_save_group(self):
        gb = QGroupBox("Save / Live Apply")
        gb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        vb = QVBoxLayout(gb)
        vb.setSpacing(8)

        self.lbl_live = QLabel("Live: parameters apply to preview automatically")
        self.lbl_live.setWordWrap(True)
        self.lbl_live.setStyleSheet("font-size: 12px; font-weight: 800; color: #aeb3c2;")
        vb.addWidget(self.lbl_live)

        self.btn_save = QPushButton("SAVE PRODUCT JSON")
        vb.addWidget(self.btn_save)

        self.lbl_save = QLabel("Saved: —")
        self.lbl_save.setWordWrap(True)
        vb.addWidget(self.lbl_save)

        self.controls_layout.addWidget(gb)

        self.btn_save.clicked.connect(self.save_product_json)

    # -------------------------
    # Live tuning
    # -------------------------
    def _connect_tuning_signals(self):
        widgets = [
            self.sp_baseplate_w_mm,
            self.sp_baseplate_h_mm,
            self.sp_tol_x_mm,
            self.sp_tol_y_mm,
            self.sp_tol_angle_deg,
            self.sp_canny_low,
            self.sp_canny_high,
            self.sp_blur,
            self.sp_clahe,
            self.sp_dilate,
            self.sp_close,
            self.sp_contrast,
            self.sp_notch_blur,
            self.sp_notch_close,
            self.sp_notch_open,
            self.sp_notch_bias,
            self.sp_notch_bottom_band,
            self.sp_notch_side_band,
            self.sp_roi_extra_pad,
        ]

        for w in widgets:
            try:
                w.valueChanged.connect(self._schedule_live_tuning_apply)
            except Exception:
                pass

    def _schedule_live_tuning_apply(self, *args):
        if self._loading_ui:
            return

        self._live_dirty = True
        self.lbl_live.setText("Live: applying unsaved parameter changes...")
        self.lbl_save.setText("Saved: unsaved changes")
        self._live_apply_timer.start(120)

    def _apply_tuning_to_live_recipe(self):
        if self.engine.recipe is None:
            return

        cfg = self._cfg_with_current_tuning(self.engine.recipe.cfg)

        try:
            self.engine.recipe.cfg.clear()
            self.engine.recipe.cfg.update(cfg)

            try:
                self.engine._bp_hyst.reset()
            except Exception:
                pass

            try:
                self.engine._stab_info = None
                self.engine._frame_i = 0
            except Exception:
                pass

            self.lbl_live.setText("Live: current parameters are being used by preview")
        except Exception as e:
            self.lbl_live.setText(f"Live: apply failed: {e}")

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
    # View mode / preview
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

    def start_preview(self):
        if self._timer.isActive():
            return

        self._apply_tuning_to_live_recipe()

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
            self._last_engine_out = None
            self.view.set_bgr(frame)
            return

        try:
            out = self.engine.process_frame(frame)
        except Exception as e:
            self.lbl_cfg_status.setText(f"Status: ENGINE FAIL: {e}")
            self.view.set_bgr(frame)
            return

        self._last_engine_out = out

        mode = (self.engine.settings.view_mode or "OVERLAY").upper()

        if mode == "RAW":
            self.view.set_bgr(out.raw_bgr if out.raw_bgr is not None else frame)

        elif mode == "PROC":
            if out.proc_bgr is not None:
                self.view.set_bgr(out.proc_bgr)
            else:
                fail = frame.copy()
                cv2.rectangle(fail, (20, 20), (900, 110), (0, 0, 0), -1)
                cv2.putText(
                    fail,
                    "PROC ERROR: out.proc_bgr is None",
                    (40, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                self.view.set_bgr(fail)

        else:
            self.view.set_bgr(out.overlay_bgr if out.overlay_bgr is not None else frame)

    # -------------------------
    # Product helpers
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
        cfg.setdefault("prefer_dark_region", True)
        cfg.setdefault("hysteresis_enabled", True)

        cfg.setdefault("roi", [0, 0, 100, 100])
        cfg.setdefault("registration_roi", cfg.get("roi", [0, 0, 100, 100]))

        cfg.setdefault("expected_center", [50.0, 50.0])
        cfg.setdefault("expected_angle", 0.0)

        if not isinstance(cfg.get("tolerance"), dict):
            legacy_tolerance = get_configured_tolerances(cfg)
            cfg["tolerance"] = {
                "x_mm": float(legacy_tolerance.get("x_mm") or 1.0),
                "y_mm": float(legacy_tolerance.get("y_mm") or 1.0),
                "angle_deg": float(legacy_tolerance.get("angle_deg") or 5.0),
            }
        cfg.pop("tolerance_px", None)

        cfg.setdefault("baseplate_width_mm", 20.4)
        cfg.setdefault("baseplate_height_mm", 26.5)
        cfg.setdefault("baseplate_dimensions_mm", {"width": 20.4, "height": 26.5})

        cfg.setdefault("canny_low", 58)
        cfg.setdefault("canny_high", 150)
        cfg.setdefault("blur_ksize", 3)
        cfg.setdefault("clahe_clip", 1.5)
        cfg.setdefault("dilate_iter", 1)
        cfg.setdefault("close_iter", 3)
        cfg.setdefault("contrast_min", 4.5)

        cfg.setdefault("notch_blur_ksize", 21)
        cfg.setdefault("notch_close_ksize", 11)
        cfg.setdefault("notch_open_ksize", 7)
        cfg.setdefault("notch_threshold_bias", 1.0)
        cfg.setdefault("notch_bottom_band_frac", 0.35)
        cfg.setdefault("notch_side_band_frac", 0.35)

        cfg.setdefault("notch_frame_roi_extra_pad", 6)
        cfg.setdefault("baseplate_roi_extra_pad", 6)

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
            cfg.setdefault("prefer_dark_region", True)
            cfg.setdefault("hysteresis_enabled", True)
            cfg.setdefault("baseplate_width_mm", 20.4)
            cfg.setdefault("baseplate_height_mm", 26.5)
            cfg.setdefault("baseplate_dimensions_mm", {"width": 20.4, "height": 26.5})

            if not isinstance(cfg.get("tolerance"), dict):
                legacy_tolerance = get_configured_tolerances(cfg)
                cfg["tolerance"] = {
                    "x_mm": float(legacy_tolerance.get("x_mm") or 1.0),
                    "y_mm": float(legacy_tolerance.get("y_mm") or 1.0),
                    "angle_deg": float(legacy_tolerance.get("angle_deg") or 5.0),
                }
            cfg.pop("tolerance_px", None)

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
    # Config load / save
    # -------------------------
    def _odd(self, value: int, minimum: int = 1) -> int:
        v = max(int(minimum), int(value))
        if v % 2 == 0:
            v += 1
        return v

    def _cfg_with_current_tuning(self, cfg: dict) -> dict:
        cfg = dict(cfg)

        bw = float(self.sp_baseplate_w_mm.value())
        bh = float(self.sp_baseplate_h_mm.value())

        cfg["baseplate_width_mm"] = bw
        cfg["baseplate_height_mm"] = bh
        cfg["baseplate_dimensions_mm"] = {
            "width": bw,
            "height": bh,
        }

        cfg["canny_low"] = int(self.sp_canny_low.value())
        cfg["canny_high"] = int(self.sp_canny_high.value())
        cfg["blur_ksize"] = self._odd(int(self.sp_blur.value()), 3)
        cfg["clahe_clip"] = float(self.sp_clahe.value())
        cfg["dilate_iter"] = int(self.sp_dilate.value())
        cfg["close_iter"] = int(self.sp_close.value())
        cfg["contrast_min"] = float(self.sp_contrast.value())

        cfg["notch_blur_ksize"] = self._odd(int(self.sp_notch_blur.value()), 3)
        cfg["notch_close_ksize"] = self._odd(int(self.sp_notch_close.value()), 1)
        cfg["notch_open_ksize"] = self._odd(int(self.sp_notch_open.value()), 1)
        cfg["notch_threshold_bias"] = float(self.sp_notch_bias.value())
        cfg["notch_bottom_band_frac"] = float(self.sp_notch_bottom_band.value())
        cfg["notch_side_band_frac"] = float(self.sp_notch_side_band.value())

        cfg["notch_frame_roi_extra_pad"] = int(self.sp_roi_extra_pad.value())
        cfg["baseplate_roi_extra_pad"] = int(self.sp_roi_extra_pad.value())

        cfg["tolerance"] = {
            "x_mm": float(self.sp_tol_x_mm.value()),
            "y_mm": float(self.sp_tol_y_mm.value()),
            "angle_deg": float(self.sp_tol_angle_deg.value()),
        }
        # Once a recipe is saved from the Calibration page, its acceptance
        # limits have a physical unit and no longer need the legacy pixel block.
        cfg.pop("tolerance_px", None)

        cfg["golden_image_path"] = "golden.png"
        cfg.setdefault("prefer_solid_edge", True)
        cfg.setdefault("prefer_dark_region", True)
        cfg.setdefault("hysteresis_enabled", True)

        return cfg

    def _get_cfg_baseplate_dims(self, cfg: dict):
        dims = cfg.get("baseplate_dimensions_mm")

        if isinstance(dims, dict):
            w = dims.get("width", dims.get("w", dims.get("W", None)))
            h = dims.get("height", dims.get("h", dims.get("H", None)))
        else:
            w = cfg.get("baseplate_width_mm", 20.4)
            h = cfg.get("baseplate_height_mm", 26.5)

        try:
            return float(w), float(h)
        except Exception:
            return 20.4, 26.5

    def _load_tuning_from_cfg(self):
        if self.engine.recipe is None:
            return

        self._loading_ui = True

        try:
            cfg = self.engine.recipe.cfg

            bw, bh = self._get_cfg_baseplate_dims(cfg)
            self.sp_baseplate_w_mm.setValue(float(bw))
            self.sp_baseplate_h_mm.setValue(float(bh))

            tolerance = get_configured_tolerances(cfg)
            self.sp_tol_x_mm.setValue(float(tolerance.get("x_mm") or 1.0))
            self.sp_tol_y_mm.setValue(float(tolerance.get("y_mm") or 1.0))
            self.sp_tol_angle_deg.setValue(float(tolerance.get("angle_deg") or 5.0))

            self.sp_canny_low.setValue(int(cfg.get("canny_low", 58)))
            self.sp_canny_high.setValue(int(cfg.get("canny_high", 150)))
            self.sp_blur.setValue(self._odd(int(cfg.get("blur_ksize", 3)), 3))
            self.sp_clahe.setValue(float(cfg.get("clahe_clip", 1.5)))
            self.sp_dilate.setValue(int(cfg.get("dilate_iter", 1)))
            self.sp_close.setValue(int(cfg.get("close_iter", 3)))
            self.sp_contrast.setValue(float(cfg.get("contrast_min", 4.5)))

            self.sp_notch_blur.setValue(self._odd(int(cfg.get("notch_blur_ksize", 21)), 3))
            self.sp_notch_close.setValue(self._odd(int(cfg.get("notch_close_ksize", 11)), 1))
            self.sp_notch_open.setValue(self._odd(int(cfg.get("notch_open_ksize", 7)), 1))
            self.sp_notch_bias.setValue(float(cfg.get("notch_threshold_bias", 1.0)))
            self.sp_notch_bottom_band.setValue(float(cfg.get("notch_bottom_band_frac", 0.35)))
            self.sp_notch_side_band.setValue(float(cfg.get("notch_side_band_frac", 0.35)))
            self.sp_roi_extra_pad.setValue(int(cfg.get("notch_frame_roi_extra_pad", cfg.get("baseplate_roi_extra_pad", 6))))

            self.lbl_save.setText("Saved: loaded from product JSON")
            self.lbl_live.setText("Live: product JSON parameters loaded")
        finally:
            self._loading_ui = False

        self._apply_tuning_to_live_recipe()
        self._update_expected_label()

    def _get_saved_px_per_mm(self, cfg: dict):
        for key in ("px_per_mm", "baseplate_px_per_mm"):
            try:
                v = float(cfg.get(key))
                if np.isfinite(v) and v > 0:
                    return v
            except Exception:
                pass

        try:
            mm_per_px = float(cfg.get("mm_per_px"))
            if np.isfinite(mm_per_px) and mm_per_px > 0:
                return 1.0 / mm_per_px
        except Exception:
            pass

        scale = cfg.get("baseplate_scale")
        if isinstance(scale, dict):
            try:
                v = float(scale.get("px_per_mm"))
                if np.isfinite(v) and v > 0:
                    return v
            except Exception:
                pass

        return None

    def _update_scale_label(self):
        if self.engine.recipe is None:
            self.lbl_scale.setText("Scale: —")
            return

        cfg = self.engine.recipe.cfg
        px_per_mm = self._get_saved_px_per_mm(cfg)

        if px_per_mm is None:
            self.lbl_scale.setText("Scale: not captured yet")
            return

        self.lbl_scale.setText(f"Scale: {px_per_mm:.3f} px/mm  ({1.0 / px_per_mm:.5f} mm/px)")

    def _update_expected_label(self):
        if self.engine.recipe is None:
            self.lbl_expected.setText("Expected: —")
            self.lbl_scale.setText("Scale: —")
            return

        cfg = self.engine.recipe.cfg
        self._update_scale_label()

        enf = cfg.get("expected_notch_frame")

        if isinstance(enf, dict):
            try:
                self.lbl_expected.setText(
                    "Expected notch-frame: "
                    f"dx={float(enf.get('dx', 0.0)):.1f}px, "
                    f"dy={float(enf.get('dy', 0.0)):.1f}px, "
                    f"relTheta={float(enf.get('relative_angle', 0.0)):.1f}"
                )
                return
            except Exception:
                pass

        c = cfg.get("expected_center")
        a = cfg.get("expected_angle")

        if c is None or a is None:
            self.lbl_expected.setText("Expected: not set")
            return

        try:
            self.lbl_expected.setText(
                f"Expected fallback: cx={float(c[0]):.1f}, cy={float(c[1]):.1f}, angle={float(a):.1f}"
            )
        except Exception:
            self.lbl_expected.setText("Expected: invalid")

    # -------------------------
    # Scale helpers
    # -------------------------
    def _scale_from_live_stab(self, stab_info: dict):
        if not isinstance(stab_info, dict):
            return None

        scale = stab_info.get("baseplate_scale_frame")
        if isinstance(scale, dict):
            try:
                px_per_mm = float(scale.get("px_per_mm"))
                if np.isfinite(px_per_mm) and px_per_mm > 0:
                    return dict(scale)
            except Exception:
                pass

        try:
            px_per_mm = float(stab_info.get("baseplate_px_per_mm_frame"))
            if np.isfinite(px_per_mm) and px_per_mm > 0:
                return {
                    "px_per_mm": float(px_per_mm),
                    "mm_per_px": float(1.0 / px_per_mm),
                }
        except Exception:
            pass

        return None

    def _write_baseplate_scale_to_cfg(self, cfg: dict, scale: dict) -> dict:
        cfg = dict(cfg)

        if not isinstance(scale, dict):
            return cfg

        try:
            px_per_mm = float(scale.get("px_per_mm"))
        except Exception:
            return cfg

        if not np.isfinite(px_per_mm) or px_per_mm <= 0:
            return cfg

        bw = float(cfg.get("baseplate_width_mm", self.sp_baseplate_w_mm.value()))
        bh = float(cfg.get("baseplate_height_mm", self.sp_baseplate_h_mm.value()))

        clean_scale = dict(scale)
        clean_scale["px_per_mm"] = float(px_per_mm)
        clean_scale["mm_per_px"] = float(1.0 / px_per_mm)
        clean_scale["baseplate_width_mm"] = float(bw)
        clean_scale["baseplate_height_mm"] = float(bh)
        clean_scale["baseplate_dimensions_mm"] = {
            "width": float(bw),
            "height": float(bh),
        }

        cfg["px_per_mm"] = float(px_per_mm)
        cfg["mm_per_px"] = float(1.0 / px_per_mm)
        cfg["baseplate_px_per_mm"] = float(px_per_mm)
        cfg["baseplate_scale"] = clean_scale

        cfg["baseplate_width_mm"] = float(bw)
        cfg["baseplate_height_mm"] = float(bh)
        cfg["baseplate_dimensions_mm"] = {
            "width": float(bw),
            "height": float(bh),
        }

        return cfg

    def _force_live_engine_output_with_cfg(self, cfg: dict):
        if self.engine.recipe is None or self._last_frame_bgr is None:
            return self._last_engine_out

        try:
            self.engine.recipe.cfg.clear()
            self.engine.recipe.cfg.update(cfg)
        except Exception:
            pass

        try:
            out = self.engine.process_frame(self._last_frame_bgr)
            self._last_engine_out = out
            return out
        except Exception:
            return self._last_engine_out

    # -------------------------
    # Capture / save
    # -------------------------
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
        cfg = self._cfg_with_current_tuning(cfg)

        out = self._force_live_engine_output_with_cfg(cfg)

        if out is None:
            self.lbl_cfg_status.setText("Status: CAPTURE FAIL: no live engine output")
            return

        stab_info = getattr(out, "stab_info", None)
        if not isinstance(stab_info, dict):
            self.lbl_cfg_status.setText("Status: CAPTURE FAIL: no stab_info")
            return

        current_measure = stab_info.get("current_notch_measure")
        if not isinstance(current_measure, dict):
            self.lbl_cfg_status.setText("Status: CAPTURE FAIL: no current_notch_measure")
            return

        center_rel = getattr(out, "center_rel", None)
        angle = getattr(out, "angle", None)
        roi_live = getattr(out, "roi_live", None)

        if center_rel is None or angle is None:
            self.lbl_cfg_status.setText("Status: CAPTURE FAIL: baseplate missing in live output")
            return

        frame_dx = float(current_measure.get("frame_dx", current_measure.get("dx", 0.0)))
        frame_dy = float(current_measure.get("frame_dy", current_measure.get("dy", 0.0)))

        cfg["expected_notch_frame"] = {
            "coord_model": "bottom_mid_local_frame",
            "dx": frame_dx,
            "dy": frame_dy,
            "frame_dx": frame_dx,
            "frame_dy": frame_dy,
            "sidewall_dx": float(current_measure.get("sidewall_dx", frame_dx)),
            "sidewall_dy": float(current_measure.get("sidewall_dy", frame_dy)),
            "relative_angle": float(current_measure.get("relative_angle", 0.0)),
            "notch_angle": float(current_measure.get("notch_angle", 0.0)),
        }

        cfg["expected_center"] = [
            float(center_rel[0]),
            float(center_rel[1]),
        ]
        cfg["expected_angle"] = float(angle)

        if roi_live is not None:
            cfg["roi"] = [int(v) for v in roi_live]
            cfg.setdefault("registration_roi", cfg["roi"])

        cfg["golden_image_path"] = "golden.png"

        scale = self._scale_from_live_stab(stab_info)
        if scale is not None:
            cfg = self._write_baseplate_scale_to_cfg(cfg, scale)

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

            enf = cfg["expected_notch_frame"]

            if scale is not None and "px_per_mm" in cfg:
                scale_txt = f" scale={float(cfg['px_per_mm']):.3f}px/mm"
            else:
                scale_txt = " scale=NOT SAVED"

            self.lbl_cfg_status.setText(
                "Status: GOLDEN + EXPECTED SAVED  "
                f"dx={enf['dx']:.1f}px dy={enf['dy']:.1f}px "
                f"relTheta={enf['relative_angle']:.1f}"
                f"{scale_txt}"
            )

        except Exception as e:
            self.lbl_cfg_status.setText(f"Status: SAVED IMAGE, JSON FAIL: {e}")

    def save_product_json(self):
        if self.engine.recipe is None:
            self.lbl_save.setText("Saved: NO PRODUCT")
            return

        recipe = self.engine.recipe
        cfg = self._cfg_with_current_tuning(recipe.cfg)

        out = self._force_live_engine_output_with_cfg(cfg)
        stab_info = getattr(out, "stab_info", None) if out is not None else None
        scale = self._scale_from_live_stab(stab_info)

        if scale is not None:
            cfg = self._write_baseplate_scale_to_cfg(cfg, scale)
        else:
            existing_px_per_mm = self._get_saved_px_per_mm(recipe.cfg)
            if existing_px_per_mm is not None:
                cfg["px_per_mm"] = float(existing_px_per_mm)
                cfg["mm_per_px"] = float(1.0 / existing_px_per_mm)
                cfg["baseplate_px_per_mm"] = float(existing_px_per_mm)

                old_scale = recipe.cfg.get("baseplate_scale")
                if isinstance(old_scale, dict):
                    cfg["baseplate_scale"] = dict(old_scale)
                else:
                    cfg["baseplate_scale"] = {
                        "px_per_mm": float(existing_px_per_mm),
                        "mm_per_px": float(1.0 / existing_px_per_mm),
                        "baseplate_width_mm": float(cfg["baseplate_width_mm"]),
                        "baseplate_height_mm": float(cfg["baseplate_height_mm"]),
                    }

        product_dir = Path(recipe.recipe_dir)
        product_dir.mkdir(parents=True, exist_ok=True)

        try:
            _safe_write_json(str(product_dir / "golden_config.json"), cfg)

            recipe.cfg.clear()
            recipe.cfg.update(cfg)

            self._live_dirty = False
            self._update_expected_label()

            if scale is not None and "px_per_mm" in cfg:
                self.lbl_save.setText(f"Saved: golden_config.json  scale={float(cfg['px_per_mm']):.3f}px/mm")
            else:
                self.lbl_save.setText("Saved: golden_config.json  scale=kept/unchanged")

            self.lbl_live.setText("Live: saved JSON parameters are active")

        except Exception as e:
            self.lbl_save.setText(f"Saved: FAIL ({e})")

    # -------------------------
    # Shutdown
    # -------------------------
    def close(self):
        try:
            self.stop_preview()
            if self._owns_cam and self.cam is not None:
                self.cam.release()
        except Exception:
            pass
