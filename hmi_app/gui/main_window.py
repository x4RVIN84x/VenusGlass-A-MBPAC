from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QFrame,
    QPushButton,
    QLabel,
    QComboBox,
    QStackedWidget,
    QSizePolicy,
)

from hmi_app.gui.styles import industrial_dark_stylesheet
from hmi_app.core.recipe_manager import RecipeManager
from hmi_app.core.engine import QCPreviewEngine
from hmi_app.io.camera import OpenCVCamera

from hmi_app.gui.pages.auto_page import AutoPage
from hmi_app.gui.pages.calibration_page import CalibrationPage
from hmi_app.gui.pages.manual_test_page import ManualTestPage
from hmi_app.gui.pages.placeholder import PlaceholderPage


class MainWindow(QMainWindow):
    """
    Main app shell.

    Important terminology cleanup:
      - The top dropdown is the PRODUCT / RECIPE selector.
      - There is no separate active config concept anymore.
      - One recipe folder == one complete product definition.
    """

    AUTO_CONTROLS_FIXED_W = 420

    def __init__(self):
        super().__init__()

        self.setWindowTitle("MBPAC QC Station (Industrial HMI)")
        self.resize(1600, 920)
        self.setStyleSheet(industrial_dark_stylesheet())

        repo_root = Path(__file__).resolve().parents[2]
        recipes_path = repo_root / "recipes"
        print("[HMI] repo_root:", repo_root)
        print("[HMI] recipes_path:", recipes_path)

        self.recipe_manager = RecipeManager(recipes_root=str(recipes_path))
        self.engine = QCPreviewEngine()

        # Shared camera for all pages.
        self.cam = OpenCVCamera(index=0, width=1280, height=720, fps=30, use_dshow=True)

        root = QWidget()
        self.setCentralWidget(root)

        outer = QVBoxLayout(root)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(12)

        # =========================
        # Top bar
        # =========================
        top = QFrame()
        top.setFixedHeight(64)
        top_lay = QHBoxLayout(top)
        top_lay.setContentsMargins(14, 10, 14, 10)
        top_lay.setSpacing(12)

        lbl_title = QLabel("MBPAC QC Station")
        lbl_title.setStyleSheet("font-size: 18px; font-weight: 900;")

        lbl_recipe = QLabel("Product/Recipe:")
        self.cmb_recipe = QComboBox()
        self.cmb_recipe.setMinimumWidth(240)

        self.badge = QLabel("IDLE")
        self.badge.setAlignment(Qt.AlignCenter)
        self.badge.setFixedHeight(28)
        self.badge.setMinimumWidth(90)
        self.badge.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.badge.setStyleSheet(
            """
            QLabel {
                background: #2a2a2e;
                color: #cfcfcf;
                border-radius: 14px;
                font-weight: 800;
                padding: 4px 12px;
            }
            """
        )

        top_lay.addWidget(lbl_title)
        top_lay.addStretch(1)
        top_lay.addWidget(lbl_recipe)
        top_lay.addWidget(self.cmb_recipe, 0)
        top_lay.addWidget(self.badge, 0)
        outer.addWidget(top)

        # =========================
        # Body
        # =========================
        body = QHBoxLayout()
        body.setSpacing(12)
        outer.addLayout(body, 1)

        nav = QFrame()
        nav.setFixedWidth(240)
        nav_lay = QVBoxLayout(nav)
        nav_lay.setContentsMargins(12, 12, 12, 12)
        nav_lay.setSpacing(10)

        self.btn_cal = QPushButton("CALIBRATION")
        self.btn_manual = QPushButton("MANUAL TEST")
        self.btn_auto = QPushButton("AUTO MODE")
        self.btn_reports = QPushButton("REPORTS")

        for b in (self.btn_cal, self.btn_manual, self.btn_auto, self.btn_reports):
            b.setMinimumHeight(56)
            nav_lay.addWidget(b)

        nav_lay.addStretch(1)
        body.addWidget(nav)

        self.stack = QStackedWidget()
        body.addWidget(self.stack, 1)

        # Real pages
        self.page_cal = CalibrationPage(
            engine=self.engine,
            recipe_manager=self.recipe_manager,
            cam=self.cam,
            on_products_changed=self.refresh_product_list,
        )
        self.page_manual = ManualTestPage(engine=self.engine, recipe_manager=self.recipe_manager, cam=self.cam)
        self.page_auto = AutoPage(engine=self.engine, cam=self.cam)
        self.page_reports = PlaceholderPage("Reports (coming next)")

        self.stack.addWidget(self.page_cal)
        self.stack.addWidget(self.page_manual)
        self.stack.addWidget(self.page_auto)
        self.stack.addWidget(self.page_reports)

        self.btn_cal.clicked.connect(lambda: self.stack.setCurrentWidget(self.page_cal))
        self.btn_manual.clicked.connect(lambda: self.stack.setCurrentWidget(self.page_manual))
        self.btn_auto.clicked.connect(lambda: self.stack.setCurrentWidget(self.page_auto))
        self.btn_reports.clicked.connect(lambda: self.stack.setCurrentWidget(self.page_reports))

        self.stack.setCurrentWidget(self.page_auto)

        # =========================
        # Recipe wiring
        # =========================
        self._loading_products = False
        self.refresh_product_list()
        self.cmb_recipe.currentTextChanged.connect(self.on_recipe_changed)

        if self.cmb_recipe.count() > 0:
            self.cmb_recipe.setCurrentIndex(0)
            self.on_recipe_changed(self.cmb_recipe.currentText())
        else:
            self.engine.recipe = None
            self.set_state("IDLE")

        QTimer.singleShot(0, self._stabilize_auto_layout)

    # -------------------------
    # Product list/load
    # -------------------------
    def refresh_product_list(self, select_name: str | None = None):
        current = (select_name or self.cmb_recipe.currentText() or "").strip()

        self._loading_products = True
        self.cmb_recipe.blockSignals(True)
        try:
            self.cmb_recipe.clear()
            products = self.recipe_manager.list_recipes()
            for p in products:
                self.cmb_recipe.addItem(p)

            if products:
                if current in products:
                    self.cmb_recipe.setCurrentText(current)
                else:
                    self.cmb_recipe.setCurrentIndex(0)
        finally:
            self.cmb_recipe.blockSignals(False)
            self._loading_products = False

        if self.cmb_recipe.count() > 0:
            self.on_recipe_changed(self.cmb_recipe.currentText())

    def on_recipe_changed(self, name: str):
        if self._loading_products:
            return

        name = (name or "").strip()
        if not name:
            self.engine.recipe = None
            self.set_state("IDLE")
            return

        try:
            recipe = self.recipe_manager.load(name)
            self.engine.set_recipe(recipe)
            self.set_state("READY")
            print(f"[HMI] loaded product/recipe: {recipe.name}")

            # Notify pages.
            try:
                self.page_cal.set_recipe_name(name)
            except Exception as e:
                print("[HMI] page_cal.set_recipe_name failed:", e)

            try:
                self.page_manual.set_recipe_name(name)
            except Exception as e:
                print("[HMI] page_manual.set_recipe_name failed:", e)

        except Exception as e:
            self.engine.recipe = None
            self.set_state("FAULT")
            print(f"[HMI] failed to load product/recipe {name}: {e}")

    # -------------------------
    # State badge
    # -------------------------
    def set_state(self, text: str):
        colors = {
            "IDLE": "#2a2a2e",
            "READY": "#1f3d2b",
            "RUNNING": "#1f2f3d",
            "FAULT": "#3d1f1f",
        }
        bg = colors.get(text, "#2a2a2e")

        if hasattr(self, "badge") and self.badge is not None:
            self.badge.setText(text)
            self.badge.setStyleSheet(
                f"""
                QLabel {{
                    background: {bg};
                    color: #ffffff;
                    border-radius: 14px;
                    font-weight: 800;
                    padding: 4px 12px;
                }}
                """
            )

    # -------------------------
    # Layout stabilization
    # -------------------------
    def _stabilize_auto_layout(self):
        try:
            page = getattr(self, "page_auto", None)
            if page is None:
                return

            fixed_w = int(self.AUTO_CONTROLS_FIXED_W)
            for attr in ("auto_controls_panel", "controls_panel", "right_panel", "panel_right", "auto_controls"):
                w = getattr(page, attr, None)
                if w is not None and isinstance(w, QWidget):
                    w.setFixedWidth(fixed_w)
                    w.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
                    break
        except Exception as e:
            print("[HMI] _stabilize_auto_layout error:", e)

    # -------------------------
    # Shutdown
    # -------------------------
    def closeEvent(self, event):
        try:
            if getattr(self, "page_auto", None) is not None:
                self.page_auto.stop()
        except Exception:
            pass

        try:
            if getattr(self, "page_cal", None) is not None:
                self.page_cal.stop_preview()
        except Exception:
            pass

        try:
            if getattr(self, "cam", None) is not None:
                self.cam.release()
        except Exception:
            pass

        super().closeEvent(event)
