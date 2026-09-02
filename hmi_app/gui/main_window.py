from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QStandardPaths
from PySide6.QtWidgets import (
    QApplication,
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
from hmi_app.gui.localization import AppLocalization
from hmi_app.core.recipe_manager import RecipeManager
from hmi_app.core.engine import QCPreviewEngine
from hmi_app.core.report_store import ReportStore
from hmi_app.io.camera import OpenCVCamera

from hmi_app.gui.pages.auto_page import AutoPage
from hmi_app.gui.pages.calibration_page import CalibrationPage
from hmi_app.gui.pages.manual_test_page import ManualTestPage
from hmi_app.gui.pages.options_page import OptionsPage
from hmi_app.gui.pages.reports_page import ReportsPage


class MainWindow(QMainWindow):
    """
    Main app shell.

    Important terminology cleanup:
      - The top dropdown is the PRODUCT / RECIPE selector.
      - There is no separate active config concept anymore.
      - One recipe folder == one complete product definition.
    """

    AUTO_CONTROLS_FIXED_W = 420
    DEFAULT_NAVIGATION_ORDER = ("auto", "calibration", "reports", "manual", "options")

    def __init__(self):
        super().__init__()

        self.localization = AppLocalization(parent=self)
        self._state_key = "IDLE"
        self.setWindowTitle(self.localization.tr("MBPAC QC Station (Industrial HMI)"))
        self.resize(1600, 920)
        self.setStyleSheet(industrial_dark_stylesheet())

        repo_root = Path(__file__).resolve().parents[2]
        recipes_path = repo_root / "recipes"
        print("[HMI] repo_root:", repo_root)
        print("[HMI] recipes_path:", recipes_path)

        self.recipe_manager = RecipeManager(recipes_root=str(recipes_path))
        self.engine = QCPreviewEngine()
        app_data = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
        report_root = Path(app_data) if app_data else (repo_root / "output")
        self.report_store = ReportStore(report_root / "inspection_history.sqlite3")

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

        self.btn_auto = QPushButton("AUTO MODE")
        self.btn_cal = QPushButton("CALIBRATION")
        self.btn_reports = QPushButton("REPORTS")
        self.btn_manual = QPushButton("MANUAL TEST")
        self.btn_options = QPushButton("OPTIONS")

        self._nav_layout = nav_lay
        self._nav_buttons = {
            "auto": self.btn_auto,
            "calibration": self.btn_cal,
            "reports": self.btn_reports,
            "manual": self.btn_manual,
            "options": self.btn_options,
        }
        self._navigation_order = list(self.DEFAULT_NAVIGATION_ORDER)

        for key in self._navigation_order:
            b = self._nav_buttons[key]
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
        self.page_manual = ManualTestPage(
            engine=self.engine,
            recipe_manager=self.recipe_manager,
            cam=self.cam,
            report_store=self.report_store,
        )
        self.page_auto = AutoPage(engine=self.engine, cam=self.cam, report_store=self.report_store)
        self.page_cal.configurationChanged.connect(self.page_auto.refresh_recipe_summary)
        self.page_options = OptionsPage(
            auto_page=self.page_auto,
            navigation_controller=self,
            localizer=self.localization,
        )
        self.page_reports = ReportsPage(
            report_store=self.report_store,
            recipe_manager=self.recipe_manager,
            localizer=self.localization,
        )

        # Pages keep the manager available for live operator-status wording.
        self.page_auto.localizer = self.localization
        self.page_cal.localizer = self.localization
        self.page_manual.localizer = self.localization

        self.stack.addWidget(self.page_cal)
        self.stack.addWidget(self.page_manual)
        self.stack.addWidget(self.page_auto)
        self.stack.addWidget(self.page_options)
        self.stack.addWidget(self.page_reports)

        self.btn_cal.clicked.connect(lambda: self._show_page(self.page_cal))
        self.btn_manual.clicked.connect(lambda: self._show_page(self.page_manual))
        self.btn_auto.clicked.connect(lambda: self._show_page(self.page_auto))
        self.btn_options.clicked.connect(lambda: self._show_page(self.page_options))
        self.btn_reports.clicked.connect(lambda: self._show_page(self.page_reports))

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

        self.localization.changed.connect(self._apply_localization)
        self._apply_localization()
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

            try:
                self.page_auto.refresh_recipe_summary()
            except Exception as e:
                print("[HMI] page_auto.refresh_recipe_summary failed:", e)

        except Exception as e:
            self.engine.recipe = None
            self.set_state("FAULT")
            print(f"[HMI] failed to load product/recipe {name}: {e}")

    # -------------------------
    # Camera/page lifecycle
    # -------------------------
    def _show_page(self, page):
        """Do not let hidden pages compete for the shared camera stream."""
        if page is not self.page_auto:
            try:
                self.page_auto.stop()
            except Exception:
                pass

        if page is not self.page_cal:
            try:
                self.page_cal.stop_preview()
            except Exception:
                pass

        self.stack.setCurrentWidget(page)

    # -------------------------
    # Navigation layout
    # -------------------------
    def navigation_order(self) -> list[str]:
        return list(self._navigation_order)

    def navigation_labels(self) -> dict[str, str]:
        return {
            "auto": self.localization.tr("Auto Mode"),
            "calibration": self.localization.tr("Calibration"),
            "reports": self.localization.tr("Reports"),
            "manual": self.localization.tr("Manual Test"),
            "options": self.localization.tr("Options"),
        }

    def default_navigation_order(self) -> list[str]:
        return list(self.DEFAULT_NAVIGATION_ORDER)

    def set_navigation_order(self, requested_order) -> list[str]:
        """Apply a valid operator-selected left-nav order without recreating pages."""
        requested = requested_order if isinstance(requested_order, (list, tuple)) else []
        order = []

        for key in requested:
            key = str(key)
            if key in self._nav_buttons and key not in order:
                order.append(key)

        for key in self.DEFAULT_NAVIGATION_ORDER:
            if key not in order:
                order.append(key)

        for button in self._nav_buttons.values():
            self._nav_layout.removeWidget(button)

        for index, key in enumerate(order):
            self._nav_layout.insertWidget(index, self._nav_buttons[key])

        self._navigation_order = order
        return self.navigation_order()

    # -------------------------
    # State badge
    # -------------------------
    def set_state(self, text: str):
        self._state_key = str(text or "IDLE").upper()
        colors = {
            "IDLE": "#2a2a2e",
            "READY": "#1f3d2b",
            "RUNNING": "#1f2f3d",
            "FAULT": "#3d1f1f",
        }
        bg = colors.get(self._state_key, "#2a2a2e")

        if hasattr(self, "badge") and self.badge is not None:
            self.badge.setText(self.localization.tr(self._state_key))
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

    def _apply_localization(self):
        """Refresh app text, direction, and report-calendar presentation live."""
        app = QApplication.instance()
        if app is not None:
            app.setLayoutDirection(Qt.RightToLeft if self.localization.is_farsi else Qt.LeftToRight)

        self.setWindowTitle(self.localization.tr("MBPAC QC Station (Industrial HMI)"))
        self.localization.apply_widget_text(self)
        self.set_state(self._state_key)

        try:
            self.page_options.retranslate_ui()
        except Exception as e:
            print("[HMI] page_options.retranslate_ui failed:", e)
        try:
            self.page_reports.apply_localization()
        except Exception as e:
            print("[HMI] page_reports.apply_localization failed:", e)
        try:
            self.page_auto.retranslate_ui()
        except Exception as e:
            print("[HMI] page_auto.retranslate_ui failed:", e)
        try:
            self.page_manual.retranslate_ui()
        except Exception as e:
            print("[HMI] page_manual.retranslate_ui failed:", e)
        try:
            self.page_cal.retranslate_ui()
        except Exception as e:
            print("[HMI] page_cal.retranslate_ui failed:", e)

        # The options-side navigation labels may have changed language.
        self.set_navigation_order(self._navigation_order)

    # -------------------------
    # Layout stabilization
    # -------------------------
    def _stabilize_auto_layout(self):
        try:
            page = getattr(self, "page_auto", None)
            if page is None:
                return

            apply_control_size = getattr(page, "_apply_control_size", None)
            if callable(apply_control_size):
                apply_control_size()
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
