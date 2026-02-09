from __future__ import annotations
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QFrame, QPushButton,
    QLabel, QComboBox, QStackedWidget
)

from hmi_app.gui.styles import industrial_dark_stylesheet
from hmi_app.core.recipe_manager import RecipeManager
from hmi_app.core.engine import QCPreviewEngine
from hmi_app.gui.pages.auto_page import AutoPage
from hmi_app.gui.pages.placeholder import PlaceholderPage
from pathlib import Path


class MainWindow(QMainWindow):
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

        root = QWidget()
        self.setCentralWidget(root)

        outer = QVBoxLayout(root)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(12)

        # Top bar
        top = QFrame()
        top.setFixedHeight(64)
        top_lay = QHBoxLayout(top)
        top_lay.setContentsMargins(14, 10, 14, 10)
        top_lay.setSpacing(12)

        lbl_title = QLabel("MBPAC QC Station")
        lbl_title.setStyleSheet("font-size: 18px; font-weight: 900;")
        top_lay.addWidget(lbl_title)

        top_lay.addStretch(1)

        lbl_recipe = QLabel("Recipe/Car:")
        self.cmb_recipe = QComboBox()

        # --- Populate recipe dropdown ---
        recipes = self.recipe_manager.list_recipes()
        self.cmb_recipe.clear()
        self.cmb_recipe.addItems(recipes)

        # --- Wire selection change ---
        self.cmb_recipe.currentTextChanged.connect(self.on_recipe_changed)

        # Auto-select first recipe (if any)
        if self.cmb_recipe.count() > 0:
            self.cmb_recipe.setCurrentIndex(0)
            self.on_recipe_changed(self.cmb_recipe.currentText())
        else:
            self.engine.recipe = None
            self.badge.setText("IDLE")

        self._load_recipe_list()

        self.lbl_state = QLabel("IDLE")
        self.lbl_state.setFixedWidth(140)
        self.lbl_state.setStyleSheet("background:#1f1f22; border:1px solid #2d2d31; border-radius: 12px; font-weight: 900; padding: 8px;")

        top_lay.addWidget(lbl_recipe)
        top_lay.addWidget(self.cmb_recipe, 0)
        top_lay.addWidget(self.lbl_state, 0)

        outer.addWidget(top)

        # Body
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

        self.page_cal = PlaceholderPage("Calibration (coming next)")
        self.page_manual = PlaceholderPage("Manual Test (coming next)")
        self.page_auto = AutoPage(engine=self.engine)
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

        # Recipe wiring
        self.cmb_recipe.currentTextChanged.connect(self.on_recipe_changed)
        if self.cmb_recipe.count() > 0:
            self.on_recipe_changed(self.cmb_recipe.currentText())

    def _load_recipe_list(self):
        self.cmb_recipe.clear()
        for r in self.recipe_manager.list_recipes():
            self.cmb_recipe.addItem(r)

    def on_recipe_changed(self, name: str):
        name = (name or "").strip()
        if not name:
            self.engine.recipe = None
            self.badge.setText("IDLE")
            return

        try:
            recipe = self.recipe_manager.load(name)
            self.engine.recipe = recipe
            self.badge.setText("READY")
            print(f"[HMI] loaded recipe: {name}")
        except Exception as e:
            self.engine.recipe = None
            self.badge.setText("FAULT")
            print(f"[HMI] failed to load recipe {name}: {e}")

    def closeEvent(self, event):
        try:
            self.page_auto.cam.release()
        except Exception:
            pass
        super().closeEvent(event)
