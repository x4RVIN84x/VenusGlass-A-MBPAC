# hmi_app/gui/main_window.py
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
from hmi_app.gui.pages.auto_page import AutoPage
from hmi_app.gui.pages.placeholder import PlaceholderPage


class MainWindow(QMainWindow):
    """
    Fix for the "video flickers between two sizes" when SEARCH <-> TRACK/PASS toggles:

    Root cause is almost always layout width changes (typically a status QLabel in the Auto controls
    panel changing sizeHint based on text length). That causes the right panel width to change a few px,
    the center video area changes a few px, and the feed appears to "breathe".

    We stabilize it here by forcing a fixed width on the AutoPage's right control panel (if we can find it),
    plus a couple safe fallback clamps on any obvious status labels.
    """

    AUTO_CONTROLS_FIXED_W = 380  # tweak if you want wider/narrower (360-420 typical)

    def __init__(self):
        super().__init__()

        self.setWindowTitle("MBPAC QC Station (Industrial HMI)")
        self.resize(1600, 920)
        self.setStyleSheet(industrial_dark_stylesheet())

        # Resolve repo root + recipes folder (repo_root/.../hmi_app/gui/main_window.py -> parents[2] == repo root)
        repo_root = Path(__file__).resolve().parents[2]
        recipes_path = repo_root / "recipes"
        print("[HMI] repo_root:", repo_root)
        print("[HMI] recipes_path:", recipes_path)

        # Core objects
        self.recipe_manager = RecipeManager(recipes_root=str(recipes_path))
        self.engine = QCPreviewEngine()

        # Root widget + outer layout
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

        # Title (left)
        lbl_title = QLabel("MBPAC QC Station")
        lbl_title.setStyleSheet("font-size: 18px; font-weight: 900;")

        # Recipe dropdown (right-side controls)
        lbl_recipe = QLabel("Recipe/Car:")
        self.cmb_recipe = QComboBox()

        # Status badge (top-right)
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


        # Layout order
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

        # Left nav
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

        # Pages stack
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

        # =========================
        # Recipe wiring
        # =========================
        self._load_recipe_list()
        self.cmb_recipe.currentTextChanged.connect(self.on_recipe_changed)

        if self.cmb_recipe.count() > 0:
            # Trigger load for the initial selection
            self.cmb_recipe.setCurrentIndex(0)
            self.on_recipe_changed(self.cmb_recipe.currentText())
        else:
            self.engine.recipe = None
            self.set_state("IDLE")

        # =========================
        # Layout stabilization (kills SEARCH<->TRACK breathing)
        # =========================
        # Delay a tick so AutoPage has built its child widgets/layouts.
        QTimer.singleShot(0, self._stabilize_auto_layout)

    def _load_recipe_list(self):
        """Populate the recipe combobox from the recipes folder."""
        self.cmb_recipe.blockSignals(True)
        try:
            self.cmb_recipe.clear()
            for r in self.recipe_manager.list_recipes():
                self.cmb_recipe.addItem(r)
        finally:
            self.cmb_recipe.blockSignals(False)

    def on_recipe_changed(self, name: str):
        name = (name or "").strip()
        if not name:
            self.engine.recipe = None
            self.set_state("IDLE")
            return

        try:
            recipe = self.recipe_manager.load(name)
            self.engine.recipe = recipe
            self.set_state("READY")
            print(f"[HMI] loaded recipe: {name}")
        except Exception as e:
            self.engine.recipe = None
            self.set_state("FAULT")
            print(f"[HMI] failed to load recipe {name}: {e}")

    def set_state(self, text: str):
        colors = {
            "IDLE": "#2a2a2e",
            "READY": "#1f3d2b",
            "RUNNING": "#1f2f3d",
            "FAULT": "#3d1f1f",
        }
        bg = colors.get(text, "#2a2a2e")

        # Update both (badge + legacy label) safely
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

        if hasattr(self, "lbl_state") and self.lbl_state is not None:
            self.lbl_state.setText(text)

    def _stabilize_auto_layout(self):
        """
        Find the Auto Controls panel (right side) and freeze its width so that
        changing status text length can't resize the panel and "breathe" the video.
        """
        try:
            page = getattr(self, "page_auto", None)
            if page is None:
                return

            fixed_w = int(self.AUTO_CONTROLS_FIXED_W)

            # 1) Best case: AutoPage exposes an attribute for the panel.
            for attr in ("auto_controls_panel", "controls_panel", "right_panel", "panel_right", "auto_controls"):
                w = getattr(page, attr, None)
                if w is not None and isinstance(w, QWidget):
                    w.setFixedWidth(fixed_w)
                    w.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
                    print(f"[HMI] fixed Auto controls panel width via attr '{attr}' -> {fixed_w}px")
                    break
            else:
                # 2) Next: try common objectNames you might have set in AutoPage.
                found = None
                for name in ("auto_controls_panel", "controls_panel", "right_panel", "auto_controls"):
                    found = page.findChild(QWidget, name)
                    if found is not None:
                        found.setFixedWidth(fixed_w)
                        found.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
                        print(f"[HMI] fixed Auto controls panel width via objectName '{name}' -> {fixed_w}px")
                        break

                # 3) Fallback heuristic: pick the right-most wide-ish QFrame/QWidget.
                if found is None:
                    candidates = []
                    for w in page.findChildren(QWidget):
                        try:
                            # Prefer frames/panels, ignore tiny widgets
                            if w.isVisible() and w.width() >= 200 and w.height() >= 200:
                                candidates.append(w)
                        except Exception:
                            pass
                    # Heuristic: right-most widget by global x
                    best = None
                    best_x = -10**9
                    for w in candidates:
                        try:
                            gx = w.mapToGlobal(w.rect().topLeft()).x()
                            if gx > best_x:
                                best_x = gx
                                best = w
                        except Exception:
                            pass
                    if best is not None:
                        best.setFixedWidth(fixed_w)
                        best.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
                        print(f"[HMI] fixed Auto controls panel width via heuristic -> {fixed_w}px")

            # Backup clamp: any status labels that might be resizing things.
            # This doesn't break anything even if it misses the real label.
            for lbl in page.findChildren(QLabel):
                try:
                    t = (lbl.text() or "")
                    if "Status:" in t or t.strip().startswith("Status"):
                        lbl.setWordWrap(False)
                        lbl.setFixedHeight(max(lbl.height(), 22))
                        # Prevent it from expanding the panel
                        lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
                except Exception:
                    pass

        except Exception as e:
            print("[HMI] _stabilize_auto_layout error:", e)

    def closeEvent(self, event):
        # Don't crash on close if camera isn't present / already released
        try:
            cam = getattr(self.page_auto, "cam", None)
            if cam is not None:
                cam.release()
        except Exception:
            pass
        super().closeEvent(event)
