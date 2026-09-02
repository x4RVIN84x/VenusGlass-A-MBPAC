from __future__ import annotations

from PySide6.QtCore import Qt, QSettings
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QSlider,
    QCheckBox,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QComboBox,
)

from hmi_app.gui.localization import AppLocalization


class OptionsPage(QWidget):
    """Workstation-level accessibility and appearance preferences."""

    SETTINGS_ORGANIZATION = "Venus Glass"
    SETTINGS_APPLICATION = "MBPAC QC Station"

    def __init__(self, *, auto_page, navigation_controller=None, localizer: AppLocalization | None = None, parent=None):
        super().__init__(parent)

        self.auto_page = auto_page
        self.navigation_controller = navigation_controller
        self._settings = QSettings(self.SETTINGS_ORGANIZATION, self.SETTINGS_APPLICATION)
        self.localizer = localizer or AppLocalization(self._settings, parent=self)
        self._loading = False

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 28, 28, 28)
        root.setSpacing(16)

        title = QLabel("Options")
        title.setStyleSheet("font-size: 24px; font-weight: 1000;")
        root.addWidget(title)

        subtitle = QLabel(
            "These settings affect this workstation only. They keep Auto Mode focused on inspection."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("font-size: 14px; color: #b7bfcd;")
        root.addWidget(subtitle)

        regional = QGroupBox("Language & Regional Settings")
        regional_lay = QVBoxLayout(regional)
        regional_lay.setSpacing(9)

        regional_lay.addWidget(QLabel("Application language"))
        self.cmb_language = QComboBox()
        self.cmb_language.addItem("English", "en")
        self.cmb_language.addItem("Farsi (فارسی)", "fa")
        self.cmb_language.setToolTip("Switches the operator interface between English and Farsi immediately.")
        regional_lay.addWidget(self.cmb_language)

        self.chk_jalali_calendar = QCheckBox("Use Solar Hijri (Jalali) calendar")
        self.chk_jalali_calendar.setToolTip(
            "Reports use Jalali dates and weeks run from Saturday through Friday."
        )
        regional_lay.addWidget(self.chk_jalali_calendar)
        jalali_note = QLabel("Weeks run Saturday through Friday")
        jalali_note.setStyleSheet("font-size: 13px; color: #b7bfcd;")
        regional_lay.addWidget(jalali_note)
        root.addWidget(regional)

        accessibility = QGroupBox("Accessibility & Interface")
        accessibility_lay = QVBoxLayout(accessibility)
        accessibility_lay.setSpacing(10)

        accessibility_lay.addWidget(QLabel("Auto Mode control and button size"))
        self.lbl_control_size = QLabel()
        self.lbl_control_size.setStyleSheet("font-size: 14px; font-weight: 900;")
        accessibility_lay.addWidget(self.lbl_control_size)

        self.sld_control_size = QSlider(Qt.Horizontal)
        self.sld_control_size.setRange(80, 140)
        self.sld_control_size.setSingleStep(5)
        self.sld_control_size.setPageStep(10)
        self.sld_control_size.setToolTip("Makes the Auto Mode controls, labels, and action buttons easier to read.")
        accessibility_lay.addWidget(self.sld_control_size)

        self.chk_reduce_motion = QCheckBox("Reduce visual alarm flashing")
        self.chk_reduce_motion.setToolTip("Keeps the visual alarm visible with a steady, lower-intensity warning.")
        accessibility_lay.addWidget(self.chk_reduce_motion)
        root.addWidget(accessibility)

        alarm = QGroupBox("Alarm")
        alarm_lay = QVBoxLayout(alarm)
        alarm_lay.setSpacing(10)

        self.chk_alarm_sound = QCheckBox("Play alarm sound")
        self.chk_alarm_sound.setToolTip("Turn this off to retain the visual lost-track alarm without sound.")
        alarm_lay.addWidget(self.chk_alarm_sound)

        alarm_note = QLabel("The visual lost-track alarm itself remains controlled in Auto Mode.")
        alarm_note.setWordWrap(True)
        alarm_note.setStyleSheet("font-size: 13px; color: #b7bfcd;")
        alarm_lay.addWidget(alarm_note)
        root.addWidget(alarm)

        navigation = QGroupBox("Navigation Order")
        navigation_lay = QVBoxLayout(navigation)
        navigation_lay.setSpacing(8)

        nav_note = QLabel("Choose a tab and use the arrows to change the left-hand tab order.")
        nav_note.setWordWrap(True)
        nav_note.setStyleSheet("font-size: 13px; color: #b7bfcd;")
        navigation_lay.addWidget(nav_note)

        self.list_navigation = QListWidget()
        self.list_navigation.setMinimumHeight(150)
        self.list_navigation.setToolTip("The selected item moves when you use the up or down buttons.")
        navigation_lay.addWidget(self.list_navigation)

        nav_buttons = QHBoxLayout()
        self.btn_tab_up = QPushButton("MOVE UP")
        self.btn_tab_down = QPushButton("MOVE DOWN")
        nav_buttons.addWidget(self.btn_tab_up)
        nav_buttons.addWidget(self.btn_tab_down)
        navigation_lay.addLayout(nav_buttons)
        root.addWidget(navigation)

        self.btn_reset = QPushButton("RESTORE DEFAULT OPTIONS")
        self.btn_reset.setMinimumHeight(42)
        root.addWidget(self.btn_reset)

        self.lbl_saved = QLabel("Options are saved automatically for this workstation.")
        self.lbl_saved.setStyleSheet("font-size: 13px; font-weight: 800; color: #8ec5ff;")
        root.addWidget(self.lbl_saved)
        root.addStretch(1)

        self.sld_control_size.valueChanged.connect(self._on_control_size_changed)
        self.chk_reduce_motion.toggled.connect(self._on_reduce_motion_changed)
        self.chk_alarm_sound.toggled.connect(self._on_alarm_sound_changed)
        self.cmb_language.currentIndexChanged.connect(self._on_language_changed)
        self.chk_jalali_calendar.toggled.connect(self._on_jalali_calendar_changed)
        self.btn_tab_up.clicked.connect(lambda: self._move_selected_tab(-1))
        self.btn_tab_down.clicked.connect(lambda: self._move_selected_tab(1))
        self.btn_reset.clicked.connect(self._restore_defaults)

        self._load_preferences()

    @staticmethod
    def _as_bool(value, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        if value is None:
            return default
        return bool(value)

    def _load_preferences(self):
        self._loading = True
        try:
            control_size = int(self._settings.value("accessibility/control_size_percent", 100))
            sound_enabled = self._as_bool(self._settings.value("alarm/sound_enabled", True), True)
            reduce_motion = self._as_bool(self._settings.value("accessibility/reduce_alarm_motion", False), False)

            self.sld_control_size.setValue(max(80, min(140, control_size)))
            self.chk_alarm_sound.setChecked(sound_enabled)
            self.chk_reduce_motion.setChecked(reduce_motion)
            language_index = self.cmb_language.findData(self.localizer.language)
            self.cmb_language.setCurrentIndex(max(0, language_index))
            self.chk_jalali_calendar.setChecked(self.localizer.use_jalali)

            self.auto_page.set_control_size_percent(self.sld_control_size.value())
            self.auto_page.set_alarm_sound_enabled(sound_enabled)
            self.auto_page.set_reduce_alarm_motion(reduce_motion)
            self._update_control_size_label()
            self._load_navigation_order()
        finally:
            self._loading = False

    def _update_control_size_label(self):
        prefix = self.localizer.tr("Current size:")
        self.lbl_control_size.setText(f"{prefix} {self.sld_control_size.value()}%")

    def _save(self, key: str, value):
        if self._loading:
            return
        self._settings.setValue(key, value)
        self._settings.sync()
        self.lbl_saved.setText(self.localizer.tr("Saved for this workstation."))

    def _on_control_size_changed(self, value: int):
        self.auto_page.set_control_size_percent(value)
        self._update_control_size_label()
        self._save("accessibility/control_size_percent", int(value))

    def _on_reduce_motion_changed(self, enabled: bool):
        self.auto_page.set_reduce_alarm_motion(enabled)
        self._save("accessibility/reduce_alarm_motion", bool(enabled))

    def _on_alarm_sound_changed(self, enabled: bool):
        self.auto_page.set_alarm_sound_enabled(enabled)
        self._save("alarm/sound_enabled", bool(enabled))

    def _on_language_changed(self, _index: int):
        if self._loading:
            return
        self.localizer.set_language(str(self.cmb_language.currentData() or "en"))
        self.lbl_saved.setText(self.localizer.tr("Saved for this workstation."))

    def _on_jalali_calendar_changed(self, enabled: bool):
        if self._loading:
            return
        self.localizer.set_use_jalali(enabled)
        self.lbl_saved.setText(self.localizer.tr("Saved for this workstation."))

    def _load_navigation_order(self):
        if self.navigation_controller is None:
            self.list_navigation.setEnabled(False)
            self.btn_tab_up.setEnabled(False)
            self.btn_tab_down.setEnabled(False)
            return

        default_order = self.navigation_controller.default_navigation_order()
        saved_order = self._settings.value("navigation/tab_order", default_order)
        if isinstance(saved_order, str):
            saved_order = [part.strip() for part in saved_order.split(",") if part.strip()]
        if not isinstance(saved_order, (list, tuple)):
            saved_order = default_order

        self._apply_navigation_order(saved_order)

    def _apply_navigation_order(self, order):
        if self.navigation_controller is None:
            return []

        applied = self.navigation_controller.set_navigation_order(order)
        labels = self.navigation_controller.navigation_labels()

        self.list_navigation.clear()
        for key in applied:
            item = QListWidgetItem(labels.get(key, str(key).replace("_", " ").title()))
            item.setData(Qt.UserRole, key)
            self.list_navigation.addItem(item)

        if self.list_navigation.count():
            self.list_navigation.setCurrentRow(0)

        return applied

    def _move_selected_tab(self, delta: int):
        if self.navigation_controller is None:
            return

        row = self.list_navigation.currentRow()
        target = row + int(delta)
        if row < 0 or target < 0 or target >= self.list_navigation.count():
            return

        order = [
            self.list_navigation.item(index).data(Qt.UserRole)
            for index in range(self.list_navigation.count())
        ]
        order[row], order[target] = order[target], order[row]
        applied = self._apply_navigation_order(order)
        self.list_navigation.setCurrentRow(target)
        self._save("navigation/tab_order", applied)

    def _restore_defaults(self):
        self.sld_control_size.setValue(100)
        self.chk_alarm_sound.setChecked(True)
        self.chk_reduce_motion.setChecked(False)
        self.localizer.set_language("en")
        self.localizer.set_use_jalali(False)
        self._loading = True
        try:
            self.cmb_language.setCurrentIndex(0)
            self.chk_jalali_calendar.setChecked(False)
        finally:
            self._loading = False
        if self.navigation_controller is not None:
            order = self._apply_navigation_order(self.navigation_controller.default_navigation_order())
            self._save("navigation/tab_order", order)
        self.lbl_saved.setText(self.localizer.tr("Restored the default options for this workstation."))

    def retranslate_ui(self):
        """Refresh strings whose contents are rebuilt outside Qt's translator system."""
        self.localizer.apply_widget_text(self)
        self._update_control_size_label()
        self._load_navigation_order()
