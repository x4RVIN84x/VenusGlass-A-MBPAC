from __future__ import annotations

from datetime import datetime, time, timedelta
import math
from typing import Optional

from PySide6.QtCore import QDate, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDateEdit,
    QFrame,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabBar,
    QTableWidget,
    QTableWidgetItem,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from hmi_app.core.report_store import ReportStore
from hmi_app.gui.localization import AppLocalization


class DonutChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._passed = 0
        self._failed = 0
        self.localizer: AppLocalization | None = None
        self.setMinimumSize(220, 220)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_counts(self, passed: int, failed: int):
        self._passed = max(0, int(passed))
        self._failed = max(0, int(failed))
        self.update()

    def sizeHint(self):
        return QSize(260, 240)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        total = self._passed + self._failed
        outer_side = max(80.0, min(self.width(), self.height()) - 24.0)
        ring = max(18.0, min(54.0, outer_side * 0.13))
        side = max(40.0, outer_side - ring)
        rect = QRectF((self.width() - side) / 2, (self.height() - side) / 2, side, side)

        painter.setPen(QPen(QColor("#2b2f38"), ring, Qt.SolidLine, Qt.RoundCap))
        painter.drawArc(rect, 90 * 16, -360 * 16)

        if total:
            pass_span = int(round(360 * 16 * self._passed / total))
            if self._passed:
                painter.setPen(QPen(QColor("#38d27b"), ring, Qt.SolidLine, Qt.RoundCap))
                painter.drawArc(rect, 90 * 16, -pass_span)
            if self._failed:
                painter.setPen(QPen(QColor("#ff6673"), ring, Qt.SolidLine, Qt.RoundCap))
                painter.drawArc(rect, 90 * 16 - pass_span, -(360 * 16 - pass_span))

        painter.setPen(QColor("#f3f5f8"))
        total_font = QFont("Segoe UI", max(18, int(side * 0.12)), QFont.Bold)
        painter.setFont(total_font)
        painter.drawText(rect, Qt.AlignCenter, str(total))
        painter.setPen(QColor("#b7bfcd"))
        painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
        label = self.localizer.tr("PROCESSED") if self.localizer is not None else "PROCESSED"
        painter.drawText(QRectF(rect.left(), rect.center().y() + side * 0.12, rect.width(), 24), Qt.AlignCenter, label)


class FailureCauseChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._data: dict[str, int] = {}
        self.localizer: AppLocalization | None = None
        self.setMinimumHeight(250)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_data(self, data: dict[str, int]):
        self._data = dict(data or {})
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(QFont("Segoe UI", 10))
        if not self._data:
            painter.setPen(QColor("#9ca6b8"))
            text = "No recorded failures in this period"
            painter.drawText(self.rect(), Qt.AlignCenter, self.localizer.tr(text) if self.localizer else text)
            return

        rows = sorted(self._data.items(), key=lambda item: (-item[1], item[0]))[:8]
        max_count = max(value for _label, value in rows) or 1
        left = 170
        right = 62
        top = 18
        row_h = max(25, min(38, (self.height() - 24) // max(1, len(rows))))
        bar_w = max(80, self.width() - left - right)

        bar_colors = ("#ff6673", "#e34d67", "#c93d5e", "#ff8a65", "#b34a6a", "#d65a8a")
        category_colors = {
            "BASEPLATE NOT FOUND": "#80461B",
            "X": "#D2042D",
            "Y": "#E97451",
            "ANGLE": "#A63D68",
        }
        for index, (label, count) in enumerate(rows):
            y = top + index * row_h
            display_label = self.localizer.tr(label) if self.localizer is not None else label
            text = display_label if len(display_label) <= 24 else display_label[:21] + "..."
            painter.setPen(QColor("#dbe4f2"))
            painter.drawText(QRectF(8, y, left - 18, row_h), Qt.AlignVCenter | Qt.AlignRight, text)

            track = QRectF(left, y + 6, bar_w, row_h - 12)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor("#282c35"))
            painter.drawRoundedRect(track, 5, 5)

            fill = QRectF(track.left(), track.top(), track.width() * count / max_count, track.height())
            painter.setBrush(QColor(category_colors.get(label.upper(), bar_colors[index % len(bar_colors)])))
            painter.drawRoundedRect(fill, 5, 5)

            painter.setPen(QColor("#f3f5f8"))
            painter.drawText(QRectF(left + bar_w + 10, y, 45, row_h), Qt.AlignVCenter | Qt.AlignLeft, str(count))


class TimelineChart(QWidget):
    """Stacked inspection columns with a period-aware time axis."""

    COLORS = {
        "PASS": "#38d27b",
        "BASEPLATE NOT FOUND": "#80461B",
        "X": "#D2042D",
        "Y": "#E97451",
        "ANGLE": "#A63D68",
        "OTHER FAIL": "#B24C63",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data: list[dict] = []
        self._granularity = "hour"
        self.localizer: AppLocalization | None = None
        self._plot_rect: Optional[QRectF] = None
        self._slot_width = 0.0
        self._last_tooltip = ""
        self.setMinimumHeight(310)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

    def set_data(self, data: list[dict], granularity: str):
        self._data = list(data or [])
        self._granularity = str(granularity or "day")
        self.update()

    def sizeHint(self):
        return QSize(900, 350)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#17191f"))

        left, right, top, bottom = 62, 20, 38, 56
        plot = QRectF(left, top, max(1, self.width() - left - right), max(1, self.height() - top - bottom))
        self._plot_rect = plot
        max_total = max((int(row.get("total", 0)) for row in self._data), default=0)
        scale_max, tick_step = self._scale_for(max_total)

        painter.setFont(QFont("Segoe UI", 9))
        painter.setPen(QColor("#9ca6b8"))
        painter.drawText(QRectF(0, 4, self.width(), 18), Qt.AlignCenter, "GLASSES PROCESSED")

        # Use a rounded scale with deliberate headroom so the tallest stack
        # and its exact total label always remain inside the plotting area.
        tick_count = int(math.ceil(scale_max / tick_step))
        for tick in range(tick_count + 1):
            value = tick * tick_step
            y = plot.bottom() - plot.height() * value / scale_max
            painter.setPen(QPen(QColor("#303642"), 1))
            painter.drawLine(plot.left(), y, plot.right(), y)
            painter.setPen(QColor("#9ca6b8"))
            label = str(int(value)) if float(value).is_integer() else f"{value:g}"
            painter.drawText(QRectF(0, y - 9, left - 10, 18), Qt.AlignRight | Qt.AlignVCenter, label)

        if not self._data:
            painter.setPen(QColor("#9ca6b8"))
            text = "No inspections recorded in this period"
            painter.drawText(plot, Qt.AlignCenter, self.localizer.tr(text) if self.localizer else text)
            return

        slot = plot.width() / max(1, len(self._data))
        self._slot_width = slot
        bar_width = max(8.0, min(52.0, slot * 0.72))
        # A day view is always labelled hour-by-hour.  Other ranges trim
        # labels only when their bucket count would make the axis unreadable.
        label_every = 1 if self._granularity == "hour" else max(1, (len(self._data) + 11) // 12)

        for index, row in enumerate(self._data):
            x = plot.left() + index * slot + (slot - bar_width) / 2
            cursor_y = plot.bottom()
            segments = row.get("segments") or {}
            for category, color in self.COLORS.items():
                count = int(segments.get(category, 0) or 0)
                if count <= 0:
                    continue
                height = plot.height() * count / scale_max
                rect = QRectF(x, cursor_y - height, bar_width, height)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(color))
                painter.drawRect(rect)
                cursor_y -= height

            # Every column gets an exact total above it, including zeroes.
            # The rounded axis scale leaves room above the stack; the clamp is
            # a final guard for very short widgets.
            total = int(row.get("total", 0))
            label_rect = QRectF(
                x - 12,
                max(plot.top() + 3, cursor_y - 23),
                bar_width + 24,
                18,
            )
            if total:
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor("#17191f"))
                painter.drawRoundedRect(label_rect.adjusted(-2, 0, 2, 0), 3, 3)
            painter.setPen(QColor("#f3f5f8"))
            painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
            painter.drawText(
                label_rect,
                Qt.AlignCenter,
                str(total),
            )

            if index % label_every == 0 or index == len(self._data) - 1:
                painter.setPen(QColor("#b7bfcd"))
                painter.setFont(QFont("Segoe UI", 8))
                label = str(row.get("label", ""))
                painter.drawText(QRectF(x - slot * 0.65, plot.bottom() + 8, slot * 1.3, 20), Qt.AlignCenter, label)

        painter.setPen(QColor("#697386"))
        painter.drawLine(plot.left(), plot.bottom(), plot.right(), plot.bottom())
        painter.drawText(QRectF(plot.left(), self.height() - 23, plot.width(), 18), Qt.AlignCenter, self._axis_label())

    def mouseMoveEvent(self, event):
        plot = self._plot_rect
        if plot is None or self._slot_width <= 0 or not self._data or not plot.contains(event.position()):
            self._last_tooltip = ""
            QToolTip.hideText()
            return super().mouseMoveEvent(event)

        index = min(len(self._data) - 1, max(0, int((event.position().x() - plot.left()) / self._slot_width)))
        row = self._data[index]
        segments = row.get("segments") or {}
        details = [
            f"{(self.localizer.tr(category) if self.localizer else category.title())}: "
            f"{int(segments.get(category, 0) or 0)}"
            for category in self.COLORS
        ]
        text = "<b>{}</b><br>Total: {}<br>{}".format(
            row.get("label", ""),
            int(row.get("total", 0)),
            "<br>".join(details),
        )
        if text != self._last_tooltip:
            self._last_tooltip = text
            QToolTip.showText(event.globalPosition().toPoint(), text, self)
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self._last_tooltip = ""
        QToolTip.hideText()
        super().leaveEvent(event)

    @staticmethod
    def _scale_for(max_total: int) -> tuple[float, float]:
        """Return a readable whole-number scale with label headroom."""
        maximum = max(0, int(max_total))
        desired_top = maximum + max(1, int(math.ceil(maximum * 0.15)))
        raw_step = max(1.0, desired_top / 4.0)
        magnitude = 10 ** math.floor(math.log10(raw_step))
        fraction = raw_step / magnitude
        for candidate in (1, 2, 2.5, 3, 4, 5, 10):
            if fraction <= candidate:
                step = candidate * magnitude
                break
        else:
            step = 10 * magnitude

        # Inspections are whole objects, so avoid fractional axis labels for
        # low counts even when the generic "nice number" algorithm suggests it.
        step = max(1.0, step)
        top = math.ceil(desired_top / step) * step
        if top <= maximum:
            top += step
        return float(top), float(step)

    def _axis_label(self) -> str:
        text = {
            "hour": "TIME · HOURS",
            "day": "TIME · DAYS",
            "week": "TIME · WEEKS",
            "month": "TIME · MONTHS",
        }.get(self._granularity, "TIME")
        return self.localizer.tr(text) if self.localizer is not None else text


def _legend_item(label: str, color: str) -> QWidget:
    widget = QWidget()
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(5)
    swatch = QFrame()
    swatch.setFixedSize(12, 12)
    swatch.setStyleSheet(f"background: {color}; border: none; border-radius: 3px;")
    layout.addWidget(swatch)
    text = QLabel(label)
    text.setStyleSheet("color: #b7bfcd; font-size: 11px;")
    layout.addWidget(text)
    return widget


class ReportsPage(QWidget):
    """Local inspection-history dashboard with period and custom-date views."""

    def __init__(
        self,
        *,
        report_store: ReportStore,
        recipe_manager=None,
        localizer: AppLocalization | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.report_store = report_store
        self.recipe_manager = recipe_manager
        self.localizer = localizer or AppLocalization(parent=self)
        self._refreshing = False

        self.setStyleSheet("""
            ReportsPage, QScrollArea, QScrollArea > QWidget, QWidget#reportsContent {
                background: #0f1014;
                color: #e9edf5;
            }
            QFrame#reportsToolbar {
                background: #151923;
                border: 1px solid #303846;
                border-radius: 12px;
            }
            QLabel#reportsEyebrow {
                color: #8ea2c8;
                font-size: 11px;
                font-weight: 800;
                letter-spacing: 1px;
            }
            QLabel#reportsTitle {
                color: #f6f8fc;
                font-size: 25px;
                font-weight: 900;
            }
            QLabel#reportsRange {
                color: #b9c9e6;
                font-size: 12px;
                font-weight: 800;
            }
            QLabel#reportsUpdated {
                color: #7f8ba1;
                font-size: 11px;
            }
            QComboBox#reportRecipe {
                background: #1d2430;
                border: 1px solid #3a4659;
                border-radius: 8px;
                color: #f2f5fb;
                min-height: 25px;
                padding: 4px 9px;
            }
            QPushButton#reportRefresh {
                background: #25375d;
                border: 1px solid #5e78af;
                color: #f6f8ff;
                border-radius: 8px;
                padding: 8px 15px;
            }
            QPushButton#reportRefresh:hover { background: #304675; }
            QTabBar::tab {
                background: #1b1d23;
                color: #dbe4f2;
                border: 1px solid #343946;
                padding: 9px 18px;
                margin-right: 4px;
                border-radius: 7px;
            }
            QTabBar::tab:selected {
                background: #31466f;
                color: #ffffff;
                border-color: #7894c7;
            }
            QTabBar::tab:hover:!selected { background: #252b3b; }
            QFrame#customRangePanel {
                background: #151923;
                border: 1px solid #303846;
                border-radius: 9px;
            }
            QDateEdit, QDateEdit QAbstractSpinBox {
                background: #1d2430;
                color: #e9edf5;
                border: 1px solid #3a4659;
                border-radius: 6px;
                padding: 6px 8px;
            }
            QDateEdit::drop-down {
                background: #252b3b;
                border-left: 1px solid #343946;
                width: 24px;
            }
            QCalendarWidget QWidget { background: #17191f; color: #e9edf5; }
            QCalendarWidget QWidget#qt_calendar_navigationbar {
                background: #1b1d23;
                color: #e9edf5;
            }
            QCalendarWidget QToolButton {
                background: #252b3b;
                color: #ffffff;
                border: 1px solid #343946;
                padding: 5px;
            }
            QCalendarWidget QSpinBox {
                background: #1b1d23;
                color: #ffffff;
                border: 1px solid #343946;
            }
            QCalendarWidget QAbstractItemView {
                background: #17191f;
                color: #e9edf5;
                selection-background-color: #40558f;
                selection-color: #ffffff;
                alternate-background-color: #1e2129;
            }
            QFrame#reportCardTotal, QFrame#reportCardPass, QFrame#reportCardFail, QFrame#reportCardRate {
                background: #151923;
                border: 1px solid #303846;
                border-radius: 10px;
            }
            QFrame#reportCardTotal { border-top: 3px solid #7b8cae; }
            QFrame#reportCardPass { border-top: 3px solid #38d27b; }
            QFrame#reportCardFail { border-top: 3px solid #ff6673; }
            QFrame#reportCardRate { border-top: 3px solid #77a5e8; }
            QLabel#metricHeading {
                color: #99a8c1;
                font-size: 10px;
                font-weight: 900;
                letter-spacing: .6px;
            }
            QFrame#reportPanel {
                background: #151923;
                border: 1px solid #303846;
                border-radius: 11px;
            }
            QLabel#sectionHeading {
                color: #c8d4e8;
                font-size: 11px;
                font-weight: 900;
                letter-spacing: .6px;
            }
            QToolTip {
                background: #1a2130;
                color: #edf3ff;
                border: 1px solid #5871a5;
                border-radius: 6px;
                padding: 6px;
            }
            QTableWidget {
                background: #17191f;
                color: #e9edf5;
                border: 1px solid #2d323d;
                gridline-color: #2d323d;
                selection-background-color: #2e385a;
                alternate-background-color: #1e2129;
            }
            QTableWidget::item { background: #17191f; color: #e9edf5; }
            QTableWidget::item:alternate { background: #1e2129; color: #e9edf5; }
            QTableCornerButton::section { background: #242832; border: 0; }
            QHeaderView::section {
                background: #242832;
                color: #dbe4f2;
                border: 0;
                border-right: 1px solid #343946;
                border-bottom: 1px solid #343946;
                padding: 7px;
                font-weight: 800;
            }
        """)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setStyleSheet(
            "QScrollArea { background: #0f1014; border: none; }"
            " QScrollArea > QWidget { background: #0f1014; border: none; }"
        )
        scroll.viewport().setStyleSheet("background: #0f1014; border: none;")
        content = QWidget()
        content.setObjectName("reportsContent")
        content.setAttribute(Qt.WA_StyledBackground, True)
        root = QVBoxLayout(content)
        root.setContentsMargins(24, 22, 24, 26)
        root.setSpacing(16)

        header_frame = QFrame()
        header_frame.setObjectName("reportsToolbar")
        header = QHBoxLayout(header_frame)
        header.setContentsMargins(18, 14, 16, 14)
        header.setSpacing(12)
        title_column = QVBoxLayout()
        title_column.setSpacing(1)
        eyebrow = QLabel("QUALITY CONTROL · PRODUCTION HISTORY")
        eyebrow.setObjectName("reportsEyebrow")
        title_column.addWidget(eyebrow)
        title = QLabel("Inspection Reports")
        title.setObjectName("reportsTitle")
        title_column.addWidget(title)
        self.lbl_range = QLabel("Today")
        self.lbl_range.setObjectName("reportsRange")
        title_column.addWidget(self.lbl_range)
        header.addLayout(title_column, 1)

        filters = QVBoxLayout()
        filters.setSpacing(4)
        filter_label = QLabel("RECIPE FILTER")
        filter_label.setObjectName("reportsEyebrow")
        filters.addWidget(filter_label)
        self.cmb_recipe = QComboBox()
        self.cmb_recipe.setObjectName("reportRecipe")
        self.cmb_recipe.setMinimumWidth(240)
        filters.addWidget(self.cmb_recipe)
        header.addLayout(filters)
        self.btn_refresh = QPushButton("REFRESH")
        self.btn_refresh.setObjectName("reportRefresh")
        self.btn_refresh.setMinimumHeight(38)
        header.addWidget(self.btn_refresh)
        root.addWidget(header_frame)

        self.period_tabs = QTabBar()
        self.period_tabs.setExpanding(False)
        for label in ("Daily", "Weekly", "Monthly", "Yearly", "Custom range"):
            self.period_tabs.addTab(label)
        self.period_tabs.setCurrentIndex(0)
        root.addWidget(self.period_tabs)

        custom_row = QHBoxLayout()
        custom_row.addWidget(QLabel("Custom start:"))
        self.date_start = QDateEdit(QDate.currentDate())
        self.localizer.configure_date_edit(self.date_start)
        self.date_start.setMaximumDate(QDate.currentDate())
        self.date_start.setMinimumWidth(140)
        custom_row.addWidget(self.date_start)
        custom_row.addWidget(QLabel("End:"))
        self.date_end = QDateEdit(QDate.currentDate())
        self.localizer.configure_date_edit(self.date_end)
        self.date_end.setMaximumDate(QDate.currentDate())
        self.date_end.setMinimumWidth(140)
        custom_row.addWidget(self.date_end)
        custom_row.addStretch(1)
        self.custom_range_frame = QFrame()
        self.custom_range_frame.setObjectName("customRangePanel")
        self.custom_range_frame.setLayout(custom_row)
        self.custom_range_frame.setVisible(False)
        root.addWidget(self.custom_range_frame)

        overview = QHBoxLayout()
        overview.setSpacing(12)
        self.card_total, self.lbl_total = self._metric_card("TOTAL GLASSES", "0", "reportCardTotal")
        self.card_pass, self.lbl_pass = self._metric_card("PASS", "0", "reportCardPass", color="#38d27b")
        self.card_fail, self.lbl_fail = self._metric_card("FAIL", "0", "reportCardFail", color="#ff6673")
        self.card_rate, self.lbl_rate = self._metric_card("PASS RATE", "—", "reportCardRate")
        for card in (self.card_total, self.card_pass, self.card_fail, self.card_rate):
            overview.addWidget(card, 1)
        root.addLayout(overview)

        charts = QHBoxLayout()
        charts.setSpacing(14)
        donut_frame = QFrame()
        donut_frame.setObjectName("reportPanel")
        donut_lay = QVBoxLayout(donut_frame)
        donut_lay.addWidget(self._section_label("PASS / FAIL SPLIT"))
        self.donut = DonutChart()
        self.donut.localizer = self.localizer
        donut_lay.addWidget(self.donut, 1)
        charts.addWidget(donut_frame, 2)

        cause_frame = QFrame()
        cause_frame.setObjectName("reportPanel")
        cause_lay = QVBoxLayout(cause_frame)
        cause_lay.addWidget(self._section_label("FAILURE CAUSES"))
        self.failure_chart = FailureCauseChart()
        self.failure_chart.localizer = self.localizer
        cause_lay.addWidget(self.failure_chart, 1)
        charts.addWidget(cause_frame, 5)
        root.addLayout(charts, 1)

        breakdown_frame = QFrame()
        breakdown_frame.setObjectName("reportPanel")
        breakdown_lay = QVBoxLayout(breakdown_frame)
        breakdown_lay.addWidget(self._section_label("FAILURE DETAIL · SIGNED DIRECTION"))
        self.table_breakdown = QTableWidget(0, 5)
        self.table_breakdown.setHorizontalHeaderLabels(
            ["FAILURE CATEGORY", "POSITIVE (+)", "NEGATIVE (−)", "ZERO / OTHER", "TOTAL"]
        )
        self._configure_table(self.table_breakdown)
        self.table_breakdown.setMinimumHeight(185)
        breakdown_lay.addWidget(self.table_breakdown)
        root.addWidget(breakdown_frame)

        timetable_frame = QFrame()
        timetable_frame.setObjectName("reportPanel")
        timetable_lay = QVBoxLayout(timetable_frame)
        timetable_lay.addWidget(self._section_label("PROCESSING TIMELINE · STACKED RESULTS"))
        self.timeline_chart = TimelineChart()
        self.timeline_chart.localizer = self.localizer
        timetable_lay.addWidget(self.timeline_chart, 1)

        legend = QHBoxLayout()
        legend.setContentsMargins(8, 0, 8, 4)
        legend.setSpacing(14)
        for category, color in TimelineChart.COLORS.items():
            legend.addWidget(_legend_item(category, color))
        legend.addStretch(1)
        timetable_lay.addLayout(legend)
        timeline_note = QLabel(
            "Timeline columns use one primary failure cause per glass; the signed detail table counts every failed measurement."
        )
        timeline_note.setStyleSheet("color: #8f9aad; font-size: 11px;")
        timeline_note.setWordWrap(True)
        timetable_lay.addWidget(timeline_note)
        root.addWidget(timetable_frame, 1)

        event_log_frame = QFrame()
        event_log_frame.setObjectName("reportPanel")
        event_log_lay = QVBoxLayout(event_log_frame)
        event_log_lay.addWidget(self._section_label("INSPECTION EVENT LOG"))
        self.lbl_event_log_hint = QLabel("Newest first · confirmed Auto and Manual Test results only.")
        self.lbl_event_log_hint.setStyleSheet("color: #8f9aad; font-size: 11px;")
        event_log_lay.addWidget(self.lbl_event_log_hint)
        self.table_event_log = QTableWidget(0, 4)
        self._configure_table(self.table_event_log)
        self.table_event_log.setMinimumHeight(280)
        self.table_event_log.setMaximumHeight(410)
        self.table_event_log.verticalHeader().setDefaultSectionSize(30)
        event_log_lay.addWidget(self.table_event_log)
        root.addWidget(event_log_frame)

        self.lbl_detail = QLabel("Reports are recorded from completed Auto and Manual Test inspections.")
        self.lbl_detail.setWordWrap(True)
        self.lbl_detail.setStyleSheet("font-size: 13px; color: #aeb8c8;")
        root.addWidget(self.lbl_detail)

        root.addStretch(1)
        scroll.setWidget(content)
        outer.addWidget(scroll)

        self.cmb_recipe.currentIndexChanged.connect(self.refresh)
        self.period_tabs.currentChanged.connect(self._on_period_changed)
        self.date_start.dateChanged.connect(self._on_custom_date_changed)
        self.date_end.dateChanged.connect(self._on_custom_date_changed)
        self.btn_refresh.clicked.connect(self.refresh)
        self._set_table_headers()
        self._load_recipe_filter()

    def _section_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("sectionHeading")
        label.setAlignment(Qt.AlignCenter)
        return label

    def _metric_card(self, title: str, value: str, object_name: str, *, color: str = "#f3f5f8"):
        frame = QFrame()
        frame.setObjectName(object_name)
        frame.setMinimumHeight(92)
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(2)
        heading = QLabel(title)
        heading.setObjectName("metricHeading")
        heading.setAlignment(Qt.AlignCenter)
        lay.addWidget(heading)
        value_label = QLabel(value)
        value_label.setAlignment(Qt.AlignCenter)
        value_label.setStyleSheet(f"font-size: 26px; font-weight: 1000; color: {color};")
        lay.addWidget(value_label)
        return frame, value_label

    def _configure_table(self, table: QTableWidget):
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setFocusPolicy(Qt.NoFocus)

    def _set_table_headers(self):
        self.table_breakdown.setHorizontalHeaderLabels(
            [
                self.localizer.tr("FAILURE CATEGORY"),
                self.localizer.tr("POSITIVE (+)"),
                self.localizer.tr("NEGATIVE (−)"),
                self.localizer.tr("ZERO / OTHER"),
                self.localizer.tr("TOTAL"),
            ]
        )
        self.table_event_log.setHorizontalHeaderLabels(
            [
                self.localizer.tr("PROCESSED AT"),
                self.localizer.tr("RECIPE"),
                self.localizer.tr("RESULT"),
                self.localizer.tr("FAILURE DETAIL"),
            ]
        )

    def _load_recipe_filter(self):
        current = self.cmb_recipe.currentData()
        self.cmb_recipe.blockSignals(True)
        try:
            self.cmb_recipe.clear()
            self.cmb_recipe.addItem(self.localizer.tr("All recipes"), None)
            names = []
            if self.recipe_manager is not None:
                try:
                    names.extend(self.recipe_manager.list_recipes())
                except Exception:
                    pass
            for name in self.report_store.available_recipes():
                if name not in names:
                    names.append(name)
            for name in sorted(set(names)):
                self.cmb_recipe.addItem(name, name)
            index = self.cmb_recipe.findData(current)
            if index >= 0:
                self.cmb_recipe.setCurrentIndex(index)
        finally:
            self.cmb_recipe.blockSignals(False)

    def _on_period_changed(self, index: int):
        self.custom_range_frame.setVisible(index == 4)
        self.refresh()

    def _on_custom_date_changed(self, _date):
        if self.period_tabs.currentIndex() == 4:
            self.refresh()

    def _selected_bounds(self):
        now = datetime.now()
        index = self.period_tabs.currentIndex()
        if index == 0:
            start = datetime.combine(now.date(), time.min)
            day_text = self.localizer.format_date(start, long=True)
            label = f"{self.localizer.tr('Today')} · {day_text}"
            end = start + timedelta(days=1)
        elif index == 1:
            start_date = now.date() - timedelta(
                days=(now.weekday() - self.localizer.python_week_start) % 7
            )
            start = datetime.combine(start_date, time.min)
            end_date = start_date + timedelta(days=6)
            label = (
                f"{self.localizer.tr('This week')} · "
                f"{self.localizer.format_date(start, long=True)} – "
                f"{self.localizer.format_date(end_date, long=True)}"
            )
            end = start + timedelta(days=7)
        elif index == 2:
            if self.localizer.use_jalali:
                calendar = self.localizer.calendar
                parts = calendar.partsFromDate(QDate(now.year, now.month, now.day))
                start_qdate = calendar.dateFromParts(parts.year, parts.month, 1)
                next_year, next_month = (parts.year + 1, 1) if parts.month == 12 else (parts.year, parts.month + 1)
                end_qdate = calendar.dateFromParts(next_year, next_month, 1)
                start = datetime.combine(start_qdate.toPython(), time.min)
                end = datetime.combine(end_qdate.toPython(), time.min)
                month_label = self.localizer.format_month(start)
            else:
                start = datetime.combine(now.date().replace(day=1), time.min)
                if start.month == 12:
                    end = start.replace(year=start.year + 1, month=1)
                else:
                    end = start.replace(month=start.month + 1)
                month_label = start.strftime("%B %Y")
            label = f"{self.localizer.tr('This month')} · {month_label}"
        elif index == 3:
            if self.localizer.use_jalali:
                calendar = self.localizer.calendar
                parts = calendar.partsFromDate(QDate(now.year, now.month, now.day))
                start_qdate = calendar.dateFromParts(parts.year, 1, 1)
                end_qdate = calendar.dateFromParts(parts.year + 1, 1, 1)
                start = datetime.combine(start_qdate.toPython(), time.min)
                end = datetime.combine(end_qdate.toPython(), time.min)
                year_text = self.localizer.digits(parts.year)
            else:
                start = datetime.combine(now.date().replace(month=1, day=1), time.min)
                end = start.replace(year=start.year + 1)
                year_text = str(start.year)
            label = f"{self.localizer.tr('This year')} · {year_text}"
        else:
            start_date = self.date_start.date().toPython()
            end_date = self.date_end.date().toPython()
            if end_date < start_date:
                start_date, end_date = end_date, start_date
            start = datetime.combine(start_date, time.min)
            label = (
                f"{self.localizer.tr('Custom')} · "
                f"{self.localizer.format_date(start_date, long=True)} – "
                f"{self.localizer.format_date(end_date, long=True)}"
            )
            end = datetime.combine(end_date + timedelta(days=1), time.min)
            return start, end, label
        return start, end, label

    def _calendar_bucket_boundaries(self, start: datetime, end: datetime, granularity: str):
        """Provide true Solar-Hijri month boundaries for the yearly chart."""
        if not self.localizer.use_jalali or granularity != "month":
            return None

        calendar = self.localizer.calendar
        boundaries = []
        cursor = start
        while cursor < end:
            boundaries.append(cursor)
            parts = calendar.partsFromDate(QDate(cursor.year, cursor.month, cursor.day))
            year, month = (parts.year + 1, 1) if parts.month == 12 else (parts.year, parts.month + 1)
            next_qdate = calendar.dateFromParts(year, month, 1)
            cursor = datetime.combine(next_qdate.toPython(), time.min)
        boundaries.append(end)
        return boundaries

    def refresh(self, *_args):
        if self._refreshing:
            return
        self._refreshing = True
        try:
            self._load_recipe_filter()
            start, end, label = self._selected_bounds()
            recipe = self.cmb_recipe.currentData() or None
            summary = self.report_store.summary(start=start, end=end, recipe=recipe)
            causes = self.report_store.failure_causes(start=start, end=end, recipe=recipe)
            breakdown = self.report_store.failure_breakdown(start=start, end=end, recipe=recipe)
            events = self.report_store.inspection_events(start=start, end=end, recipe=recipe)
            granularity = {
                0: "hour",
                1: "day",
                2: "week",
                3: "month",
            }.get(self.period_tabs.currentIndex(), "day")
            timetable = self.report_store.time_buckets(
                start=start,
                end=end,
                recipe=recipe,
                granularity=granularity,
                week_start=self.localizer.python_week_start,
                bucket_boundaries=self._calendar_bucket_boundaries(start, end, granularity),
            )
            for row in timetable:
                row["label"] = self.localizer.format_bucket(row["bucket"], granularity)

            self.lbl_range.setText(label)
            self.lbl_total.setText(self.localizer.digits(f"{summary['total']:,}"))
            self.lbl_pass.setText(self.localizer.digits(f"{summary['pass']:,}"))
            self.lbl_fail.setText(self.localizer.digits(f"{summary['fail']:,}"))
            self.lbl_rate.setText(
                self.localizer.digits(f"{summary['pass_rate']:.1f}%") if summary["total"] else "—"
            )
            self.donut.set_counts(summary["pass"], summary["fail"])
            self.failure_chart.set_data(causes)
            self._fill_breakdown_table(breakdown)
            self._fill_event_log(events)
            self.timeline_chart.set_data(timetable, granularity)
            if self.localizer.is_farsi:
                total_text = self.localizer.digits(f"{summary['total']:,}")
                self.lbl_detail.setText(
                    f"{total_text} بازرسی تکمیل‌شده در این بازه ثبت شده است. "
                    f"{self.localizer.digits(len(causes))} علت رد متمایز ثبت شد. "
                    "جهت نسبت به مرجع طلایی ذخیره‌شده است."
                )
            else:
                self.lbl_detail.setText(
                    f"{summary['total']:,} completed inspections in this range. "
                    f"{len(causes)} distinct failure cause(s) recorded. "
                    "Direction is relative to the saved golden reference."
                )
        finally:
            self._refreshing = False

    def _fill_breakdown_table(self, breakdown: dict[str, dict[str, int]]):
        rows = sorted(breakdown.items(), key=lambda item: (-item[1].get("total", 0), item[0]))
        self.table_breakdown.setRowCount(len(rows))
        for row_index, (label, values) in enumerate(rows):
            cells = (
                self.localizer.tr(label),
                self.localizer.digits(values.get("positive", 0)),
                self.localizer.digits(values.get("negative", 0)),
                self.localizer.digits(values.get("zero", 0) + values.get("other", 0)),
                self.localizer.digits(values.get("total", 0)),
            )
            for column, value in enumerate(cells):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignCenter if column else Qt.AlignLeft | Qt.AlignVCenter)
                if label == "BASEPLATE NOT FOUND":
                    item.setForeground(QColor("#ff6673"))
                self.table_breakdown.setItem(row_index, column, item)

    def _format_event_time(self, event: dict) -> str:
        timestamp = event.get("timestamp")
        if not isinstance(timestamp, datetime):
            return str(event.get("timestamp_text") or "—")
        date_text = self.localizer.format_date(timestamp, long=False)
        time_text = self.localizer.digits(timestamp.strftime("%H:%M:%S.%f")[:-3])
        return f"{date_text}  {time_text}"

    def _fill_event_log(self, events: list[dict]):
        self.table_event_log.setRowCount(len(events))
        for row_index, event in enumerate(events):
            result = str(event.get("result") or "").upper()
            cause = str(event.get("cause") or "").strip()
            cells = (
                self._format_event_time(event),
                str(event.get("recipe") or "—"),
                self.localizer.tr(result),
                self.localizer.tr(cause) if cause else "—",
            )
            for column, value in enumerate(cells):
                item = QTableWidgetItem(value)
                item.setData(Qt.UserRole, event.get("id"))
                item.setToolTip(value)
                item.setTextAlignment(
                    Qt.AlignCenter if column == 2 else Qt.AlignLeft | Qt.AlignVCenter
                )
                if column == 2:
                    item.setForeground(QColor("#38d27b") if result == "PASS" else QColor("#ff6673"))
                elif column == 3 and result == "FAIL":
                    item.setForeground(
                        QColor("#bf7446") if cause.upper() == "BASEPLATE NOT FOUND" else QColor("#ffb0b8")
                    )
                self.table_event_log.setItem(row_index, column, item)

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh()

    def apply_localization(self):
        """Apply the live language/calendar preference without changing stored data."""
        current_start = self.date_start.date()
        current_end = self.date_end.date()
        self.localizer.configure_date_edit(self.date_start)
        self.localizer.configure_date_edit(self.date_end)
        self.date_start.setDate(current_start)
        self.date_end.setDate(current_end)
        self.localizer.apply_widget_text(self)
        self._set_table_headers()
        self._load_recipe_filter()
        self.refresh()
