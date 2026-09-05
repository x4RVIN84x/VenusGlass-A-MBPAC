from __future__ import annotations

from datetime import date, datetime
from typing import Iterable

from PySide6.QtCore import QCalendar, QDate, QLocale, QObject, QSettings, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractButton,
    QComboBox,
    QDateEdit,
    QGroupBox,
    QLabel,
    QLineEdit,
    QTabBar,
    QWidget,
)


class AppLocalization(QObject):
    """Workstation-local UI language and calendar preferences.

    Inspection timestamps remain normal Gregorian datetimes in SQLite.  This
    object changes only how those dates and weeks are presented to operators.
    """

    changed = Signal()

    SETTINGS_ORGANIZATION = "Venus Glass"
    SETTINGS_APPLICATION = "MBPAC QC Station"

    PERSIAN = {
        "MBPAC QC Station": "ایستگاه کنترل کیفیت MBPAC",
        "MBPAC QC Station (Industrial HMI)": "ایستگاه کنترل کیفیت MBPAC (رابط صنعتی)",
        "Product/Recipe:": "محصول/دستورالعمل:",
        "AUTO MODE": "حالت خودکار",
        "CALIBRATION": "کالیبراسیون",
        "REPORTS": "گزارش‌ها",
        "MANUAL TEST": "آزمون دستی",
        "OPTIONS": "تنظیمات",
        "Auto Mode": "حالت خودکار",
        "Calibration": "کالیبراسیون",
        "Reports": "گزارش‌ها",
        "Manual Test": "آزمون دستی",
        "Options": "تنظیمات",
        "QUALITY CONTROL · PRODUCTION HISTORY": "کنترل کیفیت · سوابق تولید",
        "Inspection Reports": "گزارش‌های بازرسی",
        "RECIPE FILTER": "فیلتر دستورالعمل",
        "All recipes": "همه دستورالعمل‌ها",
        "REFRESH": "به‌روزرسانی",
        "Daily": "روزانه",
        "Weekly": "هفتگی",
        "Monthly": "ماهانه",
        "Yearly": "سالانه",
        "Custom range": "بازه دلخواه",
        "Custom start:": "شروع دلخواه:",
        "End:": "پایان:",
        "Today": "امروز",
        "This week": "این هفته",
        "This month": "این ماه",
        "This year": "امسال",
        "Custom": "دلخواه",
        "Week": "هفته",
        "TOTAL GLASSES": "کل شیشه‌ها",
        "PASS": "قبول",
        "FAIL": "رد",
        "PASS RATE": "نرخ قبولی",
        "PASS / FAIL SPLIT": "تفکیک قبول / رد",
        "FAILURE CAUSES": "علت‌های رد",
        "FAILURE DETAIL · SIGNED DIRECTION": "جزئیات رد · جهت‌دار",
        "INSPECTION EVENT LOG": "رویدادهای بازرسی",
        "PROCESSED AT": "زمان پردازش",
        "RECIPE": "دستورالعمل",
        "RESULT": "نتیجه",
        "FAILURE DETAIL": "جزئیات رد",
        "Newest first · confirmed Auto and Manual Test results only.": "جدیدترین ابتدا · فقط نتایج تأییدشده حالت خودکار و آزمون دستی.",
        "FAILURE CATEGORY": "دسته رد",
        "POSITIVE (+)": "مثبت (+)",
        "NEGATIVE (−)": "منفی (−)",
        "ZERO / OTHER": "صفر / سایر",
        "TOTAL": "کل",
        "PROCESSING TIMELINE · STACKED RESULTS": "روند پردازش · نتایج انباشته",
        "GLASSES PROCESSED": "شیشه‌های پردازش‌شده",
        "TIME · HOURS": "زمان · ساعت‌ها",
        "TIME · DAYS": "زمان · روزها",
        "TIME · WEEKS": "زمان · هفته‌ها",
        "TIME · MONTHS": "زمان · ماه‌ها",
        "BASEPLATE NOT FOUND": "عدم تشخیص بیس‌پلیت",
        "BASEPLATE ABSENCE": "عدم حضور بیس‌پلیت",
        "OTHER FAIL": "سایر خطاها",
        "Options are saved automatically for this workstation.": "تنظیمات برای این ایستگاه کاری خودکار ذخیره می‌شوند.",
        "These settings affect this workstation only. They keep Auto Mode focused on inspection.": "این تنظیمات فقط روی همین ایستگاه کاری اعمال می‌شوند و حالت خودکار را بر بازرسی متمرکز نگه می‌دارند.",
        "Accessibility & Interface": "دسترسی‌پذیری و رابط کاربری",
        "Auto Mode control and button size": "اندازه کنترل‌ها و دکمه‌های حالت خودکار",
        "Reduce visual alarm flashing": "کاهش چشمک هشدار دیداری",
        "Decision Confidence": "اطمینان تصمیم",
        "Result confirmation time": "زمان تأیید نتیجه",
        "Current confirmation time:": "زمان تأیید فعلی:",
        "Glass Handoff Protection": "محافظت از انتقال شیشه",
        "Clear-station delay before next glass": "زمان خالی بودن ایستگاه پیش از شیشه بعدی",
        "Current clear-station delay:": "زمان فعلی خالی بودن ایستگاه:",
        "After a result is saved, the station must remain empty for this time before another glass can be recorded.": "پس از ذخیره نتیجه، ایستگاه باید برای این مدت خالی بماند تا شیشه دیگری ثبت شود.",
        "A longer delay gives more protection against duplicate records caused by a brief tracking dropout.": "زمان طولانی‌تر، از ثبت تکراری ناشی از قطع کوتاه ردیابی بیشتر جلوگیری می‌کند.",
        "Auto Mode shows a live confidence percentage while a PASS or FAIL candidate is being confirmed.": "حالت خودکار هنگام تأیید نتیجه نامزد قبول یا رد، درصد اطمینان زنده را نشان می‌دهد.",
        "Auto Mode preserves live evidence through short track dropouts and maps it across PASS, FAIL, and baseplate missing.": "حالت خودکار شواهد زنده را در قطع‌های کوتاه ردیابی حفظ می‌کند و آن را میان قبول، رد و نبود بیس‌پلیت نمایش می‌دهد.",
        "Alarm": "هشدار",
        "Play alarm sound": "پخش صدای هشدار",
        "Navigation Order": "ترتیب زبانه‌ها",
        "MOVE UP": "انتقال به بالا",
        "MOVE DOWN": "انتقال به پایین",
        "RESTORE DEFAULT OPTIONS": "بازگردانی تنظیمات پیش‌فرض",
        "Language & Regional Settings": "زبان و تنظیمات منطقه‌ای",
        "Application language": "زبان برنامه",
        "English": "English",
        "Farsi (فارسی)": "فارسی",
        "Use Solar Hijri (Jalali) calendar": "استفاده از تقویم هجری شمسی (جلالی)",
        "Weeks run Saturday through Friday": "هفته‌ها از شنبه تا جمعه محاسبه می‌شوند",
        "Choose a tab and use the arrows to change the left-hand tab order.": "یک زبانه را انتخاب کنید و با پیکان‌ها ترتیب زبانه‌ها را تغییر دهید.",
        "Auto Controls": "کنترل‌های خودکار",
        "INSPECTION CYCLE": "چرخه بازرسی",
        "INSPECTING CURRENT GLASS": "در حال بازرسی شیشه فعلی",
        "RESULT SAVED · HOLDING CURRENT GLASS": "نتیجه ذخیره شد · شیشه فعلی هنوز در ایستگاه است",
        "RESULT SAVED · CLEAR STATION": "نتیجه ذخیره شد · در انتظار خالی شدن ایستگاه",
        "GLASS COMPLETE · HOLDING CURRENT GLASS": "اسکن شیشه کامل شد · شیشه فعلی هنوز در ایستگاه است",
        "CLEARING STATION": "در انتظار خالی شدن ایستگاه",
        "LAST SAVED:": "آخرین نتیجه ذخیره‌شده:",
        "LAST SAVED: —": "آخرین نتیجه ذخیره‌شده: —",
        "Clear-station delay:": "زمان خالی بودن ایستگاه:",
        "Reports re-arm only when clear": "گزارش‌ها فقط پس از خالی شدن ایستگاه آماده می‌شوند",
        "READY FOR NEXT GLASS": "آماده برای شیشه بعدی",
        "WAITING FOR CAMERA / RECIPE": "در انتظار دوربین / دستورالعمل",
        "A new report unlocks only after 5s of a clear station.": "گزارش جدید تنها پس از ۵ ثانیه خالی بودن ایستگاه فعال می‌شود.",
        "Show stabilizer overlay": "نمایش پوشش تثبیت‌کننده",
        "Search ROI box": "کادر ناحیه جست‌وجو",
        "Actual notch contour edge": "لبه واقعی کانتور شیار",
        "Fitted notch lines": "خطوط برازش‌شده شیار",
        "New anchors / skeleton": "لنگرها / اسکلت جدید",
        "Raw feature points": "نقاط ویژگی خام",
        "Legacy dot/line fallback debug": "اشکال‌زدایی قدیمی نقطه/خط",
        "Show diagnostic text (advanced)": "نمایش متن تشخیصی (پیشرفته)",
        "Show baseplate contour + center": "نمایش کانتور و مرکز بیس‌پلیت",
        "Enable visual lost-track alarm after 5s": "فعال‌سازی هشدار دیداری قطع ردیابی پس از ۵ ثانیه",
        "The visual lost-track alarm itself remains controlled in Auto Mode.": "خود هشدار دیداری قطع ردیابی همچنان از حالت خودکار کنترل می‌شود.",
        "DETECTION CHECK": "بررسی تشخیص",
        "Glass notch: waiting for fitted lines": "شیار شیشه: در انتظار خطوط برازش‌شده",
        "Glass notch: CONFIRMED": "شیار شیشه: تأیید شد",
        "Glass notch geometry confirmed": "هندسه شیار شیشه تأیید شد",
        "Baseplate: monitoring": "بیس‌پلیت: در حال پایش",
        "Baseplate: DETECTED": "بیس‌پلیت: تشخیص داده شد",
        "Baseplate: NOT DETECTED": "بیس‌پلیت: تشخیص داده نشد",
        "Baseplate not detected": "بیس‌پلیت تشخیص داده نشد",
        "Verifying absence": "در حال تأیید نبودن",
        "ALARM ACTIVE": "هشدار فعال",
        "BASEPLATE CHECK": "بررسی بیس‌پلیت",
        "BASEPLATE MISSING": "بیس‌پلیت یافت نشد",
        "Alarm: visual alarm disabled · baseplate check visible": "هشدار: هشدار دیداری غیرفعال است · بررسی بیس‌پلیت نمایش داده می‌شود",
        "ALARM: baseplate not found": "هشدار: بیس‌پلیت یافت نشد",
        "Alarm: reset acknowledged, re-arming in": "هشدار: بازنشانی تأیید شد، فعال‌سازی دوباره تا",
        "Alarm: baseplate warning in": "هشدار: اخطار بیس‌پلیت تا",
        "Enable display zoom around baseplate": "فعال‌سازی بزرگ‌نمایی نمایش پیرامون بیس‌پلیت",
        "Display zoom:": "بزرگ‌نمایی نمایش:",
        "Stabilize cadence (every N frames)": "تناوب تثبیت (هر N فریم)",
        "Search padding (px)": "حاشیه جست‌وجو (پیکسل)",
        "RESET / ACK ALARM": "بازنشانی / تأیید هشدار",
        "START": "شروع",
        "STOP": "توقف",
        "INSPECTION STATUS": "وضعیت بازرسی",
        "CURRENT RESULT": "نتیجه جاری",
        "CONFIDENCE": "اطمینان",
        "ACCEPTANCE LIMITS": "حدود پذیرش",
        "CALIBRATION STATUS": "وضعیت کالیبراسیون",
        "LIVE CAMERA READING": "خوانش دوربین زنده",
        "SAVED GOLDEN REFERENCE": "مرجع طلایی ذخیره‌شده",
        "Manual Test": "آزمون دستی",
        "Config:": "پیکربندی:",
        "View:": "نما:",
        "GRAB FRAME": "دریافت فریم",
        "RUN ONCE": "اجرای یک‌باره",
        "Product Management": "مدیریت محصول",
        "Current product:": "محصول فعلی:",
        "CREATE PRODUCT": "ایجاد محصول",
        "DUPLICATE SELECTED PRODUCT": "تکثیر محصول انتخاب‌شده",
        "CAPTURE GOLDEN + SET EXPECTED": "ثبت مرجع طلایی + تنظیم مقدار مرجع",
        "Preview": "پیش‌نمایش",
        "START LIVE PREVIEW": "شروع پیش‌نمایش زنده",
        "STOP PREVIEW": "توقف پیش‌نمایش",
        "PASS / FAIL Tolerances": "تلرانس‌های قبول / رد",
        "Baseplate Tuning + Scale": "تنظیم بیس‌پلیت + مقیاس",
        "Notch / Glass Edge Tuning": "تنظیم شیار / لبه شیشه",
        "Save / Live Apply": "ذخیره / اعمال زنده",
        "SAVE PRODUCT JSON": "ذخیره JSON محصول",
        "Enter physical X and Y limits in millimetres. The app saves those recipe limits and converts them to the pixels used by inspection.": "حدود فیزیکی X و Y را بر حسب میلی‌متر وارد کنید. برنامه این حدود دستورالعمل را ذخیره و به پیکسل‌های مورد استفاده در بازرسی تبدیل می‌کند.",
        "X position limit (mm)": "حد موقعیت X (میلی‌متر)",
        "Y position limit (mm)": "حد موقعیت Y (میلی‌متر)",
        "Angle limit": "حد زاویه",
        "baseplate_width_mm": "عرض بیس‌پلیت (میلی‌متر)",
        "baseplate_height_mm": "ارتفاع بیس‌پلیت (میلی‌متر)",
        "canny_low": "آستانه پایین Canny",
        "canny_high": "آستانه بالای Canny",
        "blur_ksize": "مقدار محوشدگی",
        "clahe_clip": "تقویت کنتراست محلی",
        "dilate_iter": "تکرار گسترش لبه",
        "close_iter": "تکرار بستن شکاف",
        "contrast_min": "حداقل کنتراست",
        "notch_blur_ksize": "محوشدگی شیار",
        "notch_close_ksize": "بستن شیار",
        "notch_open_ksize": "بازکردن شیار",
        "notch_threshold_bias": "تنظیم آستانه لبه تیره",
        "notch_bottom_band_frac": "بخش پایینی شیار",
        "notch_side_band_frac": "بخش کناری شیار",
        "notch_frame_roi_extra_pad": "حاشیه افزوده ناحیه شیار",
        "RAW": "خام",
        "PROC": "پردازش‌شده",
        "OVERLAY": "پوشش",
        "READY": "آماده",
        "IDLE": "بیکار",
        "RUNNING": "در حال اجرا",
        "FAULT": "خطا",
        "ANGLE": "زاویه",
        "TRACK": "در حال بررسی",
        "SEARCH": "جست‌وجو",
        "SEARCHING": "در حال جست‌وجو",
        "SETUP": "تنظیم اولیه",
        "SETUP REQUIRED": "نیازمند تنظیم",
        "ALIGNED": "هم‌راستا",
        "VERIFYING": "در حال تأیید",
        "OFFSET": "دارای انحراف",
        "LIVE": "زنده",
        "Status:": "وضعیت:",
        "Alarm:": "هشدار:",
        "Current size:": "اندازه فعلی:",
        "Scale:": "مقیاس:",
        "Expected:": "مقدار مرجع:",
        "Saved:": "ذخیره‌شده:",
        "Live:": "زنده:",
        "Tolerances:": "تلرانس‌ها:",
        "Saved for this workstation.": "برای این ایستگاه کاری ذخیره شد.",
        "Restored the default options for this workstation.": "تنظیمات پیش‌فرض این ایستگاه کاری بازگردانی شد.",
        "Within recipe limits": "در محدوده دستورالعمل",
        "Verifying position": "در حال تأیید موقعیت",
        "candidate": "نامزد",
        "confirmed": "تأیید شد",
        "confidence": "اطمینان",
        "Searching for the baseplate": "در حال جست‌وجوی بیس‌پلیت",
        "Outside acceptance limits": "خارج از حدود پذیرش",
        "Ready for inspection": "آماده بازرسی",
        "armed": "فعال",
        "disabled": "غیرفعال",
        "manually reset / monitoring again": "دستی بازنشانی شد / پایش دوباره آغاز شد",
        "Live: current parameters are being used by preview": "زنده: پارامترهای فعلی در پیش‌نمایش استفاده می‌شوند",
        "Saved: loaded from product JSON": "ذخیره‌شده: از JSON محصول بارگذاری شد",
        "Timeline columns use one primary failure cause per glass; the signed detail table counts every failed measurement.": "هر ستون زمانی یک علت اصلی رد برای هر شیشه را نشان می‌دهد؛ جدول جزئیات تمام اندازه‌گیری‌های ناموفق را می‌شمارد.",
        "No product loaded": "محصولی بارگذاری نشده است",
        "Enter X, Y, and angle limits in Calibration": "حدود X، Y و زاویه را در کالیبراسیون وارد کنید",
        "Enter X/Y limits in mm and angle limit in Calibration": "حدود X/Y را بر حسب میلی‌متر و حد زاویه را در کالیبراسیون وارد کنید",
        "No recorded failures in this period": "در این بازه خطایی ثبت نشده است",
        "No inspections recorded in this period": "در این بازه بازرسی ثبت نشده است",
        "PROCESSED": "پردازش‌شده",
    }

    _PERSIAN_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
    _JALALI_MONTHS_FA = (
        "فروردین", "اردیبهشت", "خرداد", "تیر", "مرداد", "شهریور",
        "مهر", "آبان", "آذر", "دی", "بهمن", "اسفند",
    )
    _JALALI_MONTHS_EN = (
        "Farvardin", "Ordibehesht", "Khordad", "Tir", "Mordad", "Shahrivar",
        "Mehr", "Aban", "Azar", "Dey", "Bahman", "Esfand",
    )
    _WEEKDAYS_FA = ("دوشنبه", "سه‌شنبه", "چهارشنبه", "پنجشنبه", "جمعه", "شنبه", "یکشنبه")

    def __init__(self, settings: QSettings | None = None, parent=None):
        super().__init__(parent)
        self._settings = settings or QSettings(self.SETTINGS_ORGANIZATION, self.SETTINGS_APPLICATION)
        saved_language = str(self._settings.value("regional/language", "en") or "en").lower()
        self._language = "fa" if saved_language.startswith("fa") else "en"
        self._jalali = self._as_bool(self._settings.value("regional/jalali_calendar", False), False)

    @staticmethod
    def _as_bool(value, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return default if value is None else bool(value)

    @property
    def language(self) -> str:
        return self._language

    @property
    def is_farsi(self) -> bool:
        return self._language == "fa"

    @property
    def use_jalali(self) -> bool:
        return self._jalali

    @property
    def locale(self) -> QLocale:
        return QLocale("fa_IR") if self.is_farsi else QLocale("en_US")

    @property
    def calendar(self) -> QCalendar:
        system = QCalendar.System.Jalali if self.use_jalali else QCalendar.System.Gregorian
        return QCalendar(system)

    @property
    def first_day_of_week(self):
        return Qt.Saturday if self.use_jalali else Qt.Monday

    @property
    def python_week_start(self) -> int:
        """Python weekday index: Monday=0; Saturday=5."""
        return 5 if self.use_jalali else 0

    def set_language(self, language: str):
        language = "fa" if str(language).lower().startswith("fa") else "en"
        if language == self._language:
            return
        self._language = language
        self._settings.setValue("regional/language", language)
        self._settings.sync()
        self.changed.emit()

    def set_use_jalali(self, enabled: bool):
        enabled = bool(enabled)
        if enabled == self._jalali:
            return
        self._jalali = enabled
        self._settings.setValue("regional/jalali_calendar", enabled)
        self._settings.sync()
        self.changed.emit()

    def tr(self, text: str) -> str:
        source = str(text or "")
        if not self.is_farsi:
            return source
        if source in self.PERSIAN:
            return self.PERSIAN[source]
        for prefix in ("Status: ", "Alarm: ", "Current size: ", "Scale: ", "Expected: ", "Saved: ", "Live: ", "Tolerances: "):
            if source.startswith(prefix):
                translated_prefix = self.PERSIAN.get(prefix.rstrip(), prefix).rstrip() + " "
                tail = source[len(prefix):]
                for state in ("READY", "IDLE", "RUNNING", "FAULT", "PASS", "FAIL", "SEARCH", "LIVE"):
                    if tail == state or tail.startswith(state + " "):
                        return translated_prefix + self.tr(state) + tail[len(state):]
                return translated_prefix + self.tr(tail)
        return source

    def digits(self, value) -> str:
        text = str(value)
        return text.translate(self._PERSIAN_DIGITS) if self.is_farsi else text

    def configure_date_edit(self, widget: QDateEdit):
        widget.setLocale(self.locale)
        widget.setCalendar(self.calendar)
        widget.setDisplayFormat("yyyy/MM/dd")
        widget.setCalendarPopup(True)
        popup = widget.calendarWidget()
        popup.setLocale(self.locale)
        popup.setCalendar(self.calendar)
        popup.setFirstDayOfWeek(self.first_day_of_week)

    def format_date(self, value: date | datetime, *, long: bool = False) -> str:
        current = value.date() if isinstance(value, datetime) else value
        qdate = QDate(current.year, current.month, current.day)
        if self.use_jalali:
            parts = self.calendar.partsFromDate(qdate)
            month_names = self._JALALI_MONTHS_FA if self.is_farsi else self._JALALI_MONTHS_EN
            if long:
                return f"{self.digits(parts.day)} {month_names[parts.month - 1]} {self.digits(parts.year)}"
            return "/".join((self.digits(parts.year), self.digits(f"{parts.month:02d}"), self.digits(f"{parts.day:02d}")))
        if self.is_farsi:
            return self.locale.toString(qdate, "dd MMMM yyyy")
        return current.strftime("%b %d, %Y") if long else current.strftime("%Y/%m/%d")

    def format_weekday(self, value: date | datetime) -> str:
        current = value.date() if isinstance(value, datetime) else value
        if self.is_farsi:
            return self._WEEKDAYS_FA[current.weekday()]
        return current.strftime("%a")

    def format_month(self, value: date | datetime) -> str:
        current = value.date() if isinstance(value, datetime) else value
        qdate = QDate(current.year, current.month, current.day)
        if self.use_jalali:
            parts = self.calendar.partsFromDate(qdate)
            names = self._JALALI_MONTHS_FA if self.is_farsi else self._JALALI_MONTHS_EN
            return f"{names[parts.month - 1]} {self.digits(parts.year)}"
        if self.is_farsi:
            return self.locale.toString(qdate, "MMMM yyyy")
        return current.strftime("%B %Y")

    def format_bucket(self, bucket: datetime, granularity: str) -> str:
        if granularity == "hour":
            return self.digits(f"{bucket.hour:02d}:00")
        if granularity == "day":
            return f"{self.format_weekday(bucket)} {self.digits(bucket.day if not self.use_jalali else self.calendar.partsFromDate(QDate(bucket.year, bucket.month, bucket.day)).day)}"
        if granularity == "week":
            return f"{self.tr('Week')} {self.format_date(bucket, long=False)}"
        if granularity == "month":
            if self.use_jalali:
                parts = self.calendar.partsFromDate(QDate(bucket.year, bucket.month, bucket.day))
                names = self._JALALI_MONTHS_FA if self.is_farsi else self._JALALI_MONTHS_EN
                return names[parts.month - 1]
            return bucket.strftime("%b") if not self.is_farsi else self.locale.toString(QDate(bucket.year, bucket.month, bucket.day), "MMM")
        return self.format_date(bucket)

    def apply_widget_text(self, root: QWidget):
        """Translate static labels/buttons/tabs while retaining English source text."""
        widgets: Iterable[QWidget] = (root, *root.findChildren(QWidget))
        for widget in widgets:
            if bool(widget.property("_i18n_dynamic")):
                continue
            if isinstance(widget, QTabBar):
                sources = widget.property("_i18n_tab_sources")
                if not isinstance(sources, list) or len(sources) != widget.count():
                    sources = [widget.tabText(index) for index in range(widget.count())]
                    widget.setProperty("_i18n_tab_sources", sources)
                for index, source in enumerate(sources):
                    widget.setTabText(index, self.tr(source))
                continue

            if isinstance(widget, QGroupBox):
                self._apply_text_property(widget, "_i18n_title", widget.title, widget.setTitle)
            elif isinstance(widget, (QLabel, QAbstractButton)):
                self._apply_text_property(widget, "_i18n_text", widget.text, widget.setText)

            if isinstance(widget, QLineEdit):
                self._apply_text_property(
                    widget,
                    "_i18n_placeholder",
                    widget.placeholderText,
                    widget.setPlaceholderText,
                )

            if isinstance(widget, QComboBox):
                sources = widget.property("_i18n_combo_sources")
                if not isinstance(sources, list) or len(sources) != widget.count():
                    sources = [widget.itemText(index) for index in range(widget.count())]
                    widget.setProperty("_i18n_combo_sources", sources)
                for index, source in enumerate(sources):
                    widget.setItemText(index, self.tr(source))

            tooltip = widget.toolTip()
            if tooltip:
                source = widget.property("_i18n_tooltip")
                if not isinstance(source, str):
                    source = tooltip
                    widget.setProperty("_i18n_tooltip", source)
                widget.setToolTip(self.tr(source))

    def _apply_text_property(self, widget, property_name: str, getter, setter):
        source = widget.property(property_name)
        if not isinstance(source, str):
            source = str(getter() or "")
            widget.setProperty(property_name, source)
        if source:
            setter(self.tr(source))
