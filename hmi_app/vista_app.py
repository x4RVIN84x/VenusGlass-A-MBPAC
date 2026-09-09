"""Standalone QML application entry point for VG VISTA."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QTimer, Qt, QUrl
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtWidgets import QApplication, QSplashScreen

from hmi_app.vista_backend import VistaController
from hmi_app.vista_metadata import (
    AUTHOR,
    COMPANY_NAME,
    PRODUCT_NAME,
    PRODUCT_SUBTITLE,
    PRODUCT_VERSION,
    RELEASE_DATE,
    brand_icon_path,
    qml_directory,
)


def _startup_splash() -> QSplashScreen:
    """Draw a native startup surface before the QML runtime is ready.

    The QML splash continues the visual handoff after the engine has loaded;
    this lightweight native surface prevents a blank desktop during imports.
    """
    pixmap = QPixmap(760, 430)
    pixmap.fill(QColor("#090E13"))
    painter = QPainter(pixmap)
    try:
        painter.fillRect(0, 0, pixmap.width(), 7, QColor("#20A7F5"))
        painter.setBrush(QColor("#0D2639"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(480, -160, 430, 430)

        logo = QPixmap(str(brand_icon_path()))
        if not logo.isNull():
            painter.drawPixmap(72, 110, 176, 176, logo)

        painter.setPen(QColor("#F2F7FB"))
        painter.setFont(QFont("Segoe UI Semibold", 34))
        painter.drawText(292, 156, PRODUCT_NAME)
        painter.setPen(QColor("#20A7F5"))
        painter.setFont(QFont("Segoe UI Semibold", 11))
        painter.drawText(294, 186, PRODUCT_SUBTITLE.upper())
        painter.setPen(QColor("#B3C1CC"))
        painter.setFont(QFont("Cascadia Mono", 10))
        painter.drawText(294, 240, f"VERSION {PRODUCT_VERSION}  ·  RELEASED {RELEASE_DATE.upper()}")
        painter.setPen(QColor("#778896"))
        painter.setFont(QFont("Segoe UI", 10))
        painter.drawText(294, 270, AUTHOR)
        painter.setPen(QColor("#20A7F5"))
        painter.fillRect(294, 301, 260, 3, QColor("#20A7F5"))
        painter.setPen(QColor("#B3C1CC"))
        painter.setFont(QFont("Segoe UI", 10))
        painter.drawText(294, 337, "Starting workstation services…")
    finally:
        painter.end()

    splash = QSplashScreen(pixmap, Qt.WindowType.WindowStaysOnTopHint)
    splash.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
    splash.show()
    return splash


def main() -> int:
    app = QApplication(sys.argv)
    app.setOrganizationName(COMPANY_NAME)
    app.setOrganizationDomain("venusglass.local")
    app.setApplicationName(PRODUCT_NAME)
    app.setApplicationDisplayName(PRODUCT_NAME)

    icon_path = brand_icon_path()
    if icon_path.is_file():
        app.setWindowIcon(QIcon(str(icon_path)))

    splash = _startup_splash()
    app.processEvents()

    repo_root = Path(__file__).resolve().parents[1]
    controller = VistaController(repo_root)
    engine = QQmlApplicationEngine()
    engine.addImageProvider("inspection", controller.image_provider)
    engine.rootContext().setContextProperty("vista", controller)

    main_qml = qml_directory() / "Main.qml"
    engine.load(QUrl.fromLocalFile(str(main_qml)))
    if not engine.rootObjects():
        splash.close()
        controller.shutdown()
        return 1

    # Main.qml owns a richer timed splash.  Fade this native preloader shortly
    # after a QML window exists so startup never shows a blank desktop.
    QTimer.singleShot(120, splash.close)
    app.aboutToQuit.connect(controller.shutdown)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
