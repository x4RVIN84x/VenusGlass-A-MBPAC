"""Smoke tests for the standalone VG VISTA QML shell.

The test deliberately keeps the camera closed.  It proves that a packaged
workstation can build the QML interface, its image provider, and the durable
report store without requiring a connected inspection camera.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest


# This must be set before importing any Qt GUI modules on a build agent.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QUrl
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine

from hmi_app.vista_backend import VistaController
from hmi_app.vista_metadata import qml_directory


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class VistaQmlSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QGuiApplication.instance() or QGuiApplication([])

    def test_qml_shell_loads_without_a_camera(self) -> None:
        with tempfile.TemporaryDirectory() as data_directory:
            original_data_directory = os.environ.get("VG_VISTA_DATA_DIR")
            os.environ["VG_VISTA_DATA_DIR"] = data_directory
            controller = VistaController(REPOSITORY_ROOT)
            engine = QQmlApplicationEngine()
            try:
                engine.addImageProvider("inspection", controller.image_provider)
                engine.rootContext().setContextProperty("vista", controller)
                engine.load(QUrl.fromLocalFile(str(qml_directory() / "Main.qml")))
                self.assertTrue(engine.rootObjects(), "VG VISTA QML did not create a root object")
            finally:
                controller.shutdown()
                if original_data_directory is None:
                    os.environ.pop("VG_VISTA_DATA_DIR", None)
                else:
                    os.environ["VG_VISTA_DATA_DIR"] = original_data_directory


if __name__ == "__main__":
    unittest.main()
