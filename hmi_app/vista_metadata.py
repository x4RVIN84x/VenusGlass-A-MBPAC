"""VG VISTA product identity and packaged-resource helpers."""

from __future__ import annotations

import os
import sys
from pathlib import Path


PRODUCT_NAME = "VG VISTA"
PRODUCT_SUBTITLE = "Industrial Vision & Inspection"
PRODUCT_VERSION = "1.6.1"
RELEASE_DATE = "September 9, 2026"
AUTHOR = 'Seyed Mohammad "Arvin" Afrazeh'
COMPANY_NAME = "Venus Glass"


def package_root() -> Path:
    """Return the directory containing packaged QML and image resources."""
    local_root = Path(__file__).resolve().parent
    if (local_root / "qml").is_dir():
        return local_root

    # A standalone Windows build keeps included data beside the packaged
    # modules.  This fallback also keeps the source launcher straightforward.
    frozen_root = Path(sys.executable).resolve().parent / "hmi_app"
    if (frozen_root / "qml").is_dir():
        return frozen_root
    return local_root


def qml_directory() -> Path:
    return package_root() / "qml"


def brand_icon_path() -> Path:
    return qml_directory() / "assets" / "venus-glass-logo.png"


def workstation_data_directory() -> Path:
    """Return a writable data directory that survives application upgrades.

    Installed software must not write SQLite files to ``Program Files``.  A
    commissioning technician may set ``VG_VISTA_PORTABLE=1`` (or explicitly
    choose ``VG_VISTA_DATA_DIR``) for an approved portable deployment.
    """
    configured = os.environ.get("VG_VISTA_DATA_DIR", "").strip()
    if configured:
        path = Path(configured).expanduser()
    elif os.environ.get("VG_VISTA_PORTABLE", "").strip().lower() in {"1", "true", "yes"}:
        path = Path(sys.executable).resolve().parent / "data"
    else:
        from PySide6.QtCore import QStandardPaths

        location = QStandardPaths.writableLocation(QStandardPaths.AppLocalDataLocation)
        path = Path(location) / "data" if location else Path.cwd() / "data"

    path.mkdir(parents=True, exist_ok=True)
    return path
