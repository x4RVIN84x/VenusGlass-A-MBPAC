"""Create the multi-resolution VG VISTA Windows icon from the approved logo."""

from __future__ import annotations

import argparse
from pathlib import Path


ICON_SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)


def create_icon(source: Path, destination: Path) -> None:
    """Write an ICO containing all common Windows shell and taskbar sizes."""
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - operator-facing failure
        raise SystemExit(
            "Pillow is required. Install build tools with: "
            ".\\.venv\\Scripts\\python.exe -m pip install -r requirements-build.txt"
        ) from error

    if not source.is_file():
        raise SystemExit(f"Logo source was not found: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as logo:
        # Start from a 256 px RGBA master so every Windows icon size is embedded
        # from the same source and scales cleanly in Explorer and on the taskbar.
        master = logo.convert("RGBA").resize((256, 256), Image.Resampling.LANCZOS)
        master.save(destination, format="ICO", sizes=[(size, size) for size in ICON_SIZES])

    print(f"Created {destination} with {len(ICON_SIZES)} icon resolutions.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("hmi_app/qml/assets/venus-glass-logo.png"),
        help="Approved Venus Glass PNG logo.",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("hmi_app/qml/assets/vg-vista.ico"),
        help="Generated Windows icon path.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    create_icon(arguments.source, arguments.destination)
