import sys
from PySide6.QtWidgets import QApplication
from hmi_app.gui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    # The station UI is designed for the operator's full display.  On the
    # common 125%-scaled 1080p panel this avoids an oversized fixed window
    # being clipped by the Windows taskbar.
    w.showMaximized()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
