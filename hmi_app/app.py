import sys
from PySide6.QtWidgets import QApplication
from hmi_app.gui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    # Keep Qt settings and the inspection-history database in a stable,
    # human-readable application namespace (instead of a generic "python"
    # folder when launched from source).
    app.setOrganizationName("MBPAC")
    app.setOrganizationDomain("mbpac.local")
    app.setApplicationName("MBPAC QC Station")
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
