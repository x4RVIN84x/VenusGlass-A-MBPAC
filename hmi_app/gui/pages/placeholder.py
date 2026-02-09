from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt

class PlaceholderPage(QWidget):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lbl = QLabel(title)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("font-size: 22px; font-weight: 900;")
        lay.addStretch(1)
        lay.addWidget(lbl)
        lay.addStretch(1)
