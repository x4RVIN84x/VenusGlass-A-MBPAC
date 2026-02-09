def industrial_dark_stylesheet() -> str:
    return '''
    QMainWindow { background: #0f0f10; }
    QWidget { color: #eaeaea; font-family: "Segoe UI"; font-size: 12px; }
    QFrame { background: #151517; border: 1px solid #242426; border-radius: 12px; }
    QLabel { color: #eaeaea; }
    QPushButton {
        background: #1f1f22;
        border: 1px solid #2d2d31;
        color: #f0f0f0;
        border-radius: 12px;
        padding: 10px 12px;
        font-weight: 700;
        letter-spacing: 0.2px;
    }
    QPushButton:hover { background: #252529; }
    QPushButton:pressed { background: #121214; }
    QComboBox {
        background: #1f1f22;
        border: 1px solid #2d2d31;
        border-radius: 10px;
        padding: 6px 10px;
        font-weight: 600;
    }
    QComboBox QAbstractItemView {
        background: #1b1b1e;
        selection-background-color: #2b2b30;
        border: 1px solid #2d2d31;
    }
    QSlider::groove:horizontal { height: 6px; background: #2a2a2e; border-radius: 3px; }
    QSlider::handle:horizontal { width: 16px; margin: -6px 0; border-radius: 8px; background: #c7c7c7; }
    QGroupBox {
        border: 1px solid #2d2d31;
        border-radius: 12px;
        margin-top: 10px;
        font-weight: 800;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 6px 0 6px;
        color: #d9d9d9;
    }
    '''
