def industrial_dark_stylesheet() -> str:
    """DPI-aware, high-contrast styling for an arm's-length HMI."""
    return r'''
    QMainWindow { background: #0b1017; }

    /* Points scale with the Windows display setting; fixed pixels did not. */
    QWidget {
        color: #f4f7fb;
        font-family: "Segoe UI";
        font-size: 12pt;
    }

    QWidget#MainSurface, QWidget#AutoPage, QWidget#ManualTestPage, QWidget#CalibrationPage {
        background: #0b1017;
    }

    QFrame { background: transparent; border: none; }

    QFrame#TopBar {
        background: #121923;
        border: 1px solid #2e3c4b;
        border-radius: 16px;
    }

    QFrame#NavRail {
        background: #101720;
        border: 1px solid #2a3746;
        border-radius: 16px;
    }

    QFrame#OperatorSummary, QFrame#AlarmCard {
        background: #121b26;
        border: 1px solid #334454;
        border-radius: 13px;
    }

    QLabel {
        color: #f4f7fb;
        background: transparent;
        border: none;
    }

    QLabel#AppTitle {
        color: #ffffff;
        font-size: 18pt;
        font-weight: 800;
    }

    QLabel#HeaderCaption {
        color: #9eacbb;
        font-size: 10.5pt;
        font-weight: 700;
    }

    QLabel#StateEyebrow {
        color: #94a7ba;
        font-size: 9.5pt;
        font-weight: 800;
        letter-spacing: 0.8px;
    }

    QLabel#StateLabel {
        color: #f7fbff;
        font-size: 21pt;
        font-weight: 900;
    }

    QLabel#StatusDetail {
        color: #c9d4df;
        font-size: 11pt;
        font-weight: 650;
    }

    QLabel#ToleranceCard {
        color: #dcecff;
        background: #132235;
        border: 1px solid #2d5a7c;
        border-radius: 10px;
        padding: 9px 12px;
        font-size: 10.5pt;
        font-weight: 800;
    }

    QLabel#AlarmLabel {
        color: #f5d49d;
        font-size: 10.5pt;
        font-weight: 750;
    }

    QPushButton {
        min-height: 44px;
        background: #19222d;
        border: 1px solid #334253;
        color: #f7f9fc;
        border-radius: 10px;
        padding: 8px 16px;
        font-size: 11.5pt;
        font-weight: 750;
    }

    QPushButton:hover { background: #202d3a; border-color: #4c6278; }
    QPushButton:pressed { background: #101720; }
    QPushButton:disabled { background: #121820; color: #647180; border-color: #222c37; }

    QPushButton#NavButton {
        min-height: 56px;
        background: transparent;
        border: 1px solid transparent;
        border-radius: 11px;
        color: #b9c4d0;
        font-size: 12.5pt;
        font-weight: 800;
        text-align: left;
        padding-left: 22px;
    }

    QPushButton#NavButton:hover {
        background: #17222d;
        color: #ffffff;
        border-color: #2a3b4b;
    }

    QPushButton#NavButton:checked {
        background: #11364a;
        color: #e8f8ff;
        border: 1px solid #2aaee8;
    }

    QPushButton#StartButton {
        min-height: 52px;
        background: #12613c;
        border-color: #21a663;
        color: #ffffff;
        font-size: 13pt;
        font-weight: 900;
    }
    QPushButton#StartButton:hover { background: #17764a; }

    QPushButton#StopButton {
        min-height: 52px;
        background: #63242a;
        border-color: #b84953;
        color: #ffffff;
        font-size: 13pt;
        font-weight: 900;
    }
    QPushButton#StopButton:hover { background: #7a2c34; }

    QPushButton#AlarmAckButton {
        min-height: 42px;
        background: #302419;
        border-color: #7e5c2e;
        color: #ffe6b7;
        font-size: 10.5pt;
        font-weight: 800;
    }

    QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {
        min-height: 42px;
        background: #18212b;
        border: 1px solid #344456;
        border-radius: 9px;
        padding: 3px 12px;
        color: #ffffff;
        font-size: 11.5pt;
        font-weight: 650;
        selection-background-color: #176b91;
    }

    QComboBox:focus, QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
        border: 1px solid #38bdf8;
    }

    QComboBox QAbstractItemView {
        background: #151d26;
        color: #ffffff;
        selection-background-color: #155f82;
        border: 1px solid #3a4b5d;
        padding: 6px;
    }

    QCheckBox, QRadioButton {
        color: #e3e9ef;
        font-size: 11.5pt;
        spacing: 10px;
        min-height: 30px;
    }

    QCheckBox::indicator, QRadioButton::indicator { width: 20px; height: 20px; }
    QCheckBox::indicator {
        border: 2px solid #607286;
        border-radius: 4px;
        background: #0d131a;
    }
    QCheckBox::indicator:checked { background: #1d91c4; border-color: #49c8ff; }

    QSlider { min-height: 30px; }
    QSlider::groove:horizontal { height: 8px; background: #25313e; border-radius: 4px; }
    QSlider::sub-page:horizontal { background: #258fc0; border-radius: 4px; }
    QSlider::handle:horizontal {
        width: 22px;
        height: 22px;
        margin: -7px 0;
        border-radius: 11px;
        background: #f2f7fb;
        border: 2px solid #38bdf8;
    }

    QGroupBox {
        background: #10171f;
        border: 1px solid #2c3a49;
        border-radius: 13px;
        margin-top: 18px;
        padding-top: 14px;
        font-size: 11.5pt;
        font-weight: 850;
        color: #f7f9fc;
    }

    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 14px;
        padding: 2px 8px;
        color: #9fdfff;
        background: #10171f;
    }

    QScrollArea { border: none; background: transparent; }
    QScrollArea > QWidget > QWidget { background: transparent; }
    QScrollBar:vertical {
        width: 12px;
        background: #0e141b;
        margin: 2px;
        border-radius: 6px;
    }
    QScrollBar::handle:vertical {
        min-height: 40px;
        background: #3a4c5e;
        border-radius: 5px;
    }
    QScrollBar::handle:vertical:hover { background: #50677d; }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

    QToolTip {
        color: #ffffff;
        background: #17212b;
        border: 1px solid #58728b;
        padding: 9px;
        font-size: 10.5pt;
    }
    '''
