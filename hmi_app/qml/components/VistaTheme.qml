// Central visual tokens for the VG VISTA operator interface.
//
// Keep operational states semantic: controls should use `passColor`,
// `failColor`, `trackingColor`, and `baseplateMissingColor` rather than
// hard-coded values. This lets every screen communicate the same meaning.
import QtQuick 2.15

QtObject {
    id: theme

    // Brand-neutral industrial dark palette. The high contrast is deliberate:
    // this app is intended for long shifts and bright factory environments.
    readonly property color appBackground: "#090E13"
    readonly property color workspaceBackground: "#0E151C"
    readonly property color surface: "#131C25"
    readonly property color surfaceRaised: "#18232E"
    readonly property color surfaceOverlay: "#202D39"
    readonly property color inputSurface: "#0C131A"
    readonly property color border: "#2A3A49"
    readonly property color borderStrong: "#40566B"
    readonly property color divider: "#20303D"

    readonly property color textPrimary: "#F2F7FB"
    readonly property color textSecondary: "#B3C1CC"
    readonly property color textMuted: "#778896"
    readonly property color textOnAccent: "#041019"
    readonly property color focusRing: "#4CB8FF"
    readonly property color accent: "#20A7F5"
    readonly property color accentHover: "#54BDF7"
    readonly property color accentPressed: "#0879C3"

    // Inspection states. Do not use success/fail hue alone to convey status;
    // pair these with explicit labels and an icon in production components.
    readonly property color readyColor: "#65829A"
    readonly property color trackingColor: "#E7A52B"
    readonly property color passColor: "#2BCB7F"
    readonly property color passMuted: "#1C8D5A"
    readonly property color failColor: "#D84B5B"
    readonly property color failMuted: "#A92F43"
    readonly property color baseplateMissingColor: "#80461B"
    readonly property color baseplateMissingMuted: "#5F3314"
    readonly property color offlineColor: "#70808F"
    readonly property color warningColor: "#F0B34A"

    // Eight-point-derived rhythm. Prefer these over arbitrary pixel values.
    readonly property int space1: 4
    readonly property int space2: 8
    readonly property int space3: 12
    readonly property int space4: 16
    readonly property int space5: 20
    readonly property int space6: 24
    readonly property int space8: 32
    readonly property int space10: 40
    readonly property int pageMargin: 24
    readonly property int panelPadding: 20
    readonly property int compactPanelPadding: 12

    // Sizing keeps touch targets practical while retaining a dense control-room
    // layout. Scale these later through the accessibility setting, not locally.
    readonly property int controlHeight: 44
    readonly property int compactControlHeight: 36
    readonly property int navRailWidth: 244
    readonly property int topBarHeight: 72
    readonly property int statusRibbonHeight: 116
    readonly property int radiusSmall: 6
    readonly property int radiusMedium: 10
    readonly property int radiusLarge: 14
    readonly property int borderWidth: 1
    readonly property int focusBorderWidth: 2

    // Typography deliberately favors installed Windows fonts and tabular
    // figures for measurements, timestamps, recipes, and PLC values.
    readonly property string fontFamily: "Segoe UI"
    readonly property string displayFontFamily: "Segoe UI Semibold"
    readonly property string numericFontFamily: "Cascadia Mono"
    readonly property int labelFontSize: 12
    readonly property int bodyFontSize: 14
    readonly property int controlFontSize: 15
    readonly property int sectionFontSize: 18
    readonly property int pageTitleFontSize: 28
    readonly property int metricFontSize: 32
    readonly property int heroStatusFontSize: 42

    // Motion must support clear feedback without making alarms visually noisy.
    readonly property int animationFast: 100
    readonly property int animationStandard: 180
    readonly property int animationSlow: 280
    readonly property real disabledOpacity: 0.42
    readonly property real mutedOpacity: 0.68

    // Use this helper when the UI receives a normalized inspection state.
    // It is intentionally tolerant of the legacy "BASEPLATE NOT FOUND" label.
    function stateColor(state) {
        const normalized = String(state || "").toUpperCase()
        if (normalized === "PASS")
            return passColor
        if (normalized === "FAIL")
            return failColor
        if (normalized === "BASEPLATE_WARNING" || normalized === "BASEPLATE_MISSING" || normalized === "BASEPLATE NOT FOUND")
            return baseplateMissingColor
        if (normalized === "TRACKING" || normalized === "SEARCH")
            return trackingColor
        return readyColor
    }
}
