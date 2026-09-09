import QtQuick 2.15
import QtQuick.Controls 2.15

Button {
    id: control

    property var theme
    property color accentColor: theme ? theme.accent : "#20A7F5"
    property bool destructive: false
    property bool outlined: false

    implicitHeight: theme ? theme.controlHeight : 44
    implicitWidth: 132
    hoverEnabled: true
    font.family: theme ? theme.displayFontFamily : "Segoe UI Semibold"
    font.pixelSize: theme ? theme.controlFontSize : 15

    contentItem: Text {
        text: control.text
        color: !control.enabled ? "#778896" : control.outlined ? control.accentColor : "#041019"
        font: control.font
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
    }

    background: Rectangle {
        radius: theme ? theme.radiusMedium : 10
        color: {
            if (!control.enabled)
                return "#18232E"
            if (control.outlined)
                return control.down ? "#18232E" : control.hovered ? "#202D39" : "#131C25"
            if (control.down)
                return Qt.darker(control.accentColor, 1.18)
            return control.hovered ? Qt.lighter(control.accentColor, 1.10) : control.accentColor
        }
        border.width: control.outlined ? 1 : 0
        border.color: control.enabled ? control.accentColor : "#2A3A49"
        opacity: control.enabled ? 1.0 : 0.5

        Behavior on color {
            ColorAnimation { duration: theme ? theme.animationFast : 100 }
        }
    }
}
