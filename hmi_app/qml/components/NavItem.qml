import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Button {
    id: control

    property var theme
    property bool selected: false
    property string glyph: ""

    implicitHeight: 52
    leftPadding: 14
    rightPadding: 12
    hoverEnabled: true
    font.family: theme ? theme.displayFontFamily : "Segoe UI Semibold"
    font.pixelSize: theme ? theme.bodyFontSize : 14

    contentItem: RowLayout {
        spacing: 12
        Text {
            text: control.glyph
            color: control.selected ? (theme ? theme.accent : "#20A7F5") : (theme ? theme.textMuted : "#778896")
            font.pixelSize: 18
            Layout.preferredWidth: 20
            horizontalAlignment: Text.AlignHCenter
        }
        Text {
            text: control.text
            color: control.selected ? (theme ? theme.textPrimary : "white") : (theme ? theme.textSecondary : "#B3C1CC")
            font: control.font
            Layout.fillWidth: true
            elide: Text.ElideRight
        }
        Rectangle {
            visible: control.selected
            width: 4
            height: 22
            radius: 2
            color: theme ? theme.accent : "#20A7F5"
        }
    }

    background: Rectangle {
        radius: theme ? theme.radiusMedium : 10
        color: control.selected ? "#1A2D3B" : control.hovered ? "#18232E" : "transparent"
        border.width: control.selected ? 1 : 0
        border.color: theme ? "#28506B" : "#28506B"
    }
}
