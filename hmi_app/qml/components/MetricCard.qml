import QtQuick 2.15
import QtQuick.Layouts 1.15

Rectangle {
    id: card

    property var theme
    property string caption: "METRIC"
    property string value: "—"
    property string detail: ""
    property color valueColor: theme ? theme.textPrimary : "white"

    implicitHeight: 112
    radius: theme ? theme.radiusMedium : 10
    color: theme ? theme.surface : "#131C25"
    border.width: 1
    border.color: theme ? theme.border : "#2A3A49"

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: theme ? theme.compactPanelPadding : 12
        spacing: 4
        Text {
            text: card.caption
            color: theme ? theme.textMuted : "#778896"
            font.family: theme ? theme.displayFontFamily : "Segoe UI Semibold"
            font.pixelSize: theme ? theme.labelFontSize : 12
            font.letterSpacing: 1.0
        }
        Text {
            text: card.value
            color: card.valueColor
            font.family: theme ? theme.numericFontFamily : "Cascadia Mono"
            font.pixelSize: theme ? theme.metricFontSize : 30
            font.bold: true
            Layout.fillWidth: true
            elide: Text.ElideRight
        }
        Text {
            visible: card.detail.length > 0
            text: card.detail
            color: theme ? theme.textSecondary : "#B3C1CC"
            font.family: theme ? theme.fontFamily : "Segoe UI"
            font.pixelSize: theme ? theme.labelFontSize : 12
            Layout.fillWidth: true
            elide: Text.ElideRight
        }
    }
}
