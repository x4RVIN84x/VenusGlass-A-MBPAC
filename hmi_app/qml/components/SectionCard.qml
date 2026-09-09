import QtQuick 2.15
import QtQuick.Layouts 1.15

Rectangle {
    id: card

    property var theme
    property string title: ""
    property string subtitle: ""
    default property alias body: content.data

    color: theme ? theme.surface : "#131C25"
    radius: theme ? theme.radiusMedium : 10
    border.width: 1
    border.color: theme ? theme.border : "#2A3A49"
    implicitHeight: header.implicitHeight + content.implicitHeight + (theme ? theme.panelPadding * 2 : 40)

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: theme ? theme.panelPadding : 20
        spacing: theme ? theme.space4 : 16
        ColumnLayout {
            id: header
            spacing: 3
            Text {
                text: card.title
                color: theme ? theme.textPrimary : "white"
                font.family: theme ? theme.displayFontFamily : "Segoe UI Semibold"
                font.pixelSize: theme ? theme.sectionFontSize : 18
            }
            Text {
                visible: card.subtitle.length > 0
                text: card.subtitle
                color: theme ? theme.textSecondary : "#B3C1CC"
                font.family: theme ? theme.fontFamily : "Segoe UI"
                font.pixelSize: theme ? theme.bodyFontSize : 14
                wrapMode: Text.WordWrap
                Layout.fillWidth: true
            }
        }
        ColumnLayout {
            id: content
            Layout.fillWidth: true
            spacing: theme ? theme.space3 : 12
        }
    }
}
