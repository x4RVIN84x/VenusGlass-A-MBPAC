import QtQuick 2.15
import QtQuick.Layouts 1.15

Item {
    id: splash

    property var theme
    property int displayDuration: 2600
    property bool complete: false
    signal finished()

    anchors.fill: parent
    opacity: complete ? 0 : 1
    visible: opacity > 0.01

    Behavior on opacity {
        NumberAnimation { duration: theme ? theme.animationSlow : 280; easing.type: Easing.OutCubic }
    }

    Rectangle {
        anchors.fill: parent
        color: theme ? theme.appBackground : "#090E13"
    }
    Rectangle {
        anchors.fill: parent
        color: "#071A2A"
        opacity: 0.58
    }
    Rectangle {
        width: Math.max(parent.width, parent.height) * 0.85
        height: width
        radius: width / 2
        anchors.centerIn: parent
        color: "#0C547E"
        opacity: 0.13
    }

    ColumnLayout {
        anchors.centerIn: parent
        width: Math.min(parent.width - 64, 560)
        spacing: theme ? theme.space4 : 16

        Item { Layout.fillWidth: true; Layout.preferredHeight: 12 }
        Image {
            source: "../assets/venus-glass-logo.png"
            fillMode: Image.PreserveAspectFit
            smooth: true
            asynchronous: false
            Layout.alignment: Qt.AlignHCenter
            Layout.preferredWidth: 148
            Layout.preferredHeight: 148
        }
        Text {
            text: "VG VISTA"
            color: theme ? theme.textPrimary : "white"
            font.family: theme ? theme.displayFontFamily : "Segoe UI Semibold"
            font.pixelSize: 46
            font.letterSpacing: 3.0
            Layout.alignment: Qt.AlignHCenter
        }
        Text {
            text: "INDUSTRIAL VISION & INSPECTION"
            color: theme ? theme.accent : "#20A7F5"
            font.family: theme ? theme.displayFontFamily : "Segoe UI Semibold"
            font.pixelSize: 13
            font.letterSpacing: 2.0
            Layout.alignment: Qt.AlignHCenter
        }
        Item { Layout.fillWidth: true; Layout.preferredHeight: 20 }
        Rectangle {
            Layout.alignment: Qt.AlignHCenter
            Layout.preferredWidth: 320
            Layout.preferredHeight: 3
            radius: 2
            color: theme ? theme.border : "#2A3A49"
            Rectangle {
                height: parent.height
                width: parent.width * 0.62
                radius: parent.radius
                color: theme ? theme.accent : "#20A7F5"
                SequentialAnimation on width {
                    loops: Animation.Infinite
                    NumberAnimation { to: parent.width * 0.92; duration: 840; easing.type: Easing.InOutQuad }
                    NumberAnimation { to: parent.width * 0.38; duration: 840; easing.type: Easing.InOutQuad }
                }
            }
        }
        Text {
            text: "Version 1.6.1  ·  Released September 9, 2026"
            color: theme ? theme.textSecondary : "#B3C1CC"
            font.family: theme ? theme.numericFontFamily : "Cascadia Mono"
            font.pixelSize: 13
            Layout.alignment: Qt.AlignHCenter
        }
        Text {
            text: "Seyed Mohammad \"Arvin\" Afrazeh"
            color: theme ? theme.textMuted : "#778896"
            font.family: theme ? theme.fontFamily : "Segoe UI"
            font.pixelSize: 13
            Layout.alignment: Qt.AlignHCenter
        }
    }

    Timer {
        interval: splash.displayDuration
        running: true
        repeat: false
        onTriggered: {
            splash.complete = true
            splash.finished()
        }
    }
}
