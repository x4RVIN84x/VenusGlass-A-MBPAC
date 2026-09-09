import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import "../components" as Components

Item {
    id: root

    property var theme

    Flickable {
        id: scroll
        anchors.fill: parent
        clip: true
        contentWidth: width
        contentHeight: content.implicitHeight + theme.pageMargin * 2
        boundsBehavior: Flickable.StopAtBounds
        ScrollBar.vertical: ScrollBar { }

        ColumnLayout {
            id: content
            x: theme.pageMargin
            y: theme.pageMargin
            width: Math.max(720, scroll.width - theme.pageMargin * 2)
            spacing: theme.space4

            Rectangle {
                id: statusRibbon
                Layout.fillWidth: true
                Layout.preferredHeight: theme.statusRibbonHeight
                radius: theme.radiusLarge
                color: Qt.rgba(theme.stateColor(vista.statusCode).r, theme.stateColor(vista.statusCode).g, theme.stateColor(vista.statusCode).b, 0.14)
                border.width: 1
                border.color: theme.stateColor(vista.statusCode)

                Behavior on color { ColorAnimation { duration: theme.animationStandard } }
                Behavior on border.color { ColorAnimation { duration: theme.animationStandard } }

                RowLayout {
                    anchors.fill: parent
                    anchors.margins: theme.panelPadding
                    spacing: theme.space6

                    ColumnLayout {
                        Layout.preferredWidth: 310
                        Layout.fillHeight: true
                        spacing: 4

                        Text {
                            text: "LIVE INSPECTION STATUS"
                            color: theme.textSecondary
                            font.family: theme.displayFontFamily
                            font.pixelSize: theme.labelFontSize
                            font.letterSpacing: 1.3
                        }
                        Text {
                            text: String(vista.statusCode).replace(/_/g, " ")
                            color: theme.stateColor(vista.statusCode)
                            font.family: theme.displayFontFamily
                            font.pixelSize: theme.heroStatusFontSize
                            font.letterSpacing: 0.7
                            elide: Text.ElideRight
                            Layout.fillWidth: true
                        }
                    }

                    Rectangle {
                        Layout.preferredWidth: 1
                        Layout.fillHeight: true
                        color: theme.stateColor(vista.statusCode)
                        opacity: 0.4
                    }

                    ColumnLayout {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        spacing: 5
                        Text {
                            text: "CURRENT DECISION"
                            color: theme.textMuted
                            font.family: theme.displayFontFamily
                            font.pixelSize: theme.labelFontSize
                            font.letterSpacing: 1.0
                        }
                        Text {
                            text: vista.statusText
                            color: theme.textPrimary
                            font.family: theme.displayFontFamily
                            font.pixelSize: theme.sectionFontSize
                            wrapMode: Text.WordWrap
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            verticalAlignment: Text.AlignVCenter
                        }
                    }

                    ColumnLayout {
                        Layout.preferredWidth: 122
                        spacing: 2
                        Text {
                            text: "CONFIDENCE"
                            color: theme.textMuted
                            font.family: theme.displayFontFamily
                            font.pixelSize: theme.labelFontSize
                            font.letterSpacing: 1.0
                            Layout.alignment: Qt.AlignRight
                        }
                        Text {
                            text: vista.confidencePercent + "%"
                            color: theme.stateColor(vista.statusCode)
                            font.family: theme.numericFontFamily
                            font.pixelSize: 34
                            font.bold: true
                            Layout.alignment: Qt.AlignRight
                        }
                    }
                }
            }

            RowLayout {
                Layout.fillWidth: true
                Layout.preferredHeight: 600
                spacing: theme.space4

                Rectangle {
                    id: feedPanel
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    color: "#05080C"
                    radius: theme.radiusLarge
                    border.width: 1
                    border.color: theme.borderStrong
                    clip: true

                    Image {
                        id: cameraFeed
                        anchors.fill: parent
                        anchors.margins: 1
                        source: "image://inspection/live?version=" + vista.imageVersion
                        cache: false
                        fillMode: Image.PreserveAspectFit
                        smooth: true
                        asynchronous: false
                    }

                    Rectangle {
                        anchors.left: parent.left
                        anchors.top: parent.top
                        anchors.margins: theme.space3
                        height: 30
                        width: feedTag.implicitWidth + theme.space4
                        radius: theme.radiusSmall
                        color: Qt.rgba(theme.appBackground.r, theme.appBackground.g, theme.appBackground.b, 0.82)
                        border.width: 1
                        border.color: theme.border
                        Text {
                            id: feedTag
                            anchors.centerIn: parent
                            text: vista.running ? "LIVE CAMERA · " + vista.currentRecipe : "CAMERA STANDBY"
                            color: vista.running ? theme.textPrimary : theme.textMuted
                            font.family: theme.displayFontFamily
                            font.pixelSize: 11
                            font.letterSpacing: 0.8
                        }
                    }

                    Rectangle {
                        visible: !vista.running
                        anchors.centerIn: parent
                        width: 340
                        height: 148
                        radius: theme.radiusMedium
                        color: Qt.rgba(theme.surface.r, theme.surface.g, theme.surface.b, 0.94)
                        border.width: 1
                        border.color: theme.borderStrong
                        ColumnLayout {
                            anchors.fill: parent
                            anchors.margins: theme.panelPadding
                            spacing: theme.space2
                            Text {
                                text: vista.statusCode === "CAMERA_FAULT" ? "Camera connection required" : "Inspection is on standby"
                                color: theme.textPrimary
                                font.family: theme.displayFontFamily
                                font.pixelSize: theme.sectionFontSize
                                Layout.alignment: Qt.AlignHCenter
                            }
                            Text {
                                text: vista.cameraMessage
                                color: theme.textSecondary
                                font.family: theme.fontFamily
                                font.pixelSize: theme.bodyFontSize
                                wrapMode: Text.WordWrap
                                horizontalAlignment: Text.AlignHCenter
                                Layout.fillWidth: true
                            }
                        }
                    }
                }

                Rectangle {
                    id: decisionPanel
                    Layout.preferredWidth: 340
                    Layout.fillHeight: true
                    color: theme.surface
                    radius: theme.radiusLarge
                    border.width: 1
                    border.color: theme.border

                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: theme.panelPadding
                        spacing: theme.space4

                        Text {
                            text: "INSPECTION CONTROL"
                            color: theme.textPrimary
                            font.family: theme.displayFontFamily
                            font.pixelSize: theme.sectionFontSize
                        }
                        Text {
                            text: vista.currentRecipe.length ? vista.currentRecipe : "No recipe loaded"
                            color: theme.textSecondary
                            font.family: theme.numericFontFamily
                            font.pixelSize: theme.labelFontSize
                            elide: Text.ElideRight
                            Layout.fillWidth: true
                        }

                        Rectangle {
                            Layout.fillWidth: true
                            Layout.preferredHeight: 80
                            color: Qt.rgba(theme.stateColor(vista.statusCode).r, theme.stateColor(vista.statusCode).g, theme.stateColor(vista.statusCode).b, 0.10)
                            radius: theme.radiusMedium
                            border.width: 1
                            border.color: theme.stateColor(vista.statusCode)
                            ColumnLayout {
                                anchors.fill: parent
                                anchors.margins: theme.space3
                                spacing: 2
                                Text {
                                    text: "BASEPLATE EVIDENCE"
                                    color: theme.textMuted
                                    font.family: theme.displayFontFamily
                                    font.pixelSize: 10
                                    font.letterSpacing: 1.0
                                }
                                Text {
                                    text: vista.baseplateText
                                    color: theme.textPrimary
                                    font.family: theme.fontFamily
                                    font.pixelSize: theme.bodyFontSize
                                    wrapMode: Text.WordWrap
                                    Layout.fillWidth: true
                                }
                            }
                        }

                        GridLayout {
                            Layout.fillWidth: true
                            columns: 3
                            columnSpacing: theme.space2
                            rowSpacing: theme.space2
                            Repeater {
                                model: [
                                    { "caption": "X OFFSET", "value": vista.deltaX, "detail": "" },
                                    { "caption": "Y OFFSET", "value": vista.deltaY, "detail": "" },
                                    { "caption": "ANGLE", "value": vista.angle, "detail": "" }
                                ]
                                delegate: Rectangle {
                                    required property var modelData
                                    Layout.fillWidth: true
                                    Layout.preferredHeight: 92
                                    radius: theme.radiusSmall
                                    color: theme.inputSurface
                                    border.width: 1
                                    border.color: theme.border
                                    ColumnLayout {
                                        anchors.fill: parent
                                        anchors.margins: theme.space2
                                        spacing: 2
                                        Text {
                                            text: modelData.caption
                                            color: theme.textMuted
                                            font.family: theme.displayFontFamily
                                            font.pixelSize: 9
                                            font.letterSpacing: 0.7
                                        }
                                        Text {
                                            text: modelData.value
                                            color: theme.textPrimary
                                            font.family: theme.numericFontFamily
                                            font.pixelSize: 17
                                            font.bold: true
                                            elide: Text.ElideRight
                                            Layout.fillWidth: true
                                        }
                                        Text {
                                            text: modelData.detail
                                            visible: modelData.detail.length > 0
                                            color: theme.textSecondary
                                            font.family: theme.fontFamily
                                            font.pixelSize: 10
                                        }
                                    }
                                }
                            }
                        }

                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: theme.space2
                            RowLayout {
                                Layout.fillWidth: true
                                Text {
                                    text: "DECISION CONFIDENCE"
                                    color: theme.textSecondary
                                    font.family: theme.displayFontFamily
                                    font.pixelSize: theme.labelFontSize
                                    font.letterSpacing: 0.7
                                }
                                Item { Layout.fillWidth: true }
                                Text {
                                    text: vista.confidencePercent + "%"
                                    color: theme.stateColor(vista.statusCode)
                                    font.family: theme.numericFontFamily
                                    font.pixelSize: theme.labelFontSize
                                    font.bold: true
                                }
                            }
                            Rectangle {
                                Layout.fillWidth: true
                                Layout.preferredHeight: 10
                                radius: 5
                                color: theme.inputSurface
                                border.width: 1
                                border.color: theme.border
                                Rectangle {
                                    width: parent.width * Math.min(1, Math.max(0, vista.confidencePercent / 100.0))
                                    height: parent.height
                                    radius: parent.radius
                                    color: theme.stateColor(vista.statusCode)
                                    Behavior on width { NumberAnimation { duration: theme.animationFast } }
                                    Behavior on color { ColorAnimation { duration: theme.animationStandard } }
                                }
                            }
                        }

                        Rectangle {
                            Layout.fillWidth: true
                            Layout.preferredHeight: 92
                            radius: theme.radiusMedium
                            color: theme.inputSurface
                            border.width: 1
                            border.color: theme.border
                            ColumnLayout {
                                anchors.fill: parent
                                anchors.margins: theme.space3
                                spacing: 4
                                Text {
                                    text: "INSPECTION CYCLE"
                                    color: theme.textMuted
                                    font.family: theme.displayFontFamily
                                    font.pixelSize: 10
                                    font.letterSpacing: 1.0
                                }
                                Text {
                                    text: vista.cycleText
                                    color: theme.textPrimary
                                    font.family: theme.displayFontFamily
                                    font.pixelSize: theme.bodyFontSize
                                    wrapMode: Text.WordWrap
                                    Layout.fillWidth: true
                                }
                                Text {
                                    text: vista.lastSaved
                                    color: theme.textSecondary
                                    font.family: theme.fontFamily
                                    font.pixelSize: 11
                                    elide: Text.ElideRight
                                    Layout.fillWidth: true
                                }
                            }
                        }

                        Item { Layout.fillHeight: true }

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: theme.space3
                            Components.VistaButton {
                                Layout.fillWidth: true
                                theme: root.theme
                                text: vista.running ? "RUNNING" : "START INSPECTION"
                                accentColor: theme.passColor
                                enabled: !vista.running && vista.currentRecipe.length > 0
                                onClicked: vista.startInspection()
                            }
                            Components.VistaButton {
                                Layout.fillWidth: true
                                theme: root.theme
                                text: "STOP"
                                accentColor: theme.failColor
                                enabled: vista.running
                                onClicked: vista.stopInspection()
                            }
                        }
                    }
                }
            }

            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 42
                radius: theme.radiusSmall
                color: theme.surface
                border.width: 1
                border.color: theme.border
                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: theme.space3
                    anchors.rightMargin: theme.space3
                    spacing: theme.space3
                    Rectangle {
                        Layout.preferredWidth: 8
                        Layout.preferredHeight: 8
                        radius: 4
                        color: vista.running ? theme.passColor : theme.readyColor
                    }
                    Text {
                        text: vista.running ? "Inspection service active · result recording and PLC telemetry are isolated from the UI." : "Inspection service stopped · select a recipe and start when the station is ready."
                        color: theme.textSecondary
                        font.family: theme.fontFamily
                        font.pixelSize: theme.labelFontSize
                        Layout.fillWidth: true
                        elide: Text.ElideRight
                    }
                }
            }
        }
    }
}
