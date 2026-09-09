import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import "../components" as Components

Item {
    id: page

    property var theme

    Flickable {
        id: scrollView
        anchors.fill: parent
        contentWidth: width
        contentHeight: calibrationContent.implicitHeight + (theme ? theme.pageMargin * 2 : 48)
        clip: true
        boundsBehavior: Flickable.StopAtBounds

        ScrollBar.vertical: ScrollBar {
            policy: ScrollBar.AsNeeded
            width: 10
        }

        ColumnLayout {
            id: calibrationContent
            x: theme ? theme.pageMargin : 24
            y: theme ? theme.pageMargin : 24
            width: scrollView.width - (theme ? theme.pageMargin * 2 : 48)
            spacing: theme ? theme.space5 : 20

            RowLayout {
                Layout.fillWidth: true
                spacing: theme ? theme.space4 : 16

                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 4
                    Text {
                        text: "Calibration workspace"
                        color: theme ? theme.textPrimary : "white"
                        font.family: theme ? theme.displayFontFamily : "Segoe UI Semibold"
                        font.pixelSize: theme ? theme.pageTitleFontSize : 28
                    }
                    Text {
                        text: "Controlled recipe and golden-reference setup"
                        color: theme ? theme.textSecondary : "#B3C1CC"
                        font.family: theme ? theme.fontFamily : "Segoe UI"
                        font.pixelSize: theme ? theme.bodyFontSize : 14
                    }
                }

                Rectangle {
                    implicitWidth: 180
                    implicitHeight: 34
                    radius: theme ? theme.radiusSmall : 6
                    color: theme ? theme.surfaceRaised : "#18232E"
                    border.width: 1
                    border.color: theme ? theme.warningColor : "#F0B34A"
                    Text {
                        anchors.centerIn: parent
                        text: "MIGRATION STAGED"
                        color: theme ? theme.warningColor : "#F0B34A"
                        font.family: theme ? theme.displayFontFamily : "Segoe UI Semibold"
                        font.pixelSize: theme ? theme.labelFontSize : 12
                        font.letterSpacing: 0.9
                    }
                }
            }

            Rectangle {
                Layout.fillWidth: true
                implicitHeight: 76
                radius: theme ? theme.radiusMedium : 10
                color: theme ? theme.baseplateMissingMuted : "#5F3314"
                border.width: 1
                border.color: theme ? theme.warningColor : "#F0B34A"

                RowLayout {
                    anchors.fill: parent
                    anchors.margins: theme ? theme.compactPanelPadding : 12
                    spacing: theme ? theme.space3 : 12
                    Text {
                        text: "!"
                        color: theme ? theme.warningColor : "#F0B34A"
                        font.family: theme ? theme.displayFontFamily : "Segoe UI Semibold"
                        font.pixelSize: 28
                    }
                    ColumnLayout {
                        Layout.fillWidth: true
                        spacing: 2
                        Text {
                            text: "Recipe assets are read-only in this QML release"
                            color: theme ? theme.textPrimary : "white"
                            font.family: theme ? theme.displayFontFamily : "Segoe UI Semibold"
                            font.pixelSize: theme ? theme.bodyFontSize : 14
                        }
                        Text {
                            Layout.fillWidth: true
                            text: "Golden images, tolerances, and baseplate-orientation settings cannot be changed here until their reviewed calibration workflow is ported."
                            color: theme ? theme.textSecondary : "#B3C1CC"
                            font.family: theme ? theme.fontFamily : "Segoe UI"
                            font.pixelSize: theme ? theme.labelFontSize : 12
                            wrapMode: Text.WordWrap
                        }
                    }
                }
            }

            GridLayout {
                Layout.fillWidth: true
                columns: width >= 980 ? 2 : 1
                columnSpacing: theme ? theme.space4 : 16
                rowSpacing: theme ? theme.space4 : 16

                Components.SectionCard {
                    Layout.fillWidth: true
                    theme: page.theme
                    title: "Current recipe context"
                    subtitle: "The active recipe is selected from the application header."

                    ColumnLayout {
                        Layout.fillWidth: true
                        spacing: theme ? theme.space2 : 8
                        Text {
                            text: "ACTIVE RECIPE"
                            color: theme ? theme.textMuted : "#778896"
                            font.family: theme ? theme.displayFontFamily : "Segoe UI Semibold"
                            font.pixelSize: theme ? theme.labelFontSize : 12
                            font.letterSpacing: 0.9
                        }
                        Text {
                            Layout.fillWidth: true
                            text: vista && vista.currentRecipe.length > 0 ? vista.currentRecipe : "No valid recipe loaded"
                            color: theme ? theme.accentHover : "#54BDF7"
                            font.family: theme ? theme.numericFontFamily : "Cascadia Mono"
                            font.pixelSize: theme ? theme.controlFontSize : 15
                            elide: Text.ElideRight
                        }
                        Rectangle {
                            Layout.fillWidth: true
                            implicitHeight: 1
                            color: theme ? theme.divider : "#20303D"
                        }
                        Text {
                            Layout.fillWidth: true
                            text: "The current QML runtime can load this recipe for Auto Mode, but does not grant write access to its calibration files."
                            color: theme ? theme.textSecondary : "#B3C1CC"
                            font.family: theme ? theme.fontFamily : "Segoe UI"
                            font.pixelSize: theme ? theme.bodyFontSize : 14
                            wrapMode: Text.WordWrap
                        }
                    }
                }

                Components.SectionCard {
                    Layout.fillWidth: true
                    theme: page.theme
                    title: "Migration safeguards"
                    subtitle: "The calibration port will be introduced without risking a production recipe."

                    Repeater {
                        model: [
                            "Golden capture must be reviewed before it replaces a reference.",
                            "Recipe-specific tolerances remain entered in mm / mm / deg, then converted by the vision backend.",
                            "Baseplate orientation checks remain a separate, auditable acceptance rule.",
                            "Every calibration change will record the operator, timestamp, and prior value."
                        ]
                        delegate: RowLayout {
                            Layout.fillWidth: true
                            spacing: theme ? theme.space3 : 12
                            Rectangle {
                                width: 6
                                height: 6
                                radius: 3
                                color: theme ? theme.accent : "#20A7F5"
                            }
                            Text {
                                Layout.fillWidth: true
                                text: modelData
                                color: theme ? theme.textSecondary : "#B3C1CC"
                                font.family: theme ? theme.fontFamily : "Segoe UI"
                                font.pixelSize: theme ? theme.bodyFontSize : 14
                                wrapMode: Text.WordWrap
                            }
                        }
                    }
                }
            }

            Components.SectionCard {
                Layout.fillWidth: true
                theme: page.theme
                title: "Planned calibration flow"
                subtitle: "A deliberate sequence keeps production validation explainable and reversible."

                GridLayout {
                    Layout.fillWidth: true
                    columns: width >= 1080 ? 4 : 2
                    columnSpacing: theme ? theme.space3 : 12
                    rowSpacing: theme ? theme.space3 : 12

                    Repeater {
                        model: [
                            { "step": "01", "title": "Prepare", "detail": "Select an approved recipe and verify the station." },
                            { "step": "02", "title": "Capture", "detail": "Acquire a candidate golden image with traceable metadata." },
                            { "step": "03", "title": "Verify", "detail": "Review tolerances, geometry, and orientation before activation." },
                            { "step": "04", "title": "Activate", "detail": "Commit a versioned calibration only after approval." }
                        ]
                        delegate: Rectangle {
                            Layout.fillWidth: true
                            implicitHeight: 132
                            radius: theme ? theme.radiusSmall : 6
                            color: theme ? theme.surfaceRaised : "#18232E"
                            border.width: 1
                            border.color: theme ? theme.border : "#2A3A49"

                            ColumnLayout {
                                anchors.fill: parent
                                anchors.margins: theme ? theme.compactPanelPadding : 12
                                spacing: 4
                                Text {
                                    text: modelData.step
                                    color: theme ? theme.accent : "#20A7F5"
                                    font.family: theme ? theme.numericFontFamily : "Cascadia Mono"
                                    font.pixelSize: theme ? theme.bodyFontSize : 14
                                }
                                Text {
                                    text: modelData.title
                                    color: theme ? theme.textPrimary : "white"
                                    font.family: theme ? theme.displayFontFamily : "Segoe UI Semibold"
                                    font.pixelSize: theme ? theme.controlFontSize : 15
                                }
                                Text {
                                    Layout.fillWidth: true
                                    Layout.fillHeight: true
                                    text: modelData.detail
                                    color: theme ? theme.textSecondary : "#B3C1CC"
                                    font.family: theme ? theme.fontFamily : "Segoe UI"
                                    font.pixelSize: theme ? theme.labelFontSize : 12
                                    wrapMode: Text.WordWrap
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
