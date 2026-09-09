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
        contentHeight: manualContent.implicitHeight + (theme ? theme.pageMargin * 2 : 48)
        clip: true
        boundsBehavior: Flickable.StopAtBounds

        ScrollBar.vertical: ScrollBar {
            policy: ScrollBar.AsNeeded
            width: 10
        }

        ColumnLayout {
            id: manualContent
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
                        text: "Manual inspection"
                        color: theme ? theme.textPrimary : "white"
                        font.family: theme ? theme.displayFontFamily : "Segoe UI Semibold"
                        font.pixelSize: theme ? theme.pageTitleFontSize : 28
                    }
                    Text {
                        text: "Single-frame diagnostic workflow outside the production cycle"
                        color: theme ? theme.textSecondary : "#B3C1CC"
                        font.family: theme ? theme.fontFamily : "Segoe UI"
                        font.pixelSize: theme ? theme.bodyFontSize : 14
                    }
                }

                Rectangle {
                    implicitWidth: 150
                    implicitHeight: 34
                    radius: theme ? theme.radiusSmall : 6
                    color: theme ? theme.surfaceRaised : "#18232E"
                    border.width: 1
                    border.color: theme ? theme.warningColor : "#F0B34A"
                    Text {
                        anchors.centerIn: parent
                        text: "PORT IN PROGRESS"
                        color: theme ? theme.warningColor : "#F0B34A"
                        font.family: theme ? theme.displayFontFamily : "Segoe UI Semibold"
                        font.pixelSize: theme ? theme.labelFontSize : 12
                        font.letterSpacing: 0.8
                    }
                }
            }

            Components.SectionCard {
                Layout.fillWidth: true
                theme: page.theme
                title: "Production safety boundary"
                subtitle: "Manual testing is intentionally unavailable in the first QML rollout. This page does not take a camera frame, alter the current inspection cycle, or create a report."

                RowLayout {
                    Layout.fillWidth: true
                    spacing: theme ? theme.space4 : 16

                    Rectangle {
                        Layout.preferredWidth: 52
                        Layout.preferredHeight: 52
                        radius: 26
                        color: theme ? theme.surfaceRaised : "#18232E"
                        border.width: 1
                        border.color: theme ? theme.warningColor : "#F0B34A"
                        Text {
                            anchors.centerIn: parent
                            text: "i"
                            color: theme ? theme.warningColor : "#F0B34A"
                            font.family: theme ? theme.displayFontFamily : "Segoe UI Semibold"
                            font.pixelSize: 25
                        }
                    }
                    ColumnLayout {
                        Layout.fillWidth: true
                        spacing: 3
                        Text {
                            text: "Auto Mode remains the only operational camera owner"
                            color: theme ? theme.textPrimary : "white"
                            font.family: theme ? theme.displayFontFamily : "Segoe UI Semibold"
                            font.pixelSize: theme ? theme.controlFontSize : 15
                        }
                        Text {
                            Layout.fillWidth: true
                            text: "This avoids two concurrent camera sessions or an operator accidentally mixing a diagnostic run with production records."
                            color: theme ? theme.textSecondary : "#B3C1CC"
                            font.family: theme ? theme.fontFamily : "Segoe UI"
                            font.pixelSize: theme ? theme.bodyFontSize : 14
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
                    title: "Diagnostic context"
                    subtitle: "Read-only status from the current QML inspection runtime"

                    GridLayout {
                        Layout.fillWidth: true
                        columns: 2
                        columnSpacing: theme ? theme.space4 : 16
                        rowSpacing: theme ? theme.space3 : 12

                        Text {
                            text: "ACTIVE RECIPE"
                            color: theme ? theme.textMuted : "#778896"
                            font.family: theme ? theme.displayFontFamily : "Segoe UI Semibold"
                            font.pixelSize: theme ? theme.labelFontSize : 12
                        }
                        Text {
                            Layout.fillWidth: true
                            text: vista && vista.currentRecipe.length > 0 ? vista.currentRecipe : "No valid recipe loaded"
                            color: theme ? theme.accentHover : "#54BDF7"
                            font.family: theme ? theme.numericFontFamily : "Cascadia Mono"
                            font.pixelSize: theme ? theme.bodyFontSize : 14
                            elide: Text.ElideRight
                        }
                        Text {
                            text: "AUTO MODE"
                            color: theme ? theme.textMuted : "#778896"
                            font.family: theme ? theme.displayFontFamily : "Segoe UI Semibold"
                            font.pixelSize: theme ? theme.labelFontSize : 12
                        }
                        Text {
                            text: vista && vista.running ? "Running" : "Stopped"
                            color: vista && vista.running ? (theme ? theme.passColor : "#2BCB7F") : (theme ? theme.textSecondary : "#B3C1CC")
                            font.family: theme ? theme.displayFontFamily : "Segoe UI Semibold"
                            font.pixelSize: theme ? theme.bodyFontSize : 14
                        }
                        Text {
                            text: "LIVE STATUS"
                            color: theme ? theme.textMuted : "#778896"
                            font.family: theme ? theme.displayFontFamily : "Segoe UI Semibold"
                            font.pixelSize: theme ? theme.labelFontSize : 12
                        }
                        Text {
                            Layout.fillWidth: true
                            text: vista ? vista.statusText : "Unavailable"
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
                    title: "Planned manual workflow"
                    subtitle: "The port will retain the existing diagnostic intent while making each action explicit."

                    Repeater {
                        model: [
                            { "number": "01", "text": "Capture a single frame only after the operator enters diagnostics." },
                            { "number": "02", "text": "Choose raw, processed, or overlay view without changing Auto Mode settings." },
                            { "number": "03", "text": "Run the selected recipe once and show all measurements with units." },
                            { "number": "04", "text": "Keep diagnostic output separate from production reports unless explicitly approved." }
                        ]
                        delegate: RowLayout {
                            Layout.fillWidth: true
                            spacing: theme ? theme.space3 : 12
                            Text {
                                text: modelData.number
                                color: theme ? theme.accent : "#20A7F5"
                                font.family: theme ? theme.numericFontFamily : "Cascadia Mono"
                                font.pixelSize: theme ? theme.bodyFontSize : 14
                            }
                            Text {
                                Layout.fillWidth: true
                                text: modelData.text
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
                title: "Why this is staged"
                subtitle: "A manual tool can be valuable for technicians, but it must not introduce an ambiguous camera state or a report that looks like a production inspection."

                Text {
                    Layout.fillWidth: true
                    text: "The implementation will add technician access, camera ownership checks, an explicit one-shot action, and a clearly labelled diagnostic result before it is enabled."
                    color: theme ? theme.textSecondary : "#B3C1CC"
                    font.family: theme ? theme.fontFamily : "Segoe UI"
                    font.pixelSize: theme ? theme.bodyFontSize : 14
                    wrapMode: Text.WordWrap
                }
            }
        }
    }
}
