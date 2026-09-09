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
        contentHeight: optionsContent.implicitHeight + (theme ? theme.pageMargin * 2 : 48)
        clip: true
        boundsBehavior: Flickable.StopAtBounds

        ScrollBar.vertical: ScrollBar {
            policy: ScrollBar.AsNeeded
            width: 10
        }

        ColumnLayout {
            id: optionsContent
            x: theme ? theme.pageMargin : 24
            y: theme ? theme.pageMargin : 24
            width: scrollView.width - (theme ? theme.pageMargin * 2 : 48)
            spacing: theme ? theme.space5 : 20

            ColumnLayout {
                Layout.fillWidth: true
                spacing: 4
                Text {
                    text: "Options & commissioning"
                    color: theme ? theme.textPrimary : "white"
                    font.family: theme ? theme.displayFontFamily : "Segoe UI Semibold"
                    font.pixelSize: theme ? theme.pageTitleFontSize : 28
                }
                Text {
                    text: "Workstation identity, data location, and safe integration readiness"
                    color: theme ? theme.textSecondary : "#B3C1CC"
                    font.family: theme ? theme.fontFamily : "Segoe UI"
                    font.pixelSize: theme ? theme.bodyFontSize : 14
                }
            }

            GridLayout {
                Layout.fillWidth: true
                columns: width >= 1020 ? 2 : 1
                columnSpacing: theme ? theme.space4 : 16
                rowSpacing: theme ? theme.space4 : 16

                Components.SectionCard {
                    Layout.fillWidth: true
                    theme: page.theme
                    title: "Application profile"
                    subtitle: "Installed workstation identity"

                    GridLayout {
                        Layout.fillWidth: true
                        columns: 2
                        columnSpacing: theme ? theme.space4 : 16
                        rowSpacing: theme ? theme.space3 : 12

                        Text {
                            text: "PRODUCT"
                            color: theme ? theme.textMuted : "#778896"
                            font.family: theme ? theme.displayFontFamily : "Segoe UI Semibold"
                            font.pixelSize: theme ? theme.labelFontSize : 12
                        }
                        Text {
                            Layout.fillWidth: true
                            text: vista ? vista.productName : "VG VISTA"
                            color: theme ? theme.textPrimary : "white"
                            font.family: theme ? theme.displayFontFamily : "Segoe UI Semibold"
                            font.pixelSize: theme ? theme.bodyFontSize : 14
                            elide: Text.ElideRight
                        }
                        Text {
                            text: "VERSION"
                            color: theme ? theme.textMuted : "#778896"
                            font.family: theme ? theme.displayFontFamily : "Segoe UI Semibold"
                            font.pixelSize: theme ? theme.labelFontSize : 12
                        }
                        Text {
                            text: vista ? vista.productVersion : "—"
                            color: theme ? theme.accentHover : "#54BDF7"
                            font.family: theme ? theme.numericFontFamily : "Cascadia Mono"
                            font.pixelSize: theme ? theme.bodyFontSize : 14
                        }
                        Text {
                            text: "RELEASE"
                            color: theme ? theme.textMuted : "#778896"
                            font.family: theme ? theme.displayFontFamily : "Segoe UI Semibold"
                            font.pixelSize: theme ? theme.labelFontSize : 12
                        }
                        Text {
                            Layout.fillWidth: true
                            text: vista ? vista.releaseDate : "—"
                            color: theme ? theme.textSecondary : "#B3C1CC"
                            font.family: theme ? theme.fontFamily : "Segoe UI"
                            font.pixelSize: theme ? theme.bodyFontSize : 14
                        }
                        Text {
                            text: "AUTHOR"
                            color: theme ? theme.textMuted : "#778896"
                            font.family: theme ? theme.displayFontFamily : "Segoe UI Semibold"
                            font.pixelSize: theme ? theme.labelFontSize : 12
                        }
                        Text {
                            Layout.fillWidth: true
                            text: vista ? vista.authorName : "—"
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
                    title: "Local data & audit"
                    subtitle: "Production history remains outside the installed application files."

                    Text {
                        text: "INSPECTION DATABASE"
                        color: theme ? theme.textMuted : "#778896"
                        font.family: theme ? theme.displayFontFamily : "Segoe UI Semibold"
                        font.pixelSize: theme ? theme.labelFontSize : 12
                        font.letterSpacing: 0.8
                    }
                    Text {
                        Layout.fillWidth: true
                        text: vista ? vista.databasePath : "Unavailable"
                        color: theme ? theme.accentHover : "#54BDF7"
                        font.family: theme ? theme.numericFontFamily : "Cascadia Mono"
                        font.pixelSize: theme ? theme.labelFontSize : 12
                        wrapMode: Text.WrapAnywhere
                    }
                    Rectangle {
                        Layout.fillWidth: true
                        implicitHeight: 1
                        color: theme ? theme.divider : "#20303D"
                    }
                    Text {
                        Layout.fillWidth: true
                        text: "This location is user-writable so reports survive application upgrades and the installer never needs to modify production data."
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
                title: "PLC & HMI integration"
                subtitle: "Commissioning is staged. This screen exposes status only; it cannot connect a PLC, write tags, release a conveyor, or override machine safety."

                Rectangle {
                    Layout.fillWidth: true
                    implicitHeight: 62
                    radius: theme ? theme.radiusSmall : 6
                    color: theme ? theme.surfaceRaised : "#18232E"
                    border.width: 1
                    border.color: theme ? theme.border : "#2A3A49"

                    RowLayout {
                        anchors.fill: parent
                        anchors.leftMargin: theme ? theme.compactPanelPadding : 12
                        anchors.rightMargin: theme ? theme.compactPanelPadding : 12
                        spacing: theme ? theme.space3 : 12
                        Rectangle {
                            width: 10
                            height: 10
                            radius: width / 2
                            color: theme ? theme.offlineColor : "#70808F"
                        }
                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 2
                            Text {
                                text: "COMMISSIONING STATUS"
                                color: theme ? theme.textMuted : "#778896"
                                font.family: theme ? theme.displayFontFamily : "Segoe UI Semibold"
                                font.pixelSize: theme ? theme.labelFontSize : 12
                                font.letterSpacing: 0.8
                            }
                            Text {
                                Layout.fillWidth: true
                                text: vista ? vista.plcSummary : "PLC service unavailable"
                                color: theme ? theme.textPrimary : "white"
                                font.family: theme ? theme.fontFamily : "Segoe UI"
                                font.pixelSize: theme ? theme.bodyFontSize : 14
                                wrapMode: Text.WordWrap
                            }
                        }
                    }
                }

                GridLayout {
                    Layout.fillWidth: true
                    columns: width >= 1050 ? 3 : 1
                    columnSpacing: theme ? theme.space3 : 12
                    rowSpacing: theme ? theme.space3 : 12

                    Repeater {
                        model: [
                            { "title": "VG VISTA publishes", "detail": "Inspection state, PASS / FAIL / baseplate-not-found result, confidence, measurements, cause, and a sequence number." },
                            { "title": "PLC owns", "detail": "Conveyor release, physical interlocks, E-stop logic, safety circuits, and all machine-motion permissions." },
                            { "title": "Before enablement", "detail": "An approved tag map, acknowledgement handshake, network review, timeout behavior, and engineer acceptance test are required." }
                        ]
                        delegate: Rectangle {
                            Layout.fillWidth: true
                            implicitHeight: 142
                            radius: theme ? theme.radiusSmall : 6
                            color: theme ? theme.surfaceRaised : "#18232E"
                            border.width: 1
                            border.color: theme ? theme.border : "#2A3A49"

                            ColumnLayout {
                                anchors.fill: parent
                                anchors.margins: theme ? theme.compactPanelPadding : 12
                                spacing: 6
                                Text {
                                    text: modelData.title
                                    color: theme ? theme.textPrimary : "white"
                                    font.family: theme ? theme.displayFontFamily : "Segoe UI Semibold"
                                    font.pixelSize: theme ? theme.bodyFontSize : 14
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

            Components.SectionCard {
                Layout.fillWidth: true
                theme: page.theme
                title: "Interface preferences"
                subtitle: "Accessibility, language, Jalali calendar, alarm sound, and navigation ordering will return here after their Qt settings service is ported."

                RowLayout {
                    Layout.fillWidth: true
                    spacing: theme ? theme.space3 : 12
                    Rectangle {
                        width: 8
                        height: 8
                        radius: width / 2
                        color: theme ? theme.warningColor : "#F0B34A"
                    }
                    Text {
                        Layout.fillWidth: true
                        text: "These preferences are intentionally not shown as working controls until settings are persisted and tested for the QML runtime."
                        color: theme ? theme.textSecondary : "#B3C1CC"
                        font.family: theme ? theme.fontFamily : "Segoe UI"
                        font.pixelSize: theme ? theme.bodyFontSize : 14
                        wrapMode: Text.WordWrap
                    }
                    Text {
                        text: "STAGED"
                        color: theme ? theme.warningColor : "#F0B34A"
                        font.family: theme ? theme.displayFontFamily : "Segoe UI Semibold"
                        font.pixelSize: theme ? theme.labelFontSize : 12
                        font.letterSpacing: 0.9
                    }
                }
            }
        }
    }
}
