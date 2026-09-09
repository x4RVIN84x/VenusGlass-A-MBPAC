import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import "../components" as Components

Item {
    id: page

    property var theme
    readonly property int totalCount: vista ? vista.reportTotal : 0
    readonly property int passCount: vista ? vista.reportPass : 0
    readonly property int failCount: vista ? vista.reportFail : 0
    readonly property real passRate: totalCount > 0 ? (100.0 * passCount / totalCount) : 0

    function rateText() {
        return totalCount > 0 ? passRate.toFixed(1) + "%" : "—"
    }

    Flickable {
        id: scrollView
        anchors.fill: parent
        contentWidth: width
        contentHeight: reportContent.implicitHeight + (theme ? theme.pageMargin * 2 : 48)
        clip: true
        boundsBehavior: Flickable.StopAtBounds

        ScrollBar.vertical: ScrollBar {
            policy: ScrollBar.AsNeeded
            width: 10
        }

        ColumnLayout {
            id: reportContent
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
                        text: "Inspection reports"
                        color: theme ? theme.textPrimary : "white"
                        font.family: theme ? theme.displayFontFamily : "Segoe UI Semibold"
                        font.pixelSize: theme ? theme.pageTitleFontSize : 28
                    }
                    Text {
                        text: "Today's durable inspection record for this workstation"
                        color: theme ? theme.textSecondary : "#B3C1CC"
                        font.family: theme ? theme.fontFamily : "Segoe UI"
                        font.pixelSize: theme ? theme.bodyFontSize : 14
                    }
                }

                Components.VistaButton {
                    theme: page.theme
                    text: "REFRESH"
                    accentColor: theme ? theme.accent : "#20A7F5"
                    onClicked: vista.refreshReports()
                }
            }

            Rectangle {
                Layout.fillWidth: true
                implicitHeight: 42
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
                        width: 8
                        height: 8
                        radius: width / 2
                        color: theme ? theme.accent : "#20A7F5"
                    }
                    Text {
                        text: "LIVE WINDOW"
                        color: theme ? theme.textMuted : "#778896"
                        font.family: theme ? theme.displayFontFamily : "Segoe UI Semibold"
                        font.pixelSize: theme ? theme.labelFontSize : 12
                        font.letterSpacing: 0.8
                    }
                    Text {
                        text: "Today · confirmed Auto Mode results"
                        color: theme ? theme.textPrimary : "white"
                        font.family: theme ? theme.fontFamily : "Segoe UI"
                        font.pixelSize: theme ? theme.bodyFontSize : 14
                        Layout.fillWidth: true
                    }
                    Text {
                        text: totalCount + " processed"
                        color: theme ? theme.textSecondary : "#B3C1CC"
                        font.family: theme ? theme.numericFontFamily : "Cascadia Mono"
                        font.pixelSize: theme ? theme.bodyFontSize : 14
                    }
                }
            }

            GridLayout {
                Layout.fillWidth: true
                columns: width >= 1120 ? 4 : 2
                columnSpacing: theme ? theme.space4 : 16
                rowSpacing: theme ? theme.space4 : 16

                Components.MetricCard {
                    Layout.fillWidth: true
                    theme: page.theme
                    caption: "TOTAL PROCESSED"
                    value: String(page.totalCount)
                    detail: "Confirmed inspections today"
                }
                Components.MetricCard {
                    Layout.fillWidth: true
                    theme: page.theme
                    caption: "PASS"
                    value: String(page.passCount)
                    detail: "Within active recipe limits"
                    valueColor: theme ? theme.passColor : "#2BCB7F"
                }
                Components.MetricCard {
                    Layout.fillWidth: true
                    theme: page.theme
                    caption: "FAIL"
                    value: String(page.failCount)
                    detail: "Includes baseplate-not-found events"
                    valueColor: theme ? theme.failColor : "#D84B5B"
                }
                Components.MetricCard {
                    Layout.fillWidth: true
                    theme: page.theme
                    caption: "PASS RATE"
                    value: page.rateText()
                    detail: page.totalCount > 0 ? "Based on " + page.totalCount + " inspections" : "No completed inspections yet"
                    valueColor: page.totalCount > 0 && page.passRate >= 95
                                ? (theme ? theme.passColor : "#2BCB7F")
                                : (theme ? theme.textPrimary : "white")
                }
            }

            RowLayout {
                Layout.fillWidth: true
                Layout.preferredHeight: 300
                Layout.minimumHeight: 260
                spacing: theme ? theme.space4 : 16

                Components.SectionCard {
                    Layout.fillWidth: true
                    Layout.preferredWidth: 1
                    Layout.fillHeight: true
                    theme: page.theme
                    title: "Pass / fail split"
                    subtitle: "Today's confirmed result distribution"

                    Item {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        Layout.minimumHeight: 196

                        Canvas {
                            id: resultDonut
                            anchors.centerIn: parent
                            width: Math.min(parent.width, parent.height) - 20
                            height: width
                            antialiasing: true

                            onPaint: {
                                var context = getContext("2d")
                                context.clearRect(0, 0, width, height)
                                var side = Math.min(width, height)
                                var centre = side / 2
                                var radius = Math.max(20, side / 2 - 19)
                                var lineWidth = Math.max(16, Math.min(26, side * 0.13))
                                var start = -Math.PI / 2
                                var total = Math.max(0, page.totalCount)
                                var passAngle = total > 0 ? (Math.PI * 2 * page.passCount / total) : 0
                                var failAngle = total > 0 ? (Math.PI * 2 * page.failCount / total) : 0

                                context.lineWidth = lineWidth
                                context.lineCap = "round"
                                context.strokeStyle = theme ? theme.surfaceOverlay : "#202D39"
                                context.beginPath()
                                context.arc(centre, centre, radius, start, start + Math.PI * 2, false)
                                context.stroke()

                                if (passAngle > 0) {
                                    context.strokeStyle = theme ? theme.passColor : "#2BCB7F"
                                    context.beginPath()
                                    context.arc(centre, centre, radius, start, start + passAngle, false)
                                    context.stroke()
                                }
                                if (failAngle > 0) {
                                    context.strokeStyle = theme ? theme.failColor : "#D84B5B"
                                    context.beginPath()
                                    context.arc(centre, centre, radius, start + passAngle, start + passAngle + failAngle, false)
                                    context.stroke()
                                }
                            }

                            Component.onCompleted: requestPaint()
                        }

                        Connections {
                            target: vista
                            function onReportSummaryChanged() {
                                resultDonut.requestPaint()
                            }
                        }

                        Column {
                            anchors.centerIn: parent
                            spacing: 2
                            Text {
                                width: parent.width
                                text: String(page.totalCount)
                                horizontalAlignment: Text.AlignHCenter
                                color: theme ? theme.textPrimary : "white"
                                font.family: theme ? theme.numericFontFamily : "Cascadia Mono"
                                font.pixelSize: 38
                                font.bold: true
                            }
                            Text {
                                width: parent.width
                                text: "PROCESSED"
                                horizontalAlignment: Text.AlignHCenter
                                color: theme ? theme.textMuted : "#778896"
                                font.family: theme ? theme.displayFontFamily : "Segoe UI Semibold"
                                font.pixelSize: theme ? theme.labelFontSize : 12
                                font.letterSpacing: 1.0
                            }
                        }
                    }
                }

                Components.SectionCard {
                    Layout.fillWidth: true
                    Layout.preferredWidth: 1.45
                    Layout.fillHeight: true
                    theme: page.theme
                    title: "Report integrity"
                    subtitle: "What is included in this view"

                    Repeater {
                        model: [
                            { "color": theme ? theme.passColor : "#2BCB7F", "title": "PASS", "detail": "A completed inspection within its active recipe limits." },
                            { "color": theme ? theme.failColor : "#D84B5B", "title": "FAIL", "detail": "A confirmed out-of-tolerance inspection." },
                            { "color": theme ? theme.baseplateMissingColor : "#80461B", "title": "BASEPLATE NOT FOUND", "detail": "Recorded as a FAIL only after fitted glass-notch presence is confirmed." }
                        ]

                        delegate: RowLayout {
                            Layout.fillWidth: true
                            spacing: theme ? theme.space3 : 12

                            Rectangle {
                                width: 10
                                height: 10
                                radius: width / 2
                                color: modelData.color
                            }
                            ColumnLayout {
                                Layout.fillWidth: true
                                spacing: 2
                                Text {
                                    text: modelData.title
                                    color: theme ? theme.textPrimary : "white"
                                    font.family: theme ? theme.displayFontFamily : "Segoe UI Semibold"
                                    font.pixelSize: theme ? theme.bodyFontSize : 14
                                }
                                Text {
                                    Layout.fillWidth: true
                                    text: modelData.detail
                                    color: theme ? theme.textSecondary : "#B3C1CC"
                                    font.family: theme ? theme.fontFamily : "Segoe UI"
                                    font.pixelSize: theme ? theme.labelFontSize : 12
                                    wrapMode: Text.WordWrap
                                }
                            }
                        }
                    }

                    Rectangle {
                        Layout.fillWidth: true
                        implicitHeight: 1
                        color: theme ? theme.divider : "#20303D"
                    }
                    Text {
                        Layout.fillWidth: true
                        text: "Local data store: " + (vista ? vista.databasePath : "Unavailable")
                        color: theme ? theme.textMuted : "#778896"
                        font.family: theme ? theme.numericFontFamily : "Cascadia Mono"
                        font.pixelSize: theme ? theme.labelFontSize : 12
                        elide: Text.ElideMiddle
                    }
                }
            }

            Components.SectionCard {
                Layout.fillWidth: true
                theme: page.theme
                title: "Reporting roadmap"
                subtitle: "This first QML view is intentionally limited to a live daily summary. It does not simulate historical filters or exports."

                RowLayout {
                    Layout.fillWidth: true
                    spacing: theme ? theme.space4 : 16

                    Text {
                        Layout.fillWidth: true
                        text: "Next port: selectable ranges, failure-cause drilldown, shift-aware timeline, and controlled Excel/PDF export."
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
                        font.letterSpacing: 1.0
                    }
                }
            }
        }
    }
}
