import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import "components" as Components
import "pages" as Pages

ApplicationWindow {
    id: window

    width: 1600
    height: 960
    minimumWidth: 1120
    minimumHeight: 720
    visible: true
    title: vista.productName + " · " + vista.productSubtitle
    color: theme.appBackground

    Components.VistaTheme {
        id: theme
    }

    property int currentPage: 0

    Rectangle {
        anchors.fill: parent
        color: theme.appBackground
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: theme.topBarHeight
            color: theme.surface
            border.width: 1
            border.color: theme.border

            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: theme.space6
                anchors.rightMargin: theme.space6
                spacing: theme.space4

                Image {
                    source: "assets/venus-glass-logo.png"
                    fillMode: Image.PreserveAspectFit
                    smooth: true
                    Layout.preferredWidth: 42
                    Layout.preferredHeight: 42
                }

                ColumnLayout {
                    spacing: 1
                    Layout.preferredWidth: 240

                    Text {
                        text: vista.productName
                        color: theme.textPrimary
                        font.family: theme.displayFontFamily
                        font.pixelSize: 21
                        font.letterSpacing: 1.2
                    }
                    Text {
                        text: vista.productSubtitle.toUpperCase()
                        color: theme.textMuted
                        font.family: theme.displayFontFamily
                        font.pixelSize: 10
                        font.letterSpacing: 1.5
                    }
                }

                Item { Layout.fillWidth: true }

                ColumnLayout {
                    spacing: 3
                    Layout.preferredWidth: 300

                    Text {
                        text: "ACTIVE RECIPE"
                        color: theme.textMuted
                        font.family: theme.displayFontFamily
                        font.pixelSize: theme.labelFontSize
                        font.letterSpacing: 1.0
                    }
                    ComboBox {
                        id: recipeSelector
                        Layout.fillWidth: true
                        implicitHeight: theme.compactControlHeight
                        model: vista.recipes
                        currentIndex: Math.max(0, vista.recipes.indexOf(vista.currentRecipe))
                        enabled: !vista.running && vista.recipes.length > 0
                        font.family: theme.fontFamily
                        font.pixelSize: theme.bodyFontSize

                        onActivated: vista.selectRecipe(currentText)

                        contentItem: Text {
                            leftPadding: theme.space3
                            rightPadding: theme.space8
                            text: recipeSelector.displayText
                            color: recipeSelector.enabled ? theme.textPrimary : theme.textMuted
                            font: recipeSelector.font
                            verticalAlignment: Text.AlignVCenter
                            elide: Text.ElideRight
                        }
                        background: Rectangle {
                            color: theme.inputSurface
                            radius: theme.radiusSmall
                            border.width: 1
                            border.color: recipeSelector.activeFocus ? theme.focusRing : theme.border
                        }
                        indicator: Text {
                            x: recipeSelector.width - width - theme.space3
                            y: recipeSelector.topPadding + (recipeSelector.availableHeight - height) / 2
                            text: "⌄"
                            color: theme.textSecondary
                            font.pixelSize: 16
                        }
                        popup: Popup {
                            y: recipeSelector.height + 4
                            width: recipeSelector.width
                            implicitHeight: Math.min(contentItem.implicitHeight + 12, 260)
                            padding: 6
                            contentItem: ListView {
                                clip: true
                                implicitHeight: contentHeight
                                model: recipeSelector.popup.visible ? recipeSelector.delegateModel : null
                                currentIndex: recipeSelector.highlightedIndex
                                ScrollIndicator.vertical: ScrollIndicator { }
                            }
                            background: Rectangle {
                                color: theme.surfaceRaised
                                radius: theme.radiusSmall
                                border.color: theme.borderStrong
                            }
                        }
                    }
                }

                Rectangle {
                    Layout.preferredWidth: 174
                    Layout.preferredHeight: theme.controlHeight
                    radius: theme.radiusMedium
                    color: Qt.rgba(theme.stateColor(vista.statusCode).r, theme.stateColor(vista.statusCode).g, theme.stateColor(vista.statusCode).b, 0.16)
                    border.width: 1
                    border.color: theme.stateColor(vista.statusCode)

                    RowLayout {
                        anchors.fill: parent
                        anchors.margins: theme.space3
                        spacing: theme.space2
                        Rectangle {
                            Layout.preferredWidth: 9
                            Layout.preferredHeight: 9
                            radius: 5
                            color: theme.stateColor(vista.statusCode)
                        }
                        Text {
                            Layout.fillWidth: true
                            text: String(vista.statusCode).replace(/_/g, " ")
                            color: theme.textPrimary
                            font.family: theme.displayFontFamily
                            font.pixelSize: theme.labelFontSize
                            font.letterSpacing: 0.7
                            elide: Text.ElideRight
                        }
                    }
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 0

            Rectangle {
                Layout.fillHeight: true
                Layout.preferredWidth: theme.navRailWidth
                color: theme.workspaceBackground
                border.width: 1
                border.color: theme.border

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: theme.space3
                    spacing: theme.space2

                    Text {
                        text: "WORKSPACE"
                        color: theme.textMuted
                        font.family: theme.displayFontFamily
                        font.pixelSize: theme.labelFontSize
                        font.letterSpacing: 1.3
                        Layout.leftMargin: theme.space3
                        Layout.topMargin: theme.space2
                    }

                    Components.NavItem {
                        Layout.fillWidth: true
                        theme: theme
                        text: "Auto Inspection"
                        glyph: "◉"
                        selected: window.currentPage === 0
                        onClicked: window.currentPage = 0
                    }
                    Components.NavItem {
                        Layout.fillWidth: true
                        theme: theme
                        text: "Calibration"
                        glyph: "◇"
                        selected: window.currentPage === 1
                        onClicked: window.currentPage = 1
                    }
                    Components.NavItem {
                        Layout.fillWidth: true
                        theme: theme
                        text: "Reports"
                        glyph: "▥"
                        selected: window.currentPage === 2
                        onClicked: window.currentPage = 2
                    }
                    Components.NavItem {
                        Layout.fillWidth: true
                        theme: theme
                        text: "Manual Test"
                        glyph: "◌"
                        selected: window.currentPage === 3
                        onClicked: window.currentPage = 3
                    }
                    Components.NavItem {
                        Layout.fillWidth: true
                        theme: theme
                        text: "Options"
                        glyph: "⚙"
                        selected: window.currentPage === 4
                        onClicked: window.currentPage = 4
                    }

                    Item { Layout.fillHeight: true }

                    Rectangle {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 84
                        color: theme.surface
                        radius: theme.radiusMedium
                        border.width: 1
                        border.color: theme.border

                        ColumnLayout {
                            anchors.fill: parent
                            anchors.margins: theme.space3
                            spacing: 4
                            Text {
                                text: "SYSTEM STATUS"
                                color: theme.textMuted
                                font.family: theme.displayFontFamily
                                font.pixelSize: 10
                                font.letterSpacing: 1.0
                            }
                            Text {
                                text: vista.plcSummary
                                color: theme.textSecondary
                                font.family: theme.fontFamily
                                font.pixelSize: 11
                                wrapMode: Text.WordWrap
                                Layout.fillWidth: true
                                maximumLineCount: 3
                                elide: Text.ElideRight
                            }
                        }
                    }

                    Text {
                        Layout.leftMargin: theme.space3
                        Layout.bottomMargin: theme.space2
                        text: "VG VISTA  " + vista.productVersion
                        color: theme.textMuted
                        font.family: theme.numericFontFamily
                        font.pixelSize: 10
                    }
                }
            }

            StackLayout {
                id: pageStack
                Layout.fillWidth: true
                Layout.fillHeight: true
                currentIndex: window.currentPage

                Pages.AutoPage { theme: theme }
                Pages.CalibrationPage { theme: theme }
                Pages.ReportsPage { theme: theme }
                Pages.ManualTestPage { theme: theme }
                Pages.OptionsPage { theme: theme }
            }
        }
    }

    Components.VistaSplash {
        id: splash
        z: 100
        theme: theme
    }
}
