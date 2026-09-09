"""QML-facing runtime for the standalone VG VISTA application.

The controller deliberately owns the camera and inspection lifecycle outside
QML.  The declarative UI receives properties and frames only; it cannot drive
the conveyor or access PLC transport code directly.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from threading import RLock
import time
from typing import Any, Optional

import cv2

from PySide6.QtCore import QObject, Property, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QImage
from PySide6.QtQuick import QQuickImageProvider

from hmi_app.core.engine import QCPreviewEngine
from hmi_app.core.recipe_manager import RecipeManager
from hmi_app.core.report_store import ReportStore
from hmi_app.io.camera import OpenCVCamera
from hmi_app.plc import (
    InspectionOutcome,
    InspectionPlcService,
    InspectionTelemetry,
    LiveInspectionState,
    build_decision_from_output,
    build_live_telemetry,
)
from hmi_app.vista_metadata import (
    AUTHOR,
    PRODUCT_NAME,
    PRODUCT_SUBTITLE,
    PRODUCT_VERSION,
    RELEASE_DATE,
    workstation_data_directory,
)


class VistaFrameProvider(QQuickImageProvider):
    """Thread-safe QML image provider for the latest processed camera frame."""

    def __init__(self) -> None:
        super().__init__(QQuickImageProvider.ImageType.Image)
        self._lock = RLock()
        self._image = QImage(1280, 720, QImage.Format.Format_RGB32)
        self._image.fill(QColor("#101820"))

    def set_bgr_frame(self, frame: Any) -> None:
        if frame is None:
            return
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channels = rgb.shape
            image = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()
        except Exception:
            return
        with self._lock:
            self._image = image

    def requestImage(self, _image_id, size, requested_size):  # noqa: N802 - Qt callback spelling
        with self._lock:
            image = self._image.copy()
        if requested_size.isValid() and not requested_size.isEmpty():
            image = image.scaled(
                requested_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        if size is not None:
            size.setWidth(image.width())
            size.setHeight(image.height())
        return image


class VistaController(QObject):
    """A production-safe initial QML bridge over the existing vision backend."""

    recipesChanged = Signal()
    inspectionChanged = Signal()
    imageVersionChanged = Signal()
    reportSummaryChanged = Signal()

    def __init__(self, repo_root: str | Path, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._repo_root = Path(repo_root)
        self._recipe_manager = RecipeManager(recipes_root=str(self._repo_root / "recipes"))
        self._engine = QCPreviewEngine()
        self._report_store = ReportStore(workstation_data_directory() / "inspection_history.sqlite3")
        self._plc_service = InspectionPlcService.disabled()
        self._plc_service.start()
        self._camera: Optional[OpenCVCamera] = None
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._process_frame)
        self._frame_provider = VistaFrameProvider()

        self._recipes = self._recipe_manager.list_recipes()
        self._current_recipe = ""
        self._running = False
        self._status_code = "READY"
        self._status_text = "Select a recipe and start inspection"
        self._confidence_percent = 0
        self._delta_x = "—"
        self._delta_y = "—"
        self._angle = "—"
        self._baseplate_text = "Waiting for fitted glass notch"
        self._cycle_text = "READY FOR NEXT GLASS"
        self._last_saved = "No inspection has been saved in this session"
        self._camera_message = "Camera idle"
        self._image_version = 0
        self._report_total = 0
        self._report_pass = 0
        self._report_fail = 0

        self._decision_confirmation_s = 1.0
        self._track_alarm_after_s = 5.0
        self._rearm_after_s = 5.0
        self._confidence_scores = {"PASS": 0.0, "FAIL": 0.0, "BASEPLATE": 0.0}
        self._confidence_last_updated: Optional[float] = None
        self._report_terminal_recorded = False
        self._clear_since: Optional[float] = None
        self._baseplate_missing_since: Optional[float] = None
        self._baseplate_missing_elapsed = 0.0

        if self._recipes:
            self._load_recipe(self._recipes[0])
        else:
            self._status_code = "SETUP_REQUIRED"
            self._status_text = "No loadable recipe was found"
        self.refresh_reports()

    @property
    def image_provider(self) -> VistaFrameProvider:
        return self._frame_provider

    @Property(str, constant=True)
    def productName(self) -> str:  # noqa: N802 - QML-facing API
        return PRODUCT_NAME

    @Property(str, constant=True)
    def productSubtitle(self) -> str:  # noqa: N802
        return PRODUCT_SUBTITLE

    @Property(str, constant=True)
    def productVersion(self) -> str:  # noqa: N802
        return PRODUCT_VERSION

    @Property(str, constant=True)
    def releaseDate(self) -> str:  # noqa: N802
        return RELEASE_DATE

    @Property(str, constant=True)
    def authorName(self) -> str:  # noqa: N802
        return AUTHOR

    @Property("QStringList", notify=recipesChanged)
    def recipes(self) -> list[str]:
        return list(self._recipes)

    @Property(str, notify=inspectionChanged)
    def currentRecipe(self) -> str:  # noqa: N802
        return self._current_recipe

    @Property(bool, notify=inspectionChanged)
    def running(self) -> bool:
        return self._running

    @Property(str, notify=inspectionChanged)
    def statusCode(self) -> str:  # noqa: N802
        return self._status_code

    @Property(str, notify=inspectionChanged)
    def statusText(self) -> str:  # noqa: N802
        return self._status_text

    @Property(int, notify=inspectionChanged)
    def confidencePercent(self) -> int:  # noqa: N802
        return self._confidence_percent

    @Property(str, notify=inspectionChanged)
    def deltaX(self) -> str:  # noqa: N802
        return self._delta_x

    @Property(str, notify=inspectionChanged)
    def deltaY(self) -> str:  # noqa: N802
        return self._delta_y

    @Property(str, notify=inspectionChanged)
    def angle(self) -> str:
        return self._angle

    @Property(str, notify=inspectionChanged)
    def baseplateText(self) -> str:  # noqa: N802
        return self._baseplate_text

    @Property(str, notify=inspectionChanged)
    def cycleText(self) -> str:  # noqa: N802
        return self._cycle_text

    @Property(str, notify=inspectionChanged)
    def lastSaved(self) -> str:  # noqa: N802
        return self._last_saved

    @Property(str, notify=inspectionChanged)
    def cameraMessage(self) -> str:  # noqa: N802
        return self._camera_message

    @Property(int, notify=imageVersionChanged)
    def imageVersion(self) -> int:  # noqa: N802
        return self._image_version

    @Property(int, notify=reportSummaryChanged)
    def reportTotal(self) -> int:  # noqa: N802
        return self._report_total

    @Property(int, notify=reportSummaryChanged)
    def reportPass(self) -> int:  # noqa: N802
        return self._report_pass

    @Property(int, notify=reportSummaryChanged)
    def reportFail(self) -> int:  # noqa: N802
        return self._report_fail

    @Property(str, notify=inspectionChanged)
    def plcSummary(self) -> str:  # noqa: N802
        status = self._plc_service.status()
        if status.connection_state == 0:
            return "PLC handoff staged · configuration locked until commissioning"
        if status.awaiting_acknowledgement:
            return f"PLC acknowledgement pending · sequence {status.pending_sequence}"
        return status.connection_state.name.replace("_", " ")

    @Property(str, constant=True)
    def databasePath(self) -> str:  # noqa: N802
        return str(self._report_store.db_path)

    @Slot(str)
    def selectRecipe(self, recipe_name: str) -> None:  # noqa: N802
        if self._running or not recipe_name or recipe_name == self._current_recipe:
            return
        self._load_recipe(str(recipe_name))

    @Slot()
    def startInspection(self) -> None:  # noqa: N802
        if self._running:
            return
        if self._engine.recipe is None:
            self._status_code = "SETUP_REQUIRED"
            self._status_text = "Load a valid recipe before starting"
            self.inspectionChanged.emit()
            return
        if not self._ensure_camera():
            return

        self._running = True
        self._status_code = "TRACKING"
        self._status_text = "Waiting for glass detection"
        self._cycle_text = "READY FOR NEXT GLASS"
        self._reset_cycle_state()
        self._timer.start()
        self.inspectionChanged.emit()

    @Slot()
    def stopInspection(self) -> None:  # noqa: N802
        if not self._running and self._camera is None:
            return
        self._running = False
        self._timer.stop()
        if self._camera is not None:
            self._camera.release()
            self._camera = None
        self._status_code = "READY"
        self._status_text = "Inspection stopped"
        self._cycle_text = "READY FOR NEXT GLASS"
        self._camera_message = "Camera idle"
        self.inspectionChanged.emit()

    @Slot()
    def refreshReports(self) -> None:  # noqa: N802
        self.refresh_reports()

    @Slot()
    def shutdown(self) -> None:
        self.stopInspection()
        self._plc_service.stop()

    def refresh_reports(self) -> None:
        now = datetime.now()
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        summary = self._report_store.summary(start=start, end=start + timedelta(days=1))
        self._report_total = int(summary.get("total", 0))
        self._report_pass = int(summary.get("pass", 0))
        self._report_fail = int(summary.get("fail", 0))
        self.reportSummaryChanged.emit()

    def _load_recipe(self, recipe_name: str) -> None:
        try:
            recipe = self._recipe_manager.load(recipe_name)
            self._engine.set_recipe(recipe)
            self._current_recipe = recipe.name
            self._status_code = "READY"
            self._status_text = "Recipe loaded · ready for inspection"
        except Exception as exc:
            self._engine.recipe = None
            self._current_recipe = ""
            self._status_code = "SETUP_REQUIRED"
            self._status_text = f"Recipe load failed: {exc}"
        self.inspectionChanged.emit()

    def _ensure_camera(self) -> bool:
        if self._camera is not None:
            return True
        try:
            self._camera = OpenCVCamera(index=0, width=1280, height=720, fps=30, use_dshow=True)
            self._camera_message = "Camera connected"
            return True
        except Exception as exc:
            self._camera = None
            self._status_code = "CAMERA_FAULT"
            self._status_text = "Camera unavailable · check USB connection"
            self._camera_message = str(exc)
            self.inspectionChanged.emit()
            return False

    def _process_frame(self) -> None:
        if not self._running or self._camera is None:
            return
        ok, frame = self._camera.read()
        if not ok or frame is None:
            self._status_code = "CAMERA_FAULT"
            self._status_text = "Camera frame unavailable"
            self._camera_message = "Camera read failed"
            self._publish_idle_telemetry(LiveInspectionState.CAMERA_FAULT, self._status_text)
            self.inspectionChanged.emit()
            return

        try:
            out = self._engine.process_frame(frame)
        except Exception as exc:
            self._status_code = "CAMERA_FAULT"
            self._status_text = "Inspection engine error"
            self._camera_message = str(exc)
            self._publish_idle_telemetry(LiveInspectionState.CAMERA_FAULT, self._status_text)
            self.inspectionChanged.emit()
            return

        glass_present = self._glass_present_from_out(out)
        confidence, confirmed = self._update_confidence(out, glass_present)
        self._baseplate_missing_elapsed = self._update_baseplate_presence(out, glass_present)
        self._publish_live_telemetry(out, confidence, glass_present)
        self._update_cycle_and_record(out, glass_present, confirmed)
        self._update_live_display(out, confidence, confirmed, glass_present)

        display = out.overlay_bgr if getattr(out, "overlay_bgr", None) is not None else frame
        self._frame_provider.set_bgr_frame(display)
        self._image_version += 1
        self.imageVersionChanged.emit()
        self.inspectionChanged.emit()

    def _glass_present_from_out(self, out: Any) -> bool:
        state = str(getattr(out, "state", "") or "").upper()
        if state in {"TRACK", "PASS", "FAIL"}:
            return True
        stab_info = getattr(out, "stab_info", None)
        if not isinstance(stab_info, dict) or not bool(stab_info.get("glass_presence_confirmed")):
            return False
        try:
            checked_frame = int(stab_info["glass_presence_checked_frame"])
            engine_frame = int(stab_info["engine_frame_index"])
            cadence = max(1, int(self._engine.settings.stab_every_n))
            return 0 <= engine_frame - checked_frame <= cadence + 1
        except Exception:
            return False

    def _evidence_outcome(self, out: Any, glass_present: bool) -> Optional[str]:
        state = str(getattr(out, "state", "") or "").upper()
        if state in {"PASS", "FAIL"}:
            return state
        if state == "SEARCH" and glass_present:
            return "BASEPLATE"
        return None

    def _update_confidence(self, out: Any, glass_present: bool) -> tuple[int, bool]:
        now = time.monotonic()
        elapsed = 0.0 if self._confidence_last_updated is None else min(
            0.25, max(0.0, now - self._confidence_last_updated)
        )
        self._confidence_last_updated = now
        outcome = self._evidence_outcome(out, glass_present)
        gain_per_second = 1.0 / max(0.2, self._decision_confirmation_s)
        decay_per_second = 0.18 if outcome is None else 0.46
        for key, score in self._confidence_scores.items():
            score += gain_per_second * elapsed if key == outcome else -decay_per_second * elapsed
            self._confidence_scores[key] = max(0.0, min(1.0, score))

        value = float(self._confidence_scores.get(outcome, 0.0)) if outcome else 0.0
        confirmed = bool(outcome in {"PASS", "FAIL"} and value >= 0.999)
        percent = int(min(100.0, value * 100.0))
        return (min(99, percent) if outcome in {"PASS", "FAIL"} and not confirmed else percent, confirmed)

    def _update_baseplate_presence(self, out: Any, glass_present: bool) -> float:
        state = str(getattr(out, "state", "") or "").upper()
        if state != "SEARCH" or not glass_present or self._report_terminal_recorded:
            self._baseplate_missing_since = None
            return 0.0
        now = time.monotonic()
        if self._baseplate_missing_since is None:
            self._baseplate_missing_since = now
        return max(0.0, now - self._baseplate_missing_since)

    def _update_cycle_and_record(self, out: Any, glass_present: bool, decision_confirmed: bool) -> None:
        state = str(getattr(out, "state", "") or "").upper()
        now = time.monotonic()
        if state == "SEARCH" and not glass_present:
            if self._clear_since is None:
                self._clear_since = now
            if self._report_terminal_recorded and self._plc_service.status().awaiting_acknowledgement:
                self._cycle_text = "RESULT SAVED · WAITING FOR PLC ACKNOWLEDGEMENT"
            elif self._report_terminal_recorded and now - self._clear_since >= self._rearm_after_s:
                self._reset_cycle_state()
                self._cycle_text = "READY FOR NEXT GLASS"
            elif self._report_terminal_recorded:
                self._cycle_text = f"CLEARING STATION · {max(0.0, self._rearm_after_s - (now - self._clear_since)):.1f}s"
            else:
                self._cycle_text = "READY FOR NEXT GLASS"
            return

        self._clear_since = None
        if self._report_terminal_recorded:
            self._cycle_text = "RESULT SAVED · HOLDING CURRENT GLASS"
            return
        self._cycle_text = "INSPECTING CURRENT GLASS" if glass_present else "READY FOR NEXT GLASS"

        if state in {"PASS", "FAIL"} and decision_confirmed:
            self._record_terminal_result(out, state)
        elif (
            state == "SEARCH"
            and glass_present
            and self._baseplate_missing_elapsed >= self._track_alarm_after_s
        ):
            self._record_baseplate_missing(out)

    def _record_terminal_result(self, out: Any, state: str) -> None:
        if self._report_terminal_recorded:
            return
        stab = getattr(out, "stab_info", None)
        metrics = stab.get("inspection_metrics") if isinstance(stab, dict) else []
        failed_labels = [
            str(metric.get("label", "")).replace(" OFFSET", "").strip()
            for metric in metrics or []
            if isinstance(metric, dict) and metric.get("passed") is False
        ]
        cause = " + ".join(dict.fromkeys(label for label in failed_labels if label)) if state == "FAIL" else ""
        cause = cause or ("Outside recipe limits" if state == "FAIL" else "")
        self._commit_report(out, result=state, cause=cause, metrics=metrics or [])

    def _record_baseplate_missing(self, out: Any) -> None:
        if self._report_terminal_recorded:
            return
        self._commit_report(
            out,
            result="FAIL",
            cause="BASEPLATE NOT FOUND",
            metrics=[{"label": "BASEPLATE NOT FOUND", "passed": False}],
            outcome=InspectionOutcome.BASEPLATE_NOT_FOUND,
        )

    def _commit_report(
        self,
        out: Any,
        *,
        result: str,
        cause: str,
        metrics: list[dict[str, Any]],
        outcome: Optional[InspectionOutcome] = None,
    ) -> None:
        try:
            event_id = self._report_store.record_result(
                result=result,
                recipe=self._current_recipe,
                cause=cause,
                metrics=metrics,
            )
        except Exception as exc:
            self._status_code = "FAIL"
            self._status_text = f"Inspection result could not be saved: {exc}"
            return

        self._report_terminal_recorded = True
        result_label = "BASEPLATE NOT FOUND" if outcome == InspectionOutcome.BASEPLATE_NOT_FOUND else result
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._last_saved = f"{timestamp} · {result_label}" + (f" · {cause}" if cause else "")
        self._cycle_text = "RESULT SAVED · HOLDING CURRENT GLASS"
        plc_outcome = outcome or (InspectionOutcome.PASS if result == "PASS" else InspectionOutcome.FAIL)
        try:
            decision = build_decision_from_output(
                out,
                outcome=plc_outcome,
                recipe=self._current_recipe,
                report_event_id=event_id,
                failure_cause=cause,
                confidence_percent=self._confidence_percent,
            )
            self._plc_service.submit_terminal_result(decision)
        except Exception:
            # A configured PLC handoff must never erase a durable inspection
            # record; it remains visible through the cycle/status UI.
            pass
        self.refresh_reports()

    def _update_live_display(self, out: Any, confidence: int, confirmed: bool, glass_present: bool) -> None:
        state = str(getattr(out, "state", "") or "").upper()
        self._confidence_percent = confidence
        stab = getattr(out, "stab_info", None)
        metrics = stab.get("inspection_metrics") if isinstance(stab, dict) else []
        values = {str(metric.get("label", "")).replace(" OFFSET", "").upper(): metric for metric in metrics or [] if isinstance(metric, dict)}
        self._delta_x = str(values.get("X", {}).get("value", "—"))
        self._delta_y = str(values.get("Y", {}).get("value", "—"))
        self._angle = str(values.get("ANGLE", {}).get("value", "—"))

        if state == "SEARCH" and glass_present:
            self._status_code = "BASEPLATE_MISSING" if self._baseplate_missing_elapsed >= self._track_alarm_after_s else "BASEPLATE_WARNING"
            self._status_text = (
                "Fitted glass notch confirmed · baseplate not detected "
                f"({self._baseplate_missing_elapsed:.1f}s)"
            )
            self._baseplate_text = "Notch geometry confirmed · baseplate absent"
        elif state in {"PASS", "FAIL"}:
            self._status_code = state
            result_word = "confirmed" if confirmed else "candidate"
            self._status_text = f"{state} {result_word} · {confidence}% decision confidence"
            self._baseplate_text = "Baseplate detected"
        elif state == "TRACK":
            self._status_code = "TRACKING"
            self._status_text = "Verifying fitted notch and baseplate position"
            self._baseplate_text = "Baseplate detected" if glass_present else "Waiting for fitted glass notch"
        elif state == "SETUP":
            self._status_code = "SETUP_REQUIRED"
            self._status_text = "Recipe setup required"
            self._baseplate_text = "Waiting for recipe setup"
        else:
            self._status_code = "SEARCH"
            self._status_text = "Searching for fitted glass notch"
            self._baseplate_text = "Waiting for fitted glass notch"

    def _publish_live_telemetry(self, out: Any, confidence: int, glass_present: bool) -> None:
        try:
            self._plc_service.update_live_telemetry(
                build_live_telemetry(
                    out,
                    confidence_percent=confidence,
                    glass_present=glass_present,
                    baseplate_alarm_active=self._baseplate_missing_elapsed >= self._track_alarm_after_s,
                )
            )
        except Exception:
            pass

    def _publish_idle_telemetry(self, state: LiveInspectionState, text: str) -> None:
        try:
            self._plc_service.update_live_telemetry(InspectionTelemetry(state=state, state_text=text))
        except Exception:
            pass

    def _reset_cycle_state(self) -> None:
        self._report_terminal_recorded = False
        self._clear_since = None
        self._baseplate_missing_since = None
        self._baseplate_missing_elapsed = 0.0
        self._confidence_scores = {"PASS": 0.0, "FAIL": 0.0, "BASEPLATE": 0.0}
        self._confidence_last_updated = None
        self._confidence_percent = 0
