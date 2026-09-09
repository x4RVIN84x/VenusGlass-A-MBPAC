from __future__ import annotations

from math import isfinite
from typing import Any, Optional

from hmi_app.plc.models import (
    InspectionDecision,
    InspectionOutcome,
    InspectionTelemetry,
    LiveInspectionState,
)


def _finite_number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _live_state_from_output(
    out: Any,
    *,
    glass_present: Optional[bool],
    baseplate_alarm_active: bool,
) -> LiveInspectionState:
    state = str(getattr(out, "state", "") or "").upper()
    if state == "SEARCH" and glass_present:
        return LiveInspectionState.BASEPLATE_MISSING if baseplate_alarm_active else LiveInspectionState.BASEPLATE_WARNING
    return {
        "SEARCH": LiveInspectionState.SEARCHING,
        "TRACK": LiveInspectionState.TRACKING,
        "PASS": LiveInspectionState.PASS,
        "FAIL": LiveInspectionState.FAIL,
        "SETUP": LiveInspectionState.SETUP_REQUIRED,
    }.get(state, LiveInspectionState.IDLE)


def _measurements_from_output(out: Any) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Extract explicitly millimetre/degree values without relabelling px."""
    stab_info = getattr(out, "stab_info", None)
    if not isinstance(stab_info, dict):
        return None, None, None

    dx_mm = None
    dy_mm = None
    offset = stab_info.get("current_offset_display")
    if isinstance(offset, dict) and str(offset.get("unit", "")).lower() == "mm":
        dx_mm = _finite_number(offset.get("dx"))
        dy_mm = _finite_number(offset.get("dy"))

    angle_deg = None
    metrics = stab_info.get("inspection_metrics")
    if isinstance(metrics, (tuple, list)):
        for metric in metrics:
            if not isinstance(metric, dict):
                continue
            label = str(metric.get("label", "")).replace(" OFFSET", "").strip().upper()
            if label == "ANGLE":
                angle_deg = _finite_number(metric.get("signed_measurement"))
                break

    return dx_mm, dy_mm, angle_deg


def _percent(value: Any) -> Optional[float]:
    number = _finite_number(value)
    if number is None:
        return None
    return max(0.0, min(100.0, number))


def build_live_telemetry(
    out: Any,
    *,
    confidence_percent: Any = None,
    glass_present: Optional[bool] = None,
    baseplate_alarm_active: bool = False,
) -> InspectionTelemetry:
    """Create low-risk display data; it is never a conveyor authorization."""
    dx_mm, dy_mm, angle_deg = _measurements_from_output(out)
    raw_state = str(getattr(out, "state", "") or "").upper()
    if raw_state in {"TRACK", "PASS", "FAIL"}:
        baseplate_detected = True
    elif raw_state == "SEARCH" and glass_present:
        baseplate_detected = False
    else:
        baseplate_detected = None
    return InspectionTelemetry(
        state=_live_state_from_output(
            out,
            glass_present=glass_present,
            baseplate_alarm_active=baseplate_alarm_active,
        ),
        state_text=str(getattr(out, "status_text", "") or ""),
        confidence_percent=_percent(confidence_percent),
        delta_x_mm=dx_mm,
        delta_y_mm=dy_mm,
        angle_deg=angle_deg,
        baseplate_detected=baseplate_detected,
    )


def build_decision_from_output(
    out: Any,
    *,
    outcome: InspectionOutcome,
    recipe: str = "",
    report_event_id: int = 0,
    failure_cause: str = "",
    confidence_percent: Any = None,
) -> InspectionDecision:
    """Build a committed PLC payload from a persisted Auto Mode result."""
    dx_mm, dy_mm, angle_deg = _measurements_from_output(out)
    return InspectionDecision(
        outcome=outcome,
        recipe=str(recipe or ""),
        report_event_id=max(0, int(report_event_id or 0)),
        failure_cause=str(failure_cause or ""),
        confidence_percent=_percent(confidence_percent),
        delta_x_mm=dx_mm,
        delta_y_mm=dy_mm,
        angle_deg=angle_deg,
    )
