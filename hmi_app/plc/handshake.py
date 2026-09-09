from __future__ import annotations

from typing import Optional

from hmi_app.plc.models import (
    InspectionDecision,
    InspectionOutcome,
    InspectionTelemetry,
    LiveInspectionState,
    PlcInputs,
    PlcOutputs,
)


class PendingDecisionError(RuntimeError):
    """A second result tried to overwrite one the PLC has not acknowledged."""


class PlcHandshake:
    """Pure sequence/acknowledgement state machine.

    The service synchronizes access to this object.  It does not connect to a
    PLC, choose an S7 DB layout, or operate a motor.
    """

    _MAX_SEQUENCE = 2_147_483_647

    def __init__(self) -> None:
        self._next_sequence = 1
        self._pending: Optional[InspectionDecision] = None
        self._last_acknowledged_sequence = 0

    @property
    def pending_decision(self) -> Optional[InspectionDecision]:
        return self._pending

    @property
    def last_acknowledged_sequence(self) -> int:
        return self._last_acknowledged_sequence

    def submit(self, decision: InspectionDecision) -> InspectionDecision:
        """Assign a fresh sequence without overwriting an outstanding result."""
        if self._pending is not None:
            raise PendingDecisionError(
                "Cannot publish another inspection before the PLC acknowledges "
                f"sequence {self._pending.sequence}."
            )
        if decision.sequence not in (0, None):
            raise ValueError("Only unsequenced inspection decisions can be submitted")
        if decision.outcome == InspectionOutcome.UNKNOWN:
            raise ValueError("A terminal inspection must have a real outcome")

        sequenced = decision.with_sequence(self._next_sequence)
        self._pending = sequenced
        self._next_sequence = 1 if self._next_sequence >= self._MAX_SEQUENCE else self._next_sequence + 1
        return sequenced

    def observe_inputs(self, inputs: PlcInputs) -> Optional[InspectionDecision]:
        """Clear a result only when the PLC echoes that exact sequence."""
        if self._pending is None:
            return None
        if int(inputs.acknowledgement_sequence or 0) != self._pending.sequence:
            return None

        acknowledged = self._pending
        self._last_acknowledged_sequence = acknowledged.sequence
        self._pending = None
        return acknowledged

    def build_outputs(
        self,
        *,
        telemetry: Optional[InspectionTelemetry],
        heartbeat: int,
    ) -> PlcOutputs:
        """Build one coherent output image for a transport write."""
        pending = self._pending
        telemetry = telemetry or InspectionTelemetry(state=LiveInspectionState.IDLE)
        return PlcOutputs(
            heartbeat=int(heartbeat),
            result_valid=pending is not None,
            result_sequence=0 if pending is None else pending.sequence,
            result_code=InspectionOutcome.UNKNOWN if pending is None else pending.outcome,
            result_confidence_percent=None if pending is None else pending.confidence_percent,
            result_delta_x_mm=None if pending is None else pending.delta_x_mm,
            result_delta_y_mm=None if pending is None else pending.delta_y_mm,
            result_angle_deg=None if pending is None else pending.angle_deg,
            live_state=telemetry.state,
            live_confidence_percent=telemetry.confidence_percent,
            live_delta_x_mm=telemetry.delta_x_mm,
            live_delta_y_mm=telemetry.delta_y_mm,
            live_angle_deg=telemetry.angle_deg,
            baseplate_detected=telemetry.baseplate_detected,
        )
