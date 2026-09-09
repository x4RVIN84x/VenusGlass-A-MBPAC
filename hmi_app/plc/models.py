from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import IntEnum
from typing import Optional


class InspectionOutcome(IntEnum):
    """Stable PLC result codes for an inspection that has finished."""

    UNKNOWN = 0
    PASS = 1
    FAIL = 2
    BASEPLATE_NOT_FOUND = 3


class LiveInspectionState(IntEnum):
    """Display-only live state codes for a PLC/HMI data block."""

    IDLE = 0
    SEARCHING = 1
    TRACKING = 2
    PASS = 3
    FAIL = 4
    BASEPLATE_WARNING = 5
    BASEPLATE_MISSING = 6
    SETUP_REQUIRED = 7
    CAMERA_FAULT = 8


class PlcConnectionState(IntEnum):
    DISABLED = 0
    STOPPED = 1
    CONNECTING = 2
    ONLINE = 3
    RECONNECTING = 4
    FAULT = 5


@dataclass(frozen=True)
class InspectionTelemetry:
    """Current display data; it can never authorize conveyor movement."""

    state: LiveInspectionState
    state_text: str = ""
    confidence_percent: Optional[float] = None
    delta_x_mm: Optional[float] = None
    delta_y_mm: Optional[float] = None
    angle_deg: Optional[float] = None
    baseplate_detected: Optional[bool] = None
    observed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class InspectionDecision:
    """One immutable terminal result awaiting acknowledgement from the PLC."""

    outcome: InspectionOutcome
    recipe: str = ""
    report_event_id: int = 0
    failure_cause: str = ""
    confidence_percent: Optional[float] = None
    delta_x_mm: Optional[float] = None
    delta_y_mm: Optional[float] = None
    angle_deg: Optional[float] = None
    sequence: int = 0
    committed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def with_sequence(self, sequence: int) -> "InspectionDecision":
        if sequence <= 0:
            raise ValueError("PLC result sequences must be positive")
        return replace(self, sequence=int(sequence))


@dataclass(frozen=True)
class PlcInputs:
    """Values read from the PLC-owned side of the handshake."""

    acknowledgement_sequence: int = 0
    machine_ready: bool = False
    plc_fault: bool = False


@dataclass(frozen=True)
class PlcOutputs:
    """One PC-to-PLC output image.

    There is deliberately no ``release_conveyor`` field.  A PLC must apply its
    own guards, E-stop, interlocks, and a matching acknowledged sequence before
    it releases the station.
    """

    heartbeat: int = 0
    result_valid: bool = False
    result_sequence: int = 0
    result_code: InspectionOutcome = InspectionOutcome.UNKNOWN
    result_confidence_percent: Optional[float] = None
    result_delta_x_mm: Optional[float] = None
    result_delta_y_mm: Optional[float] = None
    result_angle_deg: Optional[float] = None
    live_state: LiveInspectionState = LiveInspectionState.IDLE
    live_confidence_percent: Optional[float] = None
    live_delta_x_mm: Optional[float] = None
    live_delta_y_mm: Optional[float] = None
    live_angle_deg: Optional[float] = None
    baseplate_detected: Optional[bool] = None


@dataclass(frozen=True)
class PlcStatus:
    """Thread-safe service status for a later operator-status indicator."""

    connection_state: PlcConnectionState
    pending_sequence: int = 0
    last_acknowledged_sequence: int = 0
    last_error: str = ""

    @property
    def awaiting_acknowledgement(self) -> bool:
        return self.pending_sequence > 0
