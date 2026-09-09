"""Safe, transport-agnostic PLC handoff for completed inspections.

The package intentionally contains no PLC address, DB number, credential, or
direct conveyor-output bit.  Electrical owns that mapping; Python owns a
deterministic result/acknowledgement protocol.
"""

from hmi_app.plc.decision_builder import build_decision_from_output, build_live_telemetry
from hmi_app.plc.handshake import PendingDecisionError, PlcHandshake
from hmi_app.plc.models import (
    InspectionDecision,
    InspectionOutcome,
    InspectionTelemetry,
    LiveInspectionState,
    PlcConnectionState,
    PlcInputs,
    PlcOutputs,
    PlcStatus,
)
from hmi_app.plc.service import InspectionPlcService, PlcServiceConfig
from hmi_app.plc.transport import InMemoryPlcTransport, PlcTransport

__all__ = [
    "InspectionDecision",
    "InspectionOutcome",
    "InspectionPlcService",
    "InspectionTelemetry",
    "InMemoryPlcTransport",
    "LiveInspectionState",
    "PendingDecisionError",
    "PlcConnectionState",
    "PlcHandshake",
    "PlcInputs",
    "PlcOutputs",
    "PlcServiceConfig",
    "PlcStatus",
    "PlcTransport",
    "build_decision_from_output",
    "build_live_telemetry",
]
