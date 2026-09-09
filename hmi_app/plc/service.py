from __future__ import annotations

from dataclasses import dataclass
from threading import Event, RLock, Thread
from typing import Callable, Optional

from hmi_app.plc.handshake import PendingDecisionError, PlcHandshake
from hmi_app.plc.models import (
    InspectionDecision,
    InspectionTelemetry,
    PlcConnectionState,
    PlcStatus,
)
from hmi_app.plc.transport import PlcTransport


@dataclass(frozen=True)
class PlcServiceConfig:
    """Connection behavior; the approved S7 tag map lives outside source code."""

    enabled: bool = False
    poll_interval_s: float = 0.10
    reconnect_delay_s: float = 2.0

    def __post_init__(self) -> None:
        if self.poll_interval_s <= 0:
            raise ValueError("PLC polling interval must be positive")
        if self.reconnect_delay_s <= 0:
            raise ValueError("PLC reconnect delay must be positive")


class InspectionPlcService:
    """Background service for a PLC inspection handoff.

    The Qt thread may publish data, but the worker is the only code permitted
    to call a transport.  It holds ``result_valid`` until the PLC sends a
    matching acknowledgement sequence.  There is deliberately no method for
    writing a conveyor-release output.
    """

    def __init__(
        self,
        config: PlcServiceConfig | None = None,
        *,
        transport_factory: Optional[Callable[[], PlcTransport]] = None,
    ) -> None:
        self.config = config or PlcServiceConfig()
        self._transport_factory = transport_factory
        self._handshake = PlcHandshake()
        self._telemetry: Optional[InspectionTelemetry] = None
        self._state = PlcConnectionState.DISABLED if not self.config.enabled else PlcConnectionState.STOPPED
        self._last_error = ""
        self._heartbeat = 0
        self._lock = RLock()
        self._stop_event = Event()
        self._worker: Optional[Thread] = None

    @classmethod
    def disabled(cls) -> "InspectionPlcService":
        """Create the production-safe default until Electrical approves a map."""
        return cls(PlcServiceConfig(enabled=False))

    def start(self) -> None:
        """Start I/O only when both the service and transport are explicit."""
        if not self.config.enabled:
            with self._lock:
                self._state = PlcConnectionState.DISABLED
            return
        if self._transport_factory is None:
            with self._lock:
                self._state = PlcConnectionState.FAULT
                self._last_error = "PLC is enabled but no transport factory is configured"
            return
        if self._worker is not None and self._worker.is_alive():
            return

        self._stop_event.clear()
        self._worker = Thread(target=self._run, name="mbpac-plc", daemon=True)
        self._worker.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        self._stop_event.set()
        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=max(0.0, float(timeout_s)))
        with self._lock:
            self._state = PlcConnectionState.STOPPED if self.config.enabled else PlcConnectionState.DISABLED

    def update_live_telemetry(self, telemetry: InspectionTelemetry) -> None:
        """Store live display data without blocking a camera frame."""
        with self._lock:
            self._telemetry = telemetry

    def submit_terminal_result(self, decision: InspectionDecision) -> Optional[InspectionDecision]:
        """Hold one committed result until the PLC acknowledges its sequence."""
        if not self.config.enabled:
            return None
        with self._lock:
            return self._handshake.submit(decision)

    def status(self) -> PlcStatus:
        with self._lock:
            pending = self._handshake.pending_decision
            return PlcStatus(
                connection_state=self._state,
                pending_sequence=0 if pending is None else pending.sequence,
                last_acknowledged_sequence=self._handshake.last_acknowledged_sequence,
                last_error=self._last_error,
            )

    def _run(self) -> None:
        while not self._stop_event.is_set():
            transport: Optional[PlcTransport] = None
            try:
                with self._lock:
                    self._state = PlcConnectionState.CONNECTING
                    self._last_error = ""

                transport = self._transport_factory() if self._transport_factory is not None else None
                if transport is None:
                    raise RuntimeError("PLC transport factory returned no transport")
                transport.connect()

                with self._lock:
                    self._state = PlcConnectionState.ONLINE

                while not self._stop_event.is_set():
                    inputs = transport.read_inputs()
                    with self._lock:
                        self._handshake.observe_inputs(inputs)
                        self._heartbeat = 1 if self._heartbeat >= 2_147_483_647 else self._heartbeat + 1
                        outputs = self._handshake.build_outputs(
                            telemetry=self._telemetry,
                            heartbeat=self._heartbeat,
                        )
                    transport.write_outputs(outputs)
                    self._stop_event.wait(self.config.poll_interval_s)
            except Exception as exc:
                with self._lock:
                    self._state = PlcConnectionState.RECONNECTING
                    self._last_error = str(exc)
                self._stop_event.wait(self.config.reconnect_delay_s)
            finally:
                if transport is not None:
                    try:
                        transport.close()
                    except Exception:
                        pass

        with self._lock:
            self._state = PlcConnectionState.STOPPED


__all__ = ["InspectionPlcService", "PendingDecisionError", "PlcServiceConfig"]
