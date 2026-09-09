from __future__ import annotations

from typing import Protocol, runtime_checkable

from hmi_app.plc.models import PlcInputs, PlcOutputs


@runtime_checkable
class PlcTransport(Protocol):
    """Worker-thread-only interface for a future OPC UA or Snap7 adapter."""

    def connect(self) -> None:
        """Open the connection."""

    def read_inputs(self) -> PlcInputs:
        """Read acknowledgement and PLC health data."""

    def write_outputs(self, outputs: PlcOutputs) -> None:
        """Write a complete PC-to-PLC output image."""

    def close(self) -> None:
        """Close the connection."""


class InMemoryPlcTransport:
    """Test/simulation transport.  It cannot connect to real equipment."""

    def __init__(self) -> None:
        self.connected = False
        self.inputs = PlcInputs()
        self.writes: list[PlcOutputs] = []

    def connect(self) -> None:
        self.connected = True

    def read_inputs(self) -> PlcInputs:
        if not self.connected:
            raise RuntimeError("In-memory PLC transport is not connected")
        return self.inputs

    def write_outputs(self, outputs: PlcOutputs) -> None:
        if not self.connected:
            raise RuntimeError("In-memory PLC transport is not connected")
        self.writes.append(outputs)

    def close(self) -> None:
        self.connected = False
