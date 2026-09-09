from __future__ import annotations

import unittest
from time import monotonic, sleep

from hmi_app.plc.handshake import PendingDecisionError, PlcHandshake
from hmi_app.plc.decision_builder import build_live_telemetry
from hmi_app.plc.models import (
    InspectionDecision,
    InspectionOutcome,
    InspectionTelemetry,
    LiveInspectionState,
    PlcInputs,
)
from hmi_app.plc.service import InspectionPlcService
from hmi_app.plc.service import PlcServiceConfig
from hmi_app.plc.transport import InMemoryPlcTransport


class PlcHandshakeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.handshake = PlcHandshake()

    def test_pending_result_stays_valid_until_matching_acknowledgement(self) -> None:
        decision = self.handshake.submit(InspectionDecision(outcome=InspectionOutcome.PASS, recipe="R1"))
        self.assertEqual(decision.sequence, 1)

        before_ack = self.handshake.build_outputs(
            telemetry=InspectionTelemetry(state=LiveInspectionState.TRACKING),
            heartbeat=9,
        )
        self.assertTrue(before_ack.result_valid)
        self.assertEqual(before_ack.result_sequence, 1)
        self.assertEqual(before_ack.result_code, InspectionOutcome.PASS)

        self.assertIsNone(self.handshake.observe_inputs(PlcInputs(acknowledgement_sequence=99)))
        self.assertTrue(self.handshake.build_outputs(telemetry=None, heartbeat=10).result_valid)

        self.assertEqual(self.handshake.observe_inputs(PlcInputs(acknowledgement_sequence=1)), decision)
        after_ack = self.handshake.build_outputs(telemetry=None, heartbeat=11)
        self.assertFalse(after_ack.result_valid)
        self.assertEqual(after_ack.result_code, InspectionOutcome.UNKNOWN)
        self.assertEqual(self.handshake.last_acknowledged_sequence, 1)

    def test_pending_result_cannot_be_overwritten(self) -> None:
        self.handshake.submit(InspectionDecision(outcome=InspectionOutcome.FAIL))
        with self.assertRaises(PendingDecisionError):
            self.handshake.submit(InspectionDecision(outcome=InspectionOutcome.PASS))

    def test_baseplate_not_found_has_its_own_plc_result_code(self) -> None:
        self.handshake.submit(
            InspectionDecision(outcome=InspectionOutcome.BASEPLATE_NOT_FOUND, failure_cause="BASEPLATE NOT FOUND")
        )
        outputs = self.handshake.build_outputs(telemetry=None, heartbeat=1)
        self.assertTrue(outputs.result_valid)
        self.assertEqual(outputs.result_code, InspectionOutcome.BASEPLATE_NOT_FOUND)

    def test_disabled_service_never_publishes_a_terminal_result(self) -> None:
        service = InspectionPlcService.disabled()
        self.assertIsNone(service.submit_terminal_result(InspectionDecision(outcome=InspectionOutcome.PASS)))
        self.assertFalse(service.status().awaiting_acknowledgement)

    def test_live_telemetry_never_labels_pixel_measurements_as_millimetres(self) -> None:
        class Output:
            state = "FAIL"
            status_text = "OUTSIDE RECIPE LIMITS"
            stab_info = {
                "current_offset_display": {"unit": "px", "dx": 14.0, "dy": -9.0},
                "inspection_metrics": [
                    {"label": "ANGLE", "signed_measurement": -0.35},
                ],
            }

        telemetry = build_live_telemetry(Output(), confidence_percent=75, glass_present=True)
        self.assertEqual(telemetry.state, LiveInspectionState.FAIL)
        self.assertEqual(telemetry.confidence_percent, 75.0)
        self.assertIsNone(telemetry.delta_x_mm)
        self.assertIsNone(telemetry.delta_y_mm)
        self.assertEqual(telemetry.angle_deg, -0.35)
        self.assertTrue(telemetry.baseplate_detected)

    def test_worker_holds_result_until_the_transport_receives_matching_ack(self) -> None:
        transport = InMemoryPlcTransport()
        service = InspectionPlcService(
            PlcServiceConfig(enabled=True, poll_interval_s=0.01, reconnect_delay_s=0.01),
            transport_factory=lambda: transport,
        )

        def wait_until(predicate, timeout_s: float = 1.0) -> bool:
            deadline = monotonic() + timeout_s
            while monotonic() < deadline:
                if predicate():
                    return True
                sleep(0.005)
            return bool(predicate())

        service.start()
        try:
            self.assertTrue(wait_until(lambda: transport.connected))
            decision = service.submit_terminal_result(InspectionDecision(outcome=InspectionOutcome.PASS))
            self.assertIsNotNone(decision)
            self.assertTrue(
                wait_until(
                    lambda: any(
                        output.result_valid and output.result_sequence == decision.sequence
                        for output in transport.writes
                    )
                )
            )
            self.assertTrue(service.status().awaiting_acknowledgement)

            transport.inputs = PlcInputs(acknowledgement_sequence=decision.sequence)
            self.assertTrue(
                wait_until(
                    lambda: service.status().last_acknowledged_sequence == decision.sequence
                    and not service.status().awaiting_acknowledgement
                )
            )
            self.assertTrue(wait_until(lambda: any(not output.result_valid for output in transport.writes)))
        finally:
            service.stop()


if __name__ == "__main__":
    unittest.main()
