# PLC Integration Contract (Phase 1)

This module prepares MBPAC for an S7-1215C without writing to a controller by
default. It deliberately contains no IP address, DB number, credentials, or a
bit that can directly run the conveyor.

## Ownership

Python publishes inspection information. The PLC owns all machine outputs,
safety circuits, E-stops, guards, and conveyor-release logic. The Siemens HMI
should read normal PLC tags; Python should not control the HMI or conveyor
directly.

## Proposed DB handshake

Electrical should create a dedicated, symbolically named PLC DB and agree the
final types/addresses before a Snap7 or OPC UA adapter is enabled.

| Direction | Required field | Meaning |
| --- | --- | --- |
| PC → PLC | `QC_RESULT_VALID` | A completed result is available. |
| PC → PLC | `QC_RESULT_SEQUENCE` | Monotonic, nonzero result identity. |
| PC → PLC | `QC_RESULT_CODE` | `1=PASS`, `2=FAIL`, `3=BASEPLATE_NOT_FOUND`. |
| PC → PLC | `QC_DELTA_X_MM`, `QC_DELTA_Y_MM`, `QC_ANGLE_DEG` | Optional signed measurements; pixel values are never relabelled as mm. |
| PC → PLC | confidence, live state, heartbeat | Display/diagnostic values only. |
| PLC → PC | `QC_ACK_SEQUENCE` | Echo the exact accepted result sequence. |
| PLC → PC | `QC_MACHINE_READY`, PLC fault | Machine status only. |

The PLC should accept a result only while `QC_RESULT_VALID` is set and the
sequence differs from the last accepted sequence. It must echo that sequence
before releasing the station, and must fail safe to **hold** on a stale
heartbeat, invalid result, mismatch, communication loss, or safety fault.

## Current application behavior

Auto Mode sends live display telemetry every processed frame and publishes a
terminal result only after its local report row is saved. It uses the existing
fitted-notch glass-presence gate before it can publish the separate
`BASEPLATE_NOT_FOUND` result. Network I/O remains outside the Qt camera timer.

## Next step after electrical approval

Provide the exact DB symbols/addresses, PLC IP, access policy, and test plan.
Then add either a licensed S7-1200 OPC UA adapter (preferred) or an approved
`python-snap7` adapter behind `PlcTransport`, test with a simulator, and only
then connect the live conveyor.
