from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class ContourMeasurement:
    contour: np.ndarray
    center_xy: Tuple[float, float]
    angle_deg: float
    area: float


class ContourHysteresis:
    def __init__(
        self,
        *,
        enabled: bool = True,
        center_deadband_px: float = 3.0,
        angle_deadband_deg: float = 1.0,
        area_deadband_ratio: float = 0.06,
        hold_max_frames: int = 10,
    ):
        self.enabled = bool(enabled)
        self.center_deadband_px = float(center_deadband_px)
        self.angle_deadband_deg = float(angle_deadband_deg)
        self.area_deadband_ratio = float(area_deadband_ratio)
        self.hold_max_frames = int(hold_max_frames)

        self._last: Optional[ContourMeasurement] = None
        self._miss_hold_count = 0

    def reset(self):
        self._last = None
        self._miss_hold_count = 0

    def configure(
        self,
        *,
        enabled: bool,
        center_deadband_px: float,
        angle_deadband_deg: float,
        area_deadband_ratio: float,
        hold_max_frames: int,
    ):
        self.enabled = bool(enabled)
        self.center_deadband_px = float(center_deadband_px)
        self.angle_deadband_deg = float(angle_deadband_deg)
        self.area_deadband_ratio = float(area_deadband_ratio)
        self.hold_max_frames = int(hold_max_frames)

    def update(self, measurement: Optional[ContourMeasurement]) -> Optional[ContourMeasurement]:
        if measurement is None:
            if self._last is None:
                return None
            if self.enabled and self._miss_hold_count < max(0, self.hold_max_frames):
                self._miss_hold_count += 1
                return self._last
            self.reset()
            return None

        self._miss_hold_count = 0
        if self._last is None:
            self._last = measurement
            return measurement

        if not self.enabled:
            self._last = measurement
            return measurement

        prev = self._last
        dx = abs(float(measurement.center_xy[0]) - float(prev.center_xy[0]))
        dy = abs(float(measurement.center_xy[1]) - float(prev.center_xy[1]))
        da = abs(float(measurement.angle_deg) - float(prev.angle_deg))

        base_area = max(abs(float(prev.area)), 1e-6)
        area_delta_ratio = abs(float(measurement.area) - float(prev.area)) / base_area

        within_deadband = (
            dx <= self.center_deadband_px
            and dy <= self.center_deadband_px
            and da <= self.angle_deadband_deg
            and area_delta_ratio <= self.area_deadband_ratio
        )

        if within_deadband:
            return prev

        self._last = measurement
        return measurement
