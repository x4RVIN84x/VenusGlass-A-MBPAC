import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def compare_to_golden(detected_center, detected_angle, golden_center, golden_angle, tolerances):
    """
    detected_center: (x, y) ROI-relative (or any consistent coordinate system)
    golden_center:   (x, y) same coordinate system as detected_center
    """
    dx = abs(detected_center[0] - golden_center[0])
    dy = abs(detected_center[1] - golden_center[1])
    da = abs(detected_angle - golden_angle)

    result = {
        "dx": dx,
        "dy": dy,
        "dtheta": da,
        "pass_x": dx <= tolerances["x"],
        "pass_y": dy <= tolerances["y"],
        "pass_angle": da <= tolerances["angle"],
    }
    result["overall"] = result["pass_x"] and result["pass_y"] and result["pass_angle"]
    return result


def compare_rect_uv(detected_uv, golden_uv, tolerances_uv):
    """
    For frit-rect canonical frame comparisons (u,v in rect pixels).
    tolerances_uv = {"x":..., "y":...}  (angle handled separately)
    """
    dx = abs(detected_uv[0] - golden_uv[0])
    dy = abs(detected_uv[1] - golden_uv[1])
    return {
        "dx": dx, "dy": dy,
        "pass_x": dx <= tolerances_uv["x"],
        "pass_y": dy <= tolerances_uv["y"],
        "overall": (dx <= tolerances_uv["x"]) and (dy <= tolerances_uv["y"])
    }


# --- If you had other overlay utilities here, keep them as-is. ---
# I’m not touching your overlay_reference_and_test implementation
# because your current UI/debug workflow depends on it.
#
# (If you want, paste that function here and I’ll update it to also draw
# the frit inner box + predicted ROI mapping.)
