# compare_to_golden.py
def _ang_diff_deg(a, b):
    """Smallest absolute difference between angles in degrees."""
    d = (float(a) - float(b) + 180.0) % 360.0 - 180.0
    return abs(d)

def compare_to_golden(detected_center, detected_angle, golden_center, golden_angle, tolerances):
    dx = abs(float(detected_center[0]) - float(golden_center[0]))
    dy = abs(float(detected_center[1]) - float(golden_center[1]))
    da = _ang_diff_deg(detected_angle, golden_angle)

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
