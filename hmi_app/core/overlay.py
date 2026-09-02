from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


# ----------------------------
# Helpers
# ----------------------------
def _safe_cv_pt(pt):
    """
    Convert point-like values into a safe OpenCV integer point.

    Handles:
      [x, y]
      (x, y)
      np.array([x, y])
      nested-ish arrays

    Returns None if invalid.
    """
    if pt is None:
        return None

    try:
        arr = np.asarray(pt, dtype=np.float64).reshape(-1)
    except Exception:
        return None

    if arr.size < 2:
        return None

    x = float(arr[0])
    y = float(arr[1])

    if not np.isfinite(x) or not np.isfinite(y):
        return None

    return int(round(x)), int(round(y))


def _direction_lines_from_correction(corr: dict):
    """
    Returns operator movement text as one/two short lines.

    Preferred source:
      baseplate_correction_vector["screen_dx/screen_dy"]

    Fallback source:
      current offset/error dictionaries, inverted as a correction.
    """
    if not isinstance(corr, dict):
        return []

    unit = str(corr.get("unit", "px"))

    try:
        dx = float(corr.get("screen_dx", 0.0))
        dy = float(corr.get("screen_dy", 0.0))
    except Exception:
        return []

    lines = []

    if abs(dx) >= 0.01:
        lines.append(f"{abs(dx):.2f}{unit} {'RIGHT' if dx > 0 else 'LEFT'}")

    if abs(dy) >= 0.01:
        # Camera/image coordinates: positive y is down on screen.
        lines.append(f"{abs(dy):.2f}{unit} {'DOWN' if dy > 0 else 'UP'}")

    if not lines:
        return ["CENTERED"]

    if len(lines) == 1:
        return [f"MOVE {lines[0]}"]

    return [f"MOVE {lines[0]}", f"AND {lines[1]}"]


def _direction_text_from_correction(corr: dict) -> str:
    lines = _direction_lines_from_correction(corr)
    if not lines:
        return ""
    return " / ".join(lines)


def _as_dict(v) -> dict:
    return v if isinstance(v, dict) else {}


def _setting(settings, name: str, default):
    if settings is None:
        return default
    return getattr(settings, name, default)


def _setting_any(settings, names, default):
    if settings is None:
        return default

    for name in names:
        if hasattr(settings, name):
            return getattr(settings, name)

    return default


def _is_good_pt(pt) -> bool:
    return _safe_cv_pt(pt) is not None


def _ipt(pt) -> Tuple[int, int]:
    p = _safe_cv_pt(pt)
    if p is None:
        return 0, 0
    return p


def _as_line(line):
    if line is None:
        return None

    try:
        vals = tuple(map(float, line))
    except Exception:
        return None

    if len(vals) != 4:
        return None

    if not all(np.isfinite(vals)):
        return None

    return vals


def _as_points(pts) -> Optional[np.ndarray]:
    if pts is None:
        return None

    try:
        arr = np.asarray(pts, dtype=np.float32)
    except Exception:
        return None

    if arr.size == 0:
        return None

    try:
        arr = arr.reshape(-1, 2)
    except Exception:
        return None

    arr = arr[np.isfinite(arr).all(axis=1)]

    if len(arr) <= 0:
        return None

    return arr


def _put_text(
    vis,
    text,
    org,
    *,
    scale=0.5,
    color=(255, 255, 255),
    thickness=1,
):
    x, y = org

    cv2.putText(
        vis,
        str(text),
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        float(scale),
        (0, 0, 0),
        int(thickness) + 2,
        cv2.LINE_AA,
    )

    cv2.putText(
        vis,
        str(text),
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        float(scale),
        color,
        int(thickness),
        cv2.LINE_AA,
    )


def _draw_points(vis, pts, color=(0, 255, 255), radius=1, step=16):
    arr = _as_points(pts)

    if arr is None:
        return

    step = max(1, int(step))
    H, W = vis.shape[:2]

    for i in range(0, len(arr), step):
        x = int(round(float(arr[i, 0])))
        y = int(round(float(arr[i, 1])))

        if 0 <= x < W and 0 <= y < H:
            cv2.circle(vis, (x, y), int(radius), color, -1, lineType=cv2.LINE_AA)


def _draw_polyline(vis, cnt, color=(0, 255, 0), thickness=1, closed=True):
    if cnt is None:
        return

    try:
        arr = np.asarray(cnt, dtype=np.int32)
    except Exception:
        return

    if arr.size == 0:
        return

    try:
        arr = arr.reshape(-1, 1, 2)
    except Exception:
        return

    if len(arr) < 2:
        return

    try:
        cv2.polylines(vis, [arr], bool(closed), color, int(thickness), lineType=cv2.LINE_AA)
    except Exception:
        pass


def _draw_fitline_clipped(vis, line, color=(0, 255, 0), thickness=2):
    line = _as_line(line)

    if line is None:
        return

    vx, vy, x0, y0 = line

    H, W = vis.shape[:2]

    p1 = (int(round(x0 - vx * 5000)), int(round(y0 - vy * 5000)))
    p2 = (int(round(x0 + vx * 5000)), int(round(y0 + vy * 5000)))

    ok, cp1, cp2 = cv2.clipLine((0, 0, W, H), p1, p2)

    if ok:
        cv2.line(vis, cp1, cp2, color, int(thickness), lineType=cv2.LINE_AA)


def _draw_readable_box_text(vis, text: str, org, *, color=(255, 255, 255), border=(255, 0, 255)):
    x, y = int(org[0]), int(org[1])

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1

    (tw, th), base = cv2.getTextSize(str(text), font, scale, thickness)

    pad_x = 10
    pad_y = 8

    x0 = max(0, x - pad_x)
    y0 = max(0, y - th - pad_y)
    x1 = min(vis.shape[1] - 1, x + tw + pad_x)
    y1 = min(vis.shape[0] - 1, y + base + pad_y)

    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 0), -1, lineType=cv2.LINE_AA)
    cv2.rectangle(vis, (x0, y0), (x1, y1), border, 1, lineType=cv2.LINE_AA)

    _put_text(vis, text, (x, y), scale=scale, color=color, thickness=thickness)



def _draw_movement_badge_bottom_right(vis, corr: dict):
    """
    HUD-style movement instruction.
    Always sits bottom-right, away from the baseplate/expected-center marker.
    """
    if vis is None or not isinstance(corr, dict):
        return

    lines = _direction_lines_from_correction(corr)
    if not lines:
        return

    H, W = vis.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.58
    thick = 2
    pad_x = 18
    pad_y = 14

    sizes = [cv2.getTextSize(str(t), font, scale, thick)[0] for t in lines]
    text_w = max((s[0] for s in sizes), default=180)
    text_h = sum((s[1] for s in sizes)) + (len(lines) - 1) * 12

    box_w = int(min(max(text_w + 2 * pad_x + 12, 260), max(260, W - 48)))
    box_h = int(max(58, text_h + 2 * pad_y + 8))

    # Bottom-right, lifted above the footer text.
    margin_r = 24
    margin_b = 48

    x0 = max(18, W - box_w - margin_r)
    y0 = max(130, H - box_h - margin_b)
    x1 = min(W - 18, x0 + box_w)
    y1 = min(H - 18, y0 + box_h)

    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 0), -1, lineType=cv2.LINE_AA)
    cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 0, 255), 2, lineType=cv2.LINE_AA)

    # Magenta side strip so operator notices it without making it huge.
    cv2.rectangle(vis, (x0, y0), (x0 + 8, y1), (255, 0, 255), -1, lineType=cv2.LINE_AA)

    total_line_h = 26
    start_y = int(round((y0 + y1) * 0.5 - (len(lines) - 1) * total_line_h * 0.5 + 8))

    for i, line in enumerate(lines):
        (tw, th), _base = cv2.getTextSize(str(line), font, scale, thick)
        tx = int(round((x0 + x1) * 0.5 - tw * 0.5))
        ty = int(round(start_y + i * total_line_h))

        cv2.putText(vis, str(line), (tx, ty), font, scale, (0, 0, 0), thick + 4, cv2.LINE_AA)
        cv2.putText(vis, str(line), (tx, ty), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def _fallback_correction_from_stab(stab_info: dict):
    """
    Some recovered engine builds do not populate baseplate_correction_vector in
    fallback ROI mode. Build a usable operator correction from the signed offset.
    """
    if not isinstance(stab_info, dict):
        return None

    corr = stab_info.get("baseplate_correction_vector")
    if isinstance(corr, dict):
        return corr

    offset = stab_info.get("current_offset_display")
    if isinstance(offset, dict):
        try:
            unit = str(offset.get("unit", "px"))
            # current_offset_display is current - expected in the measurement frame,
            # so correction direction is the opposite.
            dx = -float(offset.get("dx", 0.0))
            dy = -float(offset.get("dy", 0.0))
            return {
                "unit": unit,
                "screen_dx": dx,
                "screen_dy": dy,
                "source": "fallback_from_current_offset_display",
            }
        except Exception:
            pass

    measure = stab_info.get("current_notch_measure_mm")
    if isinstance(measure, dict):
        try:
            dx = -float(measure.get("dx", 0.0))
            dy = -float(measure.get("dy", 0.0))
            return {
                "unit": "mm",
                "screen_dx": dx,
                "screen_dy": dy,
                "source": "fallback_from_current_notch_measure_mm",
            }
        except Exception:
            pass

    measure = stab_info.get("current_notch_measure")
    if isinstance(measure, dict):
        try:
            dx = -float(measure.get("dx", 0.0))
            dy = -float(measure.get("dy", 0.0))
            return {
                "unit": "px",
                "screen_dx": dx,
                "screen_dy": dy,
                "source": "fallback_from_current_notch_measure",
            }
        except Exception:
            pass

    return None


def _expected_point_from_current_and_correction(stab_info: dict):
    """
    Reconstruct expected center if engine did not explicitly expose it.
    This restores the magenta target/cross on recovered mixed modules.
    """
    if not isinstance(stab_info, dict):
        return None

    expected_pt = _safe_cv_pt(stab_info.get("expected_baseplate_center_abs"))
    if expected_pt is not None:
        return expected_pt

    current_pt = _safe_cv_pt(stab_info.get("current_baseplate_center_abs"))
    if current_pt is None:
        return None

    corr = stab_info.get("baseplate_correction_vector")
    if isinstance(corr, dict):
        try:
            if "screen_dx_px" in corr and "screen_dy_px" in corr:
                return (
                    int(round(current_pt[0] + float(corr.get("screen_dx_px", 0.0)))),
                    int(round(current_pt[1] + float(corr.get("screen_dy_px", 0.0)))),
                )

            px_per_mm = corr.get("px_per_mm", None)
            unit = str(corr.get("unit", "px")).lower()
            sx = float(corr.get("screen_dx", 0.0))
            sy = float(corr.get("screen_dy", 0.0))

            if unit == "mm" and px_per_mm is not None:
                scale = float(px_per_mm)
                sx *= scale
                sy *= scale

            return (
                int(round(current_pt[0] + sx)),
                int(round(current_pt[1] + sy)),
            )
        except Exception:
            pass

    return None



def _extract_lines_and_anchors(stab_info: dict):
    nf = stab_info.get("notch_frame")
    if not isinstance(nf, dict):
        nf = stab_info.get("current_notch_frame")
    if not isinstance(nf, dict):
        nf = _as_dict(stab_info.get("notch_frame_runtime"))

    lines = {}
    anchors = {}

    for source in (
        nf.get("lines") if isinstance(nf, dict) else None,
        nf.get("fitted_lines") if isinstance(nf, dict) else None,
        stab_info.get("notch_lines"),
        stab_info.get("current_lines"),
        stab_info.get("lines"),
    ):
        if isinstance(source, dict):
            lines.update(source)

    for source in (
        nf.get("anchors") if isinstance(nf, dict) else None,
        nf.get("points") if isinstance(nf, dict) else None,
        stab_info.get("notch_anchors"),
        stab_info.get("anchors_current"),
        stab_info.get("anchors"),
    ):
        if isinstance(source, dict):
            anchors.update(source)

    for k in ("bottom_left", "bottom_right", "bottom_mid"):
        if k in stab_info:
            anchors[k] = stab_info[k]
        if isinstance(nf, dict) and k in nf:
            anchors[k] = nf[k]

    for k in ("left_line", "right_line", "bottom_line", "bottom_ref_line"):
        if k in stab_info:
            lines[k] = stab_info[k]
        if isinstance(nf, dict) and k in nf:
            lines[k] = nf[k]

    left = (
        _as_line(lines.get("left"))
        or _as_line(lines.get("left_wall"))
        or _as_line(lines.get("left_line"))
    )

    right = (
        _as_line(lines.get("right"))
        or _as_line(lines.get("right_wall"))
        or _as_line(lines.get("right_line"))
    )

    bottom = (
        _as_line(lines.get("bottom"))
        or _as_line(lines.get("bottom_ref"))
        or _as_line(lines.get("bottom_line"))
        or _as_line(lines.get("bottom_ref_line"))
    )

    return {
        "left_line": left,
        "right_line": right,
        "bottom_line": bottom,
        "bottom_left": anchors.get("bottom_left"),
        "bottom_right": anchors.get("bottom_right"),
        "bottom_mid": anchors.get("bottom_mid"),
    }


def _draw_expected_center_guidance(vis, stab_info: dict):
    """
    Draws:
      - expected/correct baseplate center target
      - current-to-expected correction arrow
      - bottom-right movement badge for the operator
    """
    if vis is None or not isinstance(stab_info, dict):
        return

    expected_pt = _expected_point_from_current_and_correction(stab_info)
    current_pt = _safe_cv_pt(stab_info.get("current_baseplate_center_abs"))
    corr = _fallback_correction_from_stab(stab_info)

    if expected_pt is None:
        return

    ex, ey = expected_pt

    # Expected/correct center marker: magenta/cyan bullseye.
    cv2.circle(vis, (ex, ey), 22, (255, 0, 255), 3, lineType=cv2.LINE_AA)
    cv2.circle(vis, (ex, ey), 13, (255, 255, 0), 2, lineType=cv2.LINE_AA)

    cv2.drawMarker(
        vis,
        (ex, ey),
        (255, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=44,
        thickness=3,
        line_type=cv2.LINE_AA,
    )

    _put_text(
        vis,
        "TARGET",
        (ex + 16, ey - 18),
        scale=0.50,
        color=(255, 0, 255),
        thickness=1,
    )

    if current_pt is None:
        return

    cx, cy = current_pt

    # Arrow from current detected center to expected/correct center.
    cv2.arrowedLine(
        vis,
        (cx, cy),
        (ex, ey),
        (255, 0, 255),
        4,
        line_type=cv2.LINE_AA,
        tipLength=0.25,
    )

    # Small current ring so operator sees start of correction vector.
    cv2.circle(vis, (cx, cy), 15, (0, 0, 255), 2, lineType=cv2.LINE_AA)


# ----------------------------
# Stabilizer / notch debug
# ----------------------------
def draw_stab_debug(vis, stab_info: dict, *, settings=None):
    if vis is None or not isinstance(stab_info, dict):
        return

    ok = bool(stab_info.get("ok", False))
    method = str(stab_info.get("current_anchor_method", stab_info.get("anchor_method", "unknown")))

    # Support both older and newer EngineSettings names.
    show_search = bool(_setting_any(settings, ("show_search_roi", "show_stab_search_roi"), True))
    show_contour = bool(_setting_any(settings, ("show_notch_contour", "show_stab_notch_contour"), True))
    show_lines = bool(_setting_any(settings, ("show_fitted_lines", "show_stab_fitted_lines"), True))
    show_anchors = bool(_setting_any(settings, ("show_new_anchors", "show_stab_anchors"), True))
    show_points = bool(_setting_any(settings, ("show_raw_points", "show_stab_feature_points"), False))
    show_legacy = bool(_setting_any(settings, ("show_legacy_debug", "show_stab_legacy"), False))
    show_text = bool(_setting_any(settings, ("show_stabilizer_text", "show_stab_text"), True))

    solid_dbg = _as_dict(stab_info.get("solid_edge_debug"))
    dark_dbg = _as_dict(stab_info.get("dark_region_debug"))
    legacy_dbg = _as_dict(stab_info.get("legacy_lines_debug"))
    current_dbg = _as_dict(stab_info.get("current_lines_debug"))

    active_dbg = current_dbg
    if dark_dbg:
        active_dbg = dark_dbg
    elif solid_dbg:
        active_dbg = solid_dbg

    # ----------------------------
    # Search ROI
    # ----------------------------
    if show_search and "search_roi_current" in stab_info:
        try:
            x, y, w, h = map(int, stab_info["search_roi_current"])
            col = (120, 120, 120) if ok else (0, 0, 255)

            cv2.rectangle(
                vis,
                (x, y),
                (x + w, y + h),
                col,
                1,
                lineType=cv2.LINE_AA,
            )

            _put_text(vis, "search", (x + 4, y + 16), scale=0.42, color=col)
        except Exception:
            pass

    # ----------------------------
    # Actual notch contour edge
    # ----------------------------
    if show_contour:
        _draw_polyline(vis, active_dbg.get("contour_abs"), color=(0, 220, 255), thickness=2, closed=False)
        _draw_polyline(vis, active_dbg.get("lower_contour_abs"), color=(0, 220, 255), thickness=2, closed=False)
        _draw_polyline(vis, active_dbg.get("hull_abs"), color=(0, 160, 255), thickness=1, closed=True)

        pts_used = _as_dict(active_dbg.get("points_used_abs"))
        _draw_points(vis, pts_used.get("contour"), color=(0, 220, 255), radius=1, step=12)
        _draw_points(vis, pts_used.get("lower_contour"), color=(0, 220, 255), radius=1, step=8)

    # ----------------------------
    # Optional raw points
    # ----------------------------
    if show_points:
        pts_used = _as_dict(active_dbg.get("points_used_abs"))

        _draw_points(vis, pts_used.get("all"), color=(80, 220, 255), radius=1, step=20)
        _draw_points(vis, pts_used.get("left"), color=(255, 190, 0), radius=1, step=8)
        _draw_points(vis, pts_used.get("right"), color=(255, 190, 0), radius=1, step=8)
        _draw_points(vis, pts_used.get("bottom"), color=(0, 140, 255), radius=1, step=8)

    # ----------------------------
    # Fitted notch lines
    # ----------------------------
    geom = _extract_lines_and_anchors(stab_info)

    if show_lines:
        _draw_fitline_clipped(vis, geom.get("left_line"), color=(0, 255, 0), thickness=2)
        _draw_fitline_clipped(vis, geom.get("right_line"), color=(0, 255, 0), thickness=2)
        _draw_fitline_clipped(vis, geom.get("bottom_line"), color=(0, 165, 255), thickness=2)

    # ----------------------------
    # Debug anchors only
    # ----------------------------
    if show_anchors:
        for name, color in (
            ("bottom_left", (0, 165, 255)),
            ("bottom_right", (0, 165, 255)),
            ("bottom_mid", (0, 0, 255)),
        ):
            pt = geom.get(name)
            p = _safe_cv_pt(pt)
            if p is None:
                continue

            x, y = p

            cv2.circle(vis, (x, y), 7, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(vis, (x, y), 9, (0, 0, 0), 1, lineType=cv2.LINE_AA)
            _put_text(vis, name, (x + 8, y - 8), scale=0.45, color=color)

        bl = _safe_cv_pt(geom.get("bottom_left"))
        br = _safe_cv_pt(geom.get("bottom_right"))

        if bl is not None and br is not None:
            cv2.line(vis, bl, br, (0, 165, 255), 2, lineType=cv2.LINE_AA)

        # New side-wall centerline debug.
        roi_dbg = _as_dict(stab_info.get("roi_frame_debug"))
        cdy = _safe_cv_pt(roi_dbg.get("center_at_dy"))
        ldy = _safe_cv_pt(roi_dbg.get("left_at_dy"))
        rdy = _safe_cv_pt(roi_dbg.get("right_at_dy"))

        if ldy is not None and rdy is not None:
            cv2.line(vis, ldy, rdy, (255, 180, 0), 1, lineType=cv2.LINE_AA)

        if cdy is not None:
            x, y = cdy
            cv2.circle(vis, (x, y), 6, (255, 180, 0), -1, lineType=cv2.LINE_AA)
            _put_text(vis, "side_center@dy", (x + 8, y - 8), scale=0.45, color=(255, 180, 0))

    # ----------------------------
    # Expected baseplate target + correction arrow
    # ----------------------------
    # This is independent from the debug anchor toggle because operators need
    # this even when raw debug clutter is hidden.
    _draw_expected_center_guidance(vis, stab_info)

    # ----------------------------
    # Legacy fallback debug
    # ----------------------------
    if show_legacy:
        legacy_pts = _as_dict(legacy_dbg.get("points_used_abs"))

        _draw_points(vis, legacy_pts.get("left"), color=(120, 80, 255), radius=1, step=14)
        _draw_points(vis, legacy_pts.get("right"), color=(120, 80, 255), radius=1, step=14)
        _draw_points(vis, legacy_pts.get("bottom"), color=(120, 80, 255), radius=1, step=14)

        legacy_lines = _as_dict(legacy_dbg.get("lines"))
        for _name, line in legacy_lines.items():
            _draw_fitline_clipped(vis, line, color=(180, 80, 180), thickness=1)

        _put_text(vis, "legacy debug", (16, 220), scale=0.5, color=(120, 80, 255))

    # ----------------------------
    # Text
    # ----------------------------
    if show_text:
        hyst = _as_dict(stab_info.get("hysteresis"))
        bp_hyst = _as_dict(stab_info.get("baseplate_hysteresis"))
        measure = _as_dict(stab_info.get("current_notch_measure"))
        measure_mm = _as_dict(stab_info.get("current_notch_measure_mm"))
        offset_display = _as_dict(stab_info.get("current_offset_display"))
        corr = _as_dict(stab_info.get("baseplate_correction_vector"))

        text_lines = [
            f"anchor: {method}",
            f"ROI: {stab_info.get('roi_mode', '-')}",
            f"M raw delta: {hyst.get('mode', '-')}",
        ]

        if "dtranslation_px" in hyst:
            try:
                text_lines.append(f"raw dT: {float(hyst.get('dtranslation_px', 0.0)):.1f}px")
            except Exception:
                pass

        if "dangle_deg" in hyst:
            try:
                text_lines.append(f"raw dA: {float(hyst.get('dangle_deg', 0.0)):.2f}deg")
            except Exception:
                pass

        if measure_mm:
            try:
                text_lines.append(f"offset dx: {float(measure_mm.get('dx', 0.0)):+.2f}mm")
                text_lines.append(f"offset dy: {float(measure_mm.get('dy', 0.0)):+.2f}mm")
            except Exception:
                pass
        elif offset_display and offset_display.get("unit") == "mm":
            try:
                text_lines.append(f"offset dx: {float(offset_display.get('dx', 0.0)):+.2f}mm")
                text_lines.append(f"offset dy: {float(offset_display.get('dy', 0.0)):+.2f}mm")
            except Exception:
                pass
        elif measure:
            try:
                text_lines.append(f"notch dx: {float(measure.get('dx', 0.0)):+.1f}px")
                text_lines.append(f"notch dy: {float(measure.get('dy', 0.0)):+.1f}px")
            except Exception:
                pass

        if measure:
            try:
                text_lines.append(f"rel theta: {float(measure.get('relative_angle', 0.0)):+.2f}deg")
            except Exception:
                pass

        if corr:
            try:
                unit = str(corr.get("unit", "px"))
                text_lines.append(
                    f"move: {float(corr.get('screen_dx', 0.0)):+.2f}{unit}, "
                    f"{float(corr.get('screen_dy', 0.0)):+.2f}{unit}"
                )
            except Exception:
                pass

        px_per_mm = stab_info.get("baseplate_px_per_mm_saved", stab_info.get("baseplate_px_per_mm_frame"))
        if px_per_mm is not None:
            try:
                text_lines.append(f"scale: {float(px_per_mm):.2f}px/mm")
            except Exception:
                pass

        if bp_hyst:
            text_lines.append(f"baseplate: {bp_hyst.get('mode', '-')}")

        x0, y0 = 16, 145
        for i, line in enumerate(text_lines):
            _put_text(vis, line, (x0, y0 + i * 18), scale=0.5, color=(255, 255, 255))


# ----------------------------
# Baseplate overlay
# ----------------------------
def draw_baseplate_overlay(vis, roi, center_rel, contour_rel, *, roi_poly=None):
    if vis is None or roi is None:
        return

    try:
        x, y, w, h = map(int, roi)
    except Exception:
        return

    cv2.rectangle(
        vis,
        (x, y),
        (x + w, y + h),
        (255, 255, 0),
        1,
        lineType=cv2.LINE_AA,
    )

    if roi_poly is not None:
        try:
            poly = np.asarray(roi_poly, dtype=np.int32).reshape(-1, 1, 2)
            if len(poly) >= 3:
                cv2.polylines(vis, [poly], True, (255, 255, 0), 2, lineType=cv2.LINE_AA)
                _put_text(vis, "rotated live ROI", tuple(poly[0, 0]), scale=0.45, color=(255, 255, 0))
        except Exception:
            _put_text(vis, "live ROI", (x + 4, y + 18), scale=0.5, color=(255, 255, 0))
    else:
        _put_text(vis, "live ROI", (x + 4, y + 18), scale=0.5, color=(255, 255, 0))

    if contour_rel is not None:
        try:
            cnt = np.asarray(contour_rel, dtype=np.int32)

            if cnt.size > 0:
                cnt = cnt.reshape(-1, 1, 2)
                cnt_abs = cnt + np.array([[[x, y]]], dtype=np.int32)
                cv2.drawContours(vis, [cnt_abs], -1, (0, 255, 0), 2, lineType=cv2.LINE_AA)
        except Exception:
            pass

    if center_rel is not None:
        try:
            cx = int(round(x + float(center_rel[0])))
            cy = int(round(y + float(center_rel[1])))

            cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(vis, (cx, cy), 9, (255, 255, 255), 1, lineType=cv2.LINE_AA)
            _put_text(vis, "baseplate", (cx + 9, cy - 8), scale=0.45, color=(0, 0, 255))
        except Exception:
            pass


# ----------------------------
# Status box
# ----------------------------
def draw_status_box(vis, text, state="FAIL", *, metrics=None, stability=None):
    """Draw a compact operator-facing result card, not a debug console."""
    if vis is None:
        return

    state = str(state or "FAIL").upper()
    state_style = {
        "PASS": ((46, 202, 99), "PASS"),
        "TRACK": ((40, 201, 231), "VERIFYING"),
        "SEARCH": ((155, 155, 155), "SEARCHING"),
        "SETUP": ((73, 174, 243), "SETUP REQUIRED"),
        "FAIL": ((70, 70, 235), "FAIL"),
    }
    color, state_label = state_style.get(state, state_style["FAIL"])

    frame_h, frame_w = vis.shape[:2]
    x, y = 18, 18
    available_w = max(1, frame_w - (2 * x))
    card_w = min(480, available_w)
    show_metrics = isinstance(metrics, (list, tuple)) and bool(metrics) and card_w >= 360
    card_h = 178 if show_metrics else 112
    card_h = min(card_h, max(1, frame_h - (2 * y)))
    right = x + card_w
    bottom = y + card_h

    # Dark surface, thin neutral outline, and a coloured state stripe give the
    # result hierarchy without turning the camera view into a debug dashboard.
    cv2.rectangle(vis, (x, y), (right, bottom), (24, 27, 34), -1, cv2.LINE_AA)
    cv2.rectangle(vis, (x, y), (right, bottom), (80, 85, 96), 1, cv2.LINE_AA)
    cv2.rectangle(vis, (x, y), (x + 7, bottom), color, -1, cv2.LINE_AA)

    _put_text(vis, "INSPECTION RESULT", (x + 22, y + 28), scale=0.45, color=(205, 210, 220), thickness=1)
    _put_text(vis, state_label, (x + 22, y + 73), scale=0.82, color=color, thickness=2)

    if show_metrics:
        metric_x = x + 172
        row_y = y + 55

        for index, metric in enumerate(metrics[:3]):
            if not isinstance(metric, dict):
                continue

            yy = row_y + index * 36
            label = str(metric.get("label", "MEASUREMENT"))
            value = str(metric.get("value", "—"))
            limit = str(metric.get("limit", ""))
            passed = bool(metric.get("passed", False))
            value_color = (46, 202, 99) if passed else (70, 70, 235)

            _put_text(vis, label, (metric_x, yy), scale=0.38, color=(185, 190, 200), thickness=1)
            _put_text(vis, value, (metric_x, yy + 17), scale=0.52, color=value_color, thickness=1)
            _put_text(vis, limit, (metric_x + 152, yy + 17), scale=0.38, color=(185, 190, 200), thickness=1)

        if isinstance(stability, (tuple, list)) and len(stability) >= 2:
            footer = f"Verification: {int(stability[0])} / {int(stability[1])} steady frames"
        else:
            footer = str(text)
        _put_text(vis, footer, (x + 22, bottom - 16), scale=0.42, color=(205, 210, 220), thickness=1)
    else:
        _put_text(vis, str(text), (x + 22, y + 101), scale=0.46, color=(235, 238, 245), thickness=1)
