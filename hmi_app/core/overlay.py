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


def _text_background_luma(vis, text, org, *, scale=0.5, thickness=1):
    """Estimate the luminance directly behind an OpenCV text glyph."""
    if vis is None or getattr(vis, "size", 0) == 0:
        return 0.0

    try:
        x, y = int(org[0]), int(org[1])
        (tw, th), baseline = cv2.getTextSize(
            str(text), cv2.FONT_HERSHEY_SIMPLEX, float(scale), int(thickness)
        )
    except Exception:
        return 0.0

    pad = max(3, int(thickness) + 2)
    H, W = vis.shape[:2]
    x0 = max(0, min(W - 1, x - pad))
    y0 = max(0, min(H - 1, y - th - pad))
    x1 = max(x0 + 1, min(W, x + tw + pad))
    y1 = max(y0 + 1, min(H, y + baseline + pad))

    patch = vis[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0

    try:
        if patch.ndim == 3:
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        else:
            gray = patch
        return float(np.median(gray))
    except Exception:
        return 0.0


def draw_adaptive_text(
    vis,
    text,
    org,
    *,
    scale=0.5,
    thickness=1,
):
    """Draw camera-style text that flips between black and white by background.

    A thin opposite-colour halo remains in both modes, which keeps characters
    legible over mixed glass, reflections, and coloured inspection overlays.
    """
    if vis is None:
        return

    luma = _text_background_luma(vis, text, org, scale=scale, thickness=thickness)
    foreground = (8, 8, 8) if luma >= 145.0 else (255, 255, 255)
    outline = (255, 255, 255) if luma >= 145.0 else (0, 0, 0)

    x, y = int(org[0]), int(org[1])
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(
        vis,
        str(text),
        (x, y),
        font,
        float(scale),
        outline,
        max(int(thickness) + 2, 2),
        cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        str(text),
        (x, y),
        font,
        float(scale),
        foreground,
        int(thickness),
        cv2.LINE_AA,
    )


def draw_cctv_footer(vis, *, timestamp: str, product_name: str, fps: Optional[float] = None):
    """Draw persistent, report-friendly camera metadata at the bottom left."""
    if vis is None or getattr(vis, "size", 0) == 0:
        return

    H, W = vis.shape[:2]
    if H < 30 or W < 80:
        return

    stamp = str(timestamp or "---- -- -- --:--:--")
    product = str(product_name or "NO PRODUCT")
    fps_text = "--.- FPS" if fps is None else f"{float(fps):.1f} FPS"
    text = f"CAM 01 | {stamp} | {product} | {fps_text}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    available_w = max(40, W - 32)
    scale = 0.55
    (tw, _th), _base = cv2.getTextSize(text, font, scale, 1)

    if tw > available_w:
        scale = max(0.38, scale * available_w / max(1, tw))
        (tw, _th), _base = cv2.getTextSize(text, font, scale, 1)

    if tw > available_w:
        # Keep timestamp and FPS intact; shorten only the product identifier.
        product = product[: max(8, min(len(product), 16))]
        text = f"CAM 01 | {stamp} | {product} | {fps_text}"

    draw_adaptive_text(
        vis,
        text,
        (16, max(22, H - 16)),
        scale=scale,
        thickness=1,
    )


def draw_operator_measurement_hud(vis, stab_info: dict, *, org=(18, 122)):
    """Draw only the three values an operator needs to make a decision."""
    if vis is None or not isinstance(stab_info, dict):
        return

    offset = _as_dict(stab_info.get("current_offset_display"))
    measure = _as_dict(stab_info.get("current_notch_measure"))
    tolerance = _as_dict(stab_info.get("active_tolerance"))
    lines = []

    try:
        unit = str(offset.get("unit", "px"))
        dx = float(offset.get("dx"))
        dy = float(offset.get("dy"))
        lines.append(f"OFFSET  X {dx:+.2f}{unit}   Y {dy:+.2f}{unit}")
    except Exception:
        pass

    try:
        dtheta = float(measure.get("dtheta"))
        lines.append(f"ANGLE   {dtheta:+.2f} deg")
    except Exception:
        pass

    try:
        angle_limit = float(tolerance.get("angle_deg"))
        x_mm = tolerance.get("x_mm")
        y_mm = tolerance.get("y_mm")
        if x_mm is not None and y_mm is not None:
            lines.append(
                f"LIMITS  X/Y +/- {float(x_mm):.2f}mm   ANG +/- {angle_limit:.2f}deg"
            )
        else:
            lines.append(f"LIMITS  X/Y need scale   ANG +/- {angle_limit:.2f}deg")
    except Exception:
        pass

    if not lines:
        return

    H, W = vis.shape[:2]
    scale = 0.55 if W >= 1100 else 0.46
    x = max(12, int(org[0]))
    y = max(30, int(org[1]))

    for i, line in enumerate(lines[:3]):
        draw_adaptive_text(
            vis,
            line,
            (x, min(H - 20, y + i * 24)),
            scale=scale,
            thickness=1,
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

    # Always try to draw the movement badge if we can derive the correction.
    if corr is not None:
        _draw_movement_badge_bottom_right(vis, corr)

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

    draw_adaptive_text(vis, "TARGET", (ex + 16, ey - 18), scale=0.50, thickness=1)

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
    # Support both older and newer EngineSettings names.
    show_search = bool(_setting_any(settings, ("show_search_roi", "show_stab_search_roi"), True))
    show_contour = bool(_setting_any(settings, ("show_notch_contour", "show_stab_notch_contour"), True))
    show_lines = bool(_setting_any(settings, ("show_fitted_lines", "show_stab_fitted_lines"), True))
    show_anchors = bool(_setting_any(settings, ("show_new_anchors", "show_stab_anchors"), True))
    show_points = bool(_setting_any(settings, ("show_raw_points", "show_stab_feature_points"), False))
    show_legacy = bool(_setting_any(settings, ("show_legacy_debug", "show_stab_legacy"), False))
    show_text = bool(_setting_any(settings, ("show_stabilizer_text", "show_stab_text"), True))
    defer_operator_hud = bool(_setting(settings, "defer_operator_hud", False))

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

    # ----------------------------
    # Expected baseplate target + correction arrow
    # ----------------------------
    # Auto Mode redraws this after display zoom, so it is never cropped and is
    # not duplicated. Calibration/Manual overlays continue to draw it here.
    if not defer_operator_hud:
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
    if show_text and not defer_operator_hud:
        draw_operator_measurement_hud(vis, stab_info)


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
        except Exception:
            pass

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
        except Exception:
            pass


# ----------------------------
# Status box
# ----------------------------
def draw_status_box(vis, text, state="FAIL"):
    if vis is None:
        return

    if state == "PASS":
        color = (0, 200, 0)
    elif state == "TRACK":
        color = (0, 200, 200)
    elif state == "SEARCH":
        color = (180, 180, 180)
    else:
        color = (0, 0, 255)

    H, W = vis.shape[:2]
    if H < 60 or W < 150:
        return

    state_text = str(state or "FAIL").upper()
    detail = str(text or "").strip()

    if detail.upper().startswith(state_text):
        detail = detail[len(state_text):].lstrip(" :|-" )

    detail = (
        detail.replace("dx=", "X ")
        .replace("dy=", "Y ")
        .replace("dTheta=", "A ")
        .replace("dtheta=", "A ")
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    state_scale = 0.72
    detail_scale = 0.50
    state_thick = 2
    detail_thick = 1

    (state_w, _state_h), _ = cv2.getTextSize(state_text, font, state_scale, state_thick)
    max_detail_w = max(40, W - 16 - 18 - 22 - state_w - 18)

    detail_was_trimmed = False
    while detail:
        (detail_w, _detail_h), _ = cv2.getTextSize(detail, font, detail_scale, detail_thick)
        if detail_w <= max_detail_w:
            break
        detail = detail[:-2].rstrip()
        detail_was_trimmed = True

    if detail and detail_was_trimmed:
        detail = detail.rstrip(". ") + "..."

    (detail_w, _detail_h), _ = cv2.getTextSize(detail, font, detail_scale, detail_thick)
    # The detail starts after the status word, not after the dot.  Size the box
    # from that real text origin so the final angle value is never clipped.
    box_w = min(W - 32, max(230, 48 + state_w + 14 + detail_w + 16))
    box_h = 62
    x0, y0 = 16, 16
    x1, y1 = x0 + box_w, y0 + box_h

    cv2.rectangle(vis, (x0, y0), (x1, y1), (16, 16, 16), -1, lineType=cv2.LINE_AA)
    cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2, lineType=cv2.LINE_AA)
    cv2.rectangle(vis, (x0, y0), (x0 + 7, y1), color, -1, lineType=cv2.LINE_AA)
    cv2.circle(vis, (x0 + 24, y0 + box_h // 2), 8, color, -1, cv2.LINE_AA)

    cv2.putText(
        vis,
        state_text,
        (x0 + 40, y0 + 39),
        font,
        state_scale,
        (255, 255, 255),
        state_thick,
        cv2.LINE_AA,
    )

    if detail:
        cv2.putText(
            vis,
            detail,
            (x0 + 48 + state_w, y0 + 38),
            font,
            detail_scale,
            (255, 255, 255),
            detail_thick,
            cv2.LINE_AA,
        )
