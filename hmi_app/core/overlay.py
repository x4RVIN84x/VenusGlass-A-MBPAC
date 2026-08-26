from __future__ import annotations

import re
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


# Camera-HUD palette (OpenCV uses BGR).
_HUD_BG = (18, 22, 29)
_HUD_BG_ALT = (27, 33, 43)
_HUD_TEXT = (248, 250, 252)
_HUD_MUTED = (185, 196, 210)
_HUD_CYAN = (255, 205, 32)
_HUD_AMBER = (30, 190, 255)


def _hud_scale(vis) -> float:
    """Responsive scale tuned around the roughly 1050x590 Auto feed."""
    try:
        h, w = vis.shape[:2]
    except Exception:
        return 1.0
    return float(np.clip(min(w / 1050.0, h / 590.0), 0.78, 1.55))


def _rounded_panel(vis, rect, *, fill=_HUD_BG, border=(70, 82, 100), alpha=0.92, radius=12):
    """Draw a clipped translucent rounded panel without touching the text layer."""
    if vis is None or getattr(vis, "size", 0) == 0:
        return None

    h, w = vis.shape[:2]
    x0, y0, x1, y1 = map(int, rect)
    x0 = max(0, min(w - 1, x0))
    y0 = max(0, min(h - 1, y0))
    x1 = max(x0 + 1, min(w - 1, x1))
    y1 = max(y0 + 1, min(h - 1, y1))
    r = max(0, min(int(radius), (x1 - x0) // 2, (y1 - y0) // 2))

    overlay = vis.copy()
    if r <= 1:
        cv2.rectangle(overlay, (x0, y0), (x1, y1), fill, -1, cv2.LINE_AA)
    else:
        cv2.rectangle(overlay, (x0 + r, y0), (x1 - r, y1), fill, -1)
        cv2.rectangle(overlay, (x0, y0 + r), (x1, y1 - r), fill, -1)
        for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r),
                       (x0 + r, y1 - r), (x1 - r, y1 - r)):
            cv2.circle(overlay, (cx, cy), r, fill, -1, cv2.LINE_AA)

    cv2.addWeighted(overlay, float(alpha), vis, 1.0 - float(alpha), 0.0, vis)
    if border is not None:
        cv2.rectangle(vis, (x0, y0), (x1, y1), border, 1, cv2.LINE_AA)
    return x0, y0, x1, y1


def _draw_text_once(
    vis,
    text,
    org,
    *,
    scale=0.65,
    color=_HUD_TEXT,
    thickness=1,
    font=cv2.FONT_HERSHEY_SIMPLEX,
):
    """Render a single text layer; contrast comes from the HUD panel behind it."""
    cv2.putText(
        vis,
        str(text),
        (int(org[0]), int(org[1])),
        font,
        float(scale),
        color,
        max(1, int(thickness)),
        cv2.LINE_AA,
    )


def _fit_text(text: str, max_width: int, *, font, scale: float, thickness: int) -> str:
    """Ellipsize a label while retaining a useful minimum amount of context."""
    value = str(text)
    if max_width <= 0:
        return ""
    if cv2.getTextSize(value, font, scale, thickness)[0][0] <= max_width:
        return value

    suffix = "..."
    while len(value) > 4:
        value = value[:-1]
        candidate = value.rstrip() + suffix
        if cv2.getTextSize(candidate, font, scale, thickness)[0][0] <= max_width:
            return candidate
    return ""


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
    """Draw stable camera text on a compact high-contrast backing plate.

    The historical implementation sampled the live pixels every frame and
    alternated between two outlined glyph layers.  Reflections made that text
    visibly flicker.  Keeping this public name avoids breaking callers while
    the rendering itself is now deterministic and report-friendly.
    """
    if vis is None or getattr(vis, "size", 0) == 0:
        return

    x, y = int(org[0]), int(org[1])
    font = cv2.FONT_HERSHEY_SIMPLEX
    ui = _hud_scale(vis)
    scale = max(float(scale), 0.62) * max(1.0, ui)
    thickness = max(1, int(round(max(int(thickness), 1) * max(1.0, ui))))
    (tw, th), baseline = cv2.getTextSize(str(text), font, scale, thickness)
    pad_x = max(7, int(round(8 * ui)))
    pad_y = max(5, int(round(6 * ui)))
    h, w = vis.shape[:2]

    # Keep the entire label on the evidence frame even when its anchor sits
    # near a detected feature at an image edge.
    tx = max(pad_x, min(w - tw - pad_x, x))
    ty = max(th + pad_y, min(h - baseline - pad_y, y))
    _rounded_panel(
        vis,
        (tx - pad_x, ty - th - pad_y, tx + tw + pad_x, ty + baseline + pad_y),
        fill=_HUD_BG,
        border=(76, 91, 112),
        alpha=0.90,
        radius=max(5, int(round(7 * ui))),
    )
    _draw_text_once(
        vis,
        text,
        (tx, ty),
        scale=scale,
        color=_HUD_TEXT,
        thickness=thickness,
    )


def draw_cctv_footer(vis, *, timestamp: str, product_name: str, fps: Optional[float] = None):
    """Draw a large, persistent CCTV evidence bar along the full feed width."""
    if vis is None or getattr(vis, "size", 0) == 0:
        return

    H, W = vis.shape[:2]
    if H < 30 or W < 80:
        return

    ui = _hud_scale(vis)
    footer_h = max(54, int(round(62 * ui)))
    y0 = max(0, H - footer_h)
    _rounded_panel(
        vis,
        (0, y0, W - 1, H - 1),
        fill=(12, 16, 22),
        border=None,
        alpha=0.94,
        radius=0,
    )
    cv2.line(vis, (0, y0), (W - 1, y0), _HUD_CYAN, max(2, int(round(2 * ui))), cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    stamp = str(timestamp or "---- -- -- --:--:--")
    product = str(product_name or "NO PRODUCT")
    fps_text = "--.- FPS" if fps is None else f"{float(fps):.1f} FPS"
    margin = max(14, int(round(18 * ui)))
    camera_scale = 0.61 * ui
    stamp_scale = 0.75 * ui
    meta_scale = 0.58 * ui
    thick = max(1, int(round(2 * ui)))
    baseline_y = H - max(15, int(round(18 * ui)))

    camera_text = "CAM 01"
    camera_w = cv2.getTextSize(camera_text, font, camera_scale, thick)[0][0]
    stamp_w = cv2.getTextSize(stamp, font, stamp_scale, thick)[0][0]
    fps_w = cv2.getTextSize(fps_text, font, meta_scale, thick)[0][0]
    camera_x = margin
    stamp_x = camera_x + camera_w + max(22, int(round(28 * ui)))
    fps_x = W - margin - fps_w
    product_x = stamp_x + stamp_w + max(22, int(round(28 * ui)))
    product_space = fps_x - product_x - max(20, int(round(24 * ui)))
    product = _fit_text(product, product_space, font=font, scale=meta_scale, thickness=thick)

    _draw_text_once(vis, camera_text, (camera_x, baseline_y), scale=camera_scale,
                    color=_HUD_CYAN, thickness=thick)
    _draw_text_once(vis, stamp, (stamp_x, baseline_y), scale=stamp_scale,
                    color=_HUD_TEXT, thickness=thick)
    if product:
        _draw_text_once(vis, product, (product_x, baseline_y), scale=meta_scale,
                        color=_HUD_MUTED, thickness=thick)
    _draw_text_once(vis, fps_text, (fps_x, baseline_y), scale=meta_scale,
                    color=_HUD_TEXT, thickness=thick)


def draw_operator_measurement_hud(vis, stab_info: dict, *, org=(18, 122)):
    """Draw a large, stable measurement card intended for one-glance reading."""
    if vis is None or not isinstance(stab_info, dict):
        return

    offset = _as_dict(stab_info.get("current_offset_display"))
    measure = _as_dict(stab_info.get("current_notch_measure"))
    tolerance = _as_dict(stab_info.get("active_tolerance"))
    x_text = y_text = angle_text = None
    limits_text = None

    try:
        unit = str(offset.get("unit", "px"))
        dx = float(offset.get("dx"))
        dy = float(offset.get("dy"))
        x_text = f"X  {dx:+.2f} {unit}"
        y_text = f"Y  {dy:+.2f} {unit}"
    except Exception:
        pass

    try:
        dtheta = float(measure.get("dtheta"))
        angle_text = f"ANGLE  {dtheta:+.2f} deg"
    except Exception:
        pass

    try:
        angle_limit = float(tolerance.get("angle_deg"))
        x_mm = tolerance.get("x_mm")
        y_mm = tolerance.get("y_mm")
        if x_mm is not None and y_mm is not None:
            limits_text = (
                f"LIMITS   X +/- {float(x_mm):.2f} mm   "
                f"Y +/- {float(y_mm):.2f} mm   ANG +/- {angle_limit:.2f} deg"
            )
        else:
            limits_text = f"LIMITS   POSITION SCALE NEEDED   ANG +/- {angle_limit:.2f} deg"
    except Exception:
        pass

    # The acceptance limits are already permanently visible in the Auto-page
    # side card.  A large but otherwise empty camera card while SEARCHING made
    # the feed look unfinished, so only draw this when live placement exists.
    if not any((x_text, y_text, angle_text)):
        return

    H, W = vis.shape[:2]
    ui = _hud_scale(vis)
    x0 = max(12, int(org[0]))
    # Clear the result banner above it at every supported feed resolution.
    y0 = max(int(org[1]), int(round(112 * ui)))
    card_w = min(W - x0 - 14, max(450, int(round(550 * ui))))
    card_h = max(116, int(round(132 * ui)))
    if card_w < 260 or y0 >= H - 80:
        return
    y1 = min(H - 72, y0 + card_h)
    if y1 - y0 < 92:
        return

    _rounded_panel(
        vis,
        (x0, y0, x0 + card_w, y1),
        fill=_HUD_BG,
        border=(71, 86, 106),
        alpha=0.91,
        radius=max(8, int(round(12 * ui))),
    )
    cv2.rectangle(
        vis,
        (x0, y0),
        (x0 + max(6, int(round(7 * ui))), y1),
        _HUD_CYAN,
        -1,
        cv2.LINE_AA,
    )

    left = x0 + max(20, int(round(24 * ui)))
    header_y = y0 + max(21, int(round(25 * ui)))
    value_y = y0 + max(53, int(round(62 * ui)))
    angle_y = y0 + max(82, int(round(96 * ui)))
    limit_y = y1 - max(10, int(round(13 * ui)))
    label_scale = 0.46 * ui
    value_scale = 0.75 * ui
    angle_scale = 0.64 * ui
    limit_scale = 0.54 * ui
    value_thick = max(1, int(round(2 * ui)))

    _draw_text_once(vis, "LIVE POSITION", (left, header_y), scale=label_scale,
                    color=_HUD_CYAN, thickness=value_thick)
    position_parts = [part for part in (x_text, y_text) if part]
    if position_parts:
        _draw_text_once(vis, "       |       ".join(position_parts), (left, value_y),
                        scale=value_scale, color=_HUD_TEXT, thickness=value_thick)
    if angle_text:
        _draw_text_once(vis, angle_text, (left, angle_y), scale=angle_scale,
                        color=_HUD_TEXT, thickness=value_thick)
    if limits_text:
        fitted = _fit_text(limits_text, card_w - (left - x0) - 14,
                           font=cv2.FONT_HERSHEY_SIMPLEX, scale=limit_scale,
                           thickness=max(1, value_thick - 1))
        _draw_text_once(vis, fitted, (left, limit_y), scale=limit_scale,
                        color=_HUD_MUTED, thickness=max(1, value_thick - 1))


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

    ui = _hud_scale(vis)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.76 * ui
    thick = max(1, int(round(2 * ui)))
    pad_x = max(20, int(round(24 * ui)))
    footer_clearance = max(70, int(round(78 * ui)))
    title_h = max(24, int(round(27 * ui)))
    line_gap = max(31, int(round(35 * ui)))

    sizes = [cv2.getTextSize(str(t), font, scale, thick)[0] for t in lines]
    text_w = max((s[0] for s in sizes), default=220)
    box_w = int(min(max(text_w + 2 * pad_x, 330 * ui), W - 36))
    box_h = int(max(92 * ui, title_h + len(lines) * line_gap + 24 * ui))
    margin_r = max(16, int(round(20 * ui)))
    margin_b = footer_clearance

    x0 = max(14, W - box_w - margin_r)
    y0 = max(126, H - box_h - margin_b)
    x1 = min(W - 14, x0 + box_w)
    y1 = min(H - footer_clearance + 4, y0 + box_h)
    accent = (55, 215, 95) if lines == ["CENTERED"] else _HUD_AMBER

    _rounded_panel(
        vis,
        (x0, y0, x1, y1),
        fill=_HUD_BG,
        border=accent,
        alpha=0.94,
        radius=max(8, int(round(12 * ui))),
    )
    cv2.rectangle(vis, (x0, y0), (x0 + max(7, int(round(9 * ui))), y1),
                  accent, -1, cv2.LINE_AA)

    _draw_text_once(
        vis,
        "OPERATOR ADJUSTMENT",
        (x0 + pad_x, y0 + title_h),
        scale=0.43 * ui,
        color=accent,
        thickness=max(1, thick - 1),
    )

    first_y = y0 + title_h + max(29, int(round(34 * ui)))
    for i, line in enumerate(lines):
        text_value = str(line)
        tw = cv2.getTextSize(text_value, font, scale, thick)[0][0]
        tx = max(x0 + pad_x, int(round((x0 + x1) * 0.5 - tw * 0.5)))
        ty = int(round(first_y + i * line_gap))
        _draw_text_once(vis, text_value, (tx, ty), scale=scale,
                        color=_HUD_TEXT, thickness=thick)


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
    """Draw the primary inspection result as a large, stable operator banner."""
    if vis is None or getattr(vis, "size", 0) == 0:
        return

    state_text = str(state or "FAIL").upper()
    if state_text == "PASS":
        color = (60, 215, 92)
    elif state_text == "TRACK":
        color = _HUD_AMBER
    elif state_text == "SEARCH":
        color = (180, 190, 205)
    else:
        color = (60, 75, 238)

    H, W = vis.shape[:2]
    if H < 60 or W < 150:
        return

    detail = str(text or "").strip()

    if state_text == "SEARCH" and detail.upper() == "SEARCHING":
        detail = ""
    elif detail.upper().startswith(state_text):
        detail = detail[len(state_text):].lstrip(" :|-" )

    detail = re.sub(r"\bdx\s*=\s*", "X  ", detail, flags=re.IGNORECASE)
    detail = re.sub(r"\bdy\s*=\s*", "Y  ", detail, flags=re.IGNORECASE)
    detail = re.sub(r"\bdtheta\s*=\s*", "ANGLE  ", detail, flags=re.IGNORECASE)
    detail = re.sub(r"(?<=\d)(mm|px)\b", r" \1", detail, flags=re.IGNORECASE)
    detail = re.sub(r"\s+", " ", detail).strip()
    detail = re.sub(r"\s+Y\s+", "   |   Y  ", detail, flags=re.IGNORECASE)
    detail = re.sub(r"\s+ANGLE\s+", "   |   ANGLE  ", detail, flags=re.IGNORECASE)

    ui = _hud_scale(vis)
    font = cv2.FONT_HERSHEY_SIMPLEX
    state_scale = 1.02 * ui
    detail_scale = 0.69 * ui
    state_thick = max(2, int(round(2 * ui)))
    detail_thick = max(1, int(round(2 * ui)))
    box_h = max(76, int(round(86 * ui)))
    x0 = max(12, int(round(16 * ui)))
    y0 = max(12, int(round(16 * ui)))
    dot_x = x0 + max(25, int(round(29 * ui)))
    state_x = x0 + max(46, int(round(54 * ui)))
    header_y = y0 + max(19, int(round(21 * ui)))
    baseline_y = y0 + max(56, int(round(62 * ui)))
    state_w = cv2.getTextSize(state_text, font, state_scale, state_thick)[0][0]
    detail_x = state_x + state_w + max(23, int(round(30 * ui)))
    max_detail_w = max(40, W - x0 - detail_x - max(28, int(round(36 * ui))))
    detail = _fit_text(detail, max_detail_w, font=font, scale=detail_scale,
                       thickness=detail_thick)
    detail_w = cv2.getTextSize(detail, font, detail_scale, detail_thick)[0][0]
    desired_w = detail_x - x0 + detail_w + max(23, int(round(28 * ui)))
    box_w = min(W - x0 - 12, max(int(round(470 * ui)), desired_w))
    x1, y1 = x0 + box_w, y0 + box_h

    _rounded_panel(
        vis,
        (x0, y0, x1, y1),
        fill=(13, 17, 23),
        border=color,
        alpha=0.95,
        radius=max(9, int(round(13 * ui))),
    )
    cv2.rectangle(vis, (x0, y0), (x0 + max(7, int(round(9 * ui))), y1),
                  color, -1, cv2.LINE_AA)
    cv2.circle(vis, (dot_x, baseline_y - max(8, int(round(9 * ui)))),
               max(8, int(round(9 * ui))), color, -1, cv2.LINE_AA)

    _draw_text_once(vis, "INSPECTION RESULT", (state_x, header_y),
                    scale=0.39 * ui, color=_HUD_MUTED,
                    thickness=max(1, state_thick - 1))
    _draw_text_once(vis, state_text, (state_x, baseline_y), scale=state_scale,
                    color=_HUD_TEXT, thickness=state_thick)
    if detail:
        _draw_text_once(vis, detail, (detail_x, baseline_y - max(2, int(round(3 * ui)))),
                        scale=detail_scale, color=_HUD_TEXT, thickness=detail_thick)
