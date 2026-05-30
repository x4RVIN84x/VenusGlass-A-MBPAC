from __future__ import annotations

import time
from typing import Optional, Tuple, Any, Dict

import cv2
import numpy as np

import roi_stablizer
from detector import detect_baseplate

from hmi_app.core.models import Recipe, EngineSettings, QCFrameOutput
from hmi_app.core.overlay import draw_stab_debug, draw_baseplate_overlay, draw_status_box


# ----------------------------
# Basic ROI helpers
# ----------------------------
def clamp_roi(roi, W, H):
    x, y, w, h = map(int, roi)

    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))

    return (x, y, w, h)


def safe_crop(img, roi):
    roi = clamp_roi(roi, img.shape[1], img.shape[0])
    x, y, w, h = roi
    return img[y:y + h, x:x + w].copy(), roi


def _angle_wrap_deg(a: float) -> float:
    a = float(a)

    while a > 180.0:
        a -= 360.0

    while a < -180.0:
        a += 360.0

    return a


def _angle_diff_deg(a: float, b: float) -> float:
    return _angle_wrap_deg(float(a) - float(b))


def _to_bgr(gray_or_edges: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if gray_or_edges is None:
        return None

    if not isinstance(gray_or_edges, np.ndarray):
        return None

    if gray_or_edges.size == 0:
        return None

    if gray_or_edges.ndim == 2:
        return cv2.cvtColor(gray_or_edges, cv2.COLOR_GRAY2BGR)

    if gray_or_edges.ndim == 3 and gray_or_edges.shape[2] == 3:
        return gray_or_edges.copy()

    if gray_or_edges.ndim == 3 and gray_or_edges.shape[2] == 4:
        return cv2.cvtColor(gray_or_edges, cv2.COLOR_BGRA2BGR)

    return None


# ----------------------------
# Geometry helpers
# ----------------------------
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


def _as_pt(pt):
    if pt is None:
        return None

    try:
        x, y = float(pt[0]), float(pt[1])
    except Exception:
        return None

    if not np.isfinite(x) or not np.isfinite(y):
        return None

    return (x, y)


def _normalize_vec(vx: float, vy: float):
    n = float(np.hypot(vx, vy))

    if n < 1e-9:
        return None

    return (float(vx / n), float(vy / n))


def _line_intersection(l1, l2):
    l1 = _as_line(l1)
    l2 = _as_line(l2)

    if l1 is None or l2 is None:
        return None

    vx1, vy1, x1, y1 = l1
    vx2, vy2, x2, y2 = l2

    A = np.array([[vx1, -vx2], [vy1, -vy2]], dtype=np.float64)
    b = np.array([x2 - x1, y2 - y1], dtype=np.float64)

    det = float(np.linalg.det(A))
    if abs(det) < 1e-9:
        return None

    t, _u = np.linalg.solve(A, b)
    return (float(x1 + t * vx1), float(y1 + t * vy1))


def _parallel_line_through_point(line, point):
    line = _as_line(line)
    point = _as_pt(point)

    if line is None or point is None:
        return None

    vx, vy, _x0, _y0 = line
    px, py = point

    return (float(vx), float(vy), float(px), float(py))


def _rotated_rect_poly(center, w: float, h: float, angle_deg: float):
    c = _as_pt(center)

    if c is None:
        return None

    cx, cy = c
    a = np.deg2rad(float(angle_deg))

    ux, uy = float(np.cos(a)), float(np.sin(a))
    vx, vy = -uy, ux

    hw = 0.5 * float(w)
    hh = 0.5 * float(h)

    pts = np.array(
        [
            [cx - hw * ux - hh * vx, cy - hw * uy - hh * vy],
            [cx + hw * ux - hh * vx, cy + hw * uy - hh * vy],
            [cx + hw * ux + hh * vx, cy + hw * uy + hh * vy],
            [cx - hw * ux + hh * vx, cy - hw * uy + hh * vy],
        ],
        dtype=np.float32,
    )

    return pts


def _bbox_from_poly(poly, W: int, H: int, pad: int = 0):
    if poly is None:
        return None

    arr = np.asarray(poly, dtype=np.float32).reshape(-1, 2)

    x0 = int(np.floor(float(np.min(arr[:, 0])))) - int(pad)
    y0 = int(np.floor(float(np.min(arr[:, 1])))) - int(pad)
    x1 = int(np.ceil(float(np.max(arr[:, 0])))) + int(pad)
    y1 = int(np.ceil(float(np.max(arr[:, 1])))) + int(pad)

    x0 = max(0, min(x0, W - 1))
    y0 = max(0, min(y0, H - 1))
    x1 = max(x0 + 1, min(x1, W))
    y1 = max(y0 + 1, min(y1, H))

    return (x0, y0, x1 - x0, y1 - y0)


# ----------------------------
# mm conversion helpers
# ----------------------------
def _baseplate_dims_mm_from_cfg(cfg: Dict[str, Any]):
    dims = cfg.get("baseplate_dimensions_mm")

    if isinstance(dims, dict):
        w = dims.get("width", dims.get("w", dims.get("W")))
        h = dims.get("height", dims.get("h", dims.get("H")))
    else:
        w = cfg.get("baseplate_width_mm", 20.4)
        h = cfg.get("baseplate_height_mm", 26.5)

    try:
        w = float(w)
        h = float(h)
    except Exception:
        return None

    if not np.isfinite(w) or not np.isfinite(h) or w <= 0 or h <= 0:
        return None

    return (w, h)


def _estimate_px_per_mm_from_contour(contour_abs, cfg: Dict[str, Any]):
    dims = _baseplate_dims_mm_from_cfg(cfg)

    if dims is None or contour_abs is None:
        return None

    try:
        cnt = np.asarray(contour_abs, dtype=np.float32).reshape(-1, 2)
    except Exception:
        return None

    if len(cnt) < 5:
        return None

    try:
        rect = cv2.minAreaRect(cnt)
        w_px, h_px = rect[1]
    except Exception:
        return None

    w_px = float(w_px)
    h_px = float(h_px)

    if not np.isfinite(w_px) or not np.isfinite(h_px) or w_px <= 1 or h_px <= 1:
        return None

    w_mm, h_mm = dims

    px_sorted = sorted([w_px, h_px])
    mm_sorted = sorted([w_mm, h_mm])

    r1 = px_sorted[0] / mm_sorted[0]
    r2 = px_sorted[1] / mm_sorted[1]

    if not np.isfinite(r1) or not np.isfinite(r2) or r1 <= 0 or r2 <= 0:
        return None

    ratio_disagreement = abs(r1 - r2) / max(1e-6, max(r1, r2))
    px_per_mm = 0.5 * (r1 + r2)

    return {
        "px_per_mm": float(px_per_mm),
        "mm_per_px": float(1.0 / px_per_mm),
        "rect_w_px": float(w_px),
        "rect_h_px": float(h_px),
        "baseplate_width_mm": float(w_mm),
        "baseplate_height_mm": float(h_mm),
        "ratio_disagreement": float(ratio_disagreement),
    }


def _saved_px_per_mm(cfg: Dict[str, Any]):
    for key in ("px_per_mm", "baseplate_px_per_mm"):
        try:
            v = float(cfg.get(key))
            if np.isfinite(v) and v > 0:
                return v
        except Exception:
            pass

    try:
        mm_per_px = float(cfg.get("mm_per_px"))
        if np.isfinite(mm_per_px) and mm_per_px > 0:
            return 1.0 / mm_per_px
    except Exception:
        pass

    scale = cfg.get("baseplate_scale")
    if isinstance(scale, dict):
        try:
            v = float(scale.get("px_per_mm"))
            if np.isfinite(v) and v > 0:
                return v
        except Exception:
            pass

    return None


def _px_to_display(dx_px, dy_px, cfg: Dict[str, Any], frame_scale_info=None):
    px_per_mm = _saved_px_per_mm(cfg)

    if px_per_mm is None and isinstance(frame_scale_info, dict):
        try:
            v = float(frame_scale_info.get("px_per_mm"))
            if np.isfinite(v) and v > 0:
                px_per_mm = v
        except Exception:
            pass

    if px_per_mm is None or px_per_mm <= 0:
        return {
            "unit": "px",
            "dx": float(dx_px),
            "dy": float(dy_px),
            "px_per_mm": None,
        }

    return {
        "unit": "mm",
        "dx": float(dx_px) / float(px_per_mm),
        "dy": float(dy_px) / float(px_per_mm),
        "px_per_mm": float(px_per_mm),
    }


# ----------------------------
# Notch frame extraction / local coordinates
# ----------------------------
def _extract_notch_frame(stab_info: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(stab_info, dict) or not stab_info.get("ok", False):
        return None

    nf = stab_info.get("notch_frame")
    if not isinstance(nf, dict):
        nf = stab_info.get("current_notch_frame")
    if not isinstance(nf, dict):
        nf = stab_info.get("notch_frame_runtime")
    if not isinstance(nf, dict):
        nf = {}

    lines = {}
    for source in (
        nf.get("lines"),
        nf.get("fitted_lines"),
        stab_info.get("notch_lines"),
        stab_info.get("current_lines"),
        stab_info.get("lines"),
    ):
        if isinstance(source, dict):
            lines.update(source)

    anchors = {}
    for source in (
        nf.get("anchors"),
        nf.get("points"),
        stab_info.get("notch_anchors"),
        stab_info.get("anchors_current"),
        stab_info.get("anchors"),
    ):
        if isinstance(source, dict):
            anchors.update(source)

    bottom_line = (
        _as_line(nf.get("bottom_line"))
        or _as_line(nf.get("bottom_ref_line"))
        or _as_line(lines.get("bottom"))
        or _as_line(lines.get("bottom_ref"))
        or _as_line(lines.get("bottom_line"))
        or _as_line(lines.get("bottom_ref_line"))
        or _as_line(stab_info.get("bottom_line"))
        or _as_line(stab_info.get("bottom_ref_line"))
    )

    left_line = (
        _as_line(nf.get("left_line"))
        or _as_line(lines.get("left"))
        or _as_line(lines.get("left_wall"))
        or _as_line(lines.get("left_line"))
        or _as_line(stab_info.get("left_line"))
    )

    right_line = (
        _as_line(nf.get("right_line"))
        or _as_line(lines.get("right"))
        or _as_line(lines.get("right_wall"))
        or _as_line(lines.get("right_line"))
        or _as_line(stab_info.get("right_line"))
    )

    bottom_left = (
        _as_pt(nf.get("bottom_left"))
        or _as_pt(anchors.get("bottom_left"))
        or _as_pt(stab_info.get("bottom_left"))
    )

    bottom_right = (
        _as_pt(nf.get("bottom_right"))
        or _as_pt(anchors.get("bottom_right"))
        or _as_pt(stab_info.get("bottom_right"))
    )

    if bottom_line is None or left_line is None or right_line is None:
        return None

    if bottom_left is None:
        bottom_left = _line_intersection(left_line, bottom_line)

    if bottom_right is None:
        bottom_right = _line_intersection(right_line, bottom_line)

    if bottom_left is not None and bottom_right is not None:
        bottom_mid = (
            0.5 * (bottom_left[0] + bottom_right[0]),
            0.5 * (bottom_left[1] + bottom_right[1]),
        )
    else:
        bottom_mid = _as_pt(nf.get("bottom_mid")) or _as_pt(stab_info.get("bottom_mid"))

    if bottom_mid is None:
        bottom_mid = (float(bottom_line[2]), float(bottom_line[3]))

    vx, vy, _x0, _y0 = bottom_line
    u = _normalize_vec(vx, vy)

    if u is None:
        return None

    ux, uy = u

    if ux < 0:
        ux, uy = -ux, -uy

    nx, ny = uy, -ux
    n = _normalize_vec(nx, ny)

    if n is None:
        return None

    nx, ny = n

    angle_deg = float(np.degrees(np.arctan2(uy, ux)))

    return {
        "bottom_line": bottom_line,
        "left_line": left_line,
        "right_line": right_line,
        "bottom_left": bottom_left,
        "bottom_right": bottom_right,
        "bottom_mid": bottom_mid,
        "bottom_mid_source": "midpoint_bottom_left_bottom_right",
        "u": (float(ux), float(uy)),
        "n_up": (float(nx), float(ny)),
        "angle_deg": float(angle_deg),
    }


def _point_at_notch_height(notch_frame: Dict[str, Any], dy: float, dx: float = 0.0):
    if not isinstance(notch_frame, dict):
        return None

    bottom_line = _as_line(notch_frame.get("bottom_line"))
    left_line = _as_line(notch_frame.get("left_line"))
    right_line = _as_line(notch_frame.get("right_line"))

    u = notch_frame.get("u")
    n_up = notch_frame.get("n_up")

    if bottom_line is None or left_line is None or right_line is None:
        return None

    if u is None or n_up is None:
        return None

    ux, uy = float(u[0]), float(u[1])
    nx, ny = float(n_up[0]), float(n_up[1])

    _vx, _vy, x0, y0 = bottom_line
    shifted_pt = (
        float(x0 + float(dy) * nx),
        float(y0 + float(dy) * ny),
    )

    shifted_line = _parallel_line_through_point(bottom_line, shifted_pt)

    if shifted_line is None:
        return None

    left_pt = _line_intersection(left_line, shifted_line)
    right_pt = _line_intersection(right_line, shifted_line)

    if left_pt is None or right_pt is None:
        return None

    center = (
        0.5 * (left_pt[0] + right_pt[0]),
        0.5 * (left_pt[1] + right_pt[1]),
    )

    out = (
        float(center[0] + float(dx) * ux),
        float(center[1] + float(dx) * uy),
    )

    return {
        "point": out,
        "center_at_dy": center,
        "left_at_dy": left_pt,
        "right_at_dy": right_pt,
        "shifted_line": shifted_line,
    }


def _measure_in_notch_frame(point_abs, angle_deg, notch_frame: Dict[str, Any]) -> Optional[Dict[str, float]]:
    p = _as_pt(point_abs)

    if p is None or notch_frame is None:
        return None

    bottom_line = _as_line(notch_frame.get("bottom_line"))
    u = notch_frame.get("u")
    n_up = notch_frame.get("n_up")
    bottom_mid = _as_pt(notch_frame.get("bottom_mid"))

    if bottom_line is None or u is None or n_up is None or bottom_mid is None:
        return None

    px, py = p
    ux, uy = float(u[0]), float(u[1])
    nx, ny = float(n_up[0]), float(n_up[1])
    bx, by = bottom_mid

    frame_dx = (px - bx) * ux + (py - by) * uy
    frame_dy = (px - bx) * nx + (py - by) * ny

    notch_angle = float(notch_frame.get("angle_deg", 0.0))
    rel_angle = _angle_diff_deg(float(angle_deg), notch_angle)

    sidewall_dx = None
    sidewall_dy = None
    sidewall_center_at_dy = None
    sidewall_left_at_dy = None
    sidewall_right_at_dy = None

    center_data = _point_at_notch_height(notch_frame, dy=frame_dy, dx=0.0)

    if center_data is not None:
        sidewall_center_at_dy = center_data.get("center_at_dy")
        sidewall_left_at_dy = center_data.get("left_at_dy")
        sidewall_right_at_dy = center_data.get("right_at_dy")

        if sidewall_center_at_dy is not None:
            cx, cy = sidewall_center_at_dy
            sidewall_dx = (px - cx) * ux + (py - cy) * uy
            sidewall_dy = frame_dy

    out = {
        "coord_model": "bottom_mid_local_frame",
        "dx": float(frame_dx),
        "dy": float(frame_dy),
        "frame_dx": float(frame_dx),
        "frame_dy": float(frame_dy),
        "relative_angle": float(rel_angle),
        "notch_angle": float(notch_angle),
        "bottom_mid": [float(bx), float(by)],
        "x_axis": [float(ux), float(uy)],
        "y_axis": [float(nx), float(ny)],
    }

    if sidewall_dx is not None and sidewall_dy is not None:
        out["sidewall_dx"] = float(sidewall_dx)
        out["sidewall_dy"] = float(sidewall_dy)

    if sidewall_center_at_dy is not None:
        out["sidewall_center_at_dy"] = [
            float(sidewall_center_at_dy[0]),
            float(sidewall_center_at_dy[1]),
        ]

    if sidewall_left_at_dy is not None:
        out["left_at_dy"] = [
            float(sidewall_left_at_dy[0]),
            float(sidewall_left_at_dy[1]),
        ]

    if sidewall_right_at_dy is not None:
        out["right_at_dy"] = [
            float(sidewall_right_at_dy[0]),
            float(sidewall_right_at_dy[1]),
        ]

    return out


def _point_from_bottom_mid_frame(notch_frame: Dict[str, Any], dx: float, dy: float):
    if not isinstance(notch_frame, dict):
        return None

    bottom_mid = _as_pt(notch_frame.get("bottom_mid"))
    u = notch_frame.get("u")
    n_up = notch_frame.get("n_up")

    if bottom_mid is None or u is None or n_up is None:
        return None

    bx, by = bottom_mid
    ux, uy = float(u[0]), float(u[1])
    nx, ny = float(n_up[0]), float(n_up[1])

    px = float(bx + float(dx) * ux + float(dy) * nx)
    py = float(by + float(dx) * uy + float(dy) * ny)

    return (px, py), {
        "coord_model": "bottom_mid_local_frame",
        "point": (px, py),
        "bottom_mid": bottom_mid,
        "x_axis": (ux, uy),
        "y_axis": (nx, ny),
        "dx": float(dx),
        "dy": float(dy),
    }


def _point_from_sidewall_centerline_frame(notch_frame: Dict[str, Any], dx: float, dy: float):
    data = _point_at_notch_height(notch_frame, dy=float(dy), dx=float(dx))

    if data is None:
        return None

    data = dict(data)
    data["coord_model"] = "sidewall_centerline_frame"

    return data["point"], data


def _point_from_notch_frame(
    notch_frame: Dict[str, Any],
    dx: float,
    dy: float,
    *,
    coord_model: str = "bottom_mid_local_frame",
):
    model = str(coord_model or "bottom_mid_local_frame").strip().lower()

    if model in ("bottom_mid_local_frame", "bottom_mid_frame", "bottom_frame"):
        return _point_from_bottom_mid_frame(notch_frame, dx, dy)

    return _point_from_sidewall_centerline_frame(notch_frame, dx, dy)

def _expected_center_from_notch_frame(cfg: Dict[str, Any], notch_frame: Dict[str, Any]):
    """
    Rebuild the expected/correct baseplate center in the CURRENT camera frame.

    This uses the saved golden expected_notch_frame numbers, but applies them to
    the live notch frame. So if the glass is rotated or shifted, the expected
    baseplate target moves/rotates with it.

    Returns:
      (expected_center_abs, debug_dict)
    """
    if not isinstance(cfg, dict) or not isinstance(notch_frame, dict):
        return None, None

    expected_nf = cfg.get("expected_notch_frame")

    if not isinstance(expected_nf, dict):
        return None, None

    coord_model = str(expected_nf.get("coord_model", "")).strip().lower()

    if not coord_model:
        if "frame_dx" in expected_nf or "frame_dy" in expected_nf:
            coord_model = "bottom_mid_local_frame"
        else:
            coord_model = "sidewall_centerline_frame"

    if coord_model in ("bottom_mid_local_frame", "bottom_mid_frame", "bottom_frame"):
        exp_dx = float(expected_nf.get("frame_dx", expected_nf.get("dx", 0.0)))
        exp_dy = float(expected_nf.get("frame_dy", expected_nf.get("dy", 0.0)))
    else:
        exp_dx = float(expected_nf.get("sidewall_dx", expected_nf.get("dx", 0.0)))
        exp_dy = float(expected_nf.get("sidewall_dy", expected_nf.get("dy", 0.0)))

    result = _point_from_notch_frame(
        notch_frame,
        exp_dx,
        exp_dy,
        coord_model=coord_model,
    )

    if result is None:
        return None, None

    pt, dbg = result

    expected_abs = (
        float(pt[0]),
        float(pt[1]),
    )

    if isinstance(dbg, dict):
        dbg = dict(dbg)
    else:
        dbg = {}

    dbg["coord_model"] = coord_model
    dbg["expected_dx"] = float(exp_dx)
    dbg["expected_dy"] = float(exp_dy)

    return expected_abs, dbg

# ----------------------------
# Baseplate smoothing
# ----------------------------
class BaseplateHysteresis:
    def __init__(self):
        self.center_abs = None
        self.angle = None
        self.contour_abs = None
        self.miss_count = 0
        self.info = {}

    def reset(self):
        self.center_abs = None
        self.angle = None
        self.contour_abs = None
        self.miss_count = 0
        self.info = {}

    def update(
        self,
        center_abs,
        angle,
        contour_abs,
        *,
        center_alpha: float = 0.35,
        angle_alpha: float = 0.25,
        max_jump_px: float = 45.0,
        hold_frames: int = 3,
    ):
        c = _as_pt(center_abs)

        if c is None or angle is None:
            self.miss_count += 1

            if self.center_abs is not None and self.miss_count <= int(hold_frames):
                self.info = {
                    "mode": "hold_missing",
                    "accepted": False,
                    "miss_count": int(self.miss_count),
                }
                return self.center_abs, self.angle, self.contour_abs, self.info

            self.info = {
                "mode": "missing",
                "accepted": False,
                "miss_count": int(self.miss_count),
            }
            return None, None, None, self.info

        angle = float(angle)

        if self.center_abs is None:
            self.center_abs = c
            self.angle = angle
            self.contour_abs = contour_abs
            self.miss_count = 0
            self.info = {
                "mode": "init",
                "accepted": True,
                "jump_px": 0.0,
            }
            return self.center_abs, self.angle, self.contour_abs, self.info

        jump = float(np.hypot(c[0] - self.center_abs[0], c[1] - self.center_abs[1]))

        if jump > float(max_jump_px):
            self.miss_count += 1

            if self.miss_count <= int(hold_frames):
                self.info = {
                    "mode": "hold_jump",
                    "accepted": False,
                    "jump_px": float(jump),
                    "miss_count": int(self.miss_count),
                }
                return self.center_abs, self.angle, self.contour_abs, self.info

        ca = float(np.clip(center_alpha, 0.0, 1.0))
        aa = float(np.clip(angle_alpha, 0.0, 1.0))

        sx = (1.0 - ca) * self.center_abs[0] + ca * c[0]
        sy = (1.0 - ca) * self.center_abs[1] + ca * c[1]

        da = _angle_diff_deg(angle, self.angle)
        sang = _angle_wrap_deg(self.angle + aa * da)

        self.center_abs = (float(sx), float(sy))
        self.angle = float(sang)
        self.contour_abs = contour_abs
        self.miss_count = 0

        self.info = {
            "mode": "blend_accept",
            "accepted": True,
            "jump_px": float(jump),
        }

        return self.center_abs, self.angle, self.contour_abs, self.info


# ----------------------------
# PROC diagnostic helpers
# ----------------------------
def _safe_img_u8(img):
    if img is None:
        return None

    if not isinstance(img, np.ndarray):
        return None

    arr = img

    if arr.size == 0:
        return None

    if arr.dtype == np.uint8:
        return arr

    arr = arr.astype(np.float32)
    mn = float(np.nanmin(arr))
    mx = float(np.nanmax(arr))

    if not np.isfinite(mn) or not np.isfinite(mx) or abs(mx - mn) < 1e-6:
        return np.zeros(arr.shape[:2], dtype=np.uint8)

    out = (255.0 * (arr - mn) / (mx - mn)).clip(0, 255).astype(np.uint8)
    return out


def _colorize_gray(gray, cmap=cv2.COLORMAP_TURBO):
    u8 = _safe_img_u8(gray)

    if u8 is None:
        return None

    if u8.ndim == 3:
        return u8.copy()

    return cv2.applyColorMap(u8, cmap)


def _diag_text(vis, text, org, scale=0.45, color=(255, 255, 255), thickness=1):
    x, y = int(org[0]), int(org[1])

    cv2.putText(
        vis,
        str(text),
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        float(scale),
        (0, 0, 0),
        int(thickness) + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        str(text),
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        float(scale),
        color,
        int(thickness),
        cv2.LINE_AA,
    )


def _draw_fitline(vis, line, color=(0, 255, 0), thickness=2):
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


def _draw_points(vis, points, color=(0, 255, 255), radius=1, step=12):
    if points is None:
        return

    try:
        arr = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    except Exception:
        return

    if arr.size == 0:
        return

    arr = arr[np.isfinite(arr).all(axis=1)]

    if len(arr) <= 0:
        return

    H, W = vis.shape[:2]
    step = max(1, int(step))

    for i in range(0, len(arr), step):
        x = int(round(float(arr[i, 0])))
        y = int(round(float(arr[i, 1])))

        if 0 <= x < W and 0 <= y < H:
            cv2.circle(vis, (x, y), int(radius), color, -1, lineType=cv2.LINE_AA)


def _draw_contour_like(vis, contour, color=(0, 255, 0), thickness=2, closed=True):
    if contour is None:
        return

    try:
        arr = np.asarray(contour, dtype=np.int32).reshape(-1, 1, 2)
    except Exception:
        return

    if len(arr) < 2:
        return

    cv2.polylines(vis, [arr], bool(closed), color, int(thickness), lineType=cv2.LINE_AA)


def _proc_is_image(v) -> bool:
    return isinstance(v, np.ndarray) and v.size > 0 and v.ndim in (2, 3)


def _proc_find_image(v):
    if _proc_is_image(v):
        return v

    if isinstance(v, dict):
        preferred = [
            "image",
            "img",
            "bgr",
            "vis",
            "frame",
            "gray",
            "mask",
            "edges",
            "debug",
            "closed",
            "threshold",
            "dark_mask",
            "contour_mask",
        ]

        for k in preferred:
            if k in v:
                found = _proc_find_image(v[k])
                if found is not None:
                    return found

        for item in v.values():
            found = _proc_find_image(item)
            if found is not None:
                return found

    if isinstance(v, (list, tuple)):
        for item in v:
            found = _proc_find_image(item)
            if found is not None:
                return found

    return None


def _proc_pick_debug_image(debug_dict, keys):
    if not isinstance(debug_dict, dict):
        return None

    frames = debug_dict.get("dbg")

    if isinstance(frames, dict):
        for k in keys:
            if k in frames:
                found = _proc_find_image(frames.get(k))
                if found is not None:
                    return found

        for v in frames.values():
            found = _proc_find_image(v)
            if found is not None:
                return found

    for k in keys:
        if k in debug_dict:
            found = _proc_find_image(debug_dict.get(k))
            if found is not None:
                return found

    return None


def _build_proc_fallback_view(
    *,
    raw,
    roi_live=None,
    roi_poly=None,
    center_abs=None,
    contour_abs=None,
    state="PROC",
    status_text="PROC",
    reason="",
):
    if raw is None:
        return None

    try:
        H, W = raw.shape[:2]

        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

        try:
            cmap = cv2.COLORMAP_TURBO
        except Exception:
            cmap = cv2.COLORMAP_JET

        heat = cv2.applyColorMap(gray, cmap)

        edges = cv2.Canny(gray, 60, 150)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        proc = cv2.addWeighted(heat, 0.78, edges_bgr, 0.60, 0)

        for x in range(0, W, 50):
            cv2.line(proc, (x, 0), (x, H), (35, 35, 35), 1, lineType=cv2.LINE_AA)

        for y in range(0, H, 50):
            cv2.line(proc, (0, y), (W, y), (35, 35, 35), 1, lineType=cv2.LINE_AA)

        if roi_live is not None:
            try:
                x, y, w, h = map(int, roi_live)
                cv2.rectangle(proc, (x, y), (x + w, y + h), (255, 255, 0), 2, lineType=cv2.LINE_AA)
                _diag_text(proc, "LIVE ROI", (x + 6, y + 20), color=(255, 255, 0), thickness=1)
            except Exception:
                pass

        if roi_poly is not None:
            try:
                poly = np.asarray(roi_poly, dtype=np.int32).reshape(-1, 1, 2)
                if len(poly) >= 3:
                    cv2.polylines(proc, [poly], True, (255, 255, 0), 2, lineType=cv2.LINE_AA)
            except Exception:
                pass

        if contour_abs is not None:
            try:
                cnt = np.asarray(contour_abs, dtype=np.int32).reshape(-1, 1, 2)
                if len(cnt) >= 3:
                    cv2.drawContours(proc, [cnt], -1, (0, 255, 0), 2, lineType=cv2.LINE_AA)
            except Exception:
                pass

        c = _as_pt(center_abs)
        if c is not None:
            cx, cy = int(round(c[0])), int(round(c[1]))
            cv2.drawMarker(
                proc,
                (cx, cy),
                (0, 0, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=28,
                thickness=2,
                line_type=cv2.LINE_AA,
            )
            cv2.circle(proc, (cx, cy), 10, (255, 255, 255), 1, lineType=cv2.LINE_AA)
            _diag_text(proc, "BASEPLATE", (cx + 12, cy - 10), color=(0, 0, 255))

        cv2.rectangle(proc, (16, 16), (min(W - 16, 920), 112), (0, 0, 0), -1)
        cv2.rectangle(proc, (16, 16), (min(W - 16, 920), 112), (255, 255, 255), 1, lineType=cv2.LINE_AA)

        _diag_text(
            proc,
            "PROC SAFE DIAGNOSTIC VIEW",
            (32, 45),
            scale=0.75,
            color=(255, 255, 255),
            thickness=2,
        )

        _diag_text(
            proc,
            str(status_text),
            (32, 76),
            scale=0.58,
            color=(0, 255, 255),
            thickness=2,
        )

        if reason:
            _diag_text(
                proc,
                f"fallback reason: {reason}",
                (32, 101),
                scale=0.45,
                color=(80, 180, 255),
            )

        cv2.rectangle(proc, (0, 0), (W - 1, H - 1), (255, 255, 255), 1, lineType=cv2.LINE_AA)

        return proc

    except Exception:
        return raw.copy()


def _proc_make_frame_debug_view(
    *,
    raw,
    roi_live=None,
    roi_poly=None,
    center_abs=None,
    contour_abs=None,
    stab_info=None,
):
    if raw is None:
        return None

    vis = raw.copy()
    H, W = vis.shape[:2]

    gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
    heat = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    vis = cv2.addWeighted(vis, 0.25, heat, 0.75, 0)

    stab = stab_info if isinstance(stab_info, dict) else {}

    try:
        sx, sy, sw, sh = map(int, stab.get("search_roi_current", (0, 0, 0, 0)))
        if sw > 1 and sh > 1:
            cv2.rectangle(vis, (sx, sy), (sx + sw, sy + sh), (210, 210, 210), 1, cv2.LINE_AA)
            _diag_text(vis, "search", (sx + 6, sy + 18), color=(230, 230, 230))
    except Exception:
        pass

    try:
        x, y, w, h = map(int, roi_live)
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 0), 2, cv2.LINE_AA)
        _diag_text(vis, "live ROI", (x + 6, y + 22), color=(255, 255, 0), thickness=1)
    except Exception:
        pass

    if roi_poly is not None:
        try:
            poly = np.asarray(roi_poly, dtype=np.int32).reshape(-1, 1, 2)
            if len(poly) >= 3:
                cv2.polylines(vis, [poly], True, (255, 255, 0), 2, cv2.LINE_AA)
        except Exception:
            pass

    active_dbg = stab.get("current_dark_debug")
    if not isinstance(active_dbg, dict):
        active_dbg = stab.get("current_lines_debug") if isinstance(stab.get("current_lines_debug"), dict) else {}

    try:
        _draw_contour_like(vis, active_dbg.get("contour_abs"), color=(0, 255, 255), thickness=2, closed=False)
        _draw_contour_like(vis, active_dbg.get("lower_contour_abs"), color=(0, 180, 255), thickness=2, closed=False)
    except Exception:
        pass

    nf = stab.get("notch_frame_runtime")
    if not isinstance(nf, dict):
        nf = stab.get("current_notch_frame") if isinstance(stab.get("current_notch_frame"), dict) else {}

    if isinstance(nf, dict):
        try:
            _draw_fitline(vis, nf.get("left_line"), color=(0, 255, 0), thickness=2)
            _draw_fitline(vis, nf.get("right_line"), color=(0, 255, 0), thickness=2)
            _draw_fitline(vis, nf.get("bottom_line"), color=(0, 165, 255), thickness=2)
        except Exception:
            pass

    pts = {
        "bottom_left": stab.get("bottom_left"),
        "bottom_mid": stab.get("bottom_mid"),
        "bottom_right": stab.get("bottom_right"),
    }

    if isinstance(nf, dict):
        for k in pts:
            if pts[k] is None:
                pts[k] = nf.get(k)

    bl = _as_pt(pts.get("bottom_left"))
    bm = _as_pt(pts.get("bottom_mid"))
    br = _as_pt(pts.get("bottom_right"))

    if bl is not None and br is not None:
        cv2.line(
            vis,
            (int(round(bl[0])), int(round(bl[1]))),
            (int(round(br[0])), int(round(br[1]))),
            (0, 165, 255),
            3,
            cv2.LINE_AA,
        )

    for name, pt, col in (
        ("bottom_left", bl, (0, 170, 255)),
        ("bottom_mid", bm, (0, 0, 255)),
        ("bottom_right", br, (0, 170, 255)),
    ):
        if pt is None:
            continue

        px, py = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(vis, (px, py), 8, col, -1, cv2.LINE_AA)
        cv2.circle(vis, (px, py), 12, (255, 255, 255), 1, cv2.LINE_AA)
        _diag_text(vis, name, (px + 10, py - 8), color=col)

    if bm is not None:
        _diag_text(
            vis,
            "bottom_mid = midpoint(bottom_left, bottom_right)",
            (max(10, int(bm[0]) - 220), min(H - 20, int(bm[1]) + 35)),
            color=(0, 0, 255),
        )

    try:
        _draw_contour_like(vis, contour_abs, color=(0, 255, 0), thickness=2, closed=True)
    except Exception:
        pass

    c = _as_pt(center_abs)
    if c is not None:
        cx, cy = int(round(c[0])), int(round(c[1]))
        cv2.drawMarker(
            vis,
            (cx, cy),
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=30,
            thickness=2,
            line_type=cv2.LINE_AA,
        )
        _diag_text(vis, "baseplate center", (cx + 14, cy - 10), color=(0, 0, 255))

    cv2.rectangle(vis, (12, 12), (min(W - 12, 620), 72), (0, 0, 0), -1)
    cv2.rectangle(vis, (12, 12), (min(W - 12, 620), 72), (255, 255, 255), 1, cv2.LINE_AA)
    _diag_text(vis, "FRAME / ANCHOR DEBUG", (26, 40), scale=0.72, color=(255, 255, 255), thickness=2)
    _diag_text(vis, f"model={stab.get('roi_mode', '-')}", (26, 62), scale=0.45, color=(220, 220, 220))

    return vis


def _build_proc_diagnostic_view(
    *,
    raw,
    roi_live,
    roi_poly,
    center_abs,
    contour_abs,
    baseplate_dbg,
    stab_info,
    cfg,
    state,
    status_text,
    fps,
):
    if raw is None:
        return None

    H, W = raw.shape[:2]

    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    heat = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    dark = cv2.convertScaleAbs(raw, alpha=0.35, beta=0)
    proc = cv2.addWeighted(dark, 0.35, heat, 0.65, 0)

    for x in range(0, W, 40):
        cv2.line(proc, (x, 0), (x, H), (40, 40, 40), 1, lineType=cv2.LINE_AA)

    for y in range(0, H, 40):
        cv2.line(proc, (0, y), (W, y), (40, 40, 40), 1, lineType=cv2.LINE_AA)

    stab = stab_info if isinstance(stab_info, dict) else {}
    active_dbg = stab.get("current_dark_debug") if isinstance(stab.get("current_dark_debug"), dict) else {}
    legacy_dbg = stab.get("legacy_lines_debug") if isinstance(stab.get("legacy_lines_debug"), dict) else {}
    current_lines_debug = stab.get("current_lines_debug") if isinstance(stab.get("current_lines_debug"), dict) else {}

    if not active_dbg and current_lines_debug:
        active_dbg = current_lines_debug

    try:
        sx, sy, sw, sh = map(int, stab.get("search_roi_current", (0, 0, 0, 0)))
        if sw > 1 and sh > 1:
            cv2.rectangle(proc, (sx, sy), (sx + sw, sy + sh), (160, 160, 160), 1, lineType=cv2.LINE_AA)
            _diag_text(proc, "NOTCH SEARCH ROI", (sx + 6, sy + 18), color=(220, 220, 220))
    except Exception:
        pass

    try:
        x, y, w, h = map(int, roi_live)
        cv2.rectangle(proc, (x, y), (x + w, y + h), (255, 255, 0), 2, lineType=cv2.LINE_AA)
        _diag_text(proc, "LIVE BASEPLATE ROI", (x + 6, y + 20), color=(255, 255, 0), thickness=1)
    except Exception:
        pass

    if roi_poly is not None:
        try:
            poly = np.asarray(roi_poly, dtype=np.int32).reshape(-1, 1, 2)
            if len(poly) >= 3:
                cv2.polylines(proc, [poly], True, (255, 255, 0), 2, lineType=cv2.LINE_AA)
        except Exception:
            pass

    _draw_contour_like(proc, active_dbg.get("contour_abs"), color=(0, 255, 255), thickness=2, closed=False)
    _draw_contour_like(proc, active_dbg.get("lower_contour_abs"), color=(0, 210, 255), thickness=2, closed=False)
    _draw_contour_like(proc, active_dbg.get("hull_abs"), color=(0, 140, 255), thickness=1, closed=True)

    pts = active_dbg.get("points_used_abs") if isinstance(active_dbg.get("points_used_abs"), dict) else {}
    _draw_points(proc, pts.get("all"), color=(80, 220, 255), radius=1, step=14)
    _draw_points(proc, pts.get("left"), color=(0, 255, 80), radius=2, step=8)
    _draw_points(proc, pts.get("right"), color=(0, 255, 80), radius=2, step=8)
    _draw_points(proc, pts.get("bottom"), color=(0, 140, 255), radius=2, step=8)

    legacy_pts = legacy_dbg.get("points_used_abs") if isinstance(legacy_dbg.get("points_used_abs"), dict) else {}
    _draw_points(proc, legacy_pts.get("left"), color=(180, 80, 255), radius=1, step=12)
    _draw_points(proc, legacy_pts.get("right"), color=(180, 80, 255), radius=1, step=12)
    _draw_points(proc, legacy_pts.get("bottom"), color=(180, 80, 255), radius=1, step=12)

    nf = stab.get("notch_frame_runtime")
    if not isinstance(nf, dict):
        nf = stab.get("current_notch_frame") if isinstance(stab.get("current_notch_frame"), dict) else {}

    if isinstance(nf, dict):
        _draw_fitline(proc, nf.get("left_line"), color=(0, 255, 0), thickness=2)
        _draw_fitline(proc, nf.get("right_line"), color=(0, 255, 0), thickness=2)
        _draw_fitline(proc, nf.get("bottom_line"), color=(0, 165, 255), thickness=2)

    for name, col in (
        ("bottom_left", (0, 170, 255)),
        ("bottom_right", (0, 170, 255)),
        ("bottom_mid", (0, 0, 255)),
    ):
        pt = stab.get(name)
        if pt is None and isinstance(nf, dict):
            pt = nf.get(name)

        if _as_pt(pt) is not None:
            px, py = _as_pt(pt)
            cv2.circle(proc, (int(round(px)), int(round(py))), 7, col, -1, lineType=cv2.LINE_AA)
            cv2.circle(proc, (int(round(px)), int(round(py))), 10, (255, 255, 255), 1, lineType=cv2.LINE_AA)
            _diag_text(proc, name, (int(px) + 9, int(py) - 8), color=col)

    bl = _as_pt(stab.get("bottom_left"))
    br = _as_pt(stab.get("bottom_right"))
    bm = _as_pt(stab.get("bottom_mid"))

    if bl is not None and br is not None:
        cv2.line(
            proc,
            (int(round(bl[0])), int(round(bl[1]))),
            (int(round(br[0])), int(round(br[1]))),
            (0, 165, 255),
            2,
            lineType=cv2.LINE_AA,
        )

    if bm is not None:
        _diag_text(
            proc,
            "bottom_mid = midpoint(BL, BR)",
            (int(round(bm[0])) + 12, int(round(bm[1])) + 18),
            color=(0, 0, 255),
        )

    _draw_contour_like(proc, contour_abs, color=(0, 255, 0), thickness=2, closed=True)

    if center_abs is not None:
        c = _as_pt(center_abs)
        if c is not None:
            cx, cy = int(round(c[0])), int(round(c[1]))
            cv2.drawMarker(
                proc,
                (cx, cy),
                (0, 0, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=26,
                thickness=2,
                line_type=cv2.LINE_AA,
            )
            cv2.circle(proc, (cx, cy), 9, (255, 255, 255), 1, lineType=cv2.LINE_AA)
            _diag_text(proc, "BASEPLATE CENTER", (cx + 14, cy - 10), color=(0, 0, 255))

    if state == "PASS":
        state_color = (0, 255, 0)
    elif state == "TRACK":
        state_color = (0, 255, 255)
    elif state == "SEARCH":
        state_color = (190, 190, 190)
    else:
        state_color = (0, 0, 255)

    cv2.rectangle(proc, (16, 16), (min(W - 16, 880), 118), (0, 0, 0), -1)
    cv2.rectangle(proc, (16, 16), (min(W - 16, 880), 118), state_color, 2, lineType=cv2.LINE_AA)

    _diag_text(proc, "PROC DIAGNOSTIC VIEW", (32, 43), scale=0.72, color=(255, 255, 255), thickness=2)
    _diag_text(proc, status_text, (32, 76), scale=0.62, color=state_color, thickness=2)
    _diag_text(
        proc,
        f"FPS={fps:.1f}  ROI_MODE={stab.get('roi_mode', '-')}  ANCHOR={stab.get('current_anchor_method', '-')}",
        (32, 102),
        scale=0.48,
        color=(220, 220, 220),
    )

    offset = stab.get("current_offset_display") if isinstance(stab.get("current_offset_display"), dict) else {}
    measure = stab.get("current_notch_measure") if isinstance(stab.get("current_notch_measure"), dict) else {}
    params = stab.get("notch_param_debug") if isinstance(stab.get("notch_param_debug"), dict) else {}

    left_lines = []

    if offset:
        try:
            unit = offset.get("unit", "px")
            left_lines.append(
                f"offset: dx={float(offset.get('dx', 0.0)):.3f}{unit}  dy={float(offset.get('dy', 0.0)):.3f}{unit}"
            )
        except Exception:
            pass

    if measure:
        try:
            left_lines.append(
                f"notch local: dx={float(measure.get('dx', 0.0)):+.2f}px  "
                f"dy={float(measure.get('dy', 0.0)):+.2f}px  "
                f"relTheta={float(measure.get('relative_angle', 0.0)):+.2f}deg"
            )
        except Exception:
            pass

    if params:
        left_lines.append(
            "notch params: "
            f"blur={params.get('notch_blur_ksize', '-')} "
            f"close={params.get('notch_close_ksize', '-')} "
            f"open={params.get('notch_open_ksize', '-')} "
            f"bias={params.get('notch_threshold_bias', '-')}"
        )
        left_lines.append(
            "bands: "
            f"bottom={params.get('notch_bottom_band_frac', '-')} "
            f"side={params.get('notch_side_band_frac', '-')}"
        )

    scale_saved = stab.get("baseplate_px_per_mm_saved")
    scale_frame = stab.get("baseplate_px_per_mm_frame")

    if scale_saved is not None:
        try:
            left_lines.append(f"saved scale: {float(scale_saved):.3f}px/mm")
        except Exception:
            pass

    if scale_frame is not None:
        try:
            left_lines.append(f"frame scale: {float(scale_frame):.3f}px/mm")
        except Exception:
            pass

    bp_hyst = stab.get("baseplate_hysteresis") if isinstance(stab.get("baseplate_hysteresis"), dict) else {}
    if bp_hyst:
        left_lines.append(f"baseplate hysteresis: {bp_hyst.get('mode', '-')}")

    box_x0 = 16
    box_y0 = 130
    box_x1 = min(W - 240, 820)
    line_h = 24
    min_lines = 8
    box_y1 = min(H - 20, box_y0 + 42 + line_h * max(min_lines, len(left_lines)))

    cv2.rectangle(proc, (box_x0, box_y0), (box_x1, box_y1), (0, 0, 0), -1)
    cv2.rectangle(proc, (box_x0, box_y0), (box_x1, box_y1), (90, 90, 90), 1, lineType=cv2.LINE_AA)

    _diag_text(proc, "LIVE NUMBERS", (box_x0 + 12, box_y0 + 24), scale=0.55, color=(255, 255, 255), thickness=1)

    for i, line in enumerate(left_lines):
        y_line = box_y0 + 52 + i * line_h

        if y_line > box_y1 - 10:
            break

        _diag_text(proc, line, (box_x0 + 12, y_line), scale=0.48, color=(230, 230, 230))

    cv2.rectangle(proc, (0, 0), (W - 1, H - 1), (255, 255, 255), 1, lineType=cv2.LINE_AA)

    return proc


def _build_proc_payload(
    *,
    raw,
    proc_main,
    baseplate_dbg=None,
    stab_info=None,
    roi_live=None,
    roi_poly=None,
    center_abs=None,
    contour_abs=None,
    state="",
    status_text="",
    fps=0.0,
):
    stab = stab_info if isinstance(stab_info, dict) else {}

    current_dark = stab.get("current_dark_debug")
    if not isinstance(current_dark, dict):
        current_dark = {}

    base_gray = _proc_pick_debug_image(
        baseplate_dbg,
        ["gray2_contrast", "gray1_clahe", "gray0", "gray", "contrast"],
    )

    base_edges = _proc_pick_debug_image(
        baseplate_dbg,
        ["edges0", "edges1_dilate", "edges2_close", "edges", "canny"],
    )

    base_mask = _proc_pick_debug_image(
        baseplate_dbg,
        ["edges2_close", "mask", "closed", "edges1_dilate", "binary"],
    )

    frame_debug = _proc_make_frame_debug_view(
        raw=raw,
        roi_live=roi_live,
        roi_poly=roi_poly,
        center_abs=center_abs,
        contour_abs=contour_abs,
        stab_info=stab,
    )

    if base_gray is None:
        base_gray = raw

    if base_edges is None:
        try:
            g = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
            base_edges = cv2.Canny(g, 60, 150)
        except Exception:
            base_edges = raw

    if base_mask is None:
        for k in ("mask_abs", "mask", "threshold", "dark_mask", "edges", "debug", "dbg"):
            if k in current_dark:
                found = _proc_find_image(current_dark.get(k))
                if found is not None:
                    base_mask = found
                    break

    if base_mask is None:
        base_mask = base_edges

    if frame_debug is None:
        frame_debug = proc_main if proc_main is not None else raw

    feeds = {
        "base_gray": {
            "title": "BASE GRAY / CONTRAST",
            "image": base_gray,
            "help": (
                "Cleaned grayscale/contrast view before baseplate edge detection. "
                "Use it to check lighting, glare, shadows, and whether the baseplate stands out."
            ),
        },
        "base_edges": {
            "title": "BASE EDGES",
            "image": base_edges,
            "help": (
                "Edges found around the baseplate. Too many random edges means noise/glare. "
                "Missing baseplate edges means thresholds are too strict or lighting is weak."
            ),
        },
        "base_mask": {
            "title": "BASE CLOSED / MASK",
            "image": base_mask,
            "help": (
                "Cleaned segmentation/mask after gap-filling. A good mask should show a solid baseplate shape "
                "without merging into nearby shadows or reflections."
            ),
        },
        "frame_debug": {
            "title": "FRAME / ANCHOR DEBUG",
            "image": frame_debug,
            "help": (
                "Geometry used to place the live ROI: notch side lines, bottom line, bottom-left, "
                "bottom-mid, bottom-right, rotated ROI, and detected baseplate center."
            ),
        },
    }

    return {
        "main": proc_main if proc_main is not None else raw,
        "feeds": feeds,
        "stats": {
            "state": state,
            "status_text": status_text,
            "fps": float(fps or 0.0),
            "roi_mode": stab.get("roi_mode"),
            "anchor_method": stab.get("current_anchor_method"),
            "offset_display": stab.get("current_offset_display"),
            "notch_params": stab.get("notch_param_debug"),
        },
    }


# ----------------------------
# Engine
# ----------------------------
class QCPreviewEngine:
    def __init__(self):
        self.recipe: Optional[Recipe] = None
        self.settings = EngineSettings()

        self._frame_i = 0
        self._roi_live: Optional[Tuple[int, int, int, int]] = None
        self._roi_live_poly = None
        self._stab_info: Optional[Dict[str, Any]] = None

        self._stable_have = 0
        self._fps_t0 = time.time()
        self._fps = 0.0

        self._last_out: Optional[QCFrameOutput] = None
        self._bp_hyst = BaseplateHysteresis()

    def set_recipe(self, recipe: Recipe):
        self.recipe = recipe
        self._frame_i = 0
        self._roi_live = tuple(recipe.cfg["roi"])
        self._roi_live_poly = None
        self._stab_info = None
        self._stable_have = 0
        self._last_out = None
        self._bp_hyst.reset()

        try:
            roi_stablizer.reset_default_contour_hysteresis()
        except Exception:
            pass

    def get_raw_frame(self) -> Optional[np.ndarray]:
        return None if self._last_out is None else self._last_out.raw_bgr

    def get_processed_frame(self) -> Optional[np.ndarray]:
        return None if self._last_out is None else self._last_out.proc_bgr

    def get_overlay_frame(self) -> Optional[np.ndarray]:
        return None if self._last_out is None else self._last_out.overlay_bgr

    def _detector_kwargs_from_cfg(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        return dict(
            shrink_border_px=int(cfg.get("shrink_border_px", 10)),
            border_margin=int(cfg.get("border_margin", 12)),
            use_clahe=bool(cfg.get("use_clahe", True)),
            clahe_clip=float(cfg.get("clahe_clip", 1.5)),
            clahe_grid=int(cfg.get("clahe_grid", 8)),
            blur_ksize=int(cfg.get("blur_ksize", 5)),
            canny_low=int(cfg.get("canny_low", 50)),
            canny_high=int(cfg.get("canny_high", 120)),
            dilate_iter=int(cfg.get("dilate_iter", 1)),
            close_iter=int(cfg.get("close_iter", 1)),
            area_min_frac=float(cfg.get("area_min_frac", 0.005)),
            area_max_frac=float(cfg.get("area_max_frac", 0.60)),
            aspect_min=float(cfg.get("aspect_min", 0.5)),
            aspect_max=float(cfg.get("aspect_max", 2.2)),
            solidity_min=float(cfg.get("solidity_min", 0.7)),
            extent_min=float(cfg.get("extent_min", 0.25)),
            contrast_min=float(cfg.get("contrast_min", 6.0)),
        )

    def _build_notch_frame_roi(self, cfg, notch_frame, W, H):
        expected_nf = cfg.get("expected_notch_frame")

        if not isinstance(expected_nf, dict):
            return None, None, "missing_expected_notch_frame", None

        coord_model = str(expected_nf.get("coord_model", "")).strip().lower()

        if not coord_model:
            if "frame_dx" in expected_nf or "frame_dy" in expected_nf:
                coord_model = "bottom_mid_local_frame"
            else:
                coord_model = "sidewall_centerline_frame"

        if coord_model in ("bottom_mid_local_frame", "bottom_mid_frame", "bottom_frame"):
            exp_dx = float(expected_nf.get("frame_dx", expected_nf.get("dx", 0.0)))
            exp_dy = float(expected_nf.get("frame_dy", expected_nf.get("dy", 0.0)))
        else:
            exp_dx = float(expected_nf.get("sidewall_dx", expected_nf.get("dx", 0.0)))
            exp_dy = float(expected_nf.get("sidewall_dy", expected_nf.get("dy", 0.0)))

        center_and_debug = _point_from_notch_frame(
            notch_frame,
            exp_dx,
            exp_dy,
            coord_model=coord_model,
        )

        if center_and_debug is None:
            return None, None, "notch_expected_center_failed", None

        center, frame_debug = center_and_debug

        _rx, _ry, rw, rh = map(float, cfg.get("roi", (0, 0, 100, 100)))

        roi_scale = float(cfg.get("notch_frame_roi_scale", 1.0))
        extra_w = float(cfg.get("notch_frame_roi_extra_w", cfg.get("baseplate_roi_extra_pad", 35))) * 2.0
        extra_h = float(cfg.get("notch_frame_roi_extra_h", cfg.get("baseplate_roi_extra_pad", 35))) * 2.0

        live_w = max(20.0, rw * roi_scale + extra_w)
        live_h = max(20.0, rh * roi_scale + extra_h)

        notch_angle = float(notch_frame.get("angle_deg", 0.0))
        poly = _rotated_rect_poly(center, live_w, live_h, notch_angle)

        bbox = _bbox_from_poly(
            poly,
            W,
            H,
            pad=int(cfg.get("rotated_roi_bbox_pad", 8)),
        )

        if bbox is None:
            return None, None, "rotated_roi_bbox_failed", None

        roi_mode = "bottom_mid_local_rotated_roi" if coord_model in (
            "bottom_mid_local_frame",
            "bottom_mid_frame",
            "bottom_frame",
        ) else "sidewall_centerline_rotated_roi"

        if isinstance(frame_debug, dict):
            frame_debug["roi_mode"] = roi_mode
            frame_debug["expected_coord_model"] = coord_model

        return bbox, poly, roi_mode, frame_debug

    def process_frame(self, frame_bgr: np.ndarray) -> QCFrameOutput:
        if self.recipe is None:
            out = QCFrameOutput(
                overlay_bgr=frame_bgr,
                raw_bgr=frame_bgr,
                proc_bgr=None,
                proc_payload=None,
                status_text="Status: NO PRODUCT LOADED",
                state="FAIL",
                roi_live=(0, 0, 1, 1),
            )
            self._last_out = out
            return out

        cfg = self.recipe.cfg
        golden = self.recipe.golden_bgr
        Hg, Wg = golden.shape[:2]

        raw = frame_bgr

        if raw.shape[:2] != (Hg, Wg):
            raw = cv2.resize(raw, (Wg, Hg), interpolation=cv2.INTER_LINEAR)

        H, W = raw.shape[:2]

        roi_cfg = tuple(cfg["roi"])

        if self._roi_live is None:
            self._roi_live = roi_cfg

        golden_lines = cfg.get("inner_border_lines", None)
        registration_roi_golden = tuple(cfg.get("registration_roi", cfg.get("roi", roi_cfg)))

        # ----------------------------
        # Notch / ROI stabilizer
        # ----------------------------
        if (
            self.settings.stab_every_n > 0
            and (self._frame_i % int(self.settings.stab_every_n) == 0)
        ):
            moved, info = roi_stablizer.stabilize_rois_using_saved_inner_border_lines(
                current_img=raw,
                golden_img=golden,
                registration_roi_golden=registration_roi_golden,
                golden_inner_lines_abs=golden_lines or {},
                rois_golden=[roi_cfg],

                search_padding_px=int(cfg.get("search_padding_px", self.settings.search_padding_px)),

                canny_low=int(cfg.get("canny_low", 60)),
                canny_high=int(cfg.get("canny_high", 140)),

                prefer_dark_region=bool(cfg.get("prefer_dark_region", True)),
                prefer_solid_edge=bool(cfg.get("prefer_solid_edge", True)),

                notch_blur_ksize=int(cfg.get("notch_blur_ksize", 21)),
                notch_close_ksize=int(cfg.get("notch_close_ksize", 11)),
                notch_open_ksize=int(cfg.get("notch_open_ksize", 7)),
                notch_threshold_bias=float(cfg.get("notch_threshold_bias", 1.0)),
                notch_bottom_band_frac=float(cfg.get("notch_bottom_band_frac", 0.35)),
                notch_side_band_frac=float(cfg.get("notch_side_band_frac", 0.35)),

                hysteresis_enabled=bool(cfg.get("hysteresis_enabled", True)),
            )

            self._stab_info = info if isinstance(info, dict) else {}

            notch_frame = _extract_notch_frame(self._stab_info)
            bbox, poly, roi_mode, roi_frame_debug = None, None, "fallback", None

            if notch_frame is not None:
                bbox, poly, roi_mode, roi_frame_debug = self._build_notch_frame_roi(cfg, notch_frame, W, H)

            if bbox is not None:
                self._roi_live = clamp_roi(bbox, W, H)
                self._roi_live_poly = poly
            elif moved is not None and self._stab_info.get("ok"):
                self._roi_live = clamp_roi(moved[0], W, H)
                self._roi_live_poly = None
                roi_mode = "fallback_stabilized_bbox"
            else:
                self._roi_live = clamp_roi(roi_cfg, W, H)
                self._roi_live_poly = None
                roi_mode = "fallback_recipe_roi"

            if isinstance(self._stab_info, dict):
                self._stab_info["notch_frame_runtime"] = notch_frame
                self._stab_info["roi_mode"] = roi_mode
                self._stab_info["roi_frame_debug"] = roi_frame_debug
                self._stab_info["live_roi"] = self._roi_live
                self._stab_info["live_roi_poly"] = None if self._roi_live_poly is None else np.asarray(self._roi_live_poly).tolist()

                if isinstance(notch_frame, dict):
                    for k in ("bottom_left", "bottom_right", "bottom_mid"):
                        if notch_frame.get(k) is not None:
                            self._stab_info[k] = notch_frame.get(k)
                    self._stab_info["bottom_mid_source"] = "midpoint_bottom_left_bottom_right"

        crop, roi_live = safe_crop(raw, self._roi_live)

        # ----------------------------
        # Baseplate detection
        # ----------------------------
        det_kwargs = self._detector_kwargs_from_cfg(cfg)

        want_proc = bool(self.settings.compute_proc) or (str(self.settings.view_mode).upper() == "PROC")

        if want_proc:
            center_rel_raw, angle_raw, contour_rel_raw, dbg = detect_baseplate(
                crop,
                full_image_bgr=raw,
                roi_xywh_abs=roi_live,
                padding=int(cfg.get("padding", 150)),
                return_debug=True,
                **det_kwargs,
            )
        else:
            center_rel_raw, angle_raw, contour_rel_raw = detect_baseplate(
                crop,
                full_image_bgr=raw,
                roi_xywh_abs=roi_live,
                padding=int(cfg.get("padding", 150)),
                return_debug=False,
                **det_kwargs,
            )
            dbg = None

        x_roi, y_roi, _w_roi, _h_roi = roi_live

        center_abs_raw = None
        contour_abs_raw = None

        if center_rel_raw is not None:
            center_abs_raw = (
                float(x_roi + center_rel_raw[0]),
                float(y_roi + center_rel_raw[1]),
            )

        if contour_rel_raw is not None:
            try:
                contour_abs_raw = np.asarray(contour_rel_raw, dtype=np.int32).reshape(-1, 1, 2)
                contour_abs_raw = contour_abs_raw + np.array([[[x_roi, y_roi]]], dtype=np.int32)
            except Exception:
                contour_abs_raw = None

        center_abs, angle, contour_abs, bp_hyst_info = self._bp_hyst.update(
            center_abs_raw,
            angle_raw,
            contour_abs_raw,
            center_alpha=float(cfg.get("baseplate_center_alpha", 0.35)),
            angle_alpha=float(cfg.get("baseplate_angle_alpha", 0.25)),
            max_jump_px=float(cfg.get("baseplate_max_jump_px", 45.0)),
            hold_frames=int(cfg.get("baseplate_hold_frames", 3)),
        )

        center_rel = None
        contour_rel = None

        if center_abs is not None:
            center_rel = (
                float(center_abs[0] - x_roi),
                float(center_abs[1] - y_roi),
            )

        if contour_abs is not None:
            try:
                contour_rel = np.asarray(contour_abs, dtype=np.int32).reshape(-1, 1, 2)
                contour_rel = contour_rel - np.array([[[x_roi, y_roi]]], dtype=np.int32)
            except Exception:
                contour_rel = None

        # ----------------------------
        # Scale estimate
        # ----------------------------
        frame_scale_info = _estimate_px_per_mm_from_contour(contour_abs, cfg)

        if isinstance(self._stab_info, dict):
            if frame_scale_info is not None:
                self._stab_info["baseplate_scale_frame"] = frame_scale_info
                self._stab_info["baseplate_px_per_mm_frame"] = frame_scale_info["px_per_mm"]

            saved_scale = _saved_px_per_mm(cfg)
            if saved_scale is not None:
                self._stab_info["baseplate_px_per_mm_saved"] = float(saved_scale)

        # ----------------------------
        # Notch-relative measurement
        # ----------------------------
        notch_frame = _extract_notch_frame(self._stab_info)
        expected_nf = cfg.get("expected_notch_frame")

        current_nf_measure = None
        sx = sy = stheta = None
        dx_px = dy_px = dtheta = None

        if center_abs is not None and angle is not None and notch_frame is not None:
            current_nf_measure = _measure_in_notch_frame(center_abs, angle, notch_frame)

        if isinstance(expected_nf, dict) and current_nf_measure is not None:
            coord_model = str(expected_nf.get("coord_model", "")).strip().lower()

            if not coord_model:
                if "frame_dx" in expected_nf or "frame_dy" in expected_nf:
                    coord_model = "bottom_mid_local_frame"
                else:
                    coord_model = "sidewall_centerline_frame"

            if coord_model in ("bottom_mid_local_frame", "bottom_mid_frame", "bottom_frame"):
                cur_dx = float(current_nf_measure.get("frame_dx", current_nf_measure.get("dx", 0.0)))
                cur_dy = float(current_nf_measure.get("frame_dy", current_nf_measure.get("dy", 0.0)))

                exp_dx = float(expected_nf.get("frame_dx", expected_nf.get("dx", 0.0)))
                exp_dy = float(expected_nf.get("frame_dy", expected_nf.get("dy", 0.0)))
            else:
                cur_dx = float(current_nf_measure.get("sidewall_dx", current_nf_measure.get("dx", 0.0)))
                cur_dy = float(current_nf_measure.get("sidewall_dy", current_nf_measure.get("dy", 0.0)))

                exp_dx = float(expected_nf.get("sidewall_dx", expected_nf.get("dx", 0.0)))
                exp_dy = float(expected_nf.get("sidewall_dy", expected_nf.get("dy", 0.0)))

            exp_rel_a = float(expected_nf.get("relative_angle", cfg.get("expected_angle", 0.0)))

            sx = float(cur_dx - exp_dx)
            sy = float(cur_dy - exp_dy)

            cur_rel_a = float(current_nf_measure.get("relative_angle", 0.0))
            stheta = _angle_diff_deg(cur_rel_a, exp_rel_a)

            dx_px = abs(sx)
            dy_px = abs(sy)
            dtheta = abs(stheta)

            if isinstance(self._stab_info, dict):
                self._stab_info["current_notch_measure"] = current_nf_measure
                self._stab_info["expected_notch_frame"] = expected_nf
                self._stab_info["baseplate_hysteresis"] = bp_hyst_info

        elif center_rel is not None and angle is not None:
            golden_center = tuple(cfg.get("expected_center", (0.0, 0.0)))
            golden_angle = float(cfg.get("expected_angle", 0.0))

            sx = float(center_rel[0]) - float(golden_center[0])
            sy = float(center_rel[1]) - float(golden_center[1])
            stheta = _angle_diff_deg(float(angle), golden_angle)

            dx_px = abs(sx)
            dy_px = abs(sy)
            dtheta = abs(stheta)

        disp = None

        if dx_px is not None and dy_px is not None:
            disp = _px_to_display(dx_px, dy_px, cfg, frame_scale_info)

            # ----------------------------
            # Expected target center for operator guidance
            # ----------------------------
            if not isinstance(self._stab_info, dict):
                self._stab_info = {}

            if center_abs is not None:
                self._stab_info["current_baseplate_center_abs"] = [
                    float(center_abs[0]),
                    float(center_abs[1]),
                ]
            else:
                self._stab_info.pop("current_baseplate_center_abs", None)

            expected_abs, expected_dbg = _expected_center_from_notch_frame(cfg, notch_frame)

            if expected_abs is not None:
                self._stab_info["expected_baseplate_center_abs"] = [
                    float(expected_abs[0]),
                    float(expected_abs[1]),
                ]
                self._stab_info["expected_baseplate_center_debug"] = expected_dbg

                if center_abs is not None:
                    # Screen-space correction vector:
                    # from current detected center TO expected target center.
                    screen_dx_px = float(expected_abs[0] - center_abs[0])
                    screen_dy_px = float(expected_abs[1] - center_abs[1])
                    screen_dist_px = float(np.hypot(screen_dx_px, screen_dy_px))

                    px_per_mm = None

                    try:
                        saved_scale = _saved_px_per_mm(cfg)
                        if saved_scale is not None and saved_scale > 0:
                            px_per_mm = float(saved_scale)
                    except Exception:
                        px_per_mm = None

                    if px_per_mm is None and isinstance(frame_scale_info, dict):
                        try:
                            v = float(frame_scale_info.get("px_per_mm"))
                            if np.isfinite(v) and v > 0:
                                px_per_mm = v
                        except Exception:
                            px_per_mm = None

                    correction = {
                        "screen_dx_px": float(screen_dx_px),
                        "screen_dy_px": float(screen_dy_px),
                        "screen_dist_px": float(screen_dist_px),

                        # Local notch-frame signed error.
                        # sx/sy = current - expected.
                        # correction in local notch frame = -sx, -sy.
                        "local_current_minus_expected_dx_px": None if sx is None else float(sx),
                        "local_current_minus_expected_dy_px": None if sy is None else float(sy),
                        "local_correction_dx_px": None if sx is None else float(-sx),
                        "local_correction_dy_px": None if sy is None else float(-sy),
                    }

                    if px_per_mm is not None and px_per_mm > 0:
                        correction.update(
                            {
                                "unit": "mm",
                                "px_per_mm": float(px_per_mm),
                                "screen_dx": float(screen_dx_px / px_per_mm),
                                "screen_dy": float(screen_dy_px / px_per_mm),
                                "screen_dist": float(screen_dist_px / px_per_mm),
                                "local_correction_dx": None if sx is None else float((-sx) / px_per_mm),
                                "local_correction_dy": None if sy is None else float((-sy) / px_per_mm),
                            }
                        )
                    else:
                        correction.update(
                            {
                                "unit": "px",
                                "px_per_mm": None,
                                "screen_dx": float(screen_dx_px),
                                "screen_dy": float(screen_dy_px),
                                "screen_dist": float(screen_dist_px),
                                "local_correction_dx": None if sx is None else float(-sx),
                                "local_correction_dy": None if sy is None else float(-sy),
                            }
                        )

                    self._stab_info["baseplate_correction_vector"] = correction
            else:
                self._stab_info.pop("expected_baseplate_center_abs", None)
                self._stab_info.pop("expected_baseplate_center_debug", None)
                self._stab_info.pop("baseplate_correction_vector", None)

            if isinstance(self._stab_info, dict):
                self._stab_info["current_offset_px"] = {
                    "dx": float(dx_px),
                    "dy": float(dy_px),
                }
                self._stab_info["current_offset_display"] = disp

                if disp.get("unit") == "mm":
                    self._stab_info["current_notch_measure_mm"] = {
                        "dx": float(dx_px) / float(disp["px_per_mm"]),
                        "dy": float(dy_px) / float(disp["px_per_mm"]),
                        "unit": "mm",
                    }

        # ----------------------------
        # PASS / TRACK
        # ----------------------------
        tol_px = cfg.get("tolerance_px", {"x": 10, "y": 10, "angle": 5})
        tol_mm = cfg.get("tolerance_mm", None)

        state = "SEARCH"
        text = "SEARCHING????"

        if dx_px is None or dy_px is None or dtheta is None:
            self._stable_have = max(0, self._stable_have - 1)
            state = "SEARCH"
            text = "SEARCHING????"
        else:
            if isinstance(tol_mm, dict) and disp is not None and disp.get("unit") == "mm":
                check_dx = float(disp["dx"])
                check_dy = float(disp["dy"])
                ok_all = (
                    check_dx <= float(tol_mm.get("x", tol_mm.get("dx", 1.0)))
                    and check_dy <= float(tol_mm.get("y", tol_mm.get("dy", 1.0)))
                    and dtheta <= float(tol_mm.get("angle", tol_mm.get("theta", 5)))
                )
            else:
                ok_all = (
                    dx_px <= float(tol_px.get("x", 10))
                    and dy_px <= float(tol_px.get("y", 10))
                    and dtheta <= float(tol_px.get("angle", 5))
                )

            if ok_all:
                self._stable_have = min(int(self.settings.stable_need), self._stable_have + 1)
            else:
                self._stable_have = max(0, self._stable_have - 1)

            if disp is not None and disp.get("unit") == "mm":
                value_txt = f"dx={disp['dx']:.2f}mm dy={disp['dy']:.2f}mm dTheta={dtheta:.1f}"
            else:
                value_txt = f"dx={dx_px:.1f}px dy={dy_px:.1f}px dTheta={dtheta:.1f}"

            if ok_all and self._stable_have >= int(self.settings.stable_need):
                state = "PASS"
                text = f"PASS  {value_txt}"
            else:
                state = "TRACK"
                text = f"TRACK {self._stable_have}/{int(self.settings.stable_need)}  {value_txt}"

        # ----------------------------
        # FPS
        # ----------------------------
        self._frame_i += 1

        if self._frame_i % 15 == 0:
            dt = time.time() - self._fps_t0
            self._fps = 15.0 / max(1e-6, dt)
            self._fps_t0 = time.time()

        # ----------------------------
        # PROC diagnostic view + Qt dashboard payload
        # IMPORTANT:
        # proc_full/proc_payload are initialized OUTSIDE if want_proc,
        # so QCFrameOutput never sees an unbound local.
        # ----------------------------
        proc_full = None
        proc_payload = None

        if want_proc:
            try:
                proc_full = _build_proc_diagnostic_view(
                    raw=raw,
                    roi_live=roi_live,
                    roi_poly=self._roi_live_poly,
                    center_abs=center_abs,
                    contour_abs=contour_abs,
                    baseplate_dbg=dbg,
                    stab_info=self._stab_info,
                    cfg=cfg,
                    state=state,
                    status_text=text,
                    fps=self._fps,
                )

                if proc_full is None:
                    proc_full = _build_proc_fallback_view(
                        raw=raw,
                        roi_live=roi_live,
                        roi_poly=self._roi_live_poly,
                        center_abs=center_abs,
                        contour_abs=contour_abs,
                        state=state,
                        status_text=text,
                        reason="diagnostic returned None",
                    )

            except Exception as e:
                proc_full = _build_proc_fallback_view(
                    raw=raw,
                    roi_live=roi_live,
                    roi_poly=self._roi_live_poly,
                    center_abs=center_abs,
                    contour_abs=contour_abs,
                    state=state,
                    status_text=text,
                    reason=f"{type(e).__name__}: {e}",
                )

            proc_payload = _build_proc_payload(
                raw=raw,
                proc_main=proc_full,
                baseplate_dbg=dbg,
                stab_info=self._stab_info,
                roi_live=roi_live,
                roi_poly=self._roi_live_poly,
                center_abs=center_abs,
                contour_abs=contour_abs,
                state=state,
                status_text=text,
                fps=self._fps,
            )

        # ----------------------------
        # Overlay
        # ----------------------------
        overlay = raw.copy()

        if self.settings.show_stab:
            draw_stab_debug(overlay, self._stab_info, settings=self.settings)

        if self.settings.show_baseplate:
            draw_baseplate_overlay(
                overlay,
                roi_live,
                center_rel,
                contour_rel,
                roi_poly=self._roi_live_poly,
            )

        draw_status_box(overlay, text, state=state)

        bottom = f"FPS: {self._fps:.1f} | Product: {self.recipe.name}"
        y = overlay.shape[0] - 18

        cv2.putText(overlay, bottom, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(overlay, bottom, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        out = QCFrameOutput(
            overlay_bgr=overlay,
            raw_bgr=raw,
            proc_bgr=proc_full,
            proc_payload=proc_payload,
            status_text=f"Status: {state} | {text}",
            state=state,
            roi_live=roi_live,
            stab_info=self._stab_info,
            center_rel=center_rel,
            angle=angle,
            detector_dbg=dbg if want_proc else None,
        )

        self._last_out = out
        return out