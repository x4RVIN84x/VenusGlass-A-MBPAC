# roi_stablizer.py (FULL REWRITE)
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import cv2
import numpy as np

import detector


# ----------------------------
# ROI / geometry helpers
# ----------------------------
def _roi_pad(roi, pad, W, H):
    x, y, w, h = map(int, roi)

    x2 = max(0, x - int(pad))
    y2 = max(0, y - int(pad))
    w2 = min(W - x2, w + 2 * int(pad))
    h2 = min(H - y2, h + 2 * int(pad))

    return (x2, y2, w2, h2)


def _as_float_line(line):
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


def _line_intersection(l1, l2):
    l1 = _as_float_line(l1)
    l2 = _as_float_line(l2)

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


def _point_on_line_at_y(line, y_target):
    line = _as_float_line(line)
    if line is None:
        return None

    vx, vy, x0, y0 = line

    if abs(vy) < 1e-9:
        return (float(x0), float(y0))

    t = (float(y_target) - y0) / vy
    return (float(x0 + vx * t), float(y_target))


def _point_on_line_at_x(line, x_target):
    line = _as_float_line(line)
    if line is None:
        return None

    vx, vy, x0, y0 = line

    if abs(vx) < 1e-9:
        return (float(x0), float(y0))

    t = (float(x_target) - x0) / vx
    return (float(x_target), float(y0 + vy * t))


def _anchor_point_is_good(pt) -> bool:
    if pt is None:
        return False

    try:
        x, y = float(pt[0]), float(pt[1])
    except Exception:
        return False

    return bool(np.isfinite(x) and np.isfinite(y))


def _to_xy_tuple(pt):
    return (float(pt[0]), float(pt[1]))


def _overlay_safe_anchors(anchors):
    return {
        "left_top": _to_xy_tuple(anchors["left_top"]),
        "right_top": _to_xy_tuple(anchors["right_top"]),
        "bottom_mid": _to_xy_tuple(anchors["bottom_mid"]),
    }


def _odd_int(value, *, minimum: int = 1) -> int:
    try:
        v = int(value)
    except Exception:
        v = int(minimum)

    v = max(int(minimum), v)

    if v % 2 == 0:
        v += 1

    return v


def _clip_float(value, lo: float, hi: float, default: float) -> float:
    try:
        v = float(value)
    except Exception:
        v = float(default)

    if not np.isfinite(v):
        v = float(default)

    return float(np.clip(v, lo, hi))


# ----------------------------
# Golden fallback from saved lines
# ----------------------------
def _derive_bottom_mid_from_side_lines(left, right, y_bottom):
    p_left = _point_on_line_at_y(left, y_bottom)
    p_right = _point_on_line_at_y(right, y_bottom)

    if not _anchor_point_is_good(p_left) or not _anchor_point_is_good(p_right):
        return None, None

    x_mid = 0.5 * (float(p_left[0]) + float(p_right[0]))
    bottom_mid = (float(x_mid), float(y_bottom))

    debug = {
        "left_at_bottom_y": _to_xy_tuple(p_left),
        "right_at_bottom_y": _to_xy_tuple(p_right),
    }

    return bottom_mid, debug


def _derive_bottom_mid_from_lines(left, right, bottom, *, max_iter: int = 3):
    left = _as_float_line(left)
    right = _as_float_line(right)
    bottom = _as_float_line(bottom)

    if left is None or right is None or bottom is None:
        return None, None

    y_bottom = float(bottom[3])
    debug = {}

    for _ in range(max(1, int(max_iter))):
        bottom_mid, side_dbg = _derive_bottom_mid_from_side_lines(left, right, y_bottom)
        if bottom_mid is None:
            return None, None

        p_on_bottom = _point_on_line_at_x(bottom, bottom_mid[0])
        if not _anchor_point_is_good(p_on_bottom):
            return None, None

        y_bottom = float(p_on_bottom[1])
        debug = {
            **(side_dbg or {}),
            "bottom_line_at_mid_x": _to_xy_tuple(p_on_bottom),
        }

    bottom_mid, side_dbg = _derive_bottom_mid_from_side_lines(left, right, y_bottom)
    if bottom_mid is None:
        return None, None

    debug = {
        **debug,
        **(side_dbg or {}),
        "source": "saved_bottom_line",
    }

    return bottom_mid, debug


def _extract_golden_anchors_from_lines(lines_abs, y_top_ref_abs):
    if not isinstance(lines_abs, dict):
        return None

    for k in ("left", "right", "bottom"):
        if k not in lines_abs or lines_abs[k] is None:
            return None

    left = _as_float_line(lines_abs["left"])
    right = _as_float_line(lines_abs["right"])
    bottom = _as_float_line(lines_abs["bottom"])

    if left is None or right is None or bottom is None:
        return None

    left_top = _point_on_line_at_y(left, y_top_ref_abs)
    right_top = _point_on_line_at_y(right, y_top_ref_abs)

    if not _anchor_point_is_good(left_top) or not _anchor_point_is_good(right_top):
        return None

    bottom_mid, bottom_mid_debug = _derive_bottom_mid_from_lines(left, right, bottom)
    if not _anchor_point_is_good(bottom_mid):
        return None

    notch_bottom_left = _line_intersection(left, bottom)

    return {
        "left_top": _to_xy_tuple(left_top),
        "right_top": _to_xy_tuple(right_top),
        "bottom_mid": _to_xy_tuple(bottom_mid),
        "notch_bottom_left": None if notch_bottom_left is None else _to_xy_tuple(notch_bottom_left),
        "bottom_mid_debug": bottom_mid_debug,
        "source": "golden_saved_lines",
    }


# ----------------------------
# Hysteresis
# ----------------------------
def _decompose_similarity(M):
    if M is None:
        return None

    M = np.asarray(M, dtype=np.float64)

    if M.shape != (2, 3):
        return None

    if not np.isfinite(M).all():
        return None

    a = float(M[0, 0])
    b = float(M[0, 1])
    tx = float(M[0, 2])
    ty = float(M[1, 2])

    scale = float(np.sqrt(a * a + b * b))
    angle = float(np.degrees(np.arctan2(b, a)))

    return {
        "tx": tx,
        "ty": ty,
        "scale": scale,
        "angle_deg": angle,
    }


def _blend_matrix(M_old, M_new, alpha):
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return ((1.0 - alpha) * M_old.astype(np.float64) + alpha * M_new.astype(np.float64)).astype(np.float64)


@dataclass
class TopAnchorHysteresis:
    """
    Smooths the legacy top anchor span used by the fallback affine crop transform.

    The preferred bottom-frame detector avoids these top anchors entirely.
    This class only matters if the dark-region bottom-frame path fails and the
    app falls back to the old left_top/right_top/bottom_mid model.
    """

    enabled: bool = True

    mid_blend_alpha: float = 0.12
    width_blend_alpha: float = 0.05
    y_blend_alpha: float = 0.08

    max_mid_jump_px: float = 28.0
    max_width_jump_px: float = 38.0
    max_y_jump_px: float = 18.0

    hold_frames: int = 8

    _last_mid_x: Optional[float] = field(default=None, init=False)
    _last_width: Optional[float] = field(default=None, init=False)
    _last_y: Optional[float] = field(default=None, init=False)
    _held_count: int = field(default=0, init=False)
    _last_info: Dict[str, Any] = field(default_factory=dict, init=False)

    def reset(self):
        self._last_mid_x = None
        self._last_width = None
        self._last_y = None
        self._held_count = 0
        self._last_info = {}

    def update(self, left_top, right_top):
        lx, ly = float(left_top[0]), float(left_top[1])
        rx, ry = float(right_top[0]), float(right_top[1])

        raw_mid_x = 0.5 * (lx + rx)
        raw_width = abs(rx - lx)
        raw_y = 0.5 * (ly + ry)

        if not self.enabled:
            self._last_mid_x = raw_mid_x
            self._last_width = raw_width
            self._last_y = raw_y
            self._held_count = 0
            self._last_info = {
                "mode": "disabled",
                "accepted": True,
                "raw_mid_x": raw_mid_x,
                "raw_width": raw_width,
                "raw_y": raw_y,
                "mid_x": raw_mid_x,
                "width": raw_width,
                "y": raw_y,
            }
            return self._points(), self._last_info

        if self._last_mid_x is None or self._last_width is None or self._last_y is None:
            self._last_mid_x = raw_mid_x
            self._last_width = raw_width
            self._last_y = raw_y
            self._held_count = 0
            self._last_info = {
                "mode": "init",
                "accepted": True,
                "raw_mid_x": raw_mid_x,
                "raw_width": raw_width,
                "raw_y": raw_y,
                "mid_x": self._last_mid_x,
                "width": self._last_width,
                "y": self._last_y,
            }
            return self._points(), self._last_info

        mid_jump = abs(raw_mid_x - self._last_mid_x)
        width_jump = abs(raw_width - self._last_width)
        y_jump = abs(raw_y - self._last_y)

        ok_mid = mid_jump <= float(self.max_mid_jump_px)
        ok_width = width_jump <= float(self.max_width_jump_px)
        ok_y = y_jump <= float(self.max_y_jump_px)
        force_accept = self._held_count >= int(max(1, self.hold_frames))

        if ok_mid and ok_width and ok_y:
            ma = float(np.clip(self.mid_blend_alpha, 0.0, 1.0))
            wa = float(np.clip(self.width_blend_alpha, 0.0, 1.0))
            ya = float(np.clip(self.y_blend_alpha, 0.0, 1.0))

            self._last_mid_x = (1.0 - ma) * self._last_mid_x + ma * raw_mid_x
            self._last_width = (1.0 - wa) * self._last_width + wa * raw_width
            self._last_y = (1.0 - ya) * self._last_y + ya * raw_y
            self._held_count = 0

            self._last_info = {
                "mode": "blend_accept",
                "accepted": True,
                "raw_mid_x": raw_mid_x,
                "raw_width": raw_width,
                "raw_y": raw_y,
                "mid_x": self._last_mid_x,
                "width": self._last_width,
                "y": self._last_y,
                "mid_jump_px": float(mid_jump),
                "width_jump_px": float(width_jump),
                "y_jump_px": float(y_jump),
                "mid_alpha": ma,
                "width_alpha": wa,
                "y_alpha": ya,
            }

        elif force_accept:
            self._last_mid_x = raw_mid_x
            self._last_width = raw_width
            self._last_y = raw_y
            self._held_count = 0

            self._last_info = {
                "mode": "force_accept_after_hold",
                "accepted": True,
                "raw_mid_x": raw_mid_x,
                "raw_width": raw_width,
                "raw_y": raw_y,
                "mid_x": self._last_mid_x,
                "width": self._last_width,
                "y": self._last_y,
                "mid_jump_px": float(mid_jump),
                "width_jump_px": float(width_jump),
                "y_jump_px": float(y_jump),
            }

        else:
            self._held_count += 1
            self._last_info = {
                "mode": "hold_last",
                "accepted": False,
                "raw_mid_x": raw_mid_x,
                "raw_width": raw_width,
                "raw_y": raw_y,
                "mid_x": self._last_mid_x,
                "width": self._last_width,
                "y": self._last_y,
                "mid_jump_px": float(mid_jump),
                "width_jump_px": float(width_jump),
                "y_jump_px": float(y_jump),
                "held_count": int(self._held_count),
            }

        return self._points(), self._last_info

    def _points(self):
        left = (self._last_mid_x - self._last_width * 0.5, self._last_y)
        right = (self._last_mid_x + self._last_width * 0.5, self._last_y)
        return left, right


@dataclass
class ContourHysteresis:
    enabled: bool = True

    blend_alpha: float = 0.24
    micro_blend_alpha: float = 0.12
    micro_translation_px: float = 5.0
    micro_angle_deg: float = 0.45

    max_translation_jump_px: float = 36.0
    max_angle_jump_deg: float = 4.0
    max_scale_jump_frac: float = 0.045

    hold_frames: int = 8

    _last_M: Optional[np.ndarray] = field(default=None, init=False)
    _last_info: Dict[str, Any] = field(default_factory=dict, init=False)
    _held_count: int = field(default=0, init=False)

    def reset(self):
        self._last_M = None
        self._last_info = {}
        self._held_count = 0

    def update(self, M_candidate, *, confidence: Optional[Dict[str, Any]] = None):
        confidence = dict(confidence or {})

        if M_candidate is None:
            self._last_info = {
                "mode": "missing_candidate",
                "accepted": False,
                **confidence,
            }
            return self._last_M, self._last_info

        M_candidate = np.asarray(M_candidate, dtype=np.float64)

        if not self.enabled:
            self._last_M = M_candidate.copy()
            self._held_count = 0
            self._last_info = {
                "mode": "disabled",
                "accepted": True,
                "held_count": 0,
                **confidence,
            }
            return self._last_M.copy(), self._last_info

        cand = _decompose_similarity(M_candidate)
        if cand is None:
            self._last_info = {
                "mode": "bad_candidate",
                "accepted": False,
                **confidence,
            }
            return self._last_M, self._last_info

        if self._last_M is None:
            self._last_M = M_candidate.copy()
            self._held_count = 0
            self._last_info = {
                "mode": "init",
                "accepted": True,
                "held_count": 0,
                "candidate": cand,
                **confidence,
            }
            return self._last_M.copy(), self._last_info

        last = _decompose_similarity(self._last_M)
        if last is None:
            self._last_M = M_candidate.copy()
            self._held_count = 0
            self._last_info = {
                "mode": "reinit_after_bad_last",
                "accepted": True,
                "held_count": 0,
                "candidate": cand,
                **confidence,
            }
            return self._last_M.copy(), self._last_info

        dtx = cand["tx"] - last["tx"]
        dty = cand["ty"] - last["ty"]
        dtrans = float(np.hypot(dtx, dty))
        dangle = float(abs(cand["angle_deg"] - last["angle_deg"]))
        dscale = float(abs(cand["scale"] - last["scale"]) / max(1e-6, abs(last["scale"])))

        ok_translation = dtrans <= float(self.max_translation_jump_px)
        ok_angle = dangle <= float(self.max_angle_jump_deg)
        ok_scale = dscale <= float(self.max_scale_jump_frac)
        force_accept = self._held_count >= int(max(1, self.hold_frames))

        is_micro_motion = (
            dtrans <= float(self.micro_translation_px)
            and dangle <= float(self.micro_angle_deg)
            and dscale <= float(self.max_scale_jump_frac)
        )

        if ok_translation and ok_angle and ok_scale:
            alpha = self.micro_blend_alpha if is_micro_motion else self.blend_alpha
            M_out = _blend_matrix(self._last_M, M_candidate, alpha)

            self._last_M = M_out.copy()
            self._held_count = 0
            self._last_info = {
                "mode": "micro_blend_accept" if is_micro_motion else "blend_accept",
                "accepted": True,
                "held_count": 0,
                "alpha": float(alpha),
                "dtranslation_px": float(dtrans),
                "dangle_deg": float(dangle),
                "dscale_frac": float(dscale),
                "candidate": cand,
                "last": last,
                **confidence,
            }
            return self._last_M.copy(), self._last_info

        if force_accept:
            self._last_M = M_candidate.copy()
            self._held_count = 0
            self._last_info = {
                "mode": "force_accept_after_hold",
                "accepted": True,
                "held_count": 0,
                "dtranslation_px": float(dtrans),
                "dangle_deg": float(dangle),
                "dscale_frac": float(dscale),
                "candidate": cand,
                "last": last,
                **confidence,
            }
            return self._last_M.copy(), self._last_info

        self._held_count += 1
        self._last_info = {
            "mode": "hold_last",
            "accepted": False,
            "held_count": int(self._held_count),
            "dtranslation_px": float(dtrans),
            "dangle_deg": float(dangle),
            "dscale_frac": float(dscale),
            "candidate": cand,
            "last": last,
            **confidence,
        }
        return self._last_M.copy(), self._last_info


_DEFAULT_TOP_ANCHOR_HYSTERESIS = TopAnchorHysteresis()
_DEFAULT_CONTOUR_HYSTERESIS = ContourHysteresis()


def reset_default_contour_hysteresis():
    _DEFAULT_TOP_ANCHOR_HYSTERESIS.reset()
    _DEFAULT_CONTOUR_HYSTERESIS.reset()


# ----------------------------
# Transform
# ----------------------------
def estimate_similarity_from_anchors(anchors_g, anchors_c):
    src = np.array(
        [
            anchors_g["left_top"],
            anchors_g["right_top"],
            anchors_g["bottom_mid"],
        ],
        dtype=np.float32,
    )

    dst = np.array(
        [
            anchors_c["left_top"],
            anchors_c["right_top"],
            anchors_c["bottom_mid"],
        ],
        dtype=np.float32,
    )

    M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    return M, inliers


def apply_affine_to_roi(M, roi_xywh):
    x, y, w, h = map(float, roi_xywh)

    corners = np.array(
        [
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h],
        ],
        dtype=np.float32,
    )

    ones = np.ones((4, 1), dtype=np.float32)
    pts = np.hstack([corners, ones])
    warped = (M.astype(np.float32) @ pts.T).T

    x0 = float(np.min(warped[:, 0]))
    y0 = float(np.min(warped[:, 1]))
    x1 = float(np.max(warped[:, 0]))
    y1 = float(np.max(warped[:, 1]))

    return (
        int(round(x0)),
        int(round(y0)),
        int(round(x1 - x0)),
        int(round(y1 - y0)),
    )


def _detect_dark_notch(
    image,
    roi,
    *,
    blur_ksize=21,
    close_ksize=11,
    open_ksize=7,
    threshold_bias=1.0,
):
    detect_dark_region = getattr(detector, "detect_notch_dark_region_frame_local", None)
    if not callable(detect_dark_region):
        # Recovered branches can contain the legacy stabilizer beside a newer
        # detector module.  Treat the unavailable optional path as a normal
        # miss so the proven line-based fallback below can still stabilize.
        return {
            "ok": False,
            "reason": "dark_region_detector_unavailable",
        }

    return detect_dark_region(
        image,
        roi,
        blur_ksize=_odd_int(blur_ksize, minimum=3),
        close_ksize=_odd_int(close_ksize, minimum=1),
        open_ksize=_odd_int(open_ksize, minimum=1),
        threshold_bias=float(threshold_bias),
    )


def _legacy_current_anchors_from_lines(legacy_info, y_top_ref_c):
    if not isinstance(legacy_info, dict) or not legacy_info.get("ok"):
        return None

    lines = legacy_info.get("lines") or {}

    left = _as_float_line(lines.get("left"))
    right = _as_float_line(lines.get("right"))
    bottom = _as_float_line(lines.get("bottom"))

    if left is None or right is None or bottom is None:
        return None

    left_top = _point_on_line_at_y(left, y_top_ref_c)
    right_top = _point_on_line_at_y(right, y_top_ref_c)
    bottom_mid, bottom_dbg = _derive_bottom_mid_from_lines(left, right, bottom)

    if (
        not _anchor_point_is_good(left_top)
        or not _anchor_point_is_good(right_top)
        or not _anchor_point_is_good(bottom_mid)
    ):
        return None

    return {
        "left_top": _to_xy_tuple(left_top),
        "right_top": _to_xy_tuple(right_top),
        "bottom_mid": _to_xy_tuple(bottom_mid),
        "bottom_mid_debug": bottom_dbg,
        "source": "legacy_dot_band_lines",
    }


# ----------------------------
# Bottom-frame ROI mapping
# ----------------------------
def _norm2(v):
    v = np.asarray(v, dtype=np.float64).reshape(2)
    n = float(np.linalg.norm(v))

    if n < 1e-6 or not np.isfinite(n):
        return None

    return v / n


def _bottom_frame_from_notch_frame(notch_frame):
    """
    Builds a stable local coordinate frame from the actual fitted notch geometry.

    Origin:
      midpoint of left/right wall intersections with the bottom reference line.

    X axis:
      bottom-left -> bottom-right.

    Y axis:
      perpendicular to the bottom reference line, pointing upward in image coords.

    This deliberately does NOT use old top anchors / top_y.
    """
    if not isinstance(notch_frame, dict):
        return None

    left = notch_frame.get("left_line")
    right = notch_frame.get("right_line")
    bottom = notch_frame.get("bottom_line")

    bl = _line_intersection(left, bottom)
    br = _line_intersection(right, bottom)

    if not _anchor_point_is_good(bl) or not _anchor_point_is_good(br):
        return None

    bl = np.asarray(bl, dtype=np.float64)
    br = np.asarray(br, dtype=np.float64)

    x_axis = _norm2(br - bl)
    if x_axis is None:
        return None

    width = float(np.linalg.norm(br - bl))
    if width < 10 or not np.isfinite(width):
        return None

    # Image coordinates: y grows downward. This points upward when x_axis points right.
    y_axis = np.asarray([x_axis[1], -x_axis[0]], dtype=np.float64)
    if y_axis[1] > 0:
        y_axis = -y_axis

    origin = 0.5 * (bl + br)

    return {
        "origin": (float(origin[0]), float(origin[1])),
        "bottom_left": (float(bl[0]), float(bl[1])),
        "bottom_right": (float(br[0]), float(br[1])),
        "bottom_mid": (float(origin[0]), float(origin[1])),
        "x_axis": (float(x_axis[0]), float(x_axis[1])),
        "y_axis": (float(y_axis[0]), float(y_axis[1])),
        "width": float(width),
    }


def _frame_local_coords(pt, frame):
    p = np.asarray(pt, dtype=np.float64).reshape(2)
    o = np.asarray(frame["origin"], dtype=np.float64)
    x = np.asarray(frame["x_axis"], dtype=np.float64)
    y = np.asarray(frame["y_axis"], dtype=np.float64)
    d = p - o

    return float(np.dot(d, x)), float(np.dot(d, y))


def _frame_point_from_local(local_xy, frame):
    lx, ly = map(float, local_xy)
    o = np.asarray(frame["origin"], dtype=np.float64)
    x = np.asarray(frame["x_axis"], dtype=np.float64)
    y = np.asarray(frame["y_axis"], dtype=np.float64)

    p = o + lx * x + ly * y

    return (float(p[0]), float(p[1]))


def _apply_bottom_frame_to_roi(roi_xywh, frame_g, frame_c):
    """
    Transform ROI corners from golden frame to current frame using the stable
    bottom-frame coordinate system.

    Uses bottom width ratio as a similarity scale.
    """
    if frame_g is None or frame_c is None:
        return None

    wg = float(frame_g.get("width", 0.0))
    wc = float(frame_c.get("width", 0.0))

    if wg < 1e-6 or wc < 1e-6:
        return None

    scale = wc / wg

    if not np.isfinite(scale) or scale <= 0:
        return None

    x, y, w, h = map(float, roi_xywh)

    corners = [
        (x, y),
        (x + w, y),
        (x + w, y + h),
        (x, y + h),
    ]

    warped = []

    for p in corners:
        lx, ly = _frame_local_coords(p, frame_g)
        warped.append(_frame_point_from_local((lx * scale, ly * scale), frame_c))

    arr = np.asarray(warped, dtype=np.float64)

    x0 = float(np.min(arr[:, 0]))
    y0 = float(np.min(arr[:, 1]))
    x1 = float(np.max(arr[:, 0]))
    y1 = float(np.max(arr[:, 1]))

    return (
        int(round(x0)),
        int(round(y0)),
        int(round(x1 - x0)),
        int(round(y1 - y0)),
    )


def _bottom_frame_to_overlay_anchors(frame):
    if not isinstance(frame, dict):
        return {}

    out = {}

    for k in ("bottom_left", "bottom_right", "bottom_mid"):
        pt = frame.get(k)
        if _anchor_point_is_good(pt):
            out[k] = _to_xy_tuple(pt)

    return out


# ----------------------------
# Main API
# ----------------------------
def stabilize_rois_using_saved_inner_border_lines(
    *,
    current_img,
    golden_img,
    registration_roi_golden,
    golden_inner_lines_abs,
    rois_golden,
    search_padding_px=120,
    canny_low=60,
    canny_high=140,
    y_top_offset_px=0.0,

    prefer_dark_region: bool = True,
    prefer_solid_edge: bool = True,

    notch_blur_ksize: int = 21,
    notch_close_ksize: int = 11,
    notch_open_ksize: int = 7,
    notch_threshold_bias: float = 1.0,
    notch_bottom_band_frac: float = 0.35,
    notch_side_band_frac: float = 0.35,

    contour_hysteresis: Optional[ContourHysteresis] = None,
    hysteresis_enabled: bool = True,
):
    """
    Stabilize ROIs by mapping the golden product ROI into the current frame.

    Preferred path:
      golden dark-region notch frame
      -> current dark-region notch frame
      -> bottom-frame similarity transform
      -> moved ROI

    Fallback path:
      old left_top/right_top/bottom_mid affine using detector edge bands.

    Product JSON parameters enter here through engine.py. The numeric defaults
    are only emergency fallbacks for old JSON files or direct test calls.
    """
    if current_img is None or golden_img is None:
        return None, {
            "ok": False,
            "reason": "missing_images",
        }

    Hc, Wc = current_img.shape[:2]

    reg_g = tuple(map(int, registration_roi_golden))
    y_top_ref_g = float(reg_g[1]) + float(y_top_offset_px)
    y_top_ref_c = float(reg_g[1]) + float(y_top_offset_px)

    reg_search = _roi_pad(reg_g, search_padding_px, Wc, Hc)

    # Sanitize product/JSON parameters once.
    notch_blur_ksize = _odd_int(notch_blur_ksize, minimum=3)
    notch_close_ksize = _odd_int(notch_close_ksize, minimum=1)
    notch_open_ksize = _odd_int(notch_open_ksize, minimum=1)
    notch_threshold_bias = float(notch_threshold_bias)

    notch_bottom_band_frac = _clip_float(notch_bottom_band_frac, 0.05, 0.80, 0.35)
    notch_side_band_frac = _clip_float(notch_side_band_frac, 0.05, 0.80, 0.35)

    param_debug = {
        "prefer_dark_region": bool(prefer_dark_region),
        "prefer_solid_edge": bool(prefer_solid_edge),
        "notch_blur_ksize": int(notch_blur_ksize),
        "notch_close_ksize": int(notch_close_ksize),
        "notch_open_ksize": int(notch_open_ksize),
        "notch_threshold_bias": float(notch_threshold_bias),
        "notch_bottom_band_frac": float(notch_bottom_band_frac),
        "notch_side_band_frac": float(notch_side_band_frac),
        "legacy_band_bottom_frac_used": float(notch_bottom_band_frac),
        "legacy_band_side_frac_used": float(notch_side_band_frac),
    }

    golden_dark = None
    current_dark = None
    legacy_info = None
    lines_g = None

    current_method = None
    moved = None

    golden_notch_frame = None
    current_notch_frame = None
    golden_bottom_frame = None
    current_bottom_frame = None

    # --- Preferred: actual dark-region notch contour on golden and current ---
    if bool(prefer_dark_region):
        golden_dark = _detect_dark_notch(
            golden_img,
            reg_g,
            blur_ksize=notch_blur_ksize,
            close_ksize=notch_close_ksize,
            open_ksize=notch_open_ksize,
            threshold_bias=notch_threshold_bias,
        )

        current_dark = _detect_dark_notch(
            current_img,
            reg_search,
            blur_ksize=notch_blur_ksize,
            close_ksize=notch_close_ksize,
            open_ksize=notch_open_ksize,
            threshold_bias=notch_threshold_bias,
        )

        if isinstance(golden_dark, dict):
            golden_dark["notch_param_debug"] = dict(param_debug)

        if isinstance(current_dark, dict):
            current_dark["notch_param_debug"] = dict(param_debug)

        if isinstance(golden_dark, dict) and golden_dark.get("ok"):
            golden_notch_frame = golden_dark.get("notch_frame")
            golden_bottom_frame = _bottom_frame_from_notch_frame(golden_notch_frame)

        if isinstance(current_dark, dict) and current_dark.get("ok"):
            current_notch_frame = current_dark.get("notch_frame")
            current_bottom_frame = _bottom_frame_from_notch_frame(current_notch_frame)

        if golden_bottom_frame is not None and current_bottom_frame is not None:
            moved = [
                _apply_bottom_frame_to_roi(r, golden_bottom_frame, current_bottom_frame)
                for r in rois_golden
            ]

            if all(m is not None for m in moved):
                current_method = "dark_region_bottom_frame"
            else:
                moved = None

    # --- Legacy fallback: old 3-anchor affine if bottom-frame path fails ---
    legacy_anchors_g = None
    legacy_anchors_c = None
    anchors_g = None
    anchors_c = None
    M_raw = None
    M_stable = None
    inliers = None
    top_anchor_hyst_info = {
        "mode": "unused_bottom_frame",
        "accepted": True,
    }
    hysteresis_info = {
        "mode": "unused_bottom_frame",
        "accepted": True,
    }

    if moved is None:
        if not bool(prefer_solid_edge):
            return None, {
                "ok": False,
                "reason": "dark_region_failed_and_solid_edge_fallback_disabled",
                "registration_roi_golden": reg_g,
                "search_roi_current": reg_search,
                "golden_dark_debug": golden_dark,
                "current_dark_debug": current_dark,
                "notch_param_debug": param_debug,
            }

        try:
            lines_g = {
                k: tuple(map(float, golden_inner_lines_abs[k]))
                for k in ("left", "right", "bottom")
            }
        except Exception:
            lines_g = None

        if lines_g is not None:
            legacy_anchors_g = _extract_golden_anchors_from_lines(lines_g, y_top_ref_g)

        legacy_info = detector.detect_inner_border_lines_edges_local(
            current_img,
            reg_search,
            canny_low=int(canny_low),
            canny_high=int(canny_high),
            min_points=40,
            band_side_frac=float(notch_side_band_frac),
            band_bottom_frac=float(notch_bottom_band_frac),
            sample_stride=1,
        )

        if isinstance(legacy_info, dict):
            legacy_info["notch_param_debug"] = dict(param_debug)

        legacy_anchors_c = _legacy_current_anchors_from_lines(legacy_info, y_top_ref_c)

        if legacy_anchors_g is None:
            return None, {
                "ok": False,
                "reason": "golden_bottom_frame_and_legacy_anchors_failed",
                "registration_roi_golden": reg_g,
                "search_roi_current": reg_search,
                "golden_dark_debug": golden_dark,
                "current_dark_debug": current_dark,
                "legacy_lines_debug": legacy_info,
                "notch_param_debug": param_debug,
            }

        if legacy_anchors_c is None:
            return None, {
                "ok": False,
                "reason": (
                    "current_bottom_frame_and_legacy_anchors_failed: "
                    f"dark={None if current_dark is None else current_dark.get('reason')} "
                    f"legacy={None if legacy_info is None else legacy_info.get('reason')}"
                ),
                "registration_roi_golden": reg_g,
                "search_roi_current": reg_search,
                "golden_dark_debug": golden_dark,
                "current_dark_debug": current_dark,
                "legacy_lines_debug": legacy_info,
                "notch_param_debug": param_debug,
            }

        anchors_g = _overlay_safe_anchors(legacy_anchors_g)
        anchors_c = _overlay_safe_anchors(legacy_anchors_c)

        (stable_left_top, stable_right_top), top_anchor_hyst_info = _DEFAULT_TOP_ANCHOR_HYSTERESIS.update(
            anchors_c["left_top"],
            anchors_c["right_top"],
        )

        anchors_c = {
            **anchors_c,
            "left_top": _to_xy_tuple(stable_left_top),
            "right_top": _to_xy_tuple(stable_right_top),
        }

        M_raw, inliers = estimate_similarity_from_anchors(anchors_g, anchors_c)

        if M_raw is None:
            return None, {
                "ok": False,
                "reason": "legacy_transform_failed",
                "registration_roi_golden": reg_g,
                "search_roi_current": reg_search,
                "anchors_golden": anchors_g,
                "anchors_current": anchors_c,
                "current_anchor_method": "legacy_dot_band_lines",
                "current_dark_debug": current_dark,
                "legacy_lines_debug": legacy_info,
                "notch_param_debug": param_debug,
            }

        hyst = contour_hysteresis if contour_hysteresis is not None else _DEFAULT_CONTOUR_HYSTERESIS
        hyst.enabled = bool(hysteresis_enabled)

        M_stable, hysteresis_info = hyst.update(
            M_raw,
            confidence={
                "current_anchor_method": "legacy_dot_band_lines",
                "top_anchor_mode": top_anchor_hyst_info.get("mode"),
                "dark_reason": None if current_dark is None else current_dark.get("reason"),
                "legacy_reason": None if legacy_info is None else legacy_info.get("reason"),
                "notch_param_debug": dict(param_debug),
            },
        )

        if M_stable is None:
            M_stable = M_raw

        moved = [apply_affine_to_roi(M_stable, r) for r in rois_golden]
        current_method = "legacy_dot_band_lines"

    active_dbg = current_dark if current_method == "dark_region_bottom_frame" else legacy_info

    if not isinstance(active_dbg, dict):
        active_dbg = {}

    if lines_g is None:
        try:
            lines_g = {
                k: tuple(map(float, golden_inner_lines_abs[k]))
                for k in ("left", "right", "bottom")
            }
        except Exception:
            lines_g = None

    # Overlay anchors: bottom-frame path exposes only real bottom-frame points.
    if current_method == "dark_region_bottom_frame":
        frame_anchors_current = _bottom_frame_to_overlay_anchors(current_bottom_frame)
        frame_anchors_golden = _bottom_frame_to_overlay_anchors(golden_bottom_frame)

        anchors_current = frame_anchors_current
        anchors_golden = frame_anchors_golden
        anchors_current_raw = frame_anchors_current
        anchors_golden_raw = frame_anchors_golden
    else:
        frame_anchors_current = {}
        frame_anchors_golden = {}

        anchors_current = anchors_c or {}
        anchors_golden = anchors_g or {}
        anchors_current_raw = legacy_anchors_c or {}
        anchors_golden_raw = legacy_anchors_g or {}

    info = {
        "ok": True,
        "reason": "ok",

        "roi_model": "bottom_frame_similarity"
        if current_method == "dark_region_bottom_frame"
        else "legacy_three_anchor_affine",

        "M": None if M_stable is None else M_stable.tolist(),
        "M_raw": None if M_raw is None else M_raw.tolist(),
        "hysteresis": hysteresis_info,
        "inliers": None if inliers is None else inliers.astype(int).flatten().tolist(),

        "registration_roi_golden": reg_g,
        "search_roi_current": reg_search,

        "anchors_golden": anchors_golden,
        "anchors_current": anchors_current,
        "anchors_current_raw": anchors_current_raw,
        "anchors_golden_raw": anchors_golden_raw,

        "frame_anchors_current": frame_anchors_current,
        "frame_anchors_golden": frame_anchors_golden,

        "golden_bottom_frame": golden_bottom_frame,
        "current_bottom_frame": current_bottom_frame,

        "top_anchor_hysteresis": top_anchor_hyst_info,

        "anchor_model": "bottom_left/bottom_right/bottom_mid_frame"
        if current_method == "dark_region_bottom_frame"
        else "legacy_left_top/right_top/bottom_mid",

        "current_anchor_method": current_method,

        "current_notch_frame": current_notch_frame,
        "golden_notch_frame": golden_notch_frame,

        "current_dark_debug": current_dark,
        "golden_dark_debug": golden_dark,
        "legacy_lines_debug": legacy_info,
        "current_lines_debug": active_dbg,
        "golden_lines_abs": lines_g,

        # Important debug so you can prove product JSON values reached this module.
        "notch_param_debug": param_debug,
    }

    return moved, info
