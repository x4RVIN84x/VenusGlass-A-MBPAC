# roi_stablizer.py (FULL REWRITE)
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

import detector


# ----------------------------
# Basic geometry helpers
# ----------------------------
def _roi_pad(roi, pad, W, H):
    x, y, w, h = map(int, roi)
    p = int(pad)
    x2 = max(0, x - p)
    y2 = max(0, y - p)
    w2 = min(W - x2, w + 2 * p)
    h2 = min(H - y2, h + 2 * p)
    return (x2, y2, w2, h2)


def _as_float_line(line):
    if line is None:
        return None
    try:
        vals = tuple(map(float, line))
    except Exception:
        return None
    if len(vals) != 4 or not all(np.isfinite(vals)):
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
    det = np.linalg.det(A)
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


def _good_pt(pt) -> bool:
    try:
        return pt is not None and np.isfinite(float(pt[0])) and np.isfinite(float(pt[1]))
    except Exception:
        return False


def _xy(pt):
    return (float(pt[0]), float(pt[1]))


def _coerce_points_abs(points) -> Optional[np.ndarray]:
    if points is None:
        return None
    try:
        arr = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    except Exception:
        return None
    if arr.size == 0:
        return None
    arr = arr[np.isfinite(arr).all(axis=1)]
    return arr if len(arr) > 0 else None


# ----------------------------
# Golden/current anchors
# ----------------------------
def _derive_bottom_mid_from_side_lines(left, right, y_bottom):
    p_left = _point_on_line_at_y(left, y_bottom)
    p_right = _point_on_line_at_y(right, y_bottom)
    if not _good_pt(p_left) or not _good_pt(p_right):
        return None, None
    x_mid = 0.5 * (float(p_left[0]) + float(p_right[0]))
    return (float(x_mid), float(y_bottom)), {
        "left_at_bottom_y": _xy(p_left),
        "right_at_bottom_y": _xy(p_right),
    }


def _derive_bottom_mid_from_lines(left, right, bottom, *, max_iter: int = 3):
    left = _as_float_line(left)
    right = _as_float_line(right)
    bottom = _as_float_line(bottom)
    if left is None or right is None or bottom is None:
        return None, None

    y_bottom = float(bottom[3])
    debug = {}
    for _ in range(max(1, int(max_iter))):
        bm, side_dbg = _derive_bottom_mid_from_side_lines(left, right, y_bottom)
        if bm is None:
            return None, None
        p_bottom = _point_on_line_at_x(bottom, bm[0])
        if not _good_pt(p_bottom):
            return None, None
        y_bottom = float(p_bottom[1])
        debug = {**(side_dbg or {}), "bottom_line_at_mid_x": _xy(p_bottom)}

    bm, side_dbg = _derive_bottom_mid_from_side_lines(left, right, y_bottom)
    if bm is None:
        return None, None
    return bm, {**debug, **(side_dbg or {}), "source": "saved_bottom_line"}


def _golden_anchors_from_saved_lines(lines_abs, y_top_ref_abs):
    if not isinstance(lines_abs, dict):
        return None
    try:
        left = _as_float_line(lines_abs["left"])
        right = _as_float_line(lines_abs["right"])
        bottom = _as_float_line(lines_abs["bottom"])
    except Exception:
        return None
    if left is None or right is None or bottom is None:
        return None

    lt = _point_on_line_at_y(left, y_top_ref_abs)
    rt = _point_on_line_at_y(right, y_top_ref_abs)
    bm, bm_dbg = _derive_bottom_mid_from_lines(left, right, bottom)
    if not _good_pt(lt) or not _good_pt(rt) or not _good_pt(bm):
        return None

    return {
        "left_top": _xy(lt),
        "right_top": _xy(rt),
        "bottom_mid": _xy(bm),
        "source": "golden_saved_lines",
        "notch_bottom_left": None if _line_intersection(left, bottom) is None else _xy(_line_intersection(left, bottom)),
        "bottom_mid_debug": bm_dbg,
    }


def _golden_anchors_from_solid_edge(golden_img, registration_roi_golden, *, search_padding_px, canny_low, canny_high):
    if golden_img is None:
        return None, None
    Hg, Wg = golden_img.shape[:2]
    reg_search_g = _roi_pad(registration_roi_golden, search_padding_px, Wg, Hg)
    info = detector.detect_notch_solid_edge_anchors_local(
        golden_img,
        reg_search_g,
        canny_low=int(canny_low),
        canny_high=int(canny_high),
        blur_ksize=5,
        clahe_clip=1.5,
        close_iter=2,
        dilate_iter=1,
        min_component_area_px=180,
        min_anchor_points=18,
    )
    if isinstance(info, dict) and info.get("ok") and isinstance(info.get("anchors"), dict):
        a = {k: _xy(info["anchors"][k]) for k in ("left_top", "right_top", "bottom_mid")}
        a["source"] = "golden_solid_edge"
        return a, info
    return None, info


def _robust_bottom_y_from_cluster(points_abs, *, lower_percentile: float = 60.0):
    pts = _coerce_points_abs(points_abs)
    if pts is None:
        return None
    ys = pts[:, 1].astype(np.float32)
    cut = float(np.percentile(ys, float(np.clip(lower_percentile, 0.0, 95.0))))
    lower = ys[ys >= cut]
    if len(lower) <= 0:
        lower = ys
    y = float(np.median(lower))
    return y if np.isfinite(y) else None


def _current_anchors_from_legacy_lines(lines_abs, y_top_ref_abs, *, bottom_points_abs=None):
    if not isinstance(lines_abs, dict):
        return None
    try:
        left = _as_float_line(lines_abs["left"])
        right = _as_float_line(lines_abs["right"])
        bottom = _as_float_line(lines_abs["bottom"])
    except Exception:
        return None
    if left is None or right is None or bottom is None:
        return None

    lt = _point_on_line_at_y(left, y_top_ref_abs)
    rt = _point_on_line_at_y(right, y_top_ref_abs)
    yb = _robust_bottom_y_from_cluster(bottom_points_abs)
    if yb is not None:
        bm, bm_dbg = _derive_bottom_mid_from_side_lines(left, right, yb)
        if bm is not None:
            bm_dbg = {"source": "legacy_bottom_cluster", **(bm_dbg or {})}
        else:
            bm, bm_dbg = _derive_bottom_mid_from_lines(left, right, bottom)
    else:
        bm, bm_dbg = _derive_bottom_mid_from_lines(left, right, bottom)

    if not _good_pt(lt) or not _good_pt(rt) or not _good_pt(bm):
        return None

    return {
        "left_top": _xy(lt),
        "right_top": _xy(rt),
        "bottom_mid": _xy(bm),
        "source": "legacy_dot_band_lines",
        "notch_bottom_left": None if _line_intersection(left, bottom) is None else _xy(_line_intersection(left, bottom)),
        "bottom_mid_debug": bm_dbg,
    }


def _overlay_safe_anchors(anchors):
    return {
        "left_top": _xy(anchors["left_top"]),
        "right_top": _xy(anchors["right_top"]),
        "bottom_mid": _xy(anchors["bottom_mid"]),
    }


# ----------------------------
# Hysteresis
# ----------------------------
def _decompose_similarity(M):
    if M is None:
        return None
    M = np.asarray(M, dtype=np.float64)
    if M.shape != (2, 3) or not np.isfinite(M).all():
        return None
    a = float(M[0, 0])
    b = float(M[0, 1])
    return {
        "tx": float(M[0, 2]),
        "ty": float(M[1, 2]),
        "scale": float(np.sqrt(a * a + b * b)),
        "angle_deg": float(np.degrees(np.arctan2(b, a))),
    }


def _blend_matrix(M_old, M_new, alpha):
    a = float(np.clip(alpha, 0.0, 1.0))
    return ((1.0 - a) * M_old.astype(np.float64) + a * M_new.astype(np.float64)).astype(np.float64)


@dataclass
class TopAnchorHysteresis:
    """
    Stabilizes top anchors by smoothing:
      - top midpoint X
      - top span width
      - shared top Y

    This directly fixes the top notch anchors moving back/forth and dragging the ROI.
    """

    enabled: bool = True
    mid_blend_alpha: float = 0.16
    width_blend_alpha: float = 0.06
    y_blend_alpha: float = 0.10
    max_mid_jump_px: float = 26.0
    max_width_jump_px: float = 34.0
    max_y_jump_px: float = 18.0
    hold_frames: int = 8

    _last_mid_x: Optional[float] = field(default=None, init=False)
    _last_width: Optional[float] = field(default=None, init=False)
    _last_top_y: Optional[float] = field(default=None, init=False)
    _held_count: int = field(default=0, init=False)
    _last_info: Dict[str, Any] = field(default_factory=dict, init=False)

    def reset(self):
        self._last_mid_x = None
        self._last_width = None
        self._last_top_y = None
        self._held_count = 0
        self._last_info = {}

    def update(self, left_top, right_top):
        lx, ly = float(left_top[0]), float(left_top[1])
        rx, ry = float(right_top[0]), float(right_top[1])
        raw_mid_x = 0.5 * (lx + rx)
        raw_width = abs(rx - lx)
        raw_top_y = 0.5 * (ly + ry)

        if not self.enabled:
            self._last_mid_x = raw_mid_x
            self._last_width = raw_width
            self._last_top_y = raw_top_y
            self._held_count = 0
            self._last_info = {"mode": "disabled", "accepted": True, "raw_mid_x": raw_mid_x, "raw_width": raw_width, "raw_top_y": raw_top_y}
            return self._make_points(), self._last_info

        if self._last_mid_x is None or self._last_width is None or self._last_top_y is None:
            self._last_mid_x = raw_mid_x
            self._last_width = raw_width
            self._last_top_y = raw_top_y
            self._held_count = 0
            self._last_info = {"mode": "init", "accepted": True, "raw_mid_x": raw_mid_x, "raw_width": raw_width, "raw_top_y": raw_top_y}
            return self._make_points(), self._last_info

        mid_jump = abs(raw_mid_x - self._last_mid_x)
        width_jump = abs(raw_width - self._last_width)
        y_jump = abs(raw_top_y - self._last_top_y)
        ok = (
            mid_jump <= float(self.max_mid_jump_px)
            and width_jump <= float(self.max_width_jump_px)
            and y_jump <= float(self.max_y_jump_px)
        )
        force_accept = self._held_count >= int(max(1, self.hold_frames))

        if ok:
            ma = float(np.clip(self.mid_blend_alpha, 0.0, 1.0))
            wa = float(np.clip(self.width_blend_alpha, 0.0, 1.0))
            ya = float(np.clip(self.y_blend_alpha, 0.0, 1.0))
            self._last_mid_x = (1.0 - ma) * self._last_mid_x + ma * raw_mid_x
            self._last_width = (1.0 - wa) * self._last_width + wa * raw_width
            self._last_top_y = (1.0 - ya) * self._last_top_y + ya * raw_top_y
            self._held_count = 0
            mode = "blend_accept"
            accepted = True
        elif force_accept:
            self._last_mid_x = raw_mid_x
            self._last_width = raw_width
            self._last_top_y = raw_top_y
            self._held_count = 0
            mode = "force_accept_after_hold"
            accepted = True
        else:
            self._held_count += 1
            mode = "hold_last"
            accepted = False

        self._last_info = {
            "mode": mode,
            "accepted": accepted,
            "raw_mid_x": raw_mid_x,
            "raw_width": raw_width,
            "raw_top_y": raw_top_y,
            "mid_x": self._last_mid_x,
            "width": self._last_width,
            "top_y": self._last_top_y,
            "mid_jump_px": float(mid_jump),
            "width_jump_px": float(width_jump),
            "y_jump_px": float(y_jump),
            "held_count": int(self._held_count),
        }
        return self._make_points(), self._last_info

    def _make_points(self):
        lt = (float(self._last_mid_x - 0.5 * self._last_width), float(self._last_top_y))
        rt = (float(self._last_mid_x + 0.5 * self._last_width), float(self._last_top_y))
        return lt, rt


@dataclass
class ContourHysteresis:
    enabled: bool = True
    blend_alpha: float = 0.28
    micro_blend_alpha: float = 0.16
    micro_translation_px: float = 5.0
    micro_angle_deg: float = 0.45
    max_translation_jump_px: float = 34.0
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
            self._last_info = {"mode": "missing_candidate", "accepted": False, **confidence}
            return self._last_M, self._last_info
        M_candidate = np.asarray(M_candidate, dtype=np.float64)

        if not self.enabled:
            self._last_M = M_candidate.copy()
            self._held_count = 0
            self._last_info = {"mode": "disabled", "accepted": True, "held_count": 0, **confidence}
            return self._last_M.copy(), self._last_info

        cand = _decompose_similarity(M_candidate)
        if cand is None:
            self._last_info = {"mode": "bad_candidate", "accepted": False, **confidence}
            return self._last_M, self._last_info

        if self._last_M is None:
            self._last_M = M_candidate.copy()
            self._held_count = 0
            self._last_info = {"mode": "init", "accepted": True, "held_count": 0, "candidate": cand, **confidence}
            return self._last_M.copy(), self._last_info

        last = _decompose_similarity(self._last_M)
        if last is None:
            self._last_M = M_candidate.copy()
            self._held_count = 0
            self._last_info = {"mode": "reinit_after_bad_last", "accepted": True, "candidate": cand, **confidence}
            return self._last_M.copy(), self._last_info

        dtx = cand["tx"] - last["tx"]
        dty = cand["ty"] - last["ty"]
        dtrans = float(np.hypot(dtx, dty))
        dangle = float(abs(cand["angle_deg"] - last["angle_deg"]))
        dscale = float(abs(cand["scale"] - last["scale"]) / max(1e-6, abs(last["scale"])))

        ok = (
            dtrans <= float(self.max_translation_jump_px)
            and dangle <= float(self.max_angle_jump_deg)
            and dscale <= float(self.max_scale_jump_frac)
        )
        force_accept = self._held_count >= int(max(1, self.hold_frames))
        micro = dtrans <= float(self.micro_translation_px) and dangle <= float(self.micro_angle_deg)

        if ok:
            alpha = self.micro_blend_alpha if micro else self.blend_alpha
            self._last_M = _blend_matrix(self._last_M, M_candidate, alpha)
            self._held_count = 0
            mode = "micro_blend_accept" if micro else "blend_accept"
            accepted = True
        elif force_accept:
            self._last_M = M_candidate.copy()
            self._held_count = 0
            alpha = 1.0
            mode = "force_accept_after_hold"
            accepted = True
        else:
            self._held_count += 1
            alpha = 0.0
            mode = "hold_last"
            accepted = False

        self._last_info = {
            "mode": mode,
            "accepted": accepted,
            "held_count": int(self._held_count),
            "alpha": float(alpha),
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
# Transform / ROI
# ----------------------------
def estimate_similarity_from_anchors(anchors_g, anchors_c):
    src = np.array([anchors_g["left_top"], anchors_g["right_top"], anchors_g["bottom_mid"]], dtype=np.float32)
    dst = np.array([anchors_c["left_top"], anchors_c["right_top"], anchors_c["bottom_mid"]], dtype=np.float32)
    M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    return M, inliers


def apply_affine_to_roi(M, roi_xywh):
    x, y, w, h = map(float, roi_xywh)
    corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
    pts = np.hstack([corners, np.ones((4, 1), dtype=np.float32)])
    warped = (M.astype(np.float32) @ pts.T).T
    x0, y0 = float(np.min(warped[:, 0])), float(np.min(warped[:, 1]))
    x1, y1 = float(np.max(warped[:, 0])), float(np.max(warped[:, 1]))
    return (int(round(x0)), int(round(y0)), int(round(x1 - x0)), int(round(y1 - y0)))


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
    prefer_solid_edge: bool = True,
    contour_hysteresis: Optional[ContourHysteresis] = None,
    hysteresis_enabled: bool = True,
):
    if current_img is None or golden_img is None:
        return None, {"ok": False, "reason": "missing_images"}

    Hc, Wc = current_img.shape[:2]
    Hg, Wg = golden_img.shape[:2]
    reg_g = tuple(map(int, registration_roi_golden))
    y_top_ref_g = float(reg_g[1]) + float(y_top_offset_px)
    y_top_ref_c = float(reg_g[1]) + float(y_top_offset_px)

    # Golden anchors: prefer same solid-edge definition as current.
    golden_solid_info = None
    anchors_g_raw = None
    golden_anchor_method = "none"
    if bool(prefer_solid_edge):
        anchors_g_raw, golden_solid_info = _golden_anchors_from_solid_edge(
            golden_img,
            reg_g,
            search_padding_px=int(search_padding_px),
            canny_low=int(canny_low),
            canny_high=int(canny_high),
        )
        if anchors_g_raw is not None:
            golden_anchor_method = "solid_edge"

    if anchors_g_raw is None:
        try:
            lines_g = {k: tuple(map(float, golden_inner_lines_abs[k])) for k in ("left", "right", "bottom")}
        except Exception:
            return None, {"ok": False, "reason": "bad_golden_inner_lines_abs", "golden_solid_debug": golden_solid_info}
        anchors_g_raw = _golden_anchors_from_saved_lines(lines_g, y_top_ref_g)
        golden_anchor_method = "saved_lines"
        if anchors_g_raw is None:
            return None, {"ok": False, "reason": "golden_anchors_failed", "golden_solid_debug": golden_solid_info}
    else:
        lines_g = golden_inner_lines_abs

    anchors_g = _overlay_safe_anchors(anchors_g_raw)

    # Current anchors.
    reg_search = _roi_pad(reg_g, search_padding_px, Wc, Hc)
    solid_info = None
    legacy_info = None
    current_debug_info = None
    anchors_c_raw = None
    current_method = None

    if bool(prefer_solid_edge):
        solid_info = detector.detect_notch_solid_edge_anchors_local(
            current_img,
            reg_search,
            canny_low=int(canny_low),
            canny_high=int(canny_high),
            blur_ksize=5,
            clahe_clip=1.5,
            close_iter=2,
            dilate_iter=1,
            min_component_area_px=180,
            min_anchor_points=18,
        )
        if isinstance(solid_info, dict) and solid_info.get("ok") and isinstance(solid_info.get("anchors"), dict):
            anchors_c_raw = {k: _xy(solid_info["anchors"][k]) for k in ("left_top", "right_top", "bottom_mid")}
            anchors_c_raw["source"] = "solid_edge"
            current_method = "solid_edge"
            current_debug_info = solid_info

    if anchors_c_raw is None:
        legacy_info = detector.detect_inner_border_lines_edges_local(
            current_img,
            reg_search,
            canny_low=int(canny_low),
            canny_high=int(canny_high),
            min_points=40,
            band_side_frac=0.22,
            band_bottom_frac=0.25,
            sample_stride=1,
        )
        if not legacy_info.get("ok", False):
            return None, {
                "ok": False,
                "reason": f"current_anchor_failed: solid={None if solid_info is None else solid_info.get('reason')} legacy={legacy_info.get('reason')}",
                "registration_roi_golden": reg_g,
                "search_roi_current": reg_search,
                "solid_edge_debug": solid_info,
                "legacy_lines_debug": legacy_info,
            }
        pts = legacy_info.get("points_used_abs", {}) if isinstance(legacy_info, dict) else {}
        anchors_c_raw = _current_anchors_from_legacy_lines(legacy_info.get("lines"), y_top_ref_c, bottom_points_abs=pts.get("bottom"))
        if anchors_c_raw is None:
            return None, {"ok": False, "reason": "legacy_current_anchors_failed", "legacy_lines_debug": legacy_info, "solid_edge_debug": solid_info}
        current_method = "legacy_dot_band_lines"
        current_debug_info = legacy_info

    anchors_c_raw_safe = _overlay_safe_anchors(anchors_c_raw)

    # Stabilize top anchors BEFORE transform estimation.
    (lt_stable, rt_stable), top_hyst_info = _DEFAULT_TOP_ANCHOR_HYSTERESIS.update(
        anchors_c_raw_safe["left_top"],
        anchors_c_raw_safe["right_top"],
    )
    anchors_c = {
        **anchors_c_raw_safe,
        "left_top": _xy(lt_stable),
        "right_top": _xy(rt_stable),
    }

    M_raw, inliers = estimate_similarity_from_anchors(anchors_g, anchors_c)
    if M_raw is None:
        return None, {
            "ok": False,
            "reason": "transform_failed",
            "registration_roi_golden": reg_g,
            "search_roi_current": reg_search,
            "anchors_golden": anchors_g,
            "anchors_current": anchors_c,
            "solid_edge_debug": solid_info,
            "legacy_lines_debug": legacy_info,
        }

    hyst = contour_hysteresis if contour_hysteresis is not None else _DEFAULT_CONTOUR_HYSTERESIS
    hyst.enabled = bool(hysteresis_enabled)
    M_stable, hyst_info = hyst.update(
        M_raw,
        confidence={
            "current_anchor_method": current_method,
            "golden_anchor_method": golden_anchor_method,
            "top_anchor_mode": top_hyst_info.get("mode"),
            "solid_reason": None if solid_info is None else solid_info.get("reason"),
            "legacy_reason": None if legacy_info is None else legacy_info.get("reason"),
        },
    )
    if M_stable is None:
        M_stable = M_raw

    moved = [apply_affine_to_roi(M_stable, r) for r in rois_golden]
    active_dbg = current_debug_info or {}

    info = {
        "ok": True,
        "reason": "ok",
        "M": M_stable.tolist(),
        "M_raw": M_raw.tolist(),
        "hysteresis": hyst_info,
        "inliers": None if inliers is None else inliers.astype(int).flatten().tolist(),
        "registration_roi_golden": reg_g,
        "search_roi_current": reg_search,
        "y_top_ref_g": y_top_ref_g,
        "y_top_ref_c": float(0.5 * (anchors_c["left_top"][1] + anchors_c["right_top"][1])),
        "y_top_ref_c_nominal": y_top_ref_c,
        "anchors_golden": anchors_g,
        "anchors_current": anchors_c,
        "anchors_current_raw": anchors_c_raw_safe,
        "anchors_golden_raw": anchors_g_raw,
        "top_anchor_hysteresis": top_hyst_info,
        "anchor_model": "left_top/right_top/bottom_mid",
        "current_anchor_method": current_method,
        "golden_anchor_method": golden_anchor_method,
        "solid_edge_debug": solid_info,
        "golden_solid_edge_debug": golden_solid_info,
        "legacy_lines_debug": legacy_info,
        "current_lines_debug": active_dbg,
        "golden_lines_abs": lines_g,
    }
    return moved, info
