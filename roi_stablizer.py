from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, List

import cv2
import numpy as np

import detector


# ----------------------------
# Basic helpers
# ----------------------------
def _clip_float(v, lo, hi, default):
    try:
        x = float(v)
    except Exception:
        return float(default)

    if not np.isfinite(x):
        return float(default)

    return float(np.clip(x, lo, hi))


def _clamp_roi(roi, W, H):
    x, y, w, h = map(int, roi)
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return x, y, w, h


def _roi_pad(roi, pad, W, H):
    x, y, w, h = _clamp_roi(roi, W, H)
    p = int(max(0, pad))

    x2 = max(0, x - p)
    y2 = max(0, y - p)
    x3 = min(W, x + w + p)
    y3 = min(H, y + h + p)

    return int(x2), int(y2), int(max(1, x3 - x2)), int(max(1, y3 - y2))


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

    vx, vy, x0, y0 = vals
    n = float(np.hypot(vx, vy))

    if n < 1e-9:
        return None

    return float(vx / n), float(vy / n), float(x0), float(y0)


def _good_pt(pt) -> bool:
    try:
        return pt is not None and np.isfinite(float(pt[0])) and np.isfinite(float(pt[1]))
    except Exception:
        return False


def _xy(pt):
    return float(pt[0]), float(pt[1])


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

    try:
        t, _u = np.linalg.solve(A, b)
    except Exception:
        return None

    return float(x1 + t * vx1), float(y1 + t * vy1)


def _point_line_distance(pt, line):
    line = _as_float_line(line)

    if line is None or not _good_pt(pt):
        return None

    vx, vy, x0, y0 = line
    px, py = float(pt[0]), float(pt[1])

    # 2D cross product magnitude because line direction is normalized.
    return float(abs((px - x0) * vy - (py - y0) * vx))


def _line_angle_deg(line):
    line = _as_float_line(line)

    if line is None:
        return 0.0

    vx, vy, _x0, _y0 = line
    return float(np.degrees(np.arctan2(vy, vx)))


def _make_frame_from_bottom_anchors(bottom_left, bottom_right, bottom_line=None):
    """
    Bottom-frame convention:

      origin = midpoint(bottom_left, bottom_right)
      x-axis = bottom_left -> bottom_right
      y-axis = upward from the bottom edge in image space

    For a horizontal bottom line:
      x_axis = (1, 0)
      y_axis = (0, -1)

    So a baseplate above the bottom line has positive local dy.
    """
    if not _good_pt(bottom_left) or not _good_pt(bottom_right):
        return None

    bl = np.array(bottom_left, dtype=np.float64).reshape(2)
    br = np.array(bottom_right, dtype=np.float64).reshape(2)

    vec = br - bl
    width = float(np.linalg.norm(vec))

    if not np.isfinite(width) or width < 20.0:
        return None

    u = vec / width

    # If fitted bottom line exists, use its direction but keep left->right sign.
    line = _as_float_line(bottom_line)

    if line is not None:
        vx, vy, _x0, _y0 = line
        u2 = np.array([vx, vy], dtype=np.float64)

        if float(np.dot(u2, u)) < 0:
            u2 = -u2

        if np.linalg.norm(u2) > 1e-9:
            u = u2 / np.linalg.norm(u2)

    # Image y goes down, so upward normal is (uy, -ux).
    v = np.array([u[1], -u[0]], dtype=np.float64)

    origin = 0.5 * (bl + br)

    return {
        "origin": (float(origin[0]), float(origin[1])),
        "x_axis": (float(u[0]), float(u[1])),
        "y_axis": (float(v[0]), float(v[1])),
        "width_px": float(width),
        "angle_deg": float(np.degrees(np.arctan2(u[1], u[0]))),
        "bottom_left": (float(bl[0]), float(bl[1])),
        "bottom_right": (float(br[0]), float(br[1])),
        "bottom_mid": (float(origin[0]), float(origin[1])),
    }


def _point_to_frame(pt, frame):
    if not _good_pt(pt) or not isinstance(frame, dict):
        return None

    try:
        p = np.array(pt, dtype=np.float64).reshape(2)
        o = np.array(frame["origin"], dtype=np.float64).reshape(2)
        u = np.array(frame["x_axis"], dtype=np.float64).reshape(2)
        v = np.array(frame["y_axis"], dtype=np.float64).reshape(2)
    except Exception:
        return None

    d = p - o

    return float(np.dot(d, u)), float(np.dot(d, v))


def _point_from_frame(frame, dx, dy):
    if not isinstance(frame, dict):
        return None

    try:
        o = np.array(frame["origin"], dtype=np.float64).reshape(2)
        u = np.array(frame["x_axis"], dtype=np.float64).reshape(2)
        v = np.array(frame["y_axis"], dtype=np.float64).reshape(2)
        p = o + u * float(dx) + v * float(dy)
    except Exception:
        return None

    if not np.isfinite(p).all():
        return None

    return float(p[0]), float(p[1])


def _derive_bottom_anchors_from_lines(lines):
    if not isinstance(lines, dict):
        return None

    left = _as_float_line(lines.get("left"))
    right = _as_float_line(lines.get("right"))
    bottom = _as_float_line(lines.get("bottom"))

    if left is None or right is None or bottom is None:
        return None

    bl = _line_intersection(left, bottom)
    br = _line_intersection(right, bottom)

    if not _good_pt(bl) or not _good_pt(br):
        return None

    # Force bottom_mid to ALWAYS be midpoint between bottom_left and bottom_right.
    bm = (
        0.5 * (float(bl[0]) + float(br[0])),
        0.5 * (float(bl[1]) + float(br[1])),
    )

    frame = _make_frame_from_bottom_anchors(bl, br, bottom)

    if frame is None:
        return None

    frame["lines"] = {
        "left": left,
        "right": right,
        "bottom": bottom,
    }

    frame["anchors"] = {
        "bottom_left": _xy(bl),
        "bottom_right": _xy(br),
        "bottom_mid": _xy(bm),
    }

    frame["bottom_mid_source"] = "midpoint(bottom_left,bottom_right)"

    return frame


def _affine_from_frames(golden_frame, current_frame):
    """
    Build affine matrix mapping golden image points into current image points
    using bottom-frame coordinates.
    """
    if not isinstance(golden_frame, dict) or not isinstance(current_frame, dict):
        return None

    scale = float(current_frame["width_px"]) / max(1e-6, float(golden_frame["width_px"]))

    og = np.array(golden_frame["origin"], dtype=np.float64)
    ug = np.array(golden_frame["x_axis"], dtype=np.float64)
    vg = np.array(golden_frame["y_axis"], dtype=np.float64)

    oc = np.array(current_frame["origin"], dtype=np.float64)
    uc = np.array(current_frame["x_axis"], dtype=np.float64)
    vc = np.array(current_frame["y_axis"], dtype=np.float64)

    L = 100.0

    src = np.array(
        [
            og,
            og + ug * L,
            og + vg * L,
        ],
        dtype=np.float32,
    )

    dst = np.array(
        [
            oc,
            oc + uc * L * scale,
            oc + vc * L * scale,
        ],
        dtype=np.float32,
    )

    try:
        M = cv2.getAffineTransform(src, dst)
    except Exception:
        return None

    if M is None or M.shape != (2, 3) or not np.isfinite(M).all():
        return None

    return M.astype(np.float64)


def _apply_affine_to_points(M, pts_xy):
    try:
        pts = np.asarray(pts_xy, dtype=np.float64).reshape(-1, 2)
        ones = np.ones((pts.shape[0], 1), dtype=np.float64)
        pts_h = np.hstack([pts, ones])
        out = (np.asarray(M, dtype=np.float64) @ pts_h.T).T
    except Exception:
        return None

    if out.size == 0 or not np.isfinite(out).all():
        return None

    return out.astype(np.float32)


def apply_affine_to_roi_poly(M, roi_xywh):
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

    return _apply_affine_to_points(M, corners)


def apply_affine_to_roi(M, roi_xywh, W=None, H=None):
    poly = apply_affine_to_roi_poly(M, roi_xywh)

    if poly is None or len(poly) < 4:
        return tuple(map(int, roi_xywh))

    x0 = float(np.min(poly[:, 0]))
    y0 = float(np.min(poly[:, 1]))
    x1 = float(np.max(poly[:, 0]))
    y1 = float(np.max(poly[:, 1]))

    roi = (
        int(round(x0)),
        int(round(y0)),
        int(round(x1 - x0)),
        int(round(y1 - y0)),
    )

    if W is not None and H is not None:
        roi = _clamp_roi(roi, int(W), int(H))

    return roi


def _decompose_affine(M):
    if M is None:
        return None

    M = np.asarray(M, dtype=np.float64)

    if M.shape != (2, 3) or not np.isfinite(M).all():
        return None

    a = float(M[0, 0])
    b = float(M[1, 0])

    return {
        "tx": float(M[0, 2]),
        "ty": float(M[1, 2]),
        "scale": float(np.sqrt(a * a + b * b)),
        "angle_deg": float(np.degrees(np.arctan2(b, a))),
    }


def _blend_matrix(M_old, M_new, alpha):
    a = float(np.clip(alpha, 0.0, 1.0))
    return ((1.0 - a) * np.asarray(M_old, dtype=np.float64) + a * np.asarray(M_new, dtype=np.float64)).astype(np.float64)


@dataclass
class ContourHysteresis:
    enabled: bool = True
    blend_alpha: float = 0.35
    micro_blend_alpha: float = 0.18
    micro_translation_px: float = 5.0
    micro_angle_deg: float = 0.45
    max_translation_jump_px: float = 45.0
    max_angle_jump_deg: float = 5.5
    max_scale_jump_frac: float = 0.07
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

        if M_candidate.shape != (2, 3) or not np.isfinite(M_candidate).all():
            self._last_info = {"mode": "bad_candidate", "accepted": False, **confidence}
            return self._last_M, self._last_info

        if not self.enabled:
            self._last_M = M_candidate.copy()
            self._held_count = 0
            self._last_info = {"mode": "disabled", "accepted": True, "held_count": 0, **confidence}
            return self._last_M.copy(), self._last_info

        cand = _decompose_affine(M_candidate)

        if cand is None:
            self._last_info = {"mode": "bad_candidate_decompose", "accepted": False, **confidence}
            return self._last_M, self._last_info

        if self._last_M is None:
            self._last_M = M_candidate.copy()
            self._held_count = 0
            self._last_info = {"mode": "init", "accepted": True, "held_count": 0, "candidate": cand, **confidence}
            return self._last_M.copy(), self._last_info

        last = _decompose_affine(self._last_M)

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


_DEFAULT_CONTOUR_HYSTERESIS = ContourHysteresis()


def reset_default_contour_hysteresis():
    _DEFAULT_CONTOUR_HYSTERESIS.reset()


def _detect_current_notch_frame(
    *,
    current_img,
    search_roi,
    canny_low,
    canny_high,
    blur_ksize,
    clahe_clip,
    close_iter,
    dilate_iter,
    prefer_solid_edge,
    legacy_band_side_frac,
    legacy_band_bottom_frac,
):
    solid_info = None
    legacy_info = None

    if bool(prefer_solid_edge):
        solid_info = detector.detect_notch_solid_edge_anchors_local(
            current_img,
            search_roi,
            canny_low=int(canny_low),
            canny_high=int(canny_high),
            blur_ksize=int(blur_ksize),
            clahe_clip=float(clahe_clip),
            close_iter=int(close_iter),
            dilate_iter=int(dilate_iter),
            min_component_area_px=180,
            min_anchor_points=18,
        )

        if isinstance(solid_info, dict) and solid_info.get("ok"):
            frame = _derive_bottom_anchors_from_lines(solid_info.get("lines"))

            if frame is not None:
                frame["method"] = "solid_edge_bottom_frame"
                return frame, solid_info, legacy_info, "solid_edge_bottom_frame"

    legacy_info = detector.detect_inner_border_lines_edges_local(
        current_img,
        search_roi,
        canny_low=int(canny_low),
        canny_high=int(canny_high),
        min_points=40,
        band_side_frac=float(legacy_band_side_frac),
        band_bottom_frac=float(legacy_band_bottom_frac),
        sample_stride=1,
        blur_ksize=int(max(3, blur_ksize)),
    )

    if isinstance(legacy_info, dict) and legacy_info.get("ok"):
        frame = _derive_bottom_anchors_from_lines(legacy_info.get("lines"))

        if frame is not None:
            frame["method"] = "legacy_dot_band_bottom_frame"
            return frame, solid_info, legacy_info, "legacy_dot_band_bottom_frame"

    return None, solid_info, legacy_info, "failed"


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
    y_top_offset_px=0.0,  # kept for backward compatibility; no longer used
    prefer_dark_region: bool = True,  # kept for backward compatibility
    prefer_solid_edge: bool = True,
    notch_blur_ksize: int = 5,
    notch_close_ksize: int = 2,
    notch_open_ksize: int = 0,  # kept for backward compatibility
    notch_threshold_bias: float = 0.0,  # kept for backward compatibility
    contour_hysteresis: Optional[ContourHysteresis] = None,
    hysteresis_enabled: bool = True,
    notch_bottom_band_frac: float = 0.35,
    notch_side_band_frac: float = 0.35,
):
    """
    New stabilizer model:

      GOLDEN:
        saved left/right/bottom notch lines from calibration JSON

      CURRENT:
        solid-edge detector returns fitted left/right/bottom lines

      ANCHORS:
        bottom_left  = intersection(left, bottom)
        bottom_right = intersection(right, bottom)
        bottom_mid   = midpoint(bottom_left, bottom_right)

      ROI:
        mapped using bottom-frame transform, not old top_y / left_top / right_top.
    """
    if current_img is None or golden_img is None:
        return None, {"ok": False, "reason": "missing_images"}

    if not isinstance(golden_inner_lines_abs, dict):
        return None, {"ok": False, "reason": "missing_golden_inner_lines"}

    Hc, Wc = current_img.shape[:2]
    Hg, Wg = golden_img.shape[:2]

    reg_g = tuple(map(int, registration_roi_golden))
    search_roi = _roi_pad(reg_g, int(search_padding_px), Wc, Hc)

    golden_frame = _derive_bottom_anchors_from_lines(golden_inner_lines_abs)

    if golden_frame is None:
        return None, {
            "ok": False,
            "reason": "golden_bottom_frame_failed",
            "registration_roi_golden": reg_g,
            "search_roi_current": search_roi,
            "golden_lines_abs": golden_inner_lines_abs,
        }

    golden_frame["method"] = "golden_saved_bottom_frame"

    legacy_band_side_frac = _clip_float(notch_side_band_frac, 0.05, 0.80, 0.35)
    legacy_band_bottom_frac = _clip_float(notch_bottom_band_frac, 0.05, 0.80, 0.35)

    current_frame, solid_info, legacy_info, current_method = _detect_current_notch_frame(
        current_img=current_img,
        search_roi=search_roi,
        canny_low=int(canny_low),
        canny_high=int(canny_high),
        blur_ksize=int(notch_blur_ksize),
        clahe_clip=1.5,
        close_iter=int(notch_close_ksize),
        dilate_iter=1,
        prefer_solid_edge=bool(prefer_solid_edge),
        legacy_band_side_frac=legacy_band_side_frac,
        legacy_band_bottom_frac=legacy_band_bottom_frac,
    )

    if current_frame is None:
        return None, {
            "ok": False,
            "reason": "current_bottom_frame_failed",
            "registration_roi_golden": reg_g,
            "search_roi_current": search_roi,
            "golden_frame": golden_frame,
            "solid_edge_debug": solid_info,
            "legacy_lines_debug": legacy_info,
            "current_anchor_method": current_method,
            "notch_bottom_band_frac": legacy_band_bottom_frac,
            "notch_side_band_frac": legacy_band_side_frac,
        }

    M_raw = _affine_from_frames(golden_frame, current_frame)

    if M_raw is None:
        return None, {
            "ok": False,
            "reason": "bottom_frame_transform_failed",
            "registration_roi_golden": reg_g,
            "search_roi_current": search_roi,
            "golden_frame": golden_frame,
            "current_notch_frame": current_frame,
            "solid_edge_debug": solid_info,
            "legacy_lines_debug": legacy_info,
            "current_anchor_method": current_method,
        }

    hyst = contour_hysteresis if contour_hysteresis is not None else _DEFAULT_CONTOUR_HYSTERESIS
    hyst.enabled = bool(hysteresis_enabled)

    M_stable, hyst_info = hyst.update(
        M_raw,
        confidence={
            "current_anchor_method": current_method,
            "golden_anchor_method": "saved_bottom_frame",
            "solid_reason": None if solid_info is None else solid_info.get("reason"),
            "legacy_reason": None if legacy_info is None else legacy_info.get("reason"),
        },
    )

    if M_stable is None:
        M_stable = M_raw

    moved: List[Tuple[int, int, int, int]] = []
    moved_polys = []

    for r in rois_golden:
        poly = apply_affine_to_roi_poly(M_stable, r)
        roi = apply_affine_to_roi(M_stable, r, Wc, Hc)

        moved.append(roi)

        if poly is not None:
            moved_polys.append(poly.astype(np.float32))
        else:
            moved_polys.append(None)

    anchors = dict(current_frame.get("anchors") or {})
    lines = dict(current_frame.get("lines") or {})

    info = {
        "ok": True,
        "reason": "ok",

        "M": np.asarray(M_stable, dtype=np.float64).tolist(),
        "M_raw": np.asarray(M_raw, dtype=np.float64).tolist(),
        "hysteresis": hyst_info,

        "registration_roi_golden": reg_g,
        "search_roi_current": search_roi,

        "roi_mode": "bottom_mid_local_rotated_roi",
        "anchor_model": "bottom_left/bottom_right/bottom_mid",
        "current_anchor_method": current_method,
        "golden_anchor_method": "saved_bottom_frame",

        "golden_notch_frame": golden_frame,
        "current_notch_frame": current_frame,
        "notch_frame": current_frame,

        "bottom_left": anchors.get("bottom_left"),
        "bottom_right": anchors.get("bottom_right"),
        "bottom_mid": anchors.get("bottom_mid"),

        "notch_lines": lines,
        "lines": lines,
        "current_lines": lines,

        "solid_edge_debug": solid_info,
        "legacy_lines_debug": legacy_info,
        "current_lines_debug": solid_info if current_method.startswith("solid") else legacy_info,

        "roi_poly_current": None if not moved_polys else moved_polys[0],
        "roi_polys_current": moved_polys,

        "notch_bottom_band_frac": legacy_band_bottom_frac,
        "notch_side_band_frac": legacy_band_side_frac,

        # Old keys kept as harmless compatibility.
        "y_top_ref_g": None,
        "y_top_ref_c": None,
        "anchors_current": {
            "bottom_left": anchors.get("bottom_left"),
            "bottom_right": anchors.get("bottom_right"),
            "bottom_mid": anchors.get("bottom_mid"),
        },
        "anchors_golden": {
            "bottom_left": golden_frame["anchors"]["bottom_left"],
            "bottom_right": golden_frame["anchors"]["bottom_right"],
            "bottom_mid": golden_frame["anchors"]["bottom_mid"],
        },
    }

    return moved, info