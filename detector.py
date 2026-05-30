# detector.py (FULL REWRITE)
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# ----------------------------
# IO helper
# ----------------------------
def load_and_crop(image_path: str, roi_xywh):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"❌ Failed to load image: {image_path}")

    x, y, w, h = map(int, roi_xywh)
    H, W = image.shape[:2]
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))

    return image[y:y + h, x:x + w], image


# ----------------------------
# Generic helpers
# ----------------------------
def _clamp_roi_abs(image_bgr: np.ndarray, roi_xywh_abs):
    x, y, w, h = map(int, roi_xywh_abs)
    H, W = image_bgr.shape[:2]

    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))

    return x, y, w, h


def _as_points(points) -> Optional[np.ndarray]:
    if points is None:
        return None
    try:
        arr = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    except Exception:
        return None
    if arr.size == 0:
        return None
    arr = arr[np.isfinite(arr).all(axis=1)]
    if len(arr) == 0:
        return None
    return arr


def _fit_line(points) -> Optional[Tuple[float, float, float, float]]:
    pts = _as_points(points)
    if pts is None or len(pts) < 8:
        return None

    vx, vy, x0, y0 = cv2.fitLine(
        pts.reshape(-1, 1, 2),
        cv2.DIST_L2,
        0,
        0.01,
        0.01,
    ).flatten()

    vals = (float(vx), float(vy), float(x0), float(y0))
    if not np.isfinite(vals).all():
        return None
    return vals


def _line_x_at_y(line, y_target: float) -> Optional[float]:
    if line is None:
        return None
    try:
        vx, vy, x0, y0 = map(float, line)
    except Exception:
        return None
    if not np.isfinite([vx, vy, x0, y0]).all():
        return None
    if abs(vy) < 1e-9:
        return float(x0)
    t = (float(y_target) - y0) / vy
    return float(x0 + vx * t)


def _line_y_at_x(line, x_target: float) -> Optional[float]:
    if line is None:
        return None
    try:
        vx, vy, x0, y0 = map(float, line)
    except Exception:
        return None
    if not np.isfinite([vx, vy, x0, y0]).all():
        return None
    if abs(vx) < 1e-9:
        return float(y0)
    t = (float(x_target) - x0) / vx
    return float(y0 + vy * t)


def _line_intersection(line_a, line_b) -> Optional[Tuple[float, float]]:
    """
    Intersection of two cv2.fitLine-style infinite lines.

    Lines are (vx, vy, x0, y0). Returns None for bad/parallel lines.
    """
    if line_a is None or line_b is None:
        return None

    try:
        vx1, vy1, x1, y1 = map(float, line_a)
        vx2, vy2, x2, y2 = map(float, line_b)
    except Exception:
        return None

    vals = [vx1, vy1, x1, y1, vx2, vy2, x2, y2]
    if not np.isfinite(vals).all():
        return None

    n1 = float(np.hypot(vx1, vy1))
    n2 = float(np.hypot(vx2, vy2))
    if n1 < 1e-9 or n2 < 1e-9:
        return None

    vx1, vy1 = vx1 / n1, vy1 / n1
    vx2, vy2 = vx2 / n2, vy2 / n2

    A = np.array([[vx1, -vx2], [vy1, -vy2]], dtype=np.float64)
    b = np.array([x2 - x1, y2 - y1], dtype=np.float64)

    det = float(np.linalg.det(A))
    if abs(det) < 1e-9:
        return None

    try:
        t, _u = np.linalg.solve(A, b)
    except Exception:
        return None

    px = x1 + t * vx1
    py = y1 + t * vy1

    if not np.isfinite([px, py]).all():
        return None

    return float(px), float(py)


def _bottom_frame_from_lines(left_line, right_line, bottom_line):
    """
    Build the real bottom-frame anchors from fitted side and bottom lines.

    The important rule: bottom_mid is ALWAYS the midpoint of bottom_left and
    bottom_right. It never comes from the raw lower-contour median.
    """
    bottom_left = _line_intersection(left_line, bottom_line)
    bottom_right = _line_intersection(right_line, bottom_line)

    if not _good_pt(bottom_left) or not _good_pt(bottom_right):
        return None

    bl = np.asarray(bottom_left, dtype=np.float64).reshape(2)
    br = np.asarray(bottom_right, dtype=np.float64).reshape(2)

    vec = br - bl
    width = float(np.linalg.norm(vec))
    if not np.isfinite(width) or width < 10.0:
        return None

    x_axis = vec / width

    # Image coords: y grows downward. For a normal left->right bottom edge,
    # this points upward from the bottom line into the dark glass region.
    y_axis = np.asarray([x_axis[1], -x_axis[0]], dtype=np.float64)
    if y_axis[1] > 0:
        y_axis = -y_axis

    origin = 0.5 * (bl + br)

    return {
        "bottom_left": (float(bl[0]), float(bl[1])),
        "bottom_right": (float(br[0]), float(br[1])),
        "bottom_mid": (float(origin[0]), float(origin[1])),
        "origin": (float(origin[0]), float(origin[1])),
        "x_axis": (float(x_axis[0]), float(x_axis[1])),
        "y_axis": (float(y_axis[0]), float(y_axis[1])),
        "width": float(width),
        "width_px": float(width),
        "angle_deg": float(np.degrees(np.arctan2(x_axis[1], x_axis[0]))),
        "bottom_angle_deg": _bottom_line_angle_deg(bottom_line),
    }


def _point_on_line_at_y(line, y_target: float) -> Optional[Tuple[float, float]]:
    x = _line_x_at_y(line, y_target)
    if x is None:
        return None
    return (float(x), float(y_target))


def _median_point(points) -> Optional[Tuple[float, float]]:
    pts = _as_points(points)
    if pts is None:
        return None
    return (float(np.median(pts[:, 0])), float(np.median(pts[:, 1])))


def _shift_points(points: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return np.asarray(points, dtype=np.float32).reshape(-1, 2) + np.array([dx, dy], dtype=np.float32)


def _shift_line(line, dx: float, dy: float):
    if line is None:
        return None
    vx, vy, x0, y0 = map(float, line)
    return (float(vx), float(vy), float(x0 + dx), float(y0 + dy))


def _good_pt(pt) -> bool:
    try:
        return pt is not None and len(pt) >= 2 and np.isfinite(float(pt[0])) and np.isfinite(float(pt[1]))
    except Exception:
        return False


# ----------------------------
# Preprocess (baseplate debug path)
# ----------------------------
def preprocess_edges(
    bgr: np.ndarray,
    *,
    use_clahe: bool = True,
    clahe_clip: float = 1.5,
    clahe_grid: int = 8,
    bilateral_d: int = 5,
    bilateral_sigma_color: int = 40,
    bilateral_sigma_space: int = 40,
    blur_ksize: int = 5,
    canny_low: int = 50,
    canny_high: int = 120,
    dilate_iter: int = 1,
    close_iter: int = 1,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    if bgr is None or bgr.size == 0:
        raise ValueError("empty image")

    gray0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.bilateralFilter(
        gray0,
        d=int(bilateral_d),
        sigmaColor=int(bilateral_sigma_color),
        sigmaSpace=int(bilateral_sigma_space),
    )

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(int(clahe_grid), int(clahe_grid)))
        gray2 = clahe.apply(gray1)
    else:
        gray2 = cv2.equalizeHist(gray1)

    k = int(blur_ksize)
    if k % 2 == 0:
        k += 1
    k = max(3, k)

    gray3 = cv2.GaussianBlur(gray2, (k, k), 0)

    edges0 = cv2.Canny(gray3, int(canny_low), int(canny_high), apertureSize=3, L2gradient=True)
    edges1 = cv2.dilate(edges0, np.ones((3, 3), np.uint8), iterations=int(max(0, dilate_iter)))
    edges2 = cv2.morphologyEx(edges1, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=int(max(0, close_iter)))

    dbg = {
        "gray0": gray0,
        "gray1_bilateral": gray1,
        "gray2_contrast": gray2,
        "gray3_blur": gray3,
        "edges0": edges0,
        "edges1_dilate": edges1,
        "edges2_close": edges2,
    }

    return gray2, edges2, dbg


# ----------------------------
# Baseplate detection
# ----------------------------
def _contrast_score(gray: np.ndarray, cnt: np.ndarray) -> float:
    mask = np.zeros(gray.shape[:2], np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, thickness=-1)

    inside_mean = cv2.mean(gray, mask=mask)[0]
    outside_mean = cv2.mean(gray, mask=cv2.bitwise_not(mask))[0]

    return float(abs(inside_mean - outside_mean))


def _pick_best_contour(
    gray: np.ndarray,
    contours: List[np.ndarray],
    W: int,
    H: int,
    *,
    area_min_frac: float,
    area_max_frac: float,
    aspect_min: float,
    aspect_max: float,
    solidity_min: float,
    extent_min: float,
    border_margin: int,
    contrast_min: float,
) -> Optional[np.ndarray]:
    win_area = float(W * H)
    best = None
    best_score = -1e18

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        if border_margin is not None:
            if x <= border_margin or y <= border_margin or x + w >= W - border_margin or y + h >= H - border_margin:
                continue

        area = float(cv2.contourArea(c))
        if area < area_min_frac * win_area or area > area_max_frac * win_area:
            continue

        rect = cv2.minAreaRect(c)
        (_, _), (rw, rh), _ = rect

        if rw <= 0 or rh <= 0:
            continue

        ar = max(rw, rh) / max(1e-6, min(rw, rh))
        if not (aspect_min <= ar <= aspect_max):
            continue

        hull = cv2.convexHull(c)
        hull_area = float(cv2.contourArea(hull)) + 1e-6
        solidity = area / hull_area
        if solidity < solidity_min:
            continue

        extent = area / float(w * h + 1e-6)
        if extent < extent_min:
            continue

        contrast = _contrast_score(gray, c)
        if contrast < contrast_min:
            continue

        score = area + 2000.0 * solidity + 5.0 * contrast
        if score > best_score:
            best_score = score
            best = c

    return best


def _normalize_min_area_angle(rect) -> float:
    (_cx, _cy), (w, h), angle = rect

    if w < h:
        angle += 90.0

    angle -= 90.0

    if angle < -90:
        angle += 180
    elif angle > 90:
        angle -= 180

    return float(round(angle, 2))


def _detect_in_window(
    bgr_window: np.ndarray,
    *,
    shrink_border_px: int = 10,
    border_margin: int = 12,
    use_clahe: bool = True,
    clahe_clip: float = 1.5,
    clahe_grid: int = 8,
    canny_low: int = 50,
    canny_high: int = 120,
    area_min_frac: float = 0.005,
    area_max_frac: float = 0.60,
    aspect_min: float = 0.5,
    aspect_max: float = 2.2,
    solidity_min: float = 0.7,
    extent_min: float = 0.25,
    contrast_min: float = 12.0,
    blur_ksize: int = 5,
    dilate_iter: int = 1,
    close_iter: int = 1,
    return_debug: bool = False,
) -> Tuple[Optional[Tuple[float, float]], Optional[float], Optional[np.ndarray], Optional[Dict[str, Any]]]:
    if bgr_window is None or bgr_window.size == 0:
        return None, None, None, ({"ok": False, "reason": "empty_window"} if return_debug else None)

    H0, W0 = bgr_window.shape[:2]

    sx = min(int(shrink_border_px), max(0, W0 // 10))
    sy = min(int(shrink_border_px), max(0, H0 // 10))

    inner = bgr_window[sy:H0 - sy, sx:W0 - sx]

    if inner.size == 0:
        return None, None, None, ({"ok": False, "reason": "inner_empty"} if return_debug else None)

    gray, edges, dbg_frames = preprocess_edges(
        inner,
        use_clahe=use_clahe,
        clahe_clip=clahe_clip,
        clahe_grid=clahe_grid,
        blur_ksize=blur_ksize,
        canny_low=canny_low,
        canny_high=canny_high,
        dilate_iter=dilate_iter,
        close_iter=close_iter,
    )

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = _pick_best_contour(
        gray,
        contours,
        inner.shape[1],
        inner.shape[0],
        area_min_frac=area_min_frac,
        area_max_frac=area_max_frac,
        aspect_min=aspect_min,
        aspect_max=aspect_max,
        solidity_min=solidity_min,
        extent_min=extent_min,
        border_margin=border_margin,
        contrast_min=contrast_min,
    )

    relaxed_used = False

    if best is None:
        # Second chance with gentler gates. This reduces frame-to-frame dropouts
        # when lighting changes slightly or the baseplate edge touches the window
        # border for a moment.
        best = _pick_best_contour(
            gray,
            contours,
            inner.shape[1],
            inner.shape[0],
            area_min_frac=max(0.0008, float(area_min_frac) * 0.45),
            area_max_frac=min(0.90, float(area_max_frac) * 1.25),
            aspect_min=max(0.20, float(aspect_min) * 0.65),
            aspect_max=min(5.00, float(aspect_max) * 1.45),
            solidity_min=max(0.35, float(solidity_min) * 0.72),
            extent_min=max(0.08, float(extent_min) * 0.65),
            border_margin=0,
            contrast_min=max(1.5, float(contrast_min) * 0.55),
        )
        relaxed_used = best is not None

    if best is None:
        if return_debug:
            return None, None, None, {"ok": False, "reason": "no_contour", "dbg": dbg_frames}
        return None, None, None, None

    cnt_win = best + np.array([[sx, sy]], dtype=np.int32)

    rect = cv2.minAreaRect(cnt_win)
    (cx, cy), _, _ = rect
    angle = _normalize_min_area_angle(rect)

    if return_debug:
        return (
            (float(cx), float(cy)),
            angle,
            cnt_win,
            {
                "ok": True,
                "reason": "ok",
                "dbg": dbg_frames,
                "rect": rect,
                "relaxed_used": bool(relaxed_used),
            },
        )

    return (float(cx), float(cy)), angle, cnt_win, None


# ----------------------------
# Baseplate temporal stabilization / anti-flicker
# ----------------------------
_BASEPLATE_STABLE_ALPHA = 0.38
_BASEPLATE_MICRO_ALPHA = 0.18
_BASEPLATE_MICRO_JUMP_PX = 4.0
_BASEPLATE_MAX_JUMP_PX = 85.0
_BASEPLATE_MAX_ANGLE_JUMP_DEG = 22.0
_BASEPLATE_HOLD_BAD_FRAMES = 10
_BASEPLATE_HOLD_MISSING_FRAMES = 14

_BASEPLATE_TRACK = {
    "valid": False,
    "center_abs": None,
    "angle": None,
    "contour_abs": None,
    "frame_i": 0,
    "last_update_i": -10_000,
    "bad_count": 0,
    "missing_count": 0,
    "last_roi": None,
    "last_shape": None,
}


def reset_baseplate_detection_hysteresis():
    """
    Optional public reset hook.

    Call this when switching product/camera if you want to clear the anti-flicker
    memory. Existing modules do not have to call it; the cache also self-resets
    when the ROI/image shape becomes incompatible.
    """
    global _BASEPLATE_TRACK

    _BASEPLATE_TRACK = {
        "valid": False,
        "center_abs": None,
        "angle": None,
        "contour_abs": None,
        "frame_i": 0,
        "last_update_i": -10_000,
        "bad_count": 0,
        "missing_count": 0,
        "last_roi": None,
        "last_shape": None,
    }


def _angle_diff_deg_local(a, b) -> float:
    d = float(a) - float(b)
    while d > 180.0:
        d -= 360.0
    while d < -180.0:
        d += 360.0
    return float(d)


def _blend_angle_deg(old, new, alpha: float) -> float:
    return float(old + _angle_diff_deg_local(new, old) * float(alpha))


def _contour_abs_from_rel(cnt_rel, roi_xywh_abs):
    if cnt_rel is None:
        return None

    try:
        cnt = np.asarray(cnt_rel, dtype=np.int32).reshape(-1, 1, 2)
    except Exception:
        return None

    if roi_xywh_abs is None:
        return cnt.copy()

    rx, ry, _rw, _rh = map(int, roi_xywh_abs)
    return cnt + np.array([[[rx, ry]]], dtype=np.int32)


def _contour_rel_from_abs(cnt_abs, roi_xywh_abs):
    if cnt_abs is None:
        return None

    try:
        cnt = np.asarray(cnt_abs, dtype=np.int32).reshape(-1, 1, 2)
    except Exception:
        return None

    if roi_xywh_abs is None:
        return cnt.copy()

    rx, ry, _rw, _rh = map(int, roi_xywh_abs)
    return cnt - np.array([[[rx, ry]]], dtype=np.int32)


def _shift_contour(cnt_abs, dx: float, dy: float):
    if cnt_abs is None:
        return None

    try:
        cnt = np.asarray(cnt_abs, dtype=np.float32).reshape(-1, 1, 2)
    except Exception:
        return None

    cnt[:, :, 0] += float(dx)
    cnt[:, :, 1] += float(dy)

    return np.round(cnt).astype(np.int32)


def _center_abs_from_rel(center_rel, roi_xywh_abs):
    if center_rel is None:
        return None

    cx = float(center_rel[0])
    cy = float(center_rel[1])

    if roi_xywh_abs is None:
        return (cx, cy)

    rx, ry, _rw, _rh = map(float, roi_xywh_abs)
    return (float(rx + cx), float(ry + cy))


def _center_rel_from_abs(center_abs, roi_xywh_abs):
    if center_abs is None:
        return None

    cx = float(center_abs[0])
    cy = float(center_abs[1])

    if roi_xywh_abs is None:
        return (cx, cy)

    rx, ry, _rw, _rh = map(float, roi_xywh_abs)
    return (float(cx - rx), float(cy - ry))


def _roi_contains_center_with_margin(roi_xywh_abs, center_abs, margin: float = 180.0) -> bool:
    if roi_xywh_abs is None or center_abs is None:
        return True

    try:
        x, y, w, h = map(float, roi_xywh_abs)
        cx, cy = map(float, center_abs)
    except Exception:
        return False

    return bool(
        x - margin <= cx <= x + w + margin
        and y - margin <= cy <= y + h + margin
    )


def _baseplate_cache_compatible(roi_xywh_abs, image_shape) -> bool:
    tr = _BASEPLATE_TRACK

    if not tr.get("valid"):
        return False

    if tr.get("center_abs") is None:
        return False

    if not _roi_contains_center_with_margin(roi_xywh_abs, tr.get("center_abs"), margin=220.0):
        return False

    # If image size changes, drop the cache. That usually means a camera/product reload.
    last_shape = tr.get("last_shape")
    if last_shape is not None and image_shape is not None:
        try:
            if tuple(last_shape[:2]) != tuple(image_shape[:2]):
                return False
        except Exception:
            return False

    return True


def _accept_baseplate_candidate(candidate, *, mode: str):
    tr = _BASEPLATE_TRACK

    tr["valid"] = True
    tr["center_abs"] = tuple(map(float, candidate["center_abs"]))
    tr["angle"] = None if candidate.get("angle") is None else float(candidate["angle"])
    tr["contour_abs"] = None if candidate.get("contour_abs") is None else np.asarray(candidate["contour_abs"], dtype=np.int32).copy()
    tr["last_update_i"] = int(tr["frame_i"])
    tr["bad_count"] = 0
    tr["missing_count"] = 0
    tr["last_roi"] = candidate.get("roi_xywh_abs")
    tr["last_shape"] = candidate.get("image_shape")
    tr["mode"] = mode


def _held_baseplate_result(roi_xywh_abs, *, return_debug: bool, reason: str):
    tr = _BASEPLATE_TRACK
    center_rel = _center_rel_from_abs(tr.get("center_abs"), roi_xywh_abs)
    cnt_rel = _contour_rel_from_abs(tr.get("contour_abs"), roi_xywh_abs)
    angle = tr.get("angle")

    if return_debug:
        return center_rel, angle, cnt_rel, {
            "stage": "hysteresis_hold",
            "ok": True,
            "reason": reason,
            "baseplate_hysteresis": {
                "mode": reason,
                "held": True,
                "missing_count": int(tr.get("missing_count", 0)),
                "bad_count": int(tr.get("bad_count", 0)),
                "center_abs": tr.get("center_abs"),
                "angle": tr.get("angle"),
            },
        }

    return center_rel, angle, cnt_rel, None


def _stabilize_baseplate_candidate(candidate, roi_xywh_abs, image_shape, *, return_debug: bool):
    """
    Returns center_rel, angle, contour_rel, dbg_or_none.

    This is intentionally small and conservative:
      - if detection disappears for a few frames, hold last good baseplate
      - if a new contour jumps wildly, hold last good baseplate briefly
      - otherwise blend center/angle to reduce visible flicker
    """
    tr = _BASEPLATE_TRACK
    tr["frame_i"] = int(tr.get("frame_i", 0)) + 1

    if candidate is None:
        if _baseplate_cache_compatible(roi_xywh_abs, image_shape):
            tr["missing_count"] = int(tr.get("missing_count", 0)) + 1

            if tr["missing_count"] <= int(_BASEPLATE_HOLD_MISSING_FRAMES):
                return _held_baseplate_result(
                    roi_xywh_abs,
                    return_debug=return_debug,
                    reason="hold_last_missing_detection",
                )

        # No usable cache.
        if return_debug:
            return None, None, None, {
                "stage": "fail",
                "ok": False,
                "reason": "not_found",
                "baseplate_hysteresis": {
                    "mode": "no_candidate_no_cache",
                    "held": False,
                },
            }
        return None, None, None, None

    candidate["roi_xywh_abs"] = None if roi_xywh_abs is None else tuple(map(int, roi_xywh_abs))
    candidate["image_shape"] = None if image_shape is None else tuple(map(int, image_shape[:2]))

    c_abs = np.asarray(candidate["center_abs"], dtype=np.float64).reshape(2)

    if not tr.get("valid") or not _baseplate_cache_compatible(roi_xywh_abs, image_shape):
        _accept_baseplate_candidate(candidate, mode="init_accept")
        center_rel = _center_rel_from_abs(candidate["center_abs"], roi_xywh_abs)
        cnt_rel = _contour_rel_from_abs(candidate.get("contour_abs"), roi_xywh_abs)

        dbg = dict(candidate.get("dbg") or {})
        if return_debug:
            dbg["baseplate_hysteresis"] = {"mode": "init_accept", "held": False}
            return center_rel, candidate.get("angle"), cnt_rel, dbg

        return center_rel, candidate.get("angle"), cnt_rel, None

    last_c = np.asarray(tr["center_abs"], dtype=np.float64).reshape(2)
    dist = float(np.linalg.norm(c_abs - last_c))

    last_angle = tr.get("angle")
    cand_angle = candidate.get("angle")

    if last_angle is None or cand_angle is None:
        angle_jump = 0.0
    else:
        angle_jump = abs(_angle_diff_deg_local(float(cand_angle), float(last_angle)))

    huge_jump = dist > float(_BASEPLATE_MAX_JUMP_PX) or angle_jump > float(_BASEPLATE_MAX_ANGLE_JUMP_DEG)

    if huge_jump:
        tr["bad_count"] = int(tr.get("bad_count", 0)) + 1

        if tr["bad_count"] <= int(_BASEPLATE_HOLD_BAD_FRAMES):
            return _held_baseplate_result(
                roi_xywh_abs,
                return_debug=return_debug,
                reason="hold_last_jump_rejected",
            )

        # If the jump persists long enough, accept it as the new reality.
        _accept_baseplate_candidate(candidate, mode="force_accept_after_jump_hold")
        center_rel = _center_rel_from_abs(candidate["center_abs"], roi_xywh_abs)
        cnt_rel = _contour_rel_from_abs(candidate.get("contour_abs"), roi_xywh_abs)

        dbg = dict(candidate.get("dbg") or {})
        if return_debug:
            dbg["baseplate_hysteresis"] = {
                "mode": "force_accept_after_jump_hold",
                "held": False,
                "jump_px": dist,
                "angle_jump_deg": angle_jump,
            }
            return center_rel, candidate.get("angle"), cnt_rel, dbg

        return center_rel, candidate.get("angle"), cnt_rel, None

    tr["bad_count"] = 0
    tr["missing_count"] = 0

    alpha = float(_BASEPLATE_MICRO_ALPHA if dist <= _BASEPLATE_MICRO_JUMP_PX else _BASEPLATE_STABLE_ALPHA)
    filtered_c = (1.0 - alpha) * last_c + alpha * c_abs

    if last_angle is None:
        filtered_angle = cand_angle
    elif cand_angle is None:
        filtered_angle = last_angle
    else:
        filtered_angle = _blend_angle_deg(float(last_angle), float(cand_angle), alpha)

    # Shift the new contour so it visually follows the filtered center.
    contour_abs = candidate.get("contour_abs")
    if contour_abs is not None:
        dx = float(filtered_c[0] - c_abs[0])
        dy = float(filtered_c[1] - c_abs[1])
        contour_abs = _shift_contour(contour_abs, dx, dy)

    filtered = {
        "center_abs": (float(filtered_c[0]), float(filtered_c[1])),
        "angle": None if filtered_angle is None else float(filtered_angle),
        "contour_abs": contour_abs,
        "roi_xywh_abs": candidate.get("roi_xywh_abs"),
        "image_shape": candidate.get("image_shape"),
    }

    _accept_baseplate_candidate(filtered, mode="blend_accept")

    center_rel = _center_rel_from_abs(filtered["center_abs"], roi_xywh_abs)
    cnt_rel = _contour_rel_from_abs(filtered.get("contour_abs"), roi_xywh_abs)

    dbg = dict(candidate.get("dbg") or {})
    if return_debug:
        dbg["baseplate_hysteresis"] = {
            "mode": "blend_accept",
            "held": False,
            "alpha": alpha,
            "jump_px": dist,
            "angle_jump_deg": angle_jump,
            "raw_center_abs": tuple(map(float, candidate["center_abs"])),
            "filtered_center_abs": tuple(map(float, filtered["center_abs"])),
        }
        return center_rel, filtered["angle"], cnt_rel, dbg

    return center_rel, filtered["angle"], cnt_rel, None

def detect_baseplate(
    cropped_roi_bgr: np.ndarray,
    *,
    full_image_bgr: Optional[np.ndarray] = None,
    roi_xywh_abs: Optional[Tuple[int, int, int, int]] = None,
    padding: int = 150,
    return_debug: bool = False,
    **kwargs,
):
    """
    Detect the baseplate and apply a small temporal anti-flicker hold/blend.

    Public return format is unchanged:
      center_rel, angle, contour_rel
      or center_rel, angle, contour_rel, debug when return_debug=True

    The anti-flicker layer is intentionally inside detector.py so older recovered
    engine/calibration modules do not need to be rewritten again.
    """
    # Normalize ROI for stable absolute-coordinate cache.
    roi_abs = None
    if roi_xywh_abs is not None:
        try:
            roi_abs = tuple(map(int, roi_xywh_abs))
        except Exception:
            roi_abs = None

    image_shape = None
    if full_image_bgr is not None:
        image_shape = full_image_bgr.shape[:2]
    elif cropped_roi_bgr is not None:
        image_shape = cropped_roi_bgr.shape[:2]

    candidate = None
    fail_debug = None

    c, a, cnt, dbg = _detect_in_window(cropped_roi_bgr, return_debug=True, **kwargs)

    if c is not None:
        center_abs = _center_abs_from_rel(c, roi_abs)
        cnt_abs = _contour_abs_from_rel(cnt, roi_abs)

        candidate = {
            "center_abs": center_abs,
            "angle": a,
            "contour_abs": cnt_abs,
            "dbg": {"stage": "crop", **(dbg or {})},
        }

    else:
        fail_debug = {"stage": "crop_fail", **(dbg or {})}

        if full_image_bgr is not None and roi_abs is not None and int(padding) > 0:
            rx, ry, rw, rh = map(int, roi_abs)
            H, W = full_image_bgr.shape[:2]

            x0 = max(0, rx - int(padding))
            y0 = max(0, ry - int(padding))
            x1 = min(W, rx + rw + int(padding))
            y1 = min(H, ry + rh + int(padding))

            win = full_image_bgr[y0:y1, x0:x1]

            c2, a2, cnt2, dbg2 = _detect_in_window(win, return_debug=True, **kwargs)

            if c2 is not None:
                cx_abs = float(x0 + c2[0])
                cy_abs = float(y0 + c2[1])

                cnt_abs = None
                if cnt2 is not None:
                    cnt_abs = cnt2 + np.array([[x0, y0]], dtype=np.int32)

                candidate = {
                    "center_abs": (cx_abs, cy_abs),
                    "angle": a2,
                    "contour_abs": cnt_abs,
                    "dbg": {
                        "stage": "padded_full",
                        **(dbg2 or {}),
                        "pad_window_abs": (x0, y0, x1 - x0, y1 - y0),
                    },
                }
            else:
                fail_debug = {
                    "stage": "fail",
                    "ok": False,
                    "reason": "not_found",
                    "crop_debug": fail_debug,
                    "padded_debug": dbg2,
                }

    center_rel, angle, contour_rel, stable_dbg = _stabilize_baseplate_candidate(
        candidate,
        roi_abs,
        image_shape,
        return_debug=return_debug,
    )

    if return_debug:
        if stable_dbg is None:
            stable_dbg = fail_debug or {"stage": "fail", "ok": False, "reason": "not_found"}
        return center_rel, angle, contour_rel, stable_dbg

    return center_rel, angle, contour_rel



# ----------------------------
# Legacy dot-band detector (fallback only)
# ----------------------------
def detect_inner_border_lines_edges_local(
    image_bgr: np.ndarray,
    roi_xywh_abs,
    *,
    canny_low: int = 60,
    canny_high: int = 140,
    min_points: int = 40,
    band_side_frac: float = 0.22,
    band_bottom_frac: float = 0.25,
    sample_stride: int = 1,
    blur_ksize: int = 5,
):
    if image_bgr is None or image_bgr.size == 0:
        return {"ok": False, "reason": "empty_image"}

    x, y, w, h = _clamp_roi_abs(image_bgr, roi_xywh_abs)
    crop = image_bgr[y:y + h, x:x + w]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    k = int(blur_ksize)
    if k % 2 == 0:
        k += 1
    k = max(3, k)

    gray = cv2.GaussianBlur(gray, (k, k), 0)

    edges = cv2.Canny(gray, int(canny_low), int(canny_high))
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    bw = max(10, int(round(float(band_side_frac) * w)))
    bh = max(10, int(round(float(band_bottom_frac) * h)))

    left_band = edges[:, :bw]
    right_band = edges[:, w - bw:]
    bottom_band = edges[h - bh:, :]

    def band_points(band, x_off=0, y_off=0):
        ys, xs = np.where(band > 0)
        if sample_stride > 1 and len(xs) > 0:
            idx = np.arange(0, len(xs), int(sample_stride))
            xs = xs[idx]
            ys = ys[idx]
        return np.column_stack([xs + x_off, ys + y_off]).astype(np.float32)

    pts_left = band_points(left_band, 0, 0)
    pts_right = band_points(right_band, w - bw, 0)
    pts_bottom = band_points(bottom_band, 0, h - bh)

    counts = {"left": int(len(pts_left)), "right": int(len(pts_right)), "bottom": int(len(pts_bottom))}

    for k2, n in counts.items():
        if n < int(min_points):
            return {"ok": False, "reason": f"not_enough_points_{k2}", "counts": counts, "roi_used": (x, y, w, h)}

    line_left = _fit_line(pts_left)
    line_right = _fit_line(pts_right)
    line_bottom = _fit_line(pts_bottom)

    if line_left is None or line_right is None or line_bottom is None:
        return {"ok": False, "reason": "fit_failed", "counts": counts, "roi_used": (x, y, w, h)}

    lines_abs = {
        "left": _shift_line(line_left, x, y),
        "right": _shift_line(line_right, x, y),
        "bottom": _shift_line(line_bottom, x, y),
    }

    pts_left_abs = _shift_points(pts_left, x, y)
    pts_right_abs = _shift_points(pts_right, x, y)
    pts_bottom_abs = _shift_points(pts_bottom, x, y)

    all_pts_abs = np.vstack([pts_left_abs, pts_right_abs, pts_bottom_abs]).astype(np.int32)
    hull_abs = cv2.convexHull(all_pts_abs.reshape(-1, 1, 2)) if len(all_pts_abs) >= 3 else None

    return {
        "ok": True,
        "reason": "ok",
        "method": "legacy_dot_band_lines",
        "roi_used": (x, y, w, h),
        "counts": counts,
        "lines": lines_abs,
        "points_used_abs": {"left": pts_left_abs, "right": pts_right_abs, "bottom": pts_bottom_abs},
        "hull_abs": hull_abs,
    }


# ----------------------------
# True notch-edge detector: dark-region contour frame
# ----------------------------
def _smooth_column_boundary(xs: np.ndarray, ys: np.ndarray, *, radius: int = 5) -> np.ndarray:
    if len(ys) == 0:
        return ys

    radius = max(1, int(radius))
    out = np.asarray(ys, dtype=np.float32).copy()

    for i in range(len(ys)):
        a = max(0, i - radius)
        b = min(len(ys), i + radius + 1)
        out[i] = float(np.median(ys[a:b]))

    return out


def _prep_dark_region_mask(
    crop_bgr: np.ndarray,
    *,
    blur_ksize: int = 21,
    close_ksize: int = 19,
    open_ksize: int = 7,
    threshold_bias: float = 0.0,
):
    gray0 = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    # Heavy blur is intentional: it kills the printed dot field and leaves the actual glass/notch body.
    k = int(blur_ksize)
    if k % 2 == 0:
        k += 1
    k = max(7, k)
    blur = cv2.GaussianBlur(gray0, (k, k), 0)

    # Otsu usually separates dark glass from light frit/background.
    otsu_t, mask_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    if abs(float(threshold_bias)) > 1e-6:
        t = float(otsu_t) + float(threshold_bias)
        mask_inv = (blur < t).astype(np.uint8) * 255

    ck = int(close_ksize)
    if ck % 2 == 0:
        ck += 1
    ck = max(3, ck)

    ok = int(open_ksize)
    if ok % 2 == 0:
        ok += 1
    ok = max(3, ok)

    mask = cv2.morphologyEx(mask_inv, cv2.MORPH_CLOSE, np.ones((ck, ck), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((ok, ok), np.uint8), iterations=1)

    # Fill tiny holes after open/close.
    mask = cv2.medianBlur(mask, 5)

    return gray0, blur, mask_inv, mask, float(otsu_t)


def _select_dark_region_component(mask: np.ndarray):
    H, W = mask.shape[:2]
    num, labels, stats, _cent = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)

    best_label = None
    best_score = -1e18
    debug_components = []

    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])

        if area < max(250, int(0.015 * W * H)):
            continue

        bottom = y + h
        width_frac = w / max(1.0, float(W))
        height_frac = h / max(1.0, float(H))
        bottom_frac = bottom / max(1.0, float(H))

        # The desired component is the large dark glass body whose lower contour forms the notch.
        score = area + 3000.0 * bottom_frac + 1200.0 * width_frac + 600.0 * height_frac

        debug_components.append({"label": int(i), "area": area, "bbox": (x, y, w, h), "score": float(score)})

        if score > best_score:
            best_score = score
            best_label = i

    debug_components.sort(key=lambda d: d["score"], reverse=True)

    if best_label is None:
        return None, {"components": debug_components, "reason": "no_large_dark_component"}

    comp = ((labels == best_label).astype(np.uint8) * 255)
    return comp, {"components": debug_components[:8], "best_label": int(best_label), "best_score": float(best_score)}


def _extract_lower_boundary_points(component_mask: np.ndarray, *, min_dark_run_px: int = 10):
    H, W = component_mask.shape[:2]
    xs_out = []
    ys_out = []

    binary = component_mask > 0

    for x in range(W):
        ys = np.where(binary[:, x])[0]
        if len(ys) < int(min_dark_run_px):
            continue
        # Column-wise lower boundary of the dark glass region.
        xs_out.append(x)
        ys_out.append(int(np.max(ys)))

    if len(xs_out) < 40:
        return None

    xs_arr = np.asarray(xs_out, dtype=np.float32)
    ys_arr = np.asarray(ys_out, dtype=np.float32)
    ys_smooth = _smooth_column_boundary(xs_arr, ys_arr, radius=7)

    return np.column_stack([xs_arr, ys_smooth]).astype(np.float32)



def _line_point_distances(line, points) -> Optional[np.ndarray]:
    pts = _as_points(points)
    if pts is None or line is None:
        return None
    try:
        vx, vy, x0, y0 = map(float, line)
    except Exception:
        return None
    if not np.isfinite([vx, vy, x0, y0]).all():
        return None
    v = np.array([vx, vy], dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return None
    v /= n
    p0 = np.array([x0, y0], dtype=np.float64)
    d = pts.astype(np.float64) - p0[None, :]
    # 2D cross product magnitude = perpendicular distance to normalized line.
    return np.abs(d[:, 0] * v[1] - d[:, 1] * v[0]).astype(np.float32)


def _fit_line_robust(points, *, min_points: int = 12, iterations: int = 5, keep_percentile: float = 65.0):
    """
    Robust cv2.fitLine wrapper.

    It repeatedly fits a line and keeps the points closest to that line. This is
    especially useful for the bottom notch reference, where the candidate band can
    still include a little bit of the curved side walls. The final fit follows the
    straight bottom/tangent portion instead of being pulled horizontal by the full
    U-shaped contour.
    """
    pts = _as_points(points)
    if pts is None or len(pts) < int(min_points):
        return None, pts

    work = pts.copy()
    last_line = None

    for _ in range(max(1, int(iterations))):
        line = _fit_line(work)
        if line is None:
            break
        last_line = line

        dist = _line_point_distances(line, work)
        if dist is None or len(dist) < int(min_points):
            break

        q = float(np.clip(keep_percentile, 35.0, 95.0))
        cut = float(np.percentile(dist, q))
        cut = max(cut, 2.0)
        keep = work[dist <= cut]

        if len(keep) < int(min_points) or len(keep) == len(work):
            break
        work = keep

    if last_line is None:
        return None, pts

    final = _fit_line(work)
    if final is None:
        final = last_line

    return final, work


def _select_bottom_floor_points(pts: np.ndarray, *, y_min: float, y_max: float):
    """
    Selects the bottom-floor portion of the lower notch boundary.

    The old selector used every point in the lower percentile of the U shape.
    That worked when the glass was perfectly aligned, but once the glass sits at
    an angle, the side-wall curves can bias the reference. This selector finds the
    deep lower boundary first, then keeps the central/inlier portion and lets the
    robust line fit estimate the actual tilted bottom reference.
    """
    pts = _as_points(pts)
    if pts is None:
        return None

    xs = pts[:, 0]
    ys = pts[:, 1]
    y_span = max(1.0, float(y_max) - float(y_min))

    # Start from the lower contour band. This is deliberately a little generous;
    # the robust fit will reject curved side-wall tails.
    cut = float(y_min) + 0.70 * y_span
    cand = pts[ys >= cut]
    if len(cand) < 24:
        cand = pts[ys >= np.percentile(ys, 72)]
    if len(cand) < 16:
        cand = pts[ys >= np.percentile(ys, 65)]

    if len(cand) < 12:
        return cand

    cxs = cand[:, 0]
    # The floor is near the middle of the lower contour; drop the far tails first.
    xlo = float(np.percentile(cxs, 12))
    xhi = float(np.percentile(cxs, 88))
    central = cand[(cxs >= xlo) & (cxs <= xhi)]
    if len(central) >= 16:
        cand = central

    return cand.astype(np.float32)


def _bottom_line_angle_deg(line) -> Optional[float]:
    if line is None:
        return None
    try:
        vx, vy, _x0, _y0 = map(float, line)
    except Exception:
        return None
    if not np.isfinite([vx, vy]).all():
        return None
    # Normalize so the angle reads like a bottom reference from left->right.
    if vx < 0:
        vx, vy = -vx, -vy
    return float(np.degrees(np.arctan2(vy, vx)))


def _build_notch_frame_from_boundary(boundary: np.ndarray, W: int, H: int):
    pts = _as_points(boundary)
    if pts is None or len(pts) < 60:
        return None, {"reason": "not_enough_boundary_points"}

    xs = pts[:, 0]
    ys = pts[:, 1]

    y_min = float(np.percentile(ys, 3))
    y_max = float(np.percentile(ys, 97))
    y_span = max(1.0, y_max - y_min)

    # Fit the bottom reference first. This is a true tilted fit, not a forced
    # horizontal line.
    bottom_pts_seed = _select_bottom_floor_points(pts, y_min=y_min, y_max=y_max)
    if bottom_pts_seed is None or len(bottom_pts_seed) < 12:
        return None, {"reason": "not_enough_bottom_floor_points"}

    bottom_line, bottom_pts = _fit_line_robust(
        bottom_pts_seed,
        min_points=12,
        iterations=6,
        keep_percentile=62.0,
    )
    if bottom_line is None or bottom_pts is None or len(bottom_pts) < 12:
        return None, {"reason": "bottom_line_fit_failed"}

    # Only use this raw bottom-mid as a rough split point for separating side
    # wall point clouds. It is NOT exported as the actual anchor.
    bottom_mid_raw = _median_point(bottom_pts)
    if not _good_pt(bottom_mid_raw):
        return None, {"reason": "bottom_mid_seed_failed"}

    bottom_x = float(bottom_mid_raw[0])

    # Side-wall fit zones: avoid top strip and bottom-floor points.
    side_y0 = y_min + 0.18 * y_span
    side_y1 = y_min + 0.74 * y_span

    left_pts = pts[(xs < bottom_x) & (ys >= side_y0) & (ys <= side_y1)]
    right_pts = pts[(xs > bottom_x) & (ys >= side_y0) & (ys <= side_y1)]

    if len(left_pts) < 18:
        left_pts = pts[(xs < bottom_x) & (ys >= y_min + 0.12 * y_span) & (ys <= y_min + 0.82 * y_span)]
    if len(right_pts) < 18:
        right_pts = pts[(xs > bottom_x) & (ys >= y_min + 0.12 * y_span) & (ys <= y_min + 0.82 * y_span)]

    if len(left_pts) < 12 or len(right_pts) < 12:
        return None, {
            "reason": "not_enough_side_wall_points",
            "counts": {
                "left_wall": int(len(left_pts)),
                "right_wall": int(len(right_pts)),
                "bottom_seed": int(len(bottom_pts_seed)),
                "bottom_inliers": int(len(bottom_pts)),
            },
        }

    left_line, left_inliers = _fit_line_robust(left_pts, min_points=12, iterations=4, keep_percentile=78.0)
    right_line, right_inliers = _fit_line_robust(right_pts, min_points=12, iterations=4, keep_percentile=78.0)

    if left_line is None or right_line is None:
        return None, {"reason": "wall_line_fit_failed"}

    bottom_frame = _bottom_frame_from_lines(left_line, right_line, bottom_line)
    if bottom_frame is None:
        return None, {
            "reason": "bottom_left_right_intersection_failed",
            "counts": {
                "left_wall": int(len(left_pts)),
                "right_wall": int(len(right_pts)),
                "bottom_seed": int(len(bottom_pts_seed)),
                "bottom_inliers": int(len(bottom_pts)),
            },
        }

    # Compatibility top anchors for older modules. They are NOT used as the true
    # bottom-frame reference.
    top_y = float(y_min + 0.08 * y_span)
    left_top = _point_on_line_at_y(left_line, top_y)
    right_top = _point_on_line_at_y(right_line, top_y)
    if not _good_pt(left_top) or not _good_pt(right_top):
        return None, {"reason": "top_anchor_projection_failed"}

    anchors = {
        "left_top": (float(left_top[0]), float(left_top[1])),
        "right_top": (float(right_top[0]), float(right_top[1])),
        "bottom_left": bottom_frame["bottom_left"],
        "bottom_right": bottom_frame["bottom_right"],
        # Hard rule: bottom_mid is midpoint(bottom_left, bottom_right).
        "bottom_mid": bottom_frame["bottom_mid"],
    }

    frame = {
        "method": "dark_region_contour",
        "left_line": tuple(map(float, left_line)),
        "right_line": tuple(map(float, right_line)),
        "bottom_line": tuple(map(float, bottom_line)),
        "top_y": float(top_y),
        "y_min": float(y_min),
        "y_max": float(y_max),
        "y_span": float(y_span),
        **bottom_frame,
    }

    debug = {
        "counts": {
            "boundary": int(len(pts)),
            "left_wall": int(len(left_pts)),
            "left_wall_inliers": 0 if left_inliers is None else int(len(left_inliers)),
            "right_wall": int(len(right_pts)),
            "right_wall_inliers": 0 if right_inliers is None else int(len(right_inliers)),
            "bottom_seed": int(len(bottom_pts_seed)),
            "bottom": int(len(bottom_pts)),
        },
        "points_local": {
            "boundary": pts,
            "left_wall": left_inliers if left_inliers is not None else left_pts,
            "right_wall": right_inliers if right_inliers is not None else right_pts,
            "bottom": bottom_pts,
            "bottom_seed": bottom_pts_seed,
        },
        "bottom_frame_rule": "bottom_mid = midpoint(bottom_left, bottom_right)",
    }

    return {"anchors": anchors, "frame": frame}, debug

def _contour_from_mask(mask: np.ndarray):
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return contours[0]


def detect_notch_dark_region_frame_local(
    image_bgr: np.ndarray,
    roi_xywh_abs,
    *,
    blur_ksize: int = 21,
    close_ksize: int = 19,
    open_ksize: int = 7,
    threshold_bias: float = 0.0,
    min_dark_run_px: int = 10,
):
    """
    Primary notch geometry detector.

    This deliberately does NOT use dot edges as geometry.

    Pipeline:
      image crop
      -> heavily blurred dark-region segmentation
      -> largest dark glass component
      -> lower boundary of that component
      -> fit left wall / right wall / bottom reference
      -> derive anchors + notch coordinate frame

    All returned coordinates are absolute image coordinates.
    """
    if image_bgr is None or image_bgr.size == 0:
        return {"ok": False, "reason": "empty_image"}

    x, y, w, h = _clamp_roi_abs(image_bgr, roi_xywh_abs)
    crop = image_bgr[y:y + h, x:x + w]

    gray0, blur, mask_raw, mask_clean, otsu_t = _prep_dark_region_mask(
        crop,
        blur_ksize=blur_ksize,
        close_ksize=close_ksize,
        open_ksize=open_ksize,
        threshold_bias=threshold_bias,
    )

    comp_mask, comp_dbg = _select_dark_region_component(mask_clean)
    if comp_mask is None:
        return {
            "ok": False,
            "reason": comp_dbg.get("reason", "component_failed"),
            "method": "dark_region_contour",
            "roi_used": (x, y, w, h),
            "component_debug": comp_dbg,
            "dbg": {"gray0": gray0, "blur": blur, "mask_raw": mask_raw, "mask_clean": mask_clean},
        }

    boundary_local = _extract_lower_boundary_points(comp_mask, min_dark_run_px=min_dark_run_px)
    if boundary_local is None:
        return {
            "ok": False,
            "reason": "lower_boundary_failed",
            "method": "dark_region_contour",
            "roi_used": (x, y, w, h),
            "component_debug": comp_dbg,
            "dbg": {"gray0": gray0, "blur": blur, "mask_raw": mask_raw, "mask_clean": mask_clean, "component_mask": comp_mask},
        }

    model, frame_dbg = _build_notch_frame_from_boundary(boundary_local, w, h)
    if model is None:
        return {
            "ok": False,
            "reason": frame_dbg.get("reason", "notch_frame_failed"),
            "method": "dark_region_contour",
            "roi_used": (x, y, w, h),
            "component_debug": comp_dbg,
            "frame_debug": frame_dbg,
            "points_used_abs": {"boundary": _shift_points(boundary_local, x, y)},
            "dbg": {"gray0": gray0, "blur": blur, "mask_raw": mask_raw, "mask_clean": mask_clean, "component_mask": comp_mask},
        }

    anchors_local = model["anchors"]
    frame_local = model["frame"]

    def pt_abs(p):
        return (float(p[0] + x), float(p[1] + y))

    anchors_abs = {
        "left_top": pt_abs(anchors_local["left_top"]),
        "right_top": pt_abs(anchors_local["right_top"]),
        "bottom_left": pt_abs(anchors_local["bottom_left"]),
        "bottom_right": pt_abs(anchors_local["bottom_right"]),
        # Hard rule: bottom_mid is the midpoint of the two exported bottom anchors.
        "bottom_mid": (
            0.5 * (pt_abs(anchors_local["bottom_left"])[0] + pt_abs(anchors_local["bottom_right"])[0]),
            0.5 * (pt_abs(anchors_local["bottom_left"])[1] + pt_abs(anchors_local["bottom_right"])[1]),
        ),
    }

    frame_abs = {
        **frame_local,
        "left_line": _shift_line(frame_local["left_line"], x, y),
        "right_line": _shift_line(frame_local["right_line"], x, y),
        "bottom_line": _shift_line(frame_local["bottom_line"], x, y),
        "bottom_left": anchors_abs["bottom_left"],
        "bottom_right": anchors_abs["bottom_right"],
        "bottom_mid": anchors_abs["bottom_mid"],
        "origin": anchors_abs["bottom_mid"],
        "top_y": float(frame_local["top_y"] + y),
        "y_min": float(frame_local["y_min"] + y),
        "y_max": float(frame_local["y_max"] + y),
    }

    pts_local = frame_dbg.get("points_local", {})
    points_used_abs = {
        "boundary": _shift_points(boundary_local, x, y),
        "left_wall": _shift_points(pts_local.get("left_wall", np.empty((0, 2), dtype=np.float32)), x, y),
        "right_wall": _shift_points(pts_local.get("right_wall", np.empty((0, 2), dtype=np.float32)), x, y),
        "bottom": _shift_points(pts_local.get("bottom", np.empty((0, 2), dtype=np.float32)), x, y),
    }

    contour_local = _contour_from_mask(comp_mask)
    contour_abs = None
    if contour_local is not None:
        contour_abs = contour_local + np.array([[[x, y]]], dtype=np.int32)

    return {
        "ok": True,
        "reason": "ok",
        "method": "dark_region_contour",
        "roi_used": (x, y, w, h),
        "anchors": anchors_abs,
        "notch_frame": frame_abs,
        # Common direct keys for recovered modules / overlays.
        "bottom_left": anchors_abs["bottom_left"],
        "bottom_right": anchors_abs["bottom_right"],
        "bottom_mid": anchors_abs["bottom_mid"],
        "origin": anchors_abs["bottom_mid"],
        "x_axis": frame_abs.get("x_axis"),
        "y_axis": frame_abs.get("y_axis"),
        "width": frame_abs.get("width"),
        "width_px": frame_abs.get("width_px"),
        "angle_deg": frame_abs.get("angle_deg"),
        "lines": {
            "left": frame_abs["left_line"],
            "right": frame_abs["right_line"],
            "bottom": frame_abs["bottom_line"],
        },
        "points_used_abs": points_used_abs,
        "contour_abs": contour_abs,
        "component_debug": comp_dbg,
        "frame_debug": {k: v for k, v in frame_dbg.items() if k != "points_local"},
        "counts": frame_dbg.get("counts", {}),
        "threshold": {"otsu": float(otsu_t), "bias": float(threshold_bias)},
        "dbg": {"gray0": gray0, "blur": blur, "mask_raw": mask_raw, "mask_clean": mask_clean, "component_mask": comp_mask},
    }


# Backwards-compatible alias for the previous "solid edge" name.
def detect_notch_solid_edge_anchors_local(image_bgr: np.ndarray, roi_xywh_abs, **kwargs):
    return detect_notch_dark_region_frame_local(image_bgr, roi_xywh_abs, **kwargs)


# ----------------------------
# Notch-relative measurement
# ----------------------------
def measure_baseplate_against_notch_frame(center_abs, notch_frame: dict) -> Optional[Dict[str, Any]]:
    """
    Measures baseplate center in the actual notch coordinate frame.

    dx_from_notch_center:
      center_x - midpoint(left_wall_x, right_wall_x) at the baseplate's y.

    dy_from_bottom:
      bottom_line_y_at_center_x - center_y.
      Positive means the baseplate is above the bottom reference line.
    """
    if center_abs is None or not isinstance(notch_frame, dict):
        return None

    try:
        cx = float(center_abs[0])
        cy = float(center_abs[1])
    except Exception:
        return None

    left_line = notch_frame.get("left_line")
    right_line = notch_frame.get("right_line")
    bottom_line = notch_frame.get("bottom_line")

    lx = _line_x_at_y(left_line, cy)
    rx = _line_x_at_y(right_line, cy)
    by = _line_y_at_x(bottom_line, cx)

    if lx is None or rx is None or by is None:
        return None

    if not np.isfinite([lx, rx, by, cx, cy]).all():
        return None

    notch_center_x = 0.5 * (float(lx) + float(rx))
    notch_width_at_y = float(abs(rx - lx))
    dx_from_notch_center = float(cx - notch_center_x)
    dy_from_bottom = float(by - cy)

    return {
        "center_abs": (float(cx), float(cy)),
        "left_x_at_center_y": float(lx),
        "right_x_at_center_y": float(rx),
        "notch_center_x_at_center_y": float(notch_center_x),
        "notch_width_at_center_y": float(notch_width_at_y),
        "bottom_y_at_center_x": float(by),
        "dx_from_notch_center": float(dx_from_notch_center),
        "dy_from_bottom": float(dy_from_bottom),
    }


def detect_baseplate_in_roi(full_img: np.ndarray, roi_xywh_abs, *, return_debug: bool = False, **kwargs):
    if full_img is None or roi_xywh_abs is None:
        if return_debug:
            return {"ok": False, "center_abs": None, "angle": None, "contour_abs": None, "reason": "no_image_or_roi"}
        return None

    x, y, w, h = _clamp_roi_abs(full_img, roi_xywh_abs)
    crop = full_img[y:y + h, x:x + w]

    center_rel, ang, cnt_rel, dbg = detect_baseplate(
        crop,
        full_image_bgr=full_img,
        roi_xywh_abs=(x, y, w, h),
        return_debug=return_debug,
        **kwargs,
    )

    if center_rel is None:
        if return_debug:
            return {"ok": False, "center_abs": None, "angle": None, "contour_abs": None, "reason": "not_found", "dbg": dbg}
        return None

    cx_abs = float(x + center_rel[0])
    cy_abs = float(y + center_rel[1])

    cnt_abs = None
    if cnt_rel is not None:
        cnt_abs = cnt_rel + np.array([[x, y]], dtype=np.int32)

    if return_debug:
        return {"ok": True, "center_abs": (cx_abs, cy_abs), "angle": float(ang) if ang is not None else None, "contour_abs": cnt_abs, "dbg": dbg}

    return (cx_abs, cy_abs)
