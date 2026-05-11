# detector.py (REWRITE)
from __future__ import annotations

import cv2
import numpy as np
from typing import Dict, Optional, Tuple, Any, List


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
# Preprocess (debuggable)
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
    """
    Returns:
      gray_final, edges_final, debug_frames
    """
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
        clahe = cv2.createCLAHE(
            clipLimit=float(clahe_clip),
            tileGridSize=(int(clahe_grid), int(clahe_grid)),
        )
        gray2 = clahe.apply(gray1)
    else:
        gray2 = cv2.equalizeHist(gray1)

    k = int(blur_ksize)
    if k % 2 == 0:
        k += 1
    k = max(3, k)

    gray3 = cv2.GaussianBlur(gray2, (k, k), 0)

    edges0 = cv2.Canny(
        gray3,
        int(canny_low),
        int(canny_high),
        apertureSize=3,
        L2gradient=True,
    )
    edges1 = cv2.dilate(
        edges0,
        np.ones((3, 3), np.uint8),
        iterations=int(max(0, dilate_iter)),
    )
    edges2 = cv2.morphologyEx(
        edges1,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
        iterations=int(max(0, close_iter)),
    )

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
            if (
                x <= border_margin
                or y <= border_margin
                or x + w >= W - border_margin
                or y + h >= H - border_margin
            ):
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

    # make angle refer to long axis
    if w < h:
        angle += 90.0

    # vertical=0 convention
    angle -= 90.0

    # wrap to [-90, 90]
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
    """
    Detect baseplate in this window.
    Returns all coords in WINDOW coordinates.

    center_xy_window, angle_deg, contour_window, debug_dict
    """
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

    if best is None:
        if return_debug:
            return None, None, None, {
                "ok": False,
                "reason": "no_contour",
                "dbg": dbg_frames,
            }
        return None, None, None, None

    # convert contour to WINDOW coords by adding the inner offset
    cnt_win = best + np.array([[sx, sy]], dtype=np.int32)

    rect = cv2.minAreaRect(cnt_win)
    (cx, cy), _, _ = rect
    angle = _normalize_min_area_angle(rect)

    if return_debug:
        return (float(cx), float(cy)), angle, cnt_win, {
            "ok": True,
            "reason": "ok",
            "dbg": dbg_frames,
            "rect": rect,
        }

    return (float(cx), float(cy)), angle, cnt_win, None


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
    Main API:
      - Always returns center + contour in ROI-CROP coordinates.
      - angle in degrees.

    If detection fails inside the crop, and full_image + roi_xywh_abs are provided,
    it searches a padded window in full-image coords, then converts back into
    ROI-CROP coords.
    """
    c, a, cnt, dbg = _detect_in_window(
        cropped_roi_bgr,
        return_debug=return_debug,
        **kwargs,
    )

    if c is not None:
        if return_debug:
            return (c[0], c[1]), a, cnt, {"stage": "crop", **dbg}
        return (c[0], c[1]), a, cnt

    if full_image_bgr is not None and roi_xywh_abs is not None and int(padding) > 0:
        rx, ry, rw, rh = map(int, roi_xywh_abs)
        H, W = full_image_bgr.shape[:2]

        x0 = max(0, rx - int(padding))
        y0 = max(0, ry - int(padding))
        x1 = min(W, rx + rw + int(padding))
        y1 = min(H, ry + rh + int(padding))

        win = full_image_bgr[y0:y1, x0:x1]

        c2, a2, cnt2, dbg2 = _detect_in_window(
            win,
            return_debug=return_debug,
            **kwargs,
        )

        if c2 is not None:
            cx_abs = float(x0 + c2[0])
            cy_abs = float(y0 + c2[1])

            cx_rel = cx_abs - rx
            cy_rel = cy_abs - ry

            cnt_rel = None
            if cnt2 is not None:
                cnt_abs = cnt2 + np.array([[x0, y0]], dtype=np.int32)
                cnt_rel = cnt_abs - np.array([[rx, ry]], dtype=np.int32)

            if return_debug:
                return (cx_rel, cy_rel), a2, cnt_rel, {
                    "stage": "padded_full",
                    **dbg2,
                    "pad_window_abs": (x0, y0, x1 - x0, y1 - y0),
                }

            return (cx_rel, cy_rel), a2, cnt_rel

    if return_debug:
        return None, None, None, {
            "stage": "fail",
            "ok": False,
            "reason": "not_found",
        }

    return None, None, None


# ----------------------------
# Legacy inner border lines / dot-band detection
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
    """
    Legacy detector:
    Detect notch inner borders using edge sampling in 3 bands and cv2.fitLine.
    Returns lines in ABS coordinates.

    Kept as fallback/debug.
    """
    if image_bgr is None or image_bgr.size == 0:
        return {"ok": False, "reason": "empty_image"}

    x, y, w, h = map(int, roi_xywh_abs)

    H, W = image_bgr.shape[:2]
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))

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

    counts = {
        "left": int(len(pts_left)),
        "right": int(len(pts_right)),
        "bottom": int(len(pts_bottom)),
    }

    for k2, n in counts.items():
        if n < int(min_points):
            return {
                "ok": False,
                "reason": f"not_enough_points_{k2}",
                "counts": counts,
                "roi_used": (x, y, w, h),
            }

    def fit_line(pts):
        vx, vy, x0, y0 = cv2.fitLine(
            pts.reshape(-1, 1, 2),
            cv2.DIST_L2,
            0,
            0.01,
            0.01,
        ).flatten()
        return float(vx), float(vy), float(x0), float(y0)

    def shift_line(line, dx, dy):
        vx, vy, x0, y0 = line
        return (vx, vy, x0 + dx, y0 + dy)

    line_left = fit_line(pts_left)
    line_right = fit_line(pts_right)
    line_bottom = fit_line(pts_bottom)

    lines_abs = {
        "left": shift_line(line_left, x, y),
        "right": shift_line(line_right, x, y),
        "bottom": shift_line(line_bottom, x, y),
    }

    pts_left_abs = pts_left + np.array([x, y], dtype=np.float32)
    pts_right_abs = pts_right + np.array([x, y], dtype=np.float32)
    pts_bottom_abs = pts_bottom + np.array([x, y], dtype=np.float32)

    all_pts_abs = np.vstack([pts_left_abs, pts_right_abs, pts_bottom_abs]).astype(np.int32)
    hull_abs = cv2.convexHull(all_pts_abs.reshape(-1, 1, 2)) if len(all_pts_abs) >= 3 else None

    return {
        "ok": True,
        "reason": "ok",
        "roi_used": (x, y, w, h),
        "counts": counts,
        "lines": lines_abs,
        "points_used_abs": {
            "left": pts_left_abs,
            "right": pts_right_abs,
            "bottom": pts_bottom_abs,
        },
        "hull_abs": hull_abs,
    }


# ----------------------------
# Solid notch edge anchor detection
# ----------------------------
def _clamp_roi_abs(image_bgr: np.ndarray, roi_xywh_abs):
    x, y, w, h = map(int, roi_xywh_abs)
    H, W = image_bgr.shape[:2]

    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))

    return x, y, w, h


def _prep_solid_edge_mask(
    crop_bgr: np.ndarray,
    *,
    blur_ksize: int = 5,
    use_clahe: bool = True,
    clahe_clip: float = 1.5,
    canny_low: int = 45,
    canny_high: int = 125,
    close_iter: int = 2,
    dilate_iter: int = 1,
):
    gray0 = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    if use_clahe:
        clahe = cv2.createCLAHE(
            clipLimit=float(clahe_clip),
            tileGridSize=(8, 8),
        )
        gray1 = clahe.apply(gray0)
    else:
        gray1 = cv2.equalizeHist(gray0)

    k = int(blur_ksize)
    if k % 2 == 0:
        k += 1
    k = max(3, k)

    gray2 = cv2.GaussianBlur(gray1, (k, k), 0)

    edges0 = cv2.Canny(
        gray2,
        int(canny_low),
        int(canny_high),
        apertureSize=3,
        L2gradient=True,
    )

    edges1 = cv2.morphologyEx(
        edges0,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8),
        iterations=int(max(0, close_iter)),
    )

    edges2 = cv2.dilate(
        edges1,
        np.ones((3, 3), np.uint8),
        iterations=int(max(0, dilate_iter)),
    )

    return gray0, gray1, gray2, edges0, edges2


def _component_points_from_edges(edges: np.ndarray, *, min_area_px: int = 200):
    """
    Find connected edge components and return filtered point clouds.
    """
    num, labels, stats, _cent = cv2.connectedComponentsWithStats(
        (edges > 0).astype(np.uint8),
        connectivity=8,
    )

    comps = []

    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < int(min_area_px):
            continue

        ys, xs = np.where(labels == i)

        if len(xs) <= 0:
            continue

        pts = np.column_stack([xs, ys]).astype(np.float32)

        x0 = int(stats[i, cv2.CC_STAT_LEFT])
        y0 = int(stats[i, cv2.CC_STAT_TOP])
        ww = int(stats[i, cv2.CC_STAT_WIDTH])
        hh = int(stats[i, cv2.CC_STAT_HEIGHT])

        comps.append(
            {
                "area": area,
                "bbox": (x0, y0, ww, hh),
                "points": pts,
            }
        )

    return comps


def _score_notch_component(comp, W: int, H: int):
    pts = comp["points"]
    x0, y0, w, h = comp["bbox"]
    area = float(comp["area"])

    if len(pts) <= 0:
        return -1e18

    # We want a wide-ish, lower/central boundary component.
    width_score = float(w) / max(1.0, float(W))
    height_score = float(h) / max(1.0, float(H))

    cx = float(np.mean(pts[:, 0]))
    cy = float(np.mean(pts[:, 1]))

    center_penalty = abs(cx - W * 0.5) / max(1.0, W * 0.5)

    # Prefer components that include lower notch/bottom curve.
    bottom_reach = float(np.percentile(pts[:, 1], 95)) / max(1.0, float(H))

    # Avoid picking top glass horizon only.
    top_line_penalty = 1.0 if h < 0.18 * H else 0.0

    return (
        area
        + 1500.0 * width_score
        + 1200.0 * height_score
        + 1800.0 * bottom_reach
        - 800.0 * center_penalty
        - 2500.0 * top_line_penalty
    )


def _select_solid_notch_points(
    edges: np.ndarray,
    *,
    min_component_area_px: int = 200,
):
    H, W = edges.shape[:2]

    comps = _component_points_from_edges(
        edges,
        min_area_px=int(min_component_area_px),
    )

    if not comps:
        return None, {
            "components": [],
            "reason": "no_components",
        }

    scored = []
    for c in comps:
        scored.append((_score_notch_component(c, W, H), c))

    scored.sort(key=lambda t: t[0], reverse=True)
    best_score, best = scored[0]

    # Keep a compact debug list.
    dbg_components = []
    for score, c in scored[:8]:
        dbg_components.append(
            {
                "score": float(score),
                "area": int(c["area"]),
                "bbox": tuple(map(int, c["bbox"])),
            }
        )

    return best["points"], {
        "components": dbg_components,
        "best_score": float(best_score),
        "best_bbox": tuple(map(int, best["bbox"])),
    }


def _split_solid_notch_points(points_xy: np.ndarray, W: int, H: int):
    """
    Split a solid edge point cloud into left / right / bottom point groups.

    This deliberately avoids fitting over the full curved boundary.
    Instead, it uses local percentile windows.
    """
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    pts = pts[np.isfinite(pts).all(axis=1)]

    if len(pts) < 40:
        return None

    xs = pts[:, 0]
    ys = pts[:, 1]

    x10 = float(np.percentile(xs, 10))
    x35 = float(np.percentile(xs, 35))
    x65 = float(np.percentile(xs, 65))
    x90 = float(np.percentile(xs, 90))
    y10 = float(np.percentile(ys, 10))
    y25 = float(np.percentile(ys, 25))
    y75 = float(np.percentile(ys, 75))
    y90 = float(np.percentile(ys, 90))

    # Left/right arcs: use side-ish parts, avoid very top horizon noise.
    left = pts[(xs <= x35) & (ys >= y10)]
    right = pts[(xs >= x65) & (ys >= y10)]

    # Bottom arc: lower part of the contour.
    bottom = pts[ys >= y75]

    # Top anchor support: upper side regions, not the whole side curve.
    left_top = pts[(xs <= x35) & (ys <= y25)]
    right_top = pts[(xs >= x65) & (ys <= y25)]

    return {
        "left": left,
        "right": right,
        "bottom": bottom,
        "left_top": left_top,
        "right_top": right_top,
        "percentiles": {
            "x10": x10,
            "x35": x35,
            "x65": x65,
            "x90": x90,
            "y10": y10,
            "y25": y25,
            "y75": y75,
            "y90": y90,
        },
    }


def _median_point(points_xy):
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    pts = pts[np.isfinite(pts).all(axis=1)]

    if len(pts) <= 0:
        return None

    return (
        float(np.median(pts[:, 0])),
        float(np.median(pts[:, 1])),
    )


def _robust_bottom_mid_from_points(points_xy):
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    pts = pts[np.isfinite(pts).all(axis=1)]

    if len(pts) <= 0:
        return None

    ys = pts[:, 1]
    cut = float(np.percentile(ys, 65))
    low = pts[ys >= cut]

    if len(low) <= 0:
        low = pts

    return (
        float(np.median(low[:, 0])),
        float(np.median(low[:, 1])),
    )


def _line_from_points_abs(points_abs):
    pts = np.asarray(points_abs, dtype=np.float32).reshape(-1, 2)
    pts = pts[np.isfinite(pts).all(axis=1)]

    if len(pts) < 8:
        return None

    vx, vy, x0, y0 = cv2.fitLine(
        pts.reshape(-1, 1, 2),
        cv2.DIST_L2,
        0,
        0.01,
        0.01,
    ).flatten()

    return float(vx), float(vy), float(x0), float(y0)


def detect_notch_solid_edge_anchors_local(
    image_bgr: np.ndarray,
    roi_xywh_abs,
    *,
    canny_low: int = 45,
    canny_high: int = 125,
    blur_ksize: int = 5,
    clahe_clip: float = 1.5,
    close_iter: int = 2,
    dilate_iter: int = 1,
    min_component_area_px: int = 200,
    min_anchor_points: int = 20,
):
    """
    Primary stabilizer signal.

    Detects the solid dark/light notch boundary instead of relying on dot-band points.

    Returns:
      {
        ok,
        anchors: {
          left_top,
          right_top,
          bottom_mid
        },
        points_used_abs,
        contour_abs,
        lines,          # optional debug/fallback-style fitted lines
        hull_abs,
        ...
      }

    All coordinates are absolute image coordinates.
    """
    if image_bgr is None or image_bgr.size == 0:
        return {"ok": False, "reason": "empty_image"}

    x, y, w, h = _clamp_roi_abs(image_bgr, roi_xywh_abs)
    crop = image_bgr[y:y + h, x:x + w]

    gray0, gray1, gray2, edges0, edges = _prep_solid_edge_mask(
        crop,
        blur_ksize=blur_ksize,
        use_clahe=True,
        clahe_clip=clahe_clip,
        canny_low=canny_low,
        canny_high=canny_high,
        close_iter=close_iter,
        dilate_iter=dilate_iter,
    )

    pts_local, comp_dbg = _select_solid_notch_points(
        edges,
        min_component_area_px=min_component_area_px,
    )

    if pts_local is None or len(pts_local) < int(min_anchor_points) * 3:
        return {
            "ok": False,
            "reason": "solid_edge_not_found",
            "roi_used": (x, y, w, h),
            "component_debug": comp_dbg,
            "dbg": {
                "gray0": gray0,
                "gray1_contrast": gray1,
                "gray2_blur": gray2,
                "edges0": edges0,
                "edges": edges,
            },
        }

    split = _split_solid_notch_points(pts_local, w, h)

    if split is None:
        return {
            "ok": False,
            "reason": "solid_edge_split_failed",
            "roi_used": (x, y, w, h),
            "component_debug": comp_dbg,
        }

    left_top_pts = split["left_top"]
    right_top_pts = split["right_top"]
    bottom_pts = split["bottom"]

    counts_local = {
        "left_top": int(len(left_top_pts)),
        "right_top": int(len(right_top_pts)),
        "bottom": int(len(bottom_pts)),
        "all": int(len(pts_local)),
    }

    if (
        counts_local["left_top"] < int(min_anchor_points)
        or counts_local["right_top"] < int(min_anchor_points)
        or counts_local["bottom"] < int(min_anchor_points)
    ):
        return {
            "ok": False,
            "reason": "not_enough_solid_edge_anchor_points",
            "counts": counts_local,
            "roi_used": (x, y, w, h),
            "component_debug": comp_dbg,
        }

    left_top_local = _median_point(left_top_pts)
    right_top_local = _median_point(right_top_pts)
    bottom_mid_local = _robust_bottom_mid_from_points(bottom_pts)

    if left_top_local is None or right_top_local is None or bottom_mid_local is None:
        return {
            "ok": False,
            "reason": "solid_edge_anchor_calc_failed",
            "counts": counts_local,
            "roi_used": (x, y, w, h),
            "component_debug": comp_dbg,
        }

    def to_abs_pt(p):
        return (float(p[0] + x), float(p[1] + y))

    anchors_abs = {
        "left_top": to_abs_pt(left_top_local),
        "right_top": to_abs_pt(right_top_local),
        "bottom_mid": to_abs_pt(bottom_mid_local),
    }

    all_abs = pts_local + np.array([x, y], dtype=np.float32)
    left_abs = split["left"] + np.array([x, y], dtype=np.float32)
    right_abs = split["right"] + np.array([x, y], dtype=np.float32)
    bottom_abs = split["bottom"] + np.array([x, y], dtype=np.float32)
    left_top_abs = left_top_pts + np.array([x, y], dtype=np.float32)
    right_top_abs = right_top_pts + np.array([x, y], dtype=np.float32)

    hull_abs = None
    try:
        hull_abs = cv2.convexHull(all_abs.astype(np.int32).reshape(-1, 1, 2))
    except Exception:
        hull_abs = None

    # Debug lines only. roi_stablizer can ignore these.
    lines_abs = {
        "left": _line_from_points_abs(left_abs),
        "right": _line_from_points_abs(right_abs),
        "bottom": _line_from_points_abs(bottom_abs),
    }

    return {
        "ok": True,
        "reason": "ok",
        "method": "solid_edge_anchors",
        "roi_used": (x, y, w, h),
        "counts": counts_local,
        "anchors": anchors_abs,
        "points_used_abs": {
            "all": all_abs,
            "left": left_abs,
            "right": right_abs,
            "bottom": bottom_abs,
            "left_top": left_top_abs,
            "right_top": right_top_abs,
        },
        "lines": lines_abs,
        "hull_abs": hull_abs,
        "component_debug": comp_dbg,
        "dbg": {
            "gray0": gray0,
            "gray1_contrast": gray1,
            "gray2_blur": gray2,
            "edges0": edges0,
            "edges": edges,
        },
    }


def detect_baseplate_in_roi(
    full_img: np.ndarray,
    roi_xywh_abs,
    *,
    return_debug: bool = False,
    **kwargs,
):
    """
    Wrapper that returns ABS coords for center & contour consistently.
    """
    if full_img is None or roi_xywh_abs is None:
        if return_debug:
            return {
                "ok": False,
                "center_abs": None,
                "angle": None,
                "contour_abs": None,
                "reason": "no_image_or_roi",
            }
        return None

    x, y, w, h = map(int, roi_xywh_abs)
    H, W = full_img.shape[:2]

    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))

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
            return {
                "ok": False,
                "center_abs": None,
                "angle": None,
                "contour_abs": None,
                "reason": "not_found",
                "dbg": dbg,
            }
        return None

    cx_abs = float(x + center_rel[0])
    cy_abs = float(y + center_rel[1])

    cnt_abs = None
    if cnt_rel is not None:
        cnt_abs = cnt_rel + np.array([[x, y]], dtype=np.int32)

    if return_debug:
        return {
            "ok": True,
            "center_abs": (cx_abs, cy_abs),
            "angle": float(ang) if ang is not None else None,
            "contour_abs": cnt_abs,
            "dbg": dbg,
        }

    return (cx_abs, cy_abs)