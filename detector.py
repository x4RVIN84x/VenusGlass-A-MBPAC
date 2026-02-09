# detector.py (REWRITE)
import cv2
import numpy as np
from dataclasses import dataclass
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
    gray1 = cv2.bilateralFilter(gray0, d=bilateral_d,
                                sigmaColor=bilateral_sigma_color,
                                sigmaSpace=bilateral_sigma_space)

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
    (cx, cy), (w, h), angle = rect

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
    Detect baseplate in *this window*.
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
        gray, contours,
        inner.shape[1], inner.shape[0],
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
            return None, None, None, {"ok": False, "reason": "no_contour", "dbg": dbg_frames}
        return None, None, None, None

    # convert contour to WINDOW coords by adding the inner offset (sx, sy)
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
    Main API (CONSISTENT COORD FRAMES):
      - Always returns center + contour in ROI-CROP coordinates (0..w,0..h)
      - angle in degrees

    If detection fails inside the crop, and full_image+roi_xywh_abs are provided,
    it will search a padded window in full-image coordinates, then convert back
    into ROI-CROP coordinates.
    """
    # Try inside crop first
    c, a, cnt, dbg = _detect_in_window(cropped_roi_bgr, return_debug=return_debug, **kwargs)
    if c is not None:
        if return_debug:
            return (c[0], c[1]), a, cnt, {"stage": "crop", **dbg}
        return (c[0], c[1]), a, cnt

    # Fallback: padded search in full image, then convert back to ROI-crop coords
    if full_image_bgr is not None and roi_xywh_abs is not None and int(padding) > 0:
        rx, ry, rw, rh = map(int, roi_xywh_abs)
        H, W = full_image_bgr.shape[:2]
        x0 = max(0, rx - int(padding))
        y0 = max(0, ry - int(padding))
        x1 = min(W, rx + rw + int(padding))
        y1 = min(H, ry + rh + int(padding))

        win = full_image_bgr[y0:y1, x0:x1]
        c2, a2, cnt2, dbg2 = _detect_in_window(win, return_debug=return_debug, **kwargs)

        if c2 is not None:
            # window->abs
            cx_abs = float(x0 + c2[0])
            cy_abs = float(y0 + c2[1])

            # abs->roi-crop
            cx_rel = cx_abs - rx
            cy_rel = cy_abs - ry

            cnt_rel = None
            if cnt2 is not None:
                # cnt2 is in window coords -> abs -> roi-crop
                cnt_abs = cnt2 + np.array([[x0, y0]], dtype=np.int32)
                cnt_rel = cnt_abs - np.array([[rx, ry]], dtype=np.int32)

            if return_debug:
                return (cx_rel, cy_rel), a2, cnt_rel, {"stage": "padded_full", **dbg2, "pad_window_abs": (x0, y0, x1 - x0, y1 - y0)}
            return (cx_rel, cy_rel), a2, cnt_rel

    if return_debug:
        return None, None, None, {"stage": "fail", "ok": False, "reason": "not_found"}
    return None, None, None


# ----------------------------
# Inner border lines (notch) detection
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
    Detect notch inner borders using edge sampling in 3 bands and cv2.fitLine.
    Returns lines in ABS coordinates.
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

    counts = {"left": int(len(pts_left)), "right": int(len(pts_right)), "bottom": int(len(pts_bottom))}
    for k2, n in counts.items():
        if n < int(min_points):
            return {"ok": False, "reason": f"not_enough_points_{k2}", "counts": counts, "roi_used": (x, y, w, h)}

    def fit_line(pts):
        vx, vy, x0, y0 = cv2.fitLine(pts.reshape(-1, 1, 2), cv2.DIST_L2, 0, 0.01, 0.01).flatten()
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
        "points_used_abs": {"left": pts_left_abs, "right": pts_right_abs, "bottom": pts_bottom_abs},
        "hull_abs": hull_abs,
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
            return {"ok": False, "center_abs": None, "angle": None, "contour_abs": None, "reason": "no_image_or_roi"}
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
            return {"ok": False, "center_abs": None, "angle": None, "contour_abs": None, "reason": "not_found", "dbg": dbg}
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
