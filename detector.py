import cv2
import numpy as np

# ------------------------------------------------------------
# Baseplate detection (contour-based, ROI-constrained)
# ------------------------------------------------------------

def _preprocess_edges(bgr, canny_low, canny_high):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=5, sigmaColor=40, sigmaSpace=40)
    gray = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high, apertureSize=3, L2gradient=True)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    return gray, edges


def _contrast_score(gray, cnt):
    mask = np.zeros(gray.shape[:2], np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, thickness=-1)
    inside_mean = cv2.mean(gray, mask=mask)[0]
    outside_mean = cv2.mean(gray, mask=cv2.bitwise_not(mask))[0]
    return abs(inside_mean - outside_mean)


def _pick_best_contour(gray, contours, W, H,
                       area_min_frac, area_max_frac,
                       aspect_min, aspect_max,
                       solidity_min, extent_min,
                       border_margin, contrast_min):
    win_area = float(W * H)
    best = None
    best_score = -1e18

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        if border_margin is not None:
            if x <= border_margin or y <= border_margin or x + w >= W - border_margin or y + h >= H - border_margin:
                continue

        area = cv2.contourArea(c)
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
        hull_area = cv2.contourArea(hull) + 1e-6
        solidity = float(area) / hull_area
        if solidity < solidity_min:
            continue

        extent = float(area) / float(w * h + 1e-6)
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


def _detect_in_window(bgr,
                      canny_low=50,
                      canny_high=120,
                      area_min_frac=0.02,
                      area_max_frac=0.60,
                      aspect_min=0.5,
                      aspect_max=2.2,
                      solidity_min=0.7,
                      extent_min=0.25,
                      border_margin=12,
                      contrast_min=12.0,
                      shrink_border_px=10):
    if bgr is None or bgr.size == 0:
        return None, None, None

    H0, W0 = bgr.shape[:2]
    sx = min(shrink_border_px, max(0, W0 // 10))
    sy = min(shrink_border_px, max(0, H0 // 10))
    x0, y0 = sx, sy
    w0, h0 = max(1, W0 - 2 * sx), max(1, H0 - 2 * sy)
    inner = bgr[y0:y0 + h0, x0:x0 + w0]

    gray, edges = _preprocess_edges(inner, canny_low, canny_high)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = _pick_best_contour(gray, contours, w0, h0,
                              area_min_frac, area_max_frac,
                              aspect_min, aspect_max,
                              solidity_min, extent_min,
                              border_margin, contrast_min)
    if best is None:
        return None, None, None

    rect = cv2.minAreaRect(best)
    (cx, cy), _, angle = rect
    best_off = best + np.array([[x0, y0]])
    return (float(cx + x0), float(cy + y0)), float(round(angle, 2)), best_off


def detect_baseplate(cropped_roi,
                     full_image=None,
                     roi=None,
                     padding=150,
                     shrink_border_px=10,
                     border_margin=12,
                     **kwargs):
    """
    Returns:
      center_rel_to_crop (x,y), angle, contour (in crop coords)
    """
    c, a, cnt = _detect_in_window(
        cropped_roi,
        shrink_border_px=shrink_border_px,
        border_margin=border_margin,
        **kwargs
    )
    if c is not None:
        return c, a, cnt

    if full_image is not None and roi is not None and padding > 0:
        rx, ry, rw, rh = roi
        H, W = full_image.shape[:2]
        x0 = max(0, rx - padding)
        y0 = max(0, ry - padding)
        x1 = min(W, rx + rw + padding)
        y1 = min(H, ry + rh + padding)
        win = full_image[y0:y1, x0:x1]

        c2, a2, cnt2 = _detect_in_window(
            win,
            shrink_border_px=shrink_border_px,
            border_margin=border_margin,
            **kwargs
        )
        if c2 is not None:
            return (c2[0] + x0 - rx, c2[1] + y0 - ry), a2, cnt2

    return None, None, None


def detect_baseplate_in_roi(full_bgr, roi_xywh):
    """
    Close-up ROI baseplate detector.
    Returns center_abs (x,y) or None.
    """
    x, y, w, h = map(int, roi_xywh)
    H, W = full_bgr.shape[:2]
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))

    crop = full_bgr[y:y+h, x:x+w]

    # Tuned for CLOSE-UP (smaller area fractions, looser solidity/contrast)
    c_rel, ang, cnt_rel = detect_baseplate(
        crop,
        full_image=full_bgr,
        roi=(x, y, w, h),
        padding=0,
        shrink_border_px=6,
        border_margin=6,
        canny_low=35,
        canny_high=110,
        area_min_frac=0.0015,
        area_max_frac=0.25,
        aspect_min=0.6,
        aspect_max=3.5,
        solidity_min=0.45,
        extent_min=0.12,
        contrast_min=5.5,
    )

    if c_rel is None:
        return None

    return (float(x + c_rel[0]), float(y + c_rel[1]))


# ------------------------------------------------------------
# NEW: Inner border detection (edge sampling + line fitting)
# ------------------------------------------------------------

def detect_inner_border_lines_edges_local(
        image_bgr,
        roi_xywh,
        *,
        canny_low=60,
        canny_high=140,
        min_points=40,
        band_side_frac=0.22,
        band_bottom_frac=0.25,
        sample_stride=1,
):
    """
    Robust in tight notch ROI: does NOT require a closed contour.

    Samples Canny edge pixels in 3 bands:
      - left band
      - right band
      - bottom band

    Fits lines using cv2.fitLine.

    Returns dict with:
      ok, reason, roi_used, counts,
      lines (abs), points_used_abs, hull_abs
    """
    if image_bgr is None or image_bgr.size == 0:
        return {"ok": False, "reason": "empty_image"}

    x, y, w, h = map(int, roi_xywh)
    H, W = image_bgr.shape[:2]
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))

    crop = image_bgr[y:y + h, x:x + w]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, canny_low, canny_high)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    bw = max(10, int(round(band_side_frac * w)))
    bh = max(10, int(round(band_bottom_frac * h)))

    left_band = edges[:, :bw]
    right_band = edges[:, w - bw:]
    bottom_band = edges[h - bh:, :]

    def band_points(band, x_off=0, y_off=0):
        ys, xs = np.where(band > 0)
        if sample_stride > 1 and len(xs) > 0:
            idx = np.arange(0, len(xs), sample_stride)
            xs = xs[idx]
            ys = ys[idx]
        return np.column_stack([xs + x_off, ys + y_off]).astype(np.float32)

    pts_left = band_points(left_band, 0, 0)
    pts_right = band_points(right_band, w - bw, 0)
    pts_bottom = band_points(bottom_band, 0, h - bh)

    counts = {"left": int(len(pts_left)), "right": int(len(pts_right)), "bottom": int(len(pts_bottom))}
    for k, n in counts.items():
        if n < min_points:
            return {"ok": False, "reason": f"not_enough_points_{k}", "counts": counts, "roi_used": (x, y, w, h)}

    def fit_line(pts):
        vx, vy, x0, y0 = cv2.fitLine(pts.reshape(-1, 1, 2), cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        return float(vx), float(vy), float(x0), float(y0)

    line_left = fit_line(pts_left)
    line_right = fit_line(pts_right)
    line_bottom = fit_line(pts_bottom)

    def shift_line(line, dx, dy):
        vx, vy, x0, y0 = line
        return (vx, vy, x0 + dx, y0 + dy)

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
