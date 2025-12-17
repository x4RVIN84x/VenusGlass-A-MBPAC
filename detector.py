import cv2
import numpy as np

# ------------------------------------------------------------
# Load + crop by ROI (safe bounds)
# ------------------------------------------------------------
def load_and_crop(image_path, roi):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"❌ Failed to load image: {image_path}")

    x, y, w, h = roi
    H, W = image.shape[:2]
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))

    return image[y:y+h, x:x+w], image


# ------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------
def _preprocess_edges(bgr, canny_low, canny_high):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=5, sigmaColor=40, sigmaSpace=40)
    gray = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # use dynamic thresholds
    edges = cv2.Canny(blurred, canny_low, canny_high, apertureSize=3, L2gradient=True)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    return gray, edges


def _contrast_score(gray, cnt):
    mask = np.zeros(gray.shape[:2], np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, thickness=-1)
    inside_mean = cv2.mean(gray, mask=mask)[0]
    ring = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1)
    ring = cv2.subtract(ring, mask)
    if cv2.countNonZero(ring) == 0:
        return -1e9
    ring_mean = cv2.mean(gray, mask=ring)[0]
    return inside_mean - ring_mean


def _pick_best_contour(gray, contours, W, H,
                       area_min_frac, area_max_frac,
                       aspect_min, aspect_max,
                       solidity_min, extent_min,
                       border_margin, contrast_min):
    if not contours:
        return None

    win_area = float(W * H)
    cand = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        # Optional border rejection (skip if border_margin is None)
        if border_margin is not None:
            if x <= border_margin or y <= border_margin or \
               x + w >= W - border_margin or y + h >= H - border_margin:
                continue

        area = cv2.contourArea(c)
        if area < area_min_frac * win_area or area > area_max_frac * win_area:
            continue

        asp = (w / float(h)) if h else 0.0
        if not (aspect_min <= asp <= aspect_max):
            continue

        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull) or 1.0
        solidity = area / hull_area
        if solidity < solidity_min:
            continue

        extent = area / float(max(1, w * h))
        if extent < extent_min:
            continue

        contr = _contrast_score(gray, c)
        if contr < contrast_min:
            continue

        perim = cv2.arcLength(c, True)
        cand.append((contr, perim, area, c))

    if not cand:
        return None

    cand.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
    return cand[0][3]


def _detect_in_window(bgr,
                      canny_low, canny_high,
                      area_min_frac, area_max_frac,
                      aspect_min, aspect_max,
                      solidity_min, extent_min,
                      border_margin, contrast_min,
                      shrink_border_px):
    """
    Return (center_xy, angle, contour) in window coords, or (None, ..)
    """
    if bgr is None or bgr.size == 0:
        return None, None, None

    H0, W0 = bgr.shape[:2]
    sx = min(shrink_border_px, max(0, W0 // 10))
    sy = min(shrink_border_px, max(0, H0 // 10))
    x0, y0 = sx, sy
    w0, h0 = max(1, W0 - 2 * sx), max(1, H0 - 2 * sy)
    inner = bgr[y0:y0+h0, x0:x0+w0]

    gray, edges = _preprocess_edges(inner, canny_low, canny_high)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = _pick_best_contour(
        gray, contours, w0, h0,
        area_min_frac, area_max_frac,
        aspect_min, aspect_max,
        solidity_min, extent_min,
        border_margin, contrast_min
    )
    if best is None:
        return None, None, None

    (cx, cy), _, angle = cv2.minAreaRect(best)
    cx, cy = cx + x0, cy + y0  # shift back to original window
    return (int(round(cx)), int(round(cy))), float(round(angle, 2)), best + np.array([[x0, y0]])


# ------------------------------------------------------------
# Public API: robust detection with fallbacks
# ------------------------------------------------------------
def detect_baseplate(
    cropped_roi,
    *,
    full_image=None,
    roi=None,                      # [x, y, w, h]
    padding: int = 150,
    shrink_border_px: int = 10,
    canny_low: int = 50,
    canny_high: int = 120,
    area_min_frac: float = 0.02,
    area_max_frac: float = 0.60,
    aspect_min: float = 0.5,
    aspect_max: float = 2.2,
    solidity_min: float = 0.7,
    extent_min: float = 0.25,
    border_margin: int = 12,
    contrast_min: float = 12.0,
    **kwargs                        # <- swallows extra args like golden_image
):
    """
    Returns:
        (center_xy_roi_rel), angle_deg, contour_roi_rel
        - center/contour are relative to ORIGINAL calibrated ROI (roi),
          even if detection came from a padded window or full image.
        - If `roi` is None, returns coords relative to `cropped_roi`.
        - On failure: (None, None, None).
    """
    if cropped_roi is None or cropped_roi.size == 0:
        return None, None, None

    def run_on_window(img_window, to_abs=(0, 0)):
        c, a, cnt = _detect_in_window(
            img_window,
            canny_low, canny_high,
            area_min_frac, area_max_frac,
            aspect_min, aspect_max,
            solidity_min, extent_min,
            border_margin, contrast_min,
            shrink_border_px
        )
        if c is None:
            return None, None, None
        ox, oy = to_abs
        c_abs = (c[0] + ox, c[1] + oy)
        cnt_abs = cnt + np.array([[ox, oy]])
        return c_abs, a, cnt_abs

    # 1) strict in-cropped ROI (if roi provided, shift to absolute and back)
    if roi is not None:
        rx, ry, rw, rh = roi
        c_abs, ang, cnt_abs = run_on_window(cropped_roi, to_abs=(rx, ry))
        if cnt_abs is not None:
            center_rel = (c_abs[0] - rx, c_abs[1] - ry)
            cnt_rel = cnt_abs - np.array([[rx, ry]])
            return center_rel, ang, cnt_rel

    # 2) padded ROI fallback
    if full_image is not None and roi is not None:
        H, W = full_image.shape[:2]
        rx, ry, rw, rh = roi
        x_pad = max(0, rx - padding)
        y_pad = max(0, ry - padding)
        w_pad = min(rw + 2 * padding, W - x_pad)
        h_pad = min(rh + 2 * padding, H - y_pad)
        win = full_image[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
        c_abs, ang, cnt_abs = run_on_window(win, to_abs=(x_pad, y_pad))
        if cnt_abs is not None:
            center_rel = (c_abs[0] - rx, c_abs[1] - ry)
            cnt_rel = cnt_abs - np.array([[rx, ry]])
            return center_rel, ang, cnt_rel

    # 3) full image last resort
    if full_image is not None:
        c_abs, ang, cnt_abs = run_on_window(full_image, to_abs=(0, 0))
        if cnt_abs is not None:
            if roi is not None:
                rx, ry, _, _ = roi
                center_rel = (c_abs[0] - rx, c_abs[1] - ry)
                cnt_rel = cnt_abs - np.array([[rx, ry]])
            else:
                center_rel = c_abs
                cnt_rel = cnt_abs
            return center_rel, ang, cnt_rel

    return None, None, None


def detect_glass_contour(
        image,
        min_area_frac=0.3,
        max_area_frac=1.0,
        canny_low=50,
        canny_high=150,
        border_margin=None,
):
    """
    Detect the full outer contour of the glass in the image.

    Returns:
        contour (numpy array) of the largest plausible glass contour, or None if not found.
        Contour is in full-image coordinates.
    """
    if image is None or image.size == 0:
        return None

    H, W = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=5, sigmaColor=40, sigmaSpace=40)
    gray = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # dynamic Canny with accurate gradients for smooth edges
    edges = cv2.Canny(blurred, canny_low, canny_high, apertureSize=3, L2gradient=True)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    win_area = W * H
    candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        # Optional: ignore contours touching image borders (likely noise)
        if border_margin is not None:
            if x <= border_margin or y <= border_margin or x + w >= W - border_margin or y + h >= H - border_margin:
                continue

        area = cv2.contourArea(c)
        if min_area_frac * win_area <= area <= max_area_frac * win_area:
            candidates.append(c)

    if not candidates:
        return None

    # Return the largest contour by area (likely the glass)
    glass_contour = max(candidates, key=cv2.contourArea)
    return glass_contour
