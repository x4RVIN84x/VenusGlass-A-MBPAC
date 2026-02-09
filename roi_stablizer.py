# roi_stablizer.py (REWRITE)
import numpy as np
import cv2
import detector


# ----------------------------
# ROI helpers
# ----------------------------
def _roi_pad(roi, pad, W, H):
    x, y, w, h = map(int, roi)
    x2 = max(0, x - int(pad))
    y2 = max(0, y - int(pad))
    w2 = min(W - x2, w + 2 * int(pad))
    h2 = min(H - y2, h + 2 * int(pad))
    return (x2, y2, w2, h2)


def _line_intersection(l1, l2):
    """Lines are cv2.fitLine: (vx, vy, x0, y0). Returns (x, y) or None."""
    vx1, vy1, x1, y1 = map(float, l1)
    vx2, vy2, x2, y2 = map(float, l2)

    A = np.array([[vx1, -vx2],
                  [vy1, -vy2]], dtype=np.float64)
    b = np.array([x2 - x1, y2 - y1], dtype=np.float64)

    det = np.linalg.det(A)
    if abs(det) < 1e-9:
        return None

    t, _u = np.linalg.solve(A, b)
    return (float(x1 + t * vx1), float(y1 + t * vy1))


def _point_on_line_at_y(line, y_target):
    """Pick a point on fitLine at a specific y."""
    vx, vy, x0, y0 = map(float, line)
    if abs(vy) < 1e-9:
        return (float(x0), float(y0))
    t = (float(y_target) - y0) / vy
    return (float(x0 + vx * t), float(y_target))


# ----------------------------
# Anchors
# ----------------------------
def _extract_anchors_from_lines(lines_abs, y_top_ref_abs):
    """
    Anchors from left/right/bottom lines:
      - left_top at y=y_top_ref_abs
      - right_top at y=y_top_ref_abs
      - notch_bottom_left = intersection(left, bottom)
    """
    if not isinstance(lines_abs, dict):
        return None
    for k in ("left", "right", "bottom"):
        if k not in lines_abs or lines_abs[k] is None:
            return None

    left = lines_abs["left"]
    right = lines_abs["right"]
    bottom = lines_abs["bottom"]

    left_top = _point_on_line_at_y(left, y_top_ref_abs)
    right_top = _point_on_line_at_y(right, y_top_ref_abs)

    notch_bottom_left = _line_intersection(left, bottom)
    if notch_bottom_left is None:
        return None

    return {
        "left_top": left_top,
        "right_top": right_top,
        "notch_bottom_left": notch_bottom_left,
    }


def estimate_similarity_from_anchors(anchors_g, anchors_c):
    """
    Similarity-ish affine mapping golden -> current.
    Uses 3 points for estimateAffinePartial2D.
    """
    src = np.array([
        anchors_g["left_top"],
        anchors_g["right_top"],
        anchors_g["notch_bottom_left"],
    ], dtype=np.float32)

    dst = np.array([
        anchors_c["left_top"],
        anchors_c["right_top"],
        anchors_c["notch_bottom_left"],
    ], dtype=np.float32)

    M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    return M, inliers


def apply_affine_to_roi(M, roi_xywh):
    """Transform ROI corners then return tight bbox."""
    x, y, w, h = map(float, roi_xywh)
    corners = np.array([
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h],
    ], dtype=np.float32)

    ones = np.ones((4, 1), dtype=np.float32)
    pts = np.hstack([corners, ones])          # 4x3
    warped = (M.astype(np.float32) @ pts.T).T # 4x2

    x0 = float(np.min(warped[:, 0]))
    y0 = float(np.min(warped[:, 1]))
    x1 = float(np.max(warped[:, 0]))
    y1 = float(np.max(warped[:, 1]))
    return (int(round(x0)), int(round(y0)), int(round(x1 - x0)), int(round(y1 - y0)))


# ----------------------------
# Main API (stable y_top handling)
# ----------------------------
def stabilize_rois_using_saved_inner_border_lines(
    *,
    current_img,
    golden_img,
    registration_roi_golden,
    golden_inner_lines_abs,     # dict with left/right/bottom in golden ABS coords
    rois_golden,                # list of ROIs in golden ABS coords
    search_padding_px=120,
    canny_low=60,
    canny_high=140,
    # NEW: instead of pinning to golden absolute y, tie it to ROI-top with an offset
    y_top_offset_px=0.0,        # anchor-sampling y = ROI_top + offset (default: ROI_top)
):
    """
    1) Use saved golden inner lines -> golden anchors
    2) Detect current inner lines in a padded search ROI
    3) Use y_top_ref tied to current search ROI top (NOT golden absolute y):
          y_top_ref_g = reg_roi_top_g + y_top_offset
          y_top_ref_c = reg_search_top_c + y_top_offset
    4) Estimate affine partial (similarity-ish) golden->current
    5) Apply to ROIs
    """
    if current_img is None or golden_img is None:
        return None, {"ok": False, "reason": "missing_images"}

    Hc, Wc = current_img.shape[:2]

    # --- Read golden lines safely ---
    try:
        lines_g = {k: tuple(map(float, golden_inner_lines_abs[k])) for k in ("left", "right", "bottom")}
    except Exception:
        return None, {"ok": False, "reason": "bad_golden_inner_lines_abs"}

    reg_g = tuple(map(int, registration_roi_golden))
    y_top_ref_g = float(reg_g[1]) + float(y_top_offset_px)

    anchors_g = _extract_anchors_from_lines(lines_g, y_top_ref_g)
    if anchors_g is None:
        return None, {"ok": False, "reason": "golden_anchors_failed", "y_top_ref_g": y_top_ref_g}

    # --- Current detection search ROI ---
    reg_search = _roi_pad(reg_g, search_padding_px, Wc, Hc)
    y_top_ref_c = float(reg_search[1]) + float(y_top_offset_px)

    res_c = detector.detect_inner_border_lines_edges_local(
        current_img,
        reg_search,
        canny_low=canny_low,
        canny_high=canny_high,
        min_points=40,
        band_side_frac=0.22,
        band_bottom_frac=0.25,
        sample_stride=1,
    )
    if not res_c.get("ok", False):
        return None, {
            "ok": False,
            "reason": f"current_inner_failed:{res_c.get('reason')}",
            "registration_roi_golden": reg_g,
            "search_roi_current": reg_search,
            "y_top_ref_c": y_top_ref_c,
            "current_lines_debug": res_c,
        }

    lines_c = res_c.get("lines", None)
    anchors_c = _extract_anchors_from_lines(lines_c, y_top_ref_c)
    if anchors_c is None:
        return None, {
            "ok": False,
            "reason": "current_anchors_failed",
            "registration_roi_golden": reg_g,
            "search_roi_current": reg_search,
            "y_top_ref_c": y_top_ref_c,
            "current_lines_debug": res_c,
        }

    # --- Transform ---
    M, inliers = estimate_similarity_from_anchors(anchors_g, anchors_c)
    if M is None:
        return None, {
            "ok": False,
            "reason": "transform_failed",
            "registration_roi_golden": reg_g,
            "search_roi_current": reg_search,
            "y_top_ref_g": y_top_ref_g,
            "y_top_ref_c": y_top_ref_c,
            "anchors_golden": anchors_g,
            "anchors_current": anchors_c,
            "current_lines_debug": res_c,
        }

    moved = [apply_affine_to_roi(M, r) for r in rois_golden]

    info = {
        "ok": True,
        "reason": "ok",
        "M": M.tolist(),
        "inliers": None if inliers is None else inliers.astype(int).flatten().tolist(),
        "registration_roi_golden": reg_g,
        "search_roi_current": reg_search,
        "y_top_ref_g": y_top_ref_g,
        "y_top_ref_c": y_top_ref_c,
        "anchors_golden": anchors_g,
        "anchors_current": anchors_c,
        "golden_lines_abs": lines_g,
        "current_lines_debug": res_c,  # hull_abs, points_used_abs, lines, etc
    }
    return moved, info
