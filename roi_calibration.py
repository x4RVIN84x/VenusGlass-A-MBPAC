import cv2
import json
import os
import numpy as np

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
CONFIG_PATH = r"E:\ARVIN\A-MBPAC\reference_json\207_golden_config.json"  # << change if needed

# detection knobs (aligned with detector.py)
SHRINK_BORDER_PX = 10
PADDING_PX = 30
BORDER_MARGIN = 12
CANNY_LOW = 50
CANNY_HIGH = 120
AREA_MIN_FRAC = 0.02
AREA_MAX_FRAC = 0.60
ASPECT_MIN   = 0.5
ASPECT_MAX   = 2.2
SOLIDITY_MIN = 0.7
EXTENT_MIN   = 0.25
CONTRAST_MIN = 12.0

# ------------------------------------------------------------
# Mouse globals
# ------------------------------------------------------------
roi_start = None
roi_end = None
drawing = False
roi_done = False

center_abs = None
chosen_cnt = None
chosen_angle = 0.0
status_text = ""


def draw_roi(event, x, y, flags, param):
    global roi_start, roi_end, drawing, roi_done
    if event == cv2.EVENT_LBUTTONDOWN and not roi_done:
        roi_start = (x, y)
        roi_end = (x, y)
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        roi_end = (x, y)
    elif event == cv2.EVENT_LBUTTONUP and drawing:
        roi_end = (x, y)
        drawing = False
        roi_done = True


def _contrast_score(gray, cnt):
    mask = np.zeros(gray.shape[:2], np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, thickness=-1)
    inside_mean = cv2.mean(gray, mask=mask)[0]
    ring = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1)
    ring = cv2.subtract(ring, mask)
    if cv2.countNonZero(ring) == 0:
        return -1e9
    ring_mean = cv2.mean(gray, mask=ring)[0]
    return inside_mean - ring_mean  # metal brighter than backprint


def _filter_and_score_contours(gray, W, H, contours):
    roi_area = float(W * H)
    keep = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # reject border-huggers
        if x <= BORDER_MARGIN or y <= BORDER_MARGIN or \
           x + w >= W - BORDER_MARGIN or y + h >= H - BORDER_MARGIN:
            continue

        area = cv2.contourArea(cnt)
        if area < AREA_MIN_FRAC * roi_area or area > AREA_MAX_FRAC * roi_area:
            continue

        aspect = (w / float(h)) if h else 0.0
        if not (ASPECT_MIN <= aspect <= ASPECT_MAX):
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull) or 1.0
        solidity = area / hull_area
        if solidity < SOLIDITY_MIN:
            continue

        extent = area / float(w * h)
        if extent < EXTENT_MIN:
            continue

        contr = _contrast_score(gray, cnt)
        if contr < CONTRAST_MIN:
            continue

        perim = cv2.arcLength(cnt, True)
        keep.append({"cnt": cnt, "area": area, "perim": perim, "contrast": contr})

    if not keep:
        return None

    keep.sort(key=lambda s: (s["contrast"], s["perim"], s["area"]), reverse=True)
    return keep[0]["cnt"]


def _detect_in_roi(image, roi):
    """
    image: full image (BGR)
    roi: [x, y, w, h]
    returns: (center_abs_xy, angle_deg, contour_abs, ok:bool)
    """
    x, y, w, h = roi
    crop = image[y:y+h, x:x+w]
    if crop.size == 0:
        return None, None, None, False

    H0, W0 = crop.shape[:2]
    sx = min(SHRINK_BORDER_PX, max(0, W0 // 10))
    sy = min(SHRINK_BORDER_PX, max(0, H0 // 10))
    x_in, y_in = sx, sy
    w_in, h_in = max(1, W0 - 2 * sx), max(1, H0 - 2 * sy)
    inner = crop[y_in:y_in+h_in, x_in:x_in+w_in]

    gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=5, sigmaColor=40, sigmaSpace=40)
    gray = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None, False

    best = _filter_and_score_contours(gray, w_in, h_in, contours)
    if best is None:
        return None, None, None, False

    rect = cv2.minAreaRect(best)
    (cx, cy), _, angle = rect

    cx_abs = int(x + x_in + cx)
    cy_abs = int(y + y_in + cy)
    best_off = best + np.array([[x + x_in, y + y_in]])

    return (cx_abs, cy_abs), float(round(angle, 2)), best_off, True


def main():
    global center_abs, chosen_cnt, chosen_angle, status_text, roi_start, roi_end, roi_done

    # load config & image
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)
    img_path = cfg.get("golden_image_path")
    if not img_path:
        print("❌ 'golden_image_path' missing in config.")
        return
    img = cv2.imread(img_path)
    if img is None:
        print("❌ Image not found. Check path:", img_path)
        return

    clone = img.copy()
    cv2.namedWindow("Calibrator")
    cv2.setMouseCallback("Calibrator", draw_roi)

    while True:
        temp = clone.copy()

        if roi_start and roi_end:
            cv2.rectangle(temp, roi_start, roi_end, (0, 255, 0), 2)

        if roi_done and center_abs is None:
            x0, y0 = roi_start
            x1, y1 = roi_end
            roi_x = min(x0, x1)
            roi_y = min(y0, y1)
            roi_w = abs(x1 - x0)
            roi_h = abs(y1 - y0)
            roi_box = [roi_x, roi_y, roi_w, roi_h]

            # main pass
            c_abs, ang, cnt_abs, ok = _detect_in_roi(img, roi_box)
            if ok:
                status_text = "Main ROI used"
                center_abs, chosen_angle, chosen_cnt = c_abs, ang, cnt_abs
            else:
                # padded pass
                H, W = img.shape[:2]
                x_pad = max(0, roi_x - PADDING_PX)
                y_pad = max(0, roi_y - PADDING_PX)
                w_pad = min(roi_w + 2 * PADDING_PX, W - x_pad)
                h_pad = min(roi_h + 2 * PADDING_PX, H - y_pad)
                padded_box = [x_pad, y_pad, w_pad, h_pad]

                c_abs, ang, cnt_abs, ok2 = _detect_in_roi(img, padded_box)
                if ok2:
                    status_text = f"Padded ROI used (+{PADDING_PX}px)"
                    center_abs, chosen_angle, chosen_cnt = c_abs, ang, cnt_abs
                else:
                    status_text = "Fallback: using ROI center"
                    center_abs = (roi_x + roi_w // 2, roi_y + roi_h // 2)
                    chosen_angle = 0.0
                    chosen_cnt = None

        if center_abs is not None:
            cv2.circle(temp, center_abs, 5, (0, 0, 255), -1)
            cv2.line(temp, (center_abs[0], 0), (center_abs[0], temp.shape[0]), (0, 255, 255), 1)
            cv2.line(temp, (0, center_abs[1]), (temp.shape[1], center_abs[1]), (0, 255, 255), 1)

        if chosen_cnt is not None:
            cv2.drawContours(temp, [chosen_cnt], -1, (0, 255, 0), 2)

        cv2.putText(temp, "Press Q to save and quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 255, 120), 2)
        if status_text:
            cv2.putText(temp, status_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255) if "Fallback" in status_text else (0, 255, 0), 2)

        cv2.imshow("Calibrator", temp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

    # save back to JSON (ROI + ROI-relative center + angle)
    if roi_start and roi_end and center_abs is not None:
        x0, y0 = roi_start
        x1, y1 = roi_end
        roi_x = min(x0, x1)
        roi_y = min(y0, y1)
        roi_w = abs(x1 - x0)
        roi_h = abs(y1 - y0)

        cx_rel = int(center_abs[0] - roi_x)
        cy_rel = int(center_abs[1] - roi_y)

        cfg["roi"] = [roi_x, roi_y, roi_w, roi_h]
        cfg["expected_center"] = [cx_rel, cy_rel]
        cfg["expected_angle"] = float(chosen_angle)
        cfg.setdefault("tolerance_px", {"x": 10, "y": 10, "angle": 5})
        # optional scale: cfg.setdefault("px_to_mm", {"uniform": 0.10})

        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w") as f:
            json.dump(cfg, f, indent=4)

        print("✅ Saved to", CONFIG_PATH)
        print(f"   roi={cfg['roi']}")
        print(f"   expected_center (ROI-relative)={cfg['expected_center']}")
        print(f"   expected_angle={cfg['expected_angle']}°")
        print(f"   status={status_text}")
    else:
        print("⚠️ Incomplete calibration. Nothing saved.")


if __name__ == "__main__":
    main()
