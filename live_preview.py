# live_preview.py (REWRITE)
import os
import time
import json
import cv2
import numpy as np

import roi_stablizer
from detector import detect_baseplate


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
CFG_PATH = r"recipes\C270TEST_207\golden_config.json"  # <-- set your recipe here

CAM_INDEX = 0
USE_DSHOW = True

# ROI stabilization cadence (saves CPU)
STAB_EVERY_N = 6

# notch-search padding (important for live drift)
SEARCH_PADDING_PX = 120

# Stability gate (frames) to avoid flicker
STABLE_NEED = 5

# Baseplate detection defaults (tune if needed)
BP_CANNY_LOW = 50
BP_CANNY_HIGH = 120
BP_CONTRAST_MIN = 6.0

# Notch detection defaults (stabilizer)
NOTCH_CANNY_LOW = 60
NOTCH_CANNY_HIGH = 140

# Camera request (C270 supports 1280x720 on your machine)
REQ_W, REQ_H, REQ_FPS = 1280, 720, 30

WIN = "LIVE QC PREVIEW"


def _tolerances_px_from_cfg(cfg: dict):
    """Resolve physical recipe limits into pixel limits for this legacy preview."""
    if not isinstance(cfg, dict):
        return None

    raw_mm = cfg.get("tolerance_mm")
    if isinstance(raw_mm, dict):
        try:
            scale = cfg.get("px_per_mm", cfg.get("baseplate_px_per_mm"))
            if scale is None and isinstance(cfg.get("baseplate_scale"), dict):
                scale = cfg["baseplate_scale"].get("px_per_mm")
            scale = float(scale)
            if scale <= 0:
                return None
            return {
                "x": float(raw_mm["x"]) * scale,
                "y": float(raw_mm["y"]) * scale,
                "angle": float(raw_mm["angle"]),
            }
        except (TypeError, ValueError, KeyError):
            return None

    raw_px = cfg.get("tolerance_px")
    if not isinstance(raw_px, dict):
        return None
    try:
        values = {key: float(raw_px[key]) for key in ("x", "y", "angle")}
    except (TypeError, ValueError, KeyError):
        return None
    return values if all(value > 0 for value in values.values()) else None


# ---------------------------------------------------------------------
# Draw helpers
# ---------------------------------------------------------------------

import cv2
import numpy as np

def _mean_luma(img_bgr, x0, y0, x1, y1):
    """Mean luminance of a clipped ROI (0-255)."""
    H, W = img_bgr.shape[:2]
    x0 = max(0, min(W - 1, int(x0)))
    y0 = max(0, min(H - 1, int(y0)))
    x1 = max(0, min(W, int(x1)))
    y1 = max(0, min(H, int(y1)))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    roi = img_bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))

def put_cctv_word_adaptive(
    img_bgr,
    word: str,
    org,                 # bottom-left of word
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.6,
    thickness=2,
    pad=4,
    threshold=140.0,     # higher -> more likely to choose black text
):
    """
    Draw ONE word with color chosen by background under that word:
      bright bg -> black text
      dark bg   -> white text
    Also draws opposite-color outline for readability.
    """
    x, y = org
    (tw, th), baseline = cv2.getTextSize(word, font, font_scale, thickness)

    # bounding box around the word area (approx)
    x0 = x - pad
    y0 = y - th - pad
    x1 = x + tw + pad
    y1 = y + baseline + pad

    mean_l = _mean_luma(img_bgr, x0, y0, x1, y1)

    # decide foreground based on local background brightness
    if mean_l >= threshold:
        fg = (0, 0, 0)         # black text on bright background
        outline = (255, 255, 255)
    else:
        fg = (255, 255, 255)   # white text on dark background
        outline = (0, 0, 0)

    # outline then foreground (CCTV vibe)
    cv2.putText(img_bgr, word, (x, y), font, font_scale, outline, thickness + 3, cv2.LINE_AA)
    cv2.putText(img_bgr, word, (x, y), font, font_scale, fg, thickness, cv2.LINE_AA)

    return tw  # width so caller can place next word

def put_cctv_text_adaptive_words(
    img_bgr,
    text: str,
    org,                 # bottom-left start
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.6,
    thickness=2,
    word_gap=10,
    threshold=140.0,
):
    """
    Draw text word-by-word with adaptive color per word (CCTV style).
    """
    x, y = org
    for w in text.split(" "):
        if w == "":
            continue
        tw = put_cctv_word_adaptive(
            img_bgr, w, (x, y),
            font=font, font_scale=font_scale, thickness=thickness,
            threshold=threshold
        )
        x += tw + word_gap


def draw_fitline(vis, line, color=(255, 0, 0), thickness=2):
    """Draw cv2.fitLine (vx,vy,x0,y0) across the full image."""
    if line is None:
        return
    vx, vy, x0, y0 = map(float, line)
    p1 = (int(round(x0 - vx * 5000)), int(round(y0 - vy * 5000)))
    p2 = (int(round(x0 + vx * 5000)), int(round(y0 + vy * 5000)))
    cv2.line(vis, p1, p2, color, thickness, lineType=cv2.LINE_AA)


def draw_points(vis, pts, color=(0, 255, 255), radius=1, step=10):
    """Draw a lot of points without killing FPS."""
    if pts is None:
        return
    pts = np.asarray(pts)
    if pts.ndim == 3:
        pts = pts.reshape(-1, 2)
    if len(pts) == 0:
        return

    step = max(1, int(step))
    for i in range(0, len(pts), step):
        x, y = int(pts[i][0]), int(pts[i][1])
        cv2.circle(vis, (x, y), radius, color, -1, lineType=cv2.LINE_AA)


def draw_stab_debug(vis, stab_info):
    """Draw notch lines, hull, points, and anchors used for ROI stabilization."""
    if not stab_info:
        return

    ok = stab_info.get("ok", False)

    # Always show the search ROI even on fail (helps debugging)
    if "search_roi_current" in stab_info:
        x, y, w, h = stab_info["search_roi_current"]
        col = (120, 120, 120) if ok else (0, 0, 255)
        cv2.rectangle(vis, (x, y), (x + w, y + h), col, 1, lineType=cv2.LINE_AA)

    dbg = stab_info.get("current_lines_debug", {}) or {}
    lines = dbg.get("lines", None)
    pts_used = dbg.get("points_used_abs", None)
    hull = dbg.get("hull_abs", None)

    # Lines
    if isinstance(lines, dict):
        draw_fitline(vis, lines.get("left"), color=(255, 0, 0), thickness=2)
        draw_fitline(vis, lines.get("right"), color=(255, 0, 0), thickness=2)
        draw_fitline(vis, lines.get("bottom"), color=(255, 0, 0), thickness=2)

    # Hull
    if hull is not None:
        try:
            cv2.polylines(vis, [hull], True, (0, 255, 0), 2, lineType=cv2.LINE_AA)
        except Exception:
            pass

    # Points used (downsampled)
    if isinstance(pts_used, dict):
        draw_points(vis, pts_used.get("left"), color=(0, 255, 255), radius=1, step=12)
        draw_points(vis, pts_used.get("right"), color=(0, 255, 255), radius=1, step=12)
        draw_points(vis, pts_used.get("bottom"), color=(0, 255, 255), radius=1, step=12)

    # Anchors (only meaningful if ok)
    anchors = stab_info.get("anchors_current", None)
    if isinstance(anchors, dict):
        for k, pt in anchors.items():
            ax, ay = int(round(pt[0])), int(round(pt[1]))
            cv2.circle(vis, (ax, ay), 5, (0, 0, 255), -1, lineType=cv2.LINE_AA)
            cv2.putText(vis, k, (ax + 6, ay - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    # Show y_top_ref (anchor-sampling y)
    if "y_top_ref_c" in stab_info:
        y = int(round(stab_info["y_top_ref_c"]))
        cv2.line(vis, (0, y), (vis.shape[1] - 1, y), (255, 255, 255), 1, lineType=cv2.LINE_AA)

    # Status text (top-left)
    if not ok:
        reason = stab_info.get("reason", "stab_fail")
        cv2.putText(vis, f"STAB FAIL: {reason}", (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)


def clamp_roi(roi, W, H):
    x, y, w, h = map(int, roi)
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return (x, y, w, h)


def safe_crop(img, roi):
    roi = clamp_roi(roi, img.shape[1], img.shape[0])
    x, y, w, h = roi
    return img[y:y + h, x:x + w].copy(), roi


def draw_bp_overlay(vis, roi, center_rel, contour_rel):
    x, y, w, h = roi
    cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 0), 2, lineType=cv2.LINE_AA)

    if contour_rel is not None and len(contour_rel) >= 3:
        cnt_abs = contour_rel + np.array([[x, y]], dtype=np.int32)
        cv2.drawContours(vis, [cnt_abs], -1, (0, 255, 0), 2)

    if center_rel is not None:
        cx = int(round(x + center_rel[0]))
        cy = int(round(y + center_rel[1]))
        cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1, lineType=cv2.LINE_AA)


def draw_status_pill(vis, text, state="FAIL"):
    if state == "PASS":
        color = (0, 200, 0)
    elif state == "TRACK":
        color = (0, 200, 200)
    elif state == "SEARCH":
        color = (180, 180, 180)
    else:
        color = (0, 0, 255)

    cv2.rectangle(vis, (20, 20), (680, 100), (20, 20, 20), -1)
    cv2.putText(vis, text, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)



# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    # Load config
    with open(CFG_PATH, "r") as f:
        cfg = json.load(f)

    roi_cfg = tuple(cfg["roi"])
    golden_center = tuple(cfg["expected_center"])
    golden_angle = float(cfg.get("expected_angle", 0.0))
    tol = _tolerances_px_from_cfg(cfg)
    if tol is None:
        raise ValueError("Recipe is missing valid tolerances; set X/Y in mm and angle in Calibration")

    golden_img_path = cfg["golden_image_path"]
    golden_img = cv2.imread(golden_img_path)
    if golden_img is None:
        raise RuntimeError(f"Could not load golden image: {golden_img_path}")
    Hg, Wg = golden_img.shape[:2]

    # Stabilization inputs
    registration_roi_golden = tuple(cfg.get("registration_roi", cfg["roi"]))
    golden_lines = cfg.get("inner_border_lines", None)

    # Camera
    api = cv2.CAP_DSHOW if USE_DSHOW else 0
    cap = cv2.VideoCapture(CAM_INDEX, api)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(REQ_W))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(REQ_H))
    cap.set(cv2.CAP_PROP_FPS, int(REQ_FPS))

    # Runtime state
    roi_live = roi_cfg
    stab_info = None

    frame_i = 0
    t0 = time.time()
    fps = 0.0

    stable_have = 0
    last_measure = None  # (dx, dy, dtheta, ok_all)

    show_stab = True
    show_bp = True
    paused = False

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    # Where to save snapshots
    out_dir = os.path.join("output", "live_preview_snaps")
    os.makedirs(out_dir, exist_ok=True)

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break

            # Match golden geometry (only if needed)
            if frame.shape[:2] != (Hg, Wg):
                frame = cv2.resize(frame, (Wg, Hg), interpolation=cv2.INTER_LINEAR)

            H, W = frame.shape[:2]

            # ROI stabilization (not every frame)
            if golden_lines and (frame_i % STAB_EVERY_N == 0):
                moved, info = roi_stablizer.stabilize_rois_using_saved_inner_border_lines(
                    current_img=frame,
                    golden_img=golden_img,
                    registration_roi_golden=registration_roi_golden,
                    golden_inner_lines_abs=golden_lines,
                    rois_golden=[roi_cfg],
                    search_padding_px=SEARCH_PADDING_PX,
                    canny_low=NOTCH_CANNY_LOW,
                    canny_high=NOTCH_CANNY_HIGH,
                )

                stab_info = info  # <-- THIS is what you were missing

                if moved is not None and info and info.get("ok"):
                    roi_live = clamp_roi(moved[0], W, H)
                else:
                    roi_live = clamp_roi(roi_cfg, W, H)

            crop, roi_live = safe_crop(frame, roi_live)

            # Baseplate detect
            ret = detect_baseplate(
                crop,
                full_image_bgr=frame,
                roi_xywh_abs=roi_live,
                padding=150,
                shrink_border_px=10,
                canny_low=BP_CANNY_LOW,
                canny_high=BP_CANNY_HIGH,
                area_min_frac=0.005,
                contrast_min=BP_CONTRAST_MIN,
                border_margin=12,
                return_debug=False,
            )

            center_rel, angle, contour_rel = ret[:3]

            # Compare + stability gate
            state = "FAIL"
            status_text = "SEARCHING…"

            if center_rel is None or angle is None:
                stable_have = max(0, stable_have - 1)
                last_measure = None
                state = "SEARCH"
                status_text = "SEARCHING…"
            else:
                dx = abs(float(center_rel[0]) - float(golden_center[0]))
                dy = abs(float(center_rel[1]) - float(golden_center[1]))
                dtheta = abs(float(angle) - float(golden_angle))

                ok_all = (dx <= tol["x"]) and (dy <= tol["y"]) and (dtheta <= tol["angle"])
                last_measure = (dx, dy, dtheta, ok_all)

                if ok_all:
                    stable_have = min(STABLE_NEED, stable_have + 1)
                else:
                    # drift down slowly instead of instantly zeroing -> smoother UX
                    stable_have = max(0, stable_have - 1)
                    stable_have = max(0, stable_have - 1)

                if stable_have >= STABLE_NEED and ok_all:
                    state = "PASS"
                    status_text = f"PASS  dx={dx:.1f} dy={dy:.1f} dtetha={dtheta:.1f}"
                elif center_rel is not None:
                    state = "TRACK"
                    status_text = f"TRACK {stable_have}/{STABLE_NEED}  dx={dx:.1f} dy={dy:.1f} dtetha={dtheta:.1f}"
                else:
                    state = "SEARCH"
                    status_text = "SEARCHING…"

            # Compose visualization
            vis = frame.copy()

            # Draw stabilizer debug first (so BP overlay sits on top)
            if show_stab:
                draw_stab_debug(vis, stab_info)

            if show_bp:
                draw_bp_overlay(vis, roi_live, center_rel, contour_rel)

            draw_status_pill(vis, status_text, state=state)

            # FPS estimate
            frame_i += 1
            if frame_i % 15 == 0:
                dt = time.time() - t0
                fps = 15.0 / max(1e-6, dt)
                t0 = time.time()

            bottom_text = "FPS: {:.1f} | keys: [d]stab [b]bp [p]pause [s]snap [r]reset [q]quit".format(fps)

            put_cctv_text_adaptive_words(
                vis,
                bottom_text,
                (20, vis.shape[0] - 20),
                font_scale=0.6,
                thickness=2,
                threshold=140.0,  # tweak if needed
            )

            cv2.imshow(WIN, vis)

        # Key handling (works even when paused)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord("q")):
            break
        elif key == ord("d"):
            show_stab = not show_stab
        elif key == ord("b"):
            show_bp = not show_bp
        elif key == ord("p"):
            paused = not paused
        elif key == ord("r"):
            roi_live = roi_cfg
            stable_have = 0
            stab_info = None
        elif key == ord("s"):
            # Snapshot current display
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(out_dir, f"snap_{ts}.png")
            # grab last shown frame from window by re-reading via imshow buffer is not possible;
            # instead write the last 'vis' if it exists in scope
            try:
                cv2.imwrite(out_path, vis)
                print(f"[snap] saved: {out_path}")
            except Exception as e:
                print(f"[snap] failed: {e}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
