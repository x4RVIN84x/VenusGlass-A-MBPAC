# main.py
import json
import os
import cv2
import numpy as np

from detector import detect_baseplate
from compare_to_golden import compare_to_golden
import roi_stablizer


# =========================
# EDITABLE FLAGS
# =========================
AUTO_MODE = False
SHOW_OVERLAY = True
USE_ROI_STAB = True

CFG_PATH = r"C:\Users\m.afrazeh\PycharmProjects\VenusGlass-A-MBPAC\recepies\C270TEST_207\golden_config.json"
INCOMING_DIR = r"D:\Arvin\A-MBPAC\output"
OUTPUT_ROOT = r"E:\ARVIN\A-MBPAC\output"

MANUAL_TEST_IMAGE_PATH = r"D:\Arvin\A-MBPAC\images\V1.1_test\WhatsApp Image 2025-12-18 at 10.57.44 AM (1).jpeg"

# Stabilization search padding (THIS is the “padding option” you asked for)
SEARCH_PADDING_PX = 0
# Viewer controls
VIEW_MAX_W = 1500
ZOOM_MIN = 1.0
ZOOM_MAX = 8.0
ZOOM_STEP = 1.25
PAN_STEP_FRAC = 0.12


# -------------------------
# Utility
# -------------------------
def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _clamp_roi(roi, W, H):
    """Force ROI inside image."""
    if roi is None:
        return None
    x, y, w, h = map(int, roi)
    x = int(_clamp(x, 0, W - 1))
    y = int(_clamp(y, 0, H - 1))
    w = int(_clamp(w, 1, W - x))
    h = int(_clamp(h, 1, H - y))
    return (x, y, w, h)


def _safe_crop(full_img, roi):
    roi = _clamp_roi(roi, full_img.shape[1], full_img.shape[0])
    x, y, w, h = roi
    return full_img[y:y + h, x:x + w].copy(), roi


def _draw_roi(vis, roi, color, thickness=2):
    if roi is None:
        return
    x, y, w, h = map(int, roi)
    cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)


def _draw_contour_abs(vis, contour_abs, color=(0, 255, 0), thickness=2):
    if contour_abs is None:
        return
    c = np.array(contour_abs, dtype=np.int32)
    if c.ndim == 2:
        c = c.reshape((-1, 1, 2))
    if len(c) >= 3:
        cv2.drawContours(vis, [c], -1, color, thickness)


def _line_endpoints_in_image(line, W, H):
    vx, vy, x0, y0 = map(float, line)
    pts = []

    def add_if_in(xx, yy):
        if 0 <= xx <= W - 1 and 0 <= yy <= H - 1:
            pts.append((int(round(xx)), int(round(yy))))

    if abs(vx) > 1e-9:
        t = (0 - x0) / vx
        add_if_in(x0 + vx * t, y0 + vy * t)
        t = ((W - 1) - x0) / vx
        add_if_in(x0 + vx * t, y0 + vy * t)

    if abs(vy) > 1e-9:
        t = (0 - y0) / vy
        add_if_in(x0 + vx * t, y0 + vy * t)
        t = ((H - 1) - y0) / vy
        add_if_in(x0 + vx * t, y0 + vy * t)

    if len(pts) < 2:
        return None

    best = (pts[0], pts[1])
    best_d = -1
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dx = pts[i][0] - pts[j][0]
            dy = pts[i][1] - pts[j][1]
            d = dx * dx + dy * dy
            if d > best_d:
                best_d = d
                best = (pts[i], pts[j])
    return best


def _text_params_for_zoom(zoom):
    """
    Reduce text size when zooming in so labels don't occupy the whole screen.
    """
    # zoom=1 -> scale ~0.85, zoom=8 -> scale ~0.35
    scale = 0.85 / (zoom ** 0.45)
    thick = max(1, int(round(2 / (zoom ** 0.35))))
    return float(scale), int(thick)


def _put_label(vis, text, xy, color, zoom):
    scale, thick = _text_params_for_zoom(zoom)
    x, y = int(xy[0]), int(xy[1])
    cv2.putText(vis, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def _get_view_rect(img_shape, zoom, center_xy):
    H, W = img_shape[:2]
    zoom = float(_clamp(zoom, ZOOM_MIN, ZOOM_MAX))

    view_w = int(round(W / zoom))
    view_h = int(round(H / zoom))
    view_w = max(120, min(W, view_w))
    view_h = max(120, min(H, view_h))

    cx, cy = center_xy
    cx = float(_clamp(cx, 0, W - 1))
    cy = float(_clamp(cy, 0, H - 1))

    x0 = int(round(cx - view_w / 2))
    y0 = int(round(cy - view_h / 2))
    x0 = int(_clamp(x0, 0, W - view_w))
    y0 = int(_clamp(y0, 0, H - view_h))

    return x0, y0, view_w, view_h


def _make_display_size(img):
    H, W = img.shape[:2]
    disp_w = min(VIEW_MAX_W, W)
    disp_h = int(round(disp_w * (H / float(W))))
    return disp_w, disp_h


# -------------------------
# Debug drawing (calibration style)
# -------------------------
def _draw_notch_debug(vis, inner_res, zoom, draw_points=True, draw_lines=True):
    """
    Mirror calibration module visuals:
      - hull (green)
      - points_used_abs (green dots)
      - fit lines (left/right blue, bottom cyan)
    """
    if not inner_res or not inner_res.get("ok"):
        return

    H, W = vis.shape[:2]

    hull = inner_res.get("hull_abs")
    if hull is not None:
        cv2.drawContours(vis, [np.array(hull, dtype=np.int32)], -1, (0, 255, 0), 2)

    if draw_points:
        pts_abs = inner_res.get("points_used_abs", {})
        for k in ["left", "right", "bottom"]:
            arr = pts_abs.get(k)
            if arr is None:
                continue
            step = max(1, len(arr) // 350)
            for px, py in arr[::step]:
                cv2.circle(vis, (int(px), int(py)), 1, (0, 255, 0), -1)

    if draw_lines:
        lines = inner_res.get("lines", {})
        colors = {"left": (255, 0, 0), "right": (255, 0, 0), "bottom": (255, 255, 0)}
        for k in ["left", "right", "bottom"]:
            if k not in lines:
                continue
            seg = _line_endpoints_in_image(lines[k], W, H)
            if seg:
                cv2.line(vis, seg[0], seg[1], colors[k], 2)


def _draw_anchors(vis, anchors, zoom, color, prefix, show_text=True):
    if not isinstance(anchors, dict):
        return
    for k, p in anchors.items():
        try:
            x, y = int(round(p[0])), int(round(p[1]))
        except Exception:
            continue
        cv2.circle(vis, (x, y), 6, color, 2)
        if show_text:
            _put_label(vis, f"{prefix}:{k}", (x + 10, y - 8), color, zoom)


# -------------------------
# Viewer (zoom/pan + toggles)
# -------------------------
def _interactive_debug_viewer(full_img, render_fn):
    """
    render_fn(state) -> returns an image (full-res) with overlays drawn.
    Then we view it with zoom/pan like your calibration UI.
    """
    state = {
        "zoom": 1.0,
        "center": (full_img.shape[1] / 2.0, full_img.shape[0] / 2.0),

        # layer toggles
        "show_rois": True,
        "show_notch": True,
        "show_notch_points": True,
        "show_notch_lines": True,
        "show_anchors": True,
        "show_labels": True,
        "show_baseplate": True,
    }

    win = "QC (debug)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    disp_w, disp_h = _make_display_size(full_img)
    disp_size = (disp_w, disp_h)

    while True:
        overlay_full = render_fn(state)
        vx, vy, vw, vh = _get_view_rect(overlay_full.shape, state["zoom"], state["center"])
        view = overlay_full[vy:vy + vh, vx:vx + vw]
        view_disp = cv2.resize(view, disp_size, interpolation=cv2.INTER_LINEAR)

        footer = (
            "WASD pan | +/- zoom | 0 reset | "
            "R rois | N notch | P notchPts | L notchLines | A anchors | T labels | B baseplate | Esc quit"
        )
        cv2.putText(view_disp, footer, (18, view_disp.shape[0] - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3)
        cv2.putText(view_disp, footer, (18, view_disp.shape[0] - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 2)

        cv2.imshow(win, view_disp)
        key = cv2.waitKey(20) & 0xFF

        if key in (27, ord('q')):  # esc / q
            break

        # zoom
        if key in (ord('+'), ord('=')):
            state["zoom"] = float(_clamp(state["zoom"] * ZOOM_STEP, ZOOM_MIN, ZOOM_MAX))
        elif key in (ord('-'), ord('_')):
            state["zoom"] = float(_clamp(state["zoom"] / ZOOM_STEP, ZOOM_MIN, ZOOM_MAX))
        elif key == ord('0'):
            state["zoom"] = 1.0
            state["center"] = (full_img.shape[1] / 2.0, full_img.shape[0] / 2.0)

        # pan
        elif key in (ord('w'), ord('a'), ord('s'), ord('d')):
            vx, vy, vw, vh = _get_view_rect(full_img.shape, state["zoom"], state["center"])
            cx, cy = state["center"]
            if key == ord('w'):
                cy -= PAN_STEP_FRAC * vh
            elif key == ord('s'):
                cy += PAN_STEP_FRAC * vh
            elif key == ord('a'):
                cx -= PAN_STEP_FRAC * vw
            elif key == ord('d'):
                cx += PAN_STEP_FRAC * vw
            state["center"] = (_clamp(cx, 0, full_img.shape[1] - 1), _clamp(cy, 0, full_img.shape[0] - 1))

        # toggles
        elif key == ord('r'):
            state["show_rois"] = not state["show_rois"]
        elif key == ord('n'):
            state["show_notch"] = not state["show_notch"]
        elif key == ord('p'):
            state["show_notch_points"] = not state["show_notch_points"]
        elif key == ord('l'):
            state["show_notch_lines"] = not state["show_notch_lines"]
        elif key == ord('a'):
            state["show_anchors"] = not state["show_anchors"]
        elif key == ord('t'):
            state["show_labels"] = not state["show_labels"]
        elif key == ord('b'):
            state["show_baseplate"] = not state["show_baseplate"]

    cv2.destroyWindow(win)


# -------------------------
# Main processing
# -------------------------
def process_one(test_image_path: str, cfg: dict):
    if "roi" not in cfg or "expected_center" not in cfg:
        print("❌ Config missing keys: roi, expected_center")
        return

    roi_cfg = tuple(cfg["roi"])
    golden_center_roi = tuple(cfg["expected_center"])  # ROI-relative (to baseplate ROI used in calib)
    golden_angle = float(cfg.get("expected_angle", 0.0))
    tolerances = cfg.get("tolerance_px", {"x": 10, "y": 10, "angle": 5})

    golden_img_path = cfg.get("golden_image_path")
    if not golden_img_path:
        print("❌ Config missing golden_image_path")
        return

    golden_img = cv2.imread(golden_img_path)
    if golden_img is None:
        print("❌ Could not load golden image:", golden_img_path)
        return

    full_test_img = cv2.imread(test_image_path)
    if full_test_img is None:
        print("❌ Failed to load test image:", test_image_path)
        return

    # Keep geometry consistent: resize test -> golden size
    Hg, Wg = golden_img.shape[:2]
    Ht, Wt = full_test_img.shape[:2]
    if (Ht, Wt) != (Hg, Wg):
        print(f"⚠️ Resizing test image {Wt}x{Ht} -> golden {Wg}x{Hg}")
        full_test_img = cv2.resize(full_test_img, (Wg, Hg), interpolation=cv2.INTER_LINEAR)

    H, W = full_test_img.shape[:2]

    roi_for_detection = _clamp_roi(roi_cfg, W, H)
    stab_info = None

    # Stabilize ROI (notch lines)
    if USE_ROI_STAB and "inner_border_lines" in cfg:
        registration_roi_golden = tuple(cfg.get("registration_roi", cfg["roi"]))
        moved, info = roi_stablizer.stabilize_rois_using_saved_inner_border_lines(
            current_img=full_test_img,
            golden_img=golden_img,
            registration_roi_golden=registration_roi_golden,
            golden_inner_lines_abs=cfg["inner_border_lines"],
            rois_golden=[roi_cfg],
            search_padding_px=SEARCH_PADDING_PX,
            canny_low=60,
            canny_high=140,
        )
        stab_info = info
        if moved is not None and info and info.get("ok"):
            roi_for_detection = _clamp_roi(moved[0], W, H)
            print("🧭 ROI stabilized:", roi_for_detection)
        else:
            print("⚠️ ROI stabilization failed:", info)

    # Detect baseplate in moved ROI
    # Detect baseplate in moved ROI
    cropped_test, roi_for_detection = _safe_crop(full_test_img, roi_for_detection)

    ret = detect_baseplate(
        cropped_test,
        full_image_bgr=full_test_img,
        roi_xywh_abs=roi_for_detection,
        padding=150,
        shrink_border_px=10,
        canny_low=50,
        canny_high=120,
        area_min_frac=0.005,
        contrast_min=6.0,
        border_margin=12,
        return_debug=False,
    )

    test_center_rel, test_angle, test_cnt_rel = ret[:3]
    dbg = ret[3] if len(ret) > 3 else None

    test_center_rel, test_angle, test_cnt_rel = ret[:3]
    dbg = ret[3] if len(ret) > 3 else None

    if test_center_rel is None:
        print("❌ Baseplate not detected.")
        return

    abs_center = (
        roi_for_detection[0] + float(test_center_rel[0]),
        roi_for_detection[1] + float(test_center_rel[1]),
    )

    # Compare in ROI-relative coords (moved ROI frame)
    test_center_rel_to_roi = (float(test_center_rel[0]), float(test_center_rel[1]))

    result = compare_to_golden(
        detected_center=test_center_rel_to_roi,
        detected_angle=float(test_angle),
        golden_center=golden_center_roi,
        golden_angle=golden_angle,
        tolerances=tolerances,
    )

    print("\n=== QC RESULT ===")
    print("Test image:", test_image_path)
    print(f"ΔX = {result['dx']:.2f} px   PASS={result['pass_x']}")
    print(f"ΔY = {result['dy']:.2f} px   PASS={result['pass_y']}")
    print(f"Δθ = {result['dtheta']:.2f}°  PASS={result['pass_angle']}")
    print("✅ Overall:", "PASS" if result["overall"] else "FAIL")

    if not SHOW_OVERLAY:
        return

    # Precompute expected abs (in moved ROI frame)
    expected_abs = (
        roi_for_detection[0] + float(golden_center_roi[0]),
        roi_for_detection[1] + float(golden_center_roi[1]),
    )

    search_roi = None
    inner_dbg = None
    anchors_g = None
    anchors_c = None
    if stab_info and stab_info.get("ok"):
        search_roi = _clamp_roi(stab_info.get("search_roi_current"), W, H)
        inner_dbg = stab_info.get("current_lines_debug")
        anchors_g = stab_info.get("anchors_golden")
        anchors_c = stab_info.get("anchors_current")

    # Render function for the interactive viewer
    def render(state):
        zoom = float(state["zoom"])
        vis = full_test_img.copy()

        # ROIs layer
        if state["show_rois"]:
            _draw_roi(vis, _clamp_roi(roi_cfg, W, H), (0, 255, 255), 2)        # cfg ROI
            _draw_roi(vis, _clamp_roi(roi_for_detection, W, H), (255, 255, 0), 2)  # moved ROI
            _draw_roi(vis, search_roi, (255, 0, 255), 2)  # search ROI

            if state["show_labels"]:
                _put_label(vis, "cfg ROI (golden px)", (roi_cfg[0] + 8, roi_cfg[1] - 10), (0, 255, 255), zoom)
                _put_label(vis, "detect ROI (moved)", (roi_for_detection[0] + 8, roi_for_detection[1] - 10), (255, 255, 0), zoom)
                if search_roi:
                    _put_label(vis, "search ROI", (search_roi[0] + 8, search_roi[1] - 10), (255, 0, 255), zoom)

        # Notch debug layer
        if state["show_notch"] and inner_dbg and inner_dbg.get("ok"):
            _draw_notch_debug(
                vis,
                inner_dbg,
                zoom=zoom,
                draw_points=state["show_notch_points"],
                draw_lines=state["show_notch_lines"],
            )

        # Anchors layer
        if state["show_anchors"]:
            _draw_anchors(vis, anchors_g, zoom, (0, 220, 220), "Agld", show_text=state["show_labels"])
            _draw_anchors(vis, anchors_c, zoom, (255, 0, 0), "Acur", show_text=state["show_labels"])

        # Baseplate layer
        if state["show_baseplate"]:
            if test_cnt_rel is not None:
                cnt_abs = test_cnt_rel + np.array([[roi_for_detection[0], roi_for_detection[1]]], dtype=np.float32)
                _draw_contour_abs(vis, cnt_abs, color=(0, 255, 0), thickness=2)

        # Points
        cv2.circle(vis, (int(abs_center[0]), int(abs_center[1])), 7, (0, 0, 255), -1)
        cv2.circle(vis, (int(expected_abs[0]), int(expected_abs[1])), 7, (255, 0, 0), 2)

        if state["show_labels"]:
            _put_label(vis, "detected", (int(abs_center[0]) + 10, int(abs_center[1]) + 5), (0, 0, 255), zoom)
            _put_label(vis, "expected", (int(expected_abs[0]) + 10, int(expected_abs[1]) + 5), (255, 0, 0), zoom)

        # Header
        if state["show_labels"]:
            _put_label(vis, f"ROI_STAB: {'ON (OK)' if (stab_info and stab_info.get('ok')) else 'OFF'}",
                       (20, 40), (0, 0, 255), zoom)
            _put_label(vis, f"SEARCH_PADDING_PX={SEARCH_PADDING_PX}",
                       (20, 70), (0, 0, 255), zoom)

        return vis

    _interactive_debug_viewer(full_test_img, render)


def main():
    if not os.path.exists(CFG_PATH):
        print("❌ Config JSON not found:", CFG_PATH)
        return

    with open(CFG_PATH, "r") as f:
        cfg = json.load(f)

    if AUTO_MODE:
        from auto_run import run_batch  # lazy import
        os.makedirs(OUTPUT_ROOT, exist_ok=True)
        run_batch(
            config_path=CFG_PATH,
            incoming_dir=INCOMING_DIR,
            output_root=OUTPUT_ROOT,
            show_overlay=SHOW_OVERLAY,
            auto_close_sec=1.5,
        )
    else:
        test_image_path = cfg.get("test_image_path") or MANUAL_TEST_IMAGE_PATH
        if not test_image_path:
            print("❌ No manual test image path set.")
            return
        process_one(test_image_path, cfg)


if __name__ == "__main__":
    main()
