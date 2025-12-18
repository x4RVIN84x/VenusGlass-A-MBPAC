import cv2
import json
import os
import numpy as np
import detector

CONFIG_PATH = r"E:\ARVIN\A-MBPAC\reference_json\207_golden_configv2.json"

FOOTER_COLOR = (245, 245, 245)
FOOTER_SHADOW = (30, 30, 30)
STATUS_COLOR = (50, 50, 255)

ZOOM_MIN = 1.0
ZOOM_MAX = 8.0
ZOOM_STEP = 1.25
PAN_STEP_FRAC = 0.12


def _load_cfg(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _save_cfg(path, cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=4)
    print("✅ Saved to", path)


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _roi_from_points(p0, p1):
    x0, y0 = p0
    x1, y1 = p1
    x = int(min(x0, x1))
    y = int(min(y0, y1))
    w = int(abs(x1 - x0))
    h = int(abs(y1 - y0))
    return [x, y, max(1, w), max(1, h)]


def _get_view_rect(img_shape, zoom, center_xy):
    H, W = img_shape[:2]
    zoom = float(_clamp(zoom, ZOOM_MIN, ZOOM_MAX))

    view_w = int(round(W / zoom))
    view_h = int(round(H / zoom))
    view_w = max(80, min(W, view_w))
    view_h = max(80, min(H, view_h))

    cx, cy = center_xy
    cx = float(_clamp(cx, 0, W - 1))
    cy = float(_clamp(cy, 0, H - 1))

    x0 = int(round(cx - view_w / 2))
    y0 = int(round(cy - view_h / 2))
    x0 = _clamp(x0, 0, W - view_w)
    y0 = _clamp(y0, 0, H - view_h)

    return int(x0), int(y0), int(view_w), int(view_h)


def _disp_to_img(pt_xy, view_rect, disp_size):
    vx, vy, vw, vh = view_rect
    disp_w, disp_h = disp_size
    x, y = float(pt_xy[0]), float(pt_xy[1])
    ix = vx + x * (float(vw) / disp_w)
    iy = vy + y * (float(vh) / disp_h)
    return (float(ix), float(iy))


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


def _draw_overlay(img_full, state):
    vis = img_full.copy()
    H, W = vis.shape[:2]

    # ROI (yellow)
    roi = state.get("roi")
    if roi is not None:
        x, y, w, h = roi
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # inner border debug
    res = state.get("inner_res")
    if res and res.get("ok"):
        hull = res.get("hull_abs")
        if hull is not None:
            cv2.drawContours(vis, [hull], -1, (0, 255, 0), 2)

        pts_abs = res.get("points_used_abs", {})
        for k in ["left", "right", "bottom"]:
            arr = pts_abs.get(k)
            if arr is None:
                continue
            step = max(1, len(arr) // 350)
            for px, py in arr[::step]:
                cv2.circle(vis, (int(px), int(py)), 1, (0, 255, 0), -1)

        lines = res.get("lines", {})
        colors = {"left": (255, 0, 0), "right": (255, 0, 0), "bottom": (255, 255, 0)}
        for k in ["left", "right", "bottom"]:
            if k not in lines:
                continue
            seg = _line_endpoints_in_image(lines[k], W, H)
            if seg:
                cv2.line(vis, seg[0], seg[1], colors[k], 2)

    # baseplate center (red dot)
    if state.get("bp_center_abs") is not None:
        cx, cy = state["bp_center_abs"]
        cv2.circle(vis, (int(cx), int(cy)), 7, (0, 0, 255), -1)

    return vis


def main():
    cfg = _load_cfg(CONFIG_PATH)
    golden_path = cfg.get("golden_image_path")
    if not golden_path:
        print("❌ Config missing 'golden_image_path'")
        return

    img = cv2.imread(golden_path)
    if img is None:
        print("❌ Failed to load:", golden_path)
        return

    H, W = img.shape[:2]
    disp_w = min(1600, W)
    disp_h = int(round(disp_w * (H / float(W))))
    disp_size = (disp_w, disp_h)

    state = {
        "drawing": False,
        "roi_start": None,
        "roi_end": None,
        "roi_done": False,

        "roi": cfg.get("baseplate_roi") or cfg.get("registration_roi") or cfg.get("roi") or None,
        "inner_res": None,

        "click_mode": False,
        "bp_center_abs": None,
        "bp_auto": False,

        "zoom": 1.0,
        "view_center": (W / 2.0, H / 2.0),

        "status": "Draw ROI around notch/baseplate. R=recompute. C=click baseplate center.",
    }

    def recompute_all():
        if state["roi"] is None:
            state["inner_res"] = None
            state["bp_center_abs"] = None
            state["bp_auto"] = False
            state["status"] = "❌ No ROI. Draw ROI first."
            return

        # 1) inner border
        res = detector.detect_inner_border_lines_edges_local(
            img,
            state["roi"],
            canny_low=60,
            canny_high=140,
            min_points=40,
            band_side_frac=0.22,
            band_bottom_frac=0.25,
            sample_stride=1,
        )
        state["inner_res"] = res

        if not res.get("ok"):
            state["bp_center_abs"] = None
            state["bp_auto"] = False
            state["status"] = f"❌ inner border failed: {res.get('reason')} counts={res.get('counts')}"
            return

        c = res.get("counts", {})
        msg = f"✅ inner border OK. points L={c.get('left')} R={c.get('right')} B={c.get('bottom')}"

        # 2) baseplate auto attempt (CLOSE-UP ROI)
        bp = detector.detect_baseplate_in_roi(img, state["roi"])
        if bp is not None:
            state["bp_center_abs"] = bp
            state["bp_auto"] = True
            msg += " | ✅ baseplate auto"
        else:
            state["bp_center_abs"] = None
            state["bp_auto"] = False
            msg += " | ❌ baseplate not found (press C to click)"

        msg += " | Enter=save"
        state["status"] = msg

    if state["roi"] is not None:
        recompute_all()

    win = "calibrate"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, param):
        view_rect = _get_view_rect(img.shape, state["zoom"], state["view_center"])
        ix, iy = _disp_to_img((x, y), view_rect, disp_size)

        # Manual click baseplate center
        if state["click_mode"] and event == cv2.EVENT_LBUTTONDOWN:
            state["bp_center_abs"] = (float(ix), float(iy))
            state["bp_auto"] = False
            state["click_mode"] = False
            state["status"] = "✅ baseplate center set (manual click). Enter=save"
            return

        # Draw ROI
        if event == cv2.EVENT_LBUTTONDOWN:
            state["roi_start"] = (float(ix), float(iy))
            state["roi_end"] = (float(ix), float(iy))
            state["drawing"] = True
            state["roi_done"] = False
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            state["roi_end"] = (float(ix), float(iy))
        elif event == cv2.EVENT_LBUTTONUP:
            state["roi_end"] = (float(ix), float(iy))
            state["drawing"] = False
            state["roi_done"] = True

    cv2.setMouseCallback(win, on_mouse)

    while True:
        if state["drawing"] and state["roi_start"] and state["roi_end"]:
            state["roi"] = _roi_from_points(
                (int(state["roi_start"][0]), int(state["roi_start"][1])),
                (int(state["roi_end"][0]), int(state["roi_end"][1])),
            )

        if state["roi_done"]:
            state["roi_done"] = False
            state["bp_center_abs"] = None
            state["bp_auto"] = False
            recompute_all()

        overlay = _draw_overlay(img, state)

        view_rect = _get_view_rect(img.shape, state["zoom"], state["view_center"])
        vx, vy, vw, vh = view_rect
        view = overlay[vy:vy + vh, vx:vx + vw]
        view_disp = cv2.resize(view, disp_size, interpolation=cv2.INTER_LINEAR)

        cv2.putText(view_disp, state["status"], (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, STATUS_COLOR, 2)

        footer = "Draw ROI | R=recompute | C=click center | Enter=save | +/- zoom | WASD pan | 0 reset | Q/Esc quit"
        cv2.putText(view_disp, footer, (20, view_disp.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, FOOTER_SHADOW, 3)
        cv2.putText(view_disp, footer, (20, view_disp.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, FOOTER_COLOR, 2)

        cv2.imshow(win, view_disp)
        key = cv2.waitKey(20) & 0xFF

        # zoom
        if key in (ord('+'), ord('=')):
            state["zoom"] = float(_clamp(state["zoom"] * ZOOM_STEP, ZOOM_MIN, ZOOM_MAX))
        if key in (ord('-'), ord('_')):
            state["zoom"] = float(_clamp(state["zoom"] / ZOOM_STEP, ZOOM_MIN, ZOOM_MAX))
        if key == ord('0'):
            state["zoom"] = 1.0
            state["view_center"] = (W / 2.0, H / 2.0)

        # pan
        if key in (ord('w'), ord('a'), ord('s'), ord('d')):
            vx, vy, vw, vh = _get_view_rect(img.shape, state["zoom"], state["view_center"])
            cx, cy = state["view_center"]
            if key == ord('w'):
                cy -= PAN_STEP_FRAC * vh
            elif key == ord('s'):
                cy += PAN_STEP_FRAC * vh
            elif key == ord('a'):
                cx -= PAN_STEP_FRAC * vw
            elif key == ord('d'):
                cx += PAN_STEP_FRAC * vw
            state["view_center"] = (_clamp(cx, 0, W - 1), _clamp(cy, 0, H - 1))

        if key == ord('r'):
            recompute_all()

        if key == ord('c'):
            state["click_mode"] = True
            state["status"] = "🖱️ Click mode: click baseplate center (overrides auto)"

        # save (Enter)
        if key == 13:
            if state["roi"] is None:
                print("❌ No ROI.")
                continue

            res = state["inner_res"]
            if not res or not res.get("ok"):
                print("❌ Inner border not valid.")
                continue

            if state["bp_center_abs"] is None:
                print("❌ No baseplate center. Use C to click, or adjust ROI and press R.")
                state["status"] = "❌ No baseplate center. Press C to click it."
                continue

            rx, ry, _, _ = state["roi"]

            # Store ROI
            cfg["baseplate_roi"] = [int(v) for v in state["roi"]]
            cfg["roi"] = cfg["baseplate_roi"]  # compat

            # Store lines
            cfg["inner_border_lines_used"] = ["left", "right", "bottom"]
            cfg["inner_border_lines"] = {k: [float(v) for v in res["lines"][k]] for k in ["left", "right", "bottom"]}

            # Store baseplate center relative to ROI
            cfg["expected_center_in_baseplate_roi"] = [
                float(state["bp_center_abs"][0] - rx),
                float(state["bp_center_abs"][1] - ry),
            ]
            cfg["expected_center"] = cfg["expected_center_in_baseplate_roi"]  # compat

            cfg["baseplate_center_source"] = "auto" if state["bp_auto"] else "manual"

            _save_cfg(CONFIG_PATH, cfg)
            state["status"] = f"✅ Saved calibration. baseplate={cfg['baseplate_center_source']}"

        if key == 27 or key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
