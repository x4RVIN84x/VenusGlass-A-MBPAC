import cv2
import json
import os
import numpy as np
import detector

CONFIG_PATH = r"C:\Users\m.afrazeh\PycharmProjects\VenusGlass-A-MBPAC\recipes\C270TEST_207\golden_config.json"

FOOTER_COLOR  = (245, 245, 245)
FOOTER_SHADOW = (30, 30, 30)

STATUS_COLOR  = (50, 50, 255)
STATUS_SCALE  = 0.72
STATUS_THICK  = 2

ZOOM_MIN = 1.0
ZOOM_MAX = 10.0
ZOOM_STEP = 1.25
PAN_STEP_FRAC = 0.12

# baseplate detector defaults (v0-ish)
BP_KW = dict(
    padding=30,
    shrink_border_px=10,
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
)

def _normalize_contour_abs(cnt, roi):
    """
    Some detector paths return contour points ROI-relative (0..w,0..h).
    Some return absolute.
    This converts to absolute safely.
    """
    if cnt is None or len(cnt) < 3 or roi is None:
        return cnt

    rx, ry, rw, rh = roi
    pts = cnt.reshape(-1, 2).astype(np.float32)

    # Heuristic: if most points lie inside ROI extents, treat as ROI-relative
    in_rel = np.mean((pts[:, 0] >= -5) & (pts[:, 0] <= rw + 5) &
                     (pts[:, 1] >= -5) & (pts[:, 1] <= rh + 5))

    if in_rel > 0.85:
        pts[:, 0] += rx
        pts[:, 1] += ry
        return pts.reshape(cnt.shape).astype(cnt.dtype)

    return cnt


def _load_cfg(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def _save_cfg(path, cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=4)
    print("OK! Saved to", path)

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

    # registration ROI (magenta)
    reg = state.get("registration_roi")
    if reg is not None:
        x, y, w, h = reg
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 255), 2)

    # baseplate ROI (yellow)
    bp_roi = state.get("baseplate_roi")
    if bp_roi is not None:
        x, y, w, h = bp_roi
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # inner border debug (from registration ROI)
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

    # baseplate chosen contour (cyan) — only if available
    bp_cnt = state.get("bp_contour_abs")
    if bp_cnt is not None and len(bp_cnt) >= 3:
        cv2.drawContours(vis, [bp_cnt], -1, (255, 255, 0), 2)

    # baseplate center (red dot)
    bp = state.get("bp_center_abs")
    if bp is not None:
        cx, cy = bp
        cv2.circle(vis, (int(cx), int(cy)), 7, (0, 0, 255), -1)

    return vis

def main():
    cfg = _load_cfg(CONFIG_PATH)
    golden_path = cfg.get("golden_image_path")
    if not golden_path:
        print("FAIL! Config missing 'golden_image_path'")
        return

    img = cv2.imread(golden_path)
    if img is None:
        print("FAIL! Failed to load:", golden_path)
        return

    H, W = img.shape[:2]
    disp_w = min(1600, W)
    disp_h = int(round(disp_w * (H / float(W))))
    disp_size = (disp_w, disp_h)

    state = {
        # drawing state
        "drawing": False,
        "roi_start": None,
        "roi_end": None,
        "roi_done": False,

        # which ROI is being drawn right now
        # "registration" or "baseplate"
        "draw_mode": "registration",

        # stored ROIs
        "registration_roi": cfg.get("registration_roi") or None,
        "baseplate_roi": cfg.get("baseplate_roi") or cfg.get("roi") or None,

        # detection results
        "inner_res": None,
        "bp_center_abs": None,
        "bp_contour_abs": None,
        "bp_auto": False,

        # UI
        "click_mode": False,
        "zoom": 1.0,
        "view_center": (W / 2.0, H / 2.0),

        "status": "Press 1: draw REG ROI (notch) | Press 2: draw BP ROI | R=recompute | C=click BP center | Enter=save",
    }

    def recompute_all():
        # Need registration ROI to compute notch lines
        if state["registration_roi"] is None:
            state["inner_res"] = None
            state["status"] = "FAIL! No registration ROI. Press 1 and draw ROI around notch/inner contour."
            return

        # 1) inner border lines from registration ROI
        res = detector.detect_inner_border_lines_edges_local(
            img,
            state["registration_roi"],
            canny_low=60,
            canny_high=140,
            min_points=40,
            band_side_frac=0.22,
            band_bottom_frac=0.25,
            sample_stride=1,
        )
        state["inner_res"] = res

        if not res.get("ok"):
            state["status"] = f"FAIL! Inner border FAILED: {res.get('reason')} counts={res.get('counts')}"
            return

        c = res.get("counts", {})
        msg = f"OK! Notch lines OK. L={c.get('left')} R={c.get('right')} B={c.get('bottom')}"

        # 2) baseplate auto (needs baseplate ROI)
        if state["baseplate_roi"] is None:
            state["bp_center_abs"] = None
            state["bp_contour_abs"] = None
            state["bp_auto"] = False
            state["status"] = msg + " | WARNING! No BP ROI (press 2 to draw)"
            return

        dbg = detector.detect_baseplate_in_roi(img, state["baseplate_roi"], return_debug=True, **BP_KW)

        if dbg and dbg.get("ok") and dbg.get("center_abs") is not None:
            state["bp_center_abs"] = dbg["center_abs"]
            state["bp_contour_abs"] = dbg.get("contour_abs")
            state["bp_auto"] = True
            msg += " | OK! Baseplate: AUTO"
        else:
            state["bp_center_abs"] = None
            state["bp_contour_abs"] = None
            state["bp_auto"] = False
            msg += " | FAIL! Baseplate: NOT FOUND (press C to click)"

        msg += " | Enter=save"
        state["status"] = msg

    # initial recompute if we have enough data
    if state["registration_roi"] is not None:
        recompute_all()

    win = "calibrate"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, param):
        view_rect = _get_view_rect(img.shape, state["zoom"], state["view_center"])
        ix, iy = _disp_to_img((x, y), view_rect, disp_size)

        # Manual click baseplate center (override)
        if state["click_mode"] and event == cv2.EVENT_LBUTTONDOWN:
            state["bp_center_abs"] = (float(ix), float(iy))
            state["bp_contour_abs"] = None
            state["bp_auto"] = False
            state["click_mode"] = False
            state["status"] = "OK! Baseplate center set (manual click). Enter=save"
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
        # update ROI while drawing
        if state["drawing"] and state["roi_start"] and state["roi_end"]:
            roi_tmp = _roi_from_points(
                (int(state["roi_start"][0]), int(state["roi_start"][1])),
                (int(state["roi_end"][0]), int(state["roi_end"][1])),
            )
            if state["draw_mode"] == "registration":
                state["registration_roi"] = roi_tmp
            else:
                state["baseplate_roi"] = roi_tmp

        if state["roi_done"]:
            state["roi_done"] = False
            # clear derived results that depend on ROIs
            state["bp_center_abs"] = None
            state["bp_contour_abs"] = None
            state["bp_auto"] = False
            recompute_all()

        overlay = _draw_overlay(img, state)

        view_rect = _get_view_rect(img.shape, state["zoom"], state["view_center"])
        vx, vy, vw, vh = view_rect
        view = overlay[vy:vy + vh, vx:vx + vw]
        view_disp = cv2.resize(view, disp_size, interpolation=cv2.INTER_LINEAR)

        # status + footer
        cv2.putText(view_disp, state["status"], (18, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, STATUS_SCALE, STATUS_COLOR, STATUS_THICK)

        footer = "1=REG ROI | 2=BP ROI | R=recompute | C=click BP center | Enter=save | +/- zoom | WASD pan | 0 reset | Q/Esc quit"
        cv2.putText(view_disp, footer, (18, view_disp.shape[0] - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, FOOTER_SHADOW, 3)
        cv2.putText(view_disp, footer, (18, view_disp.shape[0] - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, FOOTER_COLOR, 2)

        cv2.imshow(win, view_disp)
        key = cv2.waitKey(20) & 0xFF

        # switch draw mode
        if key == ord('1'):
            state["draw_mode"] = "registration"
            state["status"] = "🟪 Draw REG ROI around notch/inner contour area (blue lines)."
        if key == ord('2'):
            state["draw_mode"] = "baseplate"
            state["status"] = "🟨 Draw BP ROI around baseplate (tight)."

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
            state["status"] = "🖱️ Click mode: click baseplate center (manual override)."

        # save (Enter)
        if key == 13:
            if state["registration_roi"] is None:
                print("FAIL! No registration_roi (press 1 and draw).")
                state["status"] = "FAIL! Need registration ROI (press 1)."
                continue

            res = state["inner_res"]
            if not res or not res.get("ok"):
                print("FAIL! Notch lines invalid. Press R or redraw registration ROI.")
                state["status"] = "FAIL! Notch lines invalid. Press R."
                continue

            if state["baseplate_roi"] is None:
                print("FAIL! No baseplate_roi (press 2 and draw).")
                state["status"] = "FAIL! Need baseplate ROI (press 2)."
                continue

            if state["bp_center_abs"] is None:
                print("FAIL! No baseplate center. Press C to click it or adjust BP ROI and press R.")
                state["status"] = "FAIL! No baseplate center. Press C."
                continue

            brx, bry, _, _ = state["baseplate_roi"]

            # Save ROIs
            cfg["registration_roi"] = [int(v) for v in state["registration_roi"]]
            cfg["baseplate_roi"] = [int(v) for v in state["baseplate_roi"]]

            # keep compat (old code expects cfg["roi"] to be baseplate ROI)
            cfg["roi"] = cfg["baseplate_roi"]

            # Save notch geometry (THIS is the “blue line” reference frame)
            cfg["inner_border_lines_used"] = ["left", "right", "bottom"]
            cfg["inner_border_lines"] = {
                k: [float(v) for v in res["lines"][k]]
                for k in ["left", "right", "bottom"]
            }

            # Save expected baseplate center relative to baseplate ROI
            cfg["expected_center_in_baseplate_roi"] = [
                float(state["bp_center_abs"][0] - brx),
                float(state["bp_center_abs"][1] - bry),
            ]
            cfg["expected_center"] = cfg["expected_center_in_baseplate_roi"]  # compat
            cfg["baseplate_center_source"] = "auto" if state["bp_auto"] else "manual"

            _save_cfg(CONFIG_PATH, cfg)
            state["status"] = f"OK! Saved. notch lines + ROIs + baseplate center ({cfg['baseplate_center_source']})"

        if key == 27 or key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
