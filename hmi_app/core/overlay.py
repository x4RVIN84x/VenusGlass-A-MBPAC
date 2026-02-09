from __future__ import annotations
import numpy as np
import cv2

def draw_fitline(vis, line, color=(255, 0, 0), thickness=2):
    if line is None:
        return
    vx, vy, x0, y0 = map(float, line)
    p1 = (int(round(x0 - vx * 5000)), int(round(y0 - vy * 5000)))
    p2 = (int(round(x0 + vx * 5000)), int(round(y0 + vy * 5000)))
    cv2.line(vis, p1, p2, color, thickness, lineType=cv2.LINE_AA)

def draw_points(vis, pts, color=(0, 255, 255), radius=1, step=12):
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

def draw_stab_debug(vis, stab_info: dict):
    if not stab_info:
        return
    ok = stab_info.get("ok", False)
    if "search_roi_current" in stab_info:
        x, y, w, h = stab_info["search_roi_current"]
        col = (120, 120, 120) if ok else (0, 0, 255)
        cv2.rectangle(vis, (x, y), (x + w, y + h), col, 1, lineType=cv2.LINE_AA)

    dbg = stab_info.get("current_lines_debug", {}) or {}
    lines = dbg.get("lines", None)
    pts_used = dbg.get("points_used_abs", None)
    hull = dbg.get("hull_abs", None)

    if isinstance(lines, dict):
        draw_fitline(vis, lines.get("left"), color=(255, 0, 0), thickness=2)
        draw_fitline(vis, lines.get("right"), color=(255, 0, 0), thickness=2)
        draw_fitline(vis, lines.get("bottom"), color=(255, 0, 0), thickness=2)

    if hull is not None:
        try:
            cv2.polylines(vis, [hull], True, (0, 255, 0), 2, lineType=cv2.LINE_AA)
        except Exception:
            pass

    if isinstance(pts_used, dict):
        draw_points(vis, pts_used.get("left"), color=(0, 255, 255), radius=1, step=12)
        draw_points(vis, pts_used.get("right"), color=(0, 255, 255), radius=1, step=12)
        draw_points(vis, pts_used.get("bottom"), color=(0, 255, 255), radius=1, step=12)

    anchors = stab_info.get("anchors_current", None)
    if isinstance(anchors, dict):
        for k, pt in anchors.items():
            ax, ay = int(round(pt[0])), int(round(pt[1]))
            cv2.circle(vis, (ax, ay), 5, (0, 0, 255), -1, lineType=cv2.LINE_AA)
            cv2.putText(vis, k, (ax + 6, ay - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    if "y_top_ref_c" in stab_info:
        y = int(round(stab_info["y_top_ref_c"]))
        cv2.line(vis, (0, y), (vis.shape[1]-1, y), (255, 255, 255), 1, lineType=cv2.LINE_AA)

def draw_baseplate_overlay(vis, roi, center_rel, contour_rel):
    x, y, w, h = roi
    cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 0), 2, lineType=cv2.LINE_AA)
    if contour_rel is not None and len(contour_rel) >= 3:
        cnt_abs = contour_rel + np.array([[x, y]], dtype=np.int32)
        cv2.drawContours(vis, [cnt_abs], -1, (0, 255, 0), 2)
    if center_rel is not None:
        cx = int(round(x + center_rel[0]))
        cy = int(round(y + center_rel[1]))
        cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1, lineType=cv2.LINE_AA)

def draw_status_box(vis, text, state="FAIL"):
    if state == "PASS":
        color = (0, 200, 0)
    elif state == "TRACK":
        color = (0, 200, 200)
    elif state == "SEARCH":
        color = (180, 180, 180)
    else:
        color = (0, 0, 255)

    cv2.rectangle(vis, (20, 20), (760, 120), (20, 20, 20), -1)
    cv2.circle(vis, (48, 70), 10, color, -1, cv2.LINE_AA)

    cv2.putText(vis, text, (70, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 6, cv2.LINE_AA)
    cv2.putText(vis, text, (70, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
