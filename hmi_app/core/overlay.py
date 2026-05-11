from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


# ----------------------------
# Small drawing helpers
# ----------------------------
def _as_dict(v) -> dict:
    return v if isinstance(v, dict) else {}


def _setting(settings, name: str, default):
    if settings is None:
        return default
    return getattr(settings, name, default)


def _is_good_pt(pt) -> bool:
    try:
        return (
            pt is not None
            and len(pt) >= 2
            and np.isfinite(float(pt[0]))
            and np.isfinite(float(pt[1]))
        )
    except Exception:
        return False


def _ipt(pt) -> Tuple[int, int]:
    return (int(round(float(pt[0]))), int(round(float(pt[1]))))


def _as_points(pts) -> Optional[np.ndarray]:
    if pts is None:
        return None

    try:
        arr = np.asarray(pts, dtype=np.float32)
    except Exception:
        return None

    if arr.size == 0:
        return None

    try:
        arr = arr.reshape(-1, 2)
    except Exception:
        return None

    arr = arr[np.isfinite(arr).all(axis=1)]

    if len(arr) <= 0:
        return None

    return arr


def _put_text(
    vis,
    text,
    org,
    *,
    scale=0.5,
    color=(255, 255, 255),
    thickness=1,
):
    x, y = org

    cv2.putText(
        vis,
        str(text),
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        float(scale),
        (0, 0, 0),
        int(thickness) + 2,
        cv2.LINE_AA,
    )

    cv2.putText(
        vis,
        str(text),
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        float(scale),
        color,
        int(thickness),
        cv2.LINE_AA,
    )


def _draw_points(vis, pts, color=(0, 255, 255), radius=1, step=16):
    arr = _as_points(pts)

    if arr is None:
        return

    step = max(1, int(step))
    H, W = vis.shape[:2]

    for i in range(0, len(arr), step):
        x = int(round(float(arr[i, 0])))
        y = int(round(float(arr[i, 1])))

        if 0 <= x < W and 0 <= y < H:
            cv2.circle(vis, (x, y), int(radius), color, -1, lineType=cv2.LINE_AA)


def _draw_polyline(vis, cnt, color=(0, 255, 0), thickness=1, closed=True):
    if cnt is None:
        return

    try:
        arr = np.asarray(cnt, dtype=np.int32)
    except Exception:
        return

    if arr.size == 0:
        return

    try:
        arr = arr.reshape(-1, 1, 2)
    except Exception:
        return

    if len(arr) < 2:
        return

    try:
        cv2.polylines(vis, [arr], bool(closed), color, int(thickness), lineType=cv2.LINE_AA)
    except Exception:
        pass


def _draw_fitline_clipped(vis, line, color=(160, 80, 160), thickness=1):
    """
    Debug only. Clipped finite line instead of huge infinite X.
    """
    if line is None:
        return

    try:
        vx, vy, x0, y0 = map(float, line)
    except Exception:
        return

    if not all(np.isfinite([vx, vy, x0, y0])):
        return

    H, W = vis.shape[:2]

    p1 = (int(round(x0 - vx * 5000)), int(round(y0 - vy * 5000)))
    p2 = (int(round(x0 + vx * 5000)), int(round(y0 + vy * 5000)))

    ok, cp1, cp2 = cv2.clipLine((0, 0, W, H), p1, p2)

    if ok:
        cv2.line(vis, cp1, cp2, color, int(thickness), lineType=cv2.LINE_AA)


# ----------------------------
# Stabilizer debug overlay
# ----------------------------
def draw_stab_debug(vis, stab_info: dict, *, settings=None):
    """
    Layered stabilizer overlay.

    Layers are controlled through EngineSettings:
      show_stab_search_roi
      show_stab_feature_points
      show_stab_anchors
      show_stab_legacy
      show_stab_text
    """
    if vis is None or not isinstance(stab_info, dict):
        return

    ok = bool(stab_info.get("ok", False))
    method = str(stab_info.get("current_anchor_method", "unknown"))

    show_search = bool(_setting(settings, "show_stab_search_roi", True))
    show_points = bool(_setting(settings, "show_stab_feature_points", True))
    show_anchors = bool(_setting(settings, "show_stab_anchors", True))
    show_legacy = bool(_setting(settings, "show_stab_legacy", False))
    show_text = bool(_setting(settings, "show_stab_text", True))

    solid_dbg = _as_dict(stab_info.get("solid_edge_debug"))
    legacy_dbg = _as_dict(stab_info.get("legacy_lines_debug"))
    current_dbg = _as_dict(stab_info.get("current_lines_debug"))

    if method == "solid_edge" and solid_dbg:
        active_dbg = solid_dbg
    elif method == "legacy_dot_band_lines" and legacy_dbg:
        active_dbg = legacy_dbg
    else:
        active_dbg = current_dbg

    # ----------------------------
    # Search ROI
    # ----------------------------
    if show_search and "search_roi_current" in stab_info:
        try:
            x, y, w, h = map(int, stab_info["search_roi_current"])
            col = (120, 120, 120) if ok else (0, 0, 255)

            cv2.rectangle(
                vis,
                (x, y),
                (x + w, y + h),
                col,
                1,
                lineType=cv2.LINE_AA,
            )

            _put_text(vis, "search", (x + 4, y + 16), scale=0.42, color=col)
        except Exception:
            pass

    # ----------------------------
    # Active feature points / contour source
    # ----------------------------
    if show_points:
        pts_used = _as_dict(active_dbg.get("points_used_abs"))

        if method == "solid_edge":
            # Solid edge source:
            # orange = top support
            # cyan = sparse solid edge
            # yellow/orange = bottom support
            _draw_points(vis, pts_used.get("left_top"), color=(255, 180, 0), radius=1, step=8)
            _draw_points(vis, pts_used.get("right_top"), color=(255, 180, 0), radius=1, step=8)
            _draw_points(vis, pts_used.get("bottom"), color=(0, 180, 255), radius=1, step=10)
            _draw_points(vis, pts_used.get("all"), color=(80, 220, 255), radius=1, step=30)

            _draw_polyline(vis, active_dbg.get("hull_abs"), color=(0, 180, 0), thickness=1, closed=True)
            _draw_polyline(vis, active_dbg.get("contour_abs"), color=(0, 128, 255), thickness=1, closed=True)

        elif method == "legacy_dot_band_lines":
            _draw_points(vis, pts_used.get("left"), color=(80, 80, 255), radius=1, step=12)
            _draw_points(vis, pts_used.get("right"), color=(80, 80, 255), radius=1, step=12)
            _draw_points(vis, pts_used.get("bottom"), color=(80, 80, 255), radius=1, step=12)

    # ----------------------------
    # Legacy fallback debug
    # ----------------------------
    if show_legacy:
        legacy_pts = _as_dict(legacy_dbg.get("points_used_abs"))

        _draw_points(vis, legacy_pts.get("left"), color=(120, 80, 255), radius=1, step=14)
        _draw_points(vis, legacy_pts.get("right"), color=(120, 80, 255), radius=1, step=14)
        _draw_points(vis, legacy_pts.get("bottom"), color=(120, 80, 255), radius=1, step=14)

        legacy_lines = _as_dict(legacy_dbg.get("lines"))
        for _name, line in legacy_lines.items():
            _draw_fitline_clipped(vis, line, color=(180, 80, 180), thickness=1)

        if method == "legacy_dot_band_lines":
            _put_text(vis, "LEGACY ACTIVE", (16, 220), scale=0.7, color=(120, 80, 255), thickness=2)
        elif legacy_dbg:
            _put_text(vis, "legacy debug available", (16, 220), scale=0.5, color=(120, 80, 255))

    # ----------------------------
    # Anchors
    # ----------------------------
    if show_anchors:
        raw_anchors = _as_dict(stab_info.get("anchors_current_raw"))
        final_anchors = _as_dict(stab_info.get("anchors_current"))

        # Raw anchors: small orange dots.
        for name in ("left_top", "right_top", "bottom_mid"):
            pt = raw_anchors.get(name)
            if _is_good_pt(pt):
                cv2.circle(vis, _ipt(pt), 3, (0, 165, 255), -1, lineType=cv2.LINE_AA)

        # Final anchors: larger clean dots.
        for name in ("left_top", "right_top", "bottom_mid"):
            pt = final_anchors.get(name)

            if not _is_good_pt(pt):
                continue

            x, y = _ipt(pt)

            if name == "bottom_mid":
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            cv2.circle(vis, (x, y), 6, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(vis, (x, y), 8, (0, 0, 0), 1, lineType=cv2.LINE_AA)
            _put_text(vis, name, (x + 8, y - 8), scale=0.45, color=color)

        lt = final_anchors.get("left_top")
        rt = final_anchors.get("right_top")
        bm = final_anchors.get("bottom_mid")

        # Final anchor skeleton = actual transform model.
        if _is_good_pt(lt) and _is_good_pt(rt):
            cv2.line(vis, _ipt(lt), _ipt(rt), (255, 255, 0), 2, lineType=cv2.LINE_AA)

        if _is_good_pt(lt) and _is_good_pt(bm):
            cv2.line(vis, _ipt(lt), _ipt(bm), (0, 255, 0), 1, lineType=cv2.LINE_AA)

        if _is_good_pt(rt) and _is_good_pt(bm):
            cv2.line(vis, _ipt(rt), _ipt(bm), (0, 255, 0), 1, lineType=cv2.LINE_AA)

        # Top Y line belongs with anchor model, not points.
        if "y_top_ref_c" in stab_info:
            try:
                yy = int(round(float(stab_info["y_top_ref_c"])))
                cv2.line(vis, (0, yy), (vis.shape[1] - 1, yy), (255, 255, 255), 1, lineType=cv2.LINE_AA)
                _put_text(vis, "top_y", (8, yy - 4), scale=0.4, color=(255, 255, 255))
            except Exception:
                pass

    # ----------------------------
    # Mini debug text
    # ----------------------------
    if show_text:
        top_hyst = _as_dict(stab_info.get("top_anchor_hysteresis"))
        transform_hyst = _as_dict(stab_info.get("hysteresis"))

        top_mode = str(top_hyst.get("mode", "-"))
        trans_mode = str(transform_hyst.get("mode", "-"))

        dtrans = transform_hyst.get("dtranslation_px", None)
        dangle = transform_hyst.get("dangle_deg", None)

        text_lines = [
            f"anchor: {method}",
            f"top_x: {top_mode}",
            f"M raw delta: {trans_mode}",
        ]

        if dtrans is not None:
            try:
                text_lines.append(f"raw dT: {float(dtrans):.1f}px")
            except Exception:
                pass

        if dangle is not None:
            try:
                text_lines.append(f"raw dA: {float(dangle):.2f}deg")
            except Exception:
                pass

        if method != "solid_edge":
            solid_reason = _as_dict(stab_info.get("solid_edge_debug")).get("reason")
            if solid_reason:
                text_lines.append(f"solid fail: {solid_reason}")

        x0, y0 = 16, 145
        for i, line in enumerate(text_lines):
            _put_text(vis, line, (x0, y0 + i * 18), scale=0.5, color=(255, 255, 255))


# ----------------------------
# Baseplate overlay
# ----------------------------
def draw_baseplate_overlay(vis, roi, center_rel, contour_rel):
    """
    Draws the actual live ROI used by detector plus detected baseplate contour.
    This is the most truthful view of what the app is using for PASS/TRACK.
    """
    if vis is None or roi is None:
        return

    try:
        x, y, w, h = map(int, roi)
    except Exception:
        return

    cv2.rectangle(
        vis,
        (x, y),
        (x + w, y + h),
        (255, 255, 0),
        2,
        lineType=cv2.LINE_AA,
    )
    _put_text(vis, "live ROI", (x + 4, y + 18), scale=0.5, color=(255, 255, 0))

    if contour_rel is not None:
        try:
            cnt = np.asarray(contour_rel, dtype=np.int32)

            if cnt.size > 0:
                cnt = cnt.reshape(-1, 1, 2)
                cnt_abs = cnt + np.array([[[x, y]]], dtype=np.int32)
                cv2.drawContours(vis, [cnt_abs], -1, (0, 255, 0), 2, lineType=cv2.LINE_AA)
        except Exception:
            pass

    if center_rel is not None:
        try:
            cx = int(round(x + float(center_rel[0])))
            cy = int(round(y + float(center_rel[1])))

            cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(vis, (cx, cy), 9, (255, 255, 255), 1, lineType=cv2.LINE_AA)
            _put_text(vis, "baseplate", (cx + 9, cy - 8), scale=0.45, color=(0, 0, 255))
        except Exception:
            pass


# ----------------------------
# Status box
# ----------------------------
def draw_status_box(vis, text, state="FAIL"):
    if vis is None:
        return

    if state == "PASS":
        color = (0, 200, 0)
    elif state == "TRACK":
        color = (0, 200, 200)
    elif state == "SEARCH":
        color = (180, 180, 180)
    else:
        color = (0, 0, 255)

    box_w = min(max(760, int(len(str(text)) * 18)), vis.shape[1] - 40)
    box_h = 100

    cv2.rectangle(vis, (20, 20), (20 + box_w, 20 + box_h), (20, 20, 20), -1)
    cv2.rectangle(vis, (20, 20), (20 + box_w, 20 + box_h), color, 2, lineType=cv2.LINE_AA)

    cv2.circle(vis, (48, 70), 10, color, -1, cv2.LINE_AA)

    _put_text(
        vis,
        text,
        (70, 80),
        scale=0.9,
        color=(255, 255, 255),
        thickness=2,
    )