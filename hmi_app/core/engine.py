from __future__ import annotations

import time
from typing import Optional, Tuple, Any, Dict

import cv2
import numpy as np

import roi_stablizer
from detector import detect_baseplate

from hmi_app.core.models import Recipe, EngineSettings, QCFrameOutput
from hmi_app.core.overlay import draw_stab_debug, draw_baseplate_overlay, draw_status_box


# ----------------------------
# Basic image / ROI helpers
# ----------------------------
def clamp_roi(roi, W, H):
    x, y, w, h = map(int, roi)

    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))

    return x, y, w, h


def safe_crop(img, roi):
    roi = clamp_roi(roi, img.shape[1], img.shape[0])
    x, y, w, h = roi
    return img[y:y + h, x:x + w].copy(), roi


def _to_bgr(img) -> Optional[np.ndarray]:
    if img is None:
        return None

    if isinstance(img, dict):
        return None

    try:
        arr = np.asarray(img)
    except Exception:
        return None

    if arr.size == 0:
        return None

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        mn = float(np.nanmin(arr))
        mx = float(np.nanmax(arr))

        if not np.isfinite(mn) or not np.isfinite(mx) or abs(mx - mn) < 1e-6:
            arr = np.zeros(arr.shape[:2], dtype=np.uint8)
        else:
            arr = (255.0 * (arr - mn) / (mx - mn)).clip(0, 255).astype(np.uint8)

    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr.copy()

    return None


def _angle_diff_deg(a, b):
    d = float(a) - float(b)

    while d > 180.0:
        d -= 360.0

    while d < -180.0:
        d += 360.0

    return float(d)


def _safe_float(v, default=None):
    try:
        x = float(v)
    except Exception:
        return default

    if not np.isfinite(x):
        return default

    return x


# ----------------------------
# Pixel/mm scale helpers
# ----------------------------
def _saved_px_per_mm(cfg: Dict[str, Any]):
    if not isinstance(cfg, dict):
        return None

    for key in (
        "baseplate_px_per_mm",
        "px_per_mm",
        "pixels_per_mm",
        "scale_px_per_mm",
    ):
        v = _safe_float(cfg.get(key), None)

        if v is not None and v > 0:
            return float(v)

    for key in (
        "baseplate_mm_per_px",
        "mm_per_px",
        "scale_mm_per_px",
    ):
        v = _safe_float(cfg.get(key), None)

        if v is not None and v > 0:
            return float(1.0 / v)

    scale = cfg.get("baseplate_scale")

    if isinstance(scale, dict):
        v = _safe_float(scale.get("px_per_mm"), None)

        if v is not None and v > 0:
            return float(v)

        v = _safe_float(scale.get("mm_per_px"), None)

        if v is not None and v > 0:
            return float(1.0 / v)

    return None


def _estimate_px_per_mm_from_contour(contour_rel, cfg: Dict[str, Any]):
    if contour_rel is None or not isinstance(cfg, dict):
        return None

    w_mm = _safe_float(cfg.get("baseplate_width_mm"), None)
    h_mm = _safe_float(cfg.get("baseplate_height_mm"), None)

    if w_mm is None or h_mm is None or w_mm <= 0 or h_mm <= 0:
        return None

    try:
        cnt = np.asarray(contour_rel, dtype=np.float32).reshape(-1, 2)
    except Exception:
        return None

    if len(cnt) < 5:
        return None

    try:
        rect = cv2.minAreaRect(cnt)
        (_cx, _cy), (rw, rh), _a = rect
    except Exception:
        return None

    rw = float(rw)
    rh = float(rh)

    if rw <= 1 or rh <= 1:
        return None

    s1a = rw / w_mm
    s1b = rh / h_mm
    err1 = abs(s1a - s1b)
    s1 = 0.5 * (s1a + s1b)

    s2a = rw / h_mm
    s2b = rh / w_mm
    err2 = abs(s2a - s2b)
    s2 = 0.5 * (s2a + s2b)

    scale = s1 if err1 <= err2 else s2

    if not np.isfinite(scale) or scale <= 0:
        return None

    return {
        "px_per_mm": float(scale),
        "source": "current_baseplate_contour",
        "rect_size_px": (rw, rh),
        "baseplate_size_mm": (float(w_mm), float(h_mm)),
    }


def _px_to_display(dx_px, dy_px, cfg, frame_scale_info=None):
    px_per_mm = _saved_px_per_mm(cfg)

    if px_per_mm is None and isinstance(frame_scale_info, dict):
        px_per_mm = _safe_float(frame_scale_info.get("px_per_mm"), None)

    if px_per_mm is not None and px_per_mm > 0:
        return {
            "unit": "mm",
            "px_per_mm": float(px_per_mm),
            "dx": float(dx_px) / float(px_per_mm),
            "dy": float(dy_px) / float(px_per_mm),
        }

    return {
        "unit": "px",
        "px_per_mm": None,
        "dx": float(dx_px),
        "dy": float(dy_px),
    }


# ----------------------------
# Notch-frame coordinate helpers
# ----------------------------
def _point_to_notch_frame(pt, notch_frame: Dict[str, Any]):
    if pt is None or not isinstance(notch_frame, dict):
        return None

    try:
        p = np.array(pt, dtype=np.float64).reshape(2)
        o = np.array(notch_frame["origin"], dtype=np.float64).reshape(2)
        u = np.array(notch_frame["x_axis"], dtype=np.float64).reshape(2)
        v = np.array(notch_frame["y_axis"], dtype=np.float64).reshape(2)
    except Exception:
        return None

    d = p - o

    return {
        "dx": float(np.dot(d, u)),
        "dy": float(np.dot(d, v)),
    }


def _point_from_notch_frame(notch_frame: Dict[str, Any], dx, dy):
    if not isinstance(notch_frame, dict):
        return None

    try:
        o = np.array(notch_frame["origin"], dtype=np.float64).reshape(2)
        u = np.array(notch_frame["x_axis"], dtype=np.float64).reshape(2)
        v = np.array(notch_frame["y_axis"], dtype=np.float64).reshape(2)
        p = o + u * float(dx) + v * float(dy)
    except Exception:
        return None

    if not np.isfinite(p).all():
        return None

    return float(p[0]), float(p[1])


def _expected_notch_frame_values(cfg: Dict[str, Any]):
    if not isinstance(cfg, dict):
        return None

    nf = cfg.get("expected_notch_frame")

    if not isinstance(nf, dict):
        return None

    dx = nf.get("frame_dx", nf.get("dx", nf.get("sidewall_dx", None)))
    dy = nf.get("frame_dy", nf.get("dy", nf.get("sidewall_dy", None)))

    dx = _safe_float(dx, None)
    dy = _safe_float(dy, None)

    if dx is None or dy is None:
        return None

    rel_angle = _safe_float(nf.get("relative_angle"), None)

    return {
        "dx": float(dx),
        "dy": float(dy),
        "relative_angle": rel_angle,
        "raw": nf,
    }


def _expected_center_from_notch_frame(cfg: Dict[str, Any], notch_frame: Dict[str, Any]):
    exp = _expected_notch_frame_values(cfg)

    if exp is None:
        return None, None

    pt = _point_from_notch_frame(notch_frame, exp["dx"], exp["dy"])

    if pt is None:
        return None, None

    return pt, {
        "expected_dx": float(exp["dx"]),
        "expected_dy": float(exp["dy"]),
        "relative_angle": exp.get("relative_angle"),
    }


def _line_angle_from_notch_frame(notch_frame):
    if not isinstance(notch_frame, dict):
        return 0.0

    a = _safe_float(notch_frame.get("angle_deg"), None)

    if a is not None:
        return float(a)

    try:
        u = np.array(notch_frame["x_axis"], dtype=np.float64).reshape(2)
        return float(np.degrees(np.arctan2(u[1], u[0])))
    except Exception:
        return 0.0


def _extract_notch_frame(stab_info):
    if not isinstance(stab_info, dict):
        return None

    for key in ("notch_frame", "current_notch_frame", "notch_frame_runtime"):
        nf = stab_info.get(key)

        if isinstance(nf, dict):
            if "origin" in nf and "x_axis" in nf and "y_axis" in nf:
                return nf

    return None


# ----------------------------
# Live rotated ROI from notch-frame
# ----------------------------
def _rotated_rect_poly(center_xy, width_px, height_px, angle_deg):
    cx, cy = map(float, center_xy)
    w = max(2.0, float(width_px))
    h = max(2.0, float(height_px))

    a = np.deg2rad(float(angle_deg))

    ux = np.array([np.cos(a), np.sin(a)], dtype=np.float64)
    uy = np.array([np.sin(a), -np.cos(a)], dtype=np.float64)

    c = np.array([cx, cy], dtype=np.float64)
    hw = 0.5 * w
    hh = 0.5 * h

    pts = np.array(
        [
            c - ux * hw - uy * hh,
            c + ux * hw - uy * hh,
            c + ux * hw + uy * hh,
            c - ux * hw + uy * hh,
        ],
        dtype=np.float32,
    )

    return pts


def _bbox_from_poly(poly, W, H, pad=8):
    if poly is None:
        return None

    try:
        pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    except Exception:
        return None

    if pts.size == 0 or not np.isfinite(pts).all():
        return None

    p = int(max(0, pad))

    x0 = int(np.floor(np.min(pts[:, 0]))) - p
    y0 = int(np.floor(np.min(pts[:, 1]))) - p
    x1 = int(np.ceil(np.max(pts[:, 0]))) + p
    y1 = int(np.ceil(np.max(pts[:, 1]))) + p

    x0 = max(0, min(x0, int(W) - 1))
    y0 = max(0, min(y0, int(H) - 1))
    x1 = max(x0 + 1, min(x1, int(W)))
    y1 = max(y0 + 1, min(y1, int(H)))

    return int(x0), int(y0), int(x1 - x0), int(y1 - y0)


def _build_live_roi_from_notch_frame(cfg, notch_frame, W, H):
    """
    Build LIVE baseplate ROI directly from the detected bottom-notch frame.

    This avoids depending on the stabilizer affine bbox if that bbox becomes stale.
    """
    if not isinstance(cfg, dict) or not isinstance(notch_frame, dict):
        return None, None, "missing_cfg_or_notch_frame", None

    expected_nf = cfg.get("expected_notch_frame")

    if not isinstance(expected_nf, dict):
        return None, None, "missing_expected_notch_frame", None

    exp_dx = _safe_float(
        expected_nf.get("frame_dx", expected_nf.get("dx", expected_nf.get("sidewall_dx", None))),
        None,
    )
    exp_dy = _safe_float(
        expected_nf.get("frame_dy", expected_nf.get("dy", expected_nf.get("sidewall_dy", None))),
        None,
    )

    if exp_dx is None or exp_dy is None:
        return None, None, "bad_expected_notch_frame_dxdy", None

    center = _point_from_notch_frame(notch_frame, exp_dx, exp_dy)

    if center is None:
        return None, None, "notch_frame_center_failed", None

    roi_size_src = cfg.get("baseplate_roi", cfg.get("roi", (0, 0, 140, 180)))

    try:
        _rx, _ry, rw, rh = map(float, roi_size_src)
    except Exception:
        rw, rh = 140.0, 180.0

    roi_scale = float(cfg.get("notch_frame_roi_scale", 1.0))

    extra_pad = float(cfg.get("baseplate_roi_extra_pad", 8.0))
    extra_w = float(cfg.get("notch_frame_roi_extra_w", extra_pad)) * 2.0
    extra_h = float(cfg.get("notch_frame_roi_extra_h", extra_pad)) * 2.0

    live_w = max(20.0, rw * roi_scale + extra_w)
    live_h = max(20.0, rh * roi_scale + extra_h)

    notch_angle = _safe_float(notch_frame.get("angle_deg"), 0.0)

    poly = _rotated_rect_poly(center, live_w, live_h, notch_angle)

    bbox = _bbox_from_poly(
        poly,
        W,
        H,
        pad=int(cfg.get("rotated_roi_bbox_pad", 8)),
    )

    if bbox is None:
        return None, None, "rotated_roi_bbox_failed", None

    frame_debug = {
        "roi_center_abs": [float(center[0]), float(center[1])],
        "expected_dx": float(exp_dx),
        "expected_dy": float(exp_dy),
        "roi_size_src": list(map(float, roi_size_src)),
        "live_w": float(live_w),
        "live_h": float(live_h),
        "notch_angle": float(notch_angle),
    }

    return bbox, poly, "expected_notch_frame_rotated_roi", frame_debug


# ----------------------------
# PROC diagnostic helpers
# ----------------------------
def _compose_proc_full(raw_bgr: np.ndarray, roi_xywh: Tuple[int, int, int, int], proc_crop_bgr: Optional[np.ndarray]) -> np.ndarray:
    H, W = raw_bgr.shape[:2]
    bg = cv2.cvtColor(cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    if proc_crop_bgr is None:
        return bg

    x, y, w, h = clamp_roi(roi_xywh, W, H)

    if w <= 1 or h <= 1:
        return bg

    proc_resized = cv2.resize(proc_crop_bgr, (w, h), interpolation=cv2.INTER_NEAREST)
    bg[y:y + h, x:x + w] = proc_resized

    return bg


def _build_proc_diagnostic_view(raw, roi_live, roi_poly, center_abs, contour_abs, stab_info, dbg, status_text):
    if raw is None:
        return None, None

    H, W = raw.shape[:2]

    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    heat = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)

    grid = heat.copy()

    step = max(40, int(round(min(W, H) / 18)))

    for x in range(0, W, step):
        cv2.line(grid, (x, 0), (x, H), (25, 25, 25), 1, cv2.LINE_AA)

    for y in range(0, H, step):
        cv2.line(grid, (0, y), (W, y), (25, 25, 25), 1, cv2.LINE_AA)

    main = cv2.addWeighted(heat, 0.78, grid, 0.22, 0)

    if roi_poly is not None:
        try:
            poly = np.asarray(roi_poly, dtype=np.int32).reshape(-1, 1, 2)

            if len(poly) >= 3:
                cv2.polylines(main, [poly], True, (255, 255, 0), 2, cv2.LINE_AA)
        except Exception:
            pass
    else:
        try:
            x, y, w, h = map(int, roi_live)
            cv2.rectangle(main, (x, y), (x + w, y + h), (255, 255, 0), 2, cv2.LINE_AA)
        except Exception:
            pass

    if isinstance(stab_info, dict):
        nf = stab_info.get("notch_frame")

        if isinstance(nf, dict):
            anchors = nf.get("anchors") or {}

            for name, color in (
                ("bottom_left", (0, 200, 255)),
                ("bottom_right", (0, 200, 255)),
                ("bottom_mid", (0, 0, 255)),
            ):
                p = anchors.get(name)

                try:
                    px, py = int(round(float(p[0]))), int(round(float(p[1])))
                    cv2.circle(main, (px, py), 7, color, -1, cv2.LINE_AA)
                    cv2.putText(main, name, (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                except Exception:
                    pass

    if center_abs is not None:
        try:
            cx, cy = int(round(float(center_abs[0]))), int(round(float(center_abs[1])))
            cv2.circle(main, (cx, cy), 8, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(main, (cx, cy), 13, (255, 255, 255), 1, cv2.LINE_AA)
        except Exception:
            pass

    if contour_abs is not None:
        try:
            cnt = np.asarray(contour_abs, dtype=np.int32).reshape(-1, 1, 2)
            cv2.drawContours(main, [cnt], -1, (0, 255, 0), 2, cv2.LINE_AA)
        except Exception:
            pass

    cv2.rectangle(main, (14, 14), (min(W - 16, 760), 88), (0, 0, 0), -1)
    cv2.rectangle(main, (14, 14), (min(W - 16, 760), 88), (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(main, "PROC DIAGNOSTIC VIEW", (28, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(main, str(status_text), (28, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1, cv2.LINE_AA)

    feeds = {}

    if isinstance(stab_info, dict):
        active_dbg = stab_info.get("solid_edge_debug") or stab_info.get("legacy_lines_debug") or {}

        if isinstance(active_dbg, dict):
            dbg_frames = active_dbg.get("dbg")

            if isinstance(dbg_frames, dict):
                for key, title, help_text in (
                    ("gray0", "Raw grayscale", "The camera image converted to brightness only."),
                    ("gray1_contrast", "Contrast image", "Enhanced contrast used before edge detection."),
                    ("gray2_blur", "Blurred image", "Slight smoothing to reduce noise."),
                    ("edges", "Solid-edge mask", "White pixels are the detected notch/glass edge candidates."),
                ):
                    if key in dbg_frames:
                        feeds[key] = {
                            "title": title,
                            "image": dbg_frames[key],
                            "help": help_text,
                        }

    if isinstance(dbg, dict):
        dframes = dbg.get("dbg")

        if isinstance(dframes, dict):
            for key, title in (
                ("gray2_contrast", "Baseplate contrast"),
                ("edges2_close", "Baseplate closed edges"),
            ):
                if key in dframes:
                    feeds[f"base_{key}"] = {
                        "title": title,
                        "image": dframes[key],
                        "help": "Baseplate detector intermediate view.",
                    }

    payload = {
        "main": main,
        "feeds": feeds,
        "stats": {
            "status": status_text,
        },
    }

    return main, payload


# ----------------------------
# Engine
# ----------------------------
class QCPreviewEngine:
    def __init__(self):
        self.recipe: Optional[Recipe] = None
        self.settings = EngineSettings()

        self._frame_i = 0
        self._roi_live: Optional[Tuple[int, int, int, int]] = None
        self._roi_live_poly = None
        self._stab_info: Optional[Dict[str, Any]] = None

        self._stable_have = 0
        self._fps_t0 = time.time()
        self._fps = 0.0

        self._last_out: Optional[QCFrameOutput] = None

    def set_recipe(self, recipe: Recipe):
        self.recipe = recipe
        self._frame_i = 0

        # Use actual baseplate ROI if present.
        self._roi_live = tuple(recipe.cfg.get("baseplate_roi", recipe.cfg.get("roi", (0, 0, 1, 1))))

        self._roi_live_poly = None
        self._stab_info = None
        self._stable_have = 0
        self._last_out = None

        try:
            roi_stablizer.reset_default_contour_hysteresis()
        except Exception:
            pass

    def get_raw_frame(self) -> Optional[np.ndarray]:
        return None if self._last_out is None else self._last_out.raw_bgr

    def get_processed_frame(self) -> Optional[np.ndarray]:
        return None if self._last_out is None else self._last_out.proc_bgr

    def get_overlay_frame(self) -> Optional[np.ndarray]:
        return None if self._last_out is None else self._last_out.overlay_bgr

    def _apply_recipe_roi_offset(self, roi, cfg, W, H):
        x, y, w, h = map(int, roi)

        x += int(cfg.get("roi_offset_x", 0))
        y += int(cfg.get("roi_offset_y", 0))

        return clamp_roi((x, y, w, h), W, H)

    def process_frame(self, frame_bgr: np.ndarray) -> QCFrameOutput:
        if self.recipe is None:
            out = QCFrameOutput(
                overlay_bgr=frame_bgr,
                raw_bgr=frame_bgr,
                proc_bgr=None,
                proc_payload=None,
                status_text="Status: NO RECIPE LOADED",
                state="FAIL",
                roi_live=(0, 0, 1, 1),
            )
            self._last_out = out
            return out

        cfg = self.recipe.cfg
        golden = self.recipe.golden_bgr
        Hg, Wg = golden.shape[:2]

        raw = frame_bgr

        if raw.shape[:2] != (Hg, Wg):
            raw = cv2.resize(raw, (Wg, Hg), interpolation=cv2.INTER_LINEAR)

        H, W = raw.shape[:2]

        # IMPORTANT:
        # baseplate_roi is the actual metal-part detector ROI.
        # roi is often the larger glass / registration ROI in older product JSON.
        roi_cfg_raw = cfg.get("baseplate_roi", cfg.get("roi", (0, 0, 1, 1)))
        roi_cfg = tuple(roi_cfg_raw)

        if self._roi_live is None:
            self._roi_live = roi_cfg

        golden_lines = cfg.get("inner_border_lines", None)
        registration_roi_golden = tuple(cfg.get("registration_roi", cfg.get("roi", roi_cfg)))

        # ----------------------------
        # ROI stabilizer
        # ----------------------------
        if (
            golden_lines
            and self.settings.stab_every_n > 0
            and (self._frame_i % int(self.settings.stab_every_n) == 0)
        ):
            moved, info = roi_stablizer.stabilize_rois_using_saved_inner_border_lines(
                current_img=raw,
                golden_img=golden,
                registration_roi_golden=registration_roi_golden,
                golden_inner_lines_abs=golden_lines,
                rois_golden=[roi_cfg],
                search_padding_px=int(self.settings.search_padding_px),

                canny_low=int(cfg.get("canny_low", 60)),
                canny_high=int(cfg.get("canny_high", 140)),

                prefer_dark_region=bool(cfg.get("prefer_dark_region", True)),
                prefer_solid_edge=bool(cfg.get("prefer_solid_edge", True)),

                notch_blur_ksize=int(cfg.get("notch_blur_ksize", 5)),
                notch_close_ksize=int(cfg.get("notch_close_ksize", 2)),
                notch_open_ksize=int(cfg.get("notch_open_ksize", 0)),
                notch_threshold_bias=float(cfg.get("notch_threshold_bias", 0.0)),
                notch_bottom_band_frac=float(cfg.get("notch_bottom_band_frac", 0.35)),
                notch_side_band_frac=float(cfg.get("notch_side_band_frac", 0.35)),

                hysteresis_enabled=bool(cfg.get("hysteresis_enabled", True)),
            )

            self._stab_info = info if isinstance(info, dict) else {}

            notch_frame = _extract_notch_frame(self._stab_info)

            bbox = None
            poly = None
            roi_mode = "fallback"
            roi_frame_debug = None

            if notch_frame is not None:
                bbox, poly, roi_mode, roi_frame_debug = _build_live_roi_from_notch_frame(
                    cfg,
                    notch_frame,
                    W,
                    H,
                )

            # Best path: direct ROI from live notch frame.
            if bbox is not None:
                self._roi_live = self._apply_recipe_roi_offset(bbox, cfg, W, H)
                self._roi_live_poly = poly

            # Fallback: stabilizer affine ROI.
            elif moved is not None and self._stab_info.get("ok"):
                self._roi_live = self._apply_recipe_roi_offset(moved[0], cfg, W, H)
                self._roi_live_poly = self._stab_info.get("roi_poly_current")
                roi_mode = "fallback_stabilized_bbox"

            # Final fallback: static baseplate ROI.
            else:
                self._roi_live = self._apply_recipe_roi_offset(roi_cfg, cfg, W, H)
                self._roi_live_poly = None
                roi_mode = "fallback_recipe_baseplate_roi"

            if isinstance(self._stab_info, dict):
                self._stab_info["notch_frame_runtime"] = notch_frame
                self._stab_info["roi_mode"] = roi_mode
                self._stab_info["roi_frame_debug"] = roi_frame_debug
                self._stab_info["live_roi"] = self._roi_live
                self._stab_info["live_roi_poly"] = (
                    None if self._roi_live_poly is None
                    else np.asarray(self._roi_live_poly).tolist()
                )

        crop, roi_live = safe_crop(raw, self._roi_live)

        # ----------------------------
        # Detector tuning
        # ----------------------------
        det_kwargs = dict(
            shrink_border_px=int(cfg.get("shrink_border_px", 10)),
            border_margin=int(cfg.get("border_margin", 12)),
            use_clahe=bool(cfg.get("use_clahe", True)),
            clahe_clip=float(cfg.get("clahe_clip", 1.5)),
            clahe_grid=int(cfg.get("clahe_grid", 8)),
            blur_ksize=int(cfg.get("blur_ksize", 5)),
            canny_low=int(cfg.get("canny_low", 50)),
            canny_high=int(cfg.get("canny_high", 120)),
            dilate_iter=int(cfg.get("dilate_iter", 1)),
            close_iter=int(cfg.get("close_iter", 1)),
            area_min_frac=float(cfg.get("area_min_frac", 0.005)),
            area_max_frac=float(cfg.get("area_max_frac", 0.60)),
            aspect_min=float(cfg.get("aspect_min", 0.5)),
            aspect_max=float(cfg.get("aspect_max", 2.2)),
            solidity_min=float(cfg.get("solidity_min", 0.7)),
            extent_min=float(cfg.get("extent_min", 0.25)),
            contrast_min=float(cfg.get("contrast_min", 6.0)),
        )

        want_proc = (
            bool(getattr(self.settings, "compute_proc", False))
            or str(getattr(self.settings, "view_mode", "OVERLAY")).upper() == "PROC"
        )

        if want_proc:
            center_rel, angle, contour_rel, dbg = detect_baseplate(
                crop,
                full_image_bgr=raw,
                roi_xywh_abs=roi_live,
                padding=int(cfg.get("padding", 150)),
                return_debug=True,
                **det_kwargs,
            )
        else:
            ret = detect_baseplate(
                crop,
                full_image_bgr=raw,
                roi_xywh_abs=roi_live,
                padding=int(cfg.get("padding", 150)),
                return_debug=False,
                **det_kwargs,
            )
            center_rel, angle, contour_rel = ret[:3]
            dbg = None

        center_abs = None
        contour_abs = None

        if center_rel is not None:
            center_abs = (
                float(roi_live[0] + center_rel[0]),
                float(roi_live[1] + center_rel[1]),
            )

        if contour_rel is not None:
            try:
                contour_abs = np.asarray(contour_rel, dtype=np.int32) + np.array([[roi_live[0], roi_live[1]]], dtype=np.int32)
            except Exception:
                contour_abs = None

        frame_scale_info = _estimate_px_per_mm_from_contour(contour_rel, cfg)

        if isinstance(self._stab_info, dict):
            saved_scale = _saved_px_per_mm(cfg)

            if saved_scale is not None:
                self._stab_info["baseplate_px_per_mm_saved"] = float(saved_scale)

            if isinstance(frame_scale_info, dict):
                self._stab_info["baseplate_px_per_mm_frame"] = float(frame_scale_info.get("px_per_mm"))
                self._stab_info["baseplate_px_per_mm_frame_debug"] = frame_scale_info

        # ----------------------------
        # Measurement
        # ----------------------------
        golden_center = tuple(cfg.get("expected_center", (0.0, 0.0)))
        golden_angle = float(cfg.get("expected_angle", 0.0))
        tol = cfg.get("tolerance_px", {"x": 10, "y": 10, "angle": 5})

        tol_x = float(tol.get("x", 10))
        tol_y = float(tol.get("y", 10))
        tol_a = float(tol.get("angle", 5))

        state = "SEARCH"
        text = "SEARCHING"

        sx = None
        sy = None
        stheta = None

        dx_px = None
        dy_px = None
        dtheta = None

        rel_theta = None
        expected_abs = None
        expected_dbg = None

        notch_frame = None

        if isinstance(self._stab_info, dict):
            notch_frame = self._stab_info.get("notch_frame") or self._stab_info.get("current_notch_frame")

        exp_nf = _expected_notch_frame_values(cfg)

        if center_abs is None or angle is None:
            self._stable_have = max(0, self._stable_have - 1)
            state = "SEARCH"
            text = "SEARCHING"

        else:
            if isinstance(notch_frame, dict) and exp_nf is not None:
                current_nf_measure = _point_to_notch_frame(center_abs, notch_frame)

                if current_nf_measure is not None:
                    sx = float(current_nf_measure["dx"]) - float(exp_nf["dx"])
                    sy = float(current_nf_measure["dy"]) - float(exp_nf["dy"])

                    dx_px = abs(float(sx))
                    dy_px = abs(float(sy))

                    notch_angle = _line_angle_from_notch_frame(notch_frame)
                    rel_theta = _angle_diff_deg(float(angle), notch_angle)

                    exp_rel_a = exp_nf.get("relative_angle")

                    if exp_rel_a is None:
                        exp_rel_a = _angle_diff_deg(golden_angle, float(exp_nf["raw"].get("notch_angle", notch_angle)))

                    stheta = _angle_diff_deg(float(rel_theta), float(exp_rel_a))
                    dtheta = abs(float(stheta))

                    expected_abs, expected_dbg = _expected_center_from_notch_frame(cfg, notch_frame)

                else:
                    sx = float(center_rel[0]) - float(golden_center[0])
                    sy = float(center_rel[1]) - float(golden_center[1])
                    stheta = _angle_diff_deg(float(angle), golden_angle)

                    dx_px = abs(float(sx))
                    dy_px = abs(float(sy))
                    dtheta = abs(float(stheta))

            else:
                sx = float(center_rel[0]) - float(golden_center[0])
                sy = float(center_rel[1]) - float(golden_center[1])
                stheta = _angle_diff_deg(float(angle), golden_angle)

                dx_px = abs(float(sx))
                dy_px = abs(float(sy))
                dtheta = abs(float(stheta))

            ok_all = (dx_px <= tol_x) and (dy_px <= tol_y) and (dtheta <= tol_a)

            if ok_all:
                self._stable_have = min(int(self.settings.stable_need), self._stable_have + 1)
            else:
                self._stable_have = max(0, self._stable_have - 1)

            disp_abs = _px_to_display(dx_px, dy_px, cfg, frame_scale_info)
            unit = disp_abs["unit"]

            if ok_all and self._stable_have >= int(self.settings.stable_need):
                state = "PASS"
                text = f"PASS  dx={disp_abs['dx']:.2f}{unit} dy={disp_abs['dy']:.2f}{unit} dTheta={dtheta:.1f}"
            else:
                state = "TRACK"
                text = f"TRACK {self._stable_have}/{int(self.settings.stable_need)}  dx={disp_abs['dx']:.2f}{unit} dy={disp_abs['dy']:.2f}{unit} dTheta={dtheta:.1f}"

        # ----------------------------
        # Store operator guidance in stab_info
        # ----------------------------
        if not isinstance(self._stab_info, dict):
            self._stab_info = {}

        if center_abs is not None:
            self._stab_info["current_baseplate_center_abs"] = [
                float(center_abs[0]),
                float(center_abs[1]),
            ]
        else:
            self._stab_info.pop("current_baseplate_center_abs", None)

        if sx is not None and sy is not None:
            disp_signed = _px_to_display(float(sx), float(sy), cfg, frame_scale_info)

            self._stab_info["current_offset_px"] = {
                "dx": float(sx),
                "dy": float(sy),
            }
            self._stab_info["current_offset_display"] = disp_signed

            self._stab_info["current_notch_measure"] = {
                "dx": float(sx),
                "dy": float(sy),
                "relative_angle": None if rel_theta is None else float(rel_theta),
                "dtheta": None if stheta is None else float(stheta),
            }

            if disp_signed.get("unit") == "mm":
                self._stab_info["current_notch_measure_mm"] = {
                    "dx": float(disp_signed["dx"]),
                    "dy": float(disp_signed["dy"]),
                    "unit": "mm",
                }
            else:
                self._stab_info.pop("current_notch_measure_mm", None)

        if expected_abs is not None:
            self._stab_info["expected_baseplate_center_abs"] = [
                float(expected_abs[0]),
                float(expected_abs[1]),
            ]
            self._stab_info["expected_baseplate_center_debug"] = expected_dbg

            if center_abs is not None:
                screen_dx_px = float(expected_abs[0] - center_abs[0])
                screen_dy_px = float(expected_abs[1] - center_abs[1])
                screen_dist_px = float(np.hypot(screen_dx_px, screen_dy_px))

                px_per_mm = _saved_px_per_mm(cfg)

                if px_per_mm is None and isinstance(frame_scale_info, dict):
                    px_per_mm = _safe_float(frame_scale_info.get("px_per_mm"), None)

                correction = {
                    "screen_dx_px": float(screen_dx_px),
                    "screen_dy_px": float(screen_dy_px),
                    "screen_dist_px": float(screen_dist_px),

                    "local_current_minus_expected_dx_px": None if sx is None else float(sx),
                    "local_current_minus_expected_dy_px": None if sy is None else float(sy),
                    "local_correction_dx_px": None if sx is None else float(-sx),
                    "local_correction_dy_px": None if sy is None else float(-sy),
                }

                if px_per_mm is not None and px_per_mm > 0:
                    correction.update(
                        {
                            "unit": "mm",
                            "px_per_mm": float(px_per_mm),
                            "screen_dx": float(screen_dx_px / px_per_mm),
                            "screen_dy": float(screen_dy_px / px_per_mm),
                            "screen_dist": float(screen_dist_px / px_per_mm),
                            "local_correction_dx": None if sx is None else float((-sx) / px_per_mm),
                            "local_correction_dy": None if sy is None else float((-sy) / px_per_mm),
                        }
                    )
                else:
                    correction.update(
                        {
                            "unit": "px",
                            "px_per_mm": None,
                            "screen_dx": float(screen_dx_px),
                            "screen_dy": float(screen_dy_px),
                            "screen_dist": float(screen_dist_px),
                            "local_correction_dx": None if sx is None else float(-sx),
                            "local_correction_dy": None if sy is None else float(-sy),
                        }
                    )

                self._stab_info["baseplate_correction_vector"] = correction
        else:
            self._stab_info.pop("expected_baseplate_center_abs", None)
            self._stab_info.pop("expected_baseplate_center_debug", None)
            self._stab_info.pop("baseplate_correction_vector", None)

        # ----------------------------
        # FPS
        # ----------------------------
        self._frame_i += 1

        if self._frame_i % 15 == 0:
            dt = time.time() - self._fps_t0
            self._fps = 15.0 / max(1e-6, dt)
            self._fps_t0 = time.time()

        # ----------------------------
        # PROC view
        # ----------------------------
        proc_full = None
        proc_payload = None

        if want_proc:
            proc_full, proc_payload = _build_proc_diagnostic_view(
                raw=raw,
                roi_live=roi_live,
                roi_poly=self._roi_live_poly,
                center_abs=center_abs,
                contour_abs=contour_abs,
                stab_info=self._stab_info,
                dbg=dbg,
                status_text=text,
            )

            if proc_full is None and isinstance(dbg, dict):
                dbg_frames = (dbg.get("dbg") or {}) if isinstance(dbg.get("dbg"), dict) else {}

                proc_candidate = (
                    dbg_frames.get("edges2_close")
                    or dbg_frames.get("edges1_dilate")
                    or dbg_frames.get("edges0")
                    or dbg_frames.get("gray2_contrast")
                    or dbg_frames.get("gray3_blur")
                )

                proc_crop_bgr = _to_bgr(proc_candidate)
                proc_full = _compose_proc_full(raw, roi_live, proc_crop_bgr)

        # ----------------------------
        # Overlay
        # ----------------------------
        overlay = raw.copy()

        if self.settings.show_stab:
            draw_stab_debug(overlay, self._stab_info, settings=self.settings)

        if self.settings.show_baseplate:
            draw_baseplate_overlay(
                overlay,
                roi_live,
                center_rel,
                contour_rel,
                roi_poly=self._roi_live_poly,
            )

        draw_status_box(overlay, text, state=state)

        bottom = (
            f"FPS: {self._fps:.1f} | "
            f"Recipe: {self.recipe.name} | "
            f"Config: {getattr(self.recipe, 'config_name', 'legacy')}"
        )

        y = overlay.shape[0] - 18

        cv2.putText(overlay, bottom, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(overlay, bottom, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        out = QCFrameOutput(
            overlay_bgr=overlay,
            raw_bgr=raw,
            proc_bgr=proc_full,
            proc_payload=proc_payload,
            status_text=f"Status: {state} | {text}",
            state=state,
            roi_live=roi_live,
            stab_info=self._stab_info,
            center_rel=center_rel,
            angle=angle,
            detector_dbg=dbg if want_proc else None,
        )

        self._last_out = out
        return out