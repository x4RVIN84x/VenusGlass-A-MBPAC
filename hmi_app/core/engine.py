from __future__ import annotations

import time
from typing import Optional, Tuple, Any, Dict

import cv2
import numpy as np

import roi_stablizer
from detector import detect_baseplate

from hmi_app.core.models import Recipe, EngineSettings, QCFrameOutput
from hmi_app.core.overlay import draw_stab_debug, draw_baseplate_overlay, draw_status_box


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


def _to_bgr(gray_or_edges: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if gray_or_edges is None:
        return None

    if gray_or_edges.ndim == 2:
        return cv2.cvtColor(gray_or_edges, cv2.COLOR_GRAY2BGR)

    if gray_or_edges.ndim == 3 and gray_or_edges.shape[2] == 3:
        return gray_or_edges

    return None


def _compose_proc_full(
    raw_bgr: np.ndarray,
    roi_xywh: Tuple[int, int, int, int],
    proc_crop_bgr: Optional[np.ndarray],
) -> np.ndarray:
    """
    PROC view:
      - full frame grayscale background
      - ROI area replaced with processed crop
    """
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


class QCPreviewEngine:
    def __init__(self):
        self.recipe: Optional[Recipe] = None
        self.settings = EngineSettings()

        self._frame_i = 0
        self._roi_live: Optional[Tuple[int, int, int, int]] = None
        self._stab_info: Optional[Dict[str, Any]] = None
        self._stable_have = 0
        self._fps_t0 = time.time()
        self._fps = 0.0

        self._last_out: Optional[QCFrameOutput] = None

    def set_recipe(self, recipe: Recipe):
        self.recipe = recipe
        self._frame_i = 0
        self._roi_live = tuple(recipe.cfg["roi"])
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
        """
        Final recipe-level nudge.

        Useful after the stabilizer is stable but systematically biased.
        Example config:
          "roi_offset_x": 0,
          "roi_offset_y": -8
        """
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

        roi_cfg = tuple(cfg["roi"])

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

                # Pull from recipe cfg.
                canny_low=int(cfg.get("canny_low", 60)),
                canny_high=int(cfg.get("canny_high", 140)),

                # Solid-edge path is now the default.
                prefer_solid_edge=bool(cfg.get("prefer_solid_edge", True)),
                hysteresis_enabled=bool(cfg.get("hysteresis_enabled", True)),
            )

            self._stab_info = info

            if moved is not None and info and info.get("ok"):
                self._roi_live = self._apply_recipe_roi_offset(moved[0], cfg, W, H)
            else:
                self._roi_live = self._apply_recipe_roi_offset(roi_cfg, cfg, W, H)

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

        want_proc = bool(self.settings.compute_proc) or (str(self.settings.view_mode).upper() == "PROC")

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

        # ----------------------------
        # PROC view
        # ----------------------------
        proc_full = None

        if want_proc and isinstance(dbg, dict):
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
        # Compare to golden
        # ----------------------------
        golden_center = tuple(cfg.get("expected_center", (0.0, 0.0)))
        golden_angle = float(cfg.get("expected_angle", 0.0))
        tol = cfg.get("tolerance_px", {"x": 10, "y": 10, "angle": 5})

        state = "SEARCH"
        text = "SEARCHING…"

        if center_rel is None or angle is None:
            self._stable_have = max(0, self._stable_have - 1)
            state = "SEARCH"
            text = "SEARCHING…"
        else:
            sx = float(center_rel[0]) - float(golden_center[0])
            sy = float(center_rel[1]) - float(golden_center[1])
            stheta = float(angle) - float(golden_angle)

            dx = abs(sx)
            dy = abs(sy)
            dtheta = abs(stheta)

            ok_all = (dx <= tol["x"]) and (dy <= tol["y"]) and (dtheta <= tol["angle"])

            if ok_all:
                self._stable_have = min(int(self.settings.stable_need), self._stable_have + 1)
            else:
                self._stable_have = max(0, self._stable_have - 1)

            if ok_all and self._stable_have >= int(self.settings.stable_need):
                state = "PASS"
                text = f"PASS  dx={dx:.1f} dy={dy:.1f} dθ={dtheta:.1f}"
            else:
                state = "TRACK"
                text = f"TRACK {self._stable_have}/{int(self.settings.stable_need)}  dx={dx:.1f} dy={dy:.1f} dθ={dtheta:.1f}"

        # ----------------------------
        # FPS
        # ----------------------------
        self._frame_i += 1

        if self._frame_i % 15 == 0:
            dt = time.time() - self._fps_t0
            self._fps = 15.0 / max(1e-6, dt)
            self._fps_t0 = time.time()

        # ----------------------------
        # Overlay
        # ----------------------------
        overlay = raw.copy()

        if self.settings.show_stab:
            draw_stab_debug(overlay, self._stab_info, settings=self.settings)

        if self.settings.show_baseplate:
            draw_baseplate_overlay(overlay, roi_live, center_rel, contour_rel)

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