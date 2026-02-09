from __future__ import annotations
import time
import cv2
import numpy as np
from typing import Optional, Tuple, Any, Dict

from hmi_app.core.models import Recipe, EngineSettings, QCFrameOutput
from hmi_app.core.overlay import draw_stab_debug, draw_baseplate_overlay, draw_status_box

import roi_stablizer
from detector import detect_baseplate


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
    return img[y:y+h, x:x+w].copy(), roi


class QCPreviewEngine:
    def __init__(self):
        self.recipe: Optional[Recipe] = None
        self.settings = EngineSettings()
        self._frame_i = 0
        self._roi_live: Optional[Tuple[int,int,int,int]] = None
        self._stab_info: Optional[Dict[str, Any]] = None
        self._stable_have = 0
        self._fps_t0 = time.time()
        self._fps = 0.0

    def set_recipe(self, recipe: Recipe):
        self.recipe = recipe
        self._frame_i = 0
        self._roi_live = tuple(recipe.cfg["roi"])
        self._stab_info = None
        self._stable_have = 0

    def process_frame(self, frame_bgr: np.ndarray) -> QCFrameOutput:
        if self.recipe is None:
            return QCFrameOutput(
                overlay_bgr=frame_bgr,
                status_text="Status: NO RECIPE LOADED",
                state="FAIL",
                roi_live=(0,0,1,1),
            )

        cfg = self.recipe.cfg
        golden = self.recipe.golden_bgr
        Hg, Wg = golden.shape[:2]

        if frame_bgr.shape[:2] != (Hg, Wg):
            frame_bgr = cv2.resize(frame_bgr, (Wg, Hg), interpolation=cv2.INTER_LINEAR)

        H, W = frame_bgr.shape[:2]

        roi_cfg = tuple(cfg["roi"])
        if self._roi_live is None:
            self._roi_live = roi_cfg

        golden_lines = cfg.get("inner_border_lines", None)
        registration_roi_golden = tuple(cfg.get("registration_roi", cfg.get("roi", roi_cfg)))

        if golden_lines and self.settings.stab_every_n > 0 and (self._frame_i % self.settings.stab_every_n == 0):
            moved, info = roi_stablizer.stabilize_rois_using_saved_inner_border_lines(
                current_img=frame_bgr,
                golden_img=golden,
                registration_roi_golden=registration_roi_golden,
                golden_inner_lines_abs=golden_lines,
                rois_golden=[roi_cfg],
                search_padding_px=int(self.settings.search_padding_px),
                canny_low=60,
                canny_high=140,
            )
            self._stab_info = info
            if moved is not None and info and info.get("ok"):
                self._roi_live = clamp_roi(moved[0], W, H)
            else:
                self._roi_live = clamp_roi(roi_cfg, W, H)

        crop, roi_live = safe_crop(frame_bgr, self._roi_live)

        ret = detect_baseplate(
            crop,
            full_image_bgr=frame_bgr,
            roi_xywh_abs=roi_live,
            padding=150,
            shrink_border_px=10,
            canny_low=50,
            canny_high=120,
            area_min_frac=0.005,
            contrast_min=float(cfg.get("contrast_min", 6.0)),
            border_margin=12,
            return_debug=False,
        )

        center_rel, angle, contour_rel = ret[:3]

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
            dx = abs(float(center_rel[0]) - float(golden_center[0]))
            dy = abs(float(center_rel[1]) - float(golden_center[1]))
            dtheta = abs(float(angle) - float(golden_angle))

            ok_all = (dx <= tol["x"]) and (dy <= tol["y"]) and (dtheta <= tol["angle"])

            if ok_all:
                self._stable_have = min(self.settings.stable_need, self._stable_have + 1)
            else:
                self._stable_have = max(0, self._stable_have - 1)

            if ok_all and self._stable_have >= self.settings.stable_need:
                state = "PASS"
                text = f"PASS  dx={dx:.1f} dy={dy:.1f} dθ={dtheta:.1f}"
            else:
                state = "TRACK"
                text = f"TRACK {self._stable_have}/{self.settings.stable_need}  dx={dx:.1f} dy={dy:.1f} dθ={dtheta:.1f}"

        self._frame_i += 1
        if self._frame_i % 15 == 0:
            dt = time.time() - self._fps_t0
            self._fps = 15.0 / max(1e-6, dt)
            self._fps_t0 = time.time()

        overlay = frame_bgr.copy()
        if self.settings.show_stab:
            draw_stab_debug(overlay, self._stab_info)
        if self.settings.show_baseplate:
            draw_baseplate_overlay(overlay, roi_live, center_rel, contour_rel)
        draw_status_box(overlay, text, state=state)

        bottom = f"FPS: {self._fps:.1f} | Recipe: {self.recipe.name}"
        y = overlay.shape[0] - 18
        cv2.putText(overlay, bottom, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 6, cv2.LINE_AA)
        cv2.putText(overlay, bottom, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

        return QCFrameOutput(
            overlay_bgr=overlay,
            status_text=f"Status: {state} | {text}",
            state=state,
            roi_live=roi_live,
            stab_info=self._stab_info,
            center_rel=center_rel,
            angle=angle,
        )
