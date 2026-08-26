from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np


@dataclass
class Recipe:
    """
    Backwards compatible recipe container.

    Legacy layout:
      recipes/<recipe>/golden_config.json
      recipes/<recipe>/golden.png

    New layout:
      recipes/<recipe>/
        active_config.txt
        configs/<config_name>/
          golden_config.json
          golden.(png|jpg|jpeg)
    """
    name: str

    # Paths
    recipe_dir: str
    config_name: str
    config_dir: str
    config_path: str
    golden_image_path: str

    # Loaded config + golden image
    cfg: Dict[str, Any]
    golden_bgr: np.ndarray

    # Flags
    is_legacy: bool = False


@dataclass
class EngineSettings:
    # ROI stabilization cadence
    stab_every_n: int = 6
    search_padding_px: int = 120

    # Master overlay toggles
    show_stab: bool = True
    show_baseplate: bool = True

    # New overlay toggles used by Auto page
    show_search_roi: bool = True
    show_notch_contour: bool = True
    show_fitted_lines: bool = True
    show_new_anchors: bool = True
    show_raw_points: bool = False
    show_legacy_debug: bool = False
    show_stabilizer_text: bool = True

    # Auto Mode redraws operator guidance after display zoom so it stays on
    # screen exactly once. Other pages keep the engine-side HUD.
    defer_operator_hud: bool = False

    # Older/legacy overlay toggle names.
    # Keep these so old pages/modules do not crash.
    show_stab_search_roi: bool = True
    show_stab_feature_points: bool = False
    show_stab_anchors: bool = True
    show_stab_legacy: bool = False
    show_stab_text: bool = True

    # PASS stability gate
    stable_need: int = 5

    # View control
    # "RAW" | "PROC" | "OVERLAY"
    view_mode: str = "OVERLAY"

    # Only compute heavy debug/proc frames when needed
    compute_proc: bool = False


@dataclass
class QCFrameOutput:
    # Always available
    overlay_bgr: np.ndarray

    # Optional, for RAW/PROC/OVERLAY switching
    raw_bgr: Optional[np.ndarray] = None
    proc_bgr: Optional[np.ndarray] = None

    # Real Qt PROC dashboard payload.
    # Shape:
    # {
    #   "main": np.ndarray,
    #   "feeds": {
    #       "base_gray": {"title": "...", "image": np.ndarray, "help": "..."},
    #       ...
    #   },
    #   "stats": {...}
    # }
    proc_payload: Optional[Dict[str, Any]] = None

    # Status + metrics
    status_text: str = ""
    state: str = "FAIL"
    roi_live: Tuple[int, int, int, int] = (0, 0, 1, 1)
    captured_at: str = ""

    stab_info: Optional[Dict[str, Any]] = None
    center_rel: Optional[Tuple[float, float]] = None
    angle: Optional[float] = None

    # Extra debug
    detector_dbg: Optional[Dict[str, Any]] = None
