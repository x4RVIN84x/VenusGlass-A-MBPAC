from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np


@dataclass
class Recipe:
    """
    Backwards compatible recipe container.

    Legacy layout:
      recipes/<recipe>/golden_config.json (and cfg contains golden_image_path)

    New layout (folder-per-config):
      recipes/<recipe>/
        active_config.txt  (optional; contains config name)
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

    # Overlay toggles
    show_stab: bool = True
    show_baseplate: bool = True

    # PASS stability gate
    stable_need: int = 5

    # View control for future UI pages
    # "RAW" | "PROC" | "OVERLAY"
    view_mode: str = "OVERLAY"

    # Only compute heavy debug/proc frames when needed
    compute_proc: bool = False


@dataclass
class QCFrameOutput:
    # Always available (Auto page uses this today)
    overlay_bgr: np.ndarray

    # Optional, for RAW/PROC/OVERLAY switching
    raw_bgr: Optional[np.ndarray] = None
    proc_bgr: Optional[np.ndarray] = None

    # Status + metrics
    status_text: str = ""
    state: str = "FAIL"
    roi_live: Tuple[int, int, int, int] = (0, 0, 1, 1)

    stab_info: Optional[Dict[str, Any]] = None
    center_rel: Optional[Tuple[float, float]] = None
    angle: Optional[float] = None

    # Extra debug (optional)
    detector_dbg: Optional[Dict[str, Any]] = None
