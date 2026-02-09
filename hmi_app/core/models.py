from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np

@dataclass
class Recipe:
    name: str
    config_path: str
    golden_image_path: str
    cfg: Dict[str, Any]
    golden_bgr: np.ndarray

@dataclass
class EngineSettings:
    stab_every_n: int = 6
    search_padding_px: int = 120
    show_stab: bool = True
    show_baseplate: bool = True
    stable_need: int = 5

@dataclass
class QCFrameOutput:
    overlay_bgr: np.ndarray
    status_text: str
    state: str
    roi_live: Tuple[int,int,int,int]
    stab_info: Optional[Dict[str, Any]] = None
    center_rel: Optional[Tuple[float,float]] = None
    angle: Optional[float] = None
