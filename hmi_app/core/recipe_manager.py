# hmi_app/core/recipe_manager.py (FULL REWRITE)
from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

import cv2

from hmi_app.core.models import Recipe


def _safe_load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_golden_image_in_dir(product_dir: str) -> Optional[str]:
    for name in ("golden.png", "golden.jpg", "golden.jpeg"):
        p = os.path.join(product_dir, name)
        if os.path.isfile(p):
            return p
    return None


class RecipeManager:
    """
    Flat product manager.

    Preferred layout:
      recipes/<PRODUCT_NAME>/
        golden_config.json
        golden.png

    Backwards compatibility:
      If old nested configs exist, they are listed as products using the name:
        <recipe>__<config>
      but new products are always flat folders.
    """

    def __init__(self, recipes_root: str = "recipes"):
        self.recipes_root = recipes_root

    # ----------------------------
    # Discovery
    # ----------------------------
    def list_recipes(self) -> List[str]:
        if not os.path.isdir(self.recipes_root):
            return []

        out: List[str] = []

        for name in sorted(os.listdir(self.recipes_root)):
            product_dir = os.path.join(self.recipes_root, name)
            if not os.path.isdir(product_dir):
                continue

            cfg = os.path.join(product_dir, "golden_config.json")
            if os.path.isfile(cfg):
                out.append(name)
                continue

            # Compatibility only: expose old configs as selectable product-like entries.
            configs_dir = os.path.join(product_dir, "configs")
            if os.path.isdir(configs_dir):
                for cfg_name in sorted(os.listdir(configs_dir)):
                    cfg_dir = os.path.join(configs_dir, cfg_name)
                    if not os.path.isdir(cfg_dir):
                        continue
                    if os.path.isfile(os.path.join(cfg_dir, "golden_config.json")):
                        out.append(f"{name}__{cfg_name}")

        return out

    # Compatibility stubs for older pages/code. Flat products do not have sub-configs.
    def list_configs(self, recipe_name: str) -> List[str]:
        return []

    def get_active_config_name(self, recipe_name: str) -> Optional[str]:
        return None

    def set_active_config_name(self, recipe_name: str, config_name: str) -> None:
        return None

    # ----------------------------
    # Loading
    # ----------------------------
    def _resolve_product_paths(self, product_name: str, config_name: Optional[str] = None) -> Tuple[str, str, str]:
        product_name = (product_name or "").strip()
        if not product_name:
            raise FileNotFoundError("No product name supplied")

        # Preferred flat path.
        product_dir = os.path.join(self.recipes_root, product_name)
        cfg_path = os.path.join(product_dir, "golden_config.json")
        if os.path.isfile(cfg_path):
            return product_name, product_dir, cfg_path

        # Compatibility: product__config maps to old nested config path.
        if "__" in product_name:
            base, cfg = product_name.split("__", 1)
            cfg_dir = os.path.join(self.recipes_root, base, "configs", cfg)
            cfg_path = os.path.join(cfg_dir, "golden_config.json")
            if os.path.isfile(cfg_path):
                return product_name, cfg_dir, cfg_path

        # Compatibility: explicit config_name for old nested path.
        if config_name:
            cfg_dir = os.path.join(self.recipes_root, product_name, "configs", config_name)
            cfg_path = os.path.join(cfg_dir, "golden_config.json")
            if os.path.isfile(cfg_path):
                return f"{product_name}__{config_name}", cfg_dir, cfg_path

        raise FileNotFoundError(f"Product folder/config not found: {product_dir}")

    def load(self, recipe_name: str, config_name: Optional[str] = None) -> Recipe:
        product_name, product_dir, cfg_path = self._resolve_product_paths(recipe_name, config_name)
        cfg = _safe_load_json(cfg_path)

        golden_path = cfg.get("golden_image_path")
        if golden_path:
            if not os.path.isabs(golden_path):
                golden_path = os.path.normpath(os.path.join(product_dir, golden_path))
        else:
            golden_path = _find_golden_image_in_dir(product_dir)
            if not golden_path:
                raise ValueError(f"golden_image_path missing and no golden image found in {product_dir}")

        golden = cv2.imread(golden_path)
        if golden is None:
            raise FileNotFoundError(f"Could not load golden image: {golden_path}")

        return Recipe(
            name=product_name,
            recipe_dir=product_dir,
            config_name=product_name,
            config_dir=product_dir,
            config_path=cfg_path,
            golden_image_path=golden_path,
            cfg=cfg,
            golden_bgr=golden,
            is_legacy=False,
        )
