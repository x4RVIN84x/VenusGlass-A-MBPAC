from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

import cv2

from hmi_app.core.models import Recipe


def _safe_load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_golden_image_in_dir(recipe_dir: str) -> Optional[str]:
    for name in ("golden.png", "golden.jpg", "golden.jpeg"):
        p = os.path.join(recipe_dir, name)
        if os.path.isfile(p):
            return p
    return None


class RecipeManager:
    """
    Flat product/recipe manager.

    One folder == one complete product definition:

      recipes/<product_name>/
        golden_config.json
        golden.png

    The old nested configs layout is still readable as a legacy fallback, but new
    products are created as flat recipe folders by CalibrationPage.
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
            recipe_dir = os.path.join(self.recipes_root, name)
            if not os.path.isdir(recipe_dir):
                continue

            if os.path.isfile(os.path.join(recipe_dir, "golden_config.json")):
                out.append(name)
                continue

            # Backward compatibility: old nested configs count as loadable recipes.
            configs_dir = os.path.join(recipe_dir, "configs")
            if os.path.isdir(configs_dir):
                for cfg_name in sorted(os.listdir(configs_dir)):
                    cfg_dir = os.path.join(configs_dir, cfg_name)
                    if os.path.isdir(cfg_dir) and os.path.isfile(os.path.join(cfg_dir, "golden_config.json")):
                        out.append(name)
                        break

        return out

    # Kept only so old pages/modules do not explode. New UI should not use it.
    def list_configs(self, recipe_name: str) -> List[str]:
        recipe_dir = os.path.join(self.recipes_root, recipe_name)
        configs_dir = os.path.join(recipe_dir, "configs")
        if not os.path.isdir(configs_dir):
            return []

        out: List[str] = []
        for cfg_name in sorted(os.listdir(configs_dir)):
            cfg_dir = os.path.join(configs_dir, cfg_name)
            if os.path.isdir(cfg_dir) and os.path.isfile(os.path.join(cfg_dir, "golden_config.json")):
                out.append(cfg_name)
        return out

    # Legacy compatibility no-ops/readers.
    def get_active_config_name(self, recipe_name: str) -> Optional[str]:
        p = os.path.join(self.recipes_root, recipe_name, "active_config.txt")
        try:
            with open(p, "r", encoding="utf-8") as f:
                s = f.read().strip()
            return s or None
        except Exception:
            return None

    def set_active_config_name(self, recipe_name: str, config_name: str) -> None:
        # Kept for old callers. New product flow does not use active_config.txt.
        recipe_dir = os.path.join(self.recipes_root, recipe_name)
        os.makedirs(recipe_dir, exist_ok=True)
        with open(os.path.join(recipe_dir, "active_config.txt"), "w", encoding="utf-8") as f:
            f.write((config_name or "").strip() + "\n")

    # ----------------------------
    # Loading
    # ----------------------------
    def _resolve_paths(self, recipe_name: str, config_name: Optional[str] = None) -> Tuple[bool, str, str, str]:
        """
        Returns:
          is_legacy_nested, recipe_dir, config_name_resolved, config_dir

        Flat preferred:
          recipes/<recipe>/golden_config.json

        Legacy nested fallback:
          recipes/<recipe>/configs/<config>/golden_config.json
        """
        recipe_dir = os.path.join(self.recipes_root, recipe_name)
        if not os.path.isdir(recipe_dir):
            raise FileNotFoundError(f"Recipe/product folder not found: {recipe_dir}")

        flat_cfg = os.path.join(recipe_dir, "golden_config.json")
        if os.path.isfile(flat_cfg):
            return False, recipe_dir, recipe_name, recipe_dir

        configs = self.list_configs(recipe_name)
        if configs:
            chosen = (config_name or self.get_active_config_name(recipe_name) or configs[0]).strip()
            if chosen not in configs:
                chosen = configs[0]
            return True, recipe_dir, chosen, os.path.join(recipe_dir, "configs", chosen)

        raise FileNotFoundError(f"Missing golden_config.json for product: {recipe_dir}")

    def load(self, recipe_name: str, config_name: Optional[str] = None) -> Recipe:
        is_legacy_nested, recipe_dir, cfg_name, config_dir = self._resolve_paths(recipe_name, config_name)

        cfg_path = os.path.join(config_dir, "golden_config.json")
        cfg = _safe_load_json(cfg_path)

        golden_path = cfg.get("golden_image_path")
        if golden_path:
            if not os.path.isabs(golden_path):
                golden_path = os.path.normpath(os.path.join(config_dir, golden_path))
        else:
            golden_path = _find_golden_image_in_dir(config_dir)
            if not golden_path:
                raise ValueError(f"golden_image_path missing and no golden image found in {config_dir}")

        golden = cv2.imread(golden_path)
        if golden is None:
            raise FileNotFoundError(f"Could not load golden image: {golden_path}")

        return Recipe(
            name=recipe_name,
            recipe_dir=recipe_dir,
            config_name=cfg_name,
            config_dir=config_dir,
            config_path=cfg_path,
            golden_image_path=golden_path,
            cfg=cfg,
            golden_bgr=golden,
            is_legacy=is_legacy_nested,
        )
