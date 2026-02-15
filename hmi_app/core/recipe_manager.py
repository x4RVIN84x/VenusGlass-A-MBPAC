from __future__ import annotations

import os
import json
from typing import List, Optional, Tuple

import cv2

from hmi_app.core.models import Recipe


def _read_text(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            s = f.read().strip()
        return s or None
    except Exception:
        return None


def _safe_load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_golden_image_in_dir(config_dir: str) -> Optional[str]:
    """
    Convention: golden.png preferred, then golden.jpg/jpeg.
    """
    for name in ("golden.png", "golden.jpg", "golden.jpeg"):
        p = os.path.join(config_dir, name)
        if os.path.isfile(p):
            return p
    return None


class RecipeManager:
    def __init__(self, recipes_root: str = "recipes"):
        self.recipes_root = recipes_root

    # ----------------------------
    # Discovery
    # ----------------------------
    def list_recipes(self) -> List[str]:
        """
        Recipe dropdown should include:
        - legacy recipes containing recipes/<name>/golden_config.json
        - new recipes containing recipes/<name>/configs/<cfg>/golden_config.json
        """
        if not os.path.isdir(self.recipes_root):
            return []

        out: List[str] = []
        for name in sorted(os.listdir(self.recipes_root)):
            recipe_dir = os.path.join(self.recipes_root, name)
            if not os.path.isdir(recipe_dir):
                continue

            legacy_cfg = os.path.join(recipe_dir, "golden_config.json")
            if os.path.isfile(legacy_cfg):
                out.append(name)
                continue

            configs_dir = os.path.join(recipe_dir, "configs")
            if os.path.isdir(configs_dir):
                # any subfolder containing golden_config.json counts
                for cfg_name in sorted(os.listdir(configs_dir)):
                    cfg_dir = os.path.join(configs_dir, cfg_name)
                    if not os.path.isdir(cfg_dir):
                        continue
                    if os.path.isfile(os.path.join(cfg_dir, "golden_config.json")):
                        out.append(name)
                        break

        return out

    def list_configs(self, recipe_name: str) -> List[str]:
        """
        New layout only: recipes/<recipe>/configs/<config_name>/
        Returns [] for legacy recipes.
        """
        recipe_dir = os.path.join(self.recipes_root, recipe_name)
        configs_dir = os.path.join(recipe_dir, "configs")
        if not os.path.isdir(configs_dir):
            return []

        out: List[str] = []
        for cfg_name in sorted(os.listdir(configs_dir)):
            cfg_dir = os.path.join(configs_dir, cfg_name)
            if not os.path.isdir(cfg_dir):
                continue
            if os.path.isfile(os.path.join(cfg_dir, "golden_config.json")):
                out.append(cfg_name)
        return out

    def get_active_config_name(self, recipe_name: str) -> Optional[str]:
        """
        Reads recipes/<recipe>/active_config.txt if present.
        Returns None if missing.
        """
        recipe_dir = os.path.join(self.recipes_root, recipe_name)
        return _read_text(os.path.join(recipe_dir, "active_config.txt"))

    def set_active_config_name(self, recipe_name: str, config_name: str) -> None:
        """
        Writes recipes/<recipe>/active_config.txt
        """
        recipe_dir = os.path.join(self.recipes_root, recipe_name)
        os.makedirs(recipe_dir, exist_ok=True)
        p = os.path.join(recipe_dir, "active_config.txt")
        with open(p, "w", encoding="utf-8") as f:
            f.write((config_name or "").strip() + "\n")

    # ----------------------------
    # Loading
    # ----------------------------
    def _resolve_mode_and_paths(
        self, recipe_name: str, config_name: Optional[str]
    ) -> Tuple[bool, str, str, str]:
        """
        Returns: (is_legacy, recipe_dir, config_name_resolved, config_dir)
        """
        recipe_dir = os.path.join(self.recipes_root, recipe_name)
        if not os.path.isdir(recipe_dir):
            raise FileNotFoundError(f"Recipe folder not found: {recipe_dir}")

        legacy_cfg_path = os.path.join(recipe_dir, "golden_config.json")
        configs_dir = os.path.join(recipe_dir, "configs")

        # Prefer new layout if configs/ exists and has at least one config folder
        if os.path.isdir(configs_dir):
            configs = self.list_configs(recipe_name)
            if configs:
                chosen = (config_name or self.get_active_config_name(recipe_name) or "").strip()
                if not chosen:
                    chosen = configs[0]
                if chosen not in configs:
                    # fallback to first available config
                    chosen = configs[0]
                config_dir = os.path.join(configs_dir, chosen)
                return False, recipe_dir, chosen, config_dir

        # Legacy fallback
        if os.path.isfile(legacy_cfg_path):
            return True, recipe_dir, "legacy", recipe_dir

        raise FileNotFoundError(
            f"Recipe has neither legacy golden_config.json nor configs/*/golden_config.json: {recipe_dir}"
        )

    def load(self, recipe_name: str, config_name: Optional[str] = None) -> Recipe:
        """
        Loads recipe+config.

        Legacy:
          recipes/<recipe>/golden_config.json (expects golden_image_path inside cfg)

        New:
          recipes/<recipe>/configs/<config>/golden_config.json
          recipes/<recipe>/configs/<config>/golden.(png|jpg|jpeg)
          (golden_image_path in cfg is optional; if present it can be relative to config_dir)
        """
        is_legacy, recipe_dir, cfg_name, config_dir = self._resolve_mode_and_paths(recipe_name, config_name)

        cfg_path = os.path.join(config_dir, "golden_config.json")
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"Missing golden_config.json: {cfg_path}")

        cfg = _safe_load_json(cfg_path)

        # Resolve golden image path
        golden_path = cfg.get("golden_image_path")

        if golden_path:
            # allow relative to config_dir
            if not os.path.isabs(golden_path):
                golden_path = os.path.normpath(os.path.join(config_dir, golden_path))
        else:
            # enforce convention in new mode
            golden_path = _find_golden_image_in_dir(config_dir)
            if not golden_path:
                # last resort: allow legacy behavior of searching cwd-relative paths
                raise ValueError(
                    f"golden_image_path missing AND no golden.(png/jpg/jpeg) found in {config_dir}"
                )

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
            is_legacy=is_legacy,
        )
