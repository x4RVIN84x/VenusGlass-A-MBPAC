from __future__ import annotations
import os, json
import cv2
from hmi_app.core.models import Recipe

class RecipeManager:
    def __init__(self, recipes_root: str = "recipes"):
        self.recipes_root = recipes_root

    def list_recipes(self):
        if not os.path.isdir(self.recipes_root):
            return []
        out = []
        for name in sorted(os.listdir(self.recipes_root)):
            p = os.path.join(self.recipes_root, name)
            if not os.path.isdir(p):
                continue
            cfg_path = os.path.join(p, "golden_config.json")
            if os.path.isfile(cfg_path):
                out.append(name)
        return out

    def load(self, recipe_name: str) -> Recipe:
        folder = os.path.join(self.recipes_root, recipe_name)
        cfg_path = os.path.join(folder, "golden_config.json")
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"Missing golden_config.json in {folder}")

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        golden_path = cfg.get("golden_image_path")
        if not golden_path:
            raise ValueError("golden_image_path missing in config")

        if not os.path.isabs(golden_path):
            golden_path = os.path.normpath(os.path.join(os.getcwd(), golden_path))

        golden = cv2.imread(golden_path)
        if golden is None:
            raise FileNotFoundError(f"Could not load golden image: {golden_path}")

        return Recipe(
            name=recipe_name,
            config_path=cfg_path,
            golden_image_path=golden_path,
            cfg=cfg,
            golden_bgr=golden,
        )
