# MBPAC HMI (PySide6) – Starter App

This is a runnable industrial dark-mode starter HMI that:
- shows live webcam feed (Logitech C270/C930e)
- lets you select a **Recipe/Car** from `recipes/<RecipeName>/golden_config.json`
- runs your current pipeline in real time (ROI stabilization + baseplate detection)
- draws baseplate contour/center and (optional) stabilizer debug overlay (lines/points/anchors)
- provides navigation skeleton for Calibration / Manual / Auto / Reports pages

## Install
```bash
pip install pyside6 opencv-python numpy
```

## Run (from your repo root)
Copy the `hmi_app/` folder into your project root (same level as `detector.py`, `roi_stablizer.py`, etc), then:
```bash
python -m hmi_app.app
```

## Recipes
Expected layout:
```
recipes/
  C270TEST_207/
    golden.jpg
    golden_config.json
  Camry_2024/
    golden.jpg
    golden_config.json
```

`golden_config.json` must contain at least:
- `golden_image_path`
- `roi`
- `expected_center`
- `tolerance_px`
Optional (for ROI stabilization):
- `registration_roi`
- `inner_border_lines`
