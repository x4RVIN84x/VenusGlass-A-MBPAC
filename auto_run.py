# auto_run.py
import os
import json
import glob
import shutil
import time
from datetime import datetime

import cv2
import numpy as np

from detector import load_and_crop, detect_baseplate
from compare_to_golden import compare_to_golden


def _load_cfg(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _copy_raw(src_path: str, dst_path: str):
    _ensure_dir(os.path.dirname(dst_path))
    shutil.copy2(src_path, dst_path)


def _save_json(data: dict, path: str):
    _ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _list_images(folder: str):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    files.sort()
    return files


def _cv_overlay_image(
    full_img_bgr: np.ndarray,
    roi,
    golden_center_roi,
    test_center_roi,
    golden_angle,
    test_angle,
    dtheta,
    test_contour=None,
    golden_contour=None,
    status="",
    px_to_mm=None,
):
    """
    Lightweight OpenCV overlay (no matplotlib) for batch mode auto-close.
    """
    x, y, w, h = roi
    out = full_img_bgr.copy()
    H, W = out.shape[:2]

    # ROI
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Points (convert ROI-rel -> ABS)
    gpt = (int(x + golden_center_roi[0]), int(y + golden_center_roi[1]))
    tpt = (int(x + test_center_roi[0]),   int(y + test_center_roi[1]))
    cv2.circle(out, gpt, 5, (255, 0, 0), -1)
    cv2.circle(out, tpt, 5, (0, 0, 255), -1)

    # Crosshairs
    cv2.line(out, (tpt[0], 0), (tpt[0], H), (0, 255, 255), 1)
    cv2.line(out, (0, tpt[1]), (W, tpt[1]), (0, 255, 255), 1)

    # Contours
    if test_contour is not None and len(test_contour) > 0:
        cv2.drawContours(out, [test_contour + np.array([[x, y]])], -1, (0, 0, 255), 2)
    if golden_contour is not None and len(golden_contour) > 0:
        cv2.drawContours(out, [golden_contour + np.array([[x, y]])], -1, (255, 0, 0), 2)

    # Deltas
    dx_px = test_center_roi[0] - golden_center_roi[0]
    dy_px = test_center_roi[1] - golden_center_roi[1]
    dx_abs, dy_abs = abs(dx_px), abs(dy_px)
    dx_mm = dy_mm = None
    if px_to_mm:
        dx_mm = dx_px * px_to_mm
        dy_mm = dy_px * px_to_mm

    # Top-left metrics box
    lines = [
        f"dX = {dx_abs:.1f} px" + (f" ({abs(dx_mm):.2f} mm)" if dx_mm is not None else ""),
        f"dY = {dy_abs:.1f} px" + (f" ({abs(dy_mm):.2f} mm)" if dy_mm is not None else ""),
        f"dA = {abs(dtheta):.2f}Deg",
    ]
    tl = (x + 10, y + 25)
    for i, line in enumerate(lines):
        cv2.putText(
            out, line, (tl[0], tl[1] + 20 * i),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

    # PASS/FAIL under the metrics
    color = (0, 255, 0) if status == "PASS" else (0, 0, 255)
    cv2.putText(
        out, status, (tl[0], tl[1] + 20 * len(lines) + 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
    )

    return out


def run_batch(
    config_path: str,
    incoming_dir: str,
    output_root: str,
    show_overlay: bool = True,
    auto_close_sec: float = 2.0,
):
    """
    Batch runner:
      - Scans incoming_dir for images
      - For each image: detect, compare, save raw/overlay/json in a dated folder
      - Optional: show overlay window and auto-close after N seconds
    """
    cfg = _load_cfg(config_path)
    roi            = cfg["roi"]
    golden_center  = cfg["expected_center"]   # ROI-relative
    golden_angle   = cfg["expected_angle"]
    tolerances     = cfg["tolerance_px"]
    golden_img_path = cfg["golden_image_path"]
    px_to_mm       = cfg.get("px_to_mm", {}).get("uniform")

    golden_img = cv2.imread(golden_img_path)
    if golden_img is None:
        print(f"❌ Could not load golden reference image: {golden_img_path}")
        return

    # golden contour (once)
    try:
        cropped_golden, _ = load_and_crop(golden_img_path, roi)
        _, _, golden_contour = detect_baseplate(cropped_golden, full_image=golden_img, roi=roi)
    except Exception:
        golden_contour = None

    images = _list_images(incoming_dir)
    if not images:
        print("ℹ️ No images found in incoming folder.")
        return

    date_folder = datetime.now().strftime("%Y-%m-%d")
    day_root = os.path.join(output_root, date_folder)
    _ensure_dir(day_root)

    processed = 0
    failed_any = False

    for img_path in images:
        try:
            ts = datetime.now().strftime("%H%M%S")
            base = os.path.splitext(os.path.basename(img_path))[0]
            run_dir = os.path.join(day_root, f"{ts}_{base}")
            _ensure_dir(run_dir)

            # copy raw
            raw_out = os.path.join(run_dir, "raw.jpg")
            _copy_raw(img_path, raw_out)

            # detect on test
            try:
                cropped_test, full_test_img = load_and_crop(img_path, roi)
            except ValueError as e:
                print(f"❌ Error loading/cropping {img_path}: {e}")
                continue

            test_center, test_angle, test_contour = detect_baseplate(
                cropped_test, full_image=full_test_img, roi=roi
            )

            if test_center is None:
                print(f"❌ No baseplate detected: {img_path}")
                # still save a JSON record
                _save_json(
                    {
                        "image": img_path,
                        "detected": False,
                        "reason": "no_contour",
                    },
                    os.path.join(run_dir, "result.json")
                )
                processed += 1
                continue

            # compare
            result = compare_to_golden(
                test_center, test_angle, golden_center, golden_angle, tolerances
            )
            status = "PASS" if result["overall"] else "FAIL"
            if not result["overall"]:
                failed_any = True

            # overlay for saving (OpenCV version for auto-close reliability)
            overlay_img = _cv_overlay_image(
                full_test_img,
                roi,
                golden_center,
                test_center,
                golden_angle,
                test_angle,
                result["dtheta"],
                test_contour=test_contour,
                golden_contour=golden_contour,
                status=status,
                px_to_mm=px_to_mm,
            )
            overlay_out = os.path.join(run_dir, "overlay.jpg")
            cv2.imwrite(overlay_out, overlay_img)

            # also (optionally) show the *matplotlib* overlay you already like
            if show_overlay:
                # Non-blocking show-and-close: create window with OpenCV
                cv2.imshow("QC Overlay (batch)", overlay_img)
                # waitKey expects milliseconds
                wait_ms = max(1, int(auto_close_sec * 1000))
                cv2.waitKey(wait_ms)
                cv2.destroyWindow("QC Overlay (batch)")

            # save JSON
            px_scale = px_to_mm if px_to_mm else None
            dx_mm = dy_mm = None
            if px_scale:
                dx_mm = (result["dx"]) * px_scale
                dy_mm = (result["dy"]) * px_scale

            _save_json(
                {
                    "image": img_path,
                    "detected": True,
                    "roi": roi,
                    "golden_center_roi": golden_center,
                    "test_center_roi": test_center,
                    "angles": {
                        "detected": test_angle,
                        "golden": golden_angle,
                        "delta": result["dtheta"],
                    },
                    "deltas": {
                        "dx_px": result["dx"],
                        "dy_px": result["dy"],
                        "dx_mm": dx_mm,
                        "dy_mm": dy_mm,
                    },
                    "tolerances": tolerances,
                    "overall": result["overall"],
                    "status": status,
                    "files": {
                        "raw": raw_out,
                        "overlay": overlay_out,
                    }
                },
                os.path.join(run_dir, "result.json")
            )

            processed += 1

        except Exception as ex:
            print(f"❌ Error processing {img_path}: {ex}")

    print(f"✅ Automation complete. Processed {processed} images.")
    print(f"📁 Output in: {day_root}")
    if failed_any:
        print("⚠️ Some glasses FAILED. See subfolders' result.json and overlays.")
