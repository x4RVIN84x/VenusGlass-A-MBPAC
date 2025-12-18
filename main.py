# main.py
import json
import os
import cv2
import numpy as np

from detector import detect_baseplate
from compare_to_golden import compare_to_golden
from auto_run import run_batch
import roi_stablizer


# =========================
# EDITABLE FLAGS
# =========================
AUTO_MODE       = False
SHOW_OVERLAY    = True
USE_ROI_STAB    = False

CFG_PATH        = r"E:\ARVIN\A-MBPAC\reference_json\207_golden_configv2.json"
INCOMING_DIR    = r"E:\ARVIN\A-MBPAC\incoming"
OUTPUT_ROOT     = r"E:\ARVIN\A-MBPAC\output"

# Manual testing image path (only used when AUTO_MODE=False)
MANUAL_TEST_IMAGE_PATH = r"E:\ARVIN\\A-MBPAC\images\V1.1_test\WhatsApp Image 2025-12-18 at 10.57.44 AM (1).jpeg"


def _as_int_roi(roi):
    x, y, w, h = roi
    return [int(round(x)), int(round(y)), int(round(w)), int(round(h))]


def _safe_crop(full_img, roi):
    """Crop full_img by roi=(x,y,w,h) safely; returns cropped image and clamped roi."""
    x, y, w, h = roi
    H, W = full_img.shape[:2]

    x = int(max(0, min(x, W - 1)))
    y = int(max(0, min(y, H - 1)))
    w = int(max(1, min(w, W - x)))
    h = int(max(1, min(h, H - y)))

    return full_img[y:y + h, x:x + w].copy(), (x, y, w, h)


def _draw_roi(vis, roi, color, thickness=2):
    x, y, w, h = roi
    cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)


def _draw_contour_abs(vis, contour_abs, color=(0, 255, 0), thickness=2):
    if contour_abs is None:
        return
    c = np.array(contour_abs, dtype=np.int32)
    cv2.drawContours(vis, [c], -1, color, thickness)


def process_one(test_image_path: str, cfg: dict):
    # Config produced by roi_calibration.py should have at least:
    # roi, expected_center, expected_angle
    if "roi" not in cfg or "expected_center" not in cfg:
        print("❌ Config missing required keys. Need: roi, expected_center (and ideally expected_angle).")
        return

    roi_cfg = tuple(cfg["roi"])
    golden_center_roi = tuple(cfg["expected_center"])  # ROI-relative
    golden_angle = float(cfg.get("expected_angle", 0.0))
    tolerances = cfg.get("tolerance_px", {"x": 10, "y": 10, "angle": 5})

    golden_img_path = cfg.get("golden_image_path", None)

    full_test_img = cv2.imread(test_image_path)
    if full_test_img is None:
        print(f"❌ Failed to load test image: {test_image_path}")
        return

    golden_img = None
    if golden_img_path:
        golden_img = cv2.imread(golden_img_path)
        if golden_img is None:
            print(f"⚠️ Could not load golden image: {golden_img_path}")
            golden_img = None

    # --------------------------------------------
    # Choose ROI for detection
    # --------------------------------------------
    roi_for_detection = roi_cfg

    if USE_ROI_STAB:
        # Your roi_stablizer.py expects images + rois (not paths)
        if golden_img is None:
            print("⚠️ USE_ROI_STAB=True but golden image missing; skipping stabilization.")
        else:
            try:
                adjusted_roi, info = roi_stablizer.stabilize_roi(
                    current_image=full_test_img,
                    current_roi=roi_cfg,
                    golden_image=golden_img,
                    golden_roi=roi_cfg,
                )
                roi_for_detection = tuple(adjusted_roi)
                print(f"🧭 ROI stabilized. offset={info.get('offset')} roi={roi_for_detection}")
            except Exception as e:
                print("⚠️ ROI stabilizer failed:", e)
                roi_for_detection = roi_cfg

    # --------------------------------------------
    # Crop + detect baseplate
    # --------------------------------------------
    cropped_test, roi_for_detection = _safe_crop(full_test_img, roi_for_detection)

    test_center_rel, test_angle, test_cnt_rel = detect_baseplate(
        cropped_test,
        full_image=full_test_img,
        roi=roi_for_detection,
        padding=150,
        shrink_border_px=10,
        canny_low=50,
        canny_high=120,
        area_min_frac=0.005,
        contrast_min=6.0,
        border_margin=None,
    )

    if test_center_rel is None:
        print("❌ Baseplate not detected.")
        if SHOW_OVERLAY:
            vis = full_test_img.copy()
            _draw_roi(vis, roi_cfg, (255, 255, 0), 2)            # cfg ROI
            _draw_roi(vis, roi_for_detection, (0, 255, 255), 2)  # detection ROI
            cv2.imshow("QC (baseplate NOT found)", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    # Convert detected center to absolute image coords
    abs_center = (
        roi_for_detection[0] + float(test_center_rel[0]),
        roi_for_detection[1] + float(test_center_rel[1]),
    )

    # Convert to cfg-ROI-relative coords for comparing against expected_center
    test_center_cfgroi = (
        abs_center[0] - roi_cfg[0],
        abs_center[1] - roi_cfg[1],
    )

    result = compare_to_golden(
        detected_center=test_center_cfgroi,
        detected_angle=float(test_angle),
        golden_center=golden_center_roi,
        golden_angle=float(golden_angle),
        tolerances=tolerances,
    )

    print("\n=== QC RESULT ===")
    print(f"Test image: {test_image_path}")
    print(f"ΔX = {result['dx']:.2f} px   (PASS: {result['pass_x']})")
    print(f"ΔY = {result['dy']:.2f} px   (PASS: {result['pass_y']})")
    print(f"Δθ = {result['dtheta']:.2f}° (PASS: {result['pass_angle']})")
    print(f"✅ Overall: {'PASS' if result['overall'] else 'FAIL'}")

    if SHOW_OVERLAY:
        vis = full_test_img.copy()

        # ROIs
        _draw_roi(vis, roi_cfg, (255, 255, 0), 2)              # cfg ROI
        _draw_roi(vis, roi_for_detection, (0, 255, 255), 2)    # detection ROI

        # chosen contour around baseplate (ABS coords)
        if test_cnt_rel is not None:
            cnt_abs = test_cnt_rel + np.array([[roi_for_detection[0], roi_for_detection[1]]], dtype=np.float32)
            _draw_contour_abs(vis, cnt_abs, color=(0, 255, 0), thickness=2)

        # detected center (red)
        cv2.circle(vis, (int(abs_center[0]), int(abs_center[1])), 7, (0, 0, 255), -1)

        # expected center (blue)
        golden_abs = (roi_cfg[0] + float(golden_center_roi[0]), roi_cfg[1] + float(golden_center_roi[1]))
        cv2.circle(vis, (int(golden_abs[0]), int(golden_abs[1])), 7, (255, 0, 0), 2)

        cv2.imshow("QC (debug)", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    if not os.path.exists(CFG_PATH):
        print("❌ Config JSON not found:", CFG_PATH)
        return

    with open(CFG_PATH, "r") as f:
        cfg = json.load(f)

    if AUTO_MODE:
        os.makedirs(OUTPUT_ROOT, exist_ok=True)
        run_batch(
            config_path=CFG_PATH,
            incoming_dir=INCOMING_DIR,
            output_root=OUTPUT_ROOT,
            show_overlay=SHOW_OVERLAY,
            auto_close_sec=1.5,
        )
    else:
        test_image_path = cfg.get("test_image_path") or MANUAL_TEST_IMAGE_PATH
        if not test_image_path:
            print("❌ No manual test image path set.")
            print("   Set MANUAL_TEST_IMAGE_PATH in main.py or put test_image_path in JSON.")
            return
        process_one(test_image_path, cfg)


if __name__ == "__main__":
    main()
