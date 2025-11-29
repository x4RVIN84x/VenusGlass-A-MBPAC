# main.py
import json
import os
import cv2

from detector import load_and_crop, detect_baseplate
from compare_to_golden import compare_to_golden, overlay_reference_and_test
from auto_run import run_batch  # <- batch runner defined below

# =========================
# EDITABLE FLAGS (no CLI)
# =========================
AUTO_MODE       = True          # True = process all images in INCOMING_DIR
SHOW_OVERLAY    = True          # Show overlay windows (single + batch)
AUTO_CLOSE_SEC  = 2.0           # Auto-close overlay in batch mode (seconds)

CONFIG_PATH     = r"E:\ARVIN\A-MBPAC\reference_json\207_golden_config.json"
TEST_IMAGE_PATH = r"E:\ARVIN\A-MBPAC\images\check_207_R1\angle_adjusted.jpg"

INCOMING_DIR    = r"E:\ARVIN\A-MBPAC\incoming"
OUTPUT_ROOT     = r"E:\ARVIN\A-MBPAC\glass_data"
# =========================


def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


def run_single(config_path: str, test_image_path: str, show_overlay: bool = True):
    cfg = load_config(config_path)
    roi            = cfg["roi"]
    golden_center  = cfg["expected_center"]     # ROI-relative
    golden_angle   = cfg["expected_angle"]
    tolerances     = cfg["tolerance_px"]
    golden_img_path = cfg["golden_image_path"]
    px_to_mm = cfg.get("px_to_mm", {}).get("uniform")

    # test image
    try:
        cropped_test, full_test_img = load_and_crop(test_image_path, roi)
    except ValueError as e:
        print(e)
        return

    # golden image (only to get golden contour for drawing)
    golden_img = cv2.imread(golden_img_path)
    if golden_img is None:
        print(f"❌ Could not load golden reference image: {golden_img_path}")
        return

    # detect on test
    test_center, test_angle, test_contour = detect_baseplate(
        cropped_test, full_image=full_test_img, roi=roi
    )
    if test_center is None:
        print("❌ No baseplate detected in test image.")
        cv2.imshow("Failed Detection", full_test_img)
        cv2.waitKey(0); cv2.destroyAllWindows()
        return

    # detect golden contour
    cropped_golden, _ = load_and_crop(golden_img_path, roi)
    _, _, golden_contour = detect_baseplate(
        cropped_golden, full_image=golden_img, roi=roi
    )

    # compare
    result = compare_to_golden(
        test_center, test_angle, golden_center, golden_angle, tolerances
    )

    print("\n📊 Comparison Results:")
    print(f"ΔX = {result['dx']} px   (PASS: {result['pass_x']})")
    print(f"ΔY = {result['dy']} px   (PASS: {result['pass_y']})")
    print(f"Δθ = {result['dtheta']}° (PASS: {result['pass_angle']})")
    if px_to_mm:
        print(f"ΔX = {result['dx']:.1f}px ({result['dx']*px_to_mm:.2f} mm)")
        print(f"ΔY = {result['dy']:.1f}px ({result['dy']*px_to_mm:.2f} mm)")
    print(f"\n✅ Overall QC Result: {'PASS' if result['overall'] else 'FAIL'}")

    if show_overlay:
        overlay_reference_and_test(
            golden_img=None,               # kept for API compatibility
            golden_center=golden_center,   # ROI-relative
            roi=roi,
            test_img=full_test_img,
            test_center=test_center,       # ROI-relative
            contour=test_contour,
            golden_contour=golden_contour,
            status="PASS" if result["overall"] else "FAIL",
            px_to_mm=px_to_mm,
        )


if __name__ == "__main__":
    if AUTO_MODE:
        run_batch(
            config_path=CONFIG_PATH,
            incoming_dir=INCOMING_DIR,
            output_root=OUTPUT_ROOT,
            show_overlay=SHOW_OVERLAY,
            auto_close_sec=AUTO_CLOSE_SEC
        )
    else:
        run_single(
            config_path=CONFIG_PATH,
            test_image_path=TEST_IMAGE_PATH,
            show_overlay=SHOW_OVERLAY
        )
