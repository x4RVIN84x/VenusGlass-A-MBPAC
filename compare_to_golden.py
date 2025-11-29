import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def compare_to_golden(detected_center, detected_angle, golden_center, golden_angle, tolerances):
    """
    detected_center: (x, y) ROI-relative from detector
    golden_center:   (x, y) ROI-relative from JSON
    """
    dx = abs(detected_center[0] - golden_center[0])
    dy = abs(detected_center[1] - golden_center[1])
    da = abs(detected_angle - golden_angle)

    result = {
        "dx": dx,
        "dy": dy,
        "dtheta": da,
        "pass_x": dx <= tolerances["x"],
        "pass_y": dy <= tolerances["y"],
        "pass_angle": da <= tolerances["angle"],
    }
    result["overall"] = result["pass_x"] and result["pass_y"] and result["pass_angle"]
    return result


def _non_overlapping_label_xy(test_pt, W, H, dx_px, dy_px, contour_bbox=None):
    """Pick a label location near test_pt that avoids the contour_bbox & stays onscreen."""
    MARGIN = 12
    LABEL_W, LABEL_H = 170, 24

    def box_intersects(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        return not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2)

    sgn_x = -1 if dx_px >= 0 else 1
    sgn_y = -1 if dy_px >= 0 else 1
    RADIUS_PIX = [90, 140, 190]
    directions = [
        (sgn_x, sgn_y),
        (sgn_x, 0), (0, sgn_y),
        (sgn_x, -sgn_y), (-sgn_x, sgn_y),
        (-sgn_x, 0), (0, -sgn_y)
    ]

    for r in RADIUS_PIX:
        for dxs, dys in directions:
            px = int(test_pt[0] + dxs * r)
            py = int(test_pt[1] + dys * r)
            box = (px, py, px + LABEL_W, py + LABEL_H)
            if box[0] < 10 or box[1] < 10 or box[2] > W - 10 or box[3] > H - 10:
                continue
            if contour_bbox and box_intersects(box, contour_bbox):
                continue
            return (px, py)

    # fallback
    px = int(np.clip(test_pt[0] + (120 if dx_px <= 0 else -120), 10, W - LABEL_W - 10))
    py = int(np.clip(test_pt[1] - 30, 10, H - LABEL_H - 10))
    return (px, py)


def overlay_reference_and_test(
    golden_img,              # kept for API compatibility (not used)
    golden_center,           # ROI-relative (from JSON)
    roi,                     # (x, y, w, h)
    test_img,                # full test image (BGR)
    test_center,             # ROI-relative center from detector
    contour=None,            # ROI-relative test contour
    golden_contour=None,     # ROI-relative golden contour
    status="FAIL",
    px_to_mm: float | None = None,
    save_path: str | None = None,      # <-- NEW: save plot to this path if provided
    auto_close_sec: float | None = None  # <-- NEW: auto close after N seconds
):
    x, y, w, h = roi
    overlay = test_img.copy()
    H, W = overlay.shape[:2]

    # ROI (green)
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert ROI-relative centers -> ABSOLUTE pixels
    golden_pt = (int(x + golden_center[0]), int(y + golden_center[1]))
    test_pt   = (int(x + test_center[0]),   int(y + test_center[1]))

    # Dots
    cv2.circle(overlay, golden_pt, 5, (255, 0, 0), -1)  # blue
    cv2.circle(overlay, test_pt,   5, (0, 0, 255), -1)  # red

    # Crosshairs from test center (yellow)
    cv2.line(overlay, (test_pt[0], 0), (test_pt[0], H), (0, 255, 255), 1)
    cv2.line(overlay, (0, test_pt[1]), (W, test_pt[1]), (0, 255, 255), 1)

    # Contours (ROI-rel -> ABS)
    cnt_abs = None
    if contour is not None and len(contour) > 0:
        cnt_abs = contour + np.array([[x, y]])
        cv2.drawContours(overlay, [cnt_abs], -1, (0, 0, 255), 2)  # red
    if golden_contour is not None and len(golden_contour) > 0:
        cv2.drawContours(overlay, [golden_contour + np.array([[x, y]])], -1, (255, 0, 0), 2)  # blue

    # deltas (ROI-relative)
    dx_px = test_center[0] - golden_center[0]
    dy_px = test_center[1] - golden_center[1]
    dx_abs = abs(dx_px)
    dy_abs = abs(dy_px)
    dx_mm = dy_mm = None
    if px_to_mm:
        dx_mm = dx_px * px_to_mm
        dy_mm = dy_px * px_to_mm

    # figure
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(overlay_rgb)
    plt.title("Reference vs Test Overlay (Zoom/Pan Enabled)")
    plt.axis("on")

    # Legend top-right
    legend_elements = [
        mpatches.Patch(color='green',  label='ROI'),
        mpatches.Patch(color='blue',   label='Golden Center + Contour'),
        mpatches.Patch(color='red',    label='Test Center + Contour'),
        mpatches.Patch(color='yellow', label='Test Crosshairs'),
    ]
    plt.legend(handles=legend_elements, loc='upper right', framealpha=0.85, fontsize=10, borderpad=0.8)

    # Top-left metrics box (px & mm)
    lines = [
        f"ΔX = {dx_abs:.1f} px" + (f" ({abs(dx_mm):.2f} mm)" if dx_mm is not None else ""),
        f"ΔY = {dy_abs:.1f} px" + (f" ({abs(dy_mm):.2f} mm)" if dy_mm is not None else "")
    ]
    tl_x, tl_y = x + 10, y + 25
    plt.text(
        tl_x, tl_y,
        "\n".join(lines),
        color="white",
        bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.3"),
        fontsize=11
    )

    # PASS/FAIL under metrics
    status_color = "lime" if status == "PASS" else "red"
    plt.text(
        tl_x, tl_y + 36,
        status,
        color=status_color,
        bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.25"),
        fontsize=12,
        weight="bold"
    )

    # Dynamic callout (dx,dy) near the test dot
    contour_bbox = None
    if cnt_abs is not None:
        cx, cy, cw, ch = cv2.boundingRect(cnt_abs)
        M = 12
        contour_bbox = (cx - M, cy - M, cx + cw + M, cy + ch + M)

    label_text = f"dx={dx_px:+.1f}px"
    if dx_mm is not None:
        label_text += f", {dx_mm:+.2f}mm"
    label_text += f"\ndy={dy_px:+.1f}px"
    if dy_mm is not None:
        label_text += f", {dy_mm:+.2f}mm"

    lbl_xy = _non_overlapping_label_xy(test_pt, W, H, dx_px, dy_px, contour_bbox)
    plt.annotate(
        label_text,
        xy=(test_pt[0], test_pt[1]),
        xytext=lbl_xy,
        textcoords='data',
        color="yellow",
        fontsize=9,
        bbox=dict(facecolor="black", alpha=0.35, pad=1.5),
        arrowprops=dict(arrowstyle="->", color="yellow", lw=1, shrinkA=4, shrinkB=4)
    )

    plt.tight_layout()

    # Save / auto-close handling
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if auto_close_sec:
        plt.pause(auto_close_sec)
        plt.close()
    else:
        plt.show()
