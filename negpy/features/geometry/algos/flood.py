"""
Algorithm 3: Flood-Fill Rebate Segmentation + Polygon Contour.
"""
from typing import Optional, Tuple
import numpy as np
import cv2
from negpy.domain.types import ROI
from negpy.features.geometry.mir import max_inscribed_rect, quad_confidence, is_plausible_quad


def _border_flood_mask(binary: np.ndarray) -> np.ndarray:
    """
    Mark all dark pixels (value=0) connected to the image border as 'rebate'.
    Uses flood-fill from a 1-px seeded border frame.
    """
    h, w = binary.shape
    # Pad with a border of 0 to guarantee connectivity from outside
    padded = np.pad(binary, 1, constant_values=0)
    # Flood fill from (0,0) which is guaranteed dark
    seed = (0, 0)
    flood = padded.copy().astype(np.uint8)
    # cv2.floodFill wants uint8, fills the connected region from seed
    fill_val = 2
    cv2.floodFill(flood, None, seed, fill_val)
    # 'rebate' = pixels that were dark AND reached by flood
    rebate_padded = (flood == fill_val)
    return rebate_padded[1:-1, 1:-1]


def _approx_to_quad(contour: np.ndarray, perimeter: float) -> Optional[np.ndarray]:
    """Simplify contour to a 4-point polygon, increasing epsilon until we get 4 points."""
    for factor in [0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.20]:
        eps = factor * perimeter
        approx = cv2.approxPolyDP(contour, eps, closed=True)
        if len(approx) <= 4:
            return approx.reshape(-1, 2).astype(float)
    return None


def _infer_missing_edge_quad(
    pts: list[np.ndarray],
    missing_side: str,
    img_h: int,
    img_w: int,
    target_ratio_str: str,
) -> np.ndarray:
    """
    Given 3 corners of a quad and the missing side label ('top','bottom','left','right'),
    compute the 4th corner and return the completed 4-point quad [[x,y],...].
    """
    try:
        w_r, h_r = map(float, target_ratio_str.split(":"))
        target_aspect = w_r / h_r
    except Exception:
        target_aspect = 1.5

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    frame_w = max(xs) - min(xs)
    frame_h = max(ys) - min(ys)

    if missing_side in ("top", "bottom"):
        inferred_h = frame_w / target_aspect
        if missing_side == "top":
            new_y = max(ys) - inferred_h
        else:
            new_y = min(ys) + inferred_h
        # Add two corners at new_y matching existing x extents
        tl = np.array([min(xs), new_y])
        tr = np.array([max(xs), new_y])
        quad_pts = pts + [tl, tr]
    else:
        inferred_w = frame_h * target_aspect
        if missing_side == "left":
            new_x = max(xs) - inferred_w
        else:
            new_x = min(xs) + inferred_w
        tl = np.array([new_x, min(ys)])
        bl = np.array([new_x, max(ys)])
        quad_pts = pts + [tl, bl]

    arr = np.array(quad_pts)
    # Sort into clockwise order: top-left, top-right, bottom-right, bottom-left
    cx, cy = arr.mean(axis=0)
    angles = np.arctan2(arr[:, 1] - cy, arr[:, 0] - cx)
    return arr[np.argsort(angles)]


def detect(
    luma: np.ndarray,
    target_ratio_str: str,
    assist_luma: Optional[float],
) -> Tuple[ROI, float]:
    img_h, img_w = luma.shape[:2]
    fallback_roi: ROI = (0, img_h, 0, img_w)

    # Threshold
    luma8 = (np.clip(luma, 0, 1) * 255).astype(np.uint8)
    if assist_luma is not None:
        thresh_val = int(np.clip(assist_luma * 255 - 5, 100, 250))
        _, binary = cv2.threshold(luma8, thresh_val, 255, cv2.THRESH_BINARY)
    else:
        _, binary = cv2.threshold(luma8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Denoise with median filter
    binary = cv2.medianBlur(binary, 5)

    dark_mask = (binary == 0)

    # Flood-fill from borders to find rebate
    rebate = _border_flood_mask(dark_mask.astype(np.uint8))
    frame_mask = (~rebate & ~dark_mask) | (~rebate & (binary > 0))

    # Morphological close to fill small gaps (sprocket holes, dust)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    frame_u8 = frame_mask.astype(np.uint8) * 255
    frame_u8 = cv2.morphologyEx(frame_u8, cv2.MORPH_CLOSE, kernel)

    # Largest connected component
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(frame_u8, connectivity=8)
    if n_labels < 2:
        return fallback_roi, 0.0

    # Skip label 0 (background); pick largest non-background component
    comp_areas = stats[1:, cv2.CC_STAT_AREA]
    best_label = int(np.argmax(comp_areas)) + 1
    min_frame_area = img_h * img_w * 0.05
    if comp_areas[best_label - 1] < min_frame_area:
        return fallback_roi, 0.0

    frame_region = (labels == best_label).astype(np.uint8) * 255

    # Contour + convex hull
    contours, _ = cv2.findContours(frame_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return fallback_roi, 0.0

    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    perimeter = float(cv2.arcLength(hull, closed=True))
    if perimeter < 10:
        return fallback_roi, 0.0

    quad_pts = _approx_to_quad(hull, perimeter)
    if quad_pts is None or len(quad_pts) < 3:
        return fallback_roi, 0.0

    border_thresh = 5  # px from image edge counts as "at boundary"

    if len(quad_pts) == 4:
        quad = quad_pts
        missing_sides = []
        for i, pt in enumerate(quad):
            x, y = pt
            if y <= border_thresh:
                missing_sides.append("top")
            elif y >= img_h - border_thresh:
                missing_sides.append("bottom")
            if x <= border_thresh:
                missing_sides.append("left")
            elif x >= img_w - border_thresh:
                missing_sides.append("right")

        if len(set(missing_sides)) == 1:
            # One full side is cut off — infer it
            missing_side = list(set(missing_sides))[0]
            interior_pts = [p for p in quad if not (
                (missing_side == "top" and p[1] <= border_thresh) or
                (missing_side == "bottom" and p[1] >= img_h - border_thresh) or
                (missing_side == "left" and p[0] <= border_thresh) or
                (missing_side == "right" and p[0] >= img_w - border_thresh)
            )]
            if len(interior_pts) >= 2:
                quad = _infer_missing_edge_quad(interior_pts, missing_side, img_h, img_w, target_ratio_str)

    else:
        # Only 3 points (one side missing from hull)
        xs = quad_pts[:, 0]
        ys = quad_pts[:, 1]
        # Determine which side is missing by which boundary is unrepresented
        near_top = any(y <= border_thresh for y in ys)
        near_bottom = any(y >= img_h - border_thresh for y in ys)
        near_left = any(x <= border_thresh for x in xs)
        near_right = any(x >= img_w - border_thresh for x in xs)

        if not near_top:
            missing = "top"
        elif not near_bottom:
            missing = "bottom"
        elif not near_left:
            missing = "left"
        elif not near_right:
            missing = "right"
        else:
            missing = "top"

        quad = _infer_missing_edge_quad(list(quad_pts), missing, img_h, img_w, target_ratio_str)

    quad = np.array(quad).reshape(-1, 2).astype(float)
    if not is_plausible_quad(quad, img_h, img_w, target_ratio_str):
        return fallback_roi, 0.0

    # Inlier fraction: ratio of frame pixels inside the convex hull vs total
    hull_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillConvexPoly(hull_mask, quad.astype(np.int32), 1)
    frame_pixels = int((frame_region > 0).sum())
    hull_pixels = int(hull_mask.sum())
    inlier_frac = float(frame_pixels) / (hull_pixels + 1e-6)
    inlier_frac = float(np.clip(inlier_frac, 0.0, 1.0))

    confidence = quad_confidence(quad, img_h, img_w, inlier_frac, target_ratio_str)
    roi = max_inscribed_rect(quad, img_h, img_w, margin_px=2.0, target_ratio_str=target_ratio_str)
    return roi, confidence
