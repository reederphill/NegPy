"""
Algorithm 1: Probabilistic Hough Line Transform + Quadrilateral Fitting.
"""

from typing import Optional, Tuple
import numpy as np
import cv2
from negpy.domain.types import ROI
from negpy.features.geometry.mir import max_inscribed_rect, quad_confidence, is_plausible_quad


def _line_intersection(l1: np.ndarray, l2: np.ndarray) -> Optional[np.ndarray]:
    """Intersect two lines in normal form [a, b, c] where ax+by=c."""
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        return None
    x = (c1 * b2 - c2 * b1) / det
    y = (a1 * c2 - a2 * c1) / det
    return np.array([x, y])


def _fit_line_pca(points: np.ndarray) -> Optional[np.ndarray]:
    """Fit a line to points via PCA. Returns [a, b, c] where ax+by=c, a²+b²=1."""
    if len(points) < 2:
        return None
    mean = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - mean)
    direction = vt[0]
    # Normal is perpendicular to direction
    normal = np.array([-direction[1], direction[0]])
    c = float(np.dot(mean, normal))
    if c < 0:
        normal = -normal
        c = -c
    return np.array([normal[0], normal[1], c])


def _infer_missing_edge(
    lines: list[Optional[np.ndarray]],
    missing_idx: int,
    img_h: int,
    img_w: int,
    target_ratio_str: str,
) -> np.ndarray:
    """Infer the missing edge line [a,b,c] (ax+by=c) from the 3 known lines."""
    opposite_idx = (missing_idx + 2) % 4
    opp = lines[opposite_idx]

    try:
        w_r, h_r = map(float, target_ratio_str.split(":"))
        target_aspect = w_r / h_r
    except Exception:
        target_aspect = 1.5

    left = lines[3]
    right = lines[1]
    frame_width: float = img_w * 0.8
    if left is not None and right is not None:
        frame_width = abs(right[2] - left[2])
    frame_height = frame_width / target_aspect

    if opp is None:
        # Fallback: image-boundary line
        if missing_idx == 0:
            return np.array([0.0, 1.0, 0.0])  # y=0
        elif missing_idx == 2:
            return np.array([0.0, 1.0, float(img_h)])  # y=img_h
        elif missing_idx == 1:
            return np.array([1.0, 0.0, float(img_w)])  # x=img_w
        else:
            return np.array([1.0, 0.0, 0.0])  # x=0

    a, b, c = opp
    offset = frame_height if missing_idx == 0 else -frame_height
    return np.array([a, b, c + offset])


def detect(
    luma: np.ndarray,
    target_ratio_str: str,
    assist_luma: Optional[float],
) -> Tuple[ROI, float]:
    img_h, img_w = luma.shape[:2]
    fallback_roi: ROI = (0, img_h, 0, img_w)

    # Bilateral-smooth then Canny with a minimum threshold floor
    luma8 = (np.clip(luma, 0, 1) * 255).astype(np.uint8)
    smooth = cv2.bilateralFilter(luma8, d=9, sigmaColor=50, sigmaSpace=50)
    grad_mag = np.sqrt(cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3) ** 2 + cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3) ** 2)
    otsu_thresh, _ = cv2.threshold(
        np.clip(grad_mag, 0, 255).astype(np.uint8),
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    canny_high = max(float(otsu_thresh) * 1.5, 30.0)
    canny_low = max(float(otsu_thresh) * 0.5, 10.0)
    edges = cv2.Canny(smooth, canny_low, canny_high)

    diag = float(np.hypot(img_h, img_w))
    min_len = int(diag * 0.10)
    lines_raw = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 360,
        threshold=30,
        minLineLength=min_len,
        maxLineGap=30,
    )

    if lines_raw is None or len(lines_raw) < 4:
        return fallback_roi, 0.0

    segments = lines_raw[:, 0]  # shape (N, 4): x1,y1,x2,y2
    mids_x = (segments[:, 0] + segments[:, 2]) / 2.0
    mids_y = (segments[:, 1] + segments[:, 3]) / 2.0

    # Angle in [0, π): 0 = horizontal line direction, π/2 = vertical line direction
    angles = (
        np.arctan2(
            (segments[:, 3] - segments[:, 1]).astype(float),
            (segments[:, 2] - segments[:, 0]).astype(float),
        )
        % np.pi
    )

    # Split into 2 orientation groups: horizontal-ish (angle < π/4 or > 3π/4) vs vertical-ish
    horiz_mask = (angles < np.pi / 4) | (angles > 3 * np.pi / 4)
    vert_mask = ~horiz_mask

    # Within each group split by position into 2 sides
    # horizontal → top (y < img_h/2) / bottom (y >= img_h/2)
    # vertical   → left (x < img_w/2) / right (x >= img_w/2)
    side_masks = {
        0: horiz_mask & (mids_y < img_h / 2),  # top
        1: vert_mask & (mids_x >= img_w / 2),  # right
        2: horiz_mask & (mids_y >= img_h / 2),  # bottom
        3: vert_mask & (mids_x < img_w / 2),  # left
    }

    lines: list[Optional[np.ndarray]] = [None, None, None, None]
    inlier_fractions: list[float] = [0.0, 0.0, 0.0, 0.0]

    for side, mask in side_masks.items():
        idx = np.where(mask)[0]
        if len(idx) < 1:
            continue
        pts: list[list[float]] = []
        for i in idx:
            x1, y1, x2, y2 = segments[i]
            pts.extend([[float(x1), float(y1)], [float(x2), float(y2)]])
        pts_arr = np.array(pts, dtype=float)
        result = _fit_line_pca(pts_arr)
        if result is None:
            continue
        lines[side] = result
        inlier_fractions[side] = len(idx) / max(len(segments), 1)

    # Detect and fill missing edges
    missing = [i for i, ln in enumerate(lines) if ln is None]
    if len(missing) > 1:
        return fallback_roi, 0.0

    if len(missing) == 1:
        mi = missing[0]
        lines[mi] = _infer_missing_edge(lines, mi, img_h, img_w, target_ratio_str)
        inlier_fractions[mi] = 0.1

    # Compute quadrilateral corners from adjacent line intersections
    # Order: top(0) ∩ right(1), right(1) ∩ bottom(2), bottom(2) ∩ left(3), left(3) ∩ top(0)
    corners = []
    for i in range(4):
        pt = _line_intersection(lines[i], lines[(i + 1) % 4])
        if pt is None:
            return fallback_roi, 0.0
        corners.append(pt)

    quad = np.array(corners)  # [[x,y], ...]
    if not is_plausible_quad(quad, img_h, img_w, target_ratio_str):
        return fallback_roi, 0.0

    mean_inlier = float(np.mean(inlier_fractions))
    confidence = quad_confidence(quad, img_h, img_w, mean_inlier, target_ratio_str)
    roi = max_inscribed_rect(quad, img_h, img_w, margin_px=2.0, target_ratio_str=target_ratio_str)
    return roi, confidence
