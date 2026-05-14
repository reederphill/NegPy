"""
Algorithm 2: Scanline Boundary Detection with Per-Side RANSAC Line Fitting.
"""
from typing import Optional, Tuple
import numpy as np
import cv2
from negpy.domain.types import ROI
from negpy.features.geometry.mir import max_inscribed_rect, quad_confidence, is_plausible_quad

_RANSAC_ITERS = 100
_INLIER_THRESH_PX = 3
_MIN_INLIER_FRAC = 0.15


def _ransac_line(points: np.ndarray, n_iter: int = _RANSAC_ITERS, thresh: float = _INLIER_THRESH_PX) -> Optional[Tuple[np.ndarray, float]]:
    """
    Fit a line to 2-D points via RANSAC.
    Returns (normal_vec [a,b,c] where ax+by=c, inlier_fraction) or None.
    Normal form: a*x + b*y = c, with a²+b²=1.
    """
    if len(points) < 2:
        return None
    best_inliers = 0
    best_line: Optional[np.ndarray] = None
    n = len(points)
    rng = np.random.default_rng(42)
    for _ in range(n_iter):
        i, j = rng.choice(n, size=2, replace=False)
        p1, p2 = points[i], points[j]
        d = p2 - p1
        length = float(np.linalg.norm(d))
        if length < 1e-6:
            continue
        normal = np.array([-d[1], d[0]]) / length
        c = float(normal @ p1)
        dists = np.abs(points @ normal - c)
        inliers = int((dists < thresh).sum())
        if inliers > best_inliers:
            best_inliers = inliers
            # Refit to inliers
            mask = dists < thresh
            in_pts = points[mask]
            mean = in_pts.mean(axis=0)
            _, _, vt = np.linalg.svd(in_pts - mean)
            direction = vt[0]
            normal_fit = np.array([-direction[1], direction[0]])
            if np.dot(normal_fit, normal) < 0:
                normal_fit = -normal_fit
            c_fit = float(normal_fit @ mean)
            best_line = np.array([normal_fit[0], normal_fit[1], c_fit])

    if best_line is None:
        return None
    inlier_frac = best_inliers / n
    return best_line, inlier_frac


def _line_intersection_normal(l1: np.ndarray, l2: np.ndarray) -> Optional[np.ndarray]:
    """Intersect two lines in normal form [a,b,c] (ax+by=c)."""
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        return None
    x = (c1 * b2 - c2 * b1) / det
    y = (a1 * c2 - a2 * c1) / det
    return np.array([x, y])


def _infer_missing_normal(
    lines: list[Optional[np.ndarray]],
    missing_idx: int,
    img_h: int,
    img_w: int,
    target_ratio_str: str,
) -> np.ndarray:
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
        # distance between parallel lines ≈ |c1 - c2| for same normal
        frame_width = abs(right[2] - left[2])
    frame_height = frame_width / target_aspect

    if opp is None:
        # Fallback: image-boundary line
        if missing_idx == 0:
            return np.array([0.0, 1.0, 0.0])          # y=0
        elif missing_idx == 2:
            return np.array([0.0, 1.0, float(img_h)])  # y=img_h
        elif missing_idx == 1:
            return np.array([1.0, 0.0, float(img_w)])  # x=img_w
        else:
            return np.array([1.0, 0.0, 0.0])           # x=0

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

    # Compute Sobel gradients
    luma_f = luma.astype(np.float32)
    gx = cv2.Sobel(luma_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(luma_f, cv2.CV_32F, 0, 1, ksize=3)

    # Expected frame occupies centre ~70%; start scanning from outer 15% inward
    margin_y = int(img_h * 0.15)
    margin_x = int(img_w * 0.15)

    edge_points: list[list[np.ndarray]] = [[], [], [], []]  # top, right, bottom, left

    # Top edge: scan each column top→down, find first strong positive gy
    for x in range(img_w):
        col = gy[:margin_y + img_h // 2, x]
        mad = float(np.median(np.abs(col - np.median(col)))) * 1.5 + 1e-6
        threshold = max(mad, 0.02)
        peaks = np.where(col > threshold)[0]
        if len(peaks):
            edge_points[0].append(np.array([float(x), float(peaks[0])]))

    # Bottom edge: scan each column bottom→up, find first strong negative gy
    for x in range(img_w):
        col = gy[img_h // 2:, x][::-1]
        mad = float(np.median(np.abs(col - np.median(col)))) * 1.5 + 1e-6
        threshold = max(mad, 0.02)
        peaks = np.where(col < -threshold)[0]
        if len(peaks):
            edge_points[2].append(np.array([float(x), float(img_h - 1 - peaks[0])]))

    # Left edge: scan each row left→right
    for y in range(img_h):
        row = gx[y, :margin_x + img_w // 2]
        mad = float(np.median(np.abs(row - np.median(row)))) * 1.5 + 1e-6
        threshold = max(mad, 0.02)
        peaks = np.where(row > threshold)[0]
        if len(peaks):
            edge_points[3].append(np.array([float(peaks[0]), float(y)]))

    # Right edge: scan each row right→left
    for y in range(img_h):
        row = gx[y, img_w // 2:][::-1]
        mad = float(np.median(np.abs(row - np.median(row)))) * 1.5 + 1e-6
        threshold = max(mad, 0.02)
        peaks = np.where(row < -threshold)[0]
        if len(peaks):
            edge_points[1].append(np.array([float(img_w - 1 - peaks[0]), float(y)]))

    lines: list[Optional[np.ndarray]] = [None, None, None, None]
    inlier_fracs: list[float] = [0.0, 0.0, 0.0, 0.0]

    for side in range(4):
        pts = edge_points[side]
        min_pts = max(5, int(0.05 * (img_h if side in (1, 3) else img_w)))
        if len(pts) < min_pts:
            continue
        pts_arr = np.array(pts)
        result = _ransac_line(pts_arr)
        if result is None:
            continue
        line, frac = result
        if frac >= _MIN_INLIER_FRAC:
            lines[side] = line
            inlier_fracs[side] = frac

    missing = [i for i, l in enumerate(lines) if l is None]
    if len(missing) > 1:
        return fallback_roi, 0.0

    if len(missing) == 1:
        mi = missing[0]
        lines[mi] = _infer_missing_normal(lines, mi, img_h, img_w, target_ratio_str)
        inlier_fracs[mi] = 0.1

    # Intersect adjacent lines for corners: top∩right, right∩bottom, bottom∩left, left∩top
    order = [0, 1, 2, 3]   # top, right, bottom, left
    corners = []
    for i in range(4):
        pt = _line_intersection_normal(lines[order[i]], lines[order[(i + 1) % 4]])
        if pt is None:
            return fallback_roi, 0.0
        corners.append(pt)

    quad = np.array(corners)  # [[x,y], ...]
    if not is_plausible_quad(quad, img_h, img_w, target_ratio_str):
        return fallback_roi, 0.0

    mean_inlier = float(np.mean(inlier_fracs))
    confidence = quad_confidence(quad, img_h, img_w, mean_inlier, target_ratio_str)
    roi = max_inscribed_rect(quad, img_h, img_w, margin_px=2.0, target_ratio_str=target_ratio_str)
    return roi, confidence
