"""
Maximum Inscribed axis-aligned Rectangle within a convex quadrilateral.
"""
from typing import Optional
import numpy as np
from negpy.domain.types import ROI
from negpy.features.geometry.logic import apply_margin_to_roi, enforce_roi_aspect_ratio


def _poly_x_span(quad: np.ndarray, y: float) -> Optional[tuple[float, float]]:
    """Horizontal span of convex quad at scanline y. Returns None if y is outside."""
    xs: list[float] = []
    n = len(quad)
    for i in range(n):
        p1 = quad[i]
        p2 = quad[(i + 1) % n]
        y1, y2 = float(p1[1]), float(p2[1])
        if y1 == y2:
            if abs(y - y1) < 0.5:
                xs.extend([float(p1[0]), float(p2[0])])
            continue
        if (y1 <= y <= y2) or (y2 <= y <= y1):
            t = (y - y1) / (y2 - y1)
            xs.append(float(p1[0]) + t * (float(p2[0]) - float(p1[0])))
    if len(xs) < 2:
        return None
    return min(xs), max(xs)


def max_inscribed_rect(
    quad: np.ndarray,
    img_h: int,
    img_w: int,
    margin_px: float,
    target_ratio_str: str,
    n_samples: int = 500,
) -> ROI:
    """
    Find the largest axis-aligned rectangle inside convex quadrilateral *quad*
    (shape [4,2] as [[x,y],...]), then apply margin and aspect ratio.
    """
    quad = quad.astype(float)

    # Collect y values to sample: vertex ys + uniform grid
    vert_ys = quad[:, 1]
    y_min = max(0.0, float(vert_ys.min()))
    y_max = min(float(img_h), float(vert_ys.max()))

    if y_max <= y_min:
        return 0, img_h, 0, img_w

    grid_ys = np.linspace(y_min, y_max, n_samples)
    sample_ys = np.unique(np.concatenate([vert_ys, grid_ys]))
    sample_ys = sample_ys[(sample_ys >= y_min) & (sample_ys <= y_max)]

    # Precompute span at each sampled y
    spans: list[tuple[float, float, float]] = []  # (y, x_left, x_right)
    for y in sample_ys:
        span = _poly_x_span(quad, y)
        if span is not None:
            xl, xr = span
            xl = max(0.0, xl)
            xr = min(float(img_w), xr)
            if xr > xl:
                spans.append((y, xl, xr))

    if not spans:
        return 0, img_h, 0, img_w

    ys_arr = np.array([s[0] for s in spans])
    xl_arr = np.array([s[1] for s in spans])
    xr_arr = np.array([s[2] for s in spans])

    best_area = 0.0
    best_roi: ROI = (0, img_h, 0, img_w)

    n = len(spans)
    for i in range(n):
        # For a convex polygon, the minimum width over [y_i, y_j] is at one endpoint.
        # Track running min as we expand downward from i.
        xl_min = xl_arr[i]
        xr_min = xr_arr[i]
        for j in range(i, n):
            xl_min = max(xl_min, xl_arr[j])   # left bound tightens inward
            xr_min = min(xr_min, xr_arr[j])   # right bound tightens inward
            if xr_min <= xl_min:
                break
            height = ys_arr[j] - ys_arr[i]
            width = xr_min - xl_min
            area = width * height
            if area > best_area:
                best_area = area
                best_roi = (
                    int(round(ys_arr[i])),
                    int(round(ys_arr[j])),
                    int(round(xl_min)),
                    int(round(xr_min)),
                )

    roi = apply_margin_to_roi(best_roi, img_h, img_w, margin_px)
    return enforce_roi_aspect_ratio(roi, img_h, img_w, target_ratio_str)


def quad_confidence(
    quad: np.ndarray,
    img_h: int,
    img_w: int,
    inlier_fraction: float,
    target_ratio_str: str,
) -> float:
    """Composite confidence score 0..1 for a detected quadrilateral."""
    if quad is None or len(quad) != 4:
        return 0.0

    # Parallelogram regularity: opposite sides should be equal length
    sides = [
        np.linalg.norm(quad[(i + 1) % 4] - quad[i])
        for i in range(4)
    ]
    if max(sides) < 1.0:
        return 0.0
    opp_diff = abs(sides[0] - sides[2]) + abs(sides[1] - sides[3])
    mean_side = np.mean(sides)
    regularity = max(0.0, 1.0 - opp_diff / (2.0 * mean_side + 1e-6))

    # Area plausibility: centred at 60% of image area, σ=25%
    img_area = float(img_h * img_w)
    d1 = quad[2] - quad[0]
    d2 = quad[3] - quad[1]
    quad_area = float(abs(d1[0] * d2[1] - d1[1] * d2[0])) / 2.0
    area_frac = quad_area / (img_area + 1e-6)
    area_score = float(np.exp(-0.5 * ((area_frac - 0.60) / 0.25) ** 2))

    # Centroid proximity to image centre, σ=20% of diagonal
    centroid = quad.mean(axis=0)
    img_centre = np.array([img_w / 2.0, img_h / 2.0])
    diag = float(np.hypot(img_h, img_w))
    dist_frac = float(np.linalg.norm(centroid - img_centre)) / (diag + 1e-6)
    centroid_score = float(np.exp(-0.5 * (dist_frac / 0.20) ** 2))

    score = (
        0.40 * np.clip(inlier_fraction, 0.0, 1.0)
        + 0.25 * regularity
        + 0.20 * area_score
        + 0.15 * centroid_score
    )
    return float(np.clip(score, 0.0, 1.0))


def is_plausible_quad(quad: np.ndarray, img_h: int, img_w: int, target_ratio_str: str) -> bool:
    """Reject quads that are implausibly sized, centred, or shaped."""
    if quad is None or len(quad) != 4:
        return False

    img_area = float(img_h * img_w)
    _d1 = quad[2] - quad[0]
    _d2 = quad[3] - quad[1]
    quad_area = float(abs(_d1[0] * _d2[1] - _d1[1] * _d2[0])) / 2.0
    area_frac = quad_area / (img_area + 1e-6)
    if area_frac < 0.20 or area_frac > 0.98:
        return False

    centroid = quad.mean(axis=0)
    img_centre = np.array([img_w / 2.0, img_h / 2.0])
    diag = float(np.hypot(img_h, img_w))
    if np.linalg.norm(centroid - img_centre) / diag > 0.35:
        return False

    try:
        w_r, h_r = map(float, target_ratio_str.split(":"))
        target_aspect = w_r / h_r
    except Exception:
        target_aspect = 1.5

    xs = quad[:, 0]
    ys = quad[:, 1]
    approx_w = float(xs.max() - xs.min())
    approx_h = float(ys.max() - ys.min())
    if approx_h < 1.0:
        return False
    approx_aspect = approx_w / approx_h
    # Allow aspect ratio up to 2x different from target
    ratio_ok = (approx_aspect / target_aspect < 2.0) and (target_aspect / approx_aspect < 2.0)
    if not ratio_ok:
        # try flipped (portrait)
        approx_aspect_v = approx_h / approx_w
        ratio_ok = (approx_aspect_v / target_aspect < 2.0) and (target_aspect / approx_aspect_v < 2.0)
    return ratio_ok
