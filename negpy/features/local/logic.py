from typing import List, Tuple

import numpy as np

from negpy.features.local.models import LocalAdjustmentsConfig


def build_ev_map(spots: List[Tuple[float, float, float, float]], img_h: int, img_w: int) -> np.ndarray:
    """
    Accumulate brush spots into a float32 EV adjustment map.

    Each spot is (nx, ny, radius, strength_ev). Spots add together additively
    with a Gaussian falloff (sigma = r/2.5), so multiple passes build up
    gradually. Positive values dodge, negative values burn.

    Uses per-spot bounding-box clipping: O(n * r²) not O(n * H * W).
    """
    ev_map = np.zeros((img_h, img_w), dtype=np.float32)
    if not spots:
        return ev_map

    short_side = float(min(img_h, img_w))

    for nx, ny, radius, strength in spots:
        cx = nx * img_w
        cy = ny * img_h
        r = radius * short_side
        if r <= 0:
            continue

        x0 = max(0, int(cx - r) - 1)
        x1 = min(img_w, int(cx + r) + 2)
        y0 = max(0, int(cy - r) - 1)
        y1 = min(img_h, int(cy + r) + 2)
        if x0 >= x1 or y0 >= y1:
            continue

        xs = np.arange(x0, x1, dtype=np.float32)
        ys = np.arange(y0, y1, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)
        dist_sq = (gx - cx) ** 2 + (gy - cy) ** 2
        sigma_sq = (r / 2.5) ** 2
        circle = np.exp(-0.5 * dist_sq / sigma_sq).astype(np.float32)
        ev_map[y0:y1, x0:x1] += circle * strength

    return ev_map


def apply_local_adjustments(img: np.ndarray, config: LocalAdjustmentsConfig) -> np.ndarray:
    """
    Apply the accumulated EV map to a linear float32 RGB image [H, W, 3].

    img * 2^(ev_map) — positive ev_map values dodge, negative burn.
    Returns the adjusted image clipped to [0, 1].
    """
    if not config.spots:
        return img

    h, w = img.shape[:2]
    ev_map = build_ev_map(config.spots, h, w)
    factor = np.power(2.0, ev_map).astype(np.float32)
    result = img.astype(np.float32, copy=True) * factor[..., np.newaxis]
    return np.clip(result, 0.0, 1.0)
