"""
Runs all 3 frame-detection algorithms in parallel and picks the highest-confidence result.
Falls back to the legacy algorithm if all confidence scores are below 0.25.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple
import numpy as np
from negpy.domain.types import ROI
from negpy.features.geometry.algos import hough, ransac, flood

_MIN_CONFIDENCE = 0.25


def detect(
    luma: np.ndarray,
    img_h: int,
    img_w: int,
    target_ratio_str: str,
    assist_luma: Optional[float],
) -> Tuple[ROI, str]:
    """
    Run all 3 algorithms concurrently. Return (best_roi, winning_algo_name).
    Returns (None, "legacy") when all algorithms score below _MIN_CONFIDENCE.
    """
    algos = [
        ("hough", hough.detect),
        ("ransac", ransac.detect),
        ("flood", flood.detect),
    ]

    results: list[Tuple[str, ROI, float]] = []

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(fn, luma, target_ratio_str, assist_luma): name
            for name, fn in algos
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                roi, confidence = future.result()
                results.append((name, roi, confidence))
            except Exception:
                pass

    if not results:
        return (0, img_h, 0, img_w), "legacy"

    best_name, best_roi, best_conf = max(results, key=lambda r: r[2])
    if best_conf < _MIN_CONFIDENCE:
        return (0, img_h, 0, img_w), "legacy"

    return best_roi, best_name
