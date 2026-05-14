"""
Tests for the new ensemble autocrop algorithms.
Each algorithm is tested against synthetic film-frame images.
"""
import numpy as np
import pytest
from negpy.features.geometry.algos import hough, ransac, flood
from negpy.features.geometry.algos.ensemble import detect as ensemble_detect
from negpy.features.geometry.mir import max_inscribed_rect, is_plausible_quad


def _make_frame(
    img_h: int = 600,
    img_w: int = 900,
    frame_y1: int = 60,
    frame_y2: int = 540,
    frame_x1: int = 90,
    frame_x2: int = 810,
    rebate_luma: float = 0.05,
    frame_luma: float = 0.85,
) -> np.ndarray:
    """Plain axis-aligned frame on dark rebate."""
    luma = np.full((img_h, img_w), rebate_luma, dtype=np.float32)
    luma[frame_y1:frame_y2, frame_x1:frame_x2] = frame_luma
    return luma


def _make_trapezoidal_frame(
    img_h: int = 600,
    img_w: int = 900,
    skew_px: int = 20,
    rebate_luma: float = 0.05,
    frame_luma: float = 0.85,
) -> np.ndarray:
    """Frame with slight keystone (left side shifted up, right side shifted down)."""
    luma = np.full((img_h, img_w), rebate_luma, dtype=np.float32)
    import cv2
    pts = np.array([
        [90, 60 - skew_px],
        [810, 60 + skew_px],
        [810, 540 + skew_px],
        [90, 540 - skew_px],
    ], dtype=np.int32)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts, 1)
    luma[mask == 1] = frame_luma
    return luma


def _make_missing_top(
    img_h: int = 600,
    img_w: int = 900,
    rebate_luma: float = 0.05,
    frame_luma: float = 0.85,
) -> np.ndarray:
    """Frame where the top edge is cut off (no rebate at top)."""
    luma = np.full((img_h, img_w), rebate_luma, dtype=np.float32)
    # Frame starts at y=0 (top clipped by film holder)
    luma[0:480, 90:810] = frame_luma
    return luma


def _roi_aspect(roi):
    y1, y2, x1, x2 = roi
    h = y2 - y1
    w = x2 - x1
    if h == 0:
        return 0.0
    return w / h


def _roi_nonempty(roi):
    y1, y2, x1, x2 = roi
    return y2 > y1 and x2 > x1


RATIO = "3:2"


# ---- MIR utility tests ----

def test_mir_axis_aligned_square():
    """MIR of a rectangular quad should return close to the quad itself (before ratio)."""
    quad = np.array([[100, 50], [800, 50], [800, 550], [100, 550]], dtype=float)
    roi = max_inscribed_rect(quad, 600, 900, margin_px=0, target_ratio_str="Free")
    y1, y2, x1, x2 = roi
    assert x2 - x1 >= 600
    assert y2 - y1 >= 480


def test_mir_enforces_ratio():
    quad = np.array([[50, 50], [850, 50], [850, 550], [50, 550]], dtype=float)
    roi = max_inscribed_rect(quad, 600, 900, margin_px=0, target_ratio_str="3:2")
    aspect = _roi_aspect(roi)
    assert abs(aspect - 1.5) < 0.05


def test_is_plausible_quad_rejects_tiny():
    quad = np.array([[400, 290], [500, 290], [500, 310], [400, 310]], dtype=float)
    assert not is_plausible_quad(quad, 600, 900, "3:2")


def test_is_plausible_quad_rejects_offcenter():
    quad = np.array([[0, 0], [100, 0], [100, 67], [0, 67]], dtype=float)
    assert not is_plausible_quad(quad, 600, 900, "3:2")


def test_is_plausible_quad_accepts_valid():
    quad = np.array([[90, 60], [810, 60], [810, 540], [90, 540]], dtype=float)
    assert is_plausible_quad(quad, 600, 900, "3:2")


# ---- Algorithm tests on synthetic frames ----

@pytest.mark.parametrize("algo_module", [hough, ransac, flood])
def test_algo_axis_aligned_frame(algo_module):
    luma = _make_frame()
    roi, confidence = algo_module.detect(luma, RATIO, None)
    assert _roi_nonempty(roi), f"{algo_module.__name__} returned empty ROI"
    assert confidence > 0.0, f"{algo_module.__name__} returned zero confidence"
    aspect = _roi_aspect(roi)
    assert abs(aspect - 1.5) < 0.1, f"{algo_module.__name__} aspect {aspect:.3f} not near 3:2"


@pytest.mark.parametrize("algo_module", [hough, ransac, flood])
def test_algo_trapezoidal_frame(algo_module):
    luma = _make_trapezoidal_frame(skew_px=15)
    roi, confidence = algo_module.detect(luma, RATIO, None)
    assert _roi_nonempty(roi), f"{algo_module.__name__} returned empty ROI on trapezoid"
    aspect = _roi_aspect(roi)
    assert abs(aspect - 1.5) < 0.15, f"{algo_module.__name__} aspect {aspect:.3f} not near 3:2"


@pytest.mark.parametrize("algo_module", [hough, ransac, flood])
def test_algo_missing_top_edge(algo_module):
    luma = _make_missing_top()
    roi, confidence = algo_module.detect(luma, RATIO, None)
    assert _roi_nonempty(roi), f"{algo_module.__name__} returned empty ROI on missing-top frame"
    aspect = _roi_aspect(roi)
    assert abs(aspect - 1.5) < 0.15, f"{algo_module.__name__} aspect {aspect:.3f} not near 3:2 (missing top)"


@pytest.mark.parametrize("algo_module", [hough, ransac, flood])
def test_algo_undetectable_falls_back(algo_module):
    """Pure black image — algorithms should return a fallback, not crash."""
    luma = np.zeros((300, 450), dtype=np.float32)
    roi, confidence = algo_module.detect(luma, RATIO, None)
    # Should not raise; roi may be full-image fallback
    assert roi is not None
    assert len(roi) == 4


# ---- Ensemble tests ----

def test_ensemble_picks_winner():
    luma = _make_frame()
    h, w = luma.shape
    roi, algo = ensemble_detect(luma, h, w, RATIO, None)
    assert _roi_nonempty(roi)
    assert algo in ("hough", "ransac", "flood", "legacy")


def test_ensemble_aspect_ratio_always_enforced():
    luma = _make_frame()
    h, w = luma.shape
    roi, _ = ensemble_detect(luma, h, w, RATIO, None)
    aspect = _roi_aspect(roi)
    # The ensemble (and any algo) must enforce 3:2
    assert abs(aspect - 1.5) < 0.1


def test_ensemble_black_image_does_not_crash():
    luma = np.zeros((300, 450), dtype=np.float32)
    h, w = luma.shape
    roi, algo = ensemble_detect(luma, h, w, RATIO, None)
    assert roi is not None
    assert len(roi) == 4


def test_full_get_autocrop_coords_integration():
    """Integration: get_autocrop_coords returns a valid 3:2 crop on a synthetic frame."""
    from negpy.features.geometry.logic import get_autocrop_coords
    img = np.stack([_make_frame()] * 3, axis=-1)
    roi = get_autocrop_coords(img, offset_px=0, target_ratio_str="3:2")
    assert _roi_nonempty(roi)
    aspect = _roi_aspect(roi)
    assert abs(aspect - 1.5) < 0.1
