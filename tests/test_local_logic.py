import numpy as np
import pytest
from negpy.features.local.logic import build_ev_map, apply_local_adjustments
from negpy.features.local.models import LocalAdjustmentsConfig


def test_build_ev_map_empty():
    ev = build_ev_map([], 100, 100)
    assert ev.shape == (100, 100)
    assert ev.max() == 0.0


def test_build_ev_map_center_spot_positive():
    spots = [(0.5, 0.5, 0.1, 1.0)]
    ev = build_ev_map(spots, 100, 100)
    # Centre of a strength=1 spot should be ~1.0
    assert ev[50, 50] == pytest.approx(1.0, abs=1e-4)
    # Corners well outside radius should be ~0
    assert ev[0, 0] == pytest.approx(0.0, abs=1e-4)


def test_build_ev_map_center_spot_negative():
    spots = [(0.5, 0.5, 0.1, -0.5)]
    ev = build_ev_map(spots, 100, 100)
    assert ev[50, 50] == pytest.approx(-0.5, abs=1e-4)
    assert ev[0, 0] == pytest.approx(0.0, abs=1e-4)


def test_build_ev_map_additive():
    """Two overlapping spots of the same sign accumulate."""
    spots = [(0.5, 0.5, 0.1, 0.5), (0.5, 0.5, 0.1, 0.5)]
    ev = build_ev_map(spots, 100, 100)
    assert ev[50, 50] == pytest.approx(1.0, abs=1e-4)


def test_build_ev_map_cancel():
    """Equal positive and negative spots cancel out."""
    spots = [(0.5, 0.5, 0.1, 1.0), (0.5, 0.5, 0.1, -1.0)]
    ev = build_ev_map(spots, 100, 100)
    assert ev[50, 50] == pytest.approx(0.0, abs=1e-4)


def test_apply_passthrough():
    img = np.full((50, 50, 3), 0.5, dtype=np.float32)
    cfg = LocalAdjustmentsConfig()
    result = apply_local_adjustments(img, cfg)
    np.testing.assert_array_equal(result, img)


def test_dodge_brightens():
    img = np.full((100, 100, 3), 0.25, dtype=np.float32)
    # strength=1.0 → ev_map centre = 1.0 → 2^1.0 × 0.25 = 0.5
    cfg = LocalAdjustmentsConfig(spots=[(0.5, 0.5, 0.1, 1.0)])
    result = apply_local_adjustments(img, cfg)
    assert result[50, 50, 0] == pytest.approx(0.5, abs=1e-4)
    assert result[2, 2, 0] == pytest.approx(0.25, abs=1e-4)


def test_burn_darkens():
    img = np.full((100, 100, 3), 0.5, dtype=np.float32)
    # strength=-1.0 → ev_map centre = -1.0 → 2^-1.0 × 0.5 = 0.25
    cfg = LocalAdjustmentsConfig(spots=[(0.5, 0.5, 0.1, -1.0)])
    result = apply_local_adjustments(img, cfg)
    assert result[50, 50, 0] == pytest.approx(0.25, abs=1e-4)
    assert result[2, 2, 0] == pytest.approx(0.5, abs=1e-4)


def test_result_clamped():
    img = np.full((50, 50, 3), 0.9, dtype=np.float32)
    cfg = LocalAdjustmentsConfig(spots=[(0.5, 0.5, 0.5, 3.0)])
    result = apply_local_adjustments(img, cfg)
    assert result.max() <= 1.0 + 1e-6


def test_dodge_and_burn_at_separate_locations():
    img = np.full((100, 100, 3), 0.5, dtype=np.float32)
    cfg = LocalAdjustmentsConfig(spots=[
        (0.2, 0.5, 0.05, 1.0),   # dodge left
        (0.8, 0.5, 0.05, -1.0),  # burn right
    ])
    result = apply_local_adjustments(img, cfg)
    assert result[50, 20, 0] > 0.5
    assert result[50, 80, 0] < 0.5
    assert result[50, 50, 0] == pytest.approx(0.5, abs=0.05)
