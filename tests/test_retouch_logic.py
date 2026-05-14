import pytest
import numpy as np
from negpy.features.retouch.logic import apply_dust_removal
from negpy.features.retouch.models import RetouchSpot


def _spot(dest_x: float, dest_y: float, source_x: float, source_y: float, radius: float) -> RetouchSpot:
    return RetouchSpot(dest_x=dest_x, dest_y=dest_y, source_x=source_x, source_y=source_y, radius=radius)


def test_manual_dust_removal_effect():
    # Use grey background and white dust (inverted film scan scenario).
    # Source is a neutral grey region offset from the bright dest.
    img = np.full((100, 100, 3), 0.5, dtype=np.float32)
    img[48:53, 48:53] = 1.0

    orig_mean = np.mean(img)
    # dest is the white spot (0.5, 0.5); source is nearby grey (0.6, 0.5)
    manual_spots = [_spot(0.5, 0.5, 0.6, 0.5, 10.0)]

    res = apply_dust_removal(
        img.copy(),
        dust_remove=False,
        dust_threshold=0.75,
        dust_size=2,
        manual_spots=manual_spots,
        scale_factor=1.0,
    )

    res_mean = np.mean(res)
    # The healing should make the white spot darker (closer to 0.5 background)
    assert res_mean < orig_mean

    spot_area = res[48:53, 48:53]
    assert np.mean(spot_area) < 0.9


def test_manual_dust_removal_no_spots():
    img = np.ones((100, 100, 3), dtype=np.float32)
    res = apply_dust_removal(
        img.copy(),
        dust_remove=False,
        dust_threshold=0.75,
        dust_size=2,
        manual_spots=[],
        scale_factor=1.0,
    )
    assert np.array_equal(img, res)


def test_auto_dust_removal_low_res():
    # Simple isolated white pixel on dark background
    img = np.zeros((100, 100, 3), dtype=np.float32)
    img[50, 50] = 1.0

    res = apply_dust_removal(
        img.copy(),
        dust_remove=True,
        dust_threshold=0.5,
        dust_size=2,
        manual_spots=[],
        scale_factor=1.0,
    )

    # The bright pixel should be gone
    assert res[50, 50, 0] < 0.5


def test_auto_dust_removal_high_res():
    # Larger spot at high scale
    img = np.zeros((200, 200, 3), dtype=np.float32)
    img[98:103, 98:103] = 1.0

    res = apply_dust_removal(
        img.copy(),
        dust_remove=True,
        dust_threshold=0.5,
        dust_size=4,
        manual_spots=[],
        scale_factor=2.0,
    )

    # The bright spot should be healed
    assert np.mean(res[98:103, 98:103]) < 0.5


def test_auto_dust_removal_cloud_protection():
    # Soft gradients should NOT be treated as dust
    y, x = np.mgrid[0:100, 0:100]
    img_gray = (np.sin(x / 10.0) * np.cos(y / 10.0) * 0.1) + 0.5
    img = np.stack([img_gray] * 3, axis=-1).astype(np.float32)

    res = apply_dust_removal(
        img.copy(),
        dust_remove=True,
        dust_threshold=0.5,
        dust_size=2,
        manual_spots=[],
        scale_factor=1.0,
    )

    # Soft gradients should remain identical or very close
    np.testing.assert_allclose(img, res, atol=0.01)


# ---------------------------------------------------------------------------
# New Poisson heal tests
# ---------------------------------------------------------------------------


def test_poisson_heal_reduces_brightness_at_dest():
    """White patch on grey background: heal from grey source reduces dest brightness."""
    img = np.full((100, 100, 3), 0.5, dtype=np.float32)
    # Bright white region at dest
    img[45:55, 45:55] = 1.0

    before_mean = float(np.mean(img[45:55, 45:55]))

    # Source at (0.75, 0.5) — entirely grey, radius 8 px → well within image
    spot = _spot(0.5, 0.5, 0.75, 0.5, 8.0)
    res = apply_dust_removal(
        img.copy(),
        dust_remove=False,
        dust_threshold=0.75,
        dust_size=2,
        manual_spots=[spot],
        scale_factor=1.0,
    )

    after_mean = float(np.mean(res[45:55, 45:55]))
    # Dest area must be darker after heal (moved toward 0.5 grey)
    assert after_mean < before_mean, f"Expected dest to darken: {before_mean:.3f} → {after_mean:.3f}"


def test_poisson_heal_transfers_source_texture():
    """Distinct texture in source region should appear at dest after heal."""
    img = np.full((200, 200, 3), 0.4, dtype=np.float32)
    # Add a distinct striped texture in the source region (top-right quadrant)
    for col in range(100, 200, 4):
        img[:100, col : col + 2] = 0.8

    # Dest is a plain grey patch in the bottom-left quadrant
    dest_region_before = img[130:160, 30:60].copy()

    spot = _spot(
        dest_x=0.225,  # (30+60)/2 / 200 = 0.225
        dest_y=0.725,  # (130+160)/2 / 200 = 0.725
        source_x=0.75,  # (100+200)/2 / 200 = 0.75
        source_y=0.25,  # (0+100)/2 / 200 = 0.25
        radius=12.0,
    )
    res = apply_dust_removal(
        img.copy(),
        dust_remove=False,
        dust_threshold=0.75,
        dust_size=2,
        manual_spots=[spot],
        scale_factor=1.0,
    )

    dest_after = res[130:160, 30:60]
    # The healed area should differ from the original plain region
    assert not np.allclose(dest_after, dest_region_before, atol=0.02), (
        "Expected healed dest to differ from original plain region (texture transfer)"
    )


def test_poisson_heal_small_radius_fallback_no_crash():
    """Radius < 3 px triggers direct-copy fallback and must not crash."""
    img = np.full((50, 50, 3), 0.5, dtype=np.float32)
    img[23:27, 23:27] = 1.0

    spot = _spot(0.5, 0.5, 0.6, 0.5, 1.0)  # radius=1, well below threshold of 3
    res = apply_dust_removal(
        img.copy(),
        dust_remove=False,
        dust_threshold=0.75,
        dust_size=2,
        manual_spots=[spot],
        scale_factor=1.0,
    )

    assert res.shape == img.shape
    assert res.dtype == np.float32


def test_poisson_heal_two_pixels_radius_fallback():
    """Radius=2 at scale 1 (still < 3) uses fallback and returns valid float32 image."""
    img = np.full((60, 60, 3), 0.3, dtype=np.float32)
    img[28:32, 28:32] = 0.9

    spot = _spot(0.5, 0.5, 0.65, 0.5, 2.0)
    res = apply_dust_removal(
        img.copy(),
        dust_remove=False,
        dust_threshold=0.75,
        dust_size=2,
        manual_spots=[spot],
        scale_factor=1.0,
    )

    assert res.shape == img.shape
    assert 0.0 <= res.max() <= 1.0 + 1e-5


# --- Laplace heal (darktable port) ------------------------------------------


from negpy.features.retouch.logic import _heal_spot_laplace  # noqa: F401, E402


def test_laplace_solver_smooths_interior():
    """Solver fills a masked interior with smooth interpolation from boundary."""
    import numpy as np
    from negpy.features.retouch.logic import _heal_laplace_nb

    h, w = 21, 21
    # diff = 1.0 everywhere; mask = 1.0 in inner circle, 0.0 on outer ring (boundary BC)
    diff = np.ones((h, w, 3), dtype=np.float32)
    mask = np.zeros((h, w), dtype=np.float32)
    cy, cx, r = h // 2, w // 2, 7
    for y in range(h):
        for x in range(w):
            if (y - cy) ** 2 + (x - cx) ** 2 < r**2:
                mask[y, x] = 1.0

    # Manually set boundary diff = 0.0 so solver converges to 0 inside
    for y in range(h):
        for x in range(w):
            if mask[y, x] == 0.0:
                diff[y, x] = 0.0

    _heal_laplace_nb(diff, mask, max_iter=2000)

    # After convergence, interior should be close to 0 (Laplace = average of boundary = 0)
    interior_max = float(np.max(np.abs(diff[mask > 0])))
    assert interior_max < 0.05, f"Interior not smoothed: max abs = {interior_max:.4f}"


def test_laplace_heal_removes_bright_defect():
    """_heal_spot_laplace heals a bright disc from a clean grey source."""
    import numpy as np
    from negpy.features.retouch.models import RetouchSpot

    rng = np.random.default_rng(0)
    img = rng.uniform(0.3, 0.5, (80, 80, 3)).astype(np.float32)
    cy, cx, r = 40, 40, 10
    for y in range(cy - r, cy + r + 1):
        for x in range(cx - r, cx + r + 1):
            if (y - cy) ** 2 + (x - cx) ** 2 <= r**2:
                img[y, x] = 0.95  # bright defect

    spot = RetouchSpot(
        dest_x=cx / 80,
        dest_y=cy / 80,
        source_x=15 / 80,
        source_y=15 / 80,
        radius=float(r),
    )
    result = _heal_spot_laplace(img.copy(), spot, scale_factor=1.0)

    orig_peak = float(img[cy, cx, 0])
    healed_peak = float(result[cy, cx, 0])
    assert healed_peak < orig_peak - 0.2, f"Expected centre darkened by >0.2; got {orig_peak:.3f} → {healed_peak:.3f}"


def test_laplace_heal_tiny_radius_fallback():
    """Radius < 2 uses direct-copy fallback (no solver crash)."""
    import numpy as np
    from negpy.features.retouch.models import RetouchSpot

    img = np.full((30, 30, 3), 0.5, dtype=np.float32)
    img[14:16, 14:16] = 1.0
    spot = RetouchSpot(dest_x=0.5, dest_y=0.5, source_x=0.7, source_y=0.5, radius=1.0)
    result = _heal_spot_laplace(img.copy(), spot, scale_factor=1.0)
    assert result.shape == img.shape
    assert result.dtype == np.float32
    assert 0.0 <= float(result.max()) <= 1.0 + 1e-5


# ---------------------------------------------------------------------------
# GPU heal smoke test
# ---------------------------------------------------------------------------


def _gpu_is_available() -> bool:
    try:
        from negpy.infrastructure.gpu.device import GPUDevice

        return GPUDevice.get().is_available
    except Exception:
        return False


@pytest.mark.skipif(not _gpu_is_available(), reason="GPU not available in this environment")
def test_gpu_heal_pipeline_smoke():
    """GPU pipeline with manual heal spots produces finite output without crashing."""
    import dataclasses

    import numpy as np

    from negpy.domain.models import WorkspaceConfig
    from negpy.features.retouch.models import RetouchConfig, RetouchSpot
    from negpy.services.rendering.gpu_engine import GPUEngine

    img = np.random.default_rng(1).uniform(0.2, 0.8, (64, 64, 3)).astype(np.float32)
    img[28:36, 28:36] = 0.95  # bright defect

    spot = RetouchSpot(dest_x=0.5, dest_y=0.5, source_x=0.2, source_y=0.2, radius=6.0)
    settings = dataclasses.replace(
        WorkspaceConfig(),
        retouch=RetouchConfig(manual_spots=[spot], dust_remove=False),
    )

    engine = GPUEngine()
    result, _ = engine.process(img, settings, scale_factor=1.0)

    assert result is not None
    assert np.all(np.isfinite(result)), "GPU heal output contains non-finite values"
    assert result.shape[0] > 0
    # The bright defect at rows 28-36, cols 28-36 should be attenuated
    defect_before = float(img[28:36, 28:36].mean())
    defect_after = float(result[28:36, 28:36].mean())
    assert defect_after < defect_before, f"GPU heal did not attenuate defect: {defect_before:.3f} → {defect_after:.3f}"
