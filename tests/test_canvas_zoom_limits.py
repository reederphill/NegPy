from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from negpy.desktop.view.canvas.widget import clamp_canvas_zoom_level


def test_clamp_canvas_zoom_level_uses_config() -> None:
    fake = SimpleNamespace(canvas_zoom_min=0.25, canvas_zoom_max=8.0)
    with patch("negpy.desktop.view.canvas.widget.APP_CONFIG", fake):
        assert clamp_canvas_zoom_level(9.0) == 8.0
        assert clamp_canvas_zoom_level(0.01) == 0.25
        assert clamp_canvas_zoom_level(2.0) == 2.0


def test_toolbar_range_matches_clamp_extremes() -> None:
    from negpy.kernel.system.config import APP_CONFIG

    zminp = int(APP_CONFIG.canvas_zoom_min * 100)
    zmaxp = int(APP_CONFIG.canvas_zoom_max * 100)
    assert clamp_canvas_zoom_level(zminp / 100.0) == APP_CONFIG.canvas_zoom_min
    assert clamp_canvas_zoom_level(zmaxp / 100.0) == APP_CONFIG.canvas_zoom_max
