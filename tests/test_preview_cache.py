from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from negpy.services.rendering.preview_cache import PreviewBufferCache, PreviewCacheKey
from negpy.services.rendering.preview_manager import PreviewManager
from negpy.desktop.workers.render import PreviewLoadTask, PreviewLoadWorker


def _small_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        preview_cache_max_entries=2,
        preview_cache_max_bytes=10**9,
        preview_render_size=2000,
        canvas_zoom_min=0.25,
        canvas_zoom_max=8.0,
    )


def test_cache_eviction_by_count() -> None:
    c = PreviewBufferCache(_small_cfg())
    a = np.zeros((4, 4, 3), dtype=np.float32)
    b = np.ones((4, 4, 3), dtype=np.float32)
    c.put(PreviewCacheKey("h1", False, "Adobe RGB", False), a, (4, 4), {})
    c.put(PreviewCacheKey("h2", False, "Adobe RGB", False), b, (4, 4), {})
    c.put(PreviewCacheKey("h3", False, "Adobe RGB", False), a.copy(), (4, 4), {})
    assert c.get(PreviewCacheKey("h1", False, "Adobe RGB", False)) is None
    assert c.get(PreviewCacheKey("h3", False, "Adobe RGB", False)) is not None


def test_cache_bypasses_second_postprocess() -> None:
    """After first load, second load with same key must not call raw.postprocess."""
    import rawpy

    rgb = np.zeros((8, 8, 3), dtype=np.uint16)
    raw = MagicMock()
    raw.raw_type = rawpy.RawType.Flat
    raw.raw_pattern = np.zeros((2, 2), dtype=np.uint8)
    raw.sizes = SimpleNamespace(raw_height=8, raw_width=8, iheight=8, iwidth=8)
    n_calls = [0]

    def _pp(**kwargs: object) -> object:
        n_calls[0] += 1
        return rgb

    raw.postprocess = _pp

    class _Ctx:
        def __enter__(self) -> MagicMock:
            return raw

        def __exit__(self, *args: object) -> None:
            return None

    pm = PreviewManager()
    with (
        patch("negpy.services.rendering.preview_manager.loader_factory") as lf,
        patch("negpy.services.rendering.preview_manager.APP_CONFIG", _small_cfg()),
    ):
        lf.get_loader.return_value = (_Ctx(), {"color_space": "Adobe RGB"})
        pm.load_linear_preview("/x.dng", file_hash="abc")
        assert n_calls[0] == 1
        pm.load_linear_preview("/x.dng", file_hash="abc")
        assert n_calls[0] == 1


def test_cache_warm_task_does_not_emit_finished() -> None:
    """Prefetch jobs populate cache only — no `finished` to the UI path."""
    pm = MagicMock()
    pm.load_linear_preview.return_value = (MagicMock(), (1, 1), {})
    w = PreviewLoadWorker(pm)
    fin = MagicMock()
    w.finished.connect(fin)
    t = PreviewLoadTask(
        file_path="/n.dng",
        workspace_color_space="Adobe RGB",
        use_camera_wb=False,
        for_cache_warm=True,
        file_hash="x",
    )
    w.process(t)
    fin.assert_not_called()
    pm.load_linear_preview.assert_called_once()
