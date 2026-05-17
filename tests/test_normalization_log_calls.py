from unittest.mock import patch

import numpy as np

from negpy.domain.interfaces import PipelineContext
from negpy.features.exposure.processor import NormalizationProcessor
from negpy.features.process.models import ProcessConfig


def _make_image(h=32, w=32):
    rng = np.random.default_rng(42)
    return (rng.random((h, w, 3)) * 0.8 + 0.1).astype(np.float32)


def test_log10_called_exactly_once_during_normalization():
    """np.log10 should be called once per process() call, not twice."""
    image = _make_image()
    config = ProcessConfig()
    context = PipelineContext(scale_factor=1.0, original_size=(32, 32), process_mode="c41")

    log_call_count = {"n": 0}
    original_log10 = np.log10

    def counting_log10(x):
        log_call_count["n"] += 1
        return original_log10(x)

    with patch("negpy.features.exposure.normalization.np.log10", side_effect=counting_log10):
        with patch("negpy.features.exposure.processor.np.log10", side_effect=counting_log10):
            proc = NormalizationProcessor(config)
            proc.process(image, context)

    assert log_call_count["n"] == 1, (
        f"np.log10 called {log_call_count['n']} times; expected 1. analyze_log_exposure_bounds should receive img_log, not raw image."
    )
