import hashlib
from unittest.mock import patch

from negpy.kernel.caching.logic import calculate_config_hash
from negpy.features.exposure.models import ExposureConfig


def test_identical_config_does_not_rehash():
    """Calling calculate_config_hash twice with identical config should not MD5-hash twice."""
    cfg = ExposureConfig()

    md5_call_count = {"n": 0}
    from negpy.kernel.caching.logic import _md5_of_serialized

    original_md5 = hashlib.md5

    def counting_md5(*args, **kwargs):
        md5_call_count["n"] += 1
        return original_md5(*args, **kwargs)

    _md5_of_serialized.cache_clear()
    try:
        with patch("negpy.kernel.caching.logic.hashlib.md5", side_effect=counting_md5):
            h1 = calculate_config_hash(cfg)
            h2 = calculate_config_hash(cfg)
    finally:
        _md5_of_serialized.cache_clear()

    assert h1 == h2
    assert md5_call_count["n"] == 1, (
        f"hashlib.md5 called {md5_call_count['n']} times for two identical configs; "
        "expected 1 (second call should be served from lru_cache)."
    )
