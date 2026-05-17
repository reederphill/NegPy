from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from negpy.services.rendering.preview_cache import PreviewBufferCache, PreviewCacheKey


def _cfg(max_entries: int = 100, max_bytes: int = 10 * 1024**3) -> SimpleNamespace:
    return SimpleNamespace(
        preview_cache_max_entries=max_entries,
        preview_cache_max_bytes=max_bytes,
    )


def _make_key(i: int) -> PreviewCacheKey:
    return PreviewCacheKey(
        file_hash=f"hash{i}",
        use_camera_wb=False,
        workspace_color_space="Adobe RGB",
        full_resolution=False,
    )


def _make_buf(h: int = 4, w: int = 4) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.float32)


def test_lru_evicts_oldest_first() -> None:
    """LRU eviction should remove the least-recently-used entry."""
    cache = PreviewBufferCache(_cfg(max_entries=3))
    for i in range(3):
        cache.put(_make_key(i), _make_buf(), (4, 4), {})

    # Access key 0 to make it recently used — key 1 becomes least-recently-used
    cache.get(_make_key(0))

    # Adding a 4th entry should evict key 1 (oldest not recently accessed)
    cache.put(_make_key(3), _make_buf(), (4, 4), {})

    assert cache.get(_make_key(0)) is not None, "key 0 was recently used — should still be cached"
    assert cache.get(_make_key(1)) is None, "key 1 was least-recently-used — should have been evicted"
    assert cache.get(_make_key(2)) is not None, "key 2 should still be cached"
    assert cache.get(_make_key(3)) is not None, "key 3 was just added — should be cached"


def test_total_bytes_tracked_correctly() -> None:
    """_total_bytes counter should stay accurate through puts, gets, and evictions."""
    cache = PreviewBufferCache(_cfg())
    buf = _make_buf(8, 8)  # 8*8*3*4 = 768 bytes

    cache.put(_make_key(0), buf, (8, 8), {})
    cache.put(_make_key(1), buf, (8, 8), {})

    expected_bytes = buf.nbytes * 2
    assert cache._total_bytes == expected_bytes, f"Expected {expected_bytes} bytes tracked, got {cache._total_bytes}"

    cache._remove_key(_make_key(0).as_tuple())
    assert cache._total_bytes == buf.nbytes, f"After removal, expected {buf.nbytes} bytes, got {cache._total_bytes}"

    cache.clear()
    assert cache._total_bytes == 0


def test_byte_eviction_respects_limit() -> None:
    """Cache should evict oldest entries when byte limit is exceeded."""
    buf = _make_buf(4, 4)  # 4*4*3*4 = 192 bytes each
    byte_limit = buf.nbytes * 2 + 1  # allow 2 entries

    cache = PreviewBufferCache(_cfg(max_bytes=byte_limit))
    for i in range(3):
        cache.put(_make_key(i), buf, (4, 4), {})

    assert len(cache._lru) <= 2, f"Expected at most 2 entries after byte eviction, got {len(cache._lru)}"
    assert cache._total_bytes <= byte_limit


def test_put_update_does_not_double_count_bytes() -> None:
    """Re-putting a key with a new buffer should not accumulate bytes for the old entry."""
    cache = PreviewBufferCache(_cfg())
    buf_small = _make_buf(4, 4)
    buf_large = _make_buf(8, 8)

    cache.put(_make_key(0), buf_small, (4, 4), {})
    cache.put(_make_key(0), buf_large, (8, 8), {})  # overwrite same key

    assert cache._total_bytes == buf_large.nbytes, f"Expected only large buffer bytes ({buf_large.nbytes}), got {cache._total_bytes}"


def test_invalidate_path_hash_removes_correct_entries() -> None:
    """invalidate_path_hash should remove only entries matching that file hash."""
    cache = PreviewBufferCache(_cfg())
    buf = _make_buf()

    # Two keys for "hash0", one for "hash1"
    key_a = PreviewCacheKey("hash0", False, "Adobe RGB", False)
    key_b = PreviewCacheKey("hash0", True, "Adobe RGB", False)
    key_c = PreviewCacheKey("hash1", False, "Adobe RGB", False)

    cache.put(key_a, buf, (4, 4), {})
    cache.put(key_b, buf, (4, 4), {})
    cache.put(key_c, buf, (4, 4), {})

    cache.invalidate_path_hash("hash0")

    assert cache.get(key_a) is None, "key_a (hash0) should have been invalidated"
    assert cache.get(key_b) is None, "key_b (hash0) should have been invalidated"
    assert cache.get(key_c) is not None, "key_c (hash1) should remain"
    assert cache._total_bytes == buf.nbytes, "Only hash1 entry bytes should remain"
