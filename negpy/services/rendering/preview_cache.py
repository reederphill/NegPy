from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Hashable, Optional


from negpy.domain.types import AppConfig, Dimensions, ImageBuffer
from negpy.kernel.system.config import APP_CONFIG
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PreviewCacheKey:
    file_hash: str
    use_camera_wb: bool
    workspace_color_space: str
    full_resolution: bool

    def as_tuple(self) -> Hashable:
        return (
            self.file_hash,
            self.use_camera_wb,
            self.workspace_color_space,
            self.full_resolution,
        )


@dataclass
class _Entry:
    buffer: ImageBuffer
    dims: Dimensions
    metadata: dict
    byte_size: int


class PreviewBufferCache:
    """In-memory LRU for decoded linear preview buffers. Evicts by entry count and approximate RSS."""

    def __init__(self, app_config: Optional[AppConfig] = None) -> None:
        self._app = app_config or APP_CONFIG
        self._lru: OrderedDict[Hashable, _Entry] = OrderedDict()
        self._total_bytes: int = 0

    def get(self, key: PreviewCacheKey) -> Optional[tuple[ImageBuffer, Dimensions, dict]]:
        t = key.as_tuple()
        ent = self._lru.get(t)
        if ent is None:
            return None
        self._lru.move_to_end(t)  # O(1) — mark as most recently used
        return ent.buffer, ent.dims, ent.metadata

    def put(self, key: PreviewCacheKey, buffer: ImageBuffer, dims: Dimensions, metadata: dict) -> None:
        t = key.as_tuple()
        b = int(buffer.nbytes)
        if t in self._lru:
            self._total_bytes -= self._lru[t].byte_size
            del self._lru[t]
        self._lru[t] = _Entry(buffer=buffer, dims=dims, metadata=dict(metadata), byte_size=b)
        self._total_bytes += b
        self._evict_if_needed()

    def invalidate_path_hash(self, file_hash: str) -> None:
        to_drop = [k for k in self._lru if isinstance(k, tuple) and k and k[0] == file_hash]
        for t in to_drop:
            self._remove_key(t)

    def clear(self) -> None:
        self._lru.clear()
        self._total_bytes = 0

    def _remove_key(self, t: Hashable) -> None:
        ent = self._lru.pop(t, None)
        if ent is not None:
            self._total_bytes -= ent.byte_size

    def _evict_if_needed(self) -> None:
        max_n = self._app.preview_cache_max_entries
        max_b = self._app.preview_cache_max_bytes

        while len(self._lru) > max_n:
            t, ent = next(iter(self._lru.items()))  # oldest entry = first in OrderedDict
            del self._lru[t]
            self._total_bytes -= ent.byte_size
            logger.debug("preview cache evict (count): dropped entry")

        while self._total_bytes > max_b and self._lru:
            t, ent = next(iter(self._lru.items()))
            del self._lru[t]
            self._total_bytes -= ent.byte_size
            logger.debug("preview cache evict (bytes): dropped entry")
