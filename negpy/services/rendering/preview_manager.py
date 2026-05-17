import io
import time
from typing import Any, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

import rawpy

from negpy.domain.types import Dimensions, ImageBuffer
from negpy.infrastructure.display.color_spaces import ColorSpaceRegistry
from negpy.infrastructure.loaders.factory import loader_factory
from negpy.infrastructure.loaders.helpers import NonStandardFileWrapper, get_best_demosaic_algorithm
from negpy.kernel.image.logic import ensure_rgb, uint16_to_float32
from negpy.kernel.image.validation import ensure_image
from negpy.kernel.system.config import APP_CONFIG
from negpy.kernel.system.logging import get_logger
from negpy.services.rendering.preview_cache import PreviewBufferCache, PreviewCacheKey

logger = get_logger(__name__)


def _output_dimensions_from_raw(raw: Any, postprocessed_h: int, postprocessed_w: int) -> Tuple[int, int]:
    """
    Returns (height, width) of the full-resolution image in image space, not the half_size postprocess output.
    """
    if isinstance(raw, NonStandardFileWrapper):
        h, w = raw.data.shape[0], raw.data.shape[1]
        return (int(h), int(w))
    try:
        s = raw.sizes
        for pair in (("iheight", "iwidth"), ("raw_height", "raw_width"), ("height", "width")):
            h_attr, w_attr = pair
            if hasattr(s, h_attr) and hasattr(s, w_attr):
                h = int(getattr(s, h_attr))
                w = int(getattr(s, w_attr))
                if h > 0 and w > 0:
                    return (h, w)
    except Exception:
        pass
    return (postprocessed_h, postprocessed_w)


class PreviewManager:
    """
    Loads RAW (and other) files for UI preview, with in-memory LRU and fast decode.
    """

    def __init__(self) -> None:
        self._cache = PreviewBufferCache(APP_CONFIG)

    @staticmethod
    def try_splash_preview(file_path: str) -> Optional[Tuple[ImageBuffer, Dimensions]]:
        """
        Quick embedded-JPEG (or half-size) RGB for first paint. Returns None if not available.
        """
        t0 = time.perf_counter()
        try:
            ctx_mgr, _metadata = loader_factory.get_loader(file_path)
        except Exception:
            return None
        try:
            with ctx_mgr as raw:
                if not hasattr(raw, "extract_thumb"):
                    return None
                try:
                    thumb = raw.extract_thumb()
                except Exception:
                    return None
                img: Optional[Image.Image] = None
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    img = Image.open(io.BytesIO(thumb.data))
                elif thumb.format == rawpy.ThumbFormat.BITMAP:
                    img = Image.fromarray(thumb.data)
                if img is None:
                    return None
                img = img.convert("RGB")
                arr = np.ascontiguousarray(np.array(img, dtype=np.float32) / 255.0)
                h, w = arr.shape[:2]
                if max(h, w) > APP_CONFIG.preview_render_size:
                    scale = APP_CONFIG.preview_render_size / max(h, w)
                    tw, th = int(w * scale), int(h * scale)
                    arr = ensure_image(cv2.resize(arr, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32))
                dh, dw = arr.shape[:2]
                full_dims = _output_dimensions_from_raw(raw, dh, dw)
                logger.debug("preview try_splash_preview ok %.3fs for %s", time.perf_counter() - t0, file_path)
                return ensure_image(arr), full_dims
        except Exception as e:
            logger.debug("preview splash skip: %s", e)
        return None

    def load_linear_preview(
        self,
        file_path: str,
        color_space: str | None = None,
        use_camera_wb: bool = False,
        full_resolution: bool = False,
        file_hash: str | None = None,
    ) -> Tuple[ImageBuffer, Dimensions, dict]:
        """
        Loads linear RGB, downsamples for display.
        If color_space is None, uses the source's declared space (metadata).
        """
        t_all = time.perf_counter()
        ctx_mgr, metadata = loader_factory.get_loader(file_path)

        if color_space is None:
            color_space = metadata.get("color_space", "Adobe RGB")

        if file_hash:
            ck = PreviewCacheKey(
                file_hash=file_hash,
                use_camera_wb=use_camera_wb,
                workspace_color_space=color_space,
                full_resolution=full_resolution,
            )
            hit = self._cache.get(ck)
            if hit is not None:
                buf, dims, meta = hit
                logger.debug(
                    "preview cache hit total %.3fs for %s",
                    time.perf_counter() - t_all,
                    file_path,
                )
                return ensure_image(buf.copy()), dims, meta

        raw_color_space = ColorSpaceRegistry.get_rawpy_space(color_space)
        t_decode = time.perf_counter()
        with ctx_mgr as raw:
            use_fast = (not full_resolution) and (not isinstance(raw, NonStandardFileWrapper))
            if use_fast:
                demosaic = rawpy.DemosaicAlgorithm.LINEAR
                post_kw: dict = {"half_size": True}
            else:
                demosaic = get_best_demosaic_algorithm(raw)
                post_kw = {}

            user_wb = None if use_camera_wb else [1, 1, 1, 1]

            t_pp = time.perf_counter()
            rgb = raw.postprocess(
                gamma=(1, 1),
                no_auto_bright=True,
                use_camera_wb=use_camera_wb,
                user_wb=user_wb,
                output_bps=16,
                output_color=raw_color_space,
                demosaic_algorithm=demosaic,
                user_flip=0,
                **post_kw,
            )
            logger.debug("raw.postprocess %.3fs (fast=%s)", time.perf_counter() - t_pp, use_fast)
            rgb = ensure_rgb(rgb)

            full_linear = uint16_to_float32(np.ascontiguousarray(rgb))
            h_p, w_p = full_linear.shape[:2]
            h_orig, w_orig = _output_dimensions_from_raw(raw, h_p, w_p)
            t_resize0 = time.perf_counter()
            max_res = APP_CONFIG.preview_render_size
            if max(h_p, w_p) > max_res and not full_resolution:
                scale = max_res / max(h_p, w_p)
                target_w = int(w_p * scale)
                target_h = int(h_p * scale)
                preview_raw = ensure_image(
                    cv2.resize(
                        full_linear,
                        (target_w, target_h),
                        interpolation=cv2.INTER_AREA,
                    )
                )
            else:
                preview_raw = full_linear.copy()
            logger.debug("preview resize+convert %.3fs", time.perf_counter() - t_resize0)
        out = ensure_image(preview_raw)
        logger.debug(
            "PreviewManager.load_linear_preview decode+resize %.3fs (total %.3fs)",
            time.perf_counter() - t_decode,
            time.perf_counter() - t_all,
        )
        if file_hash:
            ck = PreviewCacheKey(
                file_hash=file_hash,
                use_camera_wb=use_camera_wb,
                workspace_color_space=color_space,
                full_resolution=full_resolution,
            )
            self._cache.put(ck, out.copy(), (h_orig, w_orig), dict(metadata))
        return out, (h_orig, w_orig), metadata
