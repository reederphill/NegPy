import math
import numpy as np
import cv2
from numba import njit  # type: ignore
from typing import List
from negpy.domain.types import ImageBuffer, LUMA_R, LUMA_G, LUMA_B
from negpy.features.retouch.models import RetouchSpot
from negpy.kernel.image.validation import ensure_image
from negpy.kernel.image.logic import get_luminance


@njit(cache=True, fastmath=True)
def _apply_auto_retouch_jit(
    img: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    w_std: np.ndarray,
    dust_threshold: float,
    dust_size: float,
    scale_factor: float,
) -> np.ndarray:
    h, w, c = img.shape
    hit_mask = np.zeros((h, w), dtype=np.float32)

    # 1. Detection Pass
    for y in range(h):
        for x in range(w):
            l_curr = LUMA_R * img[y, x, 0] + LUMA_G * img[y, x, 1] + LUMA_B * img[y, x, 2]
            l_mean = mean[y, x]
            local_s = max(0.005, std[y, x])

            # Wide-area penalty for textures (rocks, foliage)
            w_s = max(0.0, w_std[y, x] - 0.02)
            wide_penalty = (w_s * w_s * w_s) * 800.0
            thresh = (dust_threshold * 0.4) + (local_s * 1.0) + wide_penalty

            # Multi-stage validation: Contrast, Luminance, and Z-Score
            if (l_curr - l_mean) > thresh and l_curr > 0.15 and (l_curr - l_mean) / local_s > 3.0:
                is_strong = (l_curr - l_mean) > (thresh * 2.5) or (l_curr - l_mean) > 0.25

                if 0 < y < h - 1 and 0 < x < w - 1:
                    is_max = True
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            if dy == 0 and dx == 0:
                                continue
                            nl = LUMA_R * img[y + dy, x + dx, 0] + LUMA_G * img[y + dy, x + dx, 1] + LUMA_B * img[y + dy, x + dx, 2]
                            if nl >= l_curr:
                                is_max = False
                                break
                        if not is_max:
                            break
                    if is_max or is_strong:
                        hit_mask[y, x] = 1.0
                else:
                    hit_mask[y, x] = 1.0

    # 2. Healing Pass: Stochastic Perimeter Sampling (SPS) with Soft Blending
    res = img.copy()
    exp_rad = int(max(1.0, dust_size * 0.4 * scale_factor))
    if exp_rad > 16:
        exp_rad = 16
    p_rad = exp_rad + int(3 * scale_factor)

    for y in range(h):
        for x in range(w):
            min_d2 = 1e6
            for dy in range(-exp_rad, exp_rad + 1):
                for dx in range(-exp_rad, exp_rad + 1):
                    ry, rx = y + dy, x + dx
                    if 0 <= ry < h and 0 <= rx < w and hit_mask[ry, rx] > 0.5:
                        d2 = float(dy * dy + dx * dx)
                        if d2 < min_d2:
                            min_d2 = d2

            if min_d2 < float(exp_rad * exp_rad + 1):
                dist = np.sqrt(min_d2)
                feather = 1.0 - (dist / float(exp_rad + 1.0))
                if feather < 0.0:
                    feather = 0.0
                feather = feather * feather * (3.0 - 2.0 * feather)

                if feather > 0.001:
                    s_r = np.zeros(8)
                    s_g = np.zeros(8)
                    s_b = np.zeros(8)
                    s_l = np.zeros(8)

                    # 8-point perimeter sampling
                    dy_off = np.array([-p_rad, p_rad, 0, 0, -p_rad, -p_rad, p_rad, p_rad])
                    dx_off = np.array([0, 0, -p_rad, p_rad, -p_rad, p_rad, -p_rad, p_rad])

                    for i in range(8):
                        sy, sx = y + dy_off[i], x + dx_off[i]
                        sy, sx = max(0, min(h - 1, sy)), max(0, min(w - 1, sx))
                        r, g, b = img[sy, sx, 0], img[sy, sx, 1], img[sy, sx, 2]
                        s_r[i], s_g[i], s_b[i] = r, g, b
                        s_l[i] = 0.2126 * r + 0.7152 * g + 0.0722 * b

                    # Selection sort for outlier rejection
                    for i in range(8):
                        for j in range(i + 1, 8):
                            if s_l[i] > s_l[j]:
                                s_l[i], s_l[j] = s_l[j], s_l[i]
                                s_r[i], s_r[j] = s_r[j], s_r[i]
                                s_g[i], s_g[j] = s_g[j], s_g[i]
                                s_b[i], s_b[j] = s_b[j], s_b[i]

                    # Average middle 50% (discard 2 brightest, 2 darkest)
                    bg_r = (s_r[2] + s_r[3] + s_r[4] + s_r[5]) / 4.0
                    bg_g = (s_g[2] + s_g[3] + s_g[4] + s_g[5]) / 4.0
                    bg_b = (s_b[2] + s_b[3] + s_b[4] + s_b[5]) / 4.0

                    res[y, x, 0] = img[y, x, 0] * (1.0 - feather) + bg_r * feather
                    res[y, x, 1] = img[y, x, 1] * (1.0 - feather) + bg_g * feather
                    res[y, x, 2] = img[y, x, 2] * (1.0 - feather) + bg_b * feather

    return res


@njit(cache=True, fastmath=True)
def _heal_laplace_nb(diff: np.ndarray, mask: np.ndarray, max_iter: int) -> None:
    """In-place red/black Gauss-Seidel Laplace solve — darktable heal.c port.

    diff: float32 (H, W, C) — initialised to dest-src; boundary pixels (mask==0) are
          fixed Dirichlet BCs; interior pixels (mask>0) are updated in-place.
    mask: float32 (H, W) — 1.0 inside the spot circle, 0.0 outside.
    """
    h, w, nc = diff.shape

    nmask = 0
    for y in range(h):
        for x in range(w):
            if mask[y, x] > 0:
                nmask += 1

    if nmask == 0:
        return

    sor_w = (2.0 - 1.0 / (0.1575 * math.sqrt(float(nmask)) + 0.8)) * 0.25
    eps = 0.1 / 255.0
    err_exit = eps * eps * sor_w * sor_w

    for _ in range(max_iter):
        total_err = 0.0
        for parity in range(2):
            for y in range(h):
                for x in range(w):
                    if (x + y) % 2 != parity:
                        continue
                    if mask[y, x] == 0:
                        continue
                    a = 4.0
                    if y == 0:
                        a -= 1.0
                    if y == h - 1:
                        a -= 1.0
                    if x == 0:
                        a -= 1.0
                    if x == w - 1:
                        a -= 1.0
                    for c in range(nc):
                        ns = 0.0
                        if y > 0:
                            ns += diff[y - 1, x, c]
                        if y < h - 1:
                            ns += diff[y + 1, x, c]
                        if x > 0:
                            ns += diff[y, x - 1, c]
                        if x < w - 1:
                            ns += diff[y, x + 1, c]
                        d = sor_w * (a * diff[y, x, c] - ns)
                        diff[y, x, c] -= d
                        total_err += d * d
        if total_err < err_exit:
            break


def _heal_spot_laplace(img: np.ndarray, spot: RetouchSpot, scale_factor: float, max_iter: int = 2000) -> np.ndarray:
    """Apply darktable-style Gauss-Seidel Laplace heal for a single spot."""
    h_img, w_img = img.shape[:2]

    dx = int(round(spot.dest_x * w_img))
    dy = int(round(spot.dest_y * h_img))
    sx = int(round(spot.source_x * w_img))
    sy = int(round(spot.source_y * h_img))
    r = int(max(1, spot.radius * scale_factor))

    dx = int(np.clip(dx, 0, w_img - 1))
    dy = int(np.clip(dy, 0, h_img - 1))
    sx = int(np.clip(sx, 0, w_img - 1))
    sy = int(np.clip(sy, 0, h_img - 1))

    dy0, dy1 = max(0, dy - r), min(h_img, dy + r + 1)
    dx0, dx1 = max(0, dx - r), min(w_img, dx + r + 1)
    ph = dy1 - dy0
    pw = dx1 - dx0

    if r < 2:
        result = img.copy()
        ys = np.clip(sy + np.arange(ph) + dy0 - dy, 0, h_img - 1).astype(int)
        xs = np.clip(sx + np.arange(pw) + dx0 - dx, 0, w_img - 1).astype(int)
        result[dy0:dy1, dx0:dx1] = img[np.ix_(ys, xs)]
        return ensure_image(result)

    dest_patch = img[dy0:dy1, dx0:dx1].astype(np.float32)

    ys = np.clip(sy + np.arange(ph) + dy0 - dy, 0, h_img - 1).astype(int)
    xs = np.clip(sx + np.arange(pw) + dx0 - dx, 0, w_img - 1).astype(int)
    src_patch = img[np.ix_(ys, xs)].astype(np.float32)

    py_arr = (np.arange(ph) + dy0 - dy).astype(np.float32)
    px_arr = (np.arange(pw) + dx0 - dx).astype(np.float32)
    cy_grid, cx_grid = np.meshgrid(py_arr, px_arr, indexing="ij")
    mask = ((cy_grid**2 + cx_grid**2) <= float(r * r)).astype(np.float32)

    diff = (dest_patch - src_patch).astype(np.float32)
    _heal_laplace_nb(diff, mask, max_iter)

    result = img.copy()
    healed = np.clip(src_patch + diff, 0.0, 1.0)
    result[dy0:dy1, dx0:dx1] = np.where(mask[:, :, np.newaxis] > 0, healed, dest_patch)
    return ensure_image(result)


def apply_dust_removal(
    img: ImageBuffer,
    dust_remove: bool,
    dust_threshold: float,
    dust_size: int,
    manual_spots: List[RetouchSpot],
    scale_factor: float,
) -> ImageBuffer:
    if not (dust_remove or manual_spots):
        return img

    if dust_remove:
        base_size, scale = max(1.0, float(dust_size)), max(1.0, float(scale_factor))
        v_win = int(max(3, base_size * 3.0 * scale)) * 2 + 1
        w_win = int(max(7, base_size * 4.0 * scale)) * 2 + 1

        gray = get_luminance(img)
        mean_gray = cv2.blur(gray, (v_win, v_win))
        std_gray = np.sqrt(np.clip(cv2.blur(gray**2, (v_win, v_win)) - mean_gray**2, 0, None))
        w_mean_gray = cv2.blur(gray, (w_win, w_win))
        w_std_gray = np.sqrt(np.clip(cv2.blur(gray**2, (w_win, w_win)) - w_mean_gray**2, 0, None))

        img = _apply_auto_retouch_jit(
            np.ascontiguousarray(img.astype(np.float32)),
            np.ascontiguousarray(mean_gray.astype(np.float32)),
            np.ascontiguousarray(std_gray.astype(np.float32)),
            np.ascontiguousarray(w_std_gray.astype(np.float32)),
            float(dust_threshold),
            float(dust_size),
            float(scale_factor),
        )

    for spot in manual_spots:
        img = _heal_spot_laplace(np.ascontiguousarray(img.astype(np.float32)), spot, float(scale_factor))

    return ensure_image(img)
