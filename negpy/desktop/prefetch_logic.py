from __future__ import annotations

from typing import List, Optional, Tuple


def neighbor_indices(n_files: int, current_index: int) -> List[int]:
    """
    Returns actual list indices for previous and next file, in-bounds.
    """
    if n_files <= 0 or current_index < 0 or current_index >= n_files:
        return []
    out: List[int] = []
    if current_index > 0:
        out.append(current_index - 1)
    if current_index + 1 < n_files:
        out.append(current_index + 1)
    return out


def neighbor_paths_and_hashes(files: List[dict], current_index: int) -> List[Tuple[str, Optional[str]]]:
    """
    (path, hash) for prev/next neighbors; hash may be None.
    """
    ni = neighbor_indices(len(files), current_index)
    return [(files[i]["path"], files[i].get("hash")) for i in ni]
