from __future__ import annotations

from negpy.desktop.prefetch_logic import neighbor_indices, neighbor_paths_and_hashes


def test_neighbor_indices() -> None:
    assert neighbor_indices(3, 0) == [1]
    assert neighbor_indices(3, 1) == [0, 2]
    assert neighbor_indices(1, 0) == []


def test_neighbor_paths_and_hashes() -> None:
    files = [
        {"path": "/a", "hash": "ha"},
        {"path": "/b", "hash": "hb"},
        {"path": "/c", "hash": "hc"},
    ]
    assert neighbor_paths_and_hashes(files, 0) == [("/b", "hb")]
    assert set(neighbor_paths_and_hashes(files, 1)) == {("/a", "ha"), ("/c", "hc")}
