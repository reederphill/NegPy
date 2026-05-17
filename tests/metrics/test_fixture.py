"""Unit tests for fixture.py — all network calls are mocked."""

from __future__ import annotations

from pathlib import Path

import pytest

from . import fixture


def test_env_var_takes_priority(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    real = tmp_path / "real.raf"
    real.write_bytes(b"x")
    monkeypatch.setenv("NEGPY_PERF_RAW", str(real))
    assert fixture.get_perf_raw_path() == str(real)


def test_env_var_ignored_if_file_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NEGPY_PERF_RAW", str(tmp_path / "missing.raf"))
    monkeypatch.setattr(fixture, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(fixture, "_download", lambda url, dest: False)
    assert fixture.get_perf_raw_path() is None


def test_cached_file_returned_without_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("NEGPY_PERF_RAW", raising=False)
    monkeypatch.setattr(fixture, "_CACHE_DIR", tmp_path)
    cached = tmp_path / fixture._RAF_FILENAME
    cached.write_bytes(b"cached")
    downloaded = {"called": False}

    def fake_download(url: str, dest: Path) -> bool:
        downloaded["called"] = True
        return True

    monkeypatch.setattr(fixture, "_download", fake_download)
    result = fixture.get_perf_raw_path()
    assert result == str(cached)
    assert not downloaded["called"]


def test_downloads_on_cache_miss(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("NEGPY_PERF_RAW", raising=False)
    monkeypatch.setattr(fixture, "_CACHE_DIR", tmp_path)

    def fake_download(url: str, dest: Path) -> bool:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"downloaded")
        return True

    monkeypatch.setattr(fixture, "_download", fake_download)
    result = fixture.get_perf_raw_path()
    assert result == str(tmp_path / fixture._RAF_FILENAME)
    assert Path(result).read_bytes() == b"downloaded"


def test_returns_none_when_download_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("NEGPY_PERF_RAW", raising=False)
    monkeypatch.setattr(fixture, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(fixture, "_download", lambda url, dest: False)
    assert fixture.get_perf_raw_path() is None


def test_download_creates_parent_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """_download should create parent dirs so the file can be written."""
    import io
    import urllib.request

    dest = tmp_path / "subdir" / "file.raf"
    assert not dest.parent.exists()

    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=None: io.BytesIO(b"ok"))
    ok = fixture._download("https://example.com/fake.raf", dest)
    assert ok
    assert dest.read_bytes() == b"ok"


def test_download_returns_false_on_network_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import urllib.request

    def fail(*a: object, **kw: object) -> None:
        raise OSError("network down")

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    dest = tmp_path / "out.raf"
    assert not fixture._download("https://example.com/fake.raf", dest)
    assert not dest.exists()
