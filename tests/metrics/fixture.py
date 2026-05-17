"""
Resolves a real RAW file path for performance timing tests.

Priority:
  1. NEGPY_PERF_RAW env var (local dev or self-hosted runner)
  2. ~/.cache/negpy-metrics/DSCF1276.RAF (previously cached)
  3. Download from public URL and cache
  4. None — caller should pytest.skip()
"""

from __future__ import annotations

import os
from pathlib import Path

_RAF_URL = "https://kristoffertrolle.com/raw-samples/x70/DSCF1276.RAF"
_RAF_FILENAME = "DSCF1276.RAF"
_CACHE_DIR = Path.home() / ".cache" / "negpy-metrics"


def get_perf_raw_path() -> str | None:
    env = os.environ.get("NEGPY_PERF_RAW", "").strip()
    if env and os.path.isfile(env):
        return env

    cached = _CACHE_DIR / _RAF_FILENAME
    if cached.is_file():
        return str(cached)

    if _download(_RAF_URL, cached):
        return str(cached)

    return None


def _download(url: str, dest: Path) -> bool:
    """Download *url* to *dest*, creating parent dirs. Returns True on success."""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as f:
            f.write(resp.read())
        tmp.rename(dest)
        return True
    except Exception:
        tmp.unlink(missing_ok=True)
        return False
