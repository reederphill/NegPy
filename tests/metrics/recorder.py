"""
Collect named floating-point metrics during a pytest session (e.g. wall-clock seconds).
"""

from __future__ import annotations

_metrics: dict[str, float] = {}


def clear() -> None:
    _metrics.clear()


def record(name: str, value: float) -> None:
    _metrics[name] = value


def snapshot() -> dict[str, float]:
    return dict(_metrics)
