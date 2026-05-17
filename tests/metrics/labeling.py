"""
Stable machine labels for metrics so Mac vs CI (and other hosts) are tracked separately.
"""

from __future__ import annotations

import os
import platform
import re
import socket
from typing import Any


def _slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "unknown"


def metrics_machine_label() -> str:
    """
    Human-readable, filesystem-safe id for the current host / runner.

    Override: ``NEGPY_METRICS_MACHINE=dev-mac`` (stored as slug: ``dev-mac``).
    GitHub Actions: ``github-<RUNNER_OS>-<RUNNER_ARCH>`` (e.g. ``github-linux-x64``).
    Local: ``local-<system>-<machine>-<short-hostname>``.
    """
    override = os.environ.get("NEGPY_METRICS_MACHINE", "").strip()
    if override:
        return _slug(override)
    if os.environ.get("GITHUB_ACTIONS", "").lower() == "true":
        runner_os = os.environ.get("RUNNER_OS", "unknown")
        runner_arch = os.environ.get("RUNNER_ARCH", "unknown")
        return f"github-{_slug(runner_os)}-{_slug(runner_arch)}"
    hn = socket.gethostname().split(".")[0]
    return _slug(f"local-{platform.system()}-{platform.machine()}-{hn}")


def _as_float_metrics(d: Any) -> dict[str, float]:
    if not isinstance(d, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in d.items():
        if isinstance(v, (int, float)):
            out[str(k)] = float(v)
    return out


def resolve_baseline_metrics(
    baseline: dict[str, Any],
    machine_label: str,
) -> tuple[dict[str, float] | None, str | None]:
    """
    Returns (metrics dict, None) or (None, skip reason) for pytest.

    **1) Multi-host (recommended)** — one file, trends per machine::

        {"per_machine": {"github-linux-x64": {"metrics": {"preview.load.cold_s": 2.0}}}}

    **2) Single past export** — same shape as a metrics run; used only if
    ``machine_label`` matches the current host.

    **3) Legacy** — top-level ``metrics`` and no ``machine_label``: only if
    ``NEGPY_METRICS_BASELINE_UNLABELED=1``.
    """
    per = baseline.get("per_machine")
    if isinstance(per, dict) and per:
        ent = per.get(machine_label)
        if ent is None:
            known = ", ".join(sorted(per.keys())[:16])
            more = "…" if len(per) > 16 else ""
            return (
                None,
                f"no per_machine[{machine_label!r}] in baseline (known: {known}{more}).",
            )
        if isinstance(ent, dict) and "metrics" in ent:
            m = _as_float_metrics(ent.get("metrics"))
            if m:
                return m, None
        if isinstance(ent, dict):
            m = _as_float_metrics(ent)
            if m:
                return m, None
        return None, f"per_machine[{machine_label!r}] has no numeric metrics"

    m = baseline.get("metrics")
    m = _as_float_metrics(m) if m else {}
    if not m:
        return None, "baseline has no metrics"

    bl_label = baseline.get("machine_label")
    if bl_label is not None and str(bl_label) != "" and str(bl_label) != machine_label:
        return (
            None,
            f"baseline is for machine {bl_label!r}, this run is {machine_label!r}. "
            f"Use a per_machine file or copy this host's numbers into per_machine['{machine_label}'].",
        )

    if (bl_label is None or str(bl_label) == "") and os.environ.get("NEGPY_METRICS_BASELINE_UNLABELED", "").strip() not in (
        "1",
        "true",
        "yes",
    ):
        return (
            None,
            "legacy unlabeled baseline: set NEGPY_METRICS_BASELINE_UNLABELED=1 to compare, or switch to per_machine[...] in the JSON.",
        )

    return m, None
