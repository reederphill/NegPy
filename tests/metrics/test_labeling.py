from __future__ import annotations


import pytest

from .labeling import metrics_machine_label, resolve_baseline_metrics


def test_resolve_per_machine_hits_current_label() -> None:
    base = {
        "per_machine": {
            "github-linux-x64": {"metrics": {"preview.load.cold_s": 2.0}},
        }
    }
    m, err = resolve_baseline_metrics(base, "github-linux-x64")
    assert err is None
    assert m == {"preview.load.cold_s": 2.0}


def test_resolve_per_machine_missing_key() -> None:
    m, err = resolve_baseline_metrics({"per_machine": {"other": {"metrics": {}}}}, "github-linux-x64")
    assert m is None
    assert err and "per_machine" in err


def test_resolve_single_file_matching_machine() -> None:
    m, err = resolve_baseline_metrics(
        {
            "machine_label": "dev-box",
            "metrics": {"preview.load.cold_s": 1.0},
        },
        "dev-box",
    )
    assert err is None
    assert m == {"preview.load.cold_s": 1.0}


def test_resolve_mismatch_skips() -> None:
    m, err = resolve_baseline_metrics(
        {
            "machine_label": "dev-box",
            "metrics": {"preview.load.cold_s": 1.0},
        },
        "other-host",
    )
    assert m is None
    assert err and "other-host" in err


def test_metrics_machine_label_respects_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEGPY_METRICS_MACHINE", " My MacBook ")
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    assert metrics_machine_label() == "my-macbook"


def test_metrics_machine_label_github_format(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEGPY_METRICS_MACHINE", raising=False)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("RUNNER_OS", "Linux")
    monkeypatch.setenv("RUNNER_ARCH", "X64")
    assert metrics_machine_label() == "github-linux-x64"
