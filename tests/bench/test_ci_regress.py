from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from agentfw.bench.ci_regress import (
    BASELINE_PATH,
    ci_dataset,
    compare,
    main,
)
from agentfw.bench.registry import make_detector
from agentfw.bench.replay import SplitSpec, canonical_hash, replay

NAN = math.nan
_ROOT = Path(__file__).resolve().parents[2]


def test_compare_boundaries() -> None:
    base = {"auprc": 0.80, "fpr95": 0.010}
    assert compare({"auprc": 0.790, "fpr95": 0.012}, base) == []
    assert len(compare({"auprc": 0.789, "fpr95": 0.0121}, base)) == 2


def test_compare_improvement_is_ok() -> None:
    assert compare({"auprc": 0.9, "fpr95": 0.0}, {"auprc": 0.8, "fpr95": 0.01}) == []


def test_compare_nan_rules() -> None:
    both = {"auprc": NAN, "fpr95": NAN}
    fine = {"auprc": 0.5, "fpr95": 0.1}
    assert compare(both, both) == []  # NaN -> NaN
    assert compare(fine, both) == []  # NaN -> finite is an improvement
    assert len(compare(both, fine)) == 2  # finite -> NaN fails each metric
    assert len(compare({"auprc": NAN, "fpr95": 0.1}, fine)) == 1


def test_compare_accepts_none_as_nan() -> None:
    assert compare({"auprc": 0.5, "fpr95": 0.1}, {"auprc": None, "fpr95": None}) == []  # type: ignore[dict-item]


def test_golden_static_threshold() -> None:
    rows = replay(*ci_dataset(), make_detector("static_threshold"), SplitSpec())
    golden = (_ROOT / "tests/golden/static_threshold.sha256").read_text().strip()
    assert canonical_hash(rows) == golden


def test_baseline_file_shape() -> None:
    data = json.loads(BASELINE_PATH.read_text())
    assert {"static_threshold", "engine"} <= set(data)
    for v in data.values():
        assert set(v) == {"auprc", "fpr95"}


def test_main_passes_on_committed_baseline(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "static_threshold" in out and "engine" in out


def test_main_fails_on_regression(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "baseline.json"
    bad.write_text(json.dumps({"static_threshold": {"auprc": 1.0, "fpr95": 0.0}}))
    assert main(["--baseline", str(bad)]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_write_baseline(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"static_threshold": {"auprc": 0.0, "fpr95": 0.0}}))
    assert main(["--baseline", str(path), "--write-baseline"]) == 0
    assert main(["--baseline", str(path)]) == 0
    # Only detectors already in the file are rewritten (here: static_threshold).
    written = json.loads(path.read_text())
    assert set(written) == {"static_threshold"}
    assert written["static_threshold"] == json.loads(BASELINE_PATH.read_text())["static_threshold"]
