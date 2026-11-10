"""Tier-2 exit acceptance (Task 2.11, spec §9): the brief's assertions, verbatim.

Needs the live sandbox dataset ``dataset/v1`` (built by hand; see the README's "How to build
dataset/v1") and its runs under ``results/v1/<name>/report.json``. Neither exists in a clean
checkout, and building them calls a live LLM, so this test skips when ``dataset/v1`` is absent.
The synthetic SW1 leaderboard (docs/leaderboard.md) is not a substitute for this exit.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentfw.sandbox.package import validate_dataset

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset" / "v1"
RESULTS = ROOT / "results" / "v1"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not DATASET.is_dir(), reason="dataset/v1 not built (live sandbox run; see README)"
    ),
]


def test_tier2_exit() -> None:
    assert validate_dataset(DATASET).ok
    reports = {p.parent.name: json.loads(p.read_text()) for p in RESULTS.glob("*/report.json")}
    assert {
        "engine",
        "random",
        "static_threshold",
        "isolation_forest",
        "llm_judge-sample",
        "engine-sample",
    } <= reports.keys()
    assert len({r["meta"]["dataset_sha"] for r in reports.values()}) == 1
    for name in ("engine", "engine-sample"):
        h = reports[name]["headline"]
        assert h["auprc_ci"][0] <= h["auprc"] <= h["auprc_ci"][1]
    assert "<!-- report:start -->" in (ROOT / "README.md").read_text()
