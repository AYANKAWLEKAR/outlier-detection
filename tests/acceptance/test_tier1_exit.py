"""Tier-1 exit acceptance (Task 1.15, spec §9; rulings GATE3-GATE5 in decision 008).

Dataset SW1: ``GenerateConfig(seed=0, all 20 personas, days=90, all 8 families)``, generated into a
session tmp dir, or reused from ``$AFW_TIER1_TRACES`` when that directory's manifest has exactly
this config (it must come from the current generator). Protocol: the spec's allow-all replay
(§5.2) of the default engine, through ``agentfw.bench.run`` (scores + report + ablations).

* ``test_tier1_exit`` is the brief's gate verbatim. It is a strict xfail: the gate is not met
  under the spec protocol (decision 008). If it ever passes, the xfail must be removed.
* The other exit items are real assertions; ``test_tier1_no_regression`` pins today's per-family
  numbers with a 0.1 tolerance.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import pytest

from agentfw.bench.dataset import load_dir
from agentfw.bench.metrics import benign_fpr, recall_by_family, top_reason_share
from agentfw.bench.replay import ScoreRow, SplitSpec, read_score_table
from agentfw.bench.run import run
from agentfw.core.normalize import SessionTracker, normalize
from agentfw.synth.attacks import ATTACKS
from agentfw.synth.generate import GenerateConfig, Manifest, generate
from agentfw.synth.load import CATALOG_DIR, PERSONAS_DIR, TASKS_DIR
from tests.perf.test_latency import P99_BUDGET_US

pytestmark = pytest.mark.slow

TRACES_ENV = "AFW_TIER1_TRACES"
SW1 = GenerateConfig(
    seed=0,
    personas_dir=PERSONAS_DIR,
    tasks_dir=TASKS_DIR,
    catalog_dir=CATALOG_DIR,
    days=90,
    families=sorted(ATTACKS),
    attack_rate=0.015,
)
TARGETS = {
    "magnitude_spike": "magnitude",
    "prereq_skip": "sequence",
    "novel_target": "novelty",
    "scope_blowup": "magnitude",
    "velocity_burst": "velocity",
    "exfil_volume": "magnitude",
}
ABLATIONS = [
    "engine",
    "engine-no-magnitude",
    "engine-no-novelty",
    "engine-no-velocity",
    "engine-no-sequence",
]
# Default engine, spec protocol, SW1 (evidence/gate-seed0-seed1.md, "default" row, seed 0):
# attack-level recall on warm test rows at live thresholds, gate bucket.
MEASURED_RECALL = {
    "magnitude_spike": 0.538,
    "scope_blowup": 1.000,
    "exfil_volume": 0.579,
    "prereq_skip": 0.933,
    "novel_target": 0.388,
    "velocity_burst": 0.700,
}
TOLERANCE = 0.1


def _sw1_dir(tmp: pytest.TempPathFactory) -> Path:
    given = os.environ.get(TRACES_ENV)
    if given:
        d = Path(given)
        have = Manifest.model_validate_json((d / "manifest.json").read_bytes()).config
        if have != SW1:
            raise ValueError(f"${TRACES_ENV}={d} is not the SW1 dataset: {have!r}")
        return d
    d = tmp.mktemp("sw1")
    generate(SW1, d)
    return d


@pytest.fixture(scope="session")
def tier1_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """(dataset dir, run dir): one default-engine run with the ablation table, n_boot=0."""
    traces = _sw1_dir(tmp_path_factory)
    out = run(
        traces,
        "engine",
        SplitSpec(),
        tmp_path_factory.mktemp("tier1_run"),
        n_boot=0,
        ablations=ABLATIONS,
    )
    return traces, out


@pytest.fixture(scope="session")
def tier1_rows(tier1_run: tuple[Path, Path]) -> list[ScoreRow]:
    return read_score_table(tier1_run[1] / "scores.parquet")


@pytest.fixture(scope="session")
def tier1_report(tier1_run: tuple[Path, Path]) -> dict[str, Any]:
    rep = orjson.loads((tier1_run[1] / "report.json").read_bytes())
    assert isinstance(rep, dict)
    return rep


def _warm_test(rows: list[ScoreRow]) -> list[ScoreRow]:
    return [r for r in rows if r.split == "test" and r.is_warm]


def _family_recall(warm_test: list[ScoreRow], fam: str) -> float:
    rows = [r for r in warm_test if r.attack_family in (None, fam)]
    bucket = "10-100" if fam == "magnitude_spike" else "all"
    return recall_by_family(rows, "row", level="attack")[fam][bucket].recall


@pytest.mark.xfail(
    strict=True,
    reason="Tier-1 gate not met under spec protocol; see docs/decisions/008-tier1-gate.md",
)
def test_tier1_exit(tier1_rows: list[ScoreRow]) -> None:
    """The brief's per-family gate, verbatim (recall and top-reason share >= 0.90)."""
    warm_test = _warm_test(tier1_rows)
    for fam, det in TARGETS.items():
        rows = [r for r in warm_test if r.attack_family in (None, fam)]
        bucket = "10-100" if fam == "magnitude_spike" else "all"
        cell = recall_by_family(rows, "row", level="attack")[fam][bucket]
        assert cell.recall >= 0.90, (fam, cell)
        assert top_reason_share(rows, fam, det, "row") >= 0.90, fam


def test_tier1_benign_fpr_guard(tier1_rows: list[ScoreRow]) -> None:
    assert benign_fpr(_warm_test(tier1_rows), "row") <= 0.02


def test_tier1_ablation_table(tier1_report: dict[str, Any], tier1_run: tuple[Path, Path]) -> None:
    abl = tier1_report["ablations"]
    assert abl is not None and len(abl) >= 5
    assert [a["variant"] for a in abl] == ABLATIONS
    assert all(math.isfinite(a["auprc"]) for a in abl)
    assert "Ablations" in (tier1_run[1] / "report.md").read_text()


def test_tier1_cold_start_curve_and_drift(tier1_report: dict[str, Any]) -> None:
    # The brief reads these from report["extras"]; the report stores them at the top level.
    assert len(tier1_report["cold_start"]) == 4
    assert tier1_report["drift"]["n"] > 0  # slow_drift attacks were measured


def test_tier1_score_latency_p99(tier1_rows: list[ScoreRow], tier1_run: tuple[Path, Path]) -> None:
    """Spec §6.11 on real traces: p99 of score() on warm calls with <= 20 fields < 200 us."""
    raws, _, _ = load_dir(tier1_run[0])
    tracker = SessionTracker()
    small: set[str] = set()
    for raw in sorted(raws, key=lambda r: (r.ts_unix_ms, r.session_id, r.seq_no)):
        e = normalize(raw, tracker)
        tracker.observe(e)
        if len(e.flat_args) + len(e.derived) <= 20:
            small.add(e.envelope_id)
    lat = np.asarray(
        [
            r.latency_us
            for r in tier1_rows
            if r.is_warm and r.envelope_id in small and not math.isnan(r.fused_score)
        ]
    )
    assert lat.size > 100_000
    p99 = float(np.percentile(lat, 99))
    assert p99 < P99_BUDGET_US, f"p99={p99:.1f} us"


def test_tier1_no_regression(tier1_rows: list[ScoreRow]) -> None:
    """Today's default-protocol numbers, minus a 0.1 tolerance, so regressions are caught."""
    warm_test = _warm_test(tier1_rows)
    got = {fam: _family_recall(warm_test, fam) for fam in TARGETS}
    for fam, measured in MEASURED_RECALL.items():
        assert got[fam] >= measured - TOLERANCE, (fam, got[fam], measured)
    assert min(got.values()) >= min(MEASURED_RECALL.values()) - TOLERANCE
