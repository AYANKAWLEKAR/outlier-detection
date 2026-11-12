"""Tier-1 exit acceptance under both replay protocols (spec A1.5, A1.7; D-027, D-028).

Dataset SW1: :func:`agentfw.synth.generate.sw1_config` (seed 0, all 20 personas, 90 days, all 8
families, far-mode magnitude spikes and the magnitude top-up, spec A1.4), generated into a session
tmp dir, or reused from ``$AFW_TIER1_TRACES`` when that directory's manifest has exactly this
config. The default engine is replayed on that one dataset through ``agentfw.bench.run`` under
each of the two protocols, whose labels are fixed text (spec A1.5):

* ``"review"``: oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic).
  This run also carries the ablation table.
* ``"allow"``: spec §5.2 allow-all.

``test_tier1_exit`` is the gate: the amended criterion of spec A1.7 (per gated family attack-level
recall >= 0.90 and among-the-reasons share >= 0.90; benign FPR <= 2 %), once per protocol. Its two
results are always read together; a pass under the oracle-reviewer approval loop (labels act as a
perfect human reviewer → optimistic) alone is not the Tier-1 gate being met (D-028). The
``allow`` parameter is a strict xfail (docs/decisions/008-tier1-gate.md); the ``review`` parameter
carries no mark. The top-1 share (original spec §9 criterion) is still computed and reported
beside the among-the-reasons share; it is no longer gated.

The other tests are dataset validity (``magnitude_spike`` gate cell >= 50 attacks), the benign FPR
guard, the report fields, the test's cells equal to report.json's gate cells, the failure
message's format, the regression pins against the measured seed-0 rows of
``docs/decisions/evidence/a1-gate/matrix.json`` (0.1 tolerance) and, on the review run, the
ablation table, the cold-start curve and drift, and score latency. A failure message shows both
protocols together and prints the ``magnitude_spike`` cell only next to its two companion rows
from ``matrix.json`` (spec A1.5 rules 1 and 4).
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
from agentfw.bench.gate import ORIGINAL_CRITERION_LABEL
from agentfw.bench.metrics import (
    RecallCell,
    among_reasons_share,
    benign_fpr,
    control_fpr,
    recall_by_family,
    top_reason_share,
)
from agentfw.bench.replay import (
    PROTOCOL_LABELS,
    OutcomeMode,
    ScoreRow,
    SplitSpec,
    read_score_table,
)
from agentfw.bench.run import run
from agentfw.core.normalize import SessionTracker, normalize
from agentfw.synth.generate import Manifest, generate, sw1_config
from tests.perf.test_latency import P99_BUDGET_US

pytestmark = pytest.mark.slow

TRACES_ENV = "AFW_TIER1_TRACES"
SW1 = sw1_config()
PROTOCOLS: tuple[OutcomeMode, ...] = ("review", "allow")
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
# Per family (attack-level recall, among-the-reasons share) on warm test rows at live thresholds,
# gate bucket: the default ("a1") engine on SW1 seed 0 (traces a455b10c705d), copied from the
# seed-0 rows of docs/decisions/evidence/a1-gate/matrix.json (Task 1.23; score hashes
# 1fcae9ee6bd1 review, ea8fd4d647f9 allow), recall to 3 and among to 2 decimals.
MEASURED: dict[OutcomeMode, dict[str, tuple[float, float]]] = {
    "review": {
        "magnitude_spike": (1.000, 1.00),
        "scope_blowup": (1.000, 1.00),
        "exfil_volume": (1.000, 1.00),
        "prereq_skip": (1.000, 1.00),
        "novel_target": (1.000, 1.00),
        "velocity_burst": (1.000, 1.00),
    },
    "allow": {
        "magnitude_spike": (0.938, 1.00),
        "scope_blowup": (1.000, 1.00),
        "exfil_volume": (0.579, 1.00),
        "prereq_skip": (0.533, 1.00),
        "novel_target": (0.163, 1.00),
        "velocity_burst": (0.800, 1.00),
    },
}
TOLERANCE = 0.1
MIN_MAGNITUDE_N = 50  # dataset validity (spec A1.4, A1.7; D-026)
MATRIX = Path(__file__).resolve().parents[2] / "docs/decisions/evidence/a1-gate/matrix.json"
# (dataset, engine) of the two magnitude_spike companion rows in matrix.json (spec A1.5 rule 4).
COMPANIONS = (("sw1", "pre-a1"), ("sens", "a1"))


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
def tier1_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[OutcomeMode, tuple[Path, Path]]:
    """Per protocol, (dataset dir, run dir): the default engine on one SW1 dataset, n_boot=0.
    The oracle-reviewer run carries the ablation table; the allow-all run does not."""
    traces = _sw1_dir(tmp_path_factory)
    out: dict[OutcomeMode, tuple[Path, Path]] = {}
    for mode in PROTOCOLS:
        out[mode] = (
            traces,
            run(
                traces,
                "engine",
                SplitSpec(),
                tmp_path_factory.mktemp(f"tier1_{mode}"),
                n_boot=0,
                ablations=ABLATIONS if mode == "review" else None,
                outcomes=mode,
            ),
        )
    return out


@pytest.fixture(scope="session")
def tier1_rows(
    tier1_runs: dict[OutcomeMode, tuple[Path, Path]],
) -> dict[OutcomeMode, list[ScoreRow]]:
    return {mode: read_score_table(d / "scores.parquet") for mode, (_, d) in tier1_runs.items()}


@pytest.fixture(scope="session")
def tier1_reports(
    tier1_runs: dict[OutcomeMode, tuple[Path, Path]],
) -> dict[OutcomeMode, dict[str, Any]]:
    out: dict[OutcomeMode, dict[str, Any]] = {}
    for mode, (_, d) in tier1_runs.items():
        rep = orjson.loads((d / "report.json").read_bytes())
        assert isinstance(rep, dict)
        out[mode] = rep
    return out


def _warm_test(rows: list[ScoreRow]) -> list[ScoreRow]:
    return [r for r in rows if r.split == "test" and r.is_warm]


def _bucket(fam: str) -> str:
    return "10-100" if fam == "magnitude_spike" else "all"


def _cell(rows: list[ScoreRow], fam: str) -> RecallCell:
    """Attack-level recall of ``fam`` on warm test rows at live thresholds, gate bucket."""
    fam_rows = [r for r in _warm_test(rows) if r.attack_family in (None, fam)]
    return recall_by_family(fam_rows, "row", level="attack")[fam][_bucket(fam)]


def _family_recall(warm_test: list[ScoreRow], fam: str) -> float:
    return _cell(warm_test, fam).recall  # _warm_test is idempotent


def _among(rows: list[ScoreRow], fam: str) -> float:
    """Among-the-reasons share of ``fam``'s targeted detector (spec A1.6), gate bucket."""
    return among_reasons_share(_warm_test(rows), fam, TARGETS[fam], "row", bucket=_bucket(fam))


def _companions() -> list[str]:
    """The two companion rows of the ``magnitude_spike`` cell (spec A1.5 rule 4) at SW1's seed,
    from ``matrix.json``: the pre-A1 engine on SW1 and the amended engine on the sensitivity
    dataset."""
    by_key = {
        (r["dataset"], r["engine"], r["protocol"], r["seed"]): r
        for r in orjson.loads(MATRIX.read_bytes())["rows"]
    }
    names = {
        ("sw1", "pre-a1"): "pre-A1 engine, SW1",
        ("sens", "a1"): "amended engine, sensitivity dataset (magnitude_far_mult = 1.0)",
    }
    out: list[str] = []
    for dataset, engine in COMPANIONS:
        parts = []
        for mode in PROTOCOLS:
            c = by_key[dataset, engine, mode, SW1.seed]["families"]["magnitude_spike"]
            parts.append(
                f"{PROTOCOL_LABELS[mode]}: {c['recall']:.3f} ({c['n_detected']}/{c['n_total']})"
            )
        name = f"{names[dataset, engine]} seed {SW1.seed}"
        out.append(f"  companion ({name}, matrix.json): " + "; ".join(parts))
    return out


def _summary(tier1_rows: dict[OutcomeMode, list[ScoreRow]], failed: str) -> str:
    """Failure message: the amended criterion's numbers under both protocols, each under its label
    verbatim, then benign and control FPR (spec A1.5 rules 1 and 3); the magnitude cell only with
    its companions (rule 4)."""
    lines = [f"failed: {failed}"]
    for mode in PROTOCOLS:
        rows = tier1_rows[mode]
        warm = _warm_test(rows)
        lines.append(f"{PROTOCOL_LABELS[mode]}:")
        for fam, det in TARGETS.items():
            c = _cell(rows, fam)
            top1 = top_reason_share(warm, fam, det, "row")
            lines.append(
                f"  {fam} {_bucket(fam)}: recall {c.recall:.3f} ({c.n_detected}/{c.n_total}); "
                f"among {_among(rows, fam):.2f}; top-1 {top1:.2f} ({ORIGINAL_CRITERION_LABEL})"
            )
        b, ctl = benign_fpr(warm, "row"), control_fpr(warm, "row")
        lines.append(f"  benign FPR {b:.4f}; control FPR {ctl:.4f}")
    return "\n".join([*lines, *_companions()])


_ALLOW_XFAIL = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Tier-1 gate (Amendment A1) not met under spec §5.2 allow-all: the replay learns the "
    "attack rows; see docs/decisions/008-tier1-gate.md; the oracle-reviewer approval loop (labels "
    "act as a perfect human reviewer → optimistic) result is test_tier1_exit[review]",
)


@pytest.mark.parametrize("mode", ["review", pytest.param("allow", marks=_ALLOW_XFAIL)])
def test_tier1_exit(tier1_rows: dict[OutcomeMode, list[ScoreRow]], mode: OutcomeMode) -> None:
    """Amended gate (spec A1.7), per protocol.
    "review" = oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic);
    "allow" = spec §5.2 allow-all."""
    warm_test = _warm_test(tier1_rows[mode])
    for fam, det in TARGETS.items():
        rows = [r for r in warm_test if r.attack_family in (None, fam)]
        bucket = _bucket(fam)
        recall = recall_by_family(rows, "row", level="attack")[fam][bucket].recall
        assert recall >= 0.90, _summary(tier1_rows, f"{mode} {fam} recall")
        among = among_reasons_share(rows, fam, det, "row", bucket=bucket)
        assert among >= 0.90, _summary(tier1_rows, f"{mode} {fam} among")
    fpr = benign_fpr(warm_test, "row")
    assert fpr <= 0.02, _summary(tier1_rows, f"{mode} benign FPR")


@pytest.mark.parametrize("mode", PROTOCOLS)
def test_tier1_magnitude_cell_size(
    tier1_rows: dict[OutcomeMode, list[ScoreRow]], mode: OutcomeMode
) -> None:
    """Dataset validity (spec A1.4, A1.7): the magnitude_spike gate cell holds >= 50 attacks."""
    n_total = _cell(tier1_rows[mode], "magnitude_spike").n_total
    assert n_total >= MIN_MAGNITUDE_N, mode


@pytest.mark.parametrize("mode", PROTOCOLS)
def test_tier1_benign_fpr_guard(
    tier1_rows: dict[OutcomeMode, list[ScoreRow]], mode: OutcomeMode
) -> None:
    fpr = benign_fpr(_warm_test(tier1_rows[mode]), "row")
    assert fpr <= 0.02, _summary(tier1_rows, f"{mode} benign FPR")


@pytest.mark.parametrize("mode", PROTOCOLS)
def test_tier1_report_shows_both_shares(
    tier1_reports: dict[OutcomeMode, dict[str, Any]], mode: OutcomeMode
) -> None:
    gate = tier1_reports[mode]["gate"]
    assert tier1_reports[mode]["extras"]["outcomes"] == mode
    for fam in TARGETS:
        assert {"among_reasons_share", "top_reason_share"} <= gate["cells"][fam].keys()
    # The report.json field; not a rendered claim (spec A1.5).
    assert gate["passed"] is (mode == "review")


@pytest.mark.parametrize("mode", PROTOCOLS)
def test_tier1_cells_match_report(
    tier1_rows: dict[OutcomeMode, list[ScoreRow]],
    tier1_reports: dict[OutcomeMode, dict[str, Any]],
    mode: OutcomeMode,
) -> None:
    """The test's own population (warm test rows, gate bucket) is the one report.json's gate cells
    (``gate_preview``) were computed on, so the test's numbers and the report's PASS/FAIL cannot
    drift apart (spec A1.7)."""
    rows = tier1_rows[mode]
    gate = tier1_reports[mode]["gate"]
    for fam in TARGETS:
        c, want = _cell(rows, fam), gate["cells"][fam]
        assert (c.n_detected, c.n_total) == (want["n_detected"], want["n_total"]), (mode, fam)
        assert c.recall == want["recall"], (mode, fam)
        assert _among(rows, fam) == want["among_reasons_share"], (mode, fam)
    warm = _warm_test(rows)
    assert benign_fpr(warm, "row") == gate["benign_fpr"], mode
    assert control_fpr(warm, "row") == gate["control_fpr"], mode


def test_tier1_failure_summary_follows_a1_5(
    tier1_rows: dict[OutcomeMode, list[ScoreRow]],
) -> None:
    """Every failure message (``_summary``) meets spec A1.5 rules 1, 3 and 4: both protocols in
    order, each under its label verbatim; per family recall (n/N), among share and top-1 share
    labelled as the original criterion; then benign FPR and control FPR; then the two
    ``magnitude_spike`` companion rows of SW1's seed from matrix.json."""
    matrix = {
        (r["dataset"], r["engine"], r["protocol"], r["seed"]): r
        for r in orjson.loads(MATRIX.read_bytes())["rows"]
    }
    for key in [(d, e, m, SW1.seed) for d, e in COMPANIONS for m in PROTOCOLS]:
        assert key in matrix, f"matrix.json has no companion row {key}"
    lines = _summary(tier1_rows, "probe").splitlines()
    assert lines[0] == "failed: probe"
    block = 1 + len(TARGETS) + 1
    for i, mode in enumerate(PROTOCOLS):
        head, *fams, fpr = lines[1 + i * block : 1 + (i + 1) * block]
        assert head == f"{PROTOCOL_LABELS[mode]}:"
        for fam, line in zip(TARGETS, fams, strict=True):
            c = _cell(tier1_rows[mode], fam)
            assert line.startswith(f"  {fam} {_bucket(fam)}: recall {c.recall:.3f} "), line
            assert f"({c.n_detected}/{c.n_total}); among " in line, line
            assert line.endswith(f" ({ORIGINAL_CRITERION_LABEL})") and "top-1 " in line, line
        warm = _warm_test(tier1_rows[mode])
        b, ctl = benign_fpr(warm, "row"), control_fpr(warm, "row")
        assert fpr == f"  benign FPR {b:.4f}; control FPR {ctl:.4f}", fpr
    companions = lines[1 + len(PROTOCOLS) * block :]
    assert len(companions) == len(COMPANIONS)
    for line, (dataset, engine) in zip(companions, COMPANIONS, strict=True):
        assert f"seed {SW1.seed}, matrix.json): " in line, line
        for mode in PROTOCOLS:
            c = matrix[dataset, engine, mode, SW1.seed]["families"]["magnitude_spike"]
            want = f"{PROTOCOL_LABELS[mode]}: {c['recall']:.3f} ({c['n_detected']}/{c['n_total']})"
            assert want in line, line


@pytest.mark.parametrize("mode", PROTOCOLS)
def test_tier1_no_regression(
    tier1_rows: dict[OutcomeMode, list[ScoreRow]], mode: OutcomeMode
) -> None:
    """The measured seed-0 numbers per protocol, minus a 0.1 tolerance, catch regressions."""
    warm_test = _warm_test(tier1_rows[mode])
    for fam, (recall, among) in MEASURED[mode].items():
        got_recall = _family_recall(warm_test, fam)
        assert got_recall >= recall - TOLERANCE, _summary(tier1_rows, f"{mode} {fam} recall pin")
        got_among = _among(tier1_rows[mode], fam)
        assert got_among >= among - TOLERANCE, _summary(tier1_rows, f"{mode} {fam} among pin")


def test_tier1_ablation_table(
    tier1_reports: dict[OutcomeMode, dict[str, Any]],
    tier1_runs: dict[OutcomeMode, tuple[Path, Path]],
) -> None:
    abl = tier1_reports["review"]["ablations"]
    assert abl is not None and len(abl) >= 5
    assert [a["variant"] for a in abl] == ABLATIONS
    assert all(math.isfinite(a["auprc"]) for a in abl)
    assert "Ablations" in (tier1_runs["review"][1] / "report.md").read_text()


def test_tier1_cold_start_curve_and_drift(tier1_reports: dict[OutcomeMode, dict[str, Any]]) -> None:
    # The brief reads these from report["extras"]; the report stores them at the top level.
    rep = tier1_reports["review"]
    assert len(rep["cold_start"]) == 4
    assert rep["drift"]["n"] > 0  # slow_drift attacks were measured


def test_tier1_score_latency_p99(
    tier1_rows: dict[OutcomeMode, list[ScoreRow]],
    tier1_runs: dict[OutcomeMode, tuple[Path, Path]],
) -> None:
    """Spec §6.11 on real traces: p99 of score() on warm calls with <= 20 fields < 200 us."""
    raws, _, _ = load_dir(tier1_runs["review"][0])
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
            for r in tier1_rows["review"]
            if r.is_warm and r.envelope_id in small and not math.isnan(r.fused_score)
        ]
    )
    assert lat.size > 100_000
    p99 = float(np.percentile(lat, 99))
    assert p99 < P99_BUDGET_US, f"p99={p99:.1f} us"
