"""Trajectory exit acceptance (Task 4.1): the feature-spec acceptance bullets that no unit test
pins, on the full bench.

Bench: the generator's bench config (``sw1_config``) with ``families`` extended by ``loop``,
``spin``, ``stall`` and ``splice`` (``trajectory.bench_config``), seeds 0 and 1, the time split
with a 15% calibration partition (``trajectory.SPLIT``). Datasets are generated into a session
tmp dir, or reused from ``$AFW_TRAJ_TRACES/seed-{0,1}`` when their manifests hold exactly this
config (``python -m agentfw.bench.trajectory --traces DIR`` writes the same layout). Replays run
in ``$AFW_TRAJ_WORKERS`` processes (default 4); every engine gets the dataset's schemas as hints.

Each criterion is checked under both protocols, on both seeds:

* ``"allow"``: spec §5.2 allow-all.
* ``"review"``: oracle-reviewer approval loop (labels act as a perfect human reviewer →
  optimistic).

A criterion that is not met is a strict xfail whose reason carries the measured numbers (the
Tier-1 pattern); no bar, threshold or default is changed to make one pass. The criteria:

* F2: benign vital false-flag rate <= 1% of sessions (``vital`` row: v1 + vital).
* F5: escalation (``escalate`` row) lowers the decision control FPR versus ``engine``, and does
  not reduce any family's attack-level decision recall by more than 1 point (two tests, so a
  failing clause never masks a holding one).
* F6: decision control FPR on ``benign_unusual`` <= 3% with ``provisional``, and below
  ``engine``'s (two tests).
* F7: ``engine-traj``'s halt-sensitivity curve (false halts over θ = 0.2 ... 0.8) is
  non-increasing with no step above 10 points between neighbouring θ (no cliff; measured on the
  shadow replay ``engine-traj-shadow``, whose offline halts are exact at every θ, T-43), and the
  first-flag baseline false-halts more often than ``engine-traj`` at θ_halt = 0.6.
* F10: attribution exactness: removing a group changes exactly the attacks it carried (none
  lost that it did not carry, none gained). The T-20 totals are not asserted: ``tabulate``
  makes them hold by construction (T-52).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from agentfw.bench.replay import PROTOCOL_LABELS, OutcomeMode
from agentfw.bench.trajectory import (
    TRAJ_GROUPS,
    Job,
    bench_config,
    build_report,
    ensure_dataset,
    run_jobs,
)
from agentfw.synth.generate import Manifest

pytestmark = pytest.mark.slow

TRACES_ENV = "AFW_TRAJ_TRACES"
WORKERS_ENV = "AFW_TRAJ_WORKERS"
SEEDS = (0, 1)
ROWS = ("engine", "vital", "escalate", "provisional", "engine-traj", "engine-traj-shadow")
PROTOCOLS: tuple[OutcomeMode, ...] = ("allow", "review")
MAX_STEP = 0.10
ONE_POINT = 0.01


def _root(tmp: pytest.TempPathFactory) -> Path:
    given = os.environ.get(TRACES_ENV)
    root = Path(given) if given else tmp.mktemp("traj_bench")
    for seed in SEEDS:
        d = ensure_dataset(root, seed)
        have = Manifest.model_validate_json((d / "manifest.json").read_bytes()).config
        if have != bench_config(seed):
            raise ValueError(f"{d} is not the trajectory bench dataset for seed {seed}")
    return root


@pytest.fixture(scope="session")
def report(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = _root(tmp_path_factory)
    datasets = {s: root / f"seed-{s}" for s in SEEDS}
    jobs: list[Job] = []
    for seed, d in datasets.items():
        for mode in PROTOCOLS:
            jobs += [Job("row", str(d), seed, mode, name) for name in ROWS]
            jobs += [Job("group", str(d), seed, mode, g.name) for g in TRAJ_GROUPS]
    results = run_jobs(jobs, int(os.environ.get(WORKERS_ENV, "4")))
    return build_report(results, datasets, n_boot_delta=200, latency=None, wall_s={})


def _seeds(report: dict[str, Any], mode: OutcomeMode) -> dict[str, dict[str, Any]]:
    seeds: dict[str, dict[str, Any]] = report["protocols"][mode]["seeds"]
    assert set(seeds) == {str(s) for s in SEEDS}
    return seeds


def _rows(block: dict[str, Any]) -> dict[str, Any]:
    rows: dict[str, Any] = block["rows"]
    return rows


def _xfail(reason: str) -> pytest.MarkDecorator:
    return pytest.mark.xfail(strict=True, raises=AssertionError, reason=reason)


def _modes(**xfails: str) -> list[Any]:
    """The protocol parameters, each with its strict xfail when one is given."""
    return [pytest.param(m, marks=_xfail(xfails[m])) if m in xfails else m for m in PROTOCOLS]


_F2 = (
    "F2 bar unmet (T-30, pending user decision): benign vital false-flag rate 3.85% (seed 0) / "
    "3.63% (seed 1), bar 1%; task repeat iterations re-issue the same call inside the loop window"
)


@pytest.mark.parametrize(
    ("mode", "seed"),
    [pytest.param(m, s, marks=_xfail(_F2)) for m in PROTOCOLS for s in SEEDS],
)
def test_f2_benign_vital_false_flag_rate_at_most_1pct(
    report: dict[str, Any], mode: OutcomeMode, seed: int
) -> None:
    """One case per protocol and seed, so a seed that recovers is reported on its own."""
    got = _seeds(report, mode)[str(seed)]["rows"]["vital"]["sessions"]["vital_false_flag_rate"]
    msg = f"{PROTOCOL_LABELS[mode]} seed {seed}: benign vital false-flag rate {got} (bar 0.01)"
    assert got <= 0.01, msg


_F5 = (
    "F5 unmet: escalate leaves the decision control FPR unchanged ({}); without a judge the "
    "Python re-score is the fast score (T-12, T-42; reused since T-60), so escalation changes no "
    "decision"
)


@pytest.mark.parametrize(
    "mode",
    _modes(
        allow=_F5.format("0.0265 -> 0.0265 seed 0, 0.0163 -> 0.0163 seed 1"),
        review=_F5.format("0.0737 -> 0.0737 seed 0, 0.0343 -> 0.0343 seed 1"),
    ),
)
def test_f5_escalation_lowers_control_fpr(report: dict[str, Any], mode: OutcomeMode) -> None:
    for seed, b in _seeds(report, mode).items():
        eng, esc = _rows(b)["engine"]["decisions"], _rows(b)["escalate"]["decisions"]
        msg = (
            f"{PROTOCOL_LABELS[mode]} seed {seed}: control FPR engine {eng['control_fpr']} -> "
            f"escalate {esc['control_fpr']}"
        )
        assert esc["control_fpr"] < eng["control_fpr"], msg


@pytest.mark.parametrize("mode", _modes())
def test_f5_escalation_keeps_recall_within_1_point(
    report: dict[str, Any], mode: OutcomeMode
) -> None:
    for seed, b in _seeds(report, mode).items():
        eng, esc = _rows(b)["engine"]["decisions"], _rows(b)["escalate"]["decisions"]
        assert eng["recall_by_family"], (mode, seed)
        for fam, r in eng["recall_by_family"].items():
            got = esc["recall_by_family"][fam]
            msg = f"{PROTOCOL_LABELS[mode]} seed {seed} {fam}: recall engine {r} -> escalate {got}"
            assert got >= r - ONE_POINT, msg


@pytest.mark.parametrize("mode", _modes())
def test_f6_provisional_control_fpr_at_most_3pct(report: dict[str, Any], mode: OutcomeMode) -> None:
    for seed, b in _seeds(report, mode).items():
        prov = _rows(b)["provisional"]["decisions"]["control_fpr"]
        assert prov <= 0.03, f"{PROTOCOL_LABELS[mode]} seed {seed}: provisional control FPR {prov}"


@pytest.mark.parametrize(
    "mode",
    _modes(
        allow="F6 unmet under spec §5.2 allow-all: provisional leaves the decision control FPR at "
        "engine's (0.0265 seed 0, 0.0163 seed 1); it lowers the benign FPR (-0.0011, -0.0020) but "
        "changes the decision of no flagged benign_unusual control row (only reversible band "
        "blocks become provisional; approvals never do)",
    ),
)
def test_f6_provisional_control_fpr_below_engine(report: dict[str, Any], mode: OutcomeMode) -> None:
    for seed, b in _seeds(report, mode).items():
        eng = _rows(b)["engine"]["decisions"]["control_fpr"]
        prov = _rows(b)["provisional"]["decisions"]["control_fpr"]
        msg = f"{PROTOCOL_LABELS[mode]} seed {seed}: control FPR engine {eng}, provisional {prov}"
        assert prov < eng, msg


@pytest.mark.parametrize("mode", _modes())
def test_f7_halt_sensitivity_has_no_cliff(report: dict[str, Any], mode: OutcomeMode) -> None:
    for seed, b in _seeds(report, mode).items():
        halt = _rows(b)["engine-traj-shadow"]["sessions"]["halt"]
        assert halt["exact_at_every_theta"] is True
        fhr = [x["false_halt_rate"] for x in halt["sensitivity"]]
        msg = f"{PROTOCOL_LABELS[mode]} seed {seed}: false halts over θ {fhr}"
        assert all(a >= b for a, b in zip(fhr, fhr[1:], strict=False)), msg
        assert max(a - b for a, b in zip(fhr, fhr[1:], strict=False)) <= MAX_STEP, msg


@pytest.mark.parametrize("mode", _modes())
def test_f7_first_flag_baseline_false_halts_more(report: dict[str, Any], mode: OutcomeMode) -> None:
    for seed, b in _seeds(report, mode).items():
        halt = _rows(b)["engine-traj"]["sessions"]["halt"]
        density, first = halt["density_policy"], halt["first_flag"]
        msg = (
            f"{PROTOCOL_LABELS[mode]} seed {seed}: false halts first-flag "
            f"{first['false_halt_rate']}, engine-traj {density['false_halt_rate']}"
        )
        assert first["false_halt_rate"] > density["false_halt_rate"], msg


_F10 = (
    "F10 literal exactness unmet: removing a group also gains attacks ({}); provisional suppresses "
    "blocks of reversible calls, and a removed detector moves the calibrator and later decisions "
    "(T-52)"
)


@pytest.mark.parametrize(
    "mode",
    _modes(
        allow=_F10.format(
            "seed 0: provisional 24, magnitude 2, velocity 1, sequence 1; "
            "seed 1: provisional 29, velocity 5, magnitude 4"
        ),
        review=_F10.format(
            "seed 0: provisional 24, magnitude 2, vital 2; "
            "seed 1: provisional 30, magnitude 1, velocity 1 (re-measured on c7c123f, T-64)"
        ),
    ),
)
def test_f10_attribution_is_exact(report: dict[str, Any], mode: OutcomeMode) -> None:
    for seed, b in _seeds(report, mode).items():
        att = b["attribution"]
        gained = {g: v["gained"] for g, v in att["groups"].items()}
        msg = (
            f"{PROTOCOL_LABELS[mode]} seed {seed}: attacks gained when a group is removed "
            f"{gained}; detected with every group {att['n_detected_full']}"
        )
        assert att["literal_exact"], msg
