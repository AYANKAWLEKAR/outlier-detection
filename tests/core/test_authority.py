"""Verdict authority scaled by evidence maturity (F8, Task 3.4; T-14, T-44 - T-50)."""

from __future__ import annotations

import dataclasses
import math
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path

import orjson
import pytest
from hypothesis import given
from hypothesis import strategies as st

from agentfw.bench.ci_regress import CI_DATASET
from agentfw.bench.dataset import load_dir, load_schema_hints
from agentfw.bench.replay import ScoreRow, SplitSpec, read_score_table, replay, write_score_table
from agentfw.core.config import EngineConfig
from agentfw.core.engine import KNOWN_DETECTORS, Engine
from agentfw.core.envelope import (
    Decision,
    DetectorResult,
    DetectorScore,
    Regime,
    TrajectoryInfo,
)
from agentfw.core.normalize import SessionTracker
from agentfw.core.trajectory.authority import (
    Grade,
    Maturity,
    apply_ceiling,
    contributor_floor,
    contributors,
    evidence_n,
    grade,
    maturity_of,
)
from agentfw.core.trajectory.decide import decide, session_overlay
from agentfw.core.trajectory.escalate import SEVERITY
from agentfw.core.trajectory.flags import Flag
from agentfw.core.trajectory.gate import GateOutcome
from agentfw.synth.generate import generate
from tests.core.test_gate import HINTS, Clock, charge, lookup, refund, score_alone, warm
from tests.factories import run_session

CFG = EngineConfig()
AUTH = EngineConfig(authority=True)
#: Every trajectory switch on (F8 acceptance: the audit runs with all of them).
ALL_ON = EngineConfig(
    sessions=True,
    provisional=True,
    gate=True,
    escalate=True,
    flow_ledger=True,
    seq_age_discount=True,
    br_mode="modulate",
    session_enforce=True,
    authority=True,
)
BLOCK, RA, ALLOW = Decision.BLOCK, Decision.REQUIRE_APPROVAL, Decision.ALLOW
GATE_BLOCK = GateOutcome("block", "stripe.retrieve_charge", "charge_id")


def mag(
    score: float, n: int | None, *, prior: bool = False, field: str = "amount"
) -> DetectorScore:
    ev: dict[str, object] = {"prior_used": prior}
    if n is not None:
        ev["n"] = n
    return DetectorScore("magnitude", field, score, ev)


def nov(score: float, n_total: int, *, prior: bool = False) -> DetectorScore:
    return DetectorScore("novelty", "to[]", score, {"n_total": n_total, "prior_used": prior})


def vel(score: float, n_hist: int) -> DetectorScore:
    return DetectorScore("velocity", None, score, {"n_hist": n_hist, "window_count": 9})


def seq(
    score: float, fired: str | None, *, n_trans: int, n_prereq: int, n_flow: int
) -> DetectorScore:
    ev: dict[str, object] = {
        "k": 1,
        "n_trans": n_trans,
        "n_prereq": n_prereq,
        "n": n_flow,
        "fired": fired,
        "prereq_missing": "stripe.retrieve_charge" if fired == "prereq" else None,
    }
    return DetectorScore("sequence", None, score, ev)


def vital(score: float) -> DetectorScore:
    return DetectorScore("vital", "__loop__", score, {"window": 8, "observed": 3})


def fused_of(per: Sequence[DetectorScore]) -> float:
    vs = sorted((s.score for s in per if s.detector != "vital"), reverse=True) + [0.0, 0.0]
    return min(1.0, vs[0] + (CFG.fusion_bonus if vs[1] > CFG.fusion_bonus_gate else 0.0))


def mat(per: Sequence[DetectorScore], regime: Regime = "history_only") -> Maturity:
    return maturity_of(per, fused_of(per), regime, CFG)


# -- the ceiling ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("maturity", "decision", "capped_from", "notice"),
    [
        ("mature", BLOCK, None, None),
        ("warm", RA, BLOCK, None),
        ("cold", ALLOW, BLOCK, "warn"),
        ("bare", ALLOW, BLOCK, "inform"),
    ],
)
def test_ceiling_table(
    maturity: Maturity, decision: Decision, capped_from: Decision | None, notice: str | None
) -> None:
    """T-14: mature -> block; warm -> require_approval; cold -> allow + warn; bare -> allow +
    inform (audit only). ``Decision`` gains no member."""
    assert apply_ceiling(BLOCK, maturity, None, "history_only") == (decision, capped_from, notice)


@given(
    d=st.sampled_from(list(Decision)),
    m=st.sampled_from(["mature", "warm", "cold", "bare"]),
    regime=st.sampled_from(["history+schemas", "history_only", "schemas_only", "bare"]),
    g=st.sampled_from(
        [None, GATE_BLOCK, GateOutcome("escalate", None, "x"), GateOutcome("pass", None, None)]
    ),
)
def test_ceiling_never_raises_severity(
    d: Decision, m: Maturity, regime: Regime, g: GateOutcome | None
) -> None:
    out, capped_from, notice = apply_ceiling(d, m, g, regime)
    assert SEVERITY[out] <= SEVERITY[d]
    assert (capped_from is None) == (out is d)
    assert capped_from in (None, d)
    assert notice is None or out is ALLOW
    if out is d:
        assert notice is None


def test_ceiling_leaves_what_is_under_it() -> None:
    assert apply_ceiling(RA, "warm", None, "history_only") == (RA, None, None)
    assert apply_ceiling(RA, "cold", None, "history_only") == (ALLOW, RA, "warn")
    assert apply_ceiling(ALLOW, "bare", None, "bare") == (ALLOW, None, None)


def test_gate_block_exempt_with_schemas_only() -> None:
    """F8: a missing producer for a destructive tool is an axiomatic fact when the regime has
    schemas, so F4's prerequisite block is not capped there -- and only there."""
    for regime in ("schemas_only", "history+schemas"):
        for m in ("cold", "warm", "mature"):
            assert apply_ceiling(BLOCK, m, GATE_BLOCK, regime) == (BLOCK, None, None)
    assert apply_ceiling(BLOCK, "cold", GATE_BLOCK, "history_only") == (ALLOW, BLOCK, "warn")
    assert apply_ceiling(BLOCK, "bare", GATE_BLOCK, "bare") == (ALLOW, BLOCK, "inform")
    escalated = GateOutcome("escalate", None, "charge_id")
    assert apply_ceiling(BLOCK, "cold", escalated, "schemas_only") == (ALLOW, BLOCK, "warn")


# -- maturity -----------------------------------------------------------------------------------


def test_evidence_n_reads_each_detectors_count() -> None:
    assert evidence_n(mag(0.9, 412)) == 412
    assert evidence_n(nov(0.9, 57)) == 57
    assert evidence_n(vel(0.9, 33)) == 33
    assert evidence_n(DetectorScore("magnitude", "x", 0.9, {"cold": True})) is None
    assert evidence_n(DetectorScore("magnitude", "x", 0.9, {"n": True})) is None  # a bool is no n


def test_evidence_n_of_sequence_follows_the_part_that_fired() -> None:
    """The flow part's ``n`` is the id-flow history (0 on a non-write tool): a trans- or
    prereq-driven score is measured on its own count (T-45)."""
    assert evidence_n(seq(0.8, "trans", n_trans=640, n_prereq=90, n_flow=0)) == 640
    assert evidence_n(seq(0.8, "prereq", n_trans=640, n_prereq=90, n_flow=0)) == 90
    assert evidence_n(seq(0.8, "flow", n_trans=640, n_prereq=90, n_flow=12)) == 12
    assert evidence_n(seq(0.0, None, n_trans=640, n_prereq=90, n_flow=12)) == 12


def test_sequence_evidence_carries_n_on_every_part() -> None:
    eng, clock = warm(EngineConfig(), n=40)
    r = score_alone(eng, refund(clock, "x", 0, charge(3)))
    (s,) = [s for s in r.per_detector if s.detector == "sequence"]
    assert s.evidence["n_trans"] == 40  # k=0 back-off: the unigram total of a fresh session...
    assert s.evidence["n_prereq"] == 40  # ...and the refund's own prerequisite history
    assert isinstance(s.evidence["n"], int)  # id flow (T-25), unchanged


def test_maturity_cut_points() -> None:
    """T-14: cold n < 30, warm 30 <= n < 300, mature n >= 300."""
    assert [mat([mag(0.9, n)]) for n in (0, 29, 30, 299, 300)] == [
        "cold",
        "cold",
        "warm",
        "warm",
        "mature",
    ]


def test_maturity_is_the_minimum_over_contributors() -> None:
    # top1 alone (the second does not earn the bonus): its n decides
    assert mat([mag(0.9, 1000), nov(0.45, 3)]) == "mature"
    # the second earned the bonus (> 0.5): it contributed, the minimum wins
    assert mat([mag(0.9, 1000), nov(0.55, 3)]) == "cold"
    # ties at the top all count
    assert mat([mag(0.7, 1000), vel(0.7, 40)]) == "warm"


def test_no_contributor_is_mature_and_vital_never_contributes() -> None:
    assert mat([mag(0.29, 3)]) == "mature"  # fused < 0.3: nothing to cap
    assert mat([]) == "mature"
    assert mat([vital(1.0), mag(0.2, 3)]) == "mature"  # T-07: vital is not fused
    assert mat([vital(1.0), mag(0.9, 400)]) == "mature"


def test_contributor_without_n_counts_as_cold() -> None:
    assert mat([DetectorScore("magnitude", "x", 0.9, {"non_finite": True})]) == "cold"


def test_prior_used_caps_at_warm() -> None:
    assert mat([mag(0.9, 5000, prior=True)]) == "warm"
    assert mat([nov(0.9, 5000, prior=True)]) == "warm"
    assert mat([mag(0.9, 5000, prior=True), nov(0.6, 4)]) == "cold"  # never raises a cold one
    assert mat([mag(0.9, 5000), nov(0.2, 5, prior=True)]) == "mature"  # not a contributor


def test_bare_regime_forces_bare() -> None:
    assert mat([mag(0.9, 5000)], "bare") == "bare"
    assert mat([], "bare") == "bare"
    assert mat([mag(0.9, 5000)], "schemas_only") == "mature"


def test_contributor_floor_follows_the_intervention_threshold() -> None:
    """T-50: a band intervenes from ``θ_eff - cal_delta``; when that is under 0.3 a decision can
    rest on a score under 0.3, which then contributes and is graded."""
    assert contributor_floor(None, CFG) == 0.3
    assert contributor_floor(0.6, CFG) == 0.3  # 0.5 > 0.3
    assert contributor_floor(0.3, CFG) == pytest.approx(0.2)
    low = [mag(0.25, 12)]
    assert maturity_of(low, 0.25, "history_only", CFG) == "mature"  # no theta: floor 0.3
    assert maturity_of(low, 0.25, "history_only", CFG, theta_eff=0.6) == "mature"
    assert maturity_of(low, 0.25, "history_only", CFG, theta_eff=0.24) == "cold"
    assert grade(low, "history_only", CFG, 0.24) == Grade("cold", 12)
    # a top score of 0 never contributes, even with a floor at or below 0
    assert contributors([mag(0.0, 3)], CFG, floor=-0.1) == []


def test_grade_reports_the_count_that_set_the_maturity() -> None:
    assert grade([mag(0.9, 1140)], "history_only", CFG) == Grade("mature", 1140)
    assert grade([mag(0.9, 1140), nov(0.6, 40)], "history_only", CFG) == Grade("warm", 40)
    assert grade([mag(0.9, None)], "history_only", CFG) == Grade("cold", None)
    assert grade([mag(0.9, 5000, prior=True)], "history_only", CFG) == Grade("warm", 5000)
    assert grade([mag(0.9, 1140)], "bare", CFG) == Grade("bare", None)
    assert grade([mag(0.1, 1140)], "history_only", CFG) == Grade("mature", None)


def test_trans_driven_score_is_not_cold_on_a_read_tool() -> None:
    """A read tool's flow ``n`` is 0; its surprising transition rests on the context's count."""
    assert mat([seq(0.8, "trans", n_trans=640, n_prereq=0, n_flow=0)]) == "mature"
    assert mat([seq(0.8, "prereq", n_trans=640, n_prereq=60, n_flow=0)]) == "warm"


# -- decide -----------------------------------------------------------------------------------


def info(m: Maturity | None, regime: Regime = "history_only", irr: bool = False) -> TrajectoryInfo:
    return TrajectoryInfo(regime, None, irr, 0.6, maturity=m)


def test_decide_applies_the_ceiling_only_with_authority() -> None:
    d, t = decide(0.9, info("cold"), CFG)
    assert (d, t.capped_from, t.notice) == (BLOCK, None, None)
    assert Flag.COLD_START not in t.flags
    d, t = decide(0.9, info("cold"), AUTH)
    assert (d, t.capped_from, t.notice) == (ALLOW, BLOCK, "warn")
    assert Flag.COLD_START in t.flags
    d, t = decide(0.9, info("warm"), AUTH)
    assert (d, t.capped_from, t.notice) == (RA, BLOCK, None)
    assert Flag.COLD_START not in t.flags
    d, t = decide(0.9, info("bare", "bare"), AUTH)
    assert (d, t.capped_from, t.notice) == (ALLOW, BLOCK, "inform")
    assert Flag.COLD_START not in t.flags  # the brief: cold only


def test_overlay_block_clears_the_notice() -> None:
    """T-47: the ceiling does not cap the session overlay, and a blocked call has no notice."""
    d, t = decide(0.9, info("cold"), AUTH.replace(session_enforce=True))
    assert (d, t.notice) == (ALLOW, "warn")
    d, t = session_overlay(d, t, True, AUTH.replace(session_enforce=True))
    assert (d, t.capped_from, t.notice) == (BLOCK, BLOCK, None)


def test_cold_start_flag_also_on_uncapped_calls() -> None:
    d, t = decide(0.1, info("cold"), AUTH)
    assert d is ALLOW and t.capped_from is None and Flag.COLD_START in t.flags


def test_decide_needs_maturity_with_authority() -> None:
    with pytest.raises(ValueError, match="maturity"):
        decide(0.9, info(None), AUTH)
    decide(0.9, info(None), CFG)  # off: not read


def test_authority_validated_only_when_on() -> None:
    Engine(EngineConfig(mat_warm_n=500, mat_mature_n=10))
    for bad in ({"mat_warm_n": 500, "mat_mature_n": 10}, {"mat_warm_n": 0}):
        with pytest.raises(ValueError, match="mat_"):
            Engine(AUTH.replace(**bad))


# -- the engine -------------------------------------------------------------------------------


def traj(r: DetectorResult) -> TrajectoryInfo:
    assert r.trajectory is not None
    return r.trajectory


def spike(cfg: EngineConfig, n: int = 40) -> DetectorResult:
    """A refund with an absurd amount after ``n`` benign look-up-then-refund sessions."""
    eng, clock = warm(cfg.replace(flow_ledger=True), n=n)
    tracker = SessionTracker()
    ch = charge(777)
    run_session(eng, [lookup(clock, "big", 0, ch)], tracker=tracker)
    return score_alone(eng, refund(clock, "big", 1, ch, amount=9_000_000.0), tracker)


def test_engine_caps_a_cold_block_and_reports_maturity() -> None:
    off = spike(EngineConfig())
    assert off.decision is BLOCK and traj(off).maturity == "warm"  # 40 observations
    assert traj(off).capped_from is None and traj(off).notice is None
    on = spike(AUTH)
    assert on.decision is RA and traj(on).capped_from is BLOCK and traj(on).notice is None
    cold = spike(AUTH.replace(mat_warm_n=100, mat_mature_n=200))
    assert cold.decision is ALLOW and traj(cold).maturity == "cold"
    assert traj(cold).capped_from is BLOCK and traj(cold).notice == "warn"
    assert Flag.COLD_START in traj(cold).flags
    mature = spike(AUTH.replace(mat_warm_n=10, mat_mature_n=20))
    assert mature.decision is BLOCK and traj(mature).maturity == "mature"
    assert off.fused_score == on.fused_score == cold.fused_score == mature.fused_score


def test_flags_are_derived_from_the_pre_cap_decision() -> None:
    """T-46: the ceiling moves the per-call decision only; the session window counts the call's
    own evidence (the band's BLOCK), so densities and verdicts do not depend on ``authority``."""
    on = spike(EngineConfig(sessions=True, authority=True))
    off = spike(EngineConfig(sessions=True))
    assert on.decision is RA and off.decision is BLOCK
    assert Flag.BLOCK in traj(on).flags
    assert traj(on).flags == traj(off).flags
    assert traj(on).density == traj(off).density
    assert traj(on).session_verdict == traj(off).session_verdict


@pytest.mark.parametrize("cfg", [CFG, AUTH, ALL_ON.replace(session_enforce=False)])
def test_every_reason_carries_maturity_and_regime(cfg: EngineConfig) -> None:
    r = spike(cfg)
    assert r.reasons
    t = traj(r)
    assert t.maturity is not None
    for reason in r.reasons:
        assert reason.maturity == t.maturity and reason.regime == t.regime == "history+schemas"


def test_reasons_keep_blast_radius_as_br_mode_sets_it() -> None:
    assert all(x.blast_radius is None for x in spike(CFG).reasons)
    scored = spike(EngineConfig(br_mode="score"))
    b = traj(scored).blast_radius
    assert b is not None and all(x.blast_radius == b for x in scored.reasons)


def test_score_row_maturity_and_regime_round_trip(tmp_path: Path) -> None:
    eng, clock = Engine(AUTH, hints=HINTS), Clock()
    raws = [lookup(clock, "s", 0, charge(1)), refund(clock, "s", 1, charge(1))]
    rows = replay(raws, {}, eng, SplitSpec())
    assert [r.regime for r in rows] == ["schemas_only", "schemas_only"]
    assert all(r.maturity in {"mature", "warm", "cold", "bare"} for r in rows)
    p = tmp_path / "rows.parquet"
    write_score_table(rows, p)
    assert read_score_table(p) == rows
    names = [f.name for f in dataclasses.fields(ScoreRow)]
    assert names[-3:] == ["maturity", "regime", "threshold_eff"]


# -- the CI dataset: purity and the F8 audit ------------------------------------------------------


@pytest.fixture(scope="module")
def ci_dirs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """The CI dataset with its ``schemas.jsonl`` (the bench path) and a copy without it."""
    d = tmp_path_factory.mktemp("ci")
    generate(CI_DATASET, d)
    bare = tmp_path_factory.mktemp("ci_bare")
    shutil.copytree(d, bare, dirs_exist_ok=True)
    (bare / "schemas.jsonl").unlink()
    return d, bare


def _all_on_rows(d: Path) -> list[ScoreRow]:
    """Strict replay (a ``LeakError`` fails it) with every switch on, the bench protocol."""
    raws, labels, _ = load_dir(d)
    eng = Engine(ALL_ON, enabled=KNOWN_DETECTORS, hints=load_schema_hints(d))
    return replay(raws, labels, eng, SplitSpec(), strict=True, outcomes="review")


@pytest.fixture(scope="module")
def all_on_rows(ci_dirs: tuple[Path, Path]) -> dict[str, list[ScoreRow]]:
    schemas, bare = ci_dirs
    return {"schemas": _all_on_rows(schemas), "no_schemas": _all_on_rows(bare)}


def test_strict_replay_with_every_switch_on(all_on_rows: dict[str, list[ScoreRow]]) -> None:
    """``score`` stays pure with every switch on (the replay is strict), with and without the
    dataset's schemas; every row carries a maturity and a regime."""
    for rows in all_on_rows.values():
        assert rows and all(r.maturity is not None and r.regime is not None for r in rows)
        assert all(not math.isnan(r.fused_score) for r in rows)


def _halted_before(rows: Sequence[ScoreRow]) -> list[bool]:
    """Per row: was its session halted before it (offline, as the overlay reads it; rows are in
    replay order and a halt is sticky, no release in a replay; T-43)."""
    halted: set[str] = set()
    out: list[bool] = []
    for r in rows:
        out.append(r.session_id in halted)
        if r.session_verdict == "halt_session":
            halted.add(r.session_id)
    return out


def _prereq_exempt(r: ScoreRow) -> bool:
    """The brief's one exemption: F4's prerequisite block in a regime with schemas."""
    flags = set(r.flags.split(",")) if r.flags else set()
    return "PREREQ_MISSING" in flags and r.regime is not None and "schemas" in r.regime


def _contributors(r: ScoreRow) -> list[DetectorScore]:
    per = [
        DetectorScore(x["detector"], x["field"], x["score"], x["evidence"])
        for x in orjson.loads(r.per_detector_json)
    ]
    return contributors(per, ALL_ON, contributor_floor(r.threshold_eff, ALL_ON))


def cold_or_bare_blocks(rows: Sequence[ScoreRow]) -> list[ScoreRow]:
    """F8's audit, literally: ``block`` rows with ``cold`` / ``bare`` maturity, except F4's
    prerequisite block in a schema regime."""
    return [
        r
        for r in rows
        if r.decision == "block" and r.maturity in {"cold", "bare"} and not _prereq_exempt(r)
    ]


def ungraded_blocks(rows: Sequence[ScoreRow]) -> list[ScoreRow]:
    """``block`` rows (bar the same exemption) with no contributor: their maturity is a label,
    not a grading, so the audit above would pass on them vacuously (T-50)."""
    return [
        r for r in rows if r.decision == "block" and not _prereq_exempt(r) and not _contributors(r)
    ]


def _per_call(
    rows: Sequence[ScoreRow], audit: Callable[[Sequence[ScoreRow]], list[ScoreRow]]
) -> list[ScoreRow]:
    """``audit`` over the rows whose session was not halted before them (no overlay block)."""
    return audit([r for r, h in zip(rows, _halted_before(rows), strict=True) if not h])


def _fmt(bad: Sequence[ScoreRow]) -> str:
    return str([(r.envelope_id, r.maturity, r.flags, r.session_id) for r in bad[:6]])


_OVERLAY_BARE = (
    "T-47 pending: without schemas, 6 / 956 CI rows are session-overlay BLOCKs on bare-maturity "
    "calls (2 halted benign sessions); the ceiling does not cap the overlay"
)
_OVERLAY_UNGRADED_SCHEMAS = (
    "T-47 pending: with schemas, 2 / 956 CI rows are session-overlay BLOCKs of a halted benign "
    "session on calls with no contributor (graded mature by label only)"
)


def _xfail(reason: str) -> pytest.MarkDecorator:
    return pytest.mark.xfail(strict=True, raises=AssertionError, reason=reason)


_OVERLAY_UNGRADED_BARE = (
    "T-47 pending: without schemas, 4 / 956 CI rows are session-overlay BLOCKs on calls with no "
    "contributor (bare regime)"
)


@pytest.mark.slow
@pytest.mark.parametrize(
    "name",
    ["schemas", pytest.param("no_schemas", marks=_xfail(_OVERLAY_BARE))],
)
def test_no_block_from_cold_or_bare_in_audit(
    all_on_rows: dict[str, list[ScoreRow]], name: str
) -> None:
    """F8 acceptance: no ``block`` from a ``cold`` or ``bare`` maturity in the bench logs, unless
    it is F4's prerequisite block (flag ``PREREQ_MISSING``) in a regime with schemas."""
    bad = cold_or_bare_blocks(all_on_rows[name])
    assert not bad, f"{name}: {len(bad)} {_fmt(bad)}"


@pytest.mark.slow
@pytest.mark.parametrize(
    "name",
    [
        pytest.param("schemas", marks=_xfail(_OVERLAY_UNGRADED_SCHEMAS)),
        pytest.param("no_schemas", marks=_xfail(_OVERLAY_UNGRADED_BARE)),
    ],
)
def test_every_block_rests_on_contributors(
    all_on_rows: dict[str, list[ScoreRow]], name: str
) -> None:
    """The audit is not vacuous: every ``block`` row (bar the prerequisite exemption) has a
    non-empty contributor set at its own floor (T-50)."""
    bad = ungraded_blocks(all_on_rows[name])
    assert not bad, f"{name}: {len(bad)} {_fmt(bad)}"


@pytest.mark.slow
def test_per_call_blocks_are_graded_and_never_cold_or_bare(
    all_on_rows: dict[str, list[ScoreRow]],
) -> None:
    """Both properties hold, with and without schemas, on every block the per-call pipeline
    took: what fails above is the session overlay alone (T-47). An addition to the literal
    audits, not a replacement."""
    for name, rows in all_on_rows.items():
        for audit in (cold_or_bare_blocks, ungraded_blocks):
            bad = _per_call(rows, audit)
            assert not bad, f"{name} {audit.__name__}: {_fmt(bad)}"
