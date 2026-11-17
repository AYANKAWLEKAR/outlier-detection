"""Session-level labels and metrics (Task 1.6, F2/F10)."""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import pytest

from agentfw.bench.ci_regress import CI_DATASET
from agentfw.bench.dataset import load_dir
from agentfw.bench.metrics import compute_all, per_agent_fpr_at_recall, precision_at, splice_report
from agentfw.bench.registry import make_detector
from agentfw.bench.replay import (
    ScoreRow,
    SplitSpec,
    read_score_table,
    replay,
    write_score_table,
)
from agentfw.bench.session_metrics import (
    decision_benign_fpr,
    decision_control_fpr,
    decision_flagged,
    decision_recall_by_family,
    first_flag_halts,
    halt_metrics,
    halt_sensitivity,
    length_matched,
    matched_pairs,
    session_auroc,
    session_labels,
    session_scores,
    vital_false_flag_rate,
)
from agentfw.core.config import EngineConfig
from agentfw.synth.attacks import RUNAWAY_FAMILIES
from agentfw.synth.generate import generate

_ids = itertools.count()


def _vital_json(loop: float = 0.0, spin: float = 0.0, stall: float = 0.0) -> str:
    items = [
        {"detector": "magnitude", "field": "amount", "score": 1.0, "evidence": {}},
        *(
            {"detector": "vital", "field": f, "score": s, "evidence": {}}
            for f, s in (("__loop__", loop), ("__spin__", spin), ("__stall__", stall))
        ),
    ]
    return orjson.dumps(items).decode()


def _row(session: str, **kw: Any) -> ScoreRow:
    i = next(_ids)
    base: dict[str, Any] = {
        "envelope_id": f"e{i:06d}",
        "session_id": session,
        "agent_id": "A",
        "tool_id": "zendesk.get_ticket",
        "ts_unix_ms": 1_000 + i,
        "split": "test",
        "fused_score": 0.0,
        "per_detector_json": _vital_json(),
        "label": "benign",
        "attack_family": None,
        "attack_id": None,
        "injection_level": None,
        "strength": None,
        "step": None,
        "control_type": None,
        "is_warm": True,
        "n_key_obs": 100,
        "latency_us": 10.0,
        "threshold": None,
    }
    base.update(kw)
    return ScoreRow(**base)


def _atk(session: str, family: str, **kw: Any) -> ScoreRow:
    return _row(session, label="attack", attack_family=family, attack_id=f"atk-{session}", **kw)


def test_session_labels_priority() -> None:
    rows = [
        _row("plain"),
        _row("ctl", control_type="benign_unusual"),
        _row("run"),
        _atk("run", "loop"),
        _atk("run", "stall"),
        _row("mixed"),
        _atk("mixed", "spin"),
        _atk("mixed", "magnitude_spike"),
        _atk("hit", "novel_target"),
        _row("hit"),
    ]
    assert session_labels(rows) == {
        "plain": "benign",
        "ctl": "benign",  # a control is benign
        "run": "runaway",
        "mixed": "attacked",  # attacked beats runaway
        "hit": "attacked",
    }
    assert session_labels([]) == {}


def test_vital_false_flag_rate_counts_benign_sessions_with_a_full_vital_score() -> None:
    rows = [
        _row("b1", per_detector_json=_vital_json(loop=1.0)),
        _row("b1"),
        _row("b2", per_detector_json=_vital_json(loop=0.7, stall=0.5)),  # below 1: not a flag
        _row("b3", per_detector_json=_vital_json(spin=1.0)),
        _row("b4", per_detector_json="[]"),
        _row("b5", control_type="benign_unusual", per_detector_json=_vital_json(stall=1.0)),
        # not benign sessions: outside the denominator whatever they score
        _atk("r1", "loop", per_detector_json=_vital_json(loop=1.0)),
        _row("a1", per_detector_json=_vital_json(loop=1.0)),
        _atk("a1", "magnitude_spike"),
    ]
    assert vital_false_flag_rate(rows) == pytest.approx(3 / 5)
    assert math.isnan(vital_false_flag_rate([_atk("x", "loop")]))
    assert math.isnan(vital_false_flag_rate([]))


def _replayed(
    tmp_path: Path,
    families: list[str],
    attack_rate: float = 0.15,
    split: SplitSpec = SplitSpec(),  # noqa: B008 - frozen
) -> list[ScoreRow]:
    cfg = CI_DATASET.model_copy(update={"families": families, "attack_rate": attack_rate})
    generate(cfg, tmp_path)
    raws, labels, _ = load_dir(tmp_path)
    return replay(raws, labels, make_detector("engine-vital"), split)


def test_runaway_rows_excluded_from_per_call_metrics(tmp_path: Path) -> None:
    families = [*CI_DATASET.families, *sorted(RUNAWAY_FAMILIES)]
    rows = _replayed(tmp_path, families, attack_rate=0.5)  # every runaway family present
    runaway = [r for r in rows if r.attack_family in RUNAWAY_FAMILIES]
    others = [r for r in rows if r.attack_family not in RUNAWAY_FAMILIES]
    assert {r.attack_family for r in runaway} == RUNAWAY_FAMILIES
    assert any(r.label == "attack" for r in others)
    with_runaway = compute_all(rows, warm_only=False, n_boot=0)
    without = compute_all(others, warm_only=False, n_boot=0)
    assert repr(with_runaway) == repr(without)  # repr: NaN fields compare equal
    assert not RUNAWAY_FAMILIES & set(with_runaway.per_family_attack)
    labels = session_labels(rows)
    assert {labels[r.session_id] for r in runaway} <= {"runaway", "attacked"}
    assert "runaway" in labels.values()


@pytest.fixture(scope="module")
def benign_rows(tmp_path_factory: pytest.TempPathFactory) -> list[ScoreRow]:
    return _replayed(tmp_path_factory.mktemp("benign"), [])


def test_benign_dataset_sessions_are_all_benign(benign_rows: list[ScoreRow]) -> None:
    assert set(session_labels(benign_rows).values()) == {"benign"}
    assert len(session_labels(benign_rows)) == 200


# Measured at Task 1.6 (T-30): 12 of 200 sessions (6.00%) on this dataset, all LOOP, all
# sales-ops-a; full benign SW1 3.64% (1,538 / 42,289). The bar stays 1%. This test replays
# without schemas (no hints). With the dataset's schemas.jsonl auto-loaded (Task 1.7, T-30/T-32)
# this set is also 12/200; benign SW1 rises to 3.86% (1,631 / 42,289).
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "measured 6.00% (12/200 sessions) on CI_DATASET with families=[] "
        "(full benign SW1: 3.64%, 1,538/42,289): loop on task `repeat` iterations (same call "
        "re-issued within w_loop=8); F2 semantics vs bar pending user decision"
    ),
)
def test_benign_vital_false_flag_rate_at_most_1pct(benign_rows: list[ScoreRow]) -> None:
    """F2 acceptance: loop/spin/stall flag at most 1% of benign sessions."""
    assert vital_false_flag_rate(benign_rows) <= 0.01


# -- decision metrics (Task 2.2) ---------------------------------------------------------------


def test_decision_metrics_on_handmade_rows() -> None:
    def atk(attack: str, family: str, decision: str | None, ts: int) -> ScoreRow:
        return _row(
            f"s-{attack}",
            label="attack",
            attack_family=family,
            attack_id=attack,
            decision=decision,
            ts_unix_ms=ts,
        )

    rows = [
        # magnitude_spike: a1 caught on its 2nd row, a2 never -> 1/2
        atk("a1", "magnitude_spike", "allow", 1),
        atk("a1", "magnitude_spike", "block", 2),
        atk("a2", "magnitude_spike", "allow", 3),
        # novel_target: approval counts as flagged -> 1/1
        atk("a3", "novel_target", "require_approval", 4),
        # an attack row without a decision (unscored) is skipped; a4 has no scored row at all
        atk("a4", "novel_target", None, 5),
        # runaway families are in no decision metric (T-10)
        atk("r1", "loop", "block", 6),
        # benign: 1 of 4 flagged (controls included); controls: 1 of 2
        _row("b", decision="allow"),
        _row("b", decision="block"),
        _row("c", decision="allow", control_type="benign_unusual"),
        _row("c", decision="require_approval", control_type="benign_unusual"),
        _row("b", decision=None),  # unscored: outside every denominator
    ]
    assert [decision_flagged(r) for r in rows[:4]] == [False, True, False, True]
    assert not decision_flagged(rows[4])
    assert decision_recall_by_family(rows) == {"magnitude_spike": 0.5, "novel_target": 1.0}
    assert decision_benign_fpr(rows) == pytest.approx(2 / 4)
    assert decision_control_fpr(rows) == pytest.approx(1 / 2)
    assert decision_recall_by_family([]) == {}
    assert math.isnan(decision_benign_fpr([])) and math.isnan(decision_control_fpr(rows[:4]))


def test_decision_recall_family_comes_from_the_first_row() -> None:
    def atk(family: str, decision: str, ts: int) -> ScoreRow:
        return _row(
            "s",
            label="attack",
            attack_family=family,
            attack_id="x",
            decision=decision,
            ts_unix_ms=ts,
        )

    rows = [atk("b_fam", "block", 9), atk("a_fam", "allow", 1)]
    assert decision_recall_by_family(rows) == {"a_fam": 1.0}


# -- Task 2.4: session scores, AUROC, length-matched set, the session label column -------------


def test_session_auroc_on_handmade_rows() -> None:
    rows = [
        # attacked sessions: max fused 0.9 and 0.4 (an unscored NaN row is ignored)
        _row("a1", fused_score=0.2),
        _atk("a1", "magnitude_spike", fused_score=0.9),
        _atk("a2", "splice", fused_score=0.4),
        _row("a2", fused_score=math.nan),
        # a runaway session is a positive too (T-10: scored by session metrics)
        _atk("r1", "loop", fused_score=0.7),
        # benign sessions: 0.5, 0.1, 0.4 (a tie with a2), and one never scored (dropped)
        _row("b1", fused_score=0.5),
        _row("b1", fused_score=0.3),
        _row("b2", fused_score=0.1),
        _row("b3", fused_score=0.4, control_type="benign_unusual"),
        _row("b4", fused_score=math.nan),
    ]
    scores = session_scores(rows)
    assert {k: v for k, v in scores.items() if k != "b4"} == {
        "a1": 0.9, "a2": 0.4, "r1": 0.7, "b1": 0.5, "b2": 0.1, "b3": 0.4,
    }  # fmt: skip
    assert math.isnan(scores["b4"])
    # default: attacked only (r1 is dropped, not a negative). Positives 0.9, 0.4; negatives 0.5,
    # 0.4, 0.1: 0.9 beats all 3; 0.4 beats 0.1, ties 0.4 (1/2), loses to 0.5 -> (3 + 1.5) / 6
    assert session_auroc(rows) == pytest.approx(4.5 / 6)
    assert session_auroc(rows, frozenset({"attacked"})) == session_auroc(rows)
    # runaway-inclusive: positives 0.9, 0.7, 0.4 -> (3 + 3 + 1.5) / 9
    assert session_auroc(rows, frozenset({"attacked", "runaway"})) == pytest.approx(7.5 / 9)
    assert session_auroc(rows, frozenset({"runaway"})) == pytest.approx(1.0)  # a1, a2 dropped
    with pytest.raises(ValueError, match="benign"):
        session_auroc(rows, frozenset({"attacked", "benign"}))
    assert math.isnan(session_auroc([_row("b", fused_score=0.1)]))
    assert math.isnan(session_auroc([_atk("r", "loop", fused_score=0.1), _row("b")]))  # no positive
    assert math.isnan(session_auroc([]))


def _lengths(agent: str, label: str, sizes: dict[str, int]) -> list[ScoreRow]:
    out = []
    for sess, n in sizes.items():
        for _ in range(n):
            if label == "benign":
                out.append(_row(sess, agent_id=agent))
            else:
                out.append(_atk(sess, label, agent_id=agent))
    return out


def test_length_matched_pairs_within_tol() -> None:
    rows = [
        *_lengths("A", "magnitude_spike", {"A-x1": 10, "A-x2": 11, "A-x3": 30, "A-x4": 5}),
        *_lengths("A", "benign", {"A-b1": 12, "A-b2": 9, "A-b3": 11, "A-b4": 50, "A-b5": 3}),
        *_lengths("B", "splice", {"B-x1": 8}),
        *_lengths("B", "benign", {"B-b1": 9}),
        *_lengths("A", "loop", {"A-r1": 12}),  # a runaway session is never paired (T-40)
        *_lengths("C", "novel_target", {"C-x1": 9}),  # no benign session of C: dropped
        *_lengths("D", "benign", {"D-b1": 9}),
    ]
    count = {s: sum(r.session_id == s for r in rows) for s in {r.session_id for r in rows}}
    agent = {r.session_id: r.agent_id for r in rows}
    labels = session_labels(rows)
    for seed in range(20):
        pairs = matched_pairs(rows, seed=seed)
        assert pairs == matched_pairs(rows, seed=seed)  # deterministic
        used = [s for p in pairs for s in p]
        assert len(used) == len(set(used))  # no session reused
        for atk, ben in pairs:
            assert labels[atk] == "attacked" and labels[ben] == "benign"
            assert agent[atk] == agent[ben]
            assert abs(count[atk] - count[ben]) <= 2
        # A-x1 (10), A-x2 (11) and A-x4 (5, with b5 at 3) can always pair; A-x3 (30) never can
        assert {a for a, _ in pairs} == {"A-x1", "A-x2", "A-x4", "B-x1"}
        kept = length_matched(rows, seed=seed)
        assert {r.session_id for r in kept} == set(used)
        assert kept == [r for r in rows if r.session_id in set(used)]  # row order kept
    assert len({tuple(matched_pairs(rows, seed=s)) for s in range(20)}) > 1  # the seed matters
    assert matched_pairs(rows, tol=0) == [("A-x2", "A-b3")]  # the only equal lengths
    assert length_matched([]) == [] and matched_pairs([]) == []


def test_session_label_column_filled(tmp_path: Path) -> None:
    families = [*CI_DATASET.families, "splice", "loop"]
    rows = _replayed(tmp_path, families, attack_rate=0.5)
    labels = session_labels(rows)
    assert all(r.session_label == labels[r.session_id] for r in rows)
    assert set(labels.values()) == {"benign", "runaway", "attacked"}
    spliced = {r.session_id for r in rows if r.attack_family == "splice"}
    assert spliced and {labels[s] for s in spliced} == {"attacked"}
    p = tmp_path / "scores.parquet"
    write_score_table(rows, p)
    assert read_score_table(p) == rows


def test_splice_is_reported_at_splice_free_thresholds(tmp_path: Path) -> None:
    """T-40 (review round 1): headline metrics and their thresholds are fitted on splice-free rows,
    and ``splice_report`` scores splice recall at those same thresholds."""
    # every row in test: on this dataset all splices land on sales-ops-a, inside the default
    # time split's train part
    rows = _replayed(tmp_path, ["splice"], attack_rate=0.5, split=SplitSpec(train_frac=0.0))
    without = [r for r in rows if r.attack_family != "splice"]
    assert len(without) < len(rows)
    for mode in ("per_agent", "row"):
        m = compute_all(rows, warm_only=False, breakdown_thresholds=mode, n_boot=0)
        assert repr(m) == repr(  # repr: NaN fields compare equal
            compute_all(without, warm_only=False, breakdown_thresholds=mode, n_boot=0)
        )
        assert "splice" not in m.per_family_attack
        rep = splice_report(rows, warm_only=False, breakdown_thresholds=mode)
        test_splices = {
            r.attack_id for r in rows if r.attack_family == "splice" and r.split == "test"
        }
        assert rep.attack.n_total == len(test_splices) > 0
        assert rep.envelope.n_total == sum(
            r.attack_family == "splice" and r.split == "test" for r in rows
        )
        assert 0 <= rep.attack.n_detected <= rep.attack.n_total
        assert rep.breakdown_mode == mode
    # the per-agent thresholds are those compute_all fitted, without splice rows
    pa = per_agent_fpr_at_recall([r for r in without if r.split != "calibration"])
    hit = {
        r.attack_id
        for r in rows
        if r.attack_family == "splice"
        and r.split == "test"
        and r.fused_score >= pa.thresholds.get(r.agent_id, math.inf)
    }
    assert splice_report(rows, warm_only=False).attack.n_detected == len(hit)


# -- Task 3.3: halt metrics, the sensitivity curve, the first-flag baseline (F7) -----------------


def _halt_rows() -> list[ScoreRow]:
    """Handmade sessions (density, flags) for the halt metrics; see the test for the numbers."""

    def b(sess: str, d: float, flags: str = "", **kw: Any) -> ScoreRow:
        return _row(sess, density=d, flags=flags, **kw)

    def a(sess: str, fam: str, d: float, flags: str = "") -> ScoreRow:
        return _atk(sess, fam, density=d, flags=flags)

    return [
        b("b1", 0.0), b("b1", 1 / 6), b("b1", 0.5, "LOOP"),  # flags, never halts at 0.6
        b("b2", 0.0), b("b2", 4 / 6, "LOOP"),  # a false halt
        b("b3", 0.0), b("b3", 0.0), b("b3", 0.0),
        b("b4", 0.2, "NOVEL_READ,VELOCITY_BURST", control_type="benign_unusual"),
        b("b5", 0.0, "NOVEL_READ,PROVISIONAL"),  # non-severe flags only
        b("r1", 0.0), a("r1", "loop", 1 / 6), a("r1", "loop", 0.5, "LOOP"),
        a("r1", "loop", 4 / 6, "LOOP"),  # halts 2 calls after its onset (row 1)
        a("r2", "stall", 1 / 6, "STALL"), a("r2", "stall", 2 / 6), a("r2", "stall", 0.4),
        b("a1", 0.7), a("a1", "magnitude_spike", 0.8, "BLOCK"),  # halts before its onset
        a("a2", "novel_target", 0.1),
    ]  # fmt: skip


def test_halt_metrics_on_handmade_rows() -> None:
    rows = _halt_rows()
    m = halt_metrics(rows, 0.6)
    assert m.false_halt_rate == pytest.approx(1 / 5)  # b2 of b1..b5
    assert m.recall_by_label == {"runaway": 0.5, "attacked": 0.5}  # r1 of r1, r2; a1 of a1, a2
    assert m.median_calls_from_onset == 2.0  # r1 only: a1 halted before its onset
    assert m.precision_at_base_rate == pytest.approx(precision_at(2 / 4, 1 / 5, 0.015))
    assert halt_metrics(rows, 0.6, base_rate=0.5).precision_at_base_rate == pytest.approx(
        precision_at(2 / 4, 1 / 5, 0.5)
    )
    # a halt is the first density >= theta (inclusive), so theta = 4/6 still halts r1 and b2
    assert halt_metrics(rows, 4 / 6) == m
    low = halt_metrics(rows, 0.5)
    assert low.false_halt_rate == pytest.approx(2 / 5)
    assert low.recall_by_label == {"runaway": 0.5, "attacked": 0.5}
    assert low.median_calls_from_onset == 1.0  # r1 now halts on row 2
    # Shuffled rows give the same numbers (sessions are ordered by timestamp).
    assert halt_metrics(list(reversed(rows)), 0.6) == m
    empty = halt_metrics([], 0.6)
    assert math.isnan(empty.false_halt_rate) and math.isnan(empty.median_calls_from_onset)
    assert all(math.isnan(v) for v in empty.recall_by_label.values())
    assert math.isnan(empty.precision_at_base_rate)


def test_halt_metrics_need_the_density_column() -> None:
    rows = [_row("b1"), _atk("r1", "loop")]  # density None: the session gate was shut
    with pytest.raises(ValueError, match="density"):
        halt_metrics(rows, 0.6)
    with pytest.raises(ValueError, match="density"):
        halt_sensitivity(rows)


def _random_rows(seed: int) -> list[ScoreRow]:
    rng = np.random.default_rng(seed)
    rows: list[ScoreRow] = []
    for s in range(120):
        fam = ["benign", "loop", "magnitude_spike"][int(rng.integers(3))]
        for _ in range(int(rng.integers(1, 15))):
            d = int(rng.integers(0, 7)) / 6
            if fam == "benign" or rng.random() < 0.3:
                rows.append(_row(f"s{seed}-{s}", density=d))
            else:
                rows.append(_atk(f"s{seed}-{s}", fam, density=d))
    return rows


@pytest.mark.parametrize("seed", range(5))
def test_sensitivity_curve_is_monotone_non_increasing_in_false_halts(seed: int) -> None:
    rows = _random_rows(seed)
    curve = halt_sensitivity(rows)
    thetas = [t for t, _, _ in curve]
    assert thetas == [round(0.2 + 0.05 * i, 10) for i in range(13)]  # 0.2 .. 0.8, exact grid
    assert 0.5 in thetas and 0.6 in thetas
    fh = [f for _, f, _ in curve]
    rec = [r for _, _, r in curve]
    assert all(x >= y for x, y in itertools.pairwise(fh))
    assert all(x >= y for x, y in itertools.pairwise(rec))
    assert fh[0] > fh[-1]  # the curve is exercised
    for t, f, r in curve:
        m = halt_metrics(rows, t)
        assert (f, r) == (m.false_halt_rate, m.recall_by_label["runaway"])


def test_first_flag_baseline_halts_earlier_and_more() -> None:
    rows = _halt_rows()
    density = halt_metrics(rows, 0.6)
    first = first_flag_halts(rows)
    # b1, b2 (LOOP) halt; b4's VELOCITY_BURST is severe; b3 and b5 (non-severe flags) do not
    assert first.false_halt_rate == pytest.approx(3 / 5)
    assert first.recall_by_label == {"runaway": 1.0, "attacked": 0.5}
    assert first.median_calls_from_onset == 0.0  # r1: 1, r2: 0, a1: 0
    assert first.false_halt_rate > density.false_halt_rate
    assert first.median_calls_from_onset < density.median_calls_from_onset
    for lab, rec in density.recall_by_label.items():
        assert first.recall_by_label[lab] >= rec


def test_offline_halts_match_the_engine_verdicts(tmp_path: Path) -> None:
    """Halt metrics recompute the engine's sticky halt exactly from the density column."""
    cfg = CI_DATASET.model_copy(
        update={"families": [*CI_DATASET.families, *sorted(RUNAWAY_FAMILIES)], "attack_rate": 0.5}
    )
    generate(cfg, tmp_path)
    raws, labels, _ = load_dir(tmp_path)
    det = make_detector("engine-vital", cfg=EngineConfig(sessions=True, theta_halt=0.5))
    rows = replay(raws, labels, det, SplitSpec())
    by_session: dict[str, list[ScoreRow]] = {}
    for r in rows:
        by_session.setdefault(r.session_id, []).append(r)
    n_halted = 0
    for sess in by_session.values():
        engine_halt = next(
            (i for i, r in enumerate(sess) if r.session_verdict == "halt_session"), None
        )
        offline = next(
            (i for i, r in enumerate(sess) if r.density is not None and r.density >= 0.5), None
        )
        assert engine_halt == offline
        if engine_halt is not None:  # sticky from there on
            assert all(r.session_verdict == "halt_session" for r in sess[engine_halt:])
            n_halted += 1
    assert n_halted > 0
    labels_of = session_labels(rows)
    halted = {
        s
        for s, sess in by_session.items()
        if any(r.session_verdict == "halt_session" for r in sess)
    }
    benign = [s for s, lab in labels_of.items() if lab == "benign"]
    assert halt_metrics(rows, 0.5).false_halt_rate == pytest.approx(
        sum(s in halted for s in benign) / len(benign)
    )


def test_sessions_without_an_observed_density_are_out_of_the_denominators() -> None:
    """A sampled run leaves unsampled rows' density ``None``; such a session was never
    evaluated, so it is neither halted nor not halted (T-76)."""
    rows = [_row("halt", density=0.9), _row("calm", density=0.1)]
    unscored = [_row("unscored", density=None), _row("unscored", density=None)]
    assert halt_metrics(rows, 0.5).false_halt_rate == 0.5
    assert halt_metrics(rows + unscored, 0.5).false_halt_rate == 0.5
