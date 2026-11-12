from __future__ import annotations

import math
from collections.abc import Iterable
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import pytest

from agentfw.bench.ci_regress import ci_dataset, golden_rows
from agentfw.bench.metrics import compute_all
from agentfw.bench.protocol import Detector
from agentfw.bench.registry import DETECTORS, make_detector
from agentfw.bench.replay import ScoreRow, SplitSpec, canonical_hash, replay
from agentfw.core.config import EngineConfig
from agentfw.core.detectors.base import squash
from agentfw.core.detectors.magnitude import RESP_FIELD
from agentfw.core.engine import ALL_DETECTORS, Engine
from agentfw.core.envelope import (
    ActionEnvelope,
    Decision,
    DetectorResult,
    DetectorScore,
    FieldType,
    Outcome,
    ResponseMeta,
)
from agentfw.core.hints import ToolHint
from agentfw.core.normalize import SessionTracker
from agentfw.core.state import DictStore, key_cal, key_mag, key_nov, key_prior_mag, key_prior_nov
from tests.factories import make_env

ROOT = Path(__file__).resolve().parents[2]
TOOL = "stripe.create_refund"
LIST = "zendesk.list_tickets"
T0 = 1_700_000_000_000
MIN = 60_000
# Amendment A1.3 (numeric values of BOOL-typed fields): the amended rule with the A1.2 tail score,
# and the pre-amendment behaviour pinned explicitly (not through the default, which flips later).
A1 = EngineConfig().replace(mag_bool_numeric=True, mag_score="tail")
PRE = EngineConfig().replace(mag_bool_numeric=False)


def env(
    args: dict[str, Any] | None = None,
    seq: int = 0,
    *,
    agent: str = "a1",
    tool: str = TOOL,
    session: str = "s1",
    tracker: SessionTracker | None = None,
    response_meta: ResponseMeta | None = None,
) -> ActionEnvelope:
    """One envelope per minute (``seq``); observed into ``tracker`` when one is given."""
    e = make_env(
        tracker=tracker,
        agent_id=agent,
        tool_id=tool,
        session_id=session,
        seq_no=seq,
        ts_unix_ms=T0 + seq * MIN,
        args={"amount": 37.5} if args is None else args,
        response_meta=response_meta,
    )
    if tracker is not None:
        tracker.observe(e)
    return e


def feed(eng: Engine, envs: Iterable[ActionEnvelope], outcome: Outcome = Outcome.ALLOWED) -> None:
    for e in envs:
        eng.score(e)
        eng.update(e, outcome)


def dump(eng: Engine, path: Path) -> bytes:
    eng.store.dump(path)
    return path.read_bytes()


def amounts(n: int, seed: int = 0, agent: str = "a1", start: int = 0) -> list[ActionEnvelope]:
    rng = np.random.default_rng(seed)
    return [
        env({"amount": round(float(x), 2), "currency": "usd"}, start + i, agent=agent)
        for i, x in enumerate(rng.lognormal(3.6, 0.35, n))
    ]


def by(r: DetectorResult, detector: str, field: str | None = None) -> DetectorScore:
    (s,) = [s for s in r.per_detector if s.detector == detector and s.field == field]
    return s


# -- construction ---------------------------------------------------------------------------


def test_engine_is_a_detector() -> None:
    eng: Detector = Engine()
    assert isinstance(eng, Engine)
    assert frozenset({"magnitude", "novelty", "velocity", "sequence"}) == ALL_DETECTORS


def test_unknown_detector_name_rejected() -> None:
    with pytest.raises(ValueError, match="bogus"):
        Engine(enabled=frozenset({"magnitude", "bogus"}))


def test_shared_store_is_used() -> None:
    store = DictStore()
    eng = Engine(store=store)
    feed(eng, amounts(12))
    assert eng.store is store
    assert len(store) > 0


def test_enabled_subset_limits_per_detector_output() -> None:
    eng = Engine(enabled=frozenset({"velocity"}))
    feed(eng, amounts(40))
    r = eng.score(amounts(1, seed=1, start=40)[0])
    assert {s.detector for s in r.per_detector} == {"velocity"}


def test_all_detectors_report() -> None:
    eng = Engine()
    feed(eng, amounts(40))
    r = eng.score(amounts(1, seed=1, start=40)[0])
    assert {s.detector for s in r.per_detector} == set(ALL_DETECTORS)


# -- result contents ------------------------------------------------------------------------


def test_fresh_engine_threshold_and_decision() -> None:
    cfg = EngineConfig()
    r = Engine(cfg).score(env())
    assert r.threshold == cfg.cal_theta0
    assert r.fused_score == 0.0
    assert r.decision is Decision.ALLOW
    assert r.latency_us > 0.0


def test_spike_is_blocked_with_reason() -> None:
    eng = Engine()
    feed(eng, amounts(200))
    r = eng.score(env({"amount": 37.5 * 50, "currency": "usd"}, 200))
    assert r.fused_score > 0.9
    assert r.decision is Decision.BLOCK
    assert r.reasons[0].detector == "magnitude"
    assert r.reasons[0].field == "amount"
    assert r.threshold is not None and r.fused_score >= r.threshold


def test_field_types_cover_flat_and_derived_paths() -> None:
    eng = Engine()
    args = {"amount": 10.0, "email": "bob@acme.example", "currency": "usd"}
    feed(eng, [env(args, i) for i in range(12)])
    r = eng.score(env(args, 12))
    assert dict(r.field_types) == {
        "amount": FieldType.NUMERIC,
        "email": FieldType.ID,
        "email.__domain__": FieldType.CATEGORICAL,
        "currency": FieldType.CATEGORICAL,
    }


def test_field_types_unknown_before_min_obs() -> None:
    r = Engine().score(env({"amount": 10.0, "note": "x" * 50}))
    assert dict(r.field_types) == {
        "amount": FieldType.UNKNOWN,
        "note.__strlen__": FieldType.NUMERIC,
    }


# -- update ordering and gating ------------------------------------------------------------


def test_types_observed_last_in_update() -> None:
    """Detectors in update see the types as of *before* the envelope (types are observed last)."""
    eng = Engine()
    envs = amounts(11)
    feed(eng, envs[:10])  # the 10th update assigns NUMERIC, but only after the detectors ran
    assert eng.store.get(key_mag("a1", TOOL, "amount")) is None
    feed(eng, envs[10:])
    rec = eng.store.get(key_mag("a1", TOOL, "amount"))
    assert rec is not None and rec.n == 1  # type: ignore[attr-defined]


def test_blocked_action_does_not_move_baseline(tmp_path: Path) -> None:
    # Spec behaviour: refused calls move nothing (vel_count_attempts, default on, pinned off).
    eng = Engine(EngineConfig().replace(vel_count_attempts=False))
    feed(eng, amounts(120))
    before = dump(eng, tmp_path / "before.jsonl")
    spike = env({"amount": 1e6, "currency": "xxx", "new": "cus_ABCDEFGH12"}, 120)
    r1 = eng.score(spike)
    eng.update(spike, Outcome.BLOCKED)
    eng.update(spike, Outcome.DENIED)
    assert dump(eng, tmp_path / "after.jsonl") == before
    assert eng.score(spike).fused_score == r1.fused_score


def _records(eng: Engine, path: Path) -> dict[str, dict[str, Any]]:
    return {(r := orjson.loads(ln))["key"]: r for ln in dump(eng, path).splitlines()}


@pytest.mark.parametrize("outcome", [Outcome.BLOCKED, Outcome.DENIED])
def test_default_refused_call_moves_only_the_velocity_refused_window(
    outcome: Outcome, tmp_path: Path
) -> None:
    """Default config (vel_count_attempts on, user decision): a refused call adds its minute to
    velocity's ``refused`` list and changes nothing else, in any record."""
    eng = Engine()
    assert eng.cfg.vel_count_attempts
    feed(eng, amounts(120))
    before = _records(eng, tmp_path / "before.jsonl")
    spike = env({"amount": 1e6, "currency": "xxx", "new": "cus_ABCDEFGH12"}, 120)
    r1 = eng.score(spike)
    eng.update(spike, outcome)
    after = _records(eng, tmp_path / "after.jsonl")
    assert set(after) == set(before)
    changed = sorted(k for k in after if after[k] != before[k])
    assert changed, "the refused attempt was not counted"
    for k in changed:
        new, old = after[k], before[k]
        assert new["kind"] == "vel"
        added = new["data"].pop("refused")
        assert len(added) == len(old["data"].pop("refused", [])) + 1
        assert new == old  # nothing but the refused-window list moved
    assert eng.score(spike).fused_score >= r1.fused_score


def test_approved_moves_baseline_like_allowed(tmp_path: Path) -> None:
    a, b = Engine(), Engine()
    envs = amounts(40)
    feed(a, envs, Outcome.ALLOWED)
    feed(b, envs, Outcome.APPROVED)
    assert dump(a, tmp_path / "a.jsonl") == dump(b, tmp_path / "b.jsonl")
    assert a.store.get(key_cal("a1")) is not None


def test_freetext_never_stored(tmp_path: Path) -> None:
    eng = Engine()
    body = "SECRET-TOKEN-XYZ " * 10
    for i in range(50):
        e = env({"body": body, "amount": 10.0, "items": [{"note": body}]}, i)
        eng.score(e)
        eng.update(e, Outcome.ALLOWED)
    eng.store.dump(tmp_path / "s.jsonl")
    assert "SECRET-TOKEN-XYZ" not in (tmp_path / "s.jsonl").read_text()


# -- calibrator feed: cached vs fresh score -------------------------------------------------


def _warm() -> Engine:
    eng = Engine()
    feed(eng, amounts(250))
    return eng


ODD = env({"amount": 1e6, "currency": "usd"}, 250)  # magnitude spike: fused 1.0
MID = env({"amount": 112.0, "currency": "usd"}, 251)  # mild spike: fused strictly inside (0, 1)
NOVEL = env({"amount": 40.0, "currency": "eur"}, 252)  # novel before its update, seen after


def test_update_without_cached_score_scores_before_updating(tmp_path: Path) -> None:
    eng, ref = _warm(), _warm()
    f_before = ref.score(NOVEL).fused_score
    ref.update(NOVEL, Outcome.ALLOWED)
    # Discrimination: once NOVEL is folded in, its score drops ("eur" is no longer novel), so
    # a fused computed after the detector updates would feed the calibrator a different value.
    assert ref.score(NOVEL).fused_score < f_before
    eng.update(NOVEL, Outcome.ALLOWED)  # no cached score: scored fresh, before any update
    assert dump(eng, tmp_path / "a.jsonl") == dump(ref, tmp_path / "b.jsonl")


def test_update_ignores_cache_of_other_envelope(tmp_path: Path) -> None:
    eng, ref = _warm(), _warm()
    f_odd, f_mid = eng.score(ODD).fused_score, ref.score(MID).fused_score
    assert 0.0 < f_mid < f_odd  # the check below only discriminates when the scores differ
    eng.update(MID, Outcome.ALLOWED)  # the cache holds ODD's score: it must not be used
    ref.update(MID, Outcome.ALLOWED)
    assert dump(eng, tmp_path / "a.jsonl") == dump(ref, tmp_path / "b.jsonl")


def test_calibrator_fed_the_score_that_decided(monkeypatch: pytest.MonkeyPatch) -> None:
    """Interleaved calls: the cached fused of an envelope survives another envelope's update."""
    eng = _warm()
    fed: list[float] = []
    observe = eng.calibrator.observe

    def spy(agent: str, fused: float, outcome: Outcome) -> None:
        fed.append(fused)
        observe(agent, fused, outcome)

    monkeypatch.setattr(eng.calibrator, "observe", spy)
    eur = env({"amount": 41.0, "currency": "eur"}, 253)
    f_novel = eng.score(NOVEL).fused_score
    eng.update(eur, Outcome.ALLOWED)  # no cached score for it: scored fresh; folds "eur" in
    eng.update(NOVEL, Outcome.ALLOWED)  # NOVEL's cached score, taken before that update
    assert fed[1] == f_novel
    other = _warm()  # discrimination: re-scoring NOVEL after the "eur" update gives less
    other.update(eur, Outcome.ALLOWED)
    assert other.score(NOVEL).fused_score < f_novel
    eng.update(NOVEL, Outcome.ALLOWED)  # the cache was consumed: NOVEL is now scored fresh
    assert fed[2] < f_novel


def test_reset_clears_state_and_cache(tmp_path: Path) -> None:
    eng = _warm()
    assert eng.score(ODD).fused_score == 1.0
    eng.reset()
    assert eng.state_bytes() == 0
    eng.update(ODD, Outcome.ALLOWED)  # must score fresh (0.0) against the empty state
    fresh = Engine()
    fresh.update(ODD, Outcome.ALLOWED)
    assert eng.store.get(key_cal("a1")) is not None
    assert dump(eng, tmp_path / "a.jsonl") == dump(fresh, tmp_path / "b.jsonl")


# -- hints and priors ------------------------------------------------------------------------


def test_hints_reach_detectors() -> None:
    hints = {TOOL: ToolHint(field_hints={"foo": "money"})}
    envs = [env({"foo": 5.0}, i) for i in range(11)]
    with_h, without = Engine(hints=hints), Engine()
    feed(with_h, envs[:10])
    feed(without, envs[:10])
    assert by(with_h.score(envs[10]), "magnitude", "foo").evidence["direction"] == "up"
    assert by(without.score(envs[10]), "magnitude", "foo").evidence["direction"] == "both"


def test_priors_fed_on_update_only_for_typed_values() -> None:
    eng = Engine()
    feed(eng, amounts(300, agent="veteran"))
    # Types become known at the 10th observation, so the first 10 updates feed no prior.
    rec_m = eng.store.get(key_prior_mag(TOOL, "amount"))
    rec_n = eng.store.get(key_prior_nov(TOOL, "currency"))
    assert rec_m is not None and rec_m.n == 290  # type: ignore[attr-defined]
    assert rec_n is not None and rec_n.n_total == 290  # type: ignore[attr-defined]


def test_priors_off_stores_no_prior() -> None:
    eng = Engine(EngineConfig(priors_mode="off"))
    feed(eng, amounts(100, agent="veteran"))
    assert eng.store.get(key_prior_mag(TOOL, "amount")) is None
    assert eng.store.get(key_prior_nov(TOOL, "currency")) is None


def _rookie_spike(cfg: EngineConfig) -> DetectorResult:
    eng = Engine(cfg)
    feed(eng, amounts(300, agent="veteran"))
    feed(eng, amounts(15, seed=1, agent="rookie", start=300))  # typed, but own key still cold
    return eng.score(env({"amount": 37.5 * 50, "currency": "usd"}, 315, agent="rookie"))


def test_new_agent_benefits_from_other_agents_history() -> None:
    with_p = _rookie_spike(EngineConfig())
    without = _rookie_spike(EngineConfig(priors_mode="off"))
    s_with, s_without = by(with_p, "magnitude", "amount"), by(without, "magnitude", "amount")
    assert s_with.evidence["prior_used"] is True
    assert s_with.score > 0.9
    assert s_without.evidence.get("cold") is True and s_without.score == 0.0
    assert with_p.fused_score > without.fused_score


def test_response_count_prior_fed_from_prev_response() -> None:
    """``__resp__.item_count`` priors are fed so magnitude's response signal works cold."""
    eng = Engine()
    tr = SessionTracker()
    for i in range(40):
        sid = f"v{i}"
        e1 = env(
            {"q": "open"},
            2 * i,
            agent="veteran",
            tool=LIST,
            session=sid,
            tracker=tr,
            response_meta=ResponseMeta(item_count=5 + i % 3),
        )
        e2 = env(None, 2 * i + 1, agent="veteran", session=sid, tracker=tr)
        feed(eng, [e1, e2])
    rec = eng.store.get(key_prior_mag(LIST, RESP_FIELD))
    assert rec is not None and rec.n == 40  # type: ignore[attr-defined]
    e1 = env(
        {"q": "open"},
        100,
        agent="rookie",
        tool=LIST,
        session="r",
        tracker=tr,
        response_meta=ResponseMeta(item_count=5000),
    )
    feed(eng, [e1])
    r = eng.score(env(None, 101, agent="rookie", session="r", tracker=tr))
    s = by(r, "magnitude", RESP_FIELD)
    assert s.evidence["prior_used"] is True and s.score > 0.9


# -- registry ---------------------------------------------------------------------------------


ENGINE_NAMES = [
    "engine",
    *(f"engine-no-{d}" for d in sorted(ALL_DETECTORS)),
    *(f"engine-only-{d}" for d in sorted(ALL_DETECTORS)),
    "engine-no-priors",
]


@pytest.mark.parametrize("name", ENGINE_NAMES)
def test_registry_engine_names(name: str) -> None:
    assert name in DETECTORS
    det = make_detector(name)
    assert isinstance(det, Engine)
    if name.startswith("engine-no-") and name != "engine-no-priors":
        assert det.enabled == ALL_DETECTORS - {name.removeprefix("engine-no-")}
    elif name.startswith("engine-only-"):
        assert det.enabled == frozenset({name.removeprefix("engine-only-")})
    else:
        assert det.enabled == ALL_DETECTORS
    assert det.cfg.priors_mode == ("off" if name == "engine-no-priors" else "all")


def test_registry_engine_accepts_kwargs() -> None:
    det = make_detector("engine-no-priors", cfg=EngineConfig(z0=2.0))
    assert isinstance(det, Engine)
    assert det.cfg.z0 == 2.0 and det.cfg.priors_mode == "off"


# -- CI dataset: beats static threshold, golden hash -----------------------------------------


@cache
def _ci_rows(name: str) -> list[ScoreRow]:
    return golden_rows(name)


def test_engine_beats_static_threshold_on_ci_dataset() -> None:
    eng = compute_all(_ci_rows("engine"), warm_only=False, n_boot=0).auprc
    static = compute_all(_ci_rows("static_threshold"), warm_only=False, n_boot=0).auprc
    assert eng > static


def test_golden_engine() -> None:
    golden = (ROOT / "tests/golden/engine.sha256").read_text().strip()
    assert canonical_hash(_ci_rows("engine")) == golden


def test_golden_engine_spec_allow() -> None:
    """The spec engine (``SPEC_SWITCHES``) under spec §5.2 allow-all: the pre-U1 engine golden."""
    golden = (ROOT / "tests/golden/engine-spec-allow.sha256").read_text().strip()
    assert golden == "c4042bc466f20af46d092a86e77f58d38d2ebf0caa976338f0b8da54ef3a0abd"
    assert canonical_hash(_ci_rows("engine-spec-allow")) == golden


def test_engine_replay_strict_mode_no_leak() -> None:
    raws, labels = ci_dataset()
    rows = replay(raws[:300], labels, Engine(), SplitSpec(), strict=True)
    assert len(rows) == 300


# --- docs/decisions/003-prereq-threshold.md: the default mines a 92 %-rate prerequisite --------


def _prereq_engine(cfg: EngineConfig) -> Engine:
    """100 sessions of ``Y`` (lookup ``X`` before it in 92 of them, ``Z`` otherwise)."""
    eng = Engine(cfg)
    for i in range(100):
        tr = SessionTracker()
        first = "zendesk.get_ticket" if i % 25 not in (1, 2) else "slack.post_message"
        for j, tool in enumerate((first, "stripe.create_refund")):
            e = make_env(
                tracker=tr, session_id=f"p{i}", agent_id="a1", tool_id=tool, seq_no=j,
                ts_unix_ms=1_700_000_000_000 + (2 * i + j) * 60_000, args={},
            )  # fmt: skip
            eng.score(e)
            eng.update(e, Outcome.ALLOWED)
            tr.observe(e)
    return eng


def _skip(eng: Engine) -> object:
    e = make_env(
        session_id="skip", agent_id="a1", tool_id="stripe.create_refund",
        ts_unix_ms=1_700_000_000_000 + 300 * 60_000, args={},
    )  # fmt: skip
    return by(eng.score(e), "sequence").evidence["prereq_missing"]


def test_default_engine_mines_a_92_percent_prerequisite_that_095_would_not() -> None:
    assert _skip(_prereq_engine(EngineConfig())) == "zendesk.get_ticket"
    assert _skip(_prereq_engine(EngineConfig().replace(prereq_rate=0.95))) is None


# -- Task 1.15: calibrator censoring (opt-in, docs/decisions/008-tier1-gate.md) --------------------


def _cal_spy(eng: Engine, monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, float]]:
    """Record every value the calibrator receives: ``("obs", fused)`` / ``("cens", theta)``."""
    fed: list[tuple[str, float]] = []
    observe, censored = eng.calibrator.observe, eng.calibrator.observe_censored

    def spy_obs(agent: str, fused: float, outcome: Outcome) -> None:
        fed.append(("obs", fused))
        observe(agent, fused, outcome)

    def spy_cens(agent: str, theta: float) -> None:
        fed.append(("cens", theta))
        censored(agent, theta)

    monkeypatch.setattr(eng.calibrator, "observe", spy_obs)
    monkeypatch.setattr(eng.calibrator, "observe_censored", spy_cens)
    return fed


def _warm_cfg(cfg: EngineConfig) -> Engine:
    eng = Engine(cfg)
    feed(eng, amounts(250))
    return eng


def _n_scored(eng: Engine) -> int:
    rec = eng.store.get(key_cal("a1"))
    assert rec is not None
    return int(rec.n_scored)  # type: ignore[attr-defined]


@pytest.mark.parametrize("mode", ["none", "drop", "winsorize"])
def test_cal_censor_leaves_rows_below_threshold_alone(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    eng = _warm_cfg(EngineConfig().replace(cal_censor=mode))
    fed = _cal_spy(eng, monkeypatch)
    typical = env({"amount": 36.6, "currency": "usd"}, 251)
    r = eng.score(typical)
    assert r.threshold is not None and r.fused_score < r.threshold
    eng.update(typical, Outcome.ALLOWED)
    assert fed == [("obs", r.fused_score)]


def test_cal_censor_none_learns_rows_the_engine_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    eng = _warm_cfg(EngineConfig())
    fed = _cal_spy(eng, monkeypatch)
    r = eng.score(ODD)
    assert r.decision is Decision.BLOCK
    eng.update(ODD, Outcome.ALLOWED)
    assert fed == [("obs", r.fused_score)]
    eng.score(ODD)
    eng.update(ODD, Outcome.BLOCKED)  # spec: a blocked call reaches no baseline
    assert len(fed) == 1


def test_cal_censor_drop_skips_rows_the_engine_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    eng = _warm_cfg(EngineConfig().replace(cal_censor="drop"))
    fed = _cal_spy(eng, monkeypatch)
    n0, mag0 = _n_scored(eng), eng.store.get(key_mag("a1", TOOL, "amount"))
    n_mag0 = mag0.n  # type: ignore[union-attr]
    assert eng.score(ODD).decision is Decision.BLOCK
    eng.update(ODD, Outcome.ALLOWED)  # allowed anyway (allow-all replay): detectors still learn
    assert fed == [] and _n_scored(eng) == n0
    assert eng.store.get(key_mag("a1", TOOL, "amount")).n == n_mag0 + 1  # type: ignore[union-attr]


def test_cal_censor_winsorize_feeds_the_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    eng = _warm_cfg(EngineConfig().replace(cal_censor="winsorize"))
    fed = _cal_spy(eng, monkeypatch)
    r = eng.score(ODD)
    assert r.threshold is not None and r.fused_score > r.threshold
    eng.update(ODD, Outcome.ALLOWED)
    assert fed == [("obs", r.threshold)]  # the tail mass is kept, its value is not


def test_cal_censor_winsorize_counts_blocked_mass_only_in_the_calibrator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # vel_count_attempts pinned off: only the calibrator may see the blocked call here.
    cfg = EngineConfig().replace(cal_censor="winsorize", vel_count_attempts=False)
    eng = _warm_cfg(cfg)
    ref = _warm_cfg(cfg)
    fed = _cal_spy(eng, monkeypatch)
    n0 = _n_scored(eng)
    r = eng.score(ODD)
    eng.update(ODD, Outcome.BLOCKED)
    assert fed == [("cens", r.threshold)] and _n_scored(eng) == n0 + 1
    eng.update(ODD, Outcome.DENIED)  # a human denial is not the engine's block: nothing
    assert len(fed) == 1

    # every non-calibrator record is untouched by the blocked call
    def non_cal(e: Engine, name: str) -> list[bytes]:
        lines = dump(e, tmp_path / name).splitlines()
        return [ln for ln in lines if f'"key":"{key_cal("a1")}"'.encode() not in ln]

    assert non_cal(eng, "a.jsonl") == non_cal(ref, "b.jsonl")
    assert dump(eng, tmp_path / "a.jsonl") != dump(ref, tmp_path / "b.jsonl")


def test_cal_censor_uses_the_threshold_that_decided(monkeypatch: pytest.MonkeyPatch) -> None:
    """Interleaved: another envelope's update moves the threshold between score and update."""
    eng = _warm_cfg(EngineConfig().replace(cal_censor="winsorize"))
    r = eng.score(ODD)
    for i in range(30):  # other envelopes move the calibrator between ODD's score and update
        eng.update(env({"amount": 36.6, "currency": "usd"}, 260 + i), Outcome.ALLOWED)
    assert eng.calibrator.threshold("a1") != r.threshold
    fed = _cal_spy(eng, monkeypatch)
    eng.update(ODD, Outcome.ALLOWED)
    assert fed == [("obs", r.threshold)]


def test_unknown_cal_censor_rejected() -> None:
    with pytest.raises(ValueError, match="cal_censor"):
        Engine(EngineConfig().replace(cal_censor="clip"))


def test_cal_censor_starts_only_once_the_calibrator_has_min_n_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shadow-then-censor: below ``cal_censor_min_n`` scores the calibrator learns like the spec."""
    cfg = EngineConfig().replace(cal_censor="winsorize", cal_censor_min_n=300)
    eng = _warm_cfg(cfg)  # 250 scores so far
    fed = _cal_spy(eng, monkeypatch)
    r = eng.score(ODD)
    assert r.decision is Decision.BLOCK
    eng.update(ODD, Outcome.ALLOWED)
    assert fed == [("obs", r.fused_score)]
    eng.score(ODD)
    eng.update(ODD, Outcome.BLOCKED)  # spec behaviour while in shadow: nothing learned
    assert len(fed) == 1
    for i in range(60):
        eng.update(env({"amount": 36.6, "currency": "usd"}, 300 + i), Outcome.ALLOWED)
    assert _n_scored(eng) >= 300
    fed.clear()
    r = eng.score(ODD)
    eng.update(ODD, Outcome.ALLOWED)
    assert fed == [("obs", r.threshold)]


@pytest.mark.parametrize("outcome", [Outcome.BLOCKED, Outcome.DENIED])
def test_vel_count_attempts_counts_refused_calls_in_the_window(outcome: Outcome) -> None:
    def window(eng: Engine, e: ActionEnvelope) -> int:
        (vel,) = [s for s in eng.score(e).per_detector if s.detector == "velocity"]
        v = vel.evidence["window_count"]
        assert isinstance(v, int)
        return v

    on = _warm_cfg(EngineConfig().replace(vel_count_attempts=True))
    off = _warm_cfg(EngineConfig().replace(vel_count_attempts=False))
    for eng in (on, off):
        for k in range(5):
            eng.update(env({"amount": 1e6, "currency": "usd"}, 250, session=f"r{k}"), outcome)
    probe = env({"amount": 36.6, "currency": "usd"}, 250, session="p")
    assert window(on, probe) == window(off, probe) + 5


def test_cal_censor_winsorize_ignores_blocked_calls_below_the_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A call blocked by something other than the engine (fused < theta, e.g. policy) is not
    tail mass: winsorize feeds nothing, with or without a cached score."""
    eng = _warm_cfg(EngineConfig().replace(cal_censor="winsorize"))
    fed = _cal_spy(eng, monkeypatch)
    typical = env({"amount": 36.6, "currency": "usd"}, 251)
    r = eng.score(typical)
    assert r.threshold is not None and r.fused_score < r.threshold
    eng.update(typical, Outcome.BLOCKED)  # cached score
    eng.update(typical, Outcome.BLOCKED)  # no cache: scored fresh (pure) before deciding
    assert fed == []
    theta = eng.calibrator.threshold("a1")
    eng.update(ODD, Outcome.BLOCKED)  # no cache, but the engine would block it: censored
    assert fed == [("cens", theta)]


def test_small_count_field_typed_bool_early_still_scores_a_huge_value() -> None:
    """FR I3 probe: ``qty`` mostly 1 (the first 10 all 1), sometimes 2-4, then ``qty=10000``."""
    rng = np.random.default_rng(0)
    vals = [1] * 10 + [int(x) for x in np.where(rng.random(490) < 0.8, 1, rng.integers(2, 5, 490))]
    eng, tr = Engine(), SessionTracker()
    for i, v in enumerate(vals):
        e = make_env(
            tracker=tr,
            envelope_id=f"e{i:06d}",
            session_id=f"s{i}",
            agent_id="a",
            principal_id="p",
            seq_no=0,
            ts_unix_ms=T0 + i * 10 * MIN,
            tool_id="shop.order",
            args={"qty": v},
        )
        eng.score(e)
        eng.update(e, Outcome.ALLOWED)
        tr.observe(e)
    atk = make_env(
        tracker=tr,
        envelope_id="atk",
        session_id="sx",
        agent_id="a",
        principal_id="p",
        seq_no=0,
        ts_unix_ms=T0 + len(vals) * 10 * MIN,
        tool_id="shop.order",
        args={"qty": 10000},
    )
    res = eng.score(atk)
    assert res.field_types["qty"] is FieldType.NUMERIC
    assert res.fused_score > 0.9


@pytest.mark.parametrize("kind", ["always_1", "flag_0_1"])
def test_count_field_with_only_0_1_history_scores_a_huge_value(kind: str) -> None:
    """FR2 N1 probes (pre-amendment behaviour, ``mag_bool_numeric=False``): ``qty`` always 1 (or a
    fair 0/1 flag) for 500 calls, then ``qty=10000``.

    Agent-independent level: novelty scores an unseen value at
    ``(1 - n_singletons/n_total) * min(1, n_total/novelty_full_n)``, which is exactly 1.0 here
    (no singletons, n_total >= 50); fusion takes the top score, so fused is 1.0, above any
    calibrated threshold."""
    rng = np.random.default_rng(0)
    vals = [1] * 500 if kind == "always_1" else [int(x) for x in rng.integers(0, 2, 500)]
    eng, tr = Engine(PRE), SessionTracker()
    for i, v in enumerate(vals):
        e = make_env(
            tracker=tr,
            envelope_id=f"e{i:06d}",
            session_id=f"s{i}",
            agent_id="a",
            principal_id="p",
            seq_no=0,
            ts_unix_ms=T0 + i * 10 * MIN,
            tool_id="shop.order",
            args={"qty": v},
        )
        eng.score(e)
        eng.update(e, Outcome.ALLOWED)
        tr.observe(e)
    atk = make_env(
        tracker=tr,
        envelope_id="atk",
        session_id="sx",
        agent_id="a",
        principal_id="p",
        seq_no=0,
        ts_unix_ms=T0 + len(vals) * 10 * MIN,
        tool_id="shop.order",
        args={"qty": 10000},
    )
    dumped = orjson.dumps(eng.store.get(key_nov("a", "shop.order", "qty")).to_dict())  # type: ignore[union-attr]
    res = eng.score(atk)
    assert res.field_types["qty"] is FieldType.BOOL  # scored before the update retypes it
    nov = by(res, "novelty", "qty")
    assert nov.score == pytest.approx(1.0) and nov.evidence["type_violation"] == "bool"
    assert res.fused_score == pytest.approx(1.0)
    assert orjson.dumps(eng.store.get(key_nov("a", "shop.order", "qty")).to_dict()) == dumped  # type: ignore[union-attr]
    # a genuine True/False value in the same field still scores low
    ok = make_env(
        tracker=tr,
        envelope_id="ok",
        session_id="sy",
        agent_id="a",
        principal_id="p",
        seq_no=0,
        ts_unix_ms=T0 + (len(vals) + 1) * 10 * MIN,
        tool_id="shop.order",
        args={"qty": True},
    )
    assert eng.score(ok).fused_score < 0.1


# -- BOOL-typed fields that carry numbers: pre-amendment pins (FR3) and Amendment A1.3 ------------


def _qty_engine(
    history: list[Any],
    cfg: EngineConfig = EngineConfig(),  # noqa: B008 - frozen dataclass, safe as a default
    field: str = "qty",
) -> tuple[Engine, SessionTracker]:
    eng = Engine(cfg)
    return eng, _qty_feed(eng, history, field=field)


def _qty_feed(
    eng: Engine, history: list[Any], agent: str = "a", start: int = 0, field: str = "qty"
) -> SessionTracker:
    tr = SessionTracker()
    for i, v in enumerate(history, start):
        e = _qty_env(tr, i, v, agent, field)
        eng.score(e)
        eng.update(e, Outcome.ALLOWED)
        tr.observe(e)
    return tr


def _qty_env(
    tr: SessionTracker, i: int, v: Any, agent: str = "a", field: str = "qty"
) -> ActionEnvelope:
    return make_env(
        tracker=tr,
        envelope_id=f"e{i:06d}",
        session_id=f"s{i}",
        agent_id=agent,
        principal_id="p",
        seq_no=0,
        ts_unix_ms=T0 + i * 10 * MIN,
        tool_id="shop.order",
        args={field: v},
    )


def _qty_scores(res: DetectorResult) -> list[DetectorScore]:
    return [d for d in res.per_detector if d.field == "qty"]


def test_pinned_spike_right_after_a_learned_bool_to_numeric_switch_scores_low_pre_a1() -> None:
    """PINS the pre-amendment behaviour, ``mag_bool_numeric=False``. After ``[1]*500`` and a
    learned (ALLOWED) ``2`` the field is NUMERIC with no magnitude baseline (magnitude never
    pushed the BOOL-typed history), so ``10000`` is in magnitude cold start for ``mag_cold_n``
    observations and scores low."""
    eng, tr = _qty_engine([1] * 500 + [2], PRE)
    res = eng.score(_qty_env(tr, 501, 10000))
    assert res.field_types["qty"] is FieldType.NUMERIC
    mag = by(res, "magnitude", "qty")
    assert mag.score == 0.0 and mag.evidence["cold"] is True
    assert res.fused_score < 0.1


def test_pinned_first_non_bool_number_scores_one_and_persists_while_blocked_pre_a1() -> None:
    """PINS the pre-amendment behaviour, ``mag_bool_numeric=False``. After ``[1]*60`` the first
    ``2`` scores 1.0 by novelty (``type_violation == "bool"``) whatever its magnitude, and stays
    1.0 on the next ``2`` when the first was BLOCKED (a blocked call never retypes the field)."""
    eng, tr = _qty_engine([1] * 60, PRE)
    first = _qty_env(tr, 60, 2)
    res = eng.score(first)
    assert res.field_types["qty"] is FieldType.BOOL
    nov = by(res, "novelty", "qty")
    assert nov.score == pytest.approx(1.0) and nov.evidence["type_violation"] == "bool"
    assert res.fused_score == pytest.approx(1.0)
    assert not [d for d in res.per_detector if d.detector == "magnitude" and d.field == "qty"]
    eng.update(first, Outcome.BLOCKED)
    again = eng.score(_qty_env(tr, 61, 2))
    assert again.field_types["qty"] is FieldType.BOOL
    assert by(again, "novelty", "qty").score == pytest.approx(1.0)
    assert again.fused_score == pytest.approx(1.0)
    assert eng.store.get(key_mag("a", "shop.order", "qty")) is None  # BOOL history: never pushed


def test_default_is_a1() -> None:
    """A1.9 (Task 1.21): magnitude scores BOOL-typed numbers by default; False stays selectable."""
    assert EngineConfig().mag_bool_numeric is True


def test_bool_to_numeric_retype_does_not_start_cold() -> None:
    """A1.3: magnitude pushed the BOOL-typed ``1``s, so the learned retype starts warm."""
    eng, tr = _qty_engine([1] * 500 + [2], A1)
    res = eng.score(_qty_env(tr, 501, 10000))
    assert res.field_types["qty"] is FieldType.NUMERIC
    mag = by(res, "magnitude", "qty")
    # 500 - type_min_obs (the field is UNKNOWN until then) + the learned 2
    assert "cold" not in mag.evidence and mag.evidence["n"] == 491
    assert "type_violation" not in mag.evidence  # NUMERIC now: the ordinary magnitude score
    assert mag.score > 0.999 and res.fused_score > 0.999


def test_first_non_bool_number_is_scored_by_magnitude_and_persists_while_blocked() -> None:
    eng, tr = _qty_engine([1] * 60, A1)
    first = _qty_env(tr, 60, 2)
    res = eng.score(first)
    assert res.field_types["qty"] is FieldType.BOOL
    mag = by(res, "magnitude", "qty")
    assert mag.evidence["type_violation"] == "bool" and mag.evidence["n"] == 50
    # 8.109: the field never varied, so the scale is the mag_eps floor
    assert mag.evidence["z"] == pytest.approx((math.log(3) - math.log(2)) / 0.05)
    assert mag.score > 0.999
    assert not [d for d in res.per_detector if d.detector == "novelty" and d.field == "qty"]
    eng.update(first, Outcome.BLOCKED)  # a blocked call learns nothing
    again = eng.score(_qty_env(tr, 61, 2))
    assert again.field_types["qty"] is FieldType.BOOL
    assert by(again, "magnitude", "qty").score == mag.score
    assert by(again, "magnitude", "qty").evidence["n"] == 50


def test_contradicting_value_is_scored_by_its_size() -> None:
    eng, tr = _qty_engine([0, 1] * 40, A1)  # the field varies: 0 and 1
    s = {v: by(eng.score(_qty_env(tr, 80, v)), "magnitude", "qty").score for v in (2, 5, 10, 10000)}
    # 0.048, 0.335, 0.893, 1.0
    assert s[2] < 0.1 and 0.2 < s[5] < 0.5 and 0.8 < s[10] < 0.95 and s[10000] > 0.999
    assert eng.score(_qty_env(tr, 80, 2)).fused_score < 0.1  # a benign 1 -> 2 is no longer 1.0
    for ok in (0, 1, True):  # boolean-like values: no magnitude score, no novelty score
        assert not _qty_scores(eng.score(_qty_env(tr, 80, ok)))


def test_bool_path_with_several_values_scores_only_the_contradicting_ones() -> None:
    """A1.3: the BOOL filter is per value. In a multi-valued BOOL-typed path magnitude scores the
    values that contradict BOOL and drops the boolean-like ones beside them: after ``[1, 1, 1]``
    x40 the ``0`` of ``[0, 1.0001]`` is an ordinary boolean, although its size alone would score
    1.0 against the constant-1 record of this both-direction field."""
    eng, tr = _qty_engine([[1, 1, 1]] * 40, A1, field="flags")
    res = eng.score(_qty_env(tr, 40, [0, 1.0001], field="flags"))
    assert res.field_types["flags[]"] is FieldType.BOOL
    mag = by(res, "magnitude", "flags[]")
    assert mag.evidence["direction"] == "both" and mag.evidence["n"] == 108  # 3 * (40 - 4)
    assert mag.evidence["observed"] == 1.0001 and mag.evidence["type_violation"] == "bool"
    assert mag.score < 0.1  # 0.005: a hair above the constant 1
    assert not [d for d in res.per_detector if d.detector == "novelty" and d.field == "flags[]"]
    assert res.fused_score < 0.1
    res = eng.score(_qty_env(tr, 40, [0, 1, 1], field="flags"))  # boolean-like values only
    assert res.field_types["flags[]"] is FieldType.BOOL
    assert not [d for d in res.per_detector if d.field == "flags[]"]


def test_pinned_contradiction_not_above_the_learned_range_scores_low() -> None:
    """PINS a remaining limit of A1.3 (``mag_bool_numeric=True``). Once magnitude has a reference
    it scores a contradicting value by its size alone, with the field's direction (§6.3), and
    novelty is silent. So a number outside {0, 1} that is not above the learned range scores about
    0: a negative number in an ``up`` field (``qty``: one-sided, z = 0), and a fraction between 0
    and 1 in any field. Before the amendment novelty scored each of them 1.0."""
    eng, tr = _qty_engine([0, 1] * 40, A1)
    for v in (-1, -5, -10000):
        res = eng.score(_qty_env(tr, 80, v))
        assert res.field_types["qty"] is FieldType.BOOL
        mag = by(res, "magnitude", "qty")
        assert mag.evidence["direction"] == "up" and mag.evidence["type_violation"] == "bool"
        assert mag.evidence["observed"] == v and mag.evidence["z"] == 0.0
        assert mag.score < 0.05  # 0.010: posterior(p = 0.5, tail_prior)
        assert not [d for d in res.per_detector if d.detector == "novelty" and d.field == "qty"]
        assert res.fused_score < 0.1
    half = eng.score(_qty_env(tr, 80, 0.5))
    assert by(half, "magnitude", "qty").score < 0.05  # 0.012
    assert not [d for d in half.per_detector if d.detector == "novelty" and d.field == "qty"]
    # a both-direction field (``flag``): size counts on either side, a fraction is still inside
    eng, tr = _qty_engine([0, 1] * 40, A1, field="flag")
    s: dict[float, float] = {}
    for x in (-10000, -1, 0.5):
        mag = by(eng.score(_qty_env(tr, 80, x, field="flag")), "magnitude", "flag")
        assert mag.evidence["direction"] == "both" and mag.evidence["type_violation"] == "bool"
        s[x] = mag.score
    assert s[-10000] > 0.999 and 0.2 < s[-1] < 0.5 and s[0.5] < 0.05  # 1.0, 0.392, 0.006
    # pre-amendment behaviour, ``mag_bool_numeric=False``: novelty scores each of them 1.0
    pre, tr = _qty_engine([0, 1] * 40, PRE)
    for x in (-1, -5, -10000, 0.5):
        res = pre.score(_qty_env(tr, 80, x))
        assert by(res, "novelty", "qty").score == pytest.approx(1.0)
        assert res.fused_score == pytest.approx(1.0)


def test_non_finite_in_a_bool_field_scores_one_under_a1() -> None:
    """Ruling A1-13 (``mag_bool_numeric=True``): an infinite number is a number outside {0, 1},
    so in a BOOL-typed field ``inf``, ``-inf`` and ``"1e999"`` (JSON 1e999) score 1.0 by
    magnitude through the non-finite handling (``non_finite``, ``type_violation == "bool"``), as
    in a NUMERIC-typed field, and are never pushed: an ALLOWED one can not be learned away. NaN is
    still skipped (no score, no update)."""
    eng, tr = _qty_engine([0, 1] * 40, A1)
    key = key_mag("a", "shop.order", "qty")
    rec = eng.store.get(key)
    assert rec is not None
    before = orjson.dumps(rec.to_dict())
    for i, v in enumerate((math.inf, -math.inf, "1e999"), 80):
        e = _qty_env(tr, i, v)
        res = eng.score(e)
        assert res.field_types["qty"] is FieldType.BOOL
        mag = by(res, "magnitude", "qty")
        assert mag.score == 1.0 and mag.evidence["non_finite"]
        assert mag.evidence["type_violation"] == "bool"
        assert res.fused_score > 0.999
        eng.update(e, Outcome.ALLOWED)  # learned, yet never pushed
        tr.observe(e)
        rec = eng.store.get(key)
        assert rec is not None and orjson.dumps(rec.to_dict()) == before
        again = eng.score(_qty_env(tr, i, math.inf))
        assert again.field_types["qty"] is FieldType.BOOL
        assert by(again, "magnitude", "qty").score == 1.0
    e = _qty_env(tr, 83, math.nan)
    res = eng.score(e)
    assert res.field_types["qty"] is FieldType.BOOL
    assert not _qty_scores(res) and res.fused_score < 0.1
    eng.update(e, Outcome.ALLOWED)
    rec = eng.store.get(key)
    assert rec is not None and orjson.dumps(rec.to_dict()) == before
    # in a list the path score is the maximum over its values: the infinite one scores 1.0
    eng, tr = _qty_engine([[1, 1, 1]] * 40, A1, field="flags")
    res = eng.score(_qty_env(tr, 40, [math.inf, 2], field="flags"))
    assert res.field_types["flags[]"] is FieldType.BOOL
    mag = by(res, "magnitude", "flags[]")
    assert mag.score == 1.0 and mag.evidence["non_finite"]
    assert mag.evidence["type_violation"] == "bool"


@pytest.mark.parametrize(
    ("history", "n"),
    [([True] * 60, 0), ([1] * 25, 15)],  # no magnitude record; 15 < mag_cold_n numbers, no prior
    ids=["no_record", "cold"],
)
def test_non_finite_in_a_bool_field_scores_one_without_a_usable_reference_under_a1(
    history: list[Any], n: int
) -> None:
    """Ruling A1-13 holds where magnitude has no usable reference: the non-finite check runs
    before any cold or prior logic, so ``inf`` still scores exactly 1.0 (novelty never scores it,
    ``contradicts_bool(inf)`` is False) and an ALLOWED one is never pushed."""
    eng, tr = _qty_engine(history, A1)
    key = key_mag("a", "shop.order", "qty")
    rec = eng.store.get(key)
    before = None if rec is None else orjson.dumps(rec.to_dict())
    assert (before is None) == (n == 0)
    e = _qty_env(tr, len(history), math.inf)
    res = eng.score(e)
    assert res.field_types["qty"] is FieldType.BOOL
    (mag,) = _qty_scores(res)
    assert mag.detector == "magnitude" and mag.score == 1.0
    assert mag.evidence["non_finite"] and mag.evidence["type_violation"] == "bool"
    assert mag.evidence["n"] == n and mag.evidence["prior_used"] is False
    assert res.fused_score > 0.999
    eng.update(e, Outcome.ALLOWED)
    rec = eng.store.get(key)
    assert (None if rec is None else orjson.dumps(rec.to_dict())) == before


def test_pinned_non_finite_float_in_a_bool_field_is_scored_by_no_detector_pre_a1() -> None:
    """PINS the pre-amendment behaviour, ``mag_bool_numeric=False``. ``contradicts_bool`` wants a
    finite value from a float or a string, so ``inf``, ``-inf``, ``"1e999"`` (JSON 1e999) and NaN
    in a BOOL-typed field contradict nothing: novelty drops them, magnitude skips BOOL-typed
    paths, and an ALLOWED one neither retypes the field nor is pushed."""
    eng, tr = _qty_engine([0, 1] * 40, PRE)
    key = key_mag("a", "shop.order", "qty")
    assert eng.store.get(key) is None
    for i, v in enumerate((math.inf, -math.inf, "1e999", math.nan), 80):
        e = _qty_env(tr, i, v)
        res = eng.score(e)
        assert res.field_types["qty"] is FieldType.BOOL
        assert not _qty_scores(res) and res.fused_score < 0.1
        eng.update(e, Outcome.ALLOWED)  # learned, yet nothing to learn from it
        tr.observe(e)
    assert eng.store.get(key) is None
    assert eng.score(_qty_env(tr, 84, 1)).field_types["qty"] is FieldType.BOOL
    # in a list the non-finite value is dropped and the path is scored by novelty on its finite
    # contradiction alone, never 1.0 through ``non_finite``
    eng, tr = _qty_engine([[1, 1, 1]] * 40, PRE, field="flags")
    res = eng.score(_qty_env(tr, 40, [math.inf, 2], field="flags"))
    (ds,) = [d for d in res.per_detector if d.field == "flags[]"]
    assert ds.detector == "novelty"
    assert ds.evidence["type_violation"] == "bool" and "non_finite" not in ds.evidence


def test_bool_field_without_numeric_reference_falls_back_to_novelty() -> None:
    eng, tr = _qty_engine([True] * 60, A1)  # Python bools are never pushed
    res = eng.score(_qty_env(tr, 60, 2))
    assert res.field_types["qty"] is FieldType.BOOL
    assert eng.store.get(key_mag("a", "shop.order", "qty")) is None
    nov = by(res, "novelty", "qty")
    assert nov.score == pytest.approx(1.0) and nov.evidence["type_violation"] == "bool"
    assert res.fused_score == pytest.approx(1.0)
    eng2, tr2 = _qty_engine([1] * 25, A1)  # 15 numbers learned: magnitude cold
    res2 = eng2.score(_qty_env(tr2, 25, 2))
    mag2 = by(res2, "magnitude", "qty")
    assert mag2.score == 0.0 and mag2.evidence["cold"] is True and mag2.evidence["n"] == 15
    assert mag2.evidence["type_violation"] == "bool"
    assert by(res2, "novelty", "qty").score == pytest.approx(0.3)  # (1 - 0) * 15 / novelty_full_n


def test_novelty_fallback_ends_exactly_when_magnitude_is_warm() -> None:
    """The fallback boundary is magnitude's own ``mag_cold_n``: 29 learned numbers leave novelty
    scoring the contradiction, the 30th hands the field to magnitude."""
    eng, tr = _qty_engine([1] * 39, A1)  # 39 - type_min_obs = 29 numbers pushed
    res = eng.score(_qty_env(tr, 39, 2))
    assert by(res, "magnitude", "qty").evidence["cold"] is True
    assert by(res, "novelty", "qty").score == pytest.approx(29 / 50)
    eng, tr = _qty_engine([1] * 40, A1)  # 30 numbers pushed: warm
    res = eng.score(_qty_env(tr, 40, 2))
    assert by(res, "magnitude", "qty").evidence["n"] == 30
    assert "cold" not in by(res, "magnitude", "qty").evidence
    assert not [d for d in res.per_detector if d.detector == "novelty" and d.field == "qty"]


def test_engine_without_magnitude_keeps_the_novelty_score() -> None:
    """The ``engine-no-magnitude`` ablation: nothing scores the value by size, so novelty keeps
    its FR2 N1 score (and no magnitude record is written)."""
    eng = Engine(A1, enabled=ALL_DETECTORS - {"magnitude"})
    tr = _qty_feed(eng, [1] * 60)
    res = eng.score(_qty_env(tr, 60, 2))
    assert res.field_types["qty"] is FieldType.BOOL
    nov = by(res, "novelty", "qty")
    assert nov.score == pytest.approx(1.0) and nov.evidence["type_violation"] == "bool"
    assert not [d for d in res.per_detector if d.detector == "magnitude"]
    assert eng.store.get(key_mag("a", "shop.order", "qty")) is None
    # ... also on a store where a full engine left a warm magnitude record for the field: the
    # ablation has no magnitude to score the value, so novelty must not defer to that record
    full, tr = _qty_engine([1] * 60, A1)
    rec = full.store.get(key_mag("a", "shop.order", "qty"))
    assert rec is not None and rec.to_dict()["n"] == 50
    ablated = Engine(A1, store=full.store, enabled=ALL_DETECTORS - {"magnitude"})
    res = ablated.score(_qty_env(tr, 60, 2))
    assert by(res, "novelty", "qty").score == pytest.approx(1.0)
    assert not [d for d in res.per_detector if d.detector == "magnitude"]


def _pooled_qty_engine(cfg: EngineConfig) -> tuple[Engine, SessionTracker]:
    """Agent ``b`` uses ``qty`` as a count (a warm pooled prior for the field); agent ``a`` only
    ever sent ``True``, so its field is BOOL-typed and it has no magnitude record of its own."""
    eng = Engine(cfg)
    _qty_feed(eng, [3, 4, 5, 6, 7] * 12, agent="b", start=1000)
    prior = eng.store.get(key_prior_mag("shop.order", "qty"))
    assert prior is not None and prior.to_dict()["n"] == 50  # 60 - type_min_obs
    tr = _qty_feed(eng, [True] * 60)
    assert eng.store.get(key_mag("a", "shop.order", "qty")) is None
    return eng, tr


def test_bool_field_uses_a_warm_pooled_prior_as_reference() -> None:
    """A1.3: a warm pooled prior is a usable reference too. Magnitude scores the contradicting
    value against it (by size, ``prior_used``) and novelty stays silent."""
    eng, tr = _pooled_qty_engine(A1)
    low = eng.score(_qty_env(tr, 60, 2))
    assert low.field_types["qty"] is FieldType.BOOL
    mag = by(low, "magnitude", "qty")
    assert mag.evidence["prior_used"] is True and mag.evidence["type_violation"] == "bool"
    assert mag.score < 0.1  # 2 is below the pool's median: nothing unusual about its size
    assert not [d for d in low.per_detector if d.detector == "novelty" and d.field == "qty"]
    high = eng.score(_qty_env(tr, 60, 10000))
    assert by(high, "magnitude", "qty").score > 0.999
    assert not [d for d in high.per_detector if d.detector == "novelty" and d.field == "qty"]


def test_pre_a1_ignores_a_warm_pooled_prior_in_a_bool_field() -> None:
    """Pre-amendment behaviour, ``mag_bool_numeric=False``: magnitude skips the BOOL-typed field
    even when the pool could score it, so novelty must keep its FR2 N1 score."""
    eng, tr = _pooled_qty_engine(PRE)
    res = eng.score(_qty_env(tr, 60, 2))
    assert res.field_types["qty"] is FieldType.BOOL
    nov = by(res, "novelty", "qty")
    assert nov.score == pytest.approx(1.0) and nov.evidence["type_violation"] == "bool"
    assert not [d for d in res.per_detector if d.detector == "magnitude" and d.field == "qty"]


def test_bool_numeric_spec_score_also_scores_by_size() -> None:
    """``mag_bool_numeric`` is independent of ``mag_score``: with the spec §6.3 z rule the
    contradicting value goes through ``squash``."""
    cfg = EngineConfig().replace(mag_bool_numeric=True, mag_score="z")
    eng, tr = _qty_engine([1] * 60, cfg)
    mag = by(eng.score(_qty_env(tr, 60, 2)), "magnitude", "qty")
    z = (math.log(3) - math.log(2)) / 0.05
    assert mag.evidence["type_violation"] == "bool" and "p_tail" not in mag.evidence
    assert mag.score == pytest.approx(squash(z, cfg.z0, cfg.tau))
    assert mag.score == pytest.approx(0.9989, abs=1e-4)
    eng, tr = _qty_engine([0, 1] * 40, cfg)
    res = eng.score(_qty_env(tr, 80, 2))
    assert by(res, "magnitude", "qty").score == pytest.approx(0.147, abs=1e-3)
    assert not [d for d in res.per_detector if d.detector == "novelty" and d.field == "qty"]


def test_numeric_strings_in_a_bool_field_are_learned() -> None:
    eng, tr = _qty_engine(["1"] * 60, A1)
    res = eng.score(_qty_env(tr, 60, "2"))
    assert res.field_types["qty"] is FieldType.BOOL
    mag = by(res, "magnitude", "qty")
    assert mag.score > 0.999 and mag.evidence["n"] == 50
    assert mag.evidence["type_violation"] == "bool"
    eng, tr = _qty_engine(["true", "false"] * 30, A1)  # not numbers: never pushed
    assert eng.store.get(key_mag("a", "shop.order", "qty")) is None


def test_int_too_large_for_a_float_in_a_bool_field_scores_one() -> None:
    """Every value that contradicts BOOL gets a magnitude score once novelty defers: an int that
    saturates to ``inf`` is scored 1.0 (``non_finite``), never dropped by both detectors."""
    eng, tr = _qty_engine([1] * 60, A1)
    res = eng.score(_qty_env(tr, 60, 10**400))
    mag = by(res, "magnitude", "qty")
    assert mag.score == 1.0 and mag.evidence["non_finite"] is True
    assert mag.evidence["type_violation"] == "bool" and mag.evidence["n"] == 50
    assert res.fused_score == pytest.approx(1.0)


def test_bool_numeric_score_is_pure(tmp_path: Path) -> None:
    eng, tr = _qty_engine([0, 1] * 40, A1)
    before = dump(eng, tmp_path / "before.json")
    for v in (2, 10000, True, 1):
        eng.score(_qty_env(tr, 80, v))
    assert dump(eng, tmp_path / "after.json") == before


def test_bool_numeric_learns_only_on_allowed_or_approved() -> None:
    eng, tr = _qty_engine([1] * 60, A1)
    rec = eng.store.get(key_mag("a", "shop.order", "qty"))
    assert rec is not None
    before = orjson.dumps(rec.to_dict())
    one = _qty_env(tr, 60, 1)
    for outcome in (Outcome.BLOCKED, Outcome.DENIED):
        eng.score(one)
        eng.update(one, outcome)
    assert orjson.dumps(rec.to_dict()) == before
    eng.score(one)
    eng.update(one, Outcome.APPROVED)
    assert by(eng.score(_qty_env(tr, 61, 2)), "magnitude", "qty").evidence["n"] == 51


def test_bool_numeric_does_not_feed_pooled_priors() -> None:
    """Pooled priors (§6.10) still learn NUMERIC-typed paths only."""
    eng, _ = _qty_engine([1] * 60, A1)
    assert eng.store.get(key_mag("a", "shop.order", "qty")) is not None
    assert eng.store.get(key_prior_mag("shop.order", "qty")) is None
