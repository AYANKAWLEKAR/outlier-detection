from __future__ import annotations

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
    """FR2 N1 probes: ``qty`` always 1 (or a fair 0/1 flag) for 500 calls, then ``qty=10000``.

    Agent-independent level: novelty scores an unseen value at
    ``(1 - n_singletons/n_total) * min(1, n_total/novelty_full_n)``, which is exactly 1.0 here
    (no singletons, n_total >= 50); fusion takes the top score, so fused is 1.0, above any
    calibrated threshold."""
    rng = np.random.default_rng(0)
    vals = [1] * 500 if kind == "always_1" else [int(x) for x in rng.integers(0, 2, 500)]
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


# -- FR3: pinned BOOL-field limitations (no behaviour change; Amendment A1 is where they change) --


def _qty_engine(history: list[Any]) -> tuple[Engine, SessionTracker]:
    eng, tr = Engine(), SessionTracker()
    for i, v in enumerate(history):
        e = _qty_env(tr, i, v)
        eng.score(e)
        eng.update(e, Outcome.ALLOWED)
        tr.observe(e)
    return eng, tr


def _qty_env(tr: SessionTracker, i: int, v: Any) -> ActionEnvelope:
    return make_env(
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


def test_pinned_spike_right_after_a_learned_bool_to_numeric_switch_scores_low() -> None:
    """PINS today's behaviour; Amendment A1 is where it changes. After ``[1]*500`` and a learned
    (ALLOWED) ``2`` the field is NUMERIC with no magnitude baseline (magnitude never pushed the
    BOOL-typed history), so ``10000`` is in magnitude cold start for ``mag_cold_n`` observations
    and scores low."""
    eng, tr = _qty_engine([1] * 500 + [2])
    res = eng.score(_qty_env(tr, 501, 10000))
    assert res.field_types["qty"] is FieldType.NUMERIC
    mag = by(res, "magnitude", "qty")
    assert mag.score == 0.0 and mag.evidence["cold"] is True
    assert res.fused_score < 0.1


def test_pinned_first_non_bool_number_scores_one_and_persists_while_blocked() -> None:
    """PINS today's behaviour; Amendment A1 is where it changes. After ``[1]*60`` the first ``2``
    scores 1.0 by novelty (``type_violation == "bool"``) whatever its magnitude, and stays 1.0 on
    the next ``2`` when the first was BLOCKED (a blocked call never retypes the field)."""
    eng, tr = _qty_engine([1] * 60)
    first = _qty_env(tr, 60, 2)
    res = eng.score(first)
    assert res.field_types["qty"] is FieldType.BOOL
    nov = by(res, "novelty", "qty")
    assert nov.score == pytest.approx(1.0) and nov.evidence["type_violation"] == "bool"
    assert res.fused_score == pytest.approx(1.0)
    eng.update(first, Outcome.BLOCKED)
    again = eng.score(_qty_env(tr, 61, 2))
    assert again.field_types["qty"] is FieldType.BOOL
    assert by(again, "novelty", "qty").score == pytest.approx(1.0)
    assert again.fused_score == pytest.approx(1.0)
