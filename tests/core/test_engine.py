from __future__ import annotations

from collections.abc import Iterable
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from agentfw.bench.ci_regress import ci_dataset
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
from agentfw.core.state import DictStore, key_cal, key_mag, key_prior_mag, key_prior_nov
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
    eng = Engine()
    feed(eng, amounts(120))
    before = dump(eng, tmp_path / "before.jsonl")
    spike = env({"amount": 1e6, "currency": "xxx", "new": "cus_ABCDEFGH12"}, 120)
    r1 = eng.score(spike)
    eng.update(spike, Outcome.BLOCKED)
    eng.update(spike, Outcome.DENIED)
    assert dump(eng, tmp_path / "after.jsonl") == before
    assert eng.score(spike).fused_score == r1.fused_score


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
    return replay(*ci_dataset(), make_detector(name), SplitSpec())


def test_engine_beats_static_threshold_on_ci_dataset() -> None:
    eng = compute_all(_ci_rows("engine"), warm_only=False, n_boot=0).auprc
    static = compute_all(_ci_rows("static_threshold"), warm_only=False, n_boot=0).auprc
    assert eng > static


def test_golden_engine() -> None:
    golden = (ROOT / "tests/golden/engine.sha256").read_text().strip()
    assert canonical_hash(_ci_rows("engine")) == golden


def test_engine_replay_strict_mode_no_leak() -> None:
    raws, labels = ci_dataset()
    rows = replay(raws[:300], labels, Engine(), SplitSpec(), strict=True)
    assert len(rows) == 300
