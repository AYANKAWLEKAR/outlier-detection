"""Tier-1 attack families (scope_blowup, velocity_burst, slow_drift, exfil_volume) + controls."""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from agentfw.bench.dataset import load_dir
from agentfw.core.envelope import Label, RawCall
from agentfw.core.patterns import email_domain
from agentfw.synth.attacks import ATTACKS, AttackContext, applicable, eligible, inject
from agentfw.synth.generate import GenerateConfig, generate
from agentfw.synth.load import CATALOG_DIR, PERSONAS_DIR, TASKS_DIR, TIER0_PERSONA_IDS
from agentfw.synth.sampler import sample_persona

from .conftest import World

START = 1_767_225_600_000
Ctx = Callable[..., AttackContext]


def _sessions(world: World, agent: str, days: int = 6, seed: int = 0) -> list[list[RawCall]]:
    return sample_persona(
        world.persona(agent),
        world.catalog,
        world.tasks,
        days=days,
        global_seed=seed,
        start_ms=START,
    )


def _first(sessions: list[list[RawCall]], pred: Callable[[list[RawCall]], bool]) -> int:
    return next(i for i, s in enumerate(sessions) if pred(s))


@pytest.fixture
def make_ctx(world: World) -> Ctx:
    def make(
        agent: str, sessions: list[list[RawCall]] | None = None, idx: int = 0
    ) -> AttackContext:
        return AttackContext(
            world.catalog,
            world.persona(agent),
            world.pools(agent),
            f"atk-{agent}-00000",
            sessions if sessions is not None else [],
            idx,
        )

    return make


def _by_env(session: list[RawCall]) -> dict[str, RawCall]:
    return {c.envelope_id: c for c in session}


def _seq_ok(session: list[RawCall]) -> None:
    assert [c.seq_no for c in session] == list(range(len(session)))
    ts = [c.ts_unix_ms for c in session]
    assert ts == sorted(ts)
    assert len({c.envelope_id for c in session}) == len(session)


# --- scope_blowup ---------------------------------------------------------------------------------


def test_scope_blowup_expands_record_ids(world: World, make_ctx: Ctx) -> None:
    sess = _sessions(world, "sales-ops-a")
    s = sess[_first(sess, lambda s: any(c.tool_id == "salesforce.mass_update" for c in s))]
    before = copy.deepcopy(s)
    rng = np.random.default_rng(0)
    ns = []
    for _ in range(60):
        ctx = make_ctx("sales-ops-a")
        r = ATTACKS["scope_blowup"](s, rng, ctx)
        assert r is not None and len(r.labels) == 1
        lab = r.labels[0]
        assert lab.label == "attack" and lab.attack_family == "scope_blowup"
        assert lab.strength is not None and 100 <= lab.strength <= 10_000
        new = _by_env(r.session)[lab.envelope_id]
        old = _by_env(s)[lab.envelope_id]
        key = next(k for k in new.args if new.args[k] != old.args[k])
        grown, orig = new.args[key], old.args[key]
        assert isinstance(grown, list) and isinstance(orig, list)
        assert len(grown) == lab.strength
        assert grown[: len(orig)] == orig
        if key == "record_ids":
            assert all(str(x).startswith("006") for x in grown)
            assert len(set(map(str, grown))) == len(grown)
        else:
            seen = {email_domain(str(e)) for e in orig}
            assert {email_domain(str(e)) for e in grown} <= seen
        ns.append(lab.strength)
        assert len(r.session) == len(s)
    assert s == before
    assert 0.35 < np.mean(np.log(ns) < np.log(1000)) < 0.65  # log-uniform on [100, 10000]


def test_scope_blowup_eligible_and_applicable(world: World, make_ctx: Ctx) -> None:
    assert eligible("scope_blowup", world.persona("sales-ops-a"), world.catalog)
    assert eligible("scope_blowup", world.persona("outreach-a"), world.catalog)  # mail.to
    reads = world.persona("support-refunds-a").model_copy(
        update={"tools": ["zendesk.get_ticket", "stripe.retrieve_charge"]}
    )
    assert not eligible("scope_blowup", reads, world.catalog)
    sess = _sessions(world, "sales-ops-a")
    pipeline = sess[_first(sess, lambda s: all(c.tool_id.startswith("salesforce.get") for c in s))]
    assert not applicable("scope_blowup", pipeline, make_ctx("sales-ops-a"))


# --- velocity_burst -------------------------------------------------------------------------------


def test_velocity_burst_duplicates_within_window(world: World, make_ctx: Ctx) -> None:
    sess = _sessions(world, "support-refunds-a")
    s = sess[_first(sess, lambda s: any(c.tool_id == "stripe.create_refund" for c in s))]
    before = copy.deepcopy(s)
    rng = np.random.default_rng(1)
    rs = []
    for _ in range(60):
        r = ATTACKS["velocity_burst"](s, rng, make_ctx("support-refunds-a"))
        assert r is not None
        n = r.labels[0].strength
        assert n is not None and 10 <= n <= 60 and n == int(n)
        assert len(r.labels) == n and len(r.session) == len(s) + n
        assert all(
            lab.label == "attack" and lab.attack_family == "velocity_burst" for lab in r.labels
        )
        assert {lab.strength for lab in r.labels} == {n}
        assert len({lab.attack_id for lab in r.labels}) == 1
        _seq_ok(r.session)
        dups = [_by_env(r.session)[lab.envelope_id] for lab in r.labels]
        orig_ids = {c.envelope_id for c in s}
        assert not {d.envelope_id for d in dups} & orig_ids
        src = {(d.tool_id, repr(sorted(d.args.items(), key=str))) for d in dups}
        assert len(src) == 1  # all duplicates of one write
        original = next(
            c
            for c in s
            if (c.tool_id, repr(sorted(c.args.items(), key=str))) in src and c.tool_id != ""
        )
        assert original.envelope_id in {c.envelope_id for c in r.session}
        offs = [d.ts_unix_ms - original.ts_unix_ms for d in dups]
        assert min(offs) >= 0
        rs.append((n, max(offs)))
    assert max(m for _, m in rs) <= 300_000
    assert s == before


def test_velocity_window_is_60_to_300_s(world: World, make_ctx: Ctx) -> None:
    sess = _sessions(world, "support-refunds-a")
    s = sess[_first(sess, lambda s: any(c.tool_id == "stripe.create_refund" for c in s))]
    rng = np.random.default_rng(2)
    spans = []
    for _ in range(300):
        r = ATTACKS["velocity_burst"](s, rng, make_ctx("support-refunds-a"))
        assert r is not None
        ts = [_by_env(r.session)[lab.envelope_id].ts_unix_ms for lab in r.labels]
        spans.append(max(ts) - min(ts))
    assert max(spans) <= 300_000
    assert max(spans) > 200_000  # w reaches the upper range


# --- slow_drift -----------------------------------------------------------------------------------


def _drift_sessions(world: World) -> list[list[RawCall]]:
    return _sessions(world, "support-refunds-a", days=3)


def test_slow_drift_consecutive_sessions(world: World, make_ctx: Ctx) -> None:
    sess = _drift_sessions(world)
    assert len(sess) > 80
    before = copy.deepcopy(sess)
    ctx = make_ctx("support-refunds-a", sess, 10)
    r = ATTACKS["slow_drift"](sess[10], np.random.default_rng(0), ctx)
    assert r is not None
    k = 1 + len(r.extra_sessions)
    assert 15 <= k <= 40
    assert sess == before
    labs = r.labels
    eps = {lab.strength for lab in labs}
    assert len(eps) == 1
    (e,) = eps
    assert e is not None and 0.03 <= e <= 0.10
    assert len({lab.attack_id for lab in labs}) == 1
    assert all(lab.label == "attack" and lab.attack_family == "slow_drift" for lab in labs)
    steps = [lab.step for lab in labs]
    assert all(s is not None and 1 <= s <= k for s in steps)
    int_steps = [s for s in steps if s is not None]
    assert int_steps == sorted(int_steps) and int_steps[0] == 1  # step is 1-based
    out = [r.session, *r.extra_sessions]
    for s_in, s_out in zip(sess[10 : 10 + k], out, strict=True):  # sessions are consecutive
        assert [c.envelope_id for c in s_in] == [c.envelope_id for c in s_out]
        _seq_ok(s_out)
    new = {c.envelope_id: c for s in out for c in s}
    old = {c.envelope_id: c for s in sess[10 : 10 + k] for c in s}
    ratios = []
    for lab in labs:
        assert lab.step is not None
        n, o = new[lab.envelope_id], old[lab.envelope_id]
        field_ = [a for a in n.args if n.args[a] != o.args[a]]
        assert field_ == ["amount"]
        ratios.append((lab.step, float(n.args["amount"]) / float(o.args["amount"])))  # type: ignore[arg-type]
    for step, ratio in ratios:
        assert ratio == pytest.approx((1 + e) ** step, rel=0.02, abs=0.02)
    by_step: dict[int, list[float]] = defaultdict(list)
    for step, ratio in ratios:
        by_step[step].append(ratio)
    means = [float(np.mean(by_step[s])) for s in sorted(by_step)]
    assert means == sorted(means)  # monotone multipliers


def test_slow_drift_not_enough_sessions(world: World, make_ctx: Ctx) -> None:
    sess = _drift_sessions(world)
    idx = len(sess) - 14
    ctx = make_ctx("support-refunds-a", sess, idx)
    assert not applicable("slow_drift", sess[idx], ctx)
    assert ATTACKS["slow_drift"](sess[idx], np.random.default_rng(0), ctx) is None
    idx = len(sess) - 15
    ctx = make_ctx("support-refunds-a", sess, idx)
    if applicable("slow_drift", sess[idx], ctx):
        r = ATTACKS["slow_drift"](sess[idx], np.random.default_rng(0), ctx)
        assert r is not None and len(r.extra_sessions) == 14  # k clamped to remaining


def test_inject_slow_drift_marks_all_sessions_and_does_not_reattack(
    world: World, make_ctx: Ctx
) -> None:
    sess = _drift_sessions(world)
    cfg = GenerateConfig(
        seed=0,
        personas_dir=PERSONAS_DIR,
        persona_ids=list(TIER0_PERSONA_IDS),
        tasks_dir=TASKS_DIR,
        catalog_dir=CATALOG_DIR,
        days=3,
        attack_rate=1.0,
        warmup_sessions=0,
        families=["slow_drift"],
    )
    out, labels = inject(
        {"support-refunds-a": sess},
        cfg,
        np.random.default_rng(3),
        lambda agent, aid, ps, idx: replace(make_ctx(agent, list(ps), idx), attack_id=aid),
    )
    assert len(out["support-refunds-a"]) == len(sess)
    env_session = {c.envelope_id: i for i, s in enumerate(out["support-refunds-a"]) for c in s}
    by_attack: dict[str, list[Label]] = defaultdict(list)
    for lab in labels:
        assert lab.attack_id is not None
        by_attack[lab.attack_id].append(lab)
    assert len(by_attack) >= 2
    covered: list[set[int]] = []
    for labs in by_attack.values():
        idxs = sorted({env_session[x.envelope_id] for x in labs})
        covered.append(set(range(idxs[0], idxs[-1] + 1)))
    for i, a in enumerate(covered):  # attack windows are disjoint (sessions never re-attacked)
        for b in covered[i + 1 :]:
            assert not a & b


# --- exfil_volume ---------------------------------------------------------------------------------


def _digest(world: World) -> list[RawCall]:
    sess = _sessions(world, "outreach-a")
    pair = ["slack.read_channel", "slack.post_message"]
    return sess[_first(sess, lambda s: [c.tool_id for c in s][:2] == pair)]


def test_exfil_volume_with_existing_outbound(world: World, make_ctx: Ctx) -> None:
    s = _digest(world)
    assert [c.tool_id for c in s][:2] == ["slack.read_channel", "slack.post_message"]
    before = copy.deepcopy(s)
    rng = np.random.default_rng(0)
    for _ in range(40):
        r = ATTACKS["exfil_volume"](s, rng, make_ctx("outreach-a"))
        assert r is not None and len(r.labels) == 2
        v = r.labels[0].strength
        assert v is not None and 10 <= v <= 100
        assert {lab.strength for lab in r.labels} == {v}
        assert all(lab.attack_family == "exfil_volume" for lab in r.labels)
        new = _by_env(r.session)
        read = new[r.labels[0].envelope_id]
        write = new[r.labels[1].envelope_id]
        assert read.tool_id == "slack.read_channel" and write.tool_id == "slack.post_message"
        assert read.seq_no < write.seq_no
        assert read.response_meta is not None
        assert read.response_meta.item_count == round(v * 12)  # poisson median ~= lam = 12
        assert len(r.session) == len(s)
    assert s == before


def test_exfil_volume_appends_outbound_when_missing(world: World, make_ctx: Ctx) -> None:
    s = [c for c in _digest(world) if c.tool_id == "slack.read_channel"]
    rng = np.random.default_rng(4)
    r = ATTACKS["exfil_volume"](s, rng, make_ctx("outreach-a"))
    assert r is not None and len(r.session) == len(s) + 1
    _seq_ok(r.session)
    write = _by_env(r.session)[r.labels[-1].envelope_id]
    assert write.tool_id in ("mail.send_email", "slack.post_message")
    assert write.envelope_id not in {c.envelope_id for c in s}
    assert write.seq_no == len(r.session) - 1
    assert write.ts_unix_ms >= s[-1].ts_unix_ms
    assert write.response_meta is not None
    r2 = ATTACKS["exfil_volume"](s, np.random.default_rng(4), make_ctx("outreach-a"))
    assert r2 is not None and r2.session == r.session  # deterministic


def test_exfil_volume_labels_the_call_that_carries_the_response(
    world: World, make_ctx: Ctx
) -> None:
    """§2.3 / D-004: the read's item_count is scored on the NEXT call of the session. When another
    call sits between the read and the outbound write, that next call carries the only scoreable
    evidence, so it is part of the attack (label order: read, carrier, outbound)."""
    s0 = _digest(world)
    read, post = s0[0], s0[1]
    mid = read.model_copy(
        update={
            "envelope_id": "01MIDCALL0000000000000000",
            "ts_unix_ms": read.ts_unix_ms + 1,
            "response_meta": None,  # not itself an exfil candidate
        }
    )
    s = [
        read,
        mid,
        post.model_copy(update={"ts_unix_ms": max(post.ts_unix_ms, read.ts_unix_ms + 2)}),
    ]
    s = [c.model_copy(update={"seq_no": k}) for k, c in enumerate(s)]
    r = ATTACKS["exfil_volume"](s, np.random.default_rng(0), make_ctx("outreach-a"))
    assert r is not None
    assert [lab.envelope_id for lab in r.labels] == [c.envelope_id for c in s]
    assert len({lab.strength for lab in r.labels}) == 1
    assert all(lab.attack_family == "exfil_volume" for lab in r.labels)


def test_exfil_volume_eligibility(world: World, make_ctx: Ctx) -> None:
    assert eligible("exfil_volume", world.persona("outreach-a"), world.catalog)
    assert eligible("exfil_volume", world.persona("sales-ops-a"), world.catalog)  # run_soql
    assert not eligible("exfil_volume", world.persona("support-refunds-a"), world.catalog)
    sess = _sessions(world, "support-refunds-a")
    assert not applicable("exfil_volume", sess[0], make_ctx("support-refunds-a"))


# --- benign_unusual control -----------------------------------------------------------------------


def _refund_session(world: World) -> list[RawCall]:
    sess = _sessions(world, "support-refunds-a")
    return sess[
        _first(
            sess,
            lambda s: (
                any(c.tool_id == "stripe.retrieve_charge" for c in s)
                and any(c.tool_id == "stripe.create_refund" and "amount" in c.args for c in s)
            ),
        )
    ]


def test_benign_unusual_refund_control(world: World, make_ctx: Ctx) -> None:
    s = _refund_session(world)
    before = copy.deepcopy(s)
    r = ATTACKS["benign_unusual"](s, np.random.default_rng(0), make_ctx("support-refunds-a"))
    assert r is not None and len(r.labels) == 1
    lab = r.labels[0]
    assert lab.label == "benign" and lab.control_type == "benign_unusual"
    assert lab.attack_id is None and lab.attack_family is None
    refund = _by_env(r.session)[lab.envelope_id]
    assert refund.tool_id == "stripe.create_refund"
    assert refund.args["reason"] == "manager_approved"
    p99 = math.exp(3.6 + 2.326 * 0.7)
    assert float(refund.args["amount"]) == pytest.approx(5 * p99, abs=0.01)  # type: ignore[arg-type]
    prior = [c.tool_id for c in r.session if c.seq_no < refund.seq_no]
    assert "stripe.retrieve_charge" in prior
    assert len(r.session) == len(s)
    assert s == before


def test_benign_unusual_needs_prior_lookup(world: World, make_ctx: Ctx) -> None:
    s = [c for c in _refund_session(world) if c.tool_id != "stripe.retrieve_charge"]
    assert not applicable("benign_unusual", s, make_ctx("support-refunds-a"))
    assert (
        ATTACKS["benign_unusual"](s, np.random.default_rng(0), make_ctx("support-refunds-a"))
        is None
    )
    assert eligible("benign_unusual", world.persona("support-refunds-a"), world.catalog)
    assert not eligible("benign_unusual", world.persona("outreach-a"), world.catalog)


def test_inject_controls_independent_of_attack_draw(world: World, make_ctx: Ctx) -> None:
    s = _refund_session(world)
    base = dict(
        seed=0,
        personas_dir=PERSONAS_DIR,
        persona_ids=list(TIER0_PERSONA_IDS),
        tasks_dir=TASKS_DIR,
        catalog_dir=CATALOG_DIR,
        days=1,
        warmup_sessions=0,
    )

    def run(rate: float, ctl: float, fams: list[str]) -> list[Label]:
        cfg = GenerateConfig.model_validate(
            {**base, "attack_rate": rate, "control_rate": ctl, "families": fams}
        )
        copies = [
            [c.model_copy(update={"envelope_id": f"{c.envelope_id}-{n}"}) for c in s]
            for n in range(400)
        ]
        _, labs = inject(
            {"support-refunds-a": copies},
            cfg,
            np.random.default_rng(9),
            lambda agent, aid, ps, idx: replace(make_ctx(agent), attack_id=aid),
        )
        return labs

    labs = run(0.0, 0.25, ["benign_unusual"])  # controls fire with no attacks at all
    assert labs and all(lab.control_type == "benign_unusual" for lab in labs)
    assert 60 <= len(labs) <= 140
    assert run(0.0, 0.25, ["magnitude_spike"]) == []  # controls need benign_unusual enabled
    both = run(0.5, 0.25, ["magnitude_spike", "benign_unusual"])
    ctl = [lab for lab in both if lab.label == "benign"]
    atk = [lab for lab in both if lab.label == "attack"]
    assert ctl and atk and all(lab.attack_family == "magnitude_spike" for lab in atk)
    assert not {lab.envelope_id for lab in ctl} & {lab.envelope_id for lab in atk}


# --- quarter-end sessions + all-families generation -----------------------------------------------


def _all_cfg(**kw: object) -> GenerateConfig:
    base: dict[str, object] = {
        "seed": 0,
        "personas_dir": PERSONAS_DIR,
        "persona_ids": list(TIER0_PERSONA_IDS),
        "tasks_dir": TASKS_DIR,
        "catalog_dir": CATALOG_DIR,
        "days": 100,
        "attack_rate": 0.15,
        "control_rate": 0.05,
        "families": [
            "magnitude_spike",
            "prereq_skip",
            "novel_target",
            "scope_blowup",
            "velocity_burst",
            "slow_drift",
            "exfil_volume",
            "benign_unusual",
        ],
    }
    return GenerateConfig.model_validate({**base, **kw})


def test_quarter_end_sessions_scheduled_and_labeled(tmp_path: Path) -> None:
    generate(_all_cfg(attack_rate=0.0, control_rate=0.0), tmp_path)
    raws, labels, _ = load_dir(tmp_path)
    ctl = [r for r in raws if labels[r.envelope_id].control_type == "benign_unusual"]
    assert ctl and {r.agent_id for r in ctl} == {"sales-ops-a"}
    sessions = {r.session_id for r in ctl}
    for sid in sessions:  # every envelope of such a session is a control
        assert all(
            labels[r.envelope_id].control_type == "benign_unusual"
            for r in raws
            if r.session_id == sid
        )
    # Simulated days 28-31 of a quarter-end month (Mar 2026 = days 59-89, Jun = 150-180)
    import datetime as dt

    for r in ctl:
        d = dt.datetime.fromtimestamp(r.ts_unix_ms / 1000, dt.UTC)
        assert d.month in (3, 6, 9, 12) and d.day in (28, 29, 30, 31)
    assert any(r.tool_id == "salesforce.mass_update" for r in ctl)
    assert all(lab.label == "benign" for lab in labels.values())


def test_quarter_end_absent_before_quarter_end(tmp_path: Path) -> None:
    generate(_all_cfg(days=60, attack_rate=0.0, control_rate=0.0), tmp_path)
    _, labels, _ = load_dir(tmp_path)
    assert not any(lab.control_type for lab in labels.values())


def test_label_integrity_all_eight_families(tmp_path: Path) -> None:
    generate(_all_cfg(), tmp_path)
    raws, labels, _ = load_dir(tmp_path)
    assert len(labels) == len(raws) and {r.envelope_id for r in raws} == set(labels)
    by_attack: dict[str, int] = defaultdict(int)
    fams = set()
    for lab in labels.values():
        if lab.label == "attack":
            assert lab.control_type is None and lab.attack_id is not None
            by_attack[lab.attack_id] += 1
            fams.add(lab.attack_family)
        else:
            assert lab.attack_id is None and lab.attack_family is None
    assert fams == {
        "magnitude_spike",
        "prereq_skip",
        "novel_target",
        "scope_blowup",
        "velocity_burst",
        "slow_drift",
        "exfil_volume",
    }
    assert any(lab.control_type == "benign_unusual" for lab in labels.values())
    ids = [r.envelope_id for r in raws]
    assert len(ids) == len(set(ids))
    per_session: dict[str, list[int]] = defaultdict(list)
    for r in raws:
        per_session[r.session_id].append(r.seq_no)
    assert all(sorted(v) == list(range(len(v))) for v in per_session.values())


def test_all_families_deterministic(tmp_path: Path) -> None:
    m1 = generate(_all_cfg(days=40), tmp_path / "a")
    m2 = generate(_all_cfg(days=40), tmp_path / "b")
    assert m1.files == m2.files
