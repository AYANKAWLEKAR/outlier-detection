from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import orjson
import pytest

from agentfw.core.envelope import Label, RawCall
from agentfw.core.patterns import email_domain, is_id_like
from agentfw.synth.attacks import (
    ATTACKS,
    CONTROL_P99_MULT,
    M_MAX,
    M_MIN,
    AttackContext,
    AttackResult,
    applicable,
    eligible,
    fabricate_id,
    far_bounds,
    inject,
    magnitude_spike,
    mutate_replace,
    mutate_scale,
    novel_value,
)
from agentfw.synth.generate import GenerateConfig, generate
from agentfw.synth.grammar import walk
from agentfw.synth.load import CATALOG_DIR, PERSONAS_DIR, TASKS_DIR, TIER0_PERSONA_IDS
from agentfw.synth.sampler import persona_rng

from .conftest import World


def _f(v: object) -> float:
    assert isinstance(v, int | float | str)
    return float(v)


PERSONA = "support-refunds-a"
FAMILIES = ["magnitude_spike", "prereq_skip", "novel_target"]


def _session(world: World, seed: int = 3) -> list[RawCall]:
    """A real walker session for small_refund that contains the retrieve_charge lookup."""
    p = world.persona(PERSONA)
    for s in range(seed, seed + 200):
        rng = persona_rng(s, PERSONA)
        planned = walk(world.tasks["small_refund"], p, world.catalog, rng, world.pools(PERSONA))
        tools = [c.tool_id for c in planned]
        if "stripe.create_refund" in tools and "stripe.retrieve_charge" in tools:
            return [
                RawCall(
                    envelope_id=f"env-{i}",
                    session_id="sess-1",
                    agent_id=PERSONA,
                    principal_id=f"{PERSONA}-principal",
                    seq_no=i,
                    ts_unix_ms=1_000 + i,
                    tool_id=c.tool_id,
                    args=c.args,
                    response_meta=c.response_meta,
                )
                for i, c in enumerate(planned)
            ]
    raise AssertionError("no suitable session")  # pragma: no cover


@pytest.fixture
def benign_session(world: World) -> list[RawCall]:
    return _session(world)


@pytest.fixture
def ctx(world: World, benign_session: list[RawCall]) -> Callable[[], AttackContext]:
    def make() -> AttackContext:
        return AttackContext(
            catalog=world.catalog,
            persona=world.persona(PERSONA),
            pools=world.pools(PERSONA),
            attack_id="atk-support-refunds-a-00000",
            persona_sessions=[benign_session],
            session_index=0,
        )

    return make


def test_attacks_are_pure_functions(
    benign_session: list[RawCall], ctx: Callable[[], AttackContext]
) -> None:
    for name, fn in ATTACKS.items():
        before = copy.deepcopy(benign_session)
        fn(benign_session, np.random.default_rng(0), ctx())
        assert benign_session == before, name


def test_magnitude_spike_scales_and_labels(
    benign_session: list[RawCall], ctx: Callable[[], AttackContext]
) -> None:
    r = ATTACKS["magnitude_spike"](benign_session, np.random.default_rng(0), ctx())
    assert r is not None and len(r.labels) == 1
    lab = r.labels[0]
    assert lab.label == "attack" and lab.attack_family == "magnitude_spike"
    assert lab.strength is not None and 3 <= lab.strength <= 100
    assert lab.attack_id == "atk-support-refunds-a-00000"
    assert len(r.session) == len(benign_session)
    old = next(c for c in benign_session if c.envelope_id == lab.envelope_id)
    new = next(c for c in r.session if c.envelope_id == lab.envelope_id)
    changed = [k for k in new.args if new.args[k] != old.args[k]]
    assert changed == ["amount"]
    assert _f(new.args["amount"]) == pytest.approx(_f(old.args["amount"]) * lab.strength, abs=0.01)


def test_magnitude_strength_is_log_uniform_in_range(
    benign_session: list[RawCall], ctx: Callable[[], AttackContext]
) -> None:
    rng = np.random.default_rng(1)
    ms = []
    for _ in range(400):
        r = ATTACKS["magnitude_spike"](benign_session, rng, ctx())
        assert r is not None and r.labels[0].strength is not None
        ms.append(r.labels[0].strength)
    assert min(ms) >= 3 and max(ms) <= 100
    assert 0.4 < np.mean(np.log(ms) < np.log(np.sqrt(300))) < 0.6


def test_prereq_skip_removes_lookup_and_fabricates(
    benign_session: list[RawCall], ctx: Callable[[], AttackContext]
) -> None:
    r = ATTACKS["prereq_skip"](benign_session, np.random.default_rng(0), ctx())
    assert r is not None and len(r.labels) >= 1  # target write first, then dependents
    write = next(c for c in r.session if c.envelope_id == r.labels[0].envelope_id)
    assert write.tool_id == "stripe.create_refund"
    earlier = [c for c in r.session if c.seq_no < write.seq_no]
    earlier_ids = {i for c in earlier for i in (c.response_meta.ids if c.response_meta else ())}
    assert write.args["charge_id"] not in earlier_ids
    assert "stripe.retrieve_charge" not in [c.tool_id for c in earlier]
    assert is_id_like(str(write.args["charge_id"]))
    assert [c.seq_no for c in r.session] == list(range(len(r.session)))
    kept = {c.envelope_id for c in r.session}
    assert write.envelope_id in kept
    assert kept < {c.envelope_id for c in benign_session}


def test_novel_target_replaces_with_unseen_value(
    benign_session: list[RawCall], ctx: Callable[[], AttackContext], world: World
) -> None:
    seen = set()
    for seed in range(40):
        r = ATTACKS["novel_target"](benign_session, np.random.default_rng(seed), ctx())
        assert r is not None and len(r.labels) == 1
        lab = r.labels[0]
        old = next(c for c in benign_session if c.envelope_id == lab.envelope_id)
        new = next(c for c in r.session if c.envelope_id == lab.envelope_id)
        changed = [k for k in new.args if new.args[k] != old.args[k]]
        assert len(changed) == 1
        seen.add((new.tool_id, changed[0]))
        if changed[0] == "reason":
            assert new.args["reason"] not in (world.catalog[new.tool_id].args["reason"].vocab or [])
        if changed[0] == "to":
            old_to, new_to = old.args["to"], new.args["to"]
            assert isinstance(old_to, list) and isinstance(new_to, list)
            assert all(
                (email_domain(str(e)) or "").endswith(".test") or e in old_to for e in new_to
            )
    assert ("mail.send_email", "to") in seen


def test_mutators(benign_session: list[RawCall]) -> None:
    refund = next(c for c in benign_session if c.tool_id == "stripe.create_refund")
    scaled = mutate_scale(refund, "amount", 10)
    assert _f(scaled.args["amount"]) == pytest.approx(_f(refund.args["amount"]) * 10, abs=0.01)
    assert scaled is not refund and scaled.envelope_id == refund.envelope_id
    rep = mutate_replace(refund, "reason", "zzz")
    assert rep.args["reason"] == "zzz" and refund.args["reason"] != "zzz"
    nested = mutate_replace(
        refund.model_copy(update={"args": {"to": ["a@b.example", "c@d.example"]}}), "to.1", "q"
    )
    assert nested.args["to"] == ["a@b.example", "q"]


def test_fabricate_and_novel_values(world: World) -> None:
    rng = np.random.default_rng(0)
    prefixes = {
        a.prefix for t in world.catalog.values() for a in t.args.values() if a.prefix is not None
    }
    for pre in sorted(prefixes):
        assert is_id_like(fabricate_id(pre, rng)), pre
    for _ in range(20):
        e = novel_value("email", rng)
        assert email_domain(e) is not None and (email_domain(e) or "").startswith("x-")
        assert e.endswith(".test")
        assert is_id_like(novel_value("id", rng, prefix="ba_"))
        assert novel_value("categorical", rng).startswith("x-")


def test_eligible(world: World) -> None:
    support = world.persona(PERSONA)
    for fam in FAMILIES:
        assert eligible(fam, support, world.catalog)
    # a persona with no write tools can't be attacked
    reads_only = support.model_copy(
        update={"tools": ["zendesk.get_ticket", "stripe.retrieve_charge"]}
    )
    for fam in FAMILIES:
        assert not eligible(fam, reads_only, world.catalog)
    # no money/count field and no id sourced from a read
    mail_only = support.model_copy(update={"tools": ["mail.send_email", "slack.post_message"]})
    assert not eligible("magnitude_spike", mail_only, world.catalog)
    assert not eligible("prereq_skip", mail_only, world.catalog)
    assert eligible("novel_target", mail_only, world.catalog)
    with pytest.raises(KeyError):
        eligible("nope", support, world.catalog)


def _cfg(**kw: object) -> GenerateConfig:
    base: dict[str, object] = {
        "seed": 0,
        "personas_dir": PERSONAS_DIR,
        "persona_ids": list(TIER0_PERSONA_IDS),
        "tasks_dir": TASKS_DIR,
        "catalog_dir": CATALOG_DIR,
        "days": 60,
        "families": FAMILIES,
    }
    return GenerateConfig.model_validate({**base, **kw})


def _rows(path: Path) -> list[dict[str, object]]:
    return [orjson.loads(x) for x in path.read_bytes().splitlines()]


def test_schedule_rate_and_warmup(tmp_path: Path) -> None:
    generate(_cfg(), tmp_path)
    traces = _rows(tmp_path / "traces.jsonl")
    labels = [Label.model_validate(x) for x in _rows(tmp_path / "labels.jsonl")]
    assert len(traces) == len(labels)
    by_env = {str(t["envelope_id"]): t for t in traces}
    sess_agent = {str(t["session_id"]): str(t["agent_id"]) for t in traces}
    first_ts: dict[str, int] = {}
    for t in traces:
        sid, ts = str(t["session_id"]), t["ts_unix_ms"]
        assert isinstance(ts, int)
        first_ts[sid] = min(first_ts.get(sid, 1 << 62), ts)
    attacked = {
        str(by_env[lab.envelope_id]["session_id"]) for lab in labels if lab.label == "attack"
    }
    frac = len(attacked) / len(sess_agent)
    assert 0.010 <= frac <= 0.020, frac
    for agent in {a for a in sess_agent.values()}:
        order = sorted((first_ts[s], s) for s, a in sess_agent.items() if a == agent)
        warm = {s for _, s in order[:30]}
        assert not (warm & attacked), agent
    ids = {lab.attack_id for lab in labels if lab.attack_id}
    assert all(i.startswith("atk-") and len(i.split("-")[-1]) == 5 for i in ids)
    assert {lab.attack_family for lab in labels if lab.label == "attack"} <= set(FAMILIES)


def test_attack_stream_does_not_perturb_benign(tmp_path: Path) -> None:
    generate(_cfg(days=20, attack_rate=0.0), tmp_path / "a")
    generate(_cfg(days=20, attack_rate=0.015), tmp_path / "b")
    ta, tb = _rows(tmp_path / "a" / "traces.jsonl"), _rows(tmp_path / "b" / "traces.jsonl")
    la = _rows(tmp_path / "b" / "labels.jsonl")
    attacked_sessions = {
        t["session_id"] for t, lab in zip(tb, la, strict=True) if lab["label"] == "attack"
    }
    assert attacked_sessions

    def keep(rows: list[dict[str, object]]) -> list[bytes]:
        return [
            orjson.dumps(t, option=orjson.OPT_SORT_KEYS)
            for t in rows
            if t["session_id"] not in attacked_sessions
        ]

    assert keep(ta) == keep(tb)
    assert len(ta) > len(keep(ta))


def test_generate_with_attacks_is_deterministic(tmp_path: Path) -> None:
    m1 = generate(_cfg(days=10), tmp_path / "a")
    m2 = generate(_cfg(days=10), tmp_path / "b")
    assert m1.files == m2.files


def test_unknown_family_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown attack family"):
        generate(_cfg(days=2, families=["bogus"]), tmp_path)


def test_family_choice_not_skewed(tmp_path: Path) -> None:
    # Per-session applicability legitimately favours novel_target (nearly every session has an
    # email/categorical write arg) but magnitude_spike must not be starved.
    generate(_cfg(days=120), tmp_path)
    labels = [Label.model_validate(x) for x in _rows(tmp_path / "labels.jsonl")]
    fam = {
        lab.attack_id: lab.attack_family
        for lab in labels
        if lab.attack_id is not None and lab.attack_family is not None
    }
    counts = {f: sum(1 for v in fam.values() if v == f) for f in FAMILIES}
    total = sum(counts.values())
    assert total > 60
    for n in counts.values():
        assert 0.10 <= n / total <= 0.60, counts


def test_family_choice_uniform_when_all_apply(
    benign_session: list[RawCall], ctx: Callable[[], AttackContext]
) -> None:
    assert all(applicable(f, benign_session, ctx()) for f in FAMILIES)
    n = 900
    out, labels = inject(
        {PERSONA: [benign_session] * n},
        _cfg(attack_rate=1.0, warmup_sessions=0),
        np.random.default_rng(5),
        lambda agent, aid, ps, idx: replace(ctx(), attack_id=aid),
    )
    assert len(out[PERSONA]) == n
    per = {f: len({lab.attack_id for lab in labels if lab.attack_family == f}) for f in FAMILIES}
    assert sum(per.values()) == n
    expected = n / 3
    chi2 = sum((c - expected) ** 2 / expected for c in per.values())
    assert chi2 < 13.8, per  # chi-square, 2 dof, p = 0.001


def test_applicable_matches_attack_fns(
    benign_session: list[RawCall], ctx: Callable[[], AttackContext]
) -> None:
    for name, fn in ATTACKS.items():
        got = fn(benign_session, np.random.default_rng(0), ctx())
        assert applicable(name, benign_session, ctx()) == (got is not None)
    reads = [c for c in benign_session if c.tool_id.startswith("zendesk.get")]
    for name in ATTACKS:
        assert not applicable(name, reads, ctx())


def test_session_with_no_applicable_family_stays_benign(
    world: World, benign_session: list[RawCall], ctx: Callable[[], AttackContext]
) -> None:
    reads = [c for c in benign_session if c.tool_id.startswith("zendesk.get")]
    cfg = _cfg(attack_rate=1.0, warmup_sessions=0)
    out, labels = inject(
        {PERSONA: [reads, reads]},
        cfg,
        np.random.default_rng(0),
        lambda agent, aid, ps, idx: ctx(),
    )
    assert labels == [] and out == {PERSONA: [reads, reads]}


def test_magnitude_strength_is_realized_multiplier_for_int_fields(world: World) -> None:
    call = RawCall(
        envelope_id="e1",
        session_id="s",
        agent_id="sales-ops-a",
        principal_id="p",
        seq_no=0,
        ts_unix_ms=1,
        tool_id="salesforce.create_task",
        args={"record_id": "006ABCDEFGHIJKLM", "subject": "hi", "due_in_days": 7},
    )
    sctx = AttackContext(
        world.catalog, world.persona("sales-ops-a"), {}, "atk-x-00000", [[call]], 0
    )
    for seed in range(30):
        r = ATTACKS["magnitude_spike"]([call], np.random.default_rng(seed), sctx)
        assert r is not None
        new = r.session[0].args["due_in_days"]
        assert isinstance(new, int)
        assert r.labels[0].strength == new / 7
        assert 3 <= new / 7 <= 100


def test_prereq_skip_never_present_values_are_not_dependents(
    benign_session: list[RawCall], ctx: Callable[[], AttackContext]
) -> None:
    sess = [c for c in benign_session if c.tool_id != "stripe.retrieve_charge"]
    mail = next(c for c in sess if c.tool_id == "mail.send_email")
    original_ids = {i for c in sess for i in (c.response_meta.ids if c.response_meta else ())}
    to = mail.args["to"]
    assert isinstance(to, list) and not {str(e) for e in to} & original_ids
    r = ATTACKS["prereq_skip"](sess, np.random.default_rng(0), ctx())
    assert r is not None
    assert {lab.envelope_id for lab in r.labels} == {
        next(c for c in sess if c.tool_id == "stripe.create_refund").envelope_id
    }


def test_prereq_skip_labels_dependent_whose_id_came_from_deleted_read(
    benign_session: list[RawCall], ctx: Callable[[], AttackContext]
) -> None:
    sess = [c for c in benign_session if c.tool_id != "stripe.retrieve_charge"]
    get_ticket = next(c for c in sess if c.tool_id == "zendesk.get_ticket")
    assert get_ticket.response_meta is not None
    ticket_id = get_ticket.response_meta.ids[0]
    last = sess[-1]
    follow = RawCall(
        envelope_id="env-follow",
        session_id=last.session_id,
        agent_id=last.agent_id,
        principal_id=last.principal_id,
        seq_no=last.seq_no + 1,
        ts_unix_ms=last.ts_unix_ms + 1,
        tool_id="zendesk.update_status",
        args={"ticket_id": ticket_id, "status": "solved"},
    )
    full = [*sess, follow]
    for seed in range(40):
        r = ATTACKS["prereq_skip"](full, np.random.default_rng(seed), ctx())
        assert r is not None
        first = next(c for c in r.session if c.envelope_id == r.labels[0].envelope_id)
        if first.tool_id != "stripe.create_refund":
            continue
        ids = {lab.envelope_id for lab in r.labels}
        assert "env-follow" in ids
        assert not any(c.tool_id == "mail.send_email" and c.envelope_id in ids for c in r.session)
        return
    raise AssertionError("refund never chosen")  # pragma: no cover


# --- far-mode magnitude_spike (Amendment A1.4) ----------------------------------------------------

REFUND = ("stripe.create_refund", "amount")
Far = Mapping[tuple[str, str], float]


def declared_p99(world: World) -> float:
    d = world.persona(PERSONA).fields["stripe.create_refund.amount"]
    assert d.mu is not None and d.sigma is not None
    return math.exp(d.mu + 2.326 * d.sigma)


def refund_sessions(world: World, amounts: list[float]) -> list[list[RawCall]]:
    """One copy of the benign refund session per amount (unique envelope and session ids)."""
    base = _session(world)
    return [
        [
            c.model_copy(
                update={
                    "envelope_id": f"{c.envelope_id}-{n}",
                    "session_id": f"sess-{n}",
                    "args": {**c.args, "amount": amt}
                    if c.tool_id == "stripe.create_refund"
                    else c.args,
                }
            )
            for c in base
        ]
        for n, amt in enumerate(amounts)
    ]


def far_ctx(world: World, far: Far | None, agent: str = PERSONA) -> AttackContext:
    return AttackContext(
        catalog=world.catalog,
        persona=world.persona(agent),
        pools={},
        attack_id=f"atk-{agent}-00000",
        persona_sessions=[],
        session_index=0,
        far=far,
    )


def _spiked(session: list[RawCall], res: AttackResult) -> tuple[float, float, str]:
    """``(new value, own value, envelope id)`` of the one call ``res`` changed."""
    (lab,) = res.labels
    old = next(c for c in session if c.envelope_id == lab.envelope_id)
    new = next(c for c in res.session if c.envelope_id == lab.envelope_id)
    (name,) = [k for k in new.args if new.args[k] != old.args[k]]
    return _f(new.args[name]), _f(old.args[name]), lab.envelope_id


class _FixedRng:
    """Stands in for the attack stream: candidate index ``k`` and the drawn ``ln m``."""

    def __init__(self, k: int, log_m: float) -> None:
        self.k, self.log_m = k, log_m

    def integers(self, n: int) -> int:
        assert 0 <= self.k < n
        return self.k

    def uniform(self, low: float, high: float) -> float:
        assert low - 1e-12 <= self.log_m <= high + 1e-12
        return self.log_m


def _fixed(k: int, log_m: float) -> np.random.Generator:
    return cast(np.random.Generator, _FixedRng(k, log_m))


def _task(due_in_days: int, n: int = 0) -> RawCall:
    return RawCall(
        envelope_id=f"task-{n}",
        session_id="s",
        agent_id="sales-ops-a",
        principal_id="p",
        seq_no=n,
        ts_unix_ms=1 + n,
        tool_id="salesforce.create_task",
        args={"record_id": "006ABCDEFGHIJKLM", "subject": "hi", "due_in_days": due_in_days},
    )


def test_far_bounds_is_mult_times_benign_p99(world: World) -> None:
    persona = world.persona(PERSONA)
    sessions = refund_sessions(world, amounts=[float(a) for a in range(1, 101)])  # p99 = 99.01
    far = far_bounds(sessions, persona, world.catalog)
    assert declared_p99(world) > 99.01  # the control's declared p99 wins on its amount arg
    assert far[REFUND] == pytest.approx(5.0 * max(99.01, declared_p99(world)))
    assert CONTROL_P99_MULT == 5.0
    assert far_bounds(sessions, persona, world.catalog, mult=1.0)[REFUND] * 5 == pytest.approx(
        far[REFUND]
    )
    assert far_bounds([], persona, world.catalog) == {}
    # only (write tool, money/count arg) pairs get a bound
    assert set(far) == {REFUND}
    # the sampled p99 wins when it is above the declared one (linear interpolation: 990.1)
    big = refund_sessions(world, amounts=[10.0 * a for a in range(1, 101)])
    assert declared_p99(world) < 990.1
    assert far_bounds(big, persona, world.catalog)[REFUND] == pytest.approx(5.0 * 990.1)
    assert far_bounds(big, persona, world.catalog, mult=2.0)[REFUND] == pytest.approx(2 * 990.1)


def test_far_bounds_without_a_declared_p99_and_non_positive_values(world: World) -> None:
    sales = world.persona("sales-ops-a")
    # an integer count field no control touches: the sampled p99 alone (ties: p99 == the maximum)
    tasks = [[_task(d, n)] for n, d in enumerate([3] * 50)]
    key = ("salesforce.create_task", "due_in_days")
    assert far_bounds(tasks, sales, world.catalog) == {key: pytest.approx(15.0)}
    # zero, negative, bool and non-numeric values are not benign magnitudes
    odd = [
        [_task(4, 0)],
        [_task(0, 1)],
        [_task(-7, 2)],
        [_task(4, 3).model_copy(update={"args": {"due_in_days": True}})],
        [_task(4, 4).model_copy(update={"args": {"due_in_days": "9"}})],
        [_task(4, 5).model_copy(update={"args": {"subject": "no count"}})],
    ]
    assert far_bounds(odd, sales, world.catalog) == {key: pytest.approx(20.0)}
    before = copy.deepcopy(odd)
    far_bounds(odd, sales, world.catalog)
    assert odd == before  # pure


def test_far_spike_lands_at_or_beyond_the_bound_with_strength_in_range(world: World) -> None:
    sessions = refund_sessions(world, amounts=[float(a) for a in range(1, 101)])
    far = far_bounds(sessions, world.persona(PERSONA), world.catalog)
    bound = far[REFUND]
    assert bound == pytest.approx(5.0 * declared_p99(world))  # about 932
    hits = 0
    for amount in (9.0, 9.4, 10.0, 37.25, 99.0, 400.0, 5000.0):
        (session,) = refund_sessions(world, amounts=[amount])
        before = copy.deepcopy(session)
        for seed in range(200):
            res = magnitude_spike(session, np.random.default_rng(seed), far_ctx(world, far))
            if res is None:
                assert amount * M_MAX < bound
                continue
            assert amount * M_MAX >= bound  # feasible exactly when value x 100 reaches the bound
            (lab,) = res.labels
            new, old, _ = _spiked(session, res)
            assert old == amount and new >= bound
            assert lab.strength is not None and M_MIN <= lab.strength <= M_MAX + 1e-9
            assert lab.strength == pytest.approx(new / old)
            assert lab.label == "attack" and lab.attack_family == "magnitude_spike"
            assert lab.attack_id == f"atk-{PERSONA}-00000"
            assert len(res.session) == len(session)
            hits += 1
        assert session == before  # pure
    assert hits == 6 * 200  # 9.0 x 100 < bound: never spiked; every other amount always


def test_far_multiplier_is_the_drawn_one_mapped_onto_the_feasible_range(world: World) -> None:
    (session,) = refund_sessions(world, amounts=[20.0])
    lo, hi = math.log(M_MIN), math.log(M_MAX)

    def new_value(bound: float, log_m: float) -> float:
        res = magnitude_spike(session, _fixed(0, log_m), far_ctx(world, {REFUND: bound}))
        assert res is not None
        return _spiked(session, res)[0]

    # bound 800 -> feasible multipliers [40, 100]: t = 0 -> 40, t = 1 -> 100, t = 0.5 -> sqrt(4000)
    assert new_value(800.0, lo) == pytest.approx(800.0)
    assert new_value(800.0, hi) == pytest.approx(2000.0)
    assert new_value(800.0, (lo + hi) / 2) == pytest.approx(
        20.0 * math.sqrt(40.0 * 100.0), abs=0.01
    )
    # a bound at or below 3 x the value leaves the drawn multiplier as it is (mode "own")
    for log_m in (lo, (lo + hi) / 2, 0.3 * lo + 0.7 * hi, hi):
        own = magnitude_spike(session, _fixed(0, log_m), far_ctx(world, None))
        assert own is not None
        assert new_value(60.0, log_m) == pytest.approx(_spiked(session, own)[0], abs=0.01)
        assert new_value(1.0, log_m) == pytest.approx(_spiked(session, own)[0], abs=0.01)
    # the multiplier never exceeds 100, even when the bound sits exactly at 100 x the value
    assert new_value(2000.0, lo) == new_value(2000.0, hi) == 2000.0


def test_far_float_field_fixup_rounds_the_bound_up_to_cents(world: World) -> None:
    (session,) = refund_sessions(world, amounts=[10.0])
    bound = 932.0349  # t = 0: m = 93.20349, 10.0 x m rounds DOWN to 932.03 < bound
    res = magnitude_spike(session, _fixed(0, math.log(M_MIN)), far_ctx(world, {REFUND: bound}))
    assert res is not None
    new, old, _ = _spiked(session, res)
    assert new == 932.04 and new >= bound
    assert res.labels[0].strength == pytest.approx(93.204) and new / old <= M_MAX


def test_far_picks_the_first_feasible_candidate_in_cyclic_order_from_k(world: World) -> None:
    session = [_task(3, 0), _task(1, 1), _task(2, 2)]
    key = ("salesforce.create_task", "due_in_days")
    ctx = far_ctx(world, {key: 150.0}, agent="sales-ops-a")  # feasible: task-0 (300), task-2 (200)
    mid = math.log(30.0)
    for k, want in ((0, "task-0"), (1, "task-2"), (2, "task-2")):
        res = magnitude_spike(session, _fixed(k, mid), ctx)
        assert res is not None and _spiked(session, res)[2] == want, k
    # the cycle wraps: only task-0 is feasible; k = 1 starts at task-1, passes task-2, lands on it
    wrap = far_ctx(world, {key: 250.0}, agent="sales-ops-a")
    for k in (0, 1, 2):
        res = magnitude_spike(session, _fixed(k, mid), wrap)
        assert res is not None and _spiked(session, res)[2] == "task-0", k
    # with every candidate feasible, far spikes the call mode "own" spikes
    easy = far_ctx(world, {key: 1.0}, agent="sales-ops-a")
    for seed in range(40):
        own = magnitude_spike(
            session, np.random.default_rng(seed), far_ctx(world, None, "sales-ops-a")
        )
        res = magnitude_spike(session, np.random.default_rng(seed), easy)
        assert own is not None and res is not None
        assert _spiked(session, res)[2] == _spiked(session, own)[2]
    # a candidate whose field has no bound is not feasible
    assert magnitude_spike(session, _fixed(0, mid), far_ctx(world, {}, "sales-ops-a")) is None


def test_far_mode_makes_the_same_draws_as_own(world: World) -> None:
    sessions = refund_sessions(world, amounts=[float(a) for a in range(1, 101)])
    far = far_bounds(sessions, world.persona(PERSONA), world.catalog)
    for amount, feasible in ((37.0, True), (5.0, False)):  # an infeasible session draws too
        (session,) = refund_sessions(world, amounts=[amount])
        for seed in range(20):
            a, b = np.random.default_rng(seed), np.random.default_rng(seed)
            own = magnitude_spike(session, a, far_ctx(world, None))
            res = magnitude_spike(session, b, far_ctx(world, far))
            assert own is not None and (res is not None) == feasible
            assert a.bit_generator.state == b.bit_generator.state
            assert a.random() == b.random()
    # no candidate at all: no draw in either mode
    reads = [c for c in sessions[0] if c.tool_id.startswith("zendesk.get")]
    for far_map in (None, far):
        rng = np.random.default_rng(1)
        state = copy.deepcopy(rng.bit_generator.state)
        assert magnitude_spike(reads, rng, far_ctx(world, far_map)) is None
        assert rng.bit_generator.state == state


def test_far_integer_field_fixup_within_m_max(world: World) -> None:
    key = ("salesforce.create_task", "due_in_days")
    session = [_task(1)]

    def ctx(bound: float) -> AttackContext:
        return far_ctx(world, {key: bound}, agent="sales-ops-a")

    # drawn ln m = ln 3 (t = 0): m = 39.2, 1 x 39.2 rounds to 39 < 39.2, so the fix-up applies
    res = magnitude_spike(session, _fixed(0, math.log(M_MIN)), ctx(39.2))
    assert res is not None
    new = res.session[0].args["due_in_days"]
    assert isinstance(new, int) and new == 40  # ceil(39.2), still an integer
    assert res.labels[0].strength == 40.0 <= M_MAX
    for seed in range(200):  # never below the bound, never above 100 x the value
        r = magnitude_spike(session, np.random.default_rng(seed), ctx(39.2))
        assert r is not None
        v = r.session[0].args["due_in_days"]
        assert isinstance(v, int) and 40 <= v <= 100 and r.labels[0].strength == float(v)
    # the bound exactly at 100 x the value is feasible and lands on it; ceil never passes m = 100
    for bound, want in ((100.0, 100), (99.2, 100)):
        r = magnitude_spike(session, _fixed(0, math.log(M_MIN)), ctx(bound))
        assert r is not None and r.session[0].args["due_in_days"] == want
        assert r.labels[0].strength == 100.0
    # value 1 with bound 150: 1 x 100 < 150, infeasible
    for seed in range(20):
        assert magnitude_spike(session, np.random.default_rng(seed), ctx(150.0)) is None
    # tied benign values (p99 == the maximum == 3): the bound is defined, 15 <= 3 x 100
    sales = world.persona("sales-ops-a")
    tied = far_bounds([[_task(3, n)] for n in range(40)], sales, world.catalog)
    assert tied == {key: pytest.approx(15.0)}
    for seed in range(50):
        r = magnitude_spike(
            [_task(3)], np.random.default_rng(seed), far_ctx(world, tied, "sales-ops-a")
        )
        assert r is not None and r.labels[0].strength is not None
        assert _f(r.session[0].args["due_in_days"]) >= 15 and 3 <= r.labels[0].strength <= 100


def _inject_refunds(
    world: World, amounts: list[float], **kw: object
) -> tuple[dict[str, list[list[RawCall]]], list[Label], dict[str, int], list[AttackContext]]:
    """Run the schedule over refund sessions with every session attacked (rate 1, no warm-up)."""
    sessions = refund_sessions(world, amounts)
    cfg = _cfg(**{"attack_rate": 1.0, "warmup_sessions": 0, "families": ["magnitude_spike"], **kw})
    stats: dict[str, int] = {}
    seen: list[AttackContext] = []
    shared = far_ctx(world, None)

    def factory(agent: str, aid: str, ps: object, idx: int) -> AttackContext:
        ctx = replace(shared, attack_id=aid)
        seen.append(ctx)
        return ctx

    out, labels = inject({PERSONA: sessions}, cfg, np.random.default_rng(11), factory, stats=stats)
    assert all(c.far is None for c in seen)  # the factory's contexts are never mutated
    return out, labels, stats, seen


AMOUNTS = [100.0, 100.0, 1.0, 100.0, 250.0]  # far bound 5 x 186.4 = 932: 1.0 x 100 falls short


def test_far_infeasible_session_stays_benign_and_consumes_its_attack_id(world: World) -> None:
    both = ["magnitude_spike", "benign_unusual"]
    own_out, own_labels, own_stats, _ = _inject_refunds(world, AMOUNTS, families=both)
    out, labels, stats, _ = _inject_refunds(
        world, AMOUNTS, families=both, magnitude_mode="far", control_rate=1.0
    )
    ids = [f"atk-{PERSONA}-{n:05d}" for n in range(5)]
    assert [lab.attack_id for lab in own_labels] == ids
    assert [lab.attack_id for lab in labels] == [ids[0], ids[1], ids[3], ids[4]]  # 00002 is gone
    dropped = refund_sessions(world, AMOUNTS)[2]
    assert out[PERSONA][2] == dropped  # untouched: no spike and, at control_rate 1, no control
    envs = {c.envelope_id for c in dropped}
    assert not [lab for lab in labels if lab.envelope_id in envs]
    assert all(lab.label == "attack" for lab in labels)
    assert stats == {"magnitude_first_pass": 4, "magnitude_dropped": 1, "magnitude_topup": 0}
    assert own_stats == {"magnitude_first_pass": 5, "magnitude_dropped": 0, "magnitude_topup": 0}
    bound = 5.0 * declared_p99(world)
    for i in (0, 1, 3, 4):
        refund = next(c for c in out[PERSONA][i] if c.tool_id == "stripe.create_refund")
        assert _f(refund.args["amount"]) >= bound
    # without the attack draw that same session does take the control
    _, ctl, _, _ = _inject_refunds(
        world, AMOUNTS, families=both, magnitude_mode="far", control_rate=1.0, attack_rate=0.0
    )
    assert [lab.control_type for lab in ctl if lab.envelope_id in envs] == ["benign_unusual"]


def test_far_bounds_come_from_the_untouched_sessions_and_the_configured_multiple(
    world: World,
) -> None:
    # 400 sessions at 1000.0: sampled p99 = 1000 > declared 186.4, so far = mult x 1000. A spike
    # that raised an earlier session must not raise the bound of a later one.
    amounts = [1000.0] * 400
    for mult in (5.0, 1.0):
        out, labels, stats, _ = _inject_refunds(
            world, amounts, magnitude_mode="far", magnitude_far_mult=mult
        )
        assert stats["magnitude_first_pass"] == 400 and stats["magnitude_dropped"] == 0
        news = [
            _f(c.args["amount"])
            for s in out[PERSONA]
            for c in s
            if c.tool_id == "stripe.create_refund"
        ]
        lowest = max(M_MIN, mult) * 1000.0  # the feasible range starts AT the bound (or at 3 x)
        assert lowest <= min(news) < 1.2 * lowest
        assert max(news) <= 100_000.0
    # mult = 1: the bound (1000) is below 3 x the value, so the spikes are those of mode "own"
    own_out, own_labels, _, _ = _inject_refunds(world, amounts)
    assert [lab.envelope_id for lab in labels] == [lab.envelope_id for lab in own_labels]
    assert [lab.strength for lab in labels] == pytest.approx([lab.strength for lab in own_labels])


def test_inject_stats_is_keyword_only_and_optional(world: World) -> None:
    sessions = refund_sessions(world, [100.0] * 30)
    cfg = _cfg(attack_rate=1.0, warmup_sessions=0, magnitude_mode="far")
    shared = far_ctx(world, None)  # one context object handed out for every session

    def factory(agent: str, aid: str, ps: object, idx: int) -> AttackContext:
        return shared

    with pytest.raises(TypeError):
        inject({PERSONA: sessions}, cfg, np.random.default_rng(11), factory, {})  # type: ignore[call-arg]
    a = inject({PERSONA: sessions}, cfg, np.random.default_rng(11), factory)
    stats: dict[str, int] = {"stale": 7}
    b = inject({PERSONA: sessions}, cfg, np.random.default_rng(11), factory, stats=stats)
    assert a == b
    assert set(stats) == {"stale", "magnitude_first_pass", "magnitude_dropped", "magnitude_topup"}
    assert stats["magnitude_dropped"] == stats["magnitude_topup"] == 0
    # the bounds reach magnitude_spike on a NEW context: the shared one is never written to
    assert shared.far is None and stats["magnitude_first_pass"] > 0
    bound = 5.0 * declared_p99(world)
    spiked = {lab.envelope_id for lab in b[1] if lab.attack_family == "magnitude_spike"}
    news = [_f(c.args["amount"]) for s in b[0][PERSONA] for c in s if c.envelope_id in spiked]
    assert news and min(news) >= bound
