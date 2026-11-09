from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import orjson
import pytest

from agentfw.core.envelope import Label, RawCall
from agentfw.core.patterns import email_domain, is_id_like
from agentfw.synth.attacks import (
    ATTACKS,
    AttackContext,
    applicable,
    eligible,
    fabricate_id,
    inject,
    mutate_replace,
    mutate_scale,
    novel_value,
)
from agentfw.synth.generate import GenerateConfig, generate
from agentfw.synth.grammar import walk
from agentfw.synth.load import CATALOG_DIR, PERSONAS_DIR, TASKS_DIR
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
