"""The ``splice`` family (Task 2.4, F10; T-10): three calls of another agent's benign session."""

from __future__ import annotations

import copy
from collections import defaultdict
from pathlib import Path

import numpy as np
import orjson
import pytest

from agentfw.bench.ci_regress import CI_DATASET
from agentfw.core.envelope import Label, RawCall, ResponseMeta
from agentfw.core.patterns import is_id_like
from agentfw.synth.attacks import (
    ATTACKS,
    RUNAWAY_FAMILIES,
    SPLICE_ATTACKS,
    SPLICE_DONORS_MAX,
    SPLICE_LEN,
    AttackContext,
    applicable,
    donor_sessions,
    eligible,
)
from agentfw.synth.generate import GenerateConfig, generate, sw1_config
from agentfw.synth.sampler import sample_persona

from .conftest import World

START = 1_767_225_600_000
HOST = "support-refunds-a"


def _sessions(world: World, agent: str, days: int = 6) -> list[list[RawCall]]:
    return sample_persona(
        world.persona(agent), world.catalog, world.tasks, days=days, global_seed=0, start_ms=START
    )


@pytest.fixture(scope="module")
def by_persona(world: World) -> dict[str, list[list[RawCall]]]:
    return {a: _sessions(world, a) for a in sorted(world.personas)}


def _ctx(world: World, agent: str, by_persona: dict[str, list[list[RawCall]]]) -> AttackContext:
    donors = donor_sessions(agent, world.personas, by_persona)
    return AttackContext(
        world.catalog, world.persona(agent), world.pools(agent), f"atk-{agent}-00000", [], 0,
        donors=donors,
    )  # fmt: skip


def _strings(v: object) -> list[str]:
    if isinstance(v, str):
        return [v]
    if isinstance(v, list):
        return [x for e in v for x in _strings(e)]
    if isinstance(v, dict):
        return [x for e in v.values() for x in _strings(e)]
    return []


def _id_args(world: World, c: RawCall) -> set[str]:
    return {n for n, a in world.catalog[c.tool_id].args.items() if a.type in ("id", "list_of_id")}


def test_donors_share_three_tools_earliest_first_at_most_50(
    world: World, by_persona: dict[str, list[list[RawCall]]]
) -> None:
    for agent in sorted(world.personas):
        tools = set(world.persona(agent).tools)
        donors = donor_sessions(agent, world.personas, by_persona)
        assert 0 < len(donors) <= SPLICE_DONORS_MAX
        firsts = [(s[0].ts_unix_ms, s[0].session_id) for _, s in donors]
        assert firsts == sorted(firsts)
        for other, s in donors:
            assert other != agent and {c.agent_id for c in s} == {other}
            assert len(tools & set(world.persona(other).tools)) >= 3
        # no earlier qualifying session was skipped
        cut = firsts[-1] if len(donors) == SPLICE_DONORS_MAX else (2**63, "")
        qualifying = [
            (s[0].ts_unix_ms, s[0].session_id)
            for other, ss in by_persona.items()
            if other != agent and len(tools & set(world.persona(other).tools)) >= 3
            for s in ss
            if s and (s[0].ts_unix_ms, s[0].session_id) <= cut
        ]
        assert sorted(qualifying) == firsts


def test_splice_transplants_three_calls_from_another_agent(
    world: World, by_persona: dict[str, list[list[RawCall]]]
) -> None:
    fn = SPLICE_ATTACKS["splice"]
    tools = set(world.persona(HOST).tools)
    pools = {x for p in world.pools(HOST).values() for x in p.ids}
    checked = rebound = 0
    for s in by_persona[HOST][:40]:
        ctx = _ctx(world, HOST, by_persona)
        before = copy.deepcopy(s)
        r = fn(s, np.random.default_rng(len(s)), ctx)
        assert s == before  # pure
        assert applicable("splice", s, ctx) == (r is not None)
        if r is None:
            assert len(s) < 2
            continue
        checked += 1
        assert r == fn(s, np.random.default_rng(len(s)), _ctx(world, HOST, by_persona))
        out = r.session
        assert len(out) == len(s) + SPLICE_LEN
        assert [c.seq_no for c in out] == list(range(len(out)))
        assert len({c.envelope_id for c in out}) == len(out)
        # labels: exactly the three copies, contiguous, attack / splice
        assert [(lab.label, lab.attack_family, lab.attack_id) for lab in r.labels] == [
            ("attack", "splice", ctx.attack_id)
        ] * SPLICE_LEN
        hit = {lab.envelope_id for lab in r.labels}
        pos = [i for i, c in enumerate(out) if c.envelope_id in hit]
        assert pos == list(range(pos[0], pos[0] + SPLICE_LEN)) and 1 <= pos[0] <= len(s) - 1
        kept = [c for c in out if c.envelope_id not in hit]
        assert [c.model_copy(update={"seq_no": 0}) for c in kept] == [
            c.model_copy(update={"seq_no": 0}) for c in s
        ]
        copies = [out[i] for i in pos]
        # timestamps evenly between the neighbours
        left, right = out[pos[0] - 1].ts_unix_ms, out[pos[-1] + 1].ts_unix_ms
        assert [c.ts_unix_ms for c in copies] == [
            left + (right - left) * j // (SPLICE_LEN + 1) for j in range(1, SPLICE_LEN + 1)
        ]
        # host identity, no task ids, fresh envelope ids
        for c in copies:
            assert (c.session_id, c.agent_id, c.principal_id) == (
                s[0].session_id,
                s[0].agent_id,
                s[0].principal_id,
            )
            assert c.task_ids == () and c.tool_id in tools
        donor_ids = {c.envelope_id for _, d in ctx.donors for c in d}
        assert not {c.envelope_id for c in copies} & (donor_ids | {c.envelope_id for c in s})
        # the window: three consecutive calls of one donor session of another agent, with the
        # same tools and non-id args
        windows = [
            (other, d[k : k + SPLICE_LEN])
            for other, d in ctx.donors
            for k in range(len(d) - SPLICE_LEN + 1)
            if [c.tool_id for c in d[k : k + SPLICE_LEN]] == [c.tool_id for c in copies]
            and all(
                {n: v for n, v in w.args.items() if n not in _id_args(world, w)}
                == {n: v for n, v in c.args.items() if n not in _id_args(world, c)}
                for w, c in zip(d[k : k + SPLICE_LEN], copies, strict=True)
            )
        ]
        assert windows and all(other != HOST for other, _ in windows)
        # id args: produced inside the window, seen in this session before the insertion point,
        # or drawn from the persona's pools
        seen: set[str] = set()
        for c in out[: pos[0]]:
            seen |= {v for v in _strings(c.args) if is_id_like(v)}
            seen |= set(c.response_meta.ids if c.response_meta else ())
        produced: set[str] = set()
        donor_window = windows[0][1]
        for c, w in zip(copies, donor_window, strict=True):
            for n in _id_args(world, c):
                for v in _strings(c.args.get(n)):
                    assert v in produced or v in seen or v in pools, (c.tool_id, n, v)
                rebound += c.args.get(n) != w.args.get(n)
            produced |= set(c.response_meta.ids if c.response_meta else ())
    assert checked >= 30
    assert rebound >= checked // 2  # donor ids really are rewritten to this session's


def _call(agent: str, sess: str, seq: int, tool: str, args: dict, ids: tuple[str, ...]) -> RawCall:  # type: ignore[type-arg]
    return RawCall(
        envelope_id=f"{sess}-{seq}",
        session_id=sess,
        agent_id=agent,
        principal_id=f"p-{agent}",
        seq_no=seq,
        ts_unix_ms=START + 10_000 * seq,
        tool_id=tool,
        args=args,
        response_meta=ResponseMeta(bytes=100, ids=ids, fp=tuple(range(64)) if ids else None),
    )


def test_splice_rebinds_ids_consistently_and_echoes_them_in_responses(world: World) -> None:
    """A donor id used across the window maps to ONE id of this session, a response that echoed
    the donor id echoes the new one (its stale ``fp`` dropped, as T-29), and an id produced inside
    the window keeps its in-window producer."""
    donor = [
        _call("helpdesk-triage-a", "d", 0, "zendesk.get_ticket", {"ticket_id": "tk_D0N0R0000001"},
              ("tk_D0N0R0000001", "ch_D0N0R0000001")),
        _call("helpdesk-triage-a", "d", 1, "zendesk.add_comment",
              {"ticket_id": "tk_D0N0R0000001", "body": "hi", "public": True}, ("cm_N3W00000001",)),
        _call("helpdesk-triage-a", "d", 2, "zendesk.update_status",
              {"ticket_id": "tk_D0N0R0000001", "status": "solved"}, ("tk_D0N0R0000001",)),
    ]  # fmt: skip
    host = [
        _call(HOST, "h", 0, "zendesk.get_ticket", {"ticket_id": "tk_H05T00000001"},
              ("tk_H05T00000001", "ch_H0000000001")),
        _call(HOST, "h", 1, "slack.post_message", {"channel": "x", "text": "y"}, ()),
    ]  # fmt: skip
    ctx = AttackContext(
        world.catalog, world.persona(HOST), world.pools(HOST), "atk", [], 0,
        donors=[("helpdesk-triage-a", donor)],
    )  # fmt: skip
    r = SPLICE_ATTACKS["splice"](host, np.random.default_rng(0), ctx)
    assert r is not None
    copies = r.session[1:4]
    assert [c.tool_id for c in copies] == [c.tool_id for c in donor]
    assert [c.args["ticket_id"] for c in copies] == ["tk_H05T00000001"] * 3  # the only tk_ id seen
    assert copies[1].args["body"] == "hi" and copies[2].args["status"] == "solved"
    metas = [c.response_meta for c in copies]
    assert metas[0] is not None and metas[0].fp is None
    assert metas[0].ids == ("tk_H05T00000001", "ch_D0N0R0000001")
    assert metas[1] == donor[1].response_meta  # untouched: nothing echoed
    assert metas[2] is not None and metas[2].ids == ("tk_H05T00000001",) and metas[2].fp is None
    assert r.session[0] == host[0] and r.session[4].envelope_id == host[1].envelope_id


def test_splice_needs_donors_and_two_calls(
    world: World, by_persona: dict[str, list[list[RawCall]]]
) -> None:
    s = by_persona[HOST][0]
    no_donors = AttackContext(world.catalog, world.persona(HOST), world.pools(HOST), "a", [], 0)
    assert no_donors.donors == ()
    assert not applicable("splice", s, no_donors)
    assert SPLICE_ATTACKS["splice"](s, np.random.default_rng(0), no_donors) is None
    ctx = _ctx(world, HOST, by_persona)
    assert applicable("splice", s, ctx)
    assert not applicable("splice", s[:1], ctx)
    assert eligible("splice", world.persona(HOST), world.catalog)


def test_splice_is_opt_in() -> None:
    assert set(SPLICE_ATTACKS) == {"splice"}
    assert "splice" not in ATTACKS  # sorted(ATTACKS) is SW1's family list (T-28)
    assert "splice" not in RUNAWAY_FAMILIES  # a real attack: its sessions are "attacked" (T-40)
    assert "splice" not in GenerateConfig.model_fields["families"].default
    assert "splice" not in sw1_config().families
    assert "splice" not in CI_DATASET.families


def test_donors_are_never_built_when_splice_is_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_: object) -> list[tuple[str, list[RawCall]]]:
        raise AssertionError("donor_sessions called without splice")

    monkeypatch.setattr("agentfw.synth.generate.donor_sessions", boom)
    generate(CI_DATASET, tmp_path / "default")  # default CI families
    generate(CI_DATASET.model_copy(update={"families": ["loop", "spin", "stall"]}), tmp_path / "r")
    with pytest.raises(AssertionError, match="without splice"):
        generate(CI_DATASET.model_copy(update={"families": ["splice"]}), tmp_path / "s")


def _generated(tmp_path: Path, families: list[str]) -> tuple[list[RawCall], list[Label]]:
    cfg = CI_DATASET.model_copy(update={"families": families, "attack_rate": 0.5})
    generate(cfg, tmp_path)
    raws = [RawCall.model_validate(orjson.loads(x)) for x in (tmp_path / "traces.jsonl").open()]
    labels = [Label.model_validate(orjson.loads(x)) for x in (tmp_path / "labels.jsonl").open()]
    return raws, labels


def test_generate_splices_benign_windows_of_other_agents(world: World, tmp_path: Path) -> None:
    raws, labels = _generated(tmp_path / "s", ["splice"])
    benign, _ = _generated(tmp_path / "b", [])
    spliced = {lab.envelope_id: lab for lab in labels if lab.attack_family == "splice"}
    by_atk: dict[str, list[RawCall]] = defaultdict(list)
    for c in raws:
        if c.envelope_id in spliced:
            by_atk[str(spliced[c.envelope_id].attack_id)].append(c)
    assert len(by_atk) >= 15  # 22 at this size (only outreach-a and sales-ops-a have donors)

    def shape(calls: list[RawCall]) -> tuple[tuple[str, bytes], ...]:
        """Tools and non-id args: what a splice copies verbatim (ids are rebound)."""
        return tuple(
            (
                c.tool_id,
                orjson.dumps(
                    {n: v for n, v in c.args.items() if n not in _id_args(world, c)},
                    option=orjson.OPT_SORT_KEYS,
                ),
            )
            for c in calls
        )

    benign_sessions: dict[str, list[RawCall]] = defaultdict(list)
    for c in benign:
        benign_sessions[c.session_id].append(c)
    windows = {
        (s[0].agent_id, shape(s[k : k + SPLICE_LEN]))
        for s in benign_sessions.values()
        for k in range(len(s) - SPLICE_LEN + 1)
    }
    for calls in by_atk.values():
        assert len(calls) == SPLICE_LEN
        agent = calls[0].agent_id
        assert all(c.task_ids == () and c.agent_id == agent for c in calls)
        # a benign window of some OTHER agent, taken before any injection
        assert any(a != agent and w == shape(calls) for a, w in windows)
    again, again_labels = _generated(tmp_path / "s2", ["splice"])
    assert again == raws and again_labels == labels  # deterministic
