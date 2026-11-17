"""Generator ground truth for the session ledger: task ids, fingerprints, producer edges."""

from __future__ import annotations

import numpy as np
import orjson
import pytest

from agentfw.core.envelope import RawCall
from agentfw.core.fingerprint import resp_fingerprint
from agentfw.core.normalize import SessionTracker, normalize
from agentfw.core.patterns import is_id_like
from agentfw.synth.grammar import PlannedCall, walk
from agentfw.synth.sampler import sample_persona
from tests.synth.conftest import World

Edge = tuple[str, int, str, int]  # (session, consumer seq, flat arg path, producer seq)


def _raw(pc: PlannedCall, sid: str, seq: int, agent: str, task_ids: tuple[str, ...]) -> RawCall:
    return RawCall(
        envelope_id=f"e{sid}-{seq}",
        session_id=sid,
        agent_id=agent,
        principal_id="p",
        seq_no=seq,
        ts_unix_ms=1_700_000_000_000 + seq,
        tool_id=pc.tool_id,
        args=pc.args,
        response_meta=pc.response_meta,
        task_ids=task_ids,
    )


def _walks(world: World, n: int = 300) -> list[tuple[str, list[PlannedCall]]]:
    """``n`` benign walks over every persona (id_flow_rate forced to 1.0)."""
    out: list[tuple[str, list[PlannedCall]]] = []
    ids = sorted(world.personas)
    for k in range(n):
        agent = ids[k % len(ids)]
        persona = world.persona(agent).model_copy(update={"id_flow_rate": 1.0})
        rng = np.random.default_rng(k)
        names = [t.name for t in persona.tasks if t.weight > 0]
        task = world.tasks[names[int(rng.integers(len(names)))]]
        planned = walk(task, persona, world.catalog, rng, world.pools(agent))
        if planned:
            out.append((agent, planned))
    return out


def edge_precision_recall(walks: list[tuple[str, list[PlannedCall]]]) -> tuple[float, float]:
    expected: set[Edge] = set()
    found: set[Edge] = set()
    tracker = SessionTracker()
    for n, (agent, planned) in enumerate(walks):
        sid = f"s{n}"
        seeds = tuple(dict.fromkeys(x for pc in planned for x in pc.external_ids))
        for seq, pc in enumerate(planned):
            expected.update((sid, seq, path, src) for path, src in pc.sources)
            env = normalize(_raw(pc, sid, seq, agent, seeds if seq == 0 else ()), tracker)
            found.update(
                (sid, seq, r.path, r.producer_seq)
                for r in env.ledger_refs
                if r.source == "response"
            )
            tracker.observe(env)
    hit = len(expected & found)
    return hit / len(found), hit / len(expected)


def test_ledger_recovers_grammar_edges(world: World) -> None:  # F1 acceptance
    walks = _walks(world)
    assert sum(len(pc.sources) for _, w in walks for pc in w) > 50  # the check is not vacuous
    p, r = edge_precision_recall(walks)
    assert p >= 0.99 and r >= 0.99, (p, r)


def test_sources_point_at_emitted_producers(world: World) -> None:
    for _, planned in _walks(world, 100):
        for i, pc in enumerate(planned):
            for path, src in pc.sources:
                assert 0 <= src < i and path
                assert planned[src].response_meta is not None


@pytest.fixture(scope="module")
def captured(world: World) -> tuple[list[list[RawCall]], list[list[PlannedCall]]]:
    plans: list[list[PlannedCall]] = []
    orig = walk

    def spy(*a: object, **k: object) -> list[PlannedCall]:
        out = orig(*a, **k)  # type: ignore[arg-type]
        plans.append(out)
        return out

    mp = pytest.MonkeyPatch()
    mp.setattr("agentfw.synth.sampler.walk", spy)
    try:
        sessions = sample_persona(
            world.persona("support-refunds-a"),
            world.catalog,
            world.tasks,
            days=10,
            global_seed=3,
            start_ms=1_700_000_000_000,
        )
    finally:
        mp.undo()
    return sessions, [p for p in plans if p]


def test_external_ids_seed_first_call(
    captured: tuple[list[list[RawCall]], list[list[PlannedCall]]],
) -> None:
    sessions, plans = captured
    assert len(sessions) == len(plans) > 5
    seeded = 0
    for calls, planned in zip(sessions, plans, strict=True):
        externals = [i for pc in planned for i in pc.external_ids]
        assert calls[0].task_ids == tuple(dict.fromkeys(externals))[:64]
        assert all(c.task_ids == () for c in calls[1:])
        assert all(is_id_like(i) for i in calls[0].task_ids)
        seeded += bool(externals)
    assert seeded > 0


def test_fp_is_resp_fingerprint_of_sorted_payload(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[str, int | None, bool]] = []

    def spy(text: str, item_count: int | None, error: bool) -> tuple[int, ...] | None:
        seen.append((text, item_count, error))
        return resp_fingerprint(text, item_count, error)

    monkeypatch.setattr("agentfw.synth.grammar.resp_fingerprint", spy)
    planned = walk(
        world.tasks["small_refund"],
        world.persona("support-refunds-a"),
        world.catalog,
        np.random.default_rng(1),
        world.pools("support-refunds-a"),
    )
    assert planned and len(seen) >= len(planned)
    for text, _, error in seen:
        payload = orjson.loads(text)
        assert text == orjson.dumps(payload, option=orjson.OPT_SORT_KEYS).decode()
        assert error is False
    for pc in planned:
        meta = pc.response_meta
        assert meta is not None
        text, item_count, _ = next(x for x in seen if len(x[0]) + 64 == meta.bytes)
        assert meta.fp == resp_fingerprint(text, item_count, False)


def test_no_extra_rng_draws(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    def run() -> list[dict[str, object]]:
        ss = sample_persona(
            world.persona("helpdesk-triage-a"),
            world.catalog,
            world.tasks,
            days=8,
            global_seed=11,
            start_ms=1_700_000_000_000,
        )
        rows: list[dict[str, object]] = []
        for s in ss:
            for c in s:
                d = c.model_dump(mode="json")
                d.pop("task_ids", None)
                if isinstance(d["response_meta"], dict):
                    d["response_meta"].pop("fp", None)
                rows.append(d)
        return rows

    with_fp = run()
    monkeypatch.setattr("agentfw.synth.grammar.resp_fingerprint", lambda *a, **k: None)
    without_fp = run()
    assert with_fp == without_fp and len(with_fp) > 20


def test_pool_drawn_ids_are_seeded_even_when_catalog_source_is_a_producer(world: World) -> None:
    # helpdesk tasks draw every ticket_id from the persona pool (``ext:tickets``) although the
    # catalog says zendesk's ticket_id comes from a producer; they are still task inputs.
    sessions = sample_persona(
        world.persona("helpdesk-triage-b"),
        world.catalog,
        world.tasks,
        days=10,
        global_seed=3,
        start_ms=1_700_000_000_000,
    )
    assert len(sessions) > 20
    writes = 0
    for calls in sessions:
        seeds = set(calls[0].task_ids)
        for c in calls:
            if c.tool_id in ("zendesk.get_ticket", "zendesk.add_comment", "zendesk.update_status"):
                assert c.args["ticket_id"] in seeds
                writes += c.tool_id != "zendesk.get_ticket"
    assert writes > 20


@pytest.mark.parametrize("rate", [0.0, 0.5])
def test_sources_name_a_producer_that_returned_the_value(world: World, rate: float) -> None:
    """A fabricated id (``id_flow_rate < 1``) is never credited to the bound alias's producer,
    neither directly nor through a later call that echoes it."""
    from agentfw.core.normalize import flatten

    checked = 0
    ids = sorted(world.personas)
    for k in range(300):
        agent = ids[k % len(ids)]
        persona = world.persona(agent).model_copy(update={"id_flow_rate": rate})
        rng = np.random.default_rng(k)
        names = [t.name for t in persona.tasks if t.weight > 0]
        task = world.tasks[names[int(rng.integers(len(names)))]]
        planned = walk(task, persona, world.catalog, rng, world.pools(agent))
        for pc in planned:
            flat = flatten(pc.args)[0]
            for path, src in pc.sources:
                meta = planned[src].response_meta
                assert meta is not None
                vals = [v for v in flat[path] if isinstance(v, str)]
                assert any(v in meta.ids for v in vals), (pc.tool_id, path, vals)
                checked += 1
    assert checked > 0

