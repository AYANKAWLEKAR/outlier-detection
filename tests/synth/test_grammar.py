from __future__ import annotations

import re

import numpy as np
from numpy.typing import NDArray

from agentfw.core.patterns import is_id_like
from agentfw.synth.dists import IdPool
from agentfw.synth.grammar import PlannedCall, walk
from agentfw.synth.schema import Branches, DistSpec, Persona
from tests.synth.conftest import World

Cache = dict[str, NDArray[np.float64]]
_POOLS: dict[str, dict[str, IdPool]] = {}  # building 5k-id pools per walk would dominate runtime


def _persona(world: World, pid: str, **noise: float) -> Persona:
    p = world.persona(pid)
    return p.model_copy(update={"noise": Branches(**{**p.noise.model_dump(), **noise})})


def _run(
    world: World,
    task: str,
    pid: str,
    seed: int = 1,
    cache: Cache | None = None,
    pools: dict[str, IdPool] | None = None,
    **noise: float,
) -> list[PlannedCall]:
    if pools is None:
        if pid not in _POOLS:
            _POOLS[pid] = world.pools(pid)
        pools = _POOLS[pid]
    noise.setdefault("arg_dropout", 0.0)  # keep arguments intact unless a test asks for dropout
    return walk(
        world.tasks[task],
        _persona(world, pid, **noise),
        world.catalog,
        np.random.default_rng(seed),
        pools,
        weights_cache=cache,
    )


def _ids(c: PlannedCall) -> tuple[str, ...]:
    assert c.response_meta is not None
    return c.response_meta.ids


def test_small_refund_order_and_id_flow(world: World) -> None:
    calls = walk(
        world.tasks["small_refund"],
        world.persona("support-refunds-a"),
        world.catalog,
        np.random.default_rng(1),
        world.pools("support-refunds-a"),
    )
    tools = [c.tool_id for c in calls]
    assert tools[0] == "zendesk.get_ticket" and "stripe.create_refund" in tools
    refund = next(c for c in calls if c.tool_id == "stripe.create_refund")
    assert calls[0].response_meta is not None
    assert refund.args["charge_id"] in calls[0].response_meta.ids  # id flows from ticket


def test_ticket_id_comes_from_pool_and_ids_are_id_like(world: World) -> None:
    pools = world.pools("support-refunds-a")
    calls = walk(
        world.tasks["small_refund"],
        world.persona("support-refunds-a"),
        world.catalog,
        np.random.default_rng(1),
        pools,
    )
    assert calls[0].args["ticket_id"] in pools["tickets"].ids
    assert all(is_id_like(i) for c in calls for i in _ids(c))


def test_skip_lookup_rate_matches_branch(world: World) -> None:
    rng = np.random.default_rng(2)
    p = _persona(world, "support-refunds-a", abandon=0.0)
    pools = world.pools("support-refunds-a")
    skipped = sum(
        "stripe.retrieve_charge"
        not in [c.tool_id for c in walk(world.tasks["small_refund"], p, world.catalog, rng, pools)]
        for _ in range(10_000)
    )
    assert abs(skipped / 10_000 - 0.05) < 0.01


def test_skipped_step_still_feeds_later_bindings(world: World) -> None:
    calls = _run(
        world, "small_refund", "support-refunds-a", skip_lookup=1.0, abandon=0.0, retry=0.0
    )
    assert [c.tool_id for c in calls] == [
        "zendesk.get_ticket",
        "stripe.create_refund",
        "mail.send_email",
    ]
    assert calls[1].args["charge_id"] in _ids(calls[0])


def test_retry_duplicates_args(world: World) -> None:
    calls = _run(world, "small_refund", "support-refunds-a", retry=1.0, abandon=0.0, skip_lookup=0)
    assert len(calls) % 2 == 0
    for first, second in zip(calls[0::2], calls[1::2], strict=True):
        assert first.tool_id == second.tool_id and first.args == second.args
        assert first.args is not second.args  # independent copy


def test_abandon_truncates_after_uniform_step(world: World) -> None:
    rng = np.random.default_rng(3)
    p = _persona(world, "support-refunds-a", abandon=1.0, retry=0.0, skip_lookup=0.0)
    pools = world.pools("support-refunds-a")
    lens = [
        len(walk(world.tasks["small_refund"], p, world.catalog, rng, pools)) for _ in range(4_000)
    ]
    for n in (1, 2, 3, 4):
        assert abs(lens.count(n) / 4_000 - 0.25) < 0.03
    assert max(lens) == 4


def test_no_noise_walk_is_complete_and_never_truncated(world: World) -> None:
    for seed in range(20):
        calls = _run(
            world, "small_refund", "support-refunds-a", seed, skip_lookup=0, abandon=0, retry=0
        )
        assert len(calls) == 4


def test_arg_dropout_removes_only_non_id_args(world: World) -> None:
    calls = _run(world, "escalate", "support-refunds-a", arg_dropout=1.0, abandon=0.0, retry=0.0)
    add_comment = next(c for c in calls if c.tool_id == "zendesk.add_comment")
    assert set(add_comment.args) == {"ticket_id"}
    assert "ticket_id" in calls[0].args


def test_produced_ticket_id_echoes_argument(world: World) -> None:
    calls = _run(world, "escalate", "support-refunds-a", abandon=0, retry=0)
    ticket_id = calls[0].args["ticket_id"]
    assert ticket_id in _ids(calls[0])
    assert calls[1].args["ticket_id"] == ticket_id == calls[2].args["ticket_id"]
    assert ticket_id in _ids(calls[2])  # update_status.ticket.id echoes ticket_id


def test_bool_and_freetext_shapes(world: World) -> None:
    comments = [
        c
        for s in range(300)
        for c in _run(world, "escalate", "support-refunds-a", s, abandon=0, retry=0)
        if c.tool_id == "zendesk.add_comment"
    ]
    assert all(isinstance(c.args["public"], bool) for c in comments)
    assert abs(np.mean([c.args["public"] for c in comments]) - 0.3) < 0.08
    bodies = [c.args["body"] for c in comments]
    assert all(isinstance(b, str) and len(b) >= 5 for b in bodies)
    assert any(len(str(b)) >= 40 for b in bodies)


def test_refund_amount_follows_persona_lognormal(world: World) -> None:
    amounts = [
        float(str(c.args["amount"]))
        for s in range(1_500)
        for c in _run(world, "small_refund", "support-refunds-a", s)
        if c.tool_id == "stripe.create_refund" and "amount" in c.args
    ]
    assert abs(np.median(amounts) / np.exp(3.6) - 1) < 0.08  # persona mu 3.6, not catalog 3.4
    assert all(abs(a * 100 - round(a * 100)) < 1e-6 for a in amounts)


def test_email_list_bound_value_first_then_persona_extras(world: World) -> None:
    domain = re.compile(r"^[^@]+@[a-z]+\d+\.example$")
    lens: list[int] = []
    domains: set[str] = set()
    for s in range(1_500):
        calls = _run(world, "followup_email", "support-refunds-a", s, abandon=0, retry=0)
        to = calls[1].args["to"]
        assert isinstance(to, list) and to
        assert all(isinstance(e, str) and domain.match(e) for e in to)
        lens.append(len(to))
        domains.update(str(e).split("@")[1] for e in to)
    assert min(lens) == 1 and abs(np.mean(lens) - 1.5) < 0.1  # E[max(1, Poisson(1.2))]
    assert len(domains) <= 12


def test_bound_email_is_first_list_element(world: World) -> None:
    catalog = {k: v.model_copy(deep=True) for k, v in world.catalog.items()}
    catalog["zendesk.get_ticket"].response.produces_ids.append("ticket.customer_email")
    for s in range(50):
        calls = walk(
            world.tasks["followup_email"],
            _persona(world, "support-refunds-a", abandon=0, retry=0, arg_dropout=0),
            catalog,
            np.random.default_rng(s),
            world.pools("support-refunds-a"),
        )
        to = calls[1].args["to"]
        assert isinstance(to, list) and to[0] in _ids(calls[0])


def test_persona_domain_override_applies_to_all_emails(world: World) -> None:
    outreach: set[str] = set()
    for s in range(800):
        calls = _run(world, "cold_outreach", "outreach-a", s, abandon=0, retry=0)
        for c in calls:
            if c.tool_id == "mail.send_email":
                to = c.args["to"]
                assert isinstance(to, list)
                outreach.update(str(e).split("@")[1] for e in to)
    assert 12 < len(outreach) <= 40  # persona zipf(40), not the catalog zipf(12)


def test_outreach_list_length_override(world: World) -> None:
    lens = [
        len(c.args["to"])  # type: ignore[arg-type]
        for s in range(600)
        for c in _run(world, "cold_outreach", "outreach-a", s, abandon=0, retry=0)
        if c.tool_id == "mail.send_email"
    ]
    assert min(lens) >= 1 and 2.3 < np.mean(lens) < 3.0


def test_item_count_equals_id_list_length(world: World) -> None:
    for s in range(30):
        calls = _run(world, "bulk_stage_update", "sales-ops-a", s, abandon=0, retry=0)
        q, upd = calls[0], calls[1]
        assert q.response_meta is not None and q.response_meta.item_count is not None
        assert q.response_meta.item_count == len(_ids(q))
        assert upd.args["record_ids"] == list(_ids(q))


def test_ids_truncated_to_256_but_item_count_kept(world: World) -> None:
    catalog = {k: v.model_copy(deep=True) for k, v in world.catalog.items()}
    catalog["salesforce.run_soql"].response.item_count = DistSpec(family="constant", value=1000)
    calls = walk(
        world.tasks["bulk_stage_update"],
        _persona(world, "sales-ops-a", abandon=0, retry=0),
        catalog,
        np.random.default_rng(1),
        world.pools("sales-ops-a"),
    )
    meta = calls[0].response_meta
    assert meta is not None and meta.item_count == 1000 and len(meta.ids) == 256
    assert len(calls[1].args["record_ids"]) == 1000  # type: ignore[arg-type]


def test_frac_binding_scales_bound_amount(world: World) -> None:
    for s in range(20):
        calls = _run(world, "reprice_opportunity", "sales-ops-a", s, abandon=0, retry=0)
        assert calls[1].tool_id == "salesforce.update_record"
        assert calls[1].args["record_id"] == calls[0].args["record_id"]
        assert isinstance(calls[1].args["amount"], float)


def test_repeat_multiplies_steps(world: World) -> None:
    n = [
        len(_run(world, "cold_outreach", "outreach-a", s, abandon=0, retry=0, skip_lookup=0))
        for s in range(400)
    ]
    assert all(x % 2 == 0 and x >= 2 for x in n)
    assert abs(np.mean(n) / 2 - 4.0) < 0.5


def test_gap_and_response_meta(world: World) -> None:
    calls = _run(
        world,
        "small_refund",
        "support-refunds-a",
        abandon=0,
        retry=0,
        skip_lookup=0,
        pools=world.pools("support-refunds-a"),
    )
    assert calls[0].gap_s == 0.0 and all(c.gap_s > 0 for c in calls[1:])
    for c in calls:
        m = c.response_meta
        assert m is not None and m.error is False and m.bytes is not None and m.bytes > 0
    again = _run(
        world,
        "small_refund",
        "support-refunds-a",
        abandon=0,
        retry=0,
        skip_lookup=0,
        pools=world.pools("support-refunds-a"),
    )
    assert [c.response_meta for c in calls] == [c.response_meta for c in again]


def test_deterministic_and_weights_cache_shared_across_walks(world: World) -> None:
    a = [
        _run(world, "escalate", "support-refunds-a", 5, pools=world.pools("support-refunds-a"))
        for _ in range(2)
    ]
    assert a[0] == a[1]
    cache: Cache = {}
    _run(world, "escalate", "support-refunds-a", 1, cache=cache, abandon=0, retry=0)
    snapshot = {k: v.copy() for k, v in cache.items()}
    assert snapshot  # categorical weights were drawn
    for s in range(2, 20):
        _run(world, "escalate", "support-refunds-a", s, cache=cache, abandon=0, retry=0)
    assert all(np.array_equal(snapshot[k], cache[k]) for k in snapshot)
