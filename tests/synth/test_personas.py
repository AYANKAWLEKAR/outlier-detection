"""The 20-persona benchmark roster (Task 1.13, spec 4.3)."""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from agentfw.synth.attacks import ATTACKS, eligible
from agentfw.synth.generate import GenerateConfig, generate
from agentfw.synth.load import (
    CATALOG_DIR,
    PERSONAS_DIR,
    TASKS_DIR,
    TIER0_PERSONA_IDS,
    load_catalog,
    load_personas,
    load_tasks,
    validate_world,
)
from agentfw.synth.sampler import sample_persona
from agentfw.synth.schema import Persona

from .conftest import World

START = 1_767_225_600_000

ROSTER = {
    "support-refunds-a", "support-refunds-b", "support-refunds-c", "billing-disputes-a",
    "payouts-ops-a", "sales-ops-a", "sales-ops-b", "crm-hygiene-a", "lead-router-a",
    "renewals-a", "outreach-a", "outreach-b", "newsletter-a", "slack-notifier-a",
    "helpdesk-triage-a", "helpdesk-triage-b", "escalations-a", "finance-reconcile-a",
    "data-export-a", "exec-assistant-a",
}  # fmt: skip


def test_roster_loads_with_unique_ids(world: World) -> None:
    ps = load_personas()
    assert len(ps) == 20
    assert {p.agent_id for p in ps} == ROSTER
    assert len({p.agent_id for p in ps}) == 20
    validate_world(world.catalog, ps, world.tasks)  # raises WorldError on any problem


@pytest.mark.parametrize("family", sorted(ATTACKS))
def test_every_family_has_three_eligible_personas(world: World, family: str) -> None:
    n = [p.agent_id for p in world.personas.values() if eligible(family, p, world.catalog)]
    assert len(n) >= 3, (family, n)


def test_at_least_three_personas_have_zero_id_flow(world: World) -> None:
    zero = [p for p in world.personas.values() if p.id_flow_rate == 0]
    assert len(zero) >= 3
    for p in zero:  # their writes take ids from external pools, not from the session
        for tw in p.tasks:
            for st in world.tasks[tw.name].steps:
                if world.catalog[st.tool].kind != "write":
                    continue
                for arg, b in st.args.items():
                    if world.catalog[st.tool].args[arg].type == "id":
                        assert b.startswith("ext:"), (p.agent_id, tw.name, st.tool, arg, b)


def test_at_least_three_personas_have_eight_tools(world: World) -> None:
    assert sum(len(p.tools) >= 8 for p in world.personas.values()) >= 3
    assert len(world.persona("exec-assistant-a").tools) >= 10


def _amount_medians(world: World) -> list[float]:
    out: list[float] = []
    for p in world.personas.values():
        for t in p.tools:
            for name, a in world.catalog[t].args.items():
                if world.catalog[t].kind != "write" or a.shape != "money":
                    continue
                d = p.fields.get(f"{t}.{name}") or a.dist
                if d is not None and d.family == "lognormal" and d.mu is not None:
                    out.append(math.exp(d.mu))
    return out


def test_amount_medians_span_an_order_of_magnitude(world: World) -> None:
    med = _amount_medians(world)
    assert max(med) / min(med) >= 10


def test_refund_amount_means_per_spec(world: World) -> None:
    for pid, mu in (("support-refunds-a", 3.6), ("support-refunds-b", 4.5)) + (
        ("support-refunds-c", 2.8),
    ):
        assert world.persona(pid).fields["stripe.create_refund.amount"].mu == mu


def test_payout_destination_is_a_stable_four_account_vocab(world: World) -> None:
    dest = world.catalog["stripe.create_payout"].args["destination"]
    vocab = dest.vocab or []
    assert dest.type == "categorical" and len(vocab) == 4
    assert all(v.startswith("acct_") for v in vocab)
    assert "stripe.create_payout" in world.persona("payouts-ops-a").tools


def test_specialty_personas(world: World) -> None:
    crm = world.persona("crm-hygiene-a")
    qe = {t.name: t.weight for t in crm.tasks}["quarter_end_mass_update"]
    assert qe == 0.0 and crm.quarter_end_days
    assert world.persona("newsletter-a").fields["mail.send_email.to.__len__"].lam >= 20  # type: ignore[operator]
    assert "stripe.list_charges" in world.persona("finance-reconcile-a").tools
    de = world.persona("data-export-a")
    assert {"salesforce.run_soql", "mail.send_email"} <= set(de.tools)


def test_every_persona_yields_a_session_per_day(world: World) -> None:
    for p in world.personas.values():
        s = sample_persona(p, world.catalog, world.tasks, days=30, global_seed=0, start_ms=START)
        assert len(s) >= 30, (p.agent_id, len(s))


def test_only_reserved_domains_in_persona_and_task_files() -> None:
    bad = re.compile(r"\b[\w-]+\.(com|org|net|io|co|dev|app)\b")
    for d in (PERSONAS_DIR, TASKS_DIR, CATALOG_DIR):
        for f in d.glob("*.yaml"):
            assert not bad.search(f.read_text()), f


def test_persona_ids_filter_selects_the_tier0_dataset(tmp_path: Path) -> None:
    assert TIER0_PERSONA_IDS == ("outreach-a", "sales-ops-a", "support-refunds-a")
    cfg = GenerateConfig(
        seed=0,
        personas_dir=PERSONAS_DIR,
        tasks_dir=TASKS_DIR,
        catalog_dir=CATALOG_DIR,
        days=2,
        persona_ids=list(TIER0_PERSONA_IDS),
    )
    generate(cfg, tmp_path)
    agents = {
        line.split('"agent_id":"')[1].split('"')[0]
        for line in (tmp_path / "traces.jsonl").read_text().splitlines()
    }
    assert agents <= set(TIER0_PERSONA_IDS) and len(agents) == 3
    bad = cfg.model_copy(update={"persona_ids": ["nope-a"]})
    with pytest.raises(ValueError, match="nope-a"):
        generate(bad, tmp_path / "x")


def test_load_helpers_unused_import_guard() -> None:
    assert isinstance(load_catalog(), dict) and isinstance(load_tasks(), dict)
    assert all(isinstance(p, Persona) for p in load_personas())
