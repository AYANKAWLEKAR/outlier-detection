"""Task templates + agent definitions: schema, coverage, deterministic rendering (Task 2.6)."""

from __future__ import annotations

import re
from pathlib import Path
from statistics import mean

import numpy as np
import pytest
import yaml

from agentfw.sandbox.seed.company import build_company
from agentfw.sandbox.tasks.schema import (
    AGENTS_DIR,
    TASKS_DIR,
    AgentDef,
    TaskTemplate,
    load_agents,
    load_templates,
    render,
    render_session,
)

COMPANY = build_company(0)
PERSONAS = Path(__file__).parents[2] / "agentfw" / "synth" / "personas"
KNOWN_TOOLS = {
    "zendesk.get_ticket",
    "zendesk.list_tickets",
    "zendesk.add_comment",
    "zendesk.update_status",
    "stripe.retrieve_charge",
    "stripe.list_charges",
    "stripe.create_refund",
    "stripe.retrieve_customer",
    "stripe.create_payout",
    "salesforce.run_soql",
    "salesforce.get_record",
    "salesforce.update_record",
    "salesforce.mass_update",
    "salesforce.create_task",
    "mail.send_email",
    "slack.post_message",
    "slack.read_channel",
}
CHECK_PARAMS = {
    "mailhog_has_message": {"to", "subject_contains"},
    "stripe_refund_exists": {"charge_id"},
    "sf_field_equals": {"object", "record_id", "field", "value"},
    "slack_contains": {"text"},
    "zendesk_status": {"ticket_id", "status"},
}
FORBIDDEN_WORDS = re.compile(r"firewall|attack|inject|exfiltrat|malicious|anomal", re.IGNORECASE)


@pytest.fixture(scope="module")
def templates() -> dict[str, TaskTemplate]:
    return load_templates(TASKS_DIR)


@pytest.fixture(scope="module")
def agents() -> list[AgentDef]:
    return load_agents(AGENTS_DIR)


def test_at_least_70_templates_over_6_groups(templates: dict[str, TaskTemplate]) -> None:
    assert len(templates) >= 70
    assert len(list(TASKS_DIR.glob("*.yaml"))) == 6


def test_agent_ids_mirror_synth_roster(agents: list[AgentDef]) -> None:
    roster = {yaml.safe_load(p.read_text())["agent_id"] for p in PERSONAS.glob("*.yaml")}
    assert len(agents) == 20
    assert {a.agent_id for a in agents} == roster


def test_roles_exist_and_tools_covered(
    templates: dict[str, TaskTemplate], agents: list[AgentDef]
) -> None:
    by_id = {a.agent_id: a for a in agents}
    for t in templates.values():
        assert t.roles, t.id
        for role in t.roles:
            assert role in by_id, (t.id, role)
            assert set(t.expected_tools) <= set(by_id[role].tools), (t.id, role)
        assert set(t.expected_tools) <= KNOWN_TOOLS, t.id


def test_agents_well_formed(templates: dict[str, TaskTemplate], agents: list[AgentDef]) -> None:
    for a in agents:
        assert set(a.tools) <= KNOWN_TOOLS, a.agent_id
        assert len(a.system_prompt) > 80
        assert a.sessions_per_day > 0 and a.batch_size >= 1
        mine = {t.id for t in templates.values() if a.agent_id in t.roles}
        assert len(mine) >= 3, a.agent_id
        assert a.template_mix and set(a.template_mix) <= mine, a.agent_id
        assert all(w > 0 for w in a.template_mix.values())


def test_template_ids_prompts_unique(templates: dict[str, TaskTemplate]) -> None:
    prompts = [t.prompt for t in templates.values()]
    assert len(set(prompts)) == len(prompts)
    sigs = [
        (tuple(t.expected_tools), tuple(sorted(s.kind for s in t.slots.values())))
        for t in templates.values()
    ]
    # many templates may share a tool chain, but never more than a handful
    assert max(sigs.count(s) for s in set(sigs)) <= 4


def test_success_checks_valid(templates: dict[str, TaskTemplate]) -> None:
    for t in templates.values():
        for s in t.success:
            assert s.kind in CHECK_PARAMS, (t.id, s.kind)
            assert set(s.params) == CHECK_PARAMS[s.kind], (t.id, s.kind)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_render_resolves_every_slot(templates: dict[str, TaskTemplate], seed: int) -> None:
    company = build_company(seed)
    for t in templates.values():
        r = render(t, company, np.random.default_rng([seed, 5]))
        assert not re.search(r"[{}]", r.prompt), (t.id, r.prompt)
        assert set(r.slot_values) == set(t.slots)
        assert r.expected_tools == t.expected_tools
        for s in r.success:
            assert all(v and not re.search(r"[{}]", v) for v in s.params.values()), t.id
        assert not FORBIDDEN_WORDS.search(r.prompt), t.id
        for email in re.findall(r"[\w.+-]+@[\w.-]+", r.prompt):
            assert email.rstrip(".").endswith((".example", ".test")), (t.id, email)


def test_render_deterministic(templates: dict[str, TaskTemplate]) -> None:
    for t in templates.values():
        a = render(t, COMPANY, np.random.default_rng(11))
        b = render(t, COMPANY, np.random.default_rng(11))
        assert a == b


def test_render_varies_with_rng() -> None:
    t = load_templates(TASKS_DIR)
    seen = {
        render(t["support_refund_full"], COMPANY, np.random.default_rng(i)).prompt for i in range(6)
    }
    assert len(seen) > 1


def test_filter_applies() -> None:
    t = load_templates(TASKS_DIR)["support_refund_full"]
    r = render(t, COMPANY, np.random.default_rng(3))
    ticket = r.slot_values["ticket"]
    assert "refund" in ticket["subject"].lower()
    assert ticket["id"].startswith("tk_")
    assert ticket["charge_id"].startswith("ch_")


def test_render_session_batches_numbered_queue(
    templates: dict[str, TaskTemplate], agents: list[AgentDef]
) -> None:
    for a in agents:
        s1 = render_session(a, templates, COMPANY, np.random.default_rng(5))
        s2 = render_session(a, templates, COMPANY, np.random.default_rng(5))
        assert s1 == s2
        for k in range(40):  # queues never repeat a primary record, whatever the draw
            ss = render_session(a, templates, COMPANY, np.random.default_rng(k))
            keys = [next(iter(t.slot_values.values()))["key"] for t in ss.tasks]
            assert len(set(keys)) == len(keys), (a.agent_id, k, keys)
        assert len(s1.tasks) == a.batch_size
        if a.batch_size > 1:
            for i in range(1, a.batch_size + 1):
                assert re.search(rf"^{i}\. ", s1.prompt, re.MULTILINE), (a.agent_id, i)
        # a queue never repeats the same primary record
        ids = [t.slot_values and next(iter(t.slot_values.values())).get("key") for t in s1.tasks]
        assert len(set(ids)) == len(ids) or a.batch_size == 1
        assert s1.expected_tools == [x for t in s1.tasks for x in t.expected_tools]
        assert s1.max_steps >= len(s1.expected_tools)


def test_r1_sizing_mean_tool_calls_per_session(
    templates: dict[str, TaskTemplate], agents: list[AgentDef]
) -> None:
    per_agent = []
    for a in agents:
        tot = sum(a.template_mix.values())
        mean_len = sum(
            w / tot * len(templates[i].expected_tools) for i, w in a.template_mix.items()
        )
        per_agent.append((a.agent_id, a.batch_size, mean_len))
    assert mean(b for _, b, _ in per_agent) * mean(m for _, _, m in per_agent) >= 18
    for agent_id, b, m in per_agent:  # and no agent is starved of work
        assert b * m >= 12, agent_id
