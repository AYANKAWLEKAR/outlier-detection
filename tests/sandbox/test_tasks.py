"""Task templates + agent definitions: schema, coverage, deterministic rendering (Task 2.6)."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from statistics import mean

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

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
    validate_catalog,
)
from agentfw.sandbox.tools import (
    mail_server,
    salesforce_server,
    slack_server,
    stripe_server,
    zendesk_server,
)
from agentfw.sandbox.tools.salesforce_server import sf_id
from tests.sandbox.tools.conftest import tool_names

COMPANY = build_company(0)
PERSONAS = Path(__file__).parents[2] / "agentfw" / "synth" / "personas"


def _registry_tools() -> set[str]:
    """Tool ids straight from the five servers' registries (the source of truth)."""
    servers = [
        zendesk_server.build_server(zendesk_server.FakeZendeskBackend(COMPANY)),
        stripe_server.build_server(stripe_server.FakeStripeBackend(COMPANY)),
        salesforce_server.build_server(salesforce_server.FakeSalesforceBackend(COMPANY)),
        mail_server.build_server(mail_server.FakeMailBackend()),
        slack_server.build_server(slack_server.FakeSlackBackend("C1")),
    ]
    out: set[str] = set()
    for srv in servers:
        names = asyncio.run(tool_names(srv))
        out |= {f"{srv.name}.{n}" for n in names}
    return out


KNOWN_TOOLS = _registry_tools()
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
    assert len(list(TASKS_DIR.glob("*.yaml"))) == 6  # support, billing, sales, outreach, ...


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
        assert all(w >= 0 for w in a.template_mix.values())


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


def _personas() -> dict[str, dict]:  # type: ignore[type-arg]
    out = {}
    for p in PERSONAS.glob("*.yaml"):
        d = yaml.safe_load(p.read_text())
        out[d["agent_id"]] = d
    return out


def test_catalog_consistent(templates: dict[str, TaskTemplate], agents: list[AgentDef]) -> None:
    validate_catalog(templates, agents)


def test_agent_tools_within_persona_and_all_used(
    templates: dict[str, TaskTemplate], agents: list[AgentDef]
) -> None:
    personas = _personas()
    for a in agents:
        needed = {
            tool for t in templates.values() if a.agent_id in t.roles for tool in t.expected_tools
        }
        assert set(a.tools) <= set(personas[a.agent_id]["tools"]) | needed, a.agent_id
        assert set(a.tools) == needed, (a.agent_id, set(a.tools) ^ needed)


def test_template_mix_follows_persona_task_weights(
    templates: dict[str, TaskTemplate], agents: list[AgentDef]
) -> None:
    personas = _personas()
    for a in agents:
        weights = {t["name"]: float(t["weight"]) for t in personas[a.agent_id]["tasks"]}
        assert set(a.persona_tasks) == set(weights), a.agent_id
        mapped = [tid for tids in a.persona_tasks.values() for tid in tids]
        assert sorted(mapped) == sorted(set(mapped)) and set(mapped) == set(a.template_mix)
        for task, tids in a.persona_tasks.items():
            total = sum(a.template_mix[tid] for tid in tids)
            assert abs(total - weights[task]) < 1e-6, (a.agent_id, task, total)
            per = {a.template_mix[tid] for tid in tids}
            assert max(per) - min(per) < 1e-9, (a.agent_id, task)  # split evenly


def test_quarter_end_templates_have_zero_normal_weight(
    templates: dict[str, TaskTemplate], agents: list[AgentDef]
) -> None:
    flagged = {t.id for t in templates.values() if t.quarter_end_only}
    assert flagged
    for a in agents:
        for tid, w in a.template_mix.items():
            assert (w == 0) == (tid in flagged), (a.agent_id, tid, w)


def test_batch_size_in_ruling_range(agents: list[AgentDef]) -> None:
    assert all(4 <= a.batch_size <= 6 for a in agents)


def test_status_checks_change_the_seeded_status(templates: dict[str, TaskTemplate]) -> None:
    for t in templates.values():
        for s in t.success:
            if s.kind == "zendesk_status":
                assert s.params["status"] not in {"open", "new"}, t.id  # seeded as open


def test_slack_checks_carry_a_per_record_token(templates: dict[str, TaskTemplate]) -> None:
    for t in templates.values():
        for s in t.success:
            if s.kind == "slack_contains":
                assert "{" in s.params["text"], (t.id, s.params["text"])


def test_reprice_checks_the_amount_field(templates: dict[str, TaskTemplate]) -> None:
    fields = {s.params["field"] for s in templates["reprice_opportunity"].success}
    assert "Amount" in fields
    assert "Amount" in {s.params["field"] for s in templates["renewal_uplift"].success}


def test_newsletter_recipients_come_from_queried_account(
    templates: dict[str, TaskTemplate],
) -> None:
    news = [t for t in templates.values() if "newsletter-a" in t.roles]
    assert news
    for t in news:
        for seed in range(8):
            r = render(t, COMPANY, np.random.default_rng(seed))
            acct = r.slot_values["acct"]
            name = acct.get("account_name") or acct["name"]
            account = next(a for a in COMPANY.accounts if a.name == name)
            allowed = {c.email for c in COMPANY.contacts if c.account_key == account.key}
            for s in r.success:
                if s.kind == "mailhog_has_message" and not s.params["to"].endswith(
                    "helio-ops.example"
                ):
                    assert s.params["to"] in allowed, (t.id, s.params["to"])
            for email in re.findall(r"[\w.+-]+@[\w.-]+", r.prompt):
                assert email.rstrip(".") in allowed | {
                    e for e in re.findall(r"[\w.+-]+@helio-ops\.example", r.prompt)
                }, (t.id, email)


def test_no_yaml_anchors() -> None:
    for p in [*TASKS_DIR.glob("*.yaml"), *AGENTS_DIR.glob("*.yaml")]:
        assert not re.search(r"[&*]id\d+", p.read_text()), p.name


def test_public_sf_id_helper() -> None:
    assert sf_id("Contact", "cont-0001-0").startswith("003")


def test_stripe_ids_computed_once_per_session(
    templates: dict[str, TaskTemplate],
    agents: list[AgentDef],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    real = stripe_server.fake_stripe_ids

    def counting(company):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return real(company)

    monkeypatch.setattr("agentfw.sandbox.tasks.schema.fake_stripe_ids", counting)
    agent = next(a for a in agents if a.agent_id == "finance-reconcile-a")
    render_session(agent, templates, COMPANY, np.random.default_rng(1))
    assert calls == 1


def _tpl(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "x",
        "roles": ["a"],
        "prompt": "Look at {t.id}",
        "slots": {"t": {"kind": "ticket_open"}},
        "expected_tools": ["zendesk.get_ticket"],
        "success": [],
    }
    return base | over


def test_template_validators_reject_bad_input() -> None:
    TaskTemplate.model_validate(_tpl())
    for bad in (
        _tpl(prompt="Look at {nope.id}"),  # undeclared slot
        _tpl(prompt="Look at {t.nope}"),  # unknown field for the kind
        _tpl(prompt="Look at {t}"),  # no field access
        _tpl(success=[{"kind": "zendesk_status", "params": {"ticket_id": "{t.id}"}}]),
        _tpl(success=[{"kind": "slack_contains", "params": {"text": "{zz.id}"}}]),
        _tpl(expected_tools=[]),
    ):
        with pytest.raises(ValidationError):
            TaskTemplate.model_validate(bad)


def test_agent_validators_reject_bad_input() -> None:
    good = {
        "agent_id": "a",
        "system_prompt": "p",
        "tools": ["zendesk.get_ticket"],
        "template_mix": {"x": 1.0},
        "persona_tasks": {"t": ["x"]},
        "sessions_per_day": 1.0,
        "batch_size": 4,
    }
    AgentDef.model_validate(good)
    for bad in (
        good | {"batch_size": 0},
        good | {"batch_size": 9},
        good | {"template_mix": {"x": -1.0}},
        good | {"template_mix": {}},
        good | {"tools": ["get_ticket"]},
    ):
        with pytest.raises(ValidationError):
            AgentDef.model_validate(bad)


def test_slot_field_registry_matches_resolvers() -> None:
    from agentfw.sandbox.tasks.schema import SLOT_FIELDS, SlotSource, TaskTemplate

    variants: dict[str, list[dict[str, str]]] = {
        "ticket_open": [{}],
        "customer": [{}],
        "charge_refundable": [{}],
        "contact": [{}],
        "account": [{}],
        "opportunity": [{}],
        "soql_object": [{"object": o} for o in ("Opportunity", "Contact", "Account")],
        "channel_topic": [{}],
    }
    for kind, filters in variants.items():
        for f in filters:
            t = TaskTemplate(
                id="probe",
                roles=[],
                prompt="x",
                slots={"s": SlotSource.model_validate({"kind": kind, "filter": f})},
                expected_tools=["zendesk.get_ticket"],
            )
            got = render(t, COMPANY, np.random.default_rng(0)).slot_values["s"]
            assert set(got) == set(SLOT_FIELDS[kind]), (kind, f, set(got) ^ set(SLOT_FIELDS[kind]))
