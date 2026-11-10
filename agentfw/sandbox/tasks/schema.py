"""Task templates, agent definitions and deterministic rendering against a `Company`.

Template prompts and success params use ``str.format`` with attribute access on each slot:
``"Refund ticket {ticket.id} for {ticket.customer_email}"``. Every slot value is a string;
fields per ``SlotSource.kind`` are listed in ``_RESOLVERS``. Ids (``charge_id``, ``sf_id``,
``stripe_id``) are the deterministic fake-backend ids; a live run can pass ``id_map``
({company record key: backend id}) to ``render`` to substitute the seeders' real ids.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, Field

from agentfw.sandbox.seed.company import Company
from agentfw.sandbox.tools.salesforce_server import _sf_id
from agentfw.sandbox.tools.stripe_server import fake_stripe_ids

TASKS_DIR = Path(__file__).parent
AGENTS_DIR = Path(__file__).parents[1] / "agents"

SlotKind = Literal[
    "ticket_open",
    "customer",
    "charge_refundable",
    "contact",
    "account",
    "opportunity",
    "soql_object",
    "channel_topic",
]
CheckKind = Literal[
    "mailhog_has_message",
    "stripe_refund_exists",
    "sf_field_equals",
    "slack_contains",
    "zendesk_status",
]


class SlotSource(BaseModel):
    kind: SlotKind
    filter: dict[str, str] = Field(default_factory=dict)


class SuccessCheckSpec(BaseModel):
    kind: CheckKind
    params: dict[str, str]


class TaskTemplate(BaseModel):
    id: str
    roles: list[str]
    prompt: str
    slots: dict[str, SlotSource]
    expected_tools: list[str]
    success: list[SuccessCheckSpec] = Field(default_factory=list)
    max_steps: int = 25


class AgentDef(BaseModel):
    agent_id: str
    system_prompt: str
    tools: list[str]
    template_mix: dict[str, float]
    sessions_per_day: float
    batch_size: int = 1


@dataclass(frozen=True, slots=True)
class RenderedTask:
    template_id: str
    prompt: str
    slot_values: dict[str, dict[str, str]]
    expected_tools: list[str]
    success: list[SuccessCheckSpec]
    max_steps: int = 25


@dataclass(frozen=True, slots=True)
class RenderedSession:
    agent_id: str
    prompt: str
    tasks: list[RenderedTask]
    expected_tools: list[str]
    success: list[SuccessCheckSpec]
    max_steps: int


def load_templates(directory: Path) -> dict[str, TaskTemplate]:
    out: dict[str, TaskTemplate] = {}
    for path in sorted(directory.glob("*.yaml")):
        for raw in yaml.safe_load(path.read_text()) or []:
            t = TaskTemplate.model_validate(raw)
            if t.id in out:
                raise ValueError(f"duplicate template id {t.id!r} ({path.name})")
            out[t.id] = t
    return out


def load_agents(directory: Path) -> list[AgentDef]:
    return [
        AgentDef.model_validate(yaml.safe_load(p.read_text()))
        for p in sorted(directory.glob("*.yaml"))
    ]


# ---- slot resolution -------------------------------------------------------------------------


@dataclass(slots=True)
class _Ctx:
    company: Company
    rng: np.random.Generator
    used: set[str]
    ids: dict[str, str]  # company record key -> backend id (stripe or salesforce)
    channel: str


Fields = dict[str, str]


def _money(cents: float) -> str:
    return f"{cents / 100:.2f}"


def _pick[T](ctx: _Ctx, kind: str, items: list[T], key: Callable[[T], str]) -> T:
    """Random item, preferring ones not yet used in this session's queue."""
    if not items:
        raise ValueError(f"no {kind} matches the slot filter")
    order = [int(i) for i in ctx.rng.permutation(len(items))]
    for i in order:
        if f"{kind}:{key(items[i])}" not in ctx.used:
            ctx.used.add(f"{kind}:{key(items[i])}")
            return items[i]
    return items[order[0]]


def _sfid(ctx: _Ctx, obj: str, key: str) -> str:
    return ctx.ids.get(key) or _sf_id(obj, key)


def _customer_fields(ctx: _Ctx, key: str) -> Fields:
    c = next(x for x in ctx.company.customers if x.key == key)
    return {
        "key": c.key,
        "name": c.name,
        "first_name": c.name.split()[0],
        "email": c.email,
        "stripe_id": ctx.ids.get(c.key) or fake_stripe_ids(ctx.company)[c.key],
    }


def _charge_fields(ctx: _Ctx, key: str) -> Fields:
    p = next(x for x in ctx.company.charges if x.key == key)
    cust = _customer_fields(ctx, p.customer_key)
    return {
        "key": p.key,
        "id": ctx.ids.get(p.key) or fake_stripe_ids(ctx.company)[p.key],
        "amount": _money(p.amount_cents),
        "half_amount": _money(math.floor(p.amount_cents / 2)),
        "currency": p.currency.upper(),
        "description": p.description,
        "customer_name": cust["name"],
        "customer_email": cust["email"],
        "customer_stripe_id": cust["stripe_id"],
    }


def _ticket(ctx: _Ctx, f: dict[str, str]) -> Fields:
    items = [
        t for t in ctx.company.tickets if f.get("subject_contains", "").lower() in t.subject.lower()
    ]
    t = _pick(ctx, "ticket", items, lambda x: x.key)
    ch = _charge_fields(ctx, t.charge_key)
    cust = _customer_fields(ctx, t.customer_key)
    return {
        "key": t.key,
        "id": t.key,
        "subject": t.subject,
        "body": t.body,
        "customer_name": cust["name"],
        "customer_first_name": cust["first_name"],
        "customer_email": t.customer_email,
        "charge_id": ch["id"],
        "amount": ch["amount"],
        "half_amount": ch["half_amount"],
        "currency": ch["currency"],
    }


def _customer(ctx: _Ctx, f: dict[str, str]) -> Fields:
    items = list(ctx.company.customers)
    return _customer_fields(ctx, _pick(ctx, "customer", items, lambda x: x.key).key)


def _charge(ctx: _Ctx, f: dict[str, str]) -> Fields:
    lo, hi = float(f.get("min_amount", 0)), float(f.get("max_amount", 1e12))
    items = [c for c in ctx.company.charges if lo <= c.amount_cents / 100 <= hi]
    return _charge_fields(ctx, _pick(ctx, "charge", items, lambda x: x.key).key)


def _account_fields(ctx: _Ctx, key: str) -> Fields:
    a = next(x for x in ctx.company.accounts if x.key == key)
    contacts = [c for c in ctx.company.contacts if c.account_key == key]
    c0 = contacts[0]
    cust = _customer_fields(ctx, a.customer_key)
    return {
        "key": a.key,
        "name": a.name,
        "domain": a.domain,
        "sf_id": _sfid(ctx, "Account", a.key),
        "contact_email": c0.email,
        "contact_first_name": c0.first_name,
        "contact_last_name": c0.last_name,
        "contact_sf_id": _sfid(ctx, "Contact", c0.key),
        "customer_email": cust["email"],
        "customer_name": cust["name"],
    }


def _account(ctx: _Ctx, f: dict[str, str]) -> Fields:
    return _account_fields(
        ctx, _pick(ctx, "account", list(ctx.company.accounts), lambda x: x.key).key
    )


def _contact(ctx: _Ctx, f: dict[str, str]) -> Fields:
    c = _pick(ctx, "contact", list(ctx.company.contacts), lambda x: x.key)
    acct = next(a for a in ctx.company.accounts if a.key == c.account_key)
    return {
        "key": c.key,
        "first_name": c.first_name,
        "last_name": c.last_name,
        "email": c.email,
        "sf_id": _sfid(ctx, "Contact", c.key),
        "account_name": acct.name,
        "account_domain": acct.domain,
        "account_sf_id": _sfid(ctx, "Account", acct.key),
    }


def _opportunity(ctx: _Ctx, f: dict[str, str]) -> Fields:
    items = [o for o in ctx.company.opportunities if not f.get("stage") or o.stage == f["stage"]]
    o = _pick(ctx, "opportunity", items, lambda x: x.key)
    acct = _account_fields(ctx, o.account_key)
    return {
        "key": o.key,
        "name": o.name,
        "stage": o.stage,
        "amount": str(o.amount),
        "new_amount": str(round(o.amount * 1.1)),
        "sf_id": _sfid(ctx, "Opportunity", o.key),
        "account_name": acct["name"],
        "account_sf_id": acct["sf_id"],
        "contact_email": acct["contact_email"],
        "contact_first_name": acct["contact_first_name"],
    }


def _soql(ctx: _Ctx, f: dict[str, str]) -> Fields:
    obj = f.get("object", "Opportunity")
    if obj == "Opportunity":
        stages = sorted({o.stage for o in ctx.company.opportunities} - {"Closed Won"})
        fresh = [x for x in stages if f"soqlstage:{x}" not in ctx.used] or stages
        stage = f.get("stage") or str(fresh[int(ctx.rng.integers(0, len(fresh)))])
        ctx.used.add(f"soqlstage:{stage}")
        rows = [o for o in ctx.company.opportunities if o.stage == stage]
        if not rows:
            raise ValueError(f"no opportunities in stage {stage!r}")
        first = rows[0]
        ctx.used.add(f"soql:{first.key}")
        return {
            "key": f"{obj}:{stage}",
            "object": obj,
            "stage": stage,
            "query": f"SELECT Id, Name, Amount FROM Opportunity WHERE StageName = '{stage}'",
            "count": str(len(rows)),
            "first_id": _sfid(ctx, "Opportunity", first.key),
            "first_name": first.name,
            "account_name": next(
                a.name for a in ctx.company.accounts if a.key == first.account_key
            ),
        }
    if obj == "Contact":
        a = _pick(ctx, "soqlacct", list(ctx.company.accounts), lambda x: x.key)
        rows_c = [c for c in ctx.company.contacts if c.account_key == a.key]
        sf = _sfid(ctx, "Account", a.key)
        return {
            "key": f"{obj}:{a.key}",
            "object": obj,
            "stage": "",
            "query": f"SELECT Id, FirstName, LastName, Email FROM Contact WHERE AccountId = '{sf}'",
            "count": str(len(rows_c)),
            "first_id": _sfid(ctx, "Contact", rows_c[0].key),
            "first_name": f"{rows_c[0].first_name} {rows_c[0].last_name}",
            "account_name": a.name,
        }
    a = _pick(ctx, "soqlacct", list(ctx.company.accounts), lambda x: x.key)
    return {
        "key": f"Account:{a.key}",
        "object": "Account",
        "stage": "",
        "query": f"SELECT Id, Name, Website FROM Account WHERE Website = '{a.domain}'",
        "count": "1",
        "name": a.name,
        "domain": a.domain,
        "first_id": _sfid(ctx, "Account", a.key),
        "first_name": a.name,
        "account_name": a.name,
    }


_TOPICS = [
    ("billing escalations", "weekly billing escalation roundup"),
    ("renewals", "Q4 renewals pipeline"),
    ("support backlog", "support backlog status"),
    ("product launch", "launch readiness checklist"),
    ("finance close", "month-end close schedule"),
    ("customer feedback", "customer feedback themes"),
    ("onboarding", "new customer onboarding queue"),
    ("security reviews", "vendor security review status"),
    ("holiday coverage", "holiday support coverage"),
    ("pricing change", "pricing change rollout"),
    ("partner program", "partner program updates"),
    ("data migration", "CRM data migration progress"),
    ("training", "agent training schedule"),
    ("incident follow-up", "incident follow-up actions"),
]


def _topic(ctx: _Ctx, f: dict[str, str]) -> Fields:
    topic, note = _pick(ctx, "topic", _TOPICS, lambda x: x[0])
    return {"key": f"topic:{topic}", "channel": ctx.channel, "topic": topic, "note": note}


_RESOLVERS: dict[str, Callable[[_Ctx, dict[str, str]], Fields]] = {
    "ticket_open": _ticket,
    "customer": _customer,
    "charge_refundable": _charge,
    "contact": _contact,
    "account": _account,
    "opportunity": _opportunity,
    "soql_object": _soql,
    "channel_topic": _topic,
}


def _render_task(t: TaskTemplate, ctx: _Ctx) -> RenderedTask:
    values = {name: _RESOLVERS[src.kind](ctx, src.filter) for name, src in t.slots.items()}
    ns = {name: SimpleNamespace(**v) for name, v in values.items()}
    return RenderedTask(
        t.id,
        t.prompt.format(**ns),
        values,
        list(t.expected_tools),
        [
            SuccessCheckSpec(kind=s.kind, params={k: v.format(**ns) for k, v in s.params.items()})
            for s in t.success
        ],
        t.max_steps,
    )


def render(
    t: TaskTemplate,
    company: Company,
    rng: np.random.Generator,
    *,
    id_map: dict[str, str] | None = None,
    channel: str = "#sandbox",
    used: set[str] | None = None,
) -> RenderedTask:
    ctx = _Ctx(company, rng, used if used is not None else set(), id_map or {}, channel)
    return _render_task(t, ctx)


def render_session(
    agent: AgentDef,
    templates: dict[str, TaskTemplate],
    company: Company,
    rng: np.random.Generator,
    *,
    id_map: dict[str, str] | None = None,
    channel: str = "#sandbox",
) -> RenderedSession:
    """One session's work: `batch_size` templates drawn from the agent's mix (R1 queue)."""
    ids = sorted(agent.template_mix)
    w = np.array([agent.template_mix[i] for i in ids], dtype=float)
    ctx = _Ctx(company, rng, set(), id_map or {}, channel)
    tasks: list[RenderedTask] = []
    seen: set[str] = set()
    for _ in range(agent.batch_size):
        cand = list(range(len(ids)))
        while True:  # redraw (without the colliding template) when the primary record repeats
            weights = w[cand] / w[cand].sum()
            pick = cand[int(rng.choice(len(cand), p=weights))]
            task = _render_task(templates[ids[pick]], ctx)
            key = next(iter(task.slot_values.values()))["key"]
            if key not in seen or len(cand) == 1:
                break
            cand.remove(pick)
        seen.add(key)
        tasks.append(task)
    if agent.batch_size == 1:
        prompt = tasks[0].prompt
    else:
        lines = [f"{n}. {t.prompt}" for n, t in enumerate(tasks, 1)]
        prompt = "Please work through this queue in order, then give me a one-line summary:\n\n"
        prompt += "\n".join(lines)
    return RenderedSession(
        agent.agent_id,
        prompt,
        tasks,
        [x for t in tasks for x in t.expected_tools],
        [s for t in tasks for s in t.success],
        sum(t.max_steps for t in tasks),
    )
