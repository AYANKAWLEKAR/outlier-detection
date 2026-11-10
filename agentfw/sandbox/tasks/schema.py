"""Task templates, agent definitions and deterministic rendering against a `Company`.

Template prompts and success params use ``str.format`` with attribute access on each slot:
``"Refund ticket {ticket.id} for {ticket.customer_email}"``. Every slot value is a string;
the fields available per ``SlotSource.kind`` are listed in ``SLOT_FIELDS`` (checked when a
template is loaded).

``AgentDef.persona_tasks`` maps each synth persona task to the templates implementing it;
``template_mix`` gives each task its persona weight, split evenly among its templates.
Templates flagged ``quarter_end_only`` carry weight 0 in the normal mix (their persona task has
weight 0): a runner may schedule them directly (``render``) on quarter-end days.

Ids (``charge_id``, ``sf_id``, ``stripe_id``) are the deterministic fake-backend ids;
a live run can pass ``id_map`` ({company record key: backend id}) to ``render``
to substitute the seeders' real ids.
"""

from __future__ import annotations

import math
import string
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from agentfw.sandbox.seed.company import Company
from agentfw.sandbox.tools.salesforce_server import sf_id
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


SLOT_FIELDS: dict[str, frozenset[str]] = {
    "ticket_open": frozenset(
        [
            "key",
            "id",
            "subject",
            "body",
            "customer_name",
            "customer_first_name",
            "customer_email",
            "charge_id",
            "amount",
            "half_amount",
            "currency",
        ]
    ),
    "customer": frozenset(["key", "name", "first_name", "email", "stripe_id"]),
    "charge_refundable": frozenset(
        [
            "key",
            "id",
            "amount",
            "half_amount",
            "currency",
            "description",
            "customer_name",
            "customer_email",
            "customer_stripe_id",
        ]
    ),
    "contact": frozenset(
        [
            "key",
            "first_name",
            "last_name",
            "email",
            "sf_id",
            "account_name",
            "account_domain",
            "account_sf_id",
        ]
    ),
    "account": frozenset(
        [
            "key",
            "name",
            "domain",
            "sf_id",
            "contact_email",
            "contact_first_name",
            "contact_last_name",
            "contact_sf_id",
            "customer_email",
            "customer_name",
        ]
    ),
    "opportunity": frozenset(
        [
            "key",
            "name",
            "stage",
            "amount",
            "new_amount",
            "sf_id",
            "account_name",
            "account_sf_id",
            "contact_email",
            "contact_first_name",
        ]
    ),
    "soql_object": frozenset(
        [
            "key",
            "object",
            "stage",
            "query",
            "count",
            "first_id",
            "first_name",
            "account_name",
            "name",
            "domain",
            "first_email",
            "second_email",
            "contact_email",
        ]
    ),
    "channel_topic": frozenset(["key", "channel", "topic", "note", "ref"]),
}
CHECK_PARAMS: dict[str, frozenset[str]] = {
    "mailhog_has_message": frozenset({"to", "subject_contains"}),
    "stripe_refund_exists": frozenset({"charge_id"}),
    "sf_field_equals": frozenset({"object", "record_id", "field", "value"}),
    "slack_contains": frozenset({"text"}),
    "zendesk_status": frozenset({"ticket_id", "status"}),
}
MAX_BATCH = 6


class SlotSource(BaseModel):
    kind: SlotKind
    filter: dict[str, str] = Field(default_factory=dict)


class SuccessCheckSpec(BaseModel):
    kind: CheckKind
    params: dict[str, str]

    @model_validator(mode="after")
    def _params_match_kind(self) -> SuccessCheckSpec:
        want = CHECK_PARAMS[self.kind]
        if set(self.params) != want:
            raise ValueError(f"{self.kind} params must be exactly {sorted(want)}")
        return self


def _placeholders(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for _lit, name, _spec, _conv in string.Formatter().parse(text):
        if name is None:
            continue
        slot, dot, attr = name.partition(".")
        if not dot or not attr or "." in attr or "[" in name:
            raise ValueError(f"placeholder {{{name}}} must look like {{slot.field}}")
        out.append((slot, attr))
    return out


class TaskTemplate(BaseModel):
    id: str
    roles: list[str]
    prompt: str
    slots: dict[str, SlotSource]
    expected_tools: list[str] = Field(min_length=1)
    success: list[SuccessCheckSpec] = Field(default_factory=list)
    max_steps: int = Field(default=25, ge=1)
    quarter_end_only: bool = False

    @model_validator(mode="after")
    def _placeholders_resolve(self) -> TaskTemplate:
        texts = [self.prompt] + [v for c in self.success for v in c.params.values()]
        for text in texts:
            for slot, attr in _placeholders(text):
                if slot not in self.slots:
                    raise ValueError(f"{self.id}: placeholder uses undeclared slot {slot!r}")
                if attr not in SLOT_FIELDS[self.slots[slot].kind]:
                    raise ValueError(f"{self.id}: {slot}.{attr} is not a field of that slot kind")
        return self


class AgentDef(BaseModel):
    agent_id: str
    system_prompt: str
    tools: list[str]
    template_mix: dict[str, float]
    sessions_per_day: float = Field(gt=0)
    batch_size: int = Field(default=1, ge=1, le=MAX_BATCH)  # items per session (R1)
    persona_tasks: dict[str, list[str]] = Field(default_factory=dict)

    @field_validator("tools")
    @classmethod
    def _namespaced(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v) or any(t.count(".") != 1 for t in v):
            raise ValueError("tools must be unique `server.tool` ids")
        return v

    @field_validator("template_mix")
    @classmethod
    def _mix(cls, v: dict[str, float]) -> dict[str, float]:
        if not v or any(w < 0 for w in v.values()) or sum(v.values()) <= 0:
            raise ValueError("template_mix needs non-negative weights with a positive sum")
        return v


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
    ids: dict[str, str]  # company record key -> backend id (stripe; overrides from id_map)
    channel: str


def _make_ctx(
    company: Company,
    rng: np.random.Generator,
    used: set[str] | None,
    id_map: dict[str, str] | None,
    channel: str,
) -> _Ctx:
    """Stripe ids are computed once per context (one `render` call or one session)."""
    ids = {**fake_stripe_ids(company), **(id_map or {})}
    return _Ctx(company, rng, used if used is not None else set(), ids, channel)


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
    return ctx.ids.get(key) or sf_id(obj, key)


def _customer_fields(ctx: _Ctx, key: str) -> Fields:
    c = next(x for x in ctx.company.customers if x.key == key)
    return {
        "key": c.key,
        "name": c.name,
        "first_name": c.name.split()[0],
        "email": c.email,
        "stripe_id": ctx.ids[c.key],
    }


def _charge_fields(ctx: _Ctx, key: str) -> Fields:
    p = next(x for x in ctx.company.charges if x.key == key)
    cust = _customer_fields(ctx, p.customer_key)
    return {
        "key": p.key,
        "id": ctx.ids[p.key],
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


def _soql_base(ctx: _Ctx, account_key: str) -> Fields:
    a = next(x for x in ctx.company.accounts if x.key == account_key)
    emails = [c.email for c in ctx.company.contacts if c.account_key == account_key]
    return {
        "stage": "",
        "name": a.name,
        "domain": a.domain,
        "account_name": a.name,
        "first_email": emails[0],
        "second_email": emails[1] if len(emails) > 1 else emails[0],
        "contact_email": emails[0],
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
        return _soql_base(ctx, first.account_key) | {
            "key": f"{obj}:{stage}",
            "object": obj,
            "stage": stage,
            "query": f"SELECT Id, Name, Amount FROM Opportunity WHERE StageName = '{stage}'",
            "count": str(len(rows)),
            "first_id": _sfid(ctx, "Opportunity", first.key),
            "first_name": first.name,
        }
    min_contacts = int(f.get("min_contacts", 1))
    accts = [
        a
        for a in ctx.company.accounts
        if sum(c.account_key == a.key for c in ctx.company.contacts) >= min_contacts
    ]
    a = _pick(ctx, "soqlacct", accts, lambda x: x.key)
    if obj == "Contact":
        rows_c = [c for c in ctx.company.contacts if c.account_key == a.key]
        sf = _sfid(ctx, "Account", a.key)
        return _soql_base(ctx, a.key) | {
            "key": f"{obj}:{a.key}",
            "object": obj,
            "query": f"SELECT Id, FirstName, LastName, Email FROM Contact WHERE AccountId = '{sf}'",
            "count": str(len(rows_c)),
            "first_id": _sfid(ctx, "Contact", rows_c[0].key),
            "first_name": f"{rows_c[0].first_name} {rows_c[0].last_name}",
        }
    return _soql_base(ctx, a.key) | {
        "key": f"Account:{a.key}",
        "object": "Account",
        "query": f"SELECT Id, Name, Website FROM Account WHERE Website = '{a.domain}'",
        "count": "1",
        "first_id": _sfid(ctx, "Account", a.key),
        "first_name": a.name,
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
    ref = f"REF-{int(ctx.rng.integers(10000, 100000))}"  # per-record token for Slack checks
    return {
        "key": f"topic:{topic}",
        "channel": ctx.channel,
        "topic": topic,
        "note": note,
        "ref": ref,
    }


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
    return _render_task(t, _make_ctx(company, rng, used, id_map, channel))


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
    ids = sorted(i for i, wt in agent.template_mix.items() if wt > 0)  # skips quarter-end work
    w = np.array([agent.template_mix[i] for i in ids], dtype=float)
    ctx = _make_ctx(company, rng, None, id_map, channel)
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
    return compose_session(agent.agent_id, tasks)


def compose_session(agent_id: str, tasks: list[RenderedTask]) -> RenderedSession:
    """One session from rendered tasks: a single task's prompt, or a numbered queue (R1)."""
    if not tasks:
        raise ValueError("a session needs at least one task")
    if len(tasks) == 1:
        prompt = tasks[0].prompt
    else:
        lines = [f"{n}. {t.prompt}" for n, t in enumerate(tasks, 1)]
        prompt = "Please work through this queue in order, then give me a one-line summary:\n\n"
        prompt += "\n".join(lines)
    return RenderedSession(
        agent_id,
        prompt,
        tasks,
        [x for t in tasks for x in t.expected_tools],
        [s for t in tasks for s in t.success],
        sum(t.max_steps for t in tasks),
    )


def validate_catalog(templates: dict[str, TaskTemplate], agents: list[AgentDef]) -> None:
    """Cross-file consistency: roles, mixes, persona-task maps and tool allowlists agree."""
    by_id = {a.agent_id: a for a in agents}
    for t in templates.values():
        for role in t.roles:
            if role not in by_id:
                raise ValueError(f"{t.id}: unknown role {role!r}")
            if not set(t.expected_tools) <= set(by_id[role].tools):
                raise ValueError(f"{t.id}: expected_tools not in {role}'s tool allowlist")
    for a in agents:
        mine = {t.id for t in templates.values() if a.agent_id in t.roles}
        if set(a.template_mix) != mine:
            raise ValueError(f"{a.agent_id}: template_mix keys differ from templates naming it")
        if set(a.template_mix) != {i for ids in a.persona_tasks.values() for i in ids}:
            raise ValueError(f"{a.agent_id}: persona_tasks do not cover template_mix")
        for tid, wt in a.template_mix.items():
            if (wt == 0) != templates[tid].quarter_end_only:
                raise ValueError(f"{a.agent_id}: {tid} weight 0 iff quarter_end_only")
        used = {x for tid in mine for x in templates[tid].expected_tools}
        if set(a.tools) != used:
            raise ValueError(f"{a.agent_id}: tools must equal the tools its templates use")
