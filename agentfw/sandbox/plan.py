"""Session planning for the sandbox runner: which agent, which templates, which injection.

Planning is pure and deterministic. Sessions go round-robin over the agents in sorted
``agent_id`` order (an even split, remainder to the first agents, as ruling R3), so a larger
``n_sessions`` extends a smaller plan without changing it. Each session draws ``batch_size``
templates from the agent's mix; quarter-end-only templates (weight 0) are never drawn, since
scheduling them on simulated quarter-end days is optional under ruling RS1 and is not done.

With ``AttackMix`` probabilities a session gets one injection:

* **action level**: a family whose target tool the session will call (``ACTION_TARGETS`` for the
  single-call families, a write for ``velocity_burst``, a list read for ``exfil_volume``);
* **prompt level**: a payload whose induced tool is in the agent's allowlist and whose carrier,
  restricted to ``prompt_carriers``, the session reads (a ticket it opens, an Account it fetches,
  the channel it reads). Live runs pass ``LIVE_PROMPT_CARRIERS``: only the local zendesk store can
  be snapshotted and restored. A prompt draw with no usable carrier falls back to an action-level
  plan (risk R2: action-level injection fills the attack quota).

No fitting candidate at either level means no injection for that session.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence

import numpy as np
from pydantic import BaseModel, Field, model_validator

from agentfw.sandbox.injector import (
    ACTION_FAMILIES,
    ACTION_TARGETS,
    InjectionPlan,
    PayloadSpec,
    load_payloads,
)
from agentfw.sandbox.tasks.schema import AgentDef, TaskTemplate
from agentfw.synth.sampler import stable_hash

__all__ = [
    "ALL_PROMPT_CARRIERS",
    "LIVE_PROMPT_CARRIERS",
    "TICKET_CARRIERS",
    "AttackMix",
    "SessionPlan",
    "carrier_task",
    "plan_sessions",
]

TICKET_CARRIERS = frozenset({"ticket_body", "ticket_comment"})
ALL_PROMPT_CARRIERS = TICKET_CARRIERS | {"sf_description", "slack_message"}
LIVE_PROMPT_CARRIERS = TICKET_CARRIERS  # live Salesforce / Slack cannot snapshot+restore
WRITES = frozenset(
    {
        "stripe.create_refund",
        "stripe.create_payout",
        "mail.send_email",
        "slack.post_message",
        "zendesk.add_comment",
        "zendesk.update_status",
        "salesforce.update_record",
        "salesforce.mass_update",
        "salesforce.create_task",
    }
)
LIST_READS = frozenset(
    {"stripe.list_charges", "salesforce.run_soql", "zendesk.list_tickets", "slack.read_channel"}
)


class AttackMix(BaseModel):
    action_rate: float = Field(default=0.15, ge=0.0, le=1.0)
    prompt_rate: float = Field(default=0.15, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _at_most_one(self) -> AttackMix:
        if self.action_rate + self.prompt_rate > 1.0:
            raise ValueError("action_rate + prompt_rate must be <= 1")
        return self


class SessionPlan(BaseModel):
    session_id: str
    agent_id: str
    template_ids: list[str] = Field(min_length=1)
    injection: InjectionPlan | None = None
    seed: int


def carrier_task(carrier: str, templates: Sequence[TaskTemplate]) -> int | None:
    """Index of the first template whose expected calls read the payload's carrier record."""
    for i, t in enumerate(templates):
        kinds = {(s.kind, s.filter.get("object")) for s in t.slots.values()}
        tools = set(t.expected_tools)
        if carrier in TICKET_CARRIERS:
            ok = "zendesk.get_ticket" in tools and any(k == "ticket_open" for k, _ in kinds)
        elif carrier == "sf_description":
            ok = "salesforce.get_record" in tools and (
                ("account", None) in kinds or ("soql_object", "Account") in kinds
            )
        else:  # slack_message
            ok = "slack.read_channel" in tools
        if ok:
            return i
    return None


def _action_plan(
    tools: list[str], rng: np.random.Generator, attack_id: str
) -> InjectionPlan | None:
    cands: list[tuple[str, list[str]]] = []
    for fam in sorted(ACTION_FAMILIES):
        if fam in ACTION_TARGETS:
            ok: Collection[str] = ACTION_TARGETS[fam]
        else:
            ok = WRITES if fam == "velocity_burst" else LIST_READS
        targets = sorted({t for t in tools if t in ok})
        if targets:
            cands.append((fam, targets))
    if not cands:
        return None
    fam, targets = cands[int(rng.integers(len(cands)))]
    target = targets[int(rng.integers(len(targets)))]
    return InjectionPlan(level="action", family=fam, target_tool=target, attack_id=attack_id)


def _prompt_plan(
    agent: AgentDef,
    templates: Sequence[TaskTemplate],
    payloads: Sequence[PayloadSpec],
    carriers: Collection[str],
    rng: np.random.Generator,
    attack_id: str,
) -> InjectionPlan | None:
    cands = [
        p
        for p in payloads
        if p.carrier in carriers
        and p.match.tool_id in agent.tools
        and carrier_task(p.carrier, templates) is not None
    ]
    if not cands:
        return None
    p = cands[int(rng.integers(len(cands)))]
    return InjectionPlan(
        level="prompt",
        family=p.family,
        target_tool=p.match.tool_id,
        attack_id=attack_id,
        payload_id=p.id,
    )


def plan_sessions(
    agents: Sequence[AgentDef],
    templates: Mapping[str, TaskTemplate],
    n_sessions: int,
    mix: AttackMix,
    seed: int,
    *,
    prompt_carriers: Collection[str] = ALL_PROMPT_CARRIERS,
) -> list[SessionPlan]:
    """``n_sessions`` plans, round-robin over agents (sorted by id); see the module docs."""
    if n_sessions < 0:
        raise ValueError("n_sessions must be >= 0")
    unknown = set(prompt_carriers) - ALL_PROMPT_CARRIERS
    if unknown:
        raise ValueError(f"unknown prompt carriers: {sorted(unknown)}")
    order = sorted(agents, key=lambda a: a.agent_id)
    if not order and n_sessions:
        raise ValueError("no agents to plan sessions for")
    payloads = sorted(load_payloads().values(), key=lambda p: p.id)
    plans: list[SessionPlan] = []
    for i in range(n_sessions):
        agent = order[i % len(order)]
        sid = f"s{seed}-{agent.agent_id}-{i // len(order):05d}"
        sseed = stable_hash(f"{seed}:{sid}") >> 1  # fits a signed 64-bit int
        rng = np.random.default_rng([sseed, 0])
        ids = sorted(t for t, w in agent.template_mix.items() if w > 0)
        w = np.array([agent.template_mix[t] for t in ids], dtype=float)
        picks = rng.choice(len(ids), size=agent.batch_size, p=w / w.sum())
        tids = [ids[int(k)] for k in picks]
        tmpls = [templates[t] for t in tids]
        tools = [x for t in tmpls for x in t.expected_tools]
        u = float(rng.random())
        attack_id = f"{sid}-atk"
        injection: InjectionPlan | None = None
        if u < mix.action_rate:
            injection = _action_plan(tools, rng, attack_id)
        elif u < mix.action_rate + mix.prompt_rate:
            injection = _prompt_plan(agent, tmpls, payloads, prompt_carriers, rng, attack_id)
            if injection is None:  # R2: no usable carrier, so inject at the action level
                injection = _action_plan(tools, rng, attack_id)
        plans.append(
            SessionPlan(
                session_id=sid,
                agent_id=agent.agent_id,
                template_ids=tids,
                injection=injection,
                seed=sseed,
            )
        )
    return plans
