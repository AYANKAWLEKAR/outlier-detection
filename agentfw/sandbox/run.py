"""Resumable sandbox runner: plan sessions, run them through the proxy, write benchmark sessions.

    python -m agentfw.sandbox.run --sessions N --seed S --out runs/<id> [--fake]

**Planning** (``plan_sessions``) is pure and deterministic. Sessions go round-robin over the
agents in sorted ``agent_id`` order (an even split, remainder to the first agents, as ruling R3),
so a larger ``n_sessions`` extends a smaller plan without changing it. Each session draws
``batch_size`` templates from the agent's mix (quarter-end-only templates, weight 0, are never
drawn; scheduling them on simulated quarter-end days is optional under ruling RS1 and is not
done) and, with ``AttackMix`` probabilities, one injection: an action-level family whose target
tool the session will call, or a prompt-level payload whose carrier the session reads (a ticket it
opens, an Account it fetches, the channel it reads) and whose induced tool is in the agent's
allowlist. No fitting candidate means no injection for that session.

**Executing** (``execute``) renders every plan in plan order against one run-level ``used`` record
set (ruling RS1: mutable primary records are not reused across the run's sessions while unused
ones remain; a ticket and its charge count as one record). Each session then

1. builds its backends — fresh fakes per session (``backends="fake"``, so no state leaks), or the
   shared live backends — and, for an action-level plan, an ``ActionInjector`` over separate fresh
   fakes (D-014: a mutated call never reaches the session's backends);
2. plants a prompt-level payload into the record the session reads (``record_key``);
3. connects ``mcp.Client`` per server, a ``FirewallProxy`` writing to a ``JsonlSink`` and the
   agent's client (filtered to the agent's tool allowlist) and runs ``run_session``;
4. runs the templates' success checks against the same backends, then restores planted state;
5. writes ``raw/<session>.jsonl``, ``labels/<session>.jsonl`` (one row per envelope: the
   proxy's action labels, post-hoc prompt labels, ``benign`` otherwise) and
   ``transcripts/<session>.json``, and only then appends its ``ledger.jsonl`` line.

**Resume**: sessions whose latest ledger status is ``completed`` are skipped. A crash mid-session
leaves no ledger line, so the session re-runs from scratch (its work dir is replaced). A session
that raises is recorded as ``failed`` and retried on the next run.

**Budgets**: before each session is started the runner stops scheduling once the run's LLM spend
(every ledger line's ``usd``) reaches ``cfg.llm_usd_budget`` or the Salesforce ``ApiBudget`` has
nothing left; sessions already running finish. ``RunSummary.stopped_reason`` says which.

In fake mode the agent is ``scripted_client(prepared)``: a ``ScriptedChatClient`` built from
``scripted_policy`` (prompt-level payloads are obeyed with probability ``COMPLY_PROB``, decided per
session from its seed), so the whole pipeline runs offline. Live mode (``OpenAICompatClient`` and
the real sandbox APIs) seeds the org idempotently and caches the ids in ``id_map.json``. Live
prompt-level sessions run alone (they snapshot/restore the shared local zendesk store); carriers
on backends without snapshot/restore (live Salesforce, Slack) are skipped and recorded.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import shutil
import sys
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from contextlib import AsyncExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import anyio
import numpy as np
import orjson
from mcp import Client
from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult, ListToolsResult
from pydantic import BaseModel, Field, SecretStr, model_validator

from agentfw.core.envelope import Label, RawCall
from agentfw.sandbox.agent import McpLike, Transcript, run_session
from agentfw.sandbox.checks import CheckContext, SuccessCheck
from agentfw.sandbox.config import ApiBudget, SandboxConfig
from agentfw.sandbox.injector import (
    ACTION_FAMILIES,
    ACTION_TARGETS,
    ActionInjector,
    InjectionPlan,
    PayloadSpec,
    PlantHandle,
    PromptInjector,
    label_prompt_session,
    load_payloads,
)
from agentfw.sandbox.llm import ChatClient, OpenAICompatClient, ScriptedChatClient, UsageMeter
from agentfw.sandbox.proxy_py import FirewallProxy, JsonlSink, SessionInfo, error_result
from agentfw.sandbox.proxy_py.sinks import CALLS, LABELS
from agentfw.sandbox.scripted_policy import (
    Call,
    PolicyContext,
    build_turns,
    compliant_call,
    scripted_calls,
)
from agentfw.sandbox.seed.company import Company, build_company
from agentfw.sandbox.tasks.schema import (
    AGENTS_DIR,
    TASKS_DIR,
    AgentDef,
    RenderedSession,
    TaskTemplate,
    compose_session,
    load_agents,
    load_templates,
    render,
)
from agentfw.sandbox.tools import (
    mail_server,
    salesforce_server,
    slack_server,
    stripe_server,
    zendesk_server,
)
from agentfw.synth.sampler import stable_hash

__all__ = [
    "COMPANY_SEED",
    "AttackMix",
    "PreparedSession",
    "RunSummary",
    "SessionPlan",
    "execute",
    "fake_config",
    "main",
    "plan_sessions",
    "scripted_client",
]

logger = logging.getLogger(__name__)

COMPANY_SEED = 0  # the synthetic company the live org is seeded with (tests and seeders use 0)
COMPLY_PROB = 0.5  # fake mode: chance the scripted agent obeys a planted payload
LEDGER = "ledger.jsonl"
SUMMARY = "summary.json"
ID_MAP = "id_map.json"
RAW_DIR, LABELS_DIR, TRANSCRIPTS_DIR, WORK_DIR = "raw", "labels", "transcripts", ".work"

_WRITES = frozenset(
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
_LIST_READS = frozenset(
    {"stripe.list_charges", "salesforce.run_soql", "zendesk.list_tickets", "slack.read_channel"}
)
_MUTABLE_KINDS = frozenset(
    {"ticket_open", "charge_refundable", "contact", "account", "opportunity", "soql_object"}
)
_TICKET_CARRIERS = frozenset({"ticket_body", "ticket_comment"})
_JSONL = orjson.OPT_SORT_KEYS
StopReason = Literal["llm_usd_budget", "sf_api_budget"]


# --- models --------------------------------------------------------------------------------------


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


class RunSummary(BaseModel):
    completed: int = 0
    failed: int = 0
    resumed: int = 0  # completed by an earlier invocation and skipped now
    usd: float = 0.0
    sf_calls: int = 0  # Salesforce envelopes recorded (the ApiBudget file holds real charges)
    actions: int = 0
    attacks_attempted: int = 0
    attacks_succeeded: int = 0
    stopped_reason: StopReason | None = None


# --- planning ------------------------------------------------------------------------------------


def _carrier_task(carrier: str, templates: Sequence[TaskTemplate]) -> int | None:
    """Index of the first template whose expected calls read the payload's carrier record."""
    for i, t in enumerate(templates):
        kinds = {(s.kind, s.filter.get("object")) for s in t.slots.values()}
        tools = set(t.expected_tools)
        if carrier in _TICKET_CARRIERS:
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
            ok = _WRITES if fam == "velocity_burst" else _LIST_READS
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
    rng: np.random.Generator,
    attack_id: str,
) -> InjectionPlan | None:
    cands = [
        p
        for p in payloads
        if p.match.tool_id in agent.tools and _carrier_task(p.carrier, templates) is not None
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
) -> list[SessionPlan]:
    """``n_sessions`` plans, round-robin over agents (sorted by id); see the module docs."""
    if n_sessions < 0:
        raise ValueError("n_sessions must be >= 0")
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
        u = float(rng.random())
        attack_id = f"{sid}-atk"
        injection: InjectionPlan | None = None
        if u < mix.action_rate:
            tools = [x for t in tmpls for x in t.expected_tools]
            injection = _action_plan(tools, rng, attack_id)
        elif u < mix.action_rate + mix.prompt_rate:
            injection = _prompt_plan(agent, tmpls, payloads, rng, attack_id)
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


# --- preparation (rendering against the run-level used set) --------------------------------------


@dataclass(frozen=True, slots=True)
class PreparedSession:
    plan: SessionPlan
    agent: AgentDef
    rendered: RenderedSession
    kinds: list[dict[str, str]]  # per task: slot name -> slot kind
    records: list[str]  # per task: "<kind>:<key>" of its primary (first) slot
    reused: int  # mutable primary records already used by an earlier session of the run
    carrier_task: int | None  # prompt plans: the task that reads the carrier
    carrier_key: str | None  # prompt plans: the company record to plant into
    policy: PolicyContext


def _link_used(used: set[str], company: Company) -> None:
    """A ticket and the charge it references are one mutable record (refund via either)."""
    for t in company.tickets:
        tk, ch = f"ticket:{t.key}", f"charge:{t.charge_key}"
        if tk in used:
            used.add(ch)
        elif ch in used:
            used.add(tk)


def _carrier_key(carrier: str, kinds: Mapping[str, str], values: Mapping[str, Any]) -> str | None:
    for name, kind in kinds.items():
        if carrier in _TICKET_CARRIERS and kind == "ticket_open":
            return str(values[name]["key"])
        if carrier == "sf_description" and kind == "account":
            return str(values[name]["key"])
        if carrier == "sf_description" and kind == "soql_object":
            return str(values[name]["key"]).removeprefix("Account:")
    return None


def _prepare_all(
    plans: Sequence[SessionPlan],
    templates: Mapping[str, TaskTemplate],
    agents: Mapping[str, AgentDef],
    company: Company,
    id_map: Mapping[str, str] | None,
    channel: str,
) -> list[PreparedSession]:
    policy = PolicyContext.build(company, id_map, channel)
    payloads = load_payloads()
    used: set[str] = set()
    seen: set[str] = set()
    out: list[PreparedSession] = []
    for plan in plans:
        agent = agents.get(plan.agent_id)
        if agent is None:
            raise ValueError(f"{plan.session_id}: unknown agent {plan.agent_id!r}")
        tmpls = [templates[t] for t in plan.template_ids]
        rng = np.random.default_rng([plan.seed, 1])
        tasks = []
        for t in tmpls:
            tasks.append(
                render(t, company, rng, id_map=dict(id_map or {}), channel=channel, used=used)
            )
            _link_used(used, company)
        kinds: list[dict[str, str]] = [{n: str(s.kind) for n, s in t.slots.items()} for t in tmpls]
        records: list[str] = []
        reused = 0
        for task, k in zip(tasks, kinds, strict=True):
            first = next(iter(task.slot_values))
            rec = f"{k[first]}:{task.slot_values[first]['key']}"
            records.append(rec)
            if k[first] in _MUTABLE_KINDS and rec in seen:
                reused += 1
        seen.update(records)
        carrier_task = carrier_key = None
        inj = plan.injection
        if inj is not None and inj.level == "prompt" and inj.payload_id is not None:
            carrier = payloads[inj.payload_id].carrier
            carrier_task = _carrier_task(carrier, tmpls)
            if carrier_task is not None:
                carrier_key = _carrier_key(
                    carrier, kinds[carrier_task], tasks[carrier_task].slot_values
                )
        out.append(
            PreparedSession(
                plan,
                agent,
                compose_session(agent.agent_id, tasks),
                kinds,
                records,
                reused,
                carrier_task,
                carrier_key,
                policy,
            )
        )
    return out


# --- the scripted (offline) agent ----------------------------------------------------------------


def _carrier_charge(prep: PreparedSession) -> str:
    company = prep.policy.company
    ticket = next((t for t in company.tickets if t.key == prep.carrier_key), company.tickets[0])
    return prep.policy.stripe_ids[ticket.charge_key]


def _carrier_read_index(calls: list[Call], prep: PreparedSession, carrier: str) -> int | None:
    sf_account = prep.policy.sf_ids.get(prep.carrier_key or "")
    for i, (tool, args) in enumerate(calls):
        if carrier in _TICKET_CARRIERS:
            hit = tool == "zendesk.get_ticket" and args.get("ticket_id") == prep.carrier_key
        elif carrier == "sf_description":
            hit = tool == "salesforce.get_record" and args.get("record_id") == sf_account
        else:
            hit = tool == "slack.read_channel"
        if hit:
            return i
    return None


def scripted_client(prep: PreparedSession) -> ScriptedChatClient:
    """Fake-mode agent: the scripted policy's calls for every task (see ``scripted_policy``)."""
    calls: list[Call] = []
    for task, kinds in zip(prep.rendered.tasks, prep.kinds, strict=True):
        calls.extend(scripted_calls(task, kinds, prep.policy))
    inj = prep.plan.injection
    complies = np.random.default_rng([prep.plan.seed, 3]).random() < COMPLY_PROB
    if inj is not None and inj.level == "prompt" and inj.payload_id is not None and complies:
        payload = load_payloads()[inj.payload_id]
        obey = compliant_call(payload, prep.policy, charge_id=_carrier_charge(prep))
        at = _carrier_read_index(calls, prep, payload.carrier)
        calls.insert(len(calls) if at is None else at + 1, obey)
    return ScriptedChatClient(build_turns(calls))


# --- backends ------------------------------------------------------------------------------------

_BUILDERS: dict[str, Callable[[Any], MCPServer]] = {
    "zendesk": zendesk_server.build_server,
    "stripe": stripe_server.build_server,
    "salesforce": salesforce_server.build_server,
    "slack": slack_server.build_server,
    "mail": mail_server.build_server,
}


def _fake_backends(company: Company, channel: str) -> dict[str, Any]:
    return {
        "zendesk": zendesk_server.FakeZendeskBackend(company),
        "stripe": stripe_server.FakeStripeBackend(company),
        "salesforce": salesforce_server.FakeSalesforceBackend(company),
        "slack": slack_server.FakeSlackBackend(channel),
        "mail": mail_server.FakeMailBackend(),
    }


def _fake_mailhog(mail: mail_server.FakeMailBackend) -> Callable[[str], bytes]:
    """A Mailhog ``/api/v2/messages`` view of what the fake mail backend sent."""

    def fetch(url: str) -> bytes:
        items = []
        for m in mail.sent:
            rcpts = []
            for addr in [*m["to"], *m["cc"]]:
                box, _, domain = str(addr).partition("@")
                rcpts.append({"Mailbox": box, "Domain": domain})
            items.append({"To": rcpts, "Content": {"Headers": {"Subject": [m["subject"]]}}})
        return orjson.dumps({"total": len(items), "items": items})

    return fetch


@dataclass(slots=True)
class _Backends:
    backends: dict[str, Any]
    checks: CheckContext


def _check_context(backends: Mapping[str, Any], mailhog_http: str) -> CheckContext:
    ctx = CheckContext(
        mailhog_http=mailhog_http,
        stripe=backends["stripe"],
        salesforce=backends["salesforce"],
        slack=backends["slack"],
        zendesk=backends["zendesk"],
    )
    mail = backends["mail"]
    if isinstance(mail, mail_server.FakeMailBackend):
        ctx.fetch = _fake_mailhog(mail)
    return ctx


def _fake_servers(company: Company, channel: str) -> dict[str, MCPServer]:
    return {n: _BUILDERS[n](be) for n, be in _fake_backends(company, channel).items()}


def _seed_live(
    cfg: SandboxConfig, company: Company, out_dir: Path, budget: ApiBudget
) -> dict[str, str]:
    """Idempotent seeding of the live org (ids cached in ``out_dir/id_map.json``)."""
    from agentfw.sandbox.seed.salesforce_seed import seed_salesforce
    from agentfw.sandbox.seed.stripe_seed import seed_stripe
    from agentfw.sandbox.seed.zendesk_seed import seed_zendesk

    cache = out_dir / ID_MAP
    if cache.exists():
        ids: dict[str, str] = orjson.loads(cache.read_bytes())
    else:
        ids = {**seed_stripe(cfg, company), **seed_salesforce(cfg, company, budget)}
        _write_atomic(cache, orjson.dumps(ids, option=_JSONL))
    seed_zendesk(cfg.data_dir / zendesk_server.DB_NAME, company, ids)
    return ids


def _live_backends(cfg: SandboxConfig, budget: ApiBudget) -> dict[str, Any]:
    return {
        "zendesk": zendesk_server.LiveZendeskBackend(cfg),
        "stripe": stripe_server.LiveStripeBackend(cfg),
        "salesforce": salesforce_server.LiveSalesforceBackend(cfg, budget=budget),
        "slack": slack_server.LiveSlackBackend.from_config(cfg),
        "mail": mail_server.LiveMailBackend(cfg),
    }


class _Allowlist:
    """The agent's view of the proxy: only its allowlisted tools are listed or callable."""

    def __init__(self, inner: McpLike, allowed: Collection[str]) -> None:
        self._inner = inner
        self._allowed = frozenset(allowed)

    async def list_tools(self) -> ListToolsResult:
        res = await self._inner.list_tools()
        return ListToolsResult(tools=[t for t in res.tools if t.name in self._allowed])

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> CallToolResult:
        if name not in self._allowed:
            return error_result(f"tool {name!r} is not allowed for this agent")
        return await self._inner.call_tool(name, arguments)


# --- file helpers --------------------------------------------------------------------------------


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _read_ledger(out_dir: Path) -> list[dict[str, Any]]:
    path = out_dir / LEDGER
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_bytes().splitlines():
        try:
            rows.append(orjson.loads(line))
        except orjson.JSONDecodeError:  # a line torn by a crash mid-write
            logger.warning("skipping a corrupt ledger line in %s", path)
    return rows


def _latest(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {r["session_id"]: r for r in rows}


def _describe(exc: BaseException) -> str:
    """``Type: message`` of the root cause (MCP's task groups wrap it in exception groups)."""
    while isinstance(exc, BaseExceptionGroup) and len(exc.exceptions) == 1:
        exc = exc.exceptions[0]
    return f"{type(exc).__name__}: {exc}"


def _is_subsequence(want: Sequence[str], got: Sequence[str]) -> bool:
    it = iter(got)
    return all(any(g == w for g in it) for w in want)


def _evaluate(rendered: RenderedSession, ctx: CheckContext) -> tuple[int, list[str]]:
    failed: list[str] = []
    for spec in rendered.success:
        desc = f"{spec.kind} " + " ".join(f"{k}={v}" for k, v in sorted(spec.params.items()))
        try:
            ok = SuccessCheck(spec.kind, dict(spec.params)).evaluate(ctx)
        except Exception as exc:  # boundary: a broken check is a failed check, not a crash
            ok = False
            desc += f" ({type(exc).__name__}: {exc})"
        if not ok:
            failed.append(desc)
    return len(rendered.success), failed


def _final_labels(raws: Sequence[RawCall], attacks: Sequence[Label]) -> list[Label]:
    """Exactly one label per envelope, in call order: its attack label, else benign."""
    by_id: dict[str, Label] = {}
    known = {r.envelope_id for r in raws}
    for lab in attacks:
        if lab.envelope_id not in known:
            logger.warning("dropping a label for unknown envelope %s", lab.envelope_id)
            continue
        by_id.setdefault(lab.envelope_id, lab)
    return [
        by_id.get(r.envelope_id) or Label(envelope_id=r.envelope_id, label="benign") for r in raws
    ]


def _transcript_json(sid: str, tr: Transcript) -> bytes:
    return orjson.dumps(
        {
            "session_id": sid,
            "messages": tr.messages,
            "steps": tr.steps,
            "stopped": tr.stopped,
            "usage": {
                "prompt_tokens": tr.usage.prompt_tokens,
                "completion_tokens": tr.usage.completion_tokens,
            },
        },
        option=_JSONL,
    )


# --- execution -----------------------------------------------------------------------------------


@dataclass(slots=True)
class _Runner:
    cfg: SandboxConfig
    out_dir: Path
    mode: Literal["live", "fake"]
    company: Company
    llm: ChatClient | None
    live: dict[str, Any] | None
    meter: UsageMeter
    sem: anyio.Semaphore

    def _append_ledger(self, row: dict[str, Any]) -> None:
        with (self.out_dir / LEDGER).open("ab") as f:
            f.write(orjson.dumps(row, option=_JSONL) + b"\n")

    def _session_backends(self) -> _Backends:
        if self.live is not None:
            backends = self.live
        else:
            backends = _fake_backends(self.company, self.cfg.slack_channel_id)
        return _Backends(backends, _check_context(backends, self.cfg.mailhog_http))

    @contextmanager
    def _plant(
        self, prep: PreparedSession, sb: _Backends, info: dict[str, Any]
    ) -> Iterator[PlantHandle | None]:
        inj = prep.plan.injection
        if inj is None or inj.level != "prompt":
            yield None
            return
        injector = PromptInjector(sb.backends)
        try:
            handle = injector.plant(inj, self.company, record_key=prep.carrier_key)
        except (TypeError, ValueError) as exc:  # e.g. a live backend without snapshot/restore
            logger.warning("%s: prompt injection skipped: %s", prep.plan.session_id, exc)
            info["skipped"] = f"{type(exc).__name__}: {exc}"
            yield None
            return
        try:
            yield handle
        finally:
            injector.cleanup(handle)

    async def run_guarded(self, prep: PreparedSession, tokens: int) -> None:
        sid = prep.plan.session_id
        try:
            await self._run_one(prep)
        except Exception as exc:  # boundary: one broken session must not end the run
            logger.exception("session %s failed", sid)
            shutil.rmtree(self.out_dir / WORK_DIR / sid, ignore_errors=True)
            self._append_ledger(
                {
                    "session_id": sid,
                    "agent_id": prep.plan.agent_id,
                    "status": "failed",
                    "error": _describe(exc),
                    "usd": 0.0,
                }
            )
        finally:
            for _ in range(tokens):
                self.sem.release()

    async def _run_one(self, prep: PreparedSession) -> None:
        plan, agent = prep.plan, prep.agent
        sid = plan.session_id
        work = self.out_dir / WORK_DIR / sid
        shutil.rmtree(work, ignore_errors=True)  # a crashed earlier attempt is replaced
        sink = JsonlSink(work)
        sb = self._session_backends()
        inj = plan.injection
        info: dict[str, Any] | None = None
        mutator: ActionInjector | None = None
        if inj is not None:
            info = {**inj.model_dump(), "attempted": False, "succeeded": False, "skipped": None}
        if inj is not None and inj.level == "action":
            mutator = ActionInjector(
                inj,
                np.random.default_rng([plan.seed, 2]),
                _fake_servers(self.company, self.cfg.slack_channel_id),
            )
        session_meter = UsageMeter(self.cfg.price_in_per_m, self.cfg.price_out_per_m)
        with self._plant(prep, sb, info or {}) as handle:
            async with AsyncExitStack() as stack:
                downstreams = {
                    name: await stack.enter_async_context(Client(_BUILDERS[name](be)))
                    for name, be in sb.backends.items()
                }
                proxy = FirewallProxy(
                    downstreams,
                    SessionInfo(sid, agent.agent_id, f"{agent.agent_id}-principal"),
                    sink,
                    mutator=mutator,
                    rng=np.random.default_rng([plan.seed, 4]),
                )
                client = await stack.enter_async_context(Client(proxy.build_server()))
                llm = self.llm if self.llm is not None else scripted_client(prep)
                try:
                    transcript = await run_session(
                        llm,
                        _Allowlist(client, agent.tools),
                        agent.system_prompt,
                        prep.rendered.prompt,
                        max_steps=prep.rendered.max_steps,
                        meter=session_meter,
                    )
                finally:
                    self.meter.usd += session_meter.usd
            total, failed = await anyio.to_thread.run_sync(_evaluate, prep.rendered, sb.checks)
            calls_path = work / CALLS
            raws = (
                [RawCall.model_validate_json(x) for x in calls_path.read_bytes().splitlines()]
                if calls_path.exists()
                else []
            )
            labels_path = work / LABELS
            attacks = (
                [Label.model_validate_json(x) for x in labels_path.read_bytes().splitlines()]
                if labels_path.exists()
                else []
            )
            if info is not None and mutator is not None:
                info["attempted"], info["succeeded"] = True, mutator.applied
            if info is not None and handle is not None:
                prompt_labels, ok = label_prompt_session(
                    raws, handle.payload, handle.plan.attack_id, handle.planted_at_ms
                )
                attacks += prompt_labels
                info["attempted"], info["succeeded"] = True, ok
        labels = _final_labels(raws, attacks)
        _write_atomic(
            self.out_dir / RAW_DIR / f"{sid}.jsonl",
            calls_path.read_bytes() if calls_path.exists() else b"",
        )
        _write_atomic(
            self.out_dir / LABELS_DIR / f"{sid}.jsonl",
            b"".join(
                orjson.dumps(lab.model_dump(mode="json"), option=_JSONL) + b"\n" for lab in labels
            ),
        )
        _write_atomic(
            self.out_dir / TRANSCRIPTS_DIR / f"{sid}.json", _transcript_json(sid, transcript)
        )
        tools = [r.tool_id for r in raws]
        self._append_ledger(
            {
                "session_id": sid,
                "agent_id": agent.agent_id,
                "status": "completed",
                "stopped": transcript.stopped,
                "steps": transcript.steps,
                "calls": len(raws),
                "sf_calls": sum(t.startswith("salesforce.") for t in tools),
                "usd": session_meter.usd,
                "template_ids": plan.template_ids,
                "records": prep.records,
                "reused_records": prep.reused,
                "injection": info,
                "expected_match": _is_subsequence(prep.rendered.expected_tools, tools),
                "checks": {"total": total, "passed": total - len(failed), "failed": failed},
            }
        )
        shutil.rmtree(work, ignore_errors=True)


def _stop_reason(
    meter: UsageMeter, cfg: SandboxConfig, sf_budget: ApiBudget | None
) -> StopReason | None:
    if meter.usd >= cfg.llm_usd_budget:
        return "llm_usd_budget"
    if sf_budget is not None and sf_budget.remaining() <= 0:
        return "sf_api_budget"
    return None


def _summarize(
    out_dir: Path, plans: Sequence[SessionPlan], resumed: int, stopped: StopReason | None
) -> RunSummary:
    rows = _read_ledger(out_dir)
    ids = {p.session_id for p in plans}
    latest = {sid: r for sid, r in _latest(rows).items() if sid in ids}
    done = [r for r in latest.values() if r["status"] == "completed"]
    return RunSummary(
        completed=len(done),
        failed=sum(r["status"] == "failed" for r in latest.values()),
        resumed=resumed,
        usd=sum(float(r.get("usd", 0.0)) for r in rows),
        sf_calls=sum(int(r["sf_calls"]) for r in done),
        actions=sum(int(r["calls"]) for r in done),
        attacks_attempted=sum(bool(r["injection"] and r["injection"]["attempted"]) for r in done),
        attacks_succeeded=sum(bool(r["injection"] and r["injection"]["succeeded"]) for r in done),
        stopped_reason=stopped,
    )


async def execute(
    plans: Sequence[SessionPlan],
    cfg: SandboxConfig,
    out_dir: Path,
    *,
    backends: Literal["live", "fake"] = "live",
    llm: ChatClient | None = None,
    sf_budget: ApiBudget | None = None,
) -> RunSummary:
    """Run (or resume) ``plans`` into ``out_dir``; see the module docs. ``llm`` overrides the
    agent model for every session (default: scripted policy in fake mode, the configured
    OpenAI-compatible endpoint in live mode). ``sf_budget`` defaults to the config's persisted
    Salesforce budget in live mode and to none (fake calls are not billed) in fake mode."""
    out_dir = Path(out_dir)
    for d in (RAW_DIR, LABELS_DIR, TRANSCRIPTS_DIR, WORK_DIR):
        (out_dir / d).mkdir(parents=True, exist_ok=True)
    templates = load_templates(TASKS_DIR)
    agents = {a.agent_id: a for a in load_agents(AGENTS_DIR)}
    company = build_company(COMPANY_SEED)
    live: dict[str, Any] | None = None
    id_map: dict[str, str] | None = None
    if backends == "live":
        if sf_budget is None:
            sf_budget = ApiBudget(cfg.data_dir / "sf_api_budget.json", cfg.sf_daily_api_budget)
        budget = sf_budget
        id_map = await anyio.to_thread.run_sync(lambda: _seed_live(cfg, company, out_dir, budget))
        live = _live_backends(cfg, budget)
        if llm is None:
            llm = OpenAICompatClient(
                cfg.llm_base_url, cfg.llm_api_key.get_secret_value(), cfg.llm_model
            )
    prepared = _prepare_all(plans, templates, agents, company, id_map, cfg.slack_channel_id)
    rows = _read_ledger(out_dir)
    done = {sid for sid, r in _latest(rows).items() if r["status"] == "completed"}
    meter = UsageMeter(cfg.price_in_per_m, cfg.price_out_per_m)
    meter.usd = sum(float(r.get("usd", 0.0)) for r in rows)
    sem = anyio.Semaphore(cfg.max_concurrency)
    runner = _Runner(cfg, out_dir, backends, company, llm, live, meter, sem)
    stopped: StopReason | None = None
    async with anyio.create_task_group() as tg:
        for prep in prepared:
            if prep.plan.session_id in done:
                continue
            inj = prep.plan.injection
            alone = live is not None and inj is not None and inj.level == "prompt"
            tokens = cfg.max_concurrency if alone else 1
            for _ in range(tokens):
                await sem.acquire()
            stopped = _stop_reason(meter, cfg, sf_budget)
            if stopped is not None:
                for _ in range(tokens):
                    sem.release()
                break
            tg.start_soon(runner.run_guarded, prep, tokens)
    resumed = len(done & {p.session_id for p in plans})
    summary = _summarize(out_dir, plans, resumed, stopped)
    _write_atomic(out_dir / SUMMARY, summary.model_dump_json(indent=2).encode())
    return summary


# --- configuration and CLI -----------------------------------------------------------------------


def fake_config(**overrides: Any) -> SandboxConfig:
    """An offline ``SandboxConfig``: placeholder endpoints and secrets (never used by fake
    backends or the scripted agent), zero prices and an unlimited LLM budget."""
    values: dict[str, Any] = {
        "llm_base_url": "http://localhost:0/v1",
        "llm_api_key": SecretStr("offline"),
        "llm_model": "scripted",
        "judge_base_url": "http://localhost:0/v1",
        "judge_api_key": SecretStr("offline"),
        "judge_model": "scripted",
        "stripe_secret_key": SecretStr("sk_test_offline"),
        "sf_username": "offline@sandbox.example",
        "sf_password": SecretStr("offline"),
        "sf_security_token": SecretStr("offline"),
        "slack_bot_token": SecretStr("offline"),
        "slack_channel_id": "C0SANDBOX",
        "llm_usd_budget": math.inf,
        "price_in_per_m": 0.0,
        "price_out_per_m": 0.0,
    }
    values.update(overrides)
    return SandboxConfig(**values)


def _rate(s: str) -> float:
    v = float(s)
    if not 0.0 <= v <= 1.0:
        raise argparse.ArgumentTypeError(f"must be in [0, 1], got {s}")
    return v


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m agentfw.sandbox.run", description=__doc__)
    p.add_argument("--sessions", type=int, required=True, help="number of sessions to plan")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", type=Path, required=True, help="run directory (resumable)")
    p.add_argument("--fake", action="store_true", help="fake backends + scripted agent (offline)")
    p.add_argument("--action-rate", type=_rate, default=AttackMix().action_rate)
    p.add_argument("--prompt-rate", type=_rate, default=AttackMix().prompt_rate)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    mix = AttackMix(action_rate=args.action_rate, prompt_rate=args.prompt_rate)
    plans = plan_sessions(
        load_agents(AGENTS_DIR), load_templates(TASKS_DIR), args.sessions, mix, args.seed
    )
    if args.fake:
        cfg = fake_config(data_dir=args.out / "sandbox_data")
        mode: Literal["live", "fake"] = "fake"
    else:
        cfg = SandboxConfig.from_env(os.environ)
        mode = "live"
    summary = anyio.run(lambda: execute(plans, cfg, args.out, backends=mode))
    sys.stdout.write(summary.model_dump_json(indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
