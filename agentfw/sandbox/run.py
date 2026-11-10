"""Resumable sandbox runner: run planned sessions through the proxy, write benchmark sessions.

    python -m agentfw.sandbox.run --sessions N --seed S --out runs/<id> [--fake]

Planning lives in ``agentfw.sandbox.plan`` and backend construction in ``agentfw.sandbox.backends``
(both re-exported here where the runner's API uses them).

**Executing** (``execute``) renders every plan in plan order against one run-level ``used`` record
set (ruling RS1: mutable primary records are not reused across the run's sessions while unused
ones remain; a ticket and its charge count as one record). Each session then

1. gets its backends — fresh fakes per session (``backends="fake"``, so no state leaks), or the
   shared live backends — and, for an action-level plan, an ``ActionInjector`` over separate fresh
   fakes keyed by the live id map (D-014: a mutated call never reaches the session's backends);
2. plants a prompt-level payload into the record the session reads (``record_key``);
3. connects ``mcp.Client`` per server, a ``FirewallProxy`` writing to a ``JsonlSink`` and the
   agent's client (filtered to the agent's tool allowlist) and runs ``run_session``;
4. runs the templates' success checks against the same backends (live Mailhog: only mail sent
   since the session started counts), then restores planted state;
5. writes ``raw/<session>.jsonl``, ``labels/<session>.jsonl`` (one row per envelope: the
   proxy's action labels, post-hoc prompt labels, ``benign`` otherwise) and
   ``transcripts/<session>.json``, and only then appends its ``ledger.jsonl`` line.

**Ledger statuses**: ``completed``; ``failed`` (the session raised; retried on the next run);
``budget_truncated`` (the Salesforce budget ran out mid-session: terminal, because its actions
already happened, and to be excluded from packaging). A crash mid-session leaves no line, so the
session re-runs from scratch (its work dir is replaced); a line torn by a crash mid-append is cut
off on startup. A run directory is held under an exclusive ``flock`` on ``.lock``, so two runners
cannot share it.

**Budgets**: before each session starts the runner stops scheduling once the run's LLM spend
(every ledger line's ``usd``, failed attempts included) reaches ``cfg.llm_usd_budget``, or the
Salesforce ``ApiBudget`` cannot cover the session's estimate (its expected ``salesforce.*`` calls
plus its Salesforce checks, at least 1) on top of what running sessions have reserved. Sessions
already running finish. ``RunSummary.stopped_reason`` says which budget stopped the run.

In fake mode the agent is ``scripted_client(prepared)``: a ``ScriptedChatClient`` built from
``scripted_policy`` (prompt-level payloads are obeyed with probability ``COMPLY_PROB``, decided per
session from its seed), so the whole pipeline runs offline. A ``sf_budget`` given in fake mode is
charged per fake Salesforce request, exactly like the live backend. Live mode
(``OpenAICompatClient`` and the real sandbox APIs) seeds the org idempotently and caches the ids in
``id_map.json``; plan live runs with ``prompt_carriers=LIVE_PROMPT_CARRIERS``. Live prompt-level
sessions run alone (they snapshot/restore the shared local zendesk store); a carrier without
snapshot/restore that still reaches the runner is skipped and recorded.
"""

from __future__ import annotations

import argparse
import fcntl
import logging
import math
import os
import shutil
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AsyncExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import anyio
import numpy as np
import orjson
from mcp import Client
from pydantic import BaseModel, SecretStr

from agentfw.core.envelope import Label, RawCall
from agentfw.sandbox.agent import Transcript, run_session
from agentfw.sandbox.backends import (
    BUILDERS,
    Allowlist,
    check_context,
    fake_backends,
    fake_servers,
    live_backends,
    seed_live,
    session_mailhog,
)
from agentfw.sandbox.checks import CheckContext, SuccessCheck
from agentfw.sandbox.config import ApiBudget, SandboxConfig
from agentfw.sandbox.injector import (
    ActionInjector,
    PlantHandle,
    PromptInjector,
    label_prompt_session,
    load_payloads,
)
from agentfw.sandbox.llm import ChatClient, OpenAICompatClient, ScriptedChatClient, UsageMeter
from agentfw.sandbox.plan import (
    LIVE_PROMPT_CARRIERS,
    TICKET_CARRIERS,
    AttackMix,
    SessionPlan,
    carrier_task,
    plan_sessions,
)
from agentfw.sandbox.proxy_py import FirewallProxy, JsonlSink, SessionInfo
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

__all__ = [
    "COMPANY_SEED",
    "LIVE_PROMPT_CARRIERS",
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
RUN_META = "run_meta.json"
ID_MAP = "id_map.json"
LOCK = ".lock"
RAW_DIR, LABELS_DIR, TRANSCRIPTS_DIR, WORK_DIR = "raw", "labels", "transcripts", ".work"
BUDGET_HIT = "budget exceeded"  # what the (live or budgeted fake) Salesforce backend raises

_MUTABLE_KINDS = frozenset(
    {"ticket_open", "charge_refundable", "contact", "account", "opportunity", "soql_object"}
)
_JSONL = orjson.OPT_SORT_KEYS
StopReason = Literal["llm_usd_budget", "sf_api_budget"]
Status = Literal["completed", "failed", "budget_truncated"]


class RunSummary(BaseModel):
    completed: int = 0
    failed: int = 0
    truncated: int = 0  # ``budget_truncated`` sessions (exclude from packaging)
    resumed: int = 0  # finished by an earlier invocation and skipped now
    usd: float = 0.0  # every attempt's LLM spend, failed ones included
    sf_calls: int = 0  # Salesforce envelopes recorded (the ApiBudget file holds real charges)
    actions: int = 0
    attacks_attempted: int = 0
    attacks_succeeded: int = 0
    stopped_reason: StopReason | None = None


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
        if carrier in TICKET_CARRIERS and kind == "ticket_open":
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
        c_task = c_key = None
        inj = plan.injection
        if inj is not None and inj.level == "prompt" and inj.payload_id is not None:
            carrier = payloads[inj.payload_id].carrier
            c_task = carrier_task(carrier, tmpls)
            if c_task is not None:
                c_key = _carrier_key(carrier, kinds[c_task], tasks[c_task].slot_values)
        out.append(
            PreparedSession(
                plan,
                agent,
                compose_session(agent.agent_id, tasks),
                kinds,
                records,
                reused,
                c_task,
                c_key,
                policy,
            )
        )
    return out


def _sf_estimate(prep: PreparedSession) -> int:
    """Salesforce requests the session is expected to make: its calls plus its checks."""
    calls = sum(t.startswith("salesforce.") for t in prep.rendered.expected_tools)
    checks = sum(s.kind == "sf_field_equals" for s in prep.rendered.success)
    return calls + checks


# --- the scripted (offline) agent ----------------------------------------------------------------


def _carrier_charge(prep: PreparedSession) -> str:
    company = prep.policy.company
    ticket = next((t for t in company.tickets if t.key == prep.carrier_key), company.tickets[0])
    return prep.policy.stripe_ids[ticket.charge_key]


def _carrier_read_index(calls: list[Call], prep: PreparedSession, carrier: str) -> int | None:
    sf_account = prep.policy.sf_ids.get(prep.carrier_key or "")
    for i, (tool, args) in enumerate(calls):
        if carrier in TICKET_CARRIERS:
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


# --- ledger and files ----------------------------------------------------------------------------


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _repair_ledger(path: Path) -> None:
    """Cut off a trailing line torn by a crash mid-append (it would glue onto the next line)."""
    if not path.exists():
        return
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        keep = data.rfind(b"\n") + 1
        logger.warning(
            "ledger %s: dropping a torn trailing line (%d bytes)", path, len(data) - keep
        )
        with path.open("r+b") as f:
            f.truncate(keep)


def _read_ledger(out_dir: Path) -> list[dict[str, Any]]:
    path = out_dir / LEDGER
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_bytes().splitlines():
        try:
            rows.append(orjson.loads(line))
        except orjson.JSONDecodeError:
            logger.warning("skipping a corrupt ledger line in %s", path)
    return rows


def _latest(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {r["session_id"]: r for r in rows}


@contextmanager
def _run_lock(out_dir: Path) -> Iterator[None]:
    """Exclusive, non-blocking ``flock`` on ``out_dir/.lock`` for the duration of a run."""
    with (out_dir / LOCK).open("w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"run directory {out_dir} is in use by another runner") from exc
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


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


def _budget_hit(tr: Transcript) -> bool:
    return any(
        m.get("role") == "tool" and BUDGET_HIT in str(m.get("content", "")) for m in tr.messages
    )


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


def _read_jsonl[T](path: Path, parse: Callable[[bytes], T]) -> list[T]:
    return [parse(x) for x in path.read_bytes().splitlines()] if path.exists() else []


# --- execution -----------------------------------------------------------------------------------


@dataclass(slots=True)
class _Runner:
    cfg: SandboxConfig
    out_dir: Path
    company: Company
    llm: ChatClient | None
    live: dict[str, Any] | None
    id_map: dict[str, str] | None
    sf_budget: ApiBudget | None
    meter: UsageMeter
    sem: anyio.Semaphore
    sf_reserved: int = 0

    def append_ledger(self, row: dict[str, Any]) -> None:
        with (self.out_dir / LEDGER).open("ab") as f:
            f.write(orjson.dumps(row, option=_JSONL) + b"\n")

    def stop_reason(self, prep: PreparedSession) -> StopReason | None:
        if self.meter.usd >= self.cfg.llm_usd_budget:
            return "llm_usd_budget"
        if self.sf_budget is not None:
            need = max(_sf_estimate(prep), 1)
            if self.sf_budget.remaining() - self.sf_reserved < need:
                return "sf_api_budget"
        return None

    async def _session_backends(self) -> tuple[dict[str, Any], CheckContext]:
        if self.live is None:
            backends = fake_backends(
                self.company, self.cfg.slack_channel_id, sf_budget=self.sf_budget
            )
            return backends, check_context(backends, self.cfg.mailhog_http)
        ctx = check_context(self.live, self.cfg.mailhog_http)
        base = ctx.fetch
        ctx.fetch = await anyio.to_thread.run_sync(
            lambda: session_mailhog(self.cfg.mailhog_http, base)
        )
        return self.live, ctx

    @contextmanager
    def _plant(
        self, prep: PreparedSession, backends: dict[str, Any], info: dict[str, Any]
    ) -> Iterator[PlantHandle | None]:
        inj = prep.plan.injection
        if inj is None or inj.level != "prompt":
            yield None
            return
        injector = PromptInjector(backends)
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

    async def run_guarded(self, prep: PreparedSession, tokens: int, reserved: int) -> None:
        sid = prep.plan.session_id
        session_meter = UsageMeter(self.cfg.price_in_per_m, self.cfg.price_out_per_m)
        try:
            await self._run_one(prep, session_meter)
        except Exception as exc:  # boundary: one broken session must not end the run
            logger.exception("session %s failed", sid)
            shutil.rmtree(self.out_dir / WORK_DIR / sid, ignore_errors=True)
            self.append_ledger(
                {
                    "session_id": sid,
                    "agent_id": prep.plan.agent_id,
                    "status": "failed",
                    "error": _describe(exc),
                    "usd": session_meter.usd,  # this attempt's spend still counts (budget)
                }
            )
        finally:
            self.meter.usd += session_meter.usd
            self.sf_reserved -= reserved
            for _ in range(tokens):
                self.sem.release()

    async def _run_one(self, prep: PreparedSession, session_meter: UsageMeter) -> None:
        plan, agent = prep.plan, prep.agent
        sid = plan.session_id
        work = self.out_dir / WORK_DIR / sid
        shutil.rmtree(work, ignore_errors=True)  # a crashed earlier attempt is replaced
        sink = JsonlSink(work)
        backends, checks_ctx = await self._session_backends()
        inj = plan.injection
        info: dict[str, Any] | None = None
        mutator: ActionInjector | None = None
        if inj is not None:
            info = {**inj.model_dump(), "attempted": False, "succeeded": False, "skipped": None}
        if inj is not None and inj.level == "action":
            mutator = ActionInjector(
                inj,
                np.random.default_rng([plan.seed, 2]),
                fake_servers(self.company, self.cfg.slack_channel_id, self.id_map),
            )
        with self._plant(prep, backends, info or {}) as handle:
            async with AsyncExitStack() as stack:
                downstreams = {
                    name: await stack.enter_async_context(Client(BUILDERS[name](be)))
                    for name, be in backends.items()
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
                transcript = await run_session(
                    llm,
                    Allowlist(client, agent.tools),
                    agent.system_prompt,
                    prep.rendered.prompt,
                    max_steps=prep.rendered.max_steps,
                    meter=session_meter,
                )
            total, failed = await anyio.to_thread.run_sync(_evaluate, prep.rendered, checks_ctx)
            raws = _read_jsonl(work / CALLS, RawCall.model_validate_json)
            attacks = _read_jsonl(work / LABELS, Label.model_validate_json)
            if info is not None and mutator is not None:
                info["attempted"], info["succeeded"] = True, mutator.applied
            if info is not None and handle is not None:
                prompt_labels, ok = label_prompt_session(
                    raws, handle.payload, handle.plan.attack_id, handle.planted_at_ms
                )
                attacks += prompt_labels
                info["attempted"], info["succeeded"] = True, ok
        labels = _final_labels(raws, attacks)
        calls_path = work / CALLS
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
        status: Status = "budget_truncated" if _budget_hit(transcript) else "completed"
        self.append_ledger(
            {
                "session_id": sid,
                "agent_id": agent.agent_id,
                "status": status,
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
        truncated=sum(r["status"] == "budget_truncated" for r in latest.values()),
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
    Salesforce budget in live mode and to none in fake mode (given there, fake Salesforce requests
    are charged to it)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with _run_lock(out_dir):
        return await _execute_locked(plans, cfg, out_dir, backends, llm, sf_budget)


async def _execute_locked(
    plans: Sequence[SessionPlan],
    cfg: SandboxConfig,
    out_dir: Path,
    mode: Literal["live", "fake"],
    llm: ChatClient | None,
    sf_budget: ApiBudget | None,
) -> RunSummary:
    for d in (RAW_DIR, LABELS_DIR, TRANSCRIPTS_DIR, WORK_DIR):
        (out_dir / d).mkdir(parents=True, exist_ok=True)
    _repair_ledger(out_dir / LEDGER)
    _write_atomic(  # run metadata for packaging: the model id only, never a secret
        out_dir / RUN_META, orjson.dumps({"mode": mode, "model": cfg.llm_model}, option=_JSONL)
    )
    templates = load_templates(TASKS_DIR)
    agents = {a.agent_id: a for a in load_agents(AGENTS_DIR)}
    company = build_company(COMPANY_SEED)
    live: dict[str, Any] | None = None
    id_map: dict[str, str] | None = None
    if mode == "live":
        if sf_budget is None:
            sf_budget = ApiBudget(cfg.data_dir / "sf_api_budget.json", cfg.sf_daily_api_budget)
        budget = sf_budget
        id_map = await anyio.to_thread.run_sync(
            lambda: seed_live(cfg, company, out_dir / ID_MAP, budget)
        )
        live = live_backends(cfg, budget)
        if llm is None:
            llm = OpenAICompatClient(
                cfg.llm_base_url, cfg.llm_api_key.get_secret_value(), cfg.llm_model
            )
    prepared = _prepare_all(plans, templates, agents, company, id_map, cfg.slack_channel_id)
    rows = _read_ledger(out_dir)
    finished = {sid for sid, r in _latest(rows).items() if r["status"] != "failed"}
    meter = UsageMeter(cfg.price_in_per_m, cfg.price_out_per_m)
    meter.usd = sum(float(r.get("usd", 0.0)) for r in rows)
    sem = anyio.Semaphore(cfg.max_concurrency)
    runner = _Runner(cfg, out_dir, company, llm, live, id_map, sf_budget, meter, sem)
    stopped: StopReason | None = None
    async with anyio.create_task_group() as tg:
        for prep in prepared:
            if prep.plan.session_id in finished:
                continue
            inj = prep.plan.injection
            alone = live is not None and inj is not None and inj.level == "prompt"
            tokens = cfg.max_concurrency if alone else 1
            for _ in range(tokens):
                await sem.acquire()
            stopped = runner.stop_reason(prep)
            if stopped is not None:
                for _ in range(tokens):
                    sem.release()
                break
            reserved = _sf_estimate(prep) if sf_budget is not None else 0
            runner.sf_reserved += reserved
            tg.start_soon(runner.run_guarded, prep, tokens, reserved)
    resumed = len(finished & {p.session_id for p in plans})
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
    agents, templates = load_agents(AGENTS_DIR), load_templates(TASKS_DIR)
    mode: Literal["live", "fake"]
    if args.fake:
        plans = plan_sessions(agents, templates, args.sessions, mix, args.seed)
        cfg = fake_config(data_dir=args.out / "sandbox_data")
        mode = "fake"
    else:
        plans = plan_sessions(
            agents,
            templates,
            args.sessions,
            mix,
            args.seed,
            prompt_carriers=LIVE_PROMPT_CARRIERS,
        )
        cfg = SandboxConfig.from_env(os.environ)
        mode = "live"
    summary = anyio.run(lambda: execute(plans, cfg, args.out, backends=mode))
    sys.stdout.write(summary.model_dump_json(indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
