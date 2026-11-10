"""Offline end-to-end runs: fake backends, scripted agent policy, real proxy and agent loop."""

from __future__ import annotations

import fcntl
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import orjson
import pytest

from agentfw.core.envelope import Label, RawCall
from agentfw.core.normalize import SessionTracker, normalize
from agentfw.sandbox import run as run_mod
from agentfw.sandbox.config import ApiBudget
from agentfw.sandbox.injector import ACTION_TARGETS, label_prompt_session, load_payloads
from agentfw.sandbox.llm import ChatTurn, ScriptedChatClient
from agentfw.sandbox.run import (
    AttackMix,
    RunSummary,
    SessionPlan,
    execute,
    fake_config,
    main,
    plan_sessions,
)
from agentfw.sandbox.scripted_policy import PolicyContext, build_turns, compliant_call
from agentfw.sandbox.seed.company import build_company
from agentfw.sandbox.tasks.schema import AGENTS_DIR, TASKS_DIR, load_agents, load_templates

pytestmark = pytest.mark.anyio

AGENTS = load_agents(AGENTS_DIR)
TEMPLATES = load_templates(TASKS_DIR)
BY_AGENT = {a.agent_id: a for a in AGENTS}
MUTATING = {"magnitude_spike", "prereq_skip", "novel_target", "scope_blowup"}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _ledger(out: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in (out / "ledger.jsonl").read_bytes().splitlines():
        row = orjson.loads(line)
        rows[row["session_id"]] = row  # last line per session wins
    return rows


def _raws(out: Path, sid: str) -> list[RawCall]:
    path = out / "raw" / f"{sid}.jsonl"
    return [RawCall.model_validate_json(x) for x in path.read_bytes().splitlines()]


def _labels(out: Path, sid: str) -> list[Label]:
    path = out / "labels" / f"{sid}.jsonl"
    return [Label.model_validate_json(x) for x in path.read_bytes().splitlines()]


def _is_subsequence(want: Sequence[str], got: Sequence[str]) -> bool:
    it = iter(got)
    return all(any(g == w for g in it) for w in want)


def _expected_tools(plan: SessionPlan) -> list[str]:
    return [t for tid in plan.template_ids for t in TEMPLATES[tid].expected_tools]


# -- planning ---------------------------------------------------------------------------------


def test_plan_sessions_is_deterministic_balanced_and_prefix_stable() -> None:
    plans = plan_sessions(AGENTS, TEMPLATES, 50, AttackMix(), seed=7)
    assert plans == plan_sessions(AGENTS, TEMPLATES, 50, AttackMix(), seed=7)
    assert plans != plan_sessions(AGENTS, TEMPLATES, 50, AttackMix(), seed=8)
    assert plans[:23] == plan_sessions(AGENTS, TEMPLATES, 23, AttackMix(), seed=7)
    assert len({p.session_id for p in plans}) == 50
    per_agent = Counter(p.agent_id for p in plans)
    assert set(per_agent.values()) == {2, 3}  # 50 over 20 agents: even split, remainder first
    first = sorted(BY_AGENT)[:10]
    assert all(per_agent[a] == 3 for a in first)
    for p in plans:
        agent = BY_AGENT[p.agent_id]
        assert len(p.template_ids) == agent.batch_size
        assert all(agent.template_mix[t] > 0 for t in p.template_ids)  # no quarter-end work
        assert not any(TEMPLATES[t].quarter_end_only for t in p.template_ids)


def test_injection_plans_fit_the_session() -> None:
    plans = plan_sessions(AGENTS, TEMPLATES, 400, AttackMix(), seed=0)
    payloads = load_payloads()
    levels = Counter(p.injection.level if p.injection else None for p in plans)
    assert 0.08 < levels["action"] / 400 < 0.22
    assert levels["prompt"] > 0
    for p in plans:
        inj = p.injection
        if inj is None:
            continue
        tools = set(_expected_tools(p))
        assert inj.attack_id
        if inj.level == "action":
            assert inj.target_tool in tools
            if inj.family in ACTION_TARGETS:
                assert inj.target_tool in ACTION_TARGETS[inj.family]
        else:
            assert inj.payload_id is not None
            payload = payloads[inj.payload_id]
            assert payload.family == inj.family
            assert payload.match.tool_id in BY_AGENT[p.agent_id].tools
    none = plan_sessions(AGENTS, TEMPLATES, 40, AttackMix(action_rate=0, prompt_rate=0), seed=0)
    assert all(p.injection is None for p in none)


# -- scripted policy -------------------------------------------------------------------------


def test_compliant_call_triggers_every_payload_match() -> None:
    ctx = PolicyContext.build(build_company(0), None, "C0SANDBOX")
    for payload in load_payloads().values():
        tool, args = compliant_call(payload, ctx, charge_id="ch_carrier")
        raw = RawCall(
            envelope_id="e1",
            session_id="s",
            agent_id="a",
            principal_id="p",
            seq_no=0,
            ts_unix_ms=10,
            tool_id=tool,
            args=args,
        )
        _, ok = label_prompt_session([raw], payload, "atk", planted_at_ms=0)
        assert ok, payload.id


def test_build_turns_one_call_per_step_then_a_final_answer() -> None:
    turns = build_turns([("zendesk.get_ticket", {"ticket_id": "tk_1"}), ("slack.read_channel", {})])
    assert [len(t.tool_calls) for t in turns] == [1, 1, 0]
    assert turns[0].tool_calls[0].name == "zendesk__get_ticket"
    assert orjson.loads(turns[0].tool_calls[0].arguments_json) == {"ticket_id": "tk_1"}
    assert turns[-1].content
    assert all(t.usage.prompt_tokens > 0 for t in turns)


# -- the 20-session offline integration -------------------------------------------------------


async def test_twenty_fake_sessions_end_to_end(tmp_path: Path) -> None:
    """Fake backends + the scripted policy: passing checks here prove the *plumbing* (rendering,
    proxy, envelopes, labels, check wiring) works end to end, not that a real model is competent
    at the tasks; the live integration test measures that."""
    plans = plan_sessions(AGENTS, TEMPLATES, 20, AttackMix(), seed=0)
    out = tmp_path / "run"
    summary = await execute(plans, fake_config(data_dir=tmp_path / "data"), out, backends="fake")

    assert isinstance(summary, RunSummary)
    assert (summary.completed, summary.failed, summary.stopped_reason) == (20, 0, None)
    ledger = _ledger(out)
    assert set(ledger) == {p.session_id for p in plans}
    n_actions = 0
    for plan in plans:
        sid = plan.session_id
        row = ledger[sid]
        assert row["status"] == "completed"
        raws = _raws(out, sid)
        n_actions += len(raws)
        assert [r.seq_no for r in raws] == list(range(len(raws)))
        tracker = SessionTracker()
        for raw in raws:  # every envelope passes the normalizer
            assert raw.session_id == sid and raw.agent_id == plan.agent_id
            tracker.observe(normalize(raw, tracker))
        assert _is_subsequence(_expected_tools(plan), [r.tool_id for r in raws])
        assert row["expected_match"] is True
        labels = _labels(out, sid)
        assert [lab.envelope_id for lab in labels] == [r.envelope_id for r in raws]
        inj = row["injection"]
        mutated = inj is not None and inj["level"] == "action" and inj["family"] in MUTATING
        if not mutated:
            assert row["checks"]["failed"] == [], (sid, row["checks"])
            assert row["checks"]["passed"] == row["checks"]["total"]
        transcript = orjson.loads((out / "transcripts" / f"{sid}.json").read_bytes())
        assert transcript["stopped"] == "final"
    assert summary.actions == n_actions
    assert summary.usd == 0.0  # fake_config prices are zero
    assert summary.attacks_attempted >= 1


async def test_injected_sessions_are_labeled(tmp_path: Path) -> None:
    plans = plan_sessions(AGENTS, TEMPLATES, 20, AttackMix(action_rate=0.5, prompt_rate=0.5), 3)
    out = tmp_path / "run"
    summary = await execute(plans, fake_config(data_dir=tmp_path / "data"), out, backends="fake")
    ledger = _ledger(out)
    succeeded = 0
    levels = set()
    for plan in plans:
        row = ledger[plan.session_id]
        inj = row["injection"]
        labels = _labels(out, plan.session_id)
        attacks = [lab for lab in labels if lab.label == "attack"]
        if inj is None:
            assert attacks == []
            continue
        levels.add(inj["level"])
        if inj["succeeded"]:
            succeeded += 1
            assert attacks
            assert {a.injection_level for a in attacks} == {inj["level"]}
            assert {a.attack_id for a in attacks} == {plan.injection.attack_id}  # type: ignore[union-attr]
        else:
            assert attacks == []
        assert Counter(lab.envelope_id for lab in labels).most_common(1)[0][1] == 1
    assert levels == {"action", "prompt"}
    assert summary.attacks_succeeded == succeeded > 0
    assert summary.attacks_attempted == sum(
        1 for r in ledger.values() if r["injection"] and r["injection"]["attempted"]
    )


async def test_every_template_runs_and_passes_its_checks_on_fakes(tmp_path: Path) -> None:
    plans = [
        SessionPlan(
            session_id=f"tpl-{tid}",
            agent_id=t.roles[0],
            template_ids=[tid],
            injection=None,
            seed=i,
        )
        for i, (tid, t) in enumerate(sorted(TEMPLATES.items()))
        if not t.quarter_end_only
    ]
    cfg = fake_config(data_dir=tmp_path / "data", max_concurrency=8)
    summary = await execute(plans, cfg, tmp_path / "run", backends="fake")
    assert summary.completed == len(plans)
    bad = {
        sid: row["checks"]
        for sid, row in _ledger(tmp_path / "run").items()
        if row["checks"]["failed"] or not row["expected_match"]
    }
    assert bad == {}


async def test_run_level_used_set_avoids_reusing_primary_records(tmp_path: Path) -> None:
    agent = BY_AGENT["support-refunds-a"]
    plans = plan_sessions([agent], TEMPLATES, 4, AttackMix(action_rate=0, prompt_rate=0), 0)
    out = tmp_path / "run"
    await execute(plans, fake_config(data_dir=tmp_path / "data"), out, backends="fake")
    records = [r for row in _ledger(out).values() for r in row["records"]]
    assert len(records) == 4 * agent.batch_size
    assert len(set(records)) == len(records)
    assert all(row["reused_records"] == 0 for row in _ledger(out).values())


# -- resume and budgets -----------------------------------------------------------------------


class _Crash(BaseException):
    """Simulates the process dying mid-session (not an ordinary, caught Exception)."""


class _CrashingClient:
    def __init__(self, inner: ScriptedChatClient, after: int) -> None:
        self.inner, self.after, self.n = inner, after, 0

    async def complete(
        self, messages: list[dict[str, object]], tools: list[dict[str, object]]
    ) -> ChatTurn:
        self.n += 1
        if self.n > self.after:
            raise _Crash("power cut")
        return await self.inner.complete(messages, tools)


async def test_resume_after_crash_skips_completed_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plans = plan_sessions(AGENTS, TEMPLATES, 8, AttackMix(action_rate=0, prompt_rate=0), 1)
    cfg = fake_config(data_dir=tmp_path / "data", max_concurrency=1)
    out = tmp_path / "run"
    crash_sid = plans[5].session_id
    real = run_mod.scripted_client
    built: list[str] = []

    def factory(prepared: Any) -> Any:
        built.append(prepared.plan.session_id)
        client = real(prepared)
        return _CrashingClient(client, after=3) if prepared.plan.session_id == crash_sid else client

    monkeypatch.setattr(run_mod, "scripted_client", factory)
    with pytest.raises(BaseExceptionGroup) as info:
        await execute(plans, cfg, out, backends="fake")
    assert info.group_contains(_Crash)
    done = _ledger(out)
    assert set(done) == {p.session_id for p in plans[:5]}
    assert not (out / "raw" / f"{crash_sid}.jsonl").exists()
    before = {sid: (out / "raw" / f"{sid}.jsonl").read_bytes() for sid in done}

    built.clear()

    def no_crash(prepared: Any) -> Any:
        built.append(prepared.plan.session_id)
        return real(prepared)

    monkeypatch.setattr(run_mod, "scripted_client", no_crash)
    summary = await execute(plans, cfg, out, backends="fake")
    assert built == [p.session_id for p in plans[5:]]
    assert (summary.completed, summary.resumed, summary.failed) == (8, 5, 0)
    for sid, data in before.items():
        assert (out / "raw" / f"{sid}.jsonl").read_bytes() == data
    raws = _raws(out, plans[5].session_id)
    assert [r.seq_no for r in raws] == list(range(len(raws)))  # partial run replaced, not appended
    assert len(_labels(out, plans[5].session_id)) == len(raws)


async def test_usd_budget_stops_scheduling(tmp_path: Path) -> None:
    plans = plan_sessions(AGENTS, TEMPLATES, 5, AttackMix(action_rate=0, prompt_rate=0), 0)
    cfg = fake_config(
        data_dir=tmp_path / "data",
        max_concurrency=1,
        llm_usd_budget=0.001,
        price_in_per_m=1.0,
        price_out_per_m=1.0,
    )
    summary = await execute(plans, cfg, tmp_path / "run", backends="fake")
    assert summary.stopped_reason == "llm_usd_budget"
    assert summary.completed == 1
    assert summary.usd >= 0.001
    # resuming under the same budget schedules nothing new
    again = await execute(plans, cfg, tmp_path / "run", backends="fake")
    assert (again.completed, again.stopped_reason) == (1, "llm_usd_budget")


async def test_sf_budget_stops_scheduling(tmp_path: Path) -> None:
    plans = plan_sessions(AGENTS, TEMPLATES, 3, AttackMix(action_rate=0, prompt_rate=0), 0)
    cfg = fake_config(data_dir=tmp_path / "data", sf_daily_api_budget=5)
    budget = ApiBudget(tmp_path / "data" / "sf_api_budget.json", 5)
    budget.charge(5)
    summary = await execute(plans, cfg, tmp_path / "run", backends="fake", sf_budget=budget)
    assert (summary.completed, summary.stopped_reason) == (0, "sf_api_budget")


async def test_failed_session_is_recorded_and_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plans = plan_sessions(AGENTS, TEMPLATES, 2, AttackMix(action_rate=0, prompt_rate=0), 0)
    cfg = fake_config(data_dir=tmp_path / "data", max_concurrency=1)
    real = run_mod.scripted_client

    def broken(prepared: Any) -> Any:
        if prepared.plan.session_id == plans[0].session_id:
            raise RuntimeError("cannot build policy")
        return real(prepared)

    monkeypatch.setattr(run_mod, "scripted_client", broken)
    summary = await execute(plans, cfg, tmp_path / "run", backends="fake")
    assert (summary.completed, summary.failed) == (1, 1)
    row = _ledger(tmp_path / "run")[plans[0].session_id]
    assert row["status"] == "failed" and "cannot build policy" in row["error"]
    monkeypatch.setattr(run_mod, "scripted_client", real)
    summary = await execute(plans, cfg, tmp_path / "run", backends="fake")
    assert (summary.completed, summary.failed, summary.resumed) == (2, 0, 1)


def test_cli_fake_run(tmp_path: Path) -> None:
    out = tmp_path / "runs" / "r1"
    assert main(["--sessions", "3", "--seed", "0", "--out", str(out), "--fake"]) == 0
    assert len(_ledger(out)) == 3
    summary = RunSummary.model_validate_json((out / "summary.json").read_bytes())
    assert summary.completed == 3


async def test_failed_session_spend_is_recorded_and_counts_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plans = plan_sessions(AGENTS, TEMPLATES, 2, AttackMix(action_rate=0, prompt_rate=0), 0)
    priced: dict[str, Any] = {"price_in_per_m": 1.0, "price_out_per_m": 1.0, "max_concurrency": 1}
    cfg = fake_config(data_dir=tmp_path / "data", **priced)

    def crash(*args: Any) -> Any:
        raise RuntimeError("checks crashed")  # after every scripted turn was spent

    monkeypatch.setattr(run_mod, "_evaluate", crash)
    summary = await execute(plans, cfg, tmp_path / "run", backends="fake")
    rows = _ledger(tmp_path / "run")
    assert all(r["status"] == "failed" and r["usd"] > 0 for r in rows.values())
    spent = sum(r["usd"] for r in rows.values())
    assert summary.usd == pytest.approx(spent)
    # the failed attempts' spend counts against the budget when the run resumes
    monkeypatch.undo()
    tight = fake_config(data_dir=tmp_path / "data", llm_usd_budget=spent * 0.9, **priced)
    again = await execute(plans, tight, tmp_path / "run", backends="fake")
    assert (again.completed, again.stopped_reason) == (0, "llm_usd_budget")


async def test_torn_ledger_line_is_repaired_on_startup(tmp_path: Path) -> None:
    plans = plan_sessions(AGENTS, TEMPLATES, 2, AttackMix(action_rate=0, prompt_rate=0), 0)
    cfg = fake_config(data_dir=tmp_path / "data", max_concurrency=1)
    out = tmp_path / "run"
    await execute(plans, cfg, out, backends="fake")
    with (out / "ledger.jsonl").open("ab") as f:
        f.write(b'{"session_id": "s0-torn", "sta')  # a crash mid-append
    summary = await execute(plans, cfg, out, backends="fake")
    data = (out / "ledger.jsonl").read_bytes()
    assert data.endswith(b"\n") and b"s0-torn" not in data
    assert [orjson.loads(x)["session_id"] for x in data.splitlines()] == [
        p.session_id for p in plans
    ]
    assert (summary.completed, summary.resumed) == (2, 2)


async def test_two_runners_cannot_share_a_run_dir(tmp_path: Path) -> None:
    plans = plan_sessions(AGENTS, TEMPLATES, 1, AttackMix(action_rate=0, prompt_rate=0), 0)
    out = tmp_path / "run"
    out.mkdir()
    with (out / ".lock").open("w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        with pytest.raises(RuntimeError, match="in use"):
            await execute(plans, fake_config(data_dir=tmp_path / "data"), out, backends="fake")
    summary = await execute(plans, fake_config(data_dir=tmp_path / "data"), out, backends="fake")
    assert summary.completed == 1  # the lock is free again once the other holder is gone


def _sf_plan(i: int = 0) -> SessionPlan:
    tid = "reprice_opportunity"  # 3 Salesforce calls + 2 Salesforce checks
    return SessionPlan(
        session_id=f"sf-{i}", agent_id=TEMPLATES[tid].roles[0], template_ids=[tid], seed=i
    )


async def test_fake_salesforce_calls_and_checks_are_charged_to_a_given_budget(
    tmp_path: Path,
) -> None:
    budget = ApiBudget(tmp_path / "data" / "sf_api_budget.json", 100)
    cfg = fake_config(data_dir=tmp_path / "data")
    summary = await execute([_sf_plan()], cfg, tmp_path / "run", backends="fake", sf_budget=budget)
    row = _ledger(tmp_path / "run")["sf-0"]
    assert (row["status"], row["sf_calls"], summary.sf_calls) == ("completed", 3, 3)
    assert budget.remaining() == 100 - 3 - 2


async def test_sf_budget_reserves_a_sessions_estimate_before_starting(tmp_path: Path) -> None:
    budget = ApiBudget(tmp_path / "data" / "sf_api_budget.json", 4)  # the session needs 5
    cfg = fake_config(data_dir=tmp_path / "data")
    summary = await execute([_sf_plan()], cfg, tmp_path / "run", backends="fake", sf_budget=budget)
    assert (summary.completed, summary.stopped_reason) == (0, "sf_api_budget")
    assert budget.remaining() == 4  # nothing was spent


async def test_sf_budget_hit_mid_session_marks_it_truncated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_mod, "_sf_estimate", lambda prep: 0)  # under-estimate on purpose
    budget = ApiBudget(tmp_path / "data" / "sf_api_budget.json", 2)
    cfg = fake_config(data_dir=tmp_path / "data", max_concurrency=1)
    plans = [_sf_plan(0), _sf_plan(1)]
    summary = await execute(plans, cfg, tmp_path / "run", backends="fake", sf_budget=budget)
    rows = _ledger(tmp_path / "run")
    assert rows["sf-0"]["status"] == "budget_truncated"
    assert "sf-1" not in rows
    assert (summary.completed, summary.truncated, summary.stopped_reason) == (
        0,
        1,
        "sf_api_budget",
    )
    # truncated sessions are terminal: a resume does not re-run their (already executed) actions
    again = await execute(plans[:1], cfg, tmp_path / "run", backends="fake", sf_budget=budget)
    assert (again.truncated, again.completed) == (1, 0)


class _FailingLlm:
    """An LLM whose every call after the first ``after`` turns raises an ordinary Exception."""

    def __init__(self, inner: ScriptedChatClient | None = None, after: int = 0) -> None:
        self.inner, self.after, self.n = inner, after, 0

    async def complete(
        self, messages: list[dict[str, object]], tools: list[dict[str, object]]
    ) -> ChatTurn:
        self.n += 1
        if self.inner is None or self.n > self.after:
            raise RuntimeError("provider unavailable")
        return await self.inner.complete(messages, tools)


async def test_llm_error_session_is_failed_not_completed(tmp_path: Path) -> None:
    plans = plan_sessions(AGENTS, TEMPLATES, 3, AttackMix(action_rate=1.0, prompt_rate=0), 0)
    assert all(p.injection is not None for p in plans)
    cfg = fake_config(data_dir=tmp_path / "data", max_concurrency=1)
    out = tmp_path / "run"
    summary = await execute(plans, cfg, out, backends="fake", llm=_FailingLlm())
    assert (summary.completed, summary.failed, summary.attacks_attempted) == (0, 3, 0)
    rows = _ledger(out)
    assert {r["status"] for r in rows.values()} == {"failed"}
    assert all("LLM" in r["error"] for r in rows.values())
    for p in plans:
        assert not (out / "raw" / f"{p.session_id}.jsonl").exists()
        assert not (out / ".work" / p.session_id).exists()
    # failed sessions are retried on the next run
    summary = await execute(plans, cfg, out, backends="fake")
    assert (summary.completed, summary.failed, summary.resumed) == (3, 0, 0)


async def test_llm_error_mid_session_discards_partial_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agentfw.sandbox.package import package

    plans = plan_sessions(AGENTS, TEMPLATES, 3, AttackMix(action_rate=0, prompt_rate=0), 0)
    cfg = fake_config(data_dir=tmp_path / "data", max_concurrency=1)
    out = tmp_path / "run"
    real = run_mod.scripted_client
    bad = plans[1].session_id

    def factory(prepared: Any) -> Any:
        client = real(prepared)
        return _FailingLlm(client, after=2) if prepared.plan.session_id == bad else client

    monkeypatch.setattr(run_mod, "scripted_client", factory)
    summary = await execute(plans, cfg, out, backends="fake")
    assert (summary.completed, summary.failed) == (2, 1)
    assert _ledger(out)[bad]["status"] == "failed"
    assert not (out / "raw" / f"{bad}.jsonl").exists()
    assert not (out / ".work" / bad).exists()
    manifest = package([out], tmp_path / "dataset")
    assert manifest.counts.sessions == 2
    assert manifest.source_runs[0]["excluded"] == {"failed": 1}
