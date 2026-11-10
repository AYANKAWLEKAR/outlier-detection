"""Tier 2 integration (spec 8): 20 real sessions over 5 agents, real model and sandbox APIs.

Skipped unless every ``SandboxConfig.REQUIRED_ENV`` variable is set. LLM runs vary, so the gates
are >= 18/20 rather than 20/20. A session that does not complete (a transient provider or API
failure) is a miss within that budget, not a hard failure. Every miss is written to
``integration_misses.json`` in the run directory (``AFW_LIVE_RUN_DIR``, else pytest's tmp dir).

Every planned session is built only from templates that carry a Mailhog or Stripe check, so the
Mailhog/Stripe gate is measured on all 20 sessions. The run is serial and the runner only counts
mail sent after a session started, so no session's check is satisfied by another session's mail.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import orjson
import pytest

from agentfw.core.envelope import RawCall
from agentfw.core.normalize import SessionTracker, normalize
from agentfw.sandbox.config import SandboxConfig
from agentfw.sandbox.run import SessionPlan, execute
from agentfw.sandbox.tasks.schema import (
    AGENTS_DIR,
    TASKS_DIR,
    AgentDef,
    TaskTemplate,
    load_agents,
    load_templates,
)

pytestmark = [pytest.mark.live, pytest.mark.slow, pytest.mark.anyio]

LIVE_AGENTS = (
    "support-refunds-a",
    "billing-disputes-a",
    "escalations-a",
    "outreach-a",
    "finance-reconcile-a",
)
SESSIONS_PER_AGENT = 4
MIN_OK = 18
MAILHOG_STRIPE = ("mailhog_has_message", "stripe_refund_exists")


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _live_plans(agents: list[AgentDef], templates: dict[str, TaskTemplate]) -> list[SessionPlan]:
    """20 benign sessions, each a queue of templates that all carry a Mailhog/Stripe check."""
    plans: list[SessionPlan] = []
    for agent in agents:
        eligible = sorted(
            t
            for t, w in agent.template_mix.items()
            if w > 0 and any(s.kind in MAILHOG_STRIPE for s in templates[t].success)
        )
        assert eligible, agent.agent_id
        for k in range(SESSIONS_PER_AGENT):
            start = k * agent.batch_size
            tids = [eligible[(start + j) % len(eligible)] for j in range(agent.batch_size)]
            sid = f"live-{agent.agent_id}-{k:02d}"
            plans.append(
                SessionPlan(session_id=sid, agent_id=agent.agent_id, template_ids=tids, seed=k)
            )
    return plans


async def test_twenty_live_sessions(tmp_path: Path) -> None:
    missing = [n.upper() for n in SandboxConfig.REQUIRED_ENV if n.upper() not in os.environ]
    if missing:
        pytest.skip(f"missing env: {missing}")
    cfg = SandboxConfig.from_env().model_copy(update={"max_concurrency": 1})
    templates = load_templates(TASKS_DIR)
    agents = [a for a in load_agents(AGENTS_DIR) if a.agent_id in LIVE_AGENTS]
    assert len(agents) == 5
    plans = _live_plans(agents, templates)
    assert len(plans) == 20
    out = Path(os.environ.get("AFW_LIVE_RUN_DIR", str(tmp_path / "live_run")))
    summary = await execute(plans, cfg, out, backends="live")
    assert summary.stopped_reason is None, summary

    ledger: dict[str, dict[str, Any]] = {
        row["session_id"]: row
        for row in map(orjson.loads, (out / "ledger.jsonl").read_bytes().splitlines())
    }
    misses: list[dict[str, Any]] = []
    expected_ok = checks_ok = 0
    for plan in plans:
        row = ledger.get(plan.session_id, {"status": "missing"})
        if row["status"] != "completed":
            misses.append({"session_id": plan.session_id, "status": row["status"], "row": row})
            continue
        raws = [
            RawCall.model_validate_json(x)
            for x in (out / "raw" / f"{plan.session_id}.jsonl").read_bytes().splitlines()
        ]
        tracker = SessionTracker()
        for raw in raws:  # every envelope normalizes (a hard requirement, not a tolerated miss)
            tracker.observe(normalize(raw, tracker))
        expected_ok += row["expected_match"]
        ms_failed = [f for f in row["checks"]["failed"] if f.startswith(MAILHOG_STRIPE)]
        checks_ok += not ms_failed
        if not row["expected_match"] or ms_failed:
            misses.append(
                {
                    "session_id": plan.session_id,
                    "template_ids": plan.template_ids,
                    "expected_match": row["expected_match"],
                    "failed_checks": row["checks"]["failed"],
                    "tools": [r.tool_id for r in raws],
                    "stopped": row["stopped"],
                }
            )
    (out / "integration_misses.json").write_bytes(
        orjson.dumps(misses, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
    )
    assert expected_ok >= MIN_OK, misses
    assert checks_ok >= MIN_OK, misses
