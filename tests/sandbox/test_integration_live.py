"""Tier 2 integration (spec 8): 20 real sessions over 5 agents, real model and sandbox APIs.

Skipped unless every ``SandboxConfig.REQUIRED_ENV`` variable is set. LLM runs vary, so the gates
are >= 18/20 rather than 20/20; every miss is written to ``integration_misses.json`` for review.
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
from agentfw.sandbox.run import AttackMix, execute, plan_sessions
from agentfw.sandbox.tasks.schema import AGENTS_DIR, TASKS_DIR, load_agents, load_templates

pytestmark = [pytest.mark.live, pytest.mark.slow, pytest.mark.anyio]

# Five agents whose work touches Stripe and/or Mailhog, so the spec's assertions have something
# to confirm.
LIVE_AGENTS = (
    "support-refunds-a",
    "billing-disputes-a",
    "escalations-a",
    "outreach-a",
    "finance-reconcile-a",
)
MIN_OK = 18
MAILHOG_STRIPE = ("mailhog_has_message", "stripe_refund_exists")


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def test_twenty_live_sessions(tmp_path: Path) -> None:
    missing = [n.upper() for n in SandboxConfig.REQUIRED_ENV if n.upper() not in os.environ]
    if missing:
        pytest.skip(f"missing env: {missing}")
    cfg = SandboxConfig.from_env()
    templates = load_templates(TASKS_DIR)
    agents = [a for a in load_agents(AGENTS_DIR) if a.agent_id in LIVE_AGENTS]
    assert len(agents) == 5
    plans = plan_sessions(agents, templates, 20, AttackMix(action_rate=0, prompt_rate=0), seed=0)
    out = tmp_path / "live_run"
    summary = await execute(plans, cfg, out, backends="live")
    assert summary.stopped_reason is None, summary

    ledger = {
        row["session_id"]: row
        for row in map(orjson.loads, (out / "ledger.jsonl").read_bytes().splitlines())
    }
    misses: list[dict[str, Any]] = []
    expected_ok = checks_ok = 0
    for plan in plans:
        row = ledger[plan.session_id]
        assert row["status"] == "completed", row
        raws = [
            RawCall.model_validate_json(x)
            for x in (out / "raw" / f"{plan.session_id}.jsonl").read_bytes().splitlines()
        ]
        tracker = SessionTracker()
        for raw in raws:  # every envelope normalizes
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
    Path("integration_misses.json").write_bytes(
        orjson.dumps(misses, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
    )
    assert expected_ok >= MIN_OK, misses
    assert checks_ok >= MIN_OK, misses
