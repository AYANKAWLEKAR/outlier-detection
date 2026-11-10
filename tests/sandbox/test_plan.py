"""Session planning: carriers available per backend mode and the R2 action-level fallback."""

from __future__ import annotations

from agentfw.sandbox.injector import load_payloads
from agentfw.sandbox.plan import LIVE_PROMPT_CARRIERS, AttackMix, plan_sessions
from agentfw.sandbox.tasks.schema import AGENTS_DIR, TASKS_DIR, load_agents, load_templates

AGENTS = load_agents(AGENTS_DIR)
TEMPLATES = load_templates(TASKS_DIR)
BY_AGENT = {a.agent_id: a for a in AGENTS}


def test_live_prompt_plans_use_only_carriers_that_can_be_restored() -> None:
    payloads = load_payloads()
    plans = plan_sessions(
        AGENTS, TEMPLATES, 400, AttackMix(), seed=0, prompt_carriers=LIVE_PROMPT_CARRIERS
    )
    prompt = [p.injection for p in plans if p.injection and p.injection.level == "prompt"]
    assert prompt
    for inj in prompt:
        assert inj.payload_id is not None
        assert payloads[inj.payload_id].carrier in LIVE_PROMPT_CARRIERS
    assert {"ticket_body", "ticket_comment"} == LIVE_PROMPT_CARRIERS


def test_prompt_draw_without_a_usable_carrier_falls_back_to_action_level() -> None:
    agent = BY_AGENT["crm-hygiene-a"]  # reads no tickets: no live-restorable carrier
    mix = AttackMix(action_rate=0.0, prompt_rate=1.0)
    live = plan_sessions([agent], TEMPLATES, 30, mix, 0, prompt_carriers=LIVE_PROMPT_CARRIERS)
    assert all(p.injection is not None and p.injection.level == "action" for p in live)
    fake = plan_sessions([agent], TEMPLATES, 30, mix, 0)  # every carrier available
    assert any(p.injection is not None and p.injection.level == "prompt" for p in fake)
