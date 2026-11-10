"""Latency gate (spec §6.11): p99 of ``Engine.score`` < 200 µs on warm keys with <= 20 fields."""

from __future__ import annotations

import os
from time import perf_counter_ns
from typing import Any

import numpy as np
import pytest

from agentfw.core.engine import Engine
from agentfw.core.envelope import ActionEnvelope, Outcome, RawCall, ResponseMeta
from agentfw.core.ids import ulid_from
from agentfw.core.normalize import SessionTracker, normalize

pytestmark = pytest.mark.slow

N_WARM = 5_000
N_SCORED = 10_000
P99_BUDGET_US = 200.0
AGENTS = ("agent_a", "agent_b")
TOOLS = ("stripe.create_refund", "zendesk.update_ticket")
T0 = 1_700_000_000_000


def _args(rng: np.random.Generator, i: int) -> dict[str, Any]:
    """20 fields (flat + derived): numeric, categorical, ids, an email and a bool."""
    return {
        "amount": round(float(rng.lognormal(3.6, 0.4)), 2),
        "fee": round(float(rng.lognormal(1.0, 0.3)), 2),
        "quantity": int(rng.integers(1, 5)),
        "discount_pct": float(rng.uniform(0, 20)),
        "tax": round(float(rng.lognormal(0.5, 0.2)), 2),
        "currency": ["usd", "eur", "gbp"][int(rng.integers(0, 3))],
        "status": ["open", "pending", "closed"][int(rng.integers(0, 3))],
        "priority": ["low", "normal", "high"][int(rng.integers(0, 3))],
        "channel": ["web", "email", "phone"][int(rng.integers(0, 3))],
        "reason": ["duplicate", "fraudulent", "requested_by_customer"][int(rng.integers(0, 3))],
        "region": ["us", "eu"][int(rng.integers(0, 2))],
        "customer_id": f"cus_{int(rng.integers(0, 200)):010d}",  # id + derived __prefix__
        "charge_id": f"ch_{i:012d}",  # fresh id every call, + derived __prefix__
        "email": f"user{int(rng.integers(0, 50))}@acme.example",  # id + derived __domain__
        "notify": bool(rng.integers(0, 2)),
        "lang": ["en", "de"][int(rng.integers(0, 2))],
        "source": ["agent", "api"][int(rng.integers(0, 2))],
    }


def _stream(
    n: int, start: int, rng: np.random.Generator, tracker: SessionTracker
) -> list[ActionEnvelope]:
    out: list[ActionEnvelope] = []
    for k in range(n):
        i = start + k
        agent = AGENTS[i % len(AGENTS)]
        raw = RawCall(
            envelope_id=ulid_from(T0 + i * 7_000, i),
            session_id=f"{agent}-s{i // 20}",
            agent_id=agent,
            principal_id="p1",
            seq_no=(i // len(AGENTS)) % 10,
            ts_unix_ms=T0 + i * 7_000,  # one call every 7 s: minutes roll over regularly
            tool_id=TOOLS[(i // len(AGENTS)) % len(TOOLS)],
            args=_args(rng, i),
            response_meta=ResponseMeta(item_count=int(rng.integers(1, 10))),
        )
        e = normalize(raw, tracker)
        tracker.observe(e)
        out.append(e)
    return out


@pytest.mark.skipif(os.environ.get("AFW_SKIP_PERF") == "1", reason="AFW_SKIP_PERF=1 set")
def test_score_p99_under_budget() -> None:
    rng = np.random.default_rng(0)
    tracker = SessionTracker()
    warm = _stream(N_WARM, 0, rng, tracker)
    scored = _stream(N_SCORED, N_WARM, rng, tracker)
    assert max(len(e.flat_args) + len(e.derived) for e in scored) <= 20

    eng = Engine()
    for e in warm:
        eng.score(e)
        eng.update(e, Outcome.ALLOWED)
    lat = np.empty(N_SCORED)
    for j, e in enumerate(scored):
        t0 = perf_counter_ns()
        eng.score(e)
        lat[j] = (perf_counter_ns() - t0) / 1000.0
        eng.update(e, Outcome.ALLOWED)
    p50, p99 = np.percentile(lat, [50, 99])
    print(f"Engine.score latency: p50={p50:.1f} us p99={p99:.1f} us (n={N_SCORED})")
    assert p99 < P99_BUDGET_US, f"p99={p99:.1f} us (p50={p50:.1f} us)"
