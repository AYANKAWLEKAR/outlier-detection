"""Latency gate (spec §6.11): p99 of ``Engine.score`` < 200 µs on warm keys with <= 20 fields."""

from __future__ import annotations

import os
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from agentfw.bench.dataset import load_dir, load_schema_hints
from agentfw.bench.registry import make_detector
from agentfw.bench.trajectory import bench_config
from agentfw.core.engine import Engine
from agentfw.core.envelope import ActionEnvelope, Outcome, RawCall, ResponseMeta
from agentfw.core.ids import ulid_from
from agentfw.core.normalize import SessionTracker, normalize
from agentfw.core.trajectory.flags import Flag
from agentfw.core.trajectory.session import SessionVerdict
from agentfw.synth.generate import generate

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


def _p99(eng: Engine) -> tuple[float, float]:
    """(p50, p99) of ``eng.score`` over the scored stream, after the warm-up stream."""
    rng = np.random.default_rng(0)
    tracker = SessionTracker()
    warm = _stream(N_WARM, 0, rng, tracker)
    scored = _stream(N_SCORED, N_WARM, rng, tracker)
    assert max(len(e.flat_args) + len(e.derived) for e in scored) <= 20

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
    return float(p50), float(p99)


@pytest.mark.skipif(os.environ.get("AFW_SKIP_PERF") == "1", reason="AFW_SKIP_PERF=1 set")
def test_score_p99_under_budget() -> None:
    p50, p99 = _p99(Engine())
    print(f"Engine.score latency: p50={p50:.1f} us p99={p99:.1f} us (n={N_SCORED})")
    assert p99 < P99_BUDGET_US, f"p99={p99:.1f} us (p50={p50:.1f} us)"


# -- every trajectory switch on (plan Global Constraints, Task 4.1; T-55) ------------------------

#: The bench config (``trajectory.bench_config``: SW1 + loop, spin, stall, splice) cut to 12 days
#: (about 23,000 calls): real tool calls with the dataset's captured schemas, so the gate,
#: escalation, provisional flags and session verdicts run at the bench's own rates.
TRAJ_DAYS = 12
TRAJ_WARM = 10_000
TRAJ_SCORED = 10_000


@dataclass(frozen=True, slots=True)
class TrajRun:
    lat: NDArray[np.float64]
    escalated: int
    gate_outcomes: int
    not_ok_verdicts: int
    provisional: int
    irreversible: int


@pytest.fixture(scope="module")
def traj_run(tmp_path_factory: pytest.TempPathFactory) -> TrajRun:
    """``engine-traj`` with the dataset's schemas as hints scores the calls after the first
    ``TRAJ_WARM`` (replay order, each call updated with ``ALLOWED``); ``score`` is timed on the
    next ``TRAJ_SCORED`` calls, and what the trajectory paths did on them is counted."""
    d = tmp_path_factory.mktemp("traj_perf")
    generate(bench_config(0).model_copy(update={"days": TRAJ_DAYS}), d)
    raws, _, _ = load_dir(d)
    raws = sorted(raws, key=lambda r: (r.ts_unix_ms, r.session_id, r.seq_no))
    assert len(raws) >= TRAJ_WARM + TRAJ_SCORED
    eng = make_detector("engine-traj", hints=load_schema_hints(d))
    assert isinstance(eng, Engine) and eng.cfg.br_mode == "modulate" and eng.hints
    tracker = SessionTracker()
    lat = np.empty(TRAJ_SCORED)
    esc = gate = not_ok = prov = irr = 0
    for i, raw in enumerate(raws[: TRAJ_WARM + TRAJ_SCORED]):
        e = normalize(raw, tracker)
        t0 = perf_counter_ns()
        res = eng.score(e)
        dt = (perf_counter_ns() - t0) / 1000.0
        eng.update(e, Outcome.ALLOWED)
        tracker.observe(e)
        j = i - TRAJ_WARM
        if j < 0:
            continue
        lat[j] = dt
        t = res.trajectory
        assert t is not None
        esc += t.escalated
        gate += t.gate is not None
        not_ok += t.session_verdict not in (None, SessionVerdict.OK)
        prov += Flag.PROVISIONAL in t.flags
        irr += t.irreversible
    return TrajRun(lat, esc, gate, not_ok, prov, irr)


@pytest.mark.skipif(os.environ.get("AFW_SKIP_PERF") == "1", reason="AFW_SKIP_PERF=1 set")
def test_every_switch_stream_exercises_the_trajectory_paths(traj_run: TrajRun) -> None:
    """Non-vacuity: the timed calls go through escalation, the gate, provisional flags and
    non-OK session verdicts, so the latency below is the cost of those paths."""
    r = traj_run
    print(
        f"engine-traj timed calls: escalated {r.escalated}, gate outcomes {r.gate_outcomes}, "
        f"non-OK verdicts {r.not_ok_verdicts}, provisional {r.provisional}, "
        f"irreversible {r.irreversible} (n={TRAJ_SCORED})"
    )
    assert r.escalated > 0 and r.gate_outcomes > 0
    assert r.not_ok_verdicts > 0 and r.provisional > 0 and r.irreversible > 0


@pytest.mark.skipif(os.environ.get("AFW_SKIP_PERF") == "1", reason="AFW_SKIP_PERF=1 set")
def test_score_p99_under_budget_with_every_trajectory_switch_on(traj_run: TrajRun) -> None:
    """The same bound with every trajectory switch on (``engine-traj``: v1 + vital and
    ``registry.TRAJ_SWITCHES``), on bench calls with their schemas. Was a strict xfail (p99
    207-235 us, T-55) while F5's escalation re-ran the fusion; it reuses the fused score since
    T-60 (same output), and the 1,018 escalated calls no longer pay twice."""
    p50, p99 = np.percentile(traj_run.lat, [50, 99])
    print(f"engine-traj Engine.score latency: p50={p50:.1f} us p99={p99:.1f} us (n={TRAJ_SCORED})")
    assert p99 < P99_BUDGET_US, f"p99={p99:.1f} us (p50={p50:.1f} us)"
