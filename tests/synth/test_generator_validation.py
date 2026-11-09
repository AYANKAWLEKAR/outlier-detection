"""Spec 4.7 generator validation: the synthetic data must be non-trivial and faithful."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import scipy.stats

from agentfw.bench.metrics import compute_all
from agentfw.bench.registry import make_detector
from agentfw.bench.replay import SplitSpec, replay
from agentfw.core.envelope import Label, RawCall
from agentfw.core.normalize import SessionTracker, normalize
from agentfw.synth.load import CATALOG_DIR
from agentfw.synth.load import load_catalog as _load_catalog
from agentfw.synth.sampler import sample_persona
from tests.synth.conftest import World

Dataset = tuple[list[RawCall], dict[str, Label]]
_START_MS = 1_767_225_600_000


def _benign_support_sessions(world: World, days: int, seed: int = 0) -> list[list[RawCall]]:
    return sample_persona(
        world.persona("support-refunds-a"),
        _load_catalog(CATALOG_DIR),
        world.tasks,
        days=days,
        global_seed=seed,
        start_ms=_START_MS,
    )


def sampled_field(world: World, agent_id: str, field: str, n: int) -> np.ndarray:
    """First ``n`` benign values of ``tool.arg`` (numeric) drawn by the real sampler.

    Retries (an identical consecutive call within a session) are dropped so values are independent
    draws; the sampler emits benign traffic only, and amounts are sampled before any abandon.
    """
    tool, arg = field.rsplit(".", 1)
    catalog = _load_catalog(CATALOG_DIR)
    out: list[float] = []
    days = 100
    while len(out) < n:
        out.clear()
        sessions = sample_persona(
            world.persona(agent_id),
            catalog,
            world.tasks,
            days=days,
            global_seed=0,
            start_ms=_START_MS,
        )
        for s in sessions:
            prev: object = None
            for c in s:
                if c.tool_id == tool and arg in c.args:
                    if c.args != prev:
                        out.append(float(c.args[arg]))  # type: ignore[arg-type]
                    prev = c.args
                else:
                    prev = None
        days *= 2
    return np.asarray(out[:n], dtype=np.float64)


def test_non_triviality_static_threshold(validation_dataset: Dataset) -> None:
    rows = replay(*validation_dataset, make_detector("static_threshold"), SplitSpec())
    m = compute_all(rows, warm_only=False, n_boot=0)
    assert m.n_attack_rows > 0
    assert not (m.recall_at_global_thr >= 0.95 and m.fpr95_global < 0.01)


def test_amount_distribution_matches_persona(world: World) -> None:
    amts = sampled_field(world, "support-refunds-a", "stripe.create_refund.amount", n=10_000)
    assert len(amts) == 10_000
    p = scipy.stats.kstest(amts, "lognorm", args=(0.7, 0, np.exp(3.6))).pvalue
    assert p > 0.01


def test_prereq_base_rate_matches_grammar(world: World) -> None:
    """P(retrieve_charge in the 4 prior tools | benign small_refund session reaches create_refund).

    Population: the first create_refund of each benign support-refunds-a session (the persona emits
    no attacks here; ``create_refund`` only occurs in ``small_refund``). Retries and abandonment
    are independent of the skip draw, so the rate is exactly ``1 - skip_lookup``. A long horizon
    (n ~ 7k) keeps the sampling error (~0.003) well inside the 0.01 tolerance.
    """
    skip = world.persona("support-refunds-a").noise.skip_lookup
    tracker = SessionTracker()
    hits = total = 0
    for s in _benign_support_sessions(world, days=300):
        seen_refund = False
        for raw in s:
            env = normalize(raw, tracker)
            tracker.observe(env)
            if raw.tool_id == "stripe.create_refund" and not seen_refund:
                seen_refund = True
                total += 1
                hits += "stripe.retrieve_charge" in env.prior_tools
    assert total > 5000
    assert abs(hits / total - (1 - skip)) < 0.01


def test_label_integrity(validation_dataset: Dataset) -> None:
    raws, labels = validation_dataset
    assert len(labels) == len(raws)  # exactly one label per envelope
    assert {r.envelope_id for r in raws} == set(labels)
    by_attack: dict[str, int] = defaultdict(int)
    for lab in labels.values():
        if lab.label == "attack":
            assert lab.control_type is None
            assert lab.attack_id is not None
            by_attack[lab.attack_id] += 1
        else:
            assert lab.attack_id is None
    attack_ids = {lab.attack_id for lab in labels.values() if lab.attack_id is not None}
    assert attack_ids
    assert all(by_attack[a] >= 1 for a in attack_ids)
    assert math.isfinite(len(attack_ids))
