"""Chronological score-before-update replay (spec 5.1-5.3).

Every detector and baseline runs through :func:`replay`: envelopes are sorted by
``(ts_unix_ms, session_id, seq_no)``, normalized causally, scored, and only then
used to update the detector.

``replay``'s own default outcome mode is ``"allow"``: ``ALLOWED`` for every envelope (spec §5.2:
the harness measures detection, not enforcement). The *benchmark* protocol is
:data:`BENCH_OUTCOMES` = ``"review"`` (user decision 2026-11-10, decision 008):
``agentfw.bench.run``, CI and the goldens pass it explicitly; every bench tool states its outcome
mode, and :data:`PROTOCOL_LABELS` holds the label each report must carry. Modes besides
``"allow"`` (Task 1.15):

* ``outcomes="enforce"``: ``BLOCKED`` when the detector's own ``DetectorResult.decision`` is
  ``BLOCK``, else ``ALLOWED`` -- including ``REQUIRE_APPROVAL`` calls (there is no reviewer in
  this mode) and envelopes not sampled by ``sample_ids``. A blocked call never moves a baseline.
* ``outcomes="review"``: the oracle-reviewer approval loop (§6.8 with the labels acting as a
  perfect human reviewer, hence optimistic). Every *flagged* envelope
  (decision ``BLOCK`` or ``REQUIRE_APPROVAL``) is ``APPROVED`` if benign and ``DENIED`` if labelled
  ``attack``; unflagged envelopes are ``ALLOWED`` unseen (missed attacks are learned). It reads the
  labels of flagged envelopes only (the reviewer's verdict).
* ``outcomes="oracle"``: ``BLOCKED`` for every envelope labelled ``attack``, else ``ALLOWED``. It
  reads every label, so it is a diagnostic upper bound only, never a reported benchmark number.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Literal, TypeGuard

import numpy as np
import orjson
import pyarrow as pa
import pyarrow.parquet as pq

from agentfw.bench.protocol import BatchFit, Detector
from agentfw.core.envelope import ActionEnvelope, Decision, Label, Outcome, RawCall
from agentfw.core.normalize import SessionTracker, normalize

__all__ = [
    "BENCH_OUTCOMES",
    "OUTCOME_MODES",
    "PROTOCOL_LABELS",
    "LeakError",
    "OutcomeMode",
    "ScoreRow",
    "SplitSpec",
    "assign_split",
    "canonical_hash",
    "read_score_table",
    "replay",
    "is_outcome_mode",
    "protocol_label",
    "write_score_table",
]

Split = Literal["train", "test"]
OutcomeMode = Literal["allow", "enforce", "review", "oracle"]
OUTCOME_MODES: tuple[OutcomeMode, ...] = ("allow", "enforce", "review", "oracle")
#: The benchmark protocol (user decision 2026-11-10, decision 008): the oracle-reviewer loop.
BENCH_OUTCOMES: OutcomeMode = "review"
#: The label every report header, leaderboard row or table using a mode must carry (verbatim).
PROTOCOL_LABELS: dict[OutcomeMode, str] = {
    "allow": "spec §5.2 allow-all",
    "enforce": "outcomes=enforce: not the spec 5.2 allow-all protocol",
    "review": "oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic)",
    "oracle": "diagnostic upper bound (reads every label) → optimistic",
}
_FLAGGED = frozenset({Decision.BLOCK, Decision.REQUIRE_APPROVAL})


def is_outcome_mode(x: object) -> TypeGuard[OutcomeMode]:
    """``x`` is one of :data:`OUTCOME_MODES` (e.g. a report's ``extras.outcomes``)."""
    return isinstance(x, str) and x in OUTCOME_MODES


def protocol_label(outcomes: object) -> str:
    """The label of a replay mode (``extras.outcomes``); unknown values say so explicitly."""
    if is_outcome_mode(outcomes):
        return PROTOCOL_LABELS[outcomes]
    return f"outcomes={outcomes}: unknown replay protocol"


class LeakError(RuntimeError):
    """``score()`` changed detector state."""


@dataclass(slots=True, frozen=True)
class SplitSpec:
    kind: Literal["time", "agent"] = "time"
    train_frac: float = 0.7
    seed: int = 0


@dataclass(slots=True)
class ScoreRow:
    envelope_id: str
    session_id: str
    agent_id: str
    tool_id: str
    ts_unix_ms: int
    split: str
    fused_score: float
    per_detector_json: str
    label: str
    attack_family: str | None
    attack_id: str | None
    injection_level: str | None
    strength: float | None
    step: int | None
    control_type: str | None
    is_warm: bool
    n_key_obs: int
    latency_us: float
    threshold: float | None


def _order(r: RawCall) -> tuple[int, str, int]:
    return (r.ts_unix_ms, r.session_id, r.seq_no)


def assign_split(raws: Sequence[RawCall], spec: SplitSpec) -> dict[str, Split]:
    if not raws:
        return {}
    if spec.kind == "time":
        t_min = min(r.ts_unix_ms for r in raws)
        t_max = max(r.ts_unix_ms for r in raws)
        cut = t_min + spec.train_frac * (t_max - t_min)
        return {r.envelope_id: ("train" if r.ts_unix_ms <= cut else "test") for r in raws}
    agents = sorted({r.agent_id for r in raws})
    perm = np.random.default_rng(spec.seed).permutation(len(agents))
    n_train = round(spec.train_frac * len(agents))
    train = {agents[int(i)] for i in perm[:n_train]}
    return {r.envelope_id: ("train" if r.agent_id in train else "test") for r in raws}


def _per_detector_json(res_per: Sequence[Any]) -> str:
    items = [
        {"detector": d.detector, "field": d.field, "score": d.score, "evidence": dict(d.evidence)}
        for d in res_per
    ]
    return orjson.dumps(items, option=orjson.OPT_SORT_KEYS, default=str).decode()


def replay(
    raws: Sequence[RawCall],
    labels: dict[str, Label],
    detector: Detector,
    split: SplitSpec,
    *,
    warmup_min_obs: int = 30,
    sample_ids: set[str] | None = None,
    strict: bool = False,
    outcomes: OutcomeMode = "allow",
) -> list[ScoreRow]:
    if outcomes not in OUTCOME_MODES:
        raise ValueError(f"unknown outcomes mode {outcomes!r}; known: {OUTCOME_MODES}")
    ordered = sorted(raws, key=_order)
    assignment = assign_split(ordered, split)
    detector.reset()
    tracker = SessionTracker()

    if isinstance(detector, BatchFit):
        # Fit on the train partition only, normalized causally with a throwaway tracker.
        fit_tracker = SessionTracker()
        train_envs: list[ActionEnvelope] = []
        for r in ordered:
            env = normalize(r, fit_tracker)
            fit_tracker.observe(env)
            if assignment[r.envelope_id] == "train":
                train_envs.append(env)
        detector.fit(train_envs)

    obs: Counter[tuple[str, str]] = Counter()
    rows: list[ScoreRow] = []
    for r in ordered:
        env = normalize(r, tracker)
        key = (env.agent_id, env.tool_id)
        n_obs = obs[key]
        scored = sample_ids is None or env.envelope_id in sample_ids
        fused = math.nan
        per_json = "[]"
        latency = 0.0
        threshold: float | None = None
        outcome = Outcome.ALLOWED
        flagged = False
        if scored:
            before = detector.state_bytes() if strict else 0
            t0 = perf_counter_ns()
            res = detector.score(env)
            latency = (perf_counter_ns() - t0) / 1000.0
            if strict and detector.state_bytes() != before:
                raise LeakError(f"score() changed state on {env.envelope_id}")
            fused = res.fused_score
            per_json = _per_detector_json(res.per_detector)
            threshold = res.threshold
            flagged = res.decision in _FLAGGED
            if outcomes == "enforce" and res.decision is Decision.BLOCK:
                outcome = Outcome.BLOCKED
        lab = labels.get(env.envelope_id)
        is_attack = lab is not None and lab.label == "attack"
        if outcomes == "review" and flagged:
            outcome = Outcome.DENIED if is_attack else Outcome.APPROVED
        elif outcomes == "oracle" and is_attack:
            outcome = Outcome.BLOCKED
        rows.append(
            ScoreRow(
                envelope_id=env.envelope_id,
                session_id=env.session_id,
                agent_id=env.agent_id,
                tool_id=env.tool_id,
                ts_unix_ms=env.ts_unix_ms,
                split=assignment[env.envelope_id],
                fused_score=fused,
                per_detector_json=per_json,
                label=lab.label if lab else "benign",
                attack_family=lab.attack_family if lab else None,
                attack_id=lab.attack_id if lab else None,
                injection_level=lab.injection_level if lab else None,
                strength=lab.strength if lab else None,
                step=lab.step if lab else None,
                control_type=lab.control_type if lab else None,
                is_warm=n_obs >= warmup_min_obs,
                n_key_obs=n_obs,
                latency_us=latency,
                threshold=threshold,
            )
        )
        detector.update(env, outcome)
        tracker.observe(env)
        obs[key] += 1
    return rows


def write_score_table(rows: Sequence[ScoreRow], path: str | Path) -> None:
    names = [f.name for f in fields(ScoreRow)]
    cols = {n: [getattr(r, n) for r in rows] for n in names}
    pq.write_table(pa.table(cols), str(path))


def read_score_table(path: str | Path) -> list[ScoreRow]:
    return [ScoreRow(**d) for d in pq.read_table(str(path)).to_pylist()]


def canonical_hash(rows: Sequence[ScoreRow]) -> str:
    """D-019: sha256 over canonical JSONL of ``[envelope_id, round(score, 9)]`` in row order.

    NaN scores serialize as null. Independent of Parquet bytes (metadata varies by pyarrow).
    """
    h = hashlib.sha256()
    for r in rows:
        s = r.fused_score
        h.update(orjson.dumps([r.envelope_id, None if math.isnan(s) else round(s, 9)]))
        h.update(b"\n")
    return h.hexdigest()
