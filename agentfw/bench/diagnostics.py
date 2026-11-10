"""Calibration diagnostics over a replay's score table (Task 1.14, decisions 001 / 006).

:func:`threshold_saturation` explains *why* the engine's live per-agent thresholds sit near 1.0. Per
agent: the final live threshold; benign rows with fused == 1.0 and >= 0.99; and the same counts for
**attack** rows. The bench replay allows every envelope (spec 5.2), so the calibrator learns attack
scores too, and a ``q_{1-alpha}`` sits in whichever tail mass (benign or attack) exceeds ``alpha``.
Also: which ``detector:part`` supplied the top-1 score of the benign fused == 1.0 rows.

:func:`threshold_trajectory` (Task 1.15, decision 008) follows each agent's live threshold over
time, to expose a censoring ratchet: when the calibrator stops learning the calls the engine
blocks, its ``q_{1-alpha}`` is fitted on a distribution truncated at the threshold and can only
move down.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import orjson

from agentfw.bench.replay import ScoreRow

__all__ = [
    "RATCHET_DROP",
    "SATURATED_THRESHOLD",
    "AgentSaturation",
    "SaturationReport",
    "ThetaTrajectory",
    "saturation_md",
    "threshold_saturation",
    "threshold_trajectory",
]

#: A final live threshold at or above this counts as saturated.
SATURATED_THRESHOLD = 0.999
_ONE = 1.0 - 1e-12
_NEAR = 0.99


@dataclass(slots=True, frozen=True)
class AgentSaturation:
    final_threshold: float  # live threshold on the agent's last scored row (NaN: none)
    n_benign: int  # scored benign rows, every split
    n_one: int  # of which fused == 1.0
    share_one: float
    n_benign_ge99: int  # benign rows with fused >= 0.99
    n_attack: int  # scored attack rows (the replay allows them, so the calibrator learns them)
    n_attack_one: int
    n_attack_ge99: int


@dataclass(slots=True, frozen=True)
class SaturationReport:
    agents: dict[str, AgentSaturation]
    n_saturated: int  # agents whose final threshold >= SATURATED_THRESHOLD
    top1: dict[str, int]  # "detector:part" -> benign fused==1.0 rows where it supplied top-1


def _top1_key(per_detector_json: str) -> str:
    """``detector:part`` of the highest per-detector score (first wins ties). The part is the
    field, else the sequence part that fired / the velocity signal, else ``-``."""
    best: dict[str, object] | None = None
    best_score = -math.inf
    for item in orjson.loads(per_detector_json):
        s = item.get("score")
        if isinstance(s, bool) or not isinstance(s, int | float) or math.isnan(s):
            continue
        if best is None or s > best_score:
            best, best_score = item, float(s)
    if best is None:
        return "none"
    ev = best.get("evidence")
    evd = ev if isinstance(ev, dict) else {}
    part = best.get("field") or evd.get("fired") or evd.get("signal") or "-"
    return f"{best.get('detector')}:{part}"


def threshold_saturation(rows: Sequence[ScoreRow]) -> SaturationReport:
    """Per-agent final live threshold and benign fused==1.0 share, over every scored row (all
    splits: the calibrator sees every allowed call). Attack rows count only for the threshold."""
    last: dict[str, tuple[tuple[int, str], float]] = {}
    counts: dict[str, Counter[str]] = {}
    top1: Counter[str] = Counter()
    for r in rows:
        if math.isnan(r.fused_score):
            continue
        key = (r.ts_unix_ms, r.envelope_id)
        if r.threshold is not None and (r.agent_id not in last or key >= last[r.agent_id][0]):
            last[r.agent_id] = (key, r.threshold)
        c = counts.setdefault(r.agent_id, Counter())
        kind = "attack" if r.label == "attack" else "benign"
        c[kind] += 1
        c[f"{kind}_ge99"] += r.fused_score >= _NEAR
        if r.fused_score >= _ONE:
            c[f"{kind}_one"] += 1
            if kind == "benign":
                top1[_top1_key(r.per_detector_json)] += 1
    agents: dict[str, AgentSaturation] = {}
    for agent in sorted(set(counts) | set(last)):
        c = counts.get(agent, Counter())
        nb, n1 = c["benign"], c["benign_one"]
        agents[agent] = AgentSaturation(
            final_threshold=last[agent][1] if agent in last else math.nan,
            n_benign=nb,
            n_one=n1,
            share_one=n1 / nb if nb else math.nan,
            n_benign_ge99=c["benign_ge99"],
            n_attack=c["attack"],
            n_attack_one=c["attack_one"],
            n_attack_ge99=c["attack_ge99"],
        )
    n_sat = sum(a.final_threshold >= SATURATED_THRESHOLD for a in agents.values())
    return SaturationReport(agents, n_sat, dict(top1.most_common()))


def saturation_md(rep: SaturationReport) -> list[str]:
    """Markdown lines: per-agent table, then the top-1 attribution of benign fused==1.0 rows."""
    out = [
        f"{rep.n_saturated} of {len(rep.agents)} agents end with a live threshold >= "
        f"{SATURATED_THRESHOLD}. All splits; the calibrator learns benign AND attack rows:",
        "",
        "| agent | final live threshold | benign rows | benign == 1.0 (share) | benign >= 0.99 "
        "| attack rows | attack == 1.0 | attack >= 0.99 | all rows >= 0.99 (share) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for agent, a in rep.agents.items():
        n_all = a.n_benign + a.n_attack
        ge = a.n_benign_ge99 + a.n_attack_ge99
        out.append(
            f"| {agent} | {a.final_threshold:.4f} | {a.n_benign} | {a.n_one} ({a.share_one:.4f}) "
            f"| {a.n_benign_ge99} | {a.n_attack} | {a.n_attack_one} | {a.n_attack_ge99} "
            f"| {ge} ({ge / n_all if n_all else math.nan:.4f}) |"
        )
    total = sum(rep.top1.values())
    out += ["", "Top-1 detector:part of benign fused == 1.0 rows:", "", "| top-1 | rows | share |"]
    out.append("|---|---|---|")
    for k, n in rep.top1.items():
        out.append(f"| {k} | {n} | {n / total:.3f} |")
    return out


# --- threshold trajectories (Task 1.15) ---------------------------------------------------------

#: An agent whose live threshold fell by more than this over the test period counts as ratcheting.
RATCHET_DROP = 0.05
_QUARTERS = (0.25, 0.5, 0.75, 1.0)


@dataclass(slots=True, frozen=True)
class ThetaTrajectory:
    end_train: float  # live threshold on the agent's last scored train row (NaN: none)
    test_quarters: tuple[float, ...]  # on its scored test rows at 25/50/75/100 % (by count)
    min_test: float
    end: float  # last scored row overall

    @property
    def drift(self) -> float:
        """``end - end_train`` (NaN without train rows)."""
        return self.end - self.end_train


def threshold_trajectory(rows: Sequence[ScoreRow]) -> dict[str, ThetaTrajectory]:
    """Per agent (sorted), over scored rows with a threshold, ordered by (ts, envelope_id)."""
    by: dict[str, list[tuple[str, float]]] = {}
    for r in sorted(rows, key=lambda r: (r.ts_unix_ms, r.envelope_id)):
        if not math.isnan(r.fused_score) and r.threshold is not None:
            by.setdefault(r.agent_id, []).append((r.split, r.threshold))
    out: dict[str, ThetaTrajectory] = {}
    for agent in sorted(by):
        rs = by[agent]
        train = [t for split, t in rs if split == "train"]
        test = [t for split, t in rs if split == "test"]
        out[agent] = ThetaTrajectory(
            end_train=train[-1] if train else math.nan,
            test_quarters=tuple(test[max(0, math.ceil(f * len(test)) - 1)] for f in _QUARTERS)
            if test
            else (),
            min_test=min(test) if test else math.nan,
            end=rs[-1][1],
        )
    return out
