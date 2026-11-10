"""Stratified subsampling of the test split for slow detectors (spec 5.5, LLM judge)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import TypedDict

import numpy as np

from agentfw.bench.replay import SplitSpec, assign_split
from agentfw.core.envelope import Label, RawCall
from agentfw.synth.sampler import stable_hash

__all__ = ["SampleCounts", "sample_counts", "stratified_sample"]


class SampleCounts(TypedDict):
    n: int
    n_attack: int
    n_benign: int
    attack_by_family: dict[str, int]
    benign_by_agent: dict[str, int]


def _allocate(sizes: Mapping[str, int], total: int) -> dict[str, int]:
    """Equal share per stratum, capped by its size; leftover is redistributed (water-filling)."""
    take = dict.fromkeys(sizes, 0)
    left = min(total, sum(sizes.values()))
    open_ = sorted(k for k, s in sizes.items() if s > 0)
    while left > 0 and open_:
        share, extra = divmod(left, len(open_))
        nxt: list[str] = []
        for i, k in enumerate(open_):
            want = share + (1 if i < extra else 0)
            got = min(want, sizes[k] - take[k])
            take[k] += got
            left -= got
            if take[k] < sizes[k]:
                nxt.append(k)
        open_ = nxt
    return take


def _pick(strata: Mapping[str, list[str]], total: int, rng: np.random.Generator) -> set[str]:
    take = _allocate({k: len(v) for k, v in strata.items()}, total)
    out: set[str] = set()
    for k in sorted(strata):
        ids = sorted(strata[k])
        if take[k]:
            idx = rng.choice(len(ids), size=take[k], replace=False)
            out.update(ids[int(i)] for i in idx)
    return out


def stratified_sample(
    raws: Sequence[RawCall],
    labels: Mapping[str, Label],
    n: int = 2000,
    seed: int = 0,
    split: SplitSpec | None = None,
) -> set[str]:
    """Envelope ids of a test-split sample: <= n/2 attacks by family, the rest benign by agent.

    Allocation is equal per stratum (capped by availability), so rare families and small agents
    stay represented. Deterministic per ``seed``. If one side runs out, the other fills up to ``n``
    (or everything available). The sample's attack prevalence is therefore far above the
    dataset's; prevalence-sensitive metrics (AUPRC) are not comparable with a full replay.
    """
    assignment = assign_split(raws, split or SplitSpec())
    attacks: dict[str, list[str]] = defaultdict(list)
    benign: dict[str, list[str]] = defaultdict(list)
    for r in raws:
        if assignment[r.envelope_id] != "test":
            continue
        lab = labels.get(r.envelope_id)
        if lab is not None and lab.label == "attack":
            attacks[lab.attack_family or "unknown"].append(r.envelope_id)
        else:
            benign[r.agent_id].append(r.envelope_id)
    rng = np.random.default_rng([seed, stable_hash("sample")])
    chosen = _pick(attacks, n // 2, rng)
    chosen |= _pick(benign, n - len(chosen), rng)
    return chosen


def sample_counts(
    raws: Sequence[RawCall], labels: Mapping[str, Label], ids: set[str]
) -> SampleCounts:
    """Actual composition of a sample, for the report."""
    fam: dict[str, int] = defaultdict(int)
    agent: dict[str, int] = defaultdict(int)
    for r in raws:
        if r.envelope_id not in ids:
            continue
        lab = labels.get(r.envelope_id)
        if lab is not None and lab.label == "attack":
            fam[lab.attack_family or "unknown"] += 1
        else:
            agent[r.agent_id] += 1
    return {
        "n": len(ids),
        "n_attack": sum(fam.values()),
        "n_benign": sum(agent.values()),
        "attack_by_family": dict(sorted(fam.items())),
        "benign_by_agent": dict(sorted(agent.items())),
    }
