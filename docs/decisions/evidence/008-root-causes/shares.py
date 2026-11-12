"""Task 1.15: attribution and review load of one candidate (evidence for decision 008).

    uv run python docs/decisions/evidence/008-root-causes/shares.py TRACES_DIR OUTCOMES 'JSON_PARAMS'

e.g. ... traces/sweep-seed0 review '{"vel_count_attempts": true, "seq_prereq_graded": true,
"novelty_full_n": 50}'. JSON_PARAMS apply on top of the spec engine: the config is
spec_config(params) = EngineConfig().replace(**{**SPEC_SWITCHES, **params}), so '{}' is the spec
engine even though the engine defaults changed later (user decision, Task U1; Amendment A1 scoring,
Task 1.21: SPEC_SWITCHES includes PRE_A1_SWITCHES, so every candidate is the engine before
Amendment A1). Per gate family (warm test, live thresholds): detected attacks, the top-1
detector:part of each detected attack's representative row (the brief's top_reason_share), and
reason_share = share of detected attacks with the targeted detector among the reasons (score >
reason_min) of a flagged row. Then flagged calls (block + require_approval) per agent-day.
"""

import collections
import json
import math
import sys

import orjson

from agentfw.bench.ablation import replay_detector
from agentfw.bench.metrics import GATE_FAMILIES, _bucket_of
from agentfw.bench.replay import SplitSpec
from agentfw.bench.sweeps import SPEC_SWITCHES
from agentfw.core.config import EngineConfig


def spec_config(params: dict[str, object]) -> EngineConfig:
    """The spec engine (``SPEC_SWITCHES``) with ``params`` on top."""
    return EngineConfig().replace(**{**SPEC_SWITCHES, **params})


def main() -> None:
    traces, outcomes, params = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
    cfg = spec_config(params)
    rows, _ = replay_detector(traces, "engine", SplitSpec(), cfg, outcomes)  # type: ignore[arg-type]
    warm = [r for r in rows if r.split == "test" and r.is_warm and not math.isnan(r.fused_score)]
    groups = collections.defaultdict(list)
    for r in warm:
        if r.label == "attack":
            groups[r.attack_id].append(r)
    for fam, (bucket, det) in GATE_FAMILIES.items():
        tops: collections.Counter[str] = collections.Counter()
        reason = detected = 0
        for rs in groups.values():
            rs.sort(key=lambda r: (r.ts_unix_ms, r.envelope_id))
            if rs[0].attack_family != fam or (
                bucket != "all" and _bucket_of(fam, rs[0].strength) != bucket
            ):
                continue
            flagged = [r for r in rs if r.threshold is not None and r.fused_score >= r.threshold]
            if not flagged:
                continue
            detected += 1
            rep = max(flagged, key=lambda r: r.fused_score)
            top = max(orjson.loads(rep.per_detector_json), key=lambda i: i["score"])
            ev = top.get("evidence") or {}
            tops[f"{top['detector']}:{top.get('field') or ev.get('fired') or ev.get('signal')}"] += 1
            reason += any(
                any(i["detector"] == det and i["score"] > cfg.reason_min for i in orjson.loads(f.per_detector_json))
                for f in flagged
            )
        share = reason / detected if detected else math.nan
        print(f"{fam:16s} detected={detected:3d} reason_share={share:.2f} top1={tops.most_common(4)}")
    for split in ("train", "test"):
        rs = [r for r in rows if r.split == split and not math.isnan(r.fused_score)]
        days = (max(r.ts_unix_ms for r in rs) - min(r.ts_unix_ms for r in rs)) / 86_400_000
        agents = len({r.agent_id for r in rs})
        fl = [r for r in rs if r.threshold is not None and r.fused_score >= r.threshold - cfg.cal_delta]
        ben = [r for r in fl if r.label != "attack"]
        print(
            f"{split}: calls/agent-day={len(rs) / agents / days:.0f} "
            f"flagged/agent-day={len(fl) / agents / days:.1f} "
            f"benign flagged/agent-day={len(ben) / agents / days:.1f}"
        )


if __name__ == "__main__":
    main()
