# 004 — Velocity long window: dense ring vs sparse counters

Spec §10 open decision 4 (and D-010). Status: decided (Task 1.14).

## Context

§6.4 keeps a short (10 min) and a long (24 h) window per (agent, tool) and per money field. A dense
1440-bucket ring of `(count, sum)` costs ≈ 23 KB/key in binary (16 B × 1440) — the spec's "25 KB/key".
D-010 already stores the 24 h window sparsely (`{minute: (count, sum)}` for active minutes, pruned to
`vel_long_min`). The long window is **not scored** in v1 (the score uses the short window against its
active-minute history), so the choice affects memory, not detection, today.

## Options

1. Dense ring (spec text).
2. Sparse active-minute map (D-010, current).
3. Drop the long window (`vel_long_min = 0`) until a detector scores it.
4. Decayed counters (2 floats/key) — not implemented; would lose the per-minute shape.

## Evidence

`evidence/sweeps-seed0.md`, suite `vel_long`; `sweeps.velocity_memory` measures the final engine state
(124 velocity keys after 90 days, 20 agents):

| layout | JSON bytes (all velocity keys) | binary estimate |
|---|---|---|
| dense 1440-bucket ring | 456,104 | 2,856,960 (23,040 / key) |
| sparse (current) | 46,761 | 37,128 (12.5 active minutes / key on average) |
| no long window | 37,096 | — |

Detection is identical for `vel_long_min` 1440 vs 0: every gate cell, AUPRC 0.550, benign FPR 0.0027,
control FPR 0.062 (velocity_burst recall 0.600 in both). The whole velocity state is 0.4 % of the
engine's 11.3 MB on this dataset; novelty dominates (engine-no-novelty state is 3.2 MB).

## Decision

Keep the **sparse** long window (D-010 confirmed). It is ~10× smaller than a dense ring in JSON and
~77× in binary, it keeps the per-minute history that a future long-window or drift signal needs, and
it costs only ≈ 78 B/key (21 % of velocity state) over dropping it. No default changes. If Tier 4's
Redis budget gets tight, option 3 is free to take while the window stays unscored.
