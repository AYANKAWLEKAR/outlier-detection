# 006 — Calibration target FPR `cal_alpha`: 0.5 % or 1 %

Spec §10 open decision 6. Status: decided (Task 1.14) — keep 0.005, revisit after the calibrator fix.

## Context

Per-agent threshold `θ = q_{1−α}` of the agent's allowed fused scores (P² tail), blended with
`θ0 = 0.6` until 500 scores (§6.8). The spec default is α = 0.005; the question is whether 0.01 trades
better against the control set.

## Options

`cal_alpha` ∈ {0.005, 0.01}.

## Evidence

`evidence/sweeps-seed0.md`, suite `alpha`. Live = the engine's own thresholds; offcal = offline
reference thresholds (per-agent exact `1 − α` quantile of warm train benign scores), shown because the
live thresholds saturate (below):

| α | live min gate recall | live novel_target | live velocity_burst | live benign FPR | live control FPR | offcal min recall | offcal benign FPR | offcal control FPR |
|---|---|---|---|---|---|---|---|---|
| 0.005 | 0.388 | 0.388 | 0.600 | 0.0027 | 0.062 | 0.579 | 0.0153 | 0.157 |
| 0.01 | 0.449 | 0.449 | 0.700 | 0.0029 | 0.063 | 0.632 | 0.0295 | 0.166 |

The live calibrator barely reacts to α because its thresholds sit at 1.0: at α = 0.005 the final live
threshold is ≥ 0.999 for 17 of 20 agents. Fused scores of exactly 1.0 make up 0.05–2.5 % of each
agent's allowed calls (top-1 ≥ 0.85 plus the 0.15 agreement bonus, or a saturated sigmoid), so
`q_{0.995}` and even `q_{0.99}` *are* 1.0, and only envelopes that fuse to 1.0 are flagged. The P²
tail itself is accurate (decision 001: ≤ 0.02 error); the saturation is in the score distribution. In
the unsaturated offline reference, α = 0.01 doubles benign FPR (1.5 % → 3.0 %) for +5 pp min recall.

## Decision

Keep **`cal_alpha = 0.005`**. The live gain from 0.01 (+3 novel_target and +5 velocity_burst attacks out
of 49/50) is a by-product of saturated thresholds, it does not pass the gate either, and once the
thresholds stop saturating the same switch costs 2× benign FPR. The lever for the Tier-1 gate is the
benign 1.0 mass and the calibrator's treatment of it (e.g. calibrating on top-1 before the bonus,
excluding cold/prior-scored calls, or a decayed tail) — escalated to the controller in the Task 1.14
report, not decided here.
