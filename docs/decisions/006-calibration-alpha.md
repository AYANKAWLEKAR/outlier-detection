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

The live calibrator barely reacts to α because its thresholds saturate. Evidence: the `saturation`
section of `evidence/sweeps-seed0.md`, produced by `agentfw.bench.diagnostics.threshold_saturation`
(sweeps suite `saturation`, run at the current default config):

- **17 of 20 agents** end with a live threshold ≥ 0.999.
- Those 17 agents are exactly the ones whose rows fusing ≥ 0.99 exceed α = 0.5 % of all their
  calls. The other three are data-export-a 0.42 %, helpdesk-triage-b 0.28 % and
  slack-notifier-a 0.24 %, with final θ 0.964 / 0.967 / 0.973.
- **Most of that tail mass is attack rows.** The bench replay allows every envelope (§5.2), so the
  calibrator learns attack scores too. Attack rows are 0.7–13.4 % of each agent's rows, and many of
  them fuse to 1.0 (multi-envelope velocity bursts, scope blowups). Examples:
  - outreach-b: 3 benign rows ≥ 0.99 out of 21,069, but 407 attack rows ≥ 0.99.
  - renewals-a: 0 benign rows ≥ 0.99, but 63 attack rows ≥ 0.99.
- Benign rows alone exceed 0.5 % at ≥ 0.99 in only **5** agents: billing-disputes-a, newsletter-a,
  sales-ops-b, support-refunds-a, support-refunds-c. Benign fused == 1.0 shares range 0–1.86 %.
- Top-1 attribution of benign fused == 1.0 rows (638 rows): sequence prereq 47 %,
  magnitude `to.__len__` 11 %, sequence flow 11 %, sequence trans 9 %, velocity count 9 %,
  magnitude `amount` 5 %, then a long tail.
- The P² tail itself is accurate (decision 001: ≤ 0.02 error).

This replaces the explanation in the first version of this record, which attributed the
saturation to benign 1.0 mass alone. In the unsaturated offline reference, α = 0.01 doubles
benign FPR (1.5 % → 3.0 %) for +5 pp min recall.

## Decision

Keep **`cal_alpha = 0.005`**. The live gain from 0.01 (+3 novel_target and +5 velocity_burst attacks out
of 49/50) is a by-product of saturated thresholds, it does not pass the gate either, and once the
thresholds stop saturating the same switch costs 2× benign FPR. The lever for the Tier-1 gate is what the calibrator is fed:
- the bench replay feeds it attack scores. A realistic protocol would not learn flagged/blocked
  envelopes, or would calibrate on benign-labelled rows only;
- a smaller lever is benign 1.0 mass, chiefly the prereq rule.

This is escalated to the controller (ruling GATE1, Task 1.15) and not decided here.
