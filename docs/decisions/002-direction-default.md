# 002 — `direction` default for unhinted numerics

Spec §10 open decision 2. Status: decided (Task 1.14).

## Context

Magnitude scores one-sided (`up`) for money/count-hinted fields and names matching
`(^|_)(amount|count|qty|total|len|strlen)`, and two-sided (`both`) otherwise (§6.3, ruling R-DIR). The
question: should *unhinted, unnamed* numerics default to `up` (ignore unusually small values) to cut
control false positives? Task 1.14 adds `EngineConfig.mag_direction_default` (default `"both"`) to measure it.

## Options

1. `both` for unhinted numerics (spec default; catches abnormally small *and* large values).
2. `up` for all numerics.

## Evidence

`evidence/sweeps-seed0.md`, suite `direction` (warm test rows, live thresholds):

| setting | AUPRC | FPR@95 per-agent (recall) | benign FPR | control FPR | min gate recall |
|---|---|---|---|---|---|
| both | 0.550 | 0.563 (0.951) | 0.0027 | 0.062 | 0.388 |
| up | 0.550 | 0.564 (0.951) | 0.0027 | 0.062 | 0.388 |

Every gate cell is identical. In the 20-persona world almost every numeric field is up-named
(`amount`, `__len__`, `__strlen__`, `item_count`); the main exception is `due_in_days`, whose
magnitude_spike attacks are *upward*; the controls (manager-approved large refunds, quarter-end mass
updates) are upward on up-named fields too, so the switch has nothing to act on in this world.

## Decision

Keep **`both`** (no change). The control set shows no FPR difference to buy, and `both` keeps the ability
to flag abnormally small unhinted values. The switch stays in `EngineConfig` so Tier 2 (sandbox traces
with more varied schemas) can re-run this sweep; revisit if a control family exercises unhinted fields.
