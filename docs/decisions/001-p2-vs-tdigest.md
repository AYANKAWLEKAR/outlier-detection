# 001 — P² vs t-digest for the magnitude estimator

Spec §10 open decision 1. Status: decided (Task 1.14).

## Context

Magnitude keeps P² markers at q ∈ {0.05, 0.25, 0.5, 0.75, 0.95} over `y = sign(x)·log1p(|x|)` per
(agent, tool, path) (§6.3); the calibrator keeps a 3-marker P² tail at `1 − cal_alpha` over fused scores
(§6.8). The question is whether P²'s tail accuracy is good enough or a t-digest is needed. Note that the
magnitude *score* only reads q25/q50/q75 (`z = (y − q50) / max(IQR/1.349, eps)`); q05/q95 feed only the
`percentile` evidence string.

## Options

1. P² (current; O(1) state, 5 markers, deterministic, no dependency).
2. t-digest (better extreme-tail accuracy; new dependency, larger state, non-trivial to port to Go for Tier 4).

## Evidence

Dataset and command: see `evidence/sweeps-seed0.md` (suite `tail`);
`agentfw.bench.sweeps.quantile_tail_error` / `field_tail_errors` / `calibration_tail_error`.
`tdigest` is not installed in this environment, so option 2 was not measured (`quantile_tail_error(...,
"tdigest")` raises `ImportError`); the decision rests on whether P²'s error matters at all.

P² vs exact (`numpy.quantile`, linear), streamed in replay order, 103 numeric keys with n ≥ 200, error
in robust-sigma units (the unit of magnitude's z; z0 = 3):

| q | median | p90 | max |
|---|---|---|---|
| 0.05 | 0.0006 | 0.035 | 0.125 |
| 0.25 | 0.0061 | 0.066 | 0.766 |
| 0.5 | 0.0062 | 0.061 | 0.842 |
| 0.75 | 0.0068 | 0.060 | 0.331 |
| 0.95 | 0.0333 | 0.226 | 1.296 |

The maxima are all discrete or near-constant fields whose IQR collapses to `eps` (`query.__strlen__` of
templated SOQL, scale 0.107; `to.__len__`, `due_in_days` small integers), where any estimator's error
in sigma units is inflated by the tiny scale. Calibrator tail (fused-score units, per agent): median
0.0002, max 0.019 at α = 0.005; median 0.0002, max 0.083 at α = 0.01 (finance-reconcile-a).

## Decision

Keep **P²**. The quantiles that drive scoring (q25/q50/q75) are within 0.07 σ of exact for 90 % of keys —
two orders of magnitude below the z0 = 3 decision scale — and the calibrator's tail error is ≤ 0.02 at
the default α. No measured failure justifies a new dependency or a heavier Go port. Revisit only if a
future detector scores on q05/q95 directly (p90 error 0.23 σ at q95), and then benchmark t-digest with
the same function.
