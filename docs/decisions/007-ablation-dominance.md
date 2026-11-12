# 007 — Ablation dominance does not hold (finding, not a default change)

Task 1.14 finding, recorded so the strict xfail in
`tests/bench/test_ablation_sweeps.py::test_full_engine_auprc_at_least_every_single_detector_ablation`
has a reference. Status: open; nothing is decided here.

## Context

The Task 1.14 brief expected "`engine` AUPRC ≥ every `engine-no-*`": removing any detector should not
improve the full engine's ranking.

## Options

1. Treat a violation as a test failure (gate red).
2. Record it and keep the property asserted per variant, with any violating case as a **strict**
   xfail. That case turns red as soon as it starts holding (XPASS), so the xfail can't silently go stale.

## Evidence

Warm test rows, AUPRC over envelopes.

- `tiny_dataset` (3 Tier-0 personas, 5 days, 18 warm test attack rows; measured at prereq_rate 0.95, still
  violated at 0.90 per the strict xfail): engine 0.449 <
  engine-no-magnitude 0.500. The other single-detector ablations are lower (no-novelty 0.288,
  no-velocity 0.424, no-sequence 0.324).
- 90-day seed-0 dataset (`evidence/sweeps-seed0.md`, suite `ablation`, prereq_rate 0.95 base):
  engine 0.550 < engine-no-novelty 0.608 and < engine-no-sequence 0.593. engine-only-velocity
  reaches 0.721. engine-no-magnitude 0.548 and engine-no-velocity 0.162 hold the property.
- Caveat on the metric: envelope-level AUPRC is dominated by `velocity_burst`, whose attacks have
  10–60 positive envelopes each, while most other families have one. Removing novelty or sequence
  takes out benign noise that outranks burst envelopes.
- Per-family recall in the same table shows each detector carrying its targeted family. For
  example, without sequence prereq_skip recall drops 0.867 → 0, and without novelty novel_target
  recall drops 0.388 → 0.

## Decision

Option 2. The `engine-no-magnitude` case is a strict xfail on `tiny_dataset`, and the other three
ablations stay asserted. Ablation conclusions should be read from per-family attack-level recall, not
envelope AUPRC. Whether the Tier-1 "ablation table" exit criterion needs an attack-level AUPRC is left
to the controller (Task 1.15).

## Addendum 2026-11-11 (Amendment A1, Task 1.21): the `tiny_dataset` xfail XPASSes and is removed

With the Amendment A1 scoring as the engine default (`vel_count_stat = "poisson"`,
`mag_score = "tail"`, `mag_bool_numeric = True`; spec A1.1 to A1.3), the strict xfail on
`engine-no-magnitude` XPASSed, so the mark is removed and the property is asserted for all four
single-detector ablations on `tiny_dataset`. Warm test AUPRC, replay outcomes `allow`, measured on
the implementation:

| engine configuration | engine | engine-no-magnitude | engine-no-novelty | engine-no-velocity | engine-no-sequence |
|---|---|---|---|---|---|
| defaults (Amendment A1 scoring) | 0.608 | 0.522 | 0.381 | 0.557 | 0.485 |
| `PRE_A1_SWITCHES` (pre-amendment scoring) | 0.472 | 0.521 | 0.268 | 0.436 | 0.380 |

The pre-amendment row still violates the property (0.472 < 0.521; it was 0.449 < 0.500 when this
record was written, before the user-decision defaults of decision 008). The change comes from the
scoring, not from the data: the dataset is the same, and `engine-no-magnitude` barely moves
(0.522 vs 0.521, the velocity count statistic only). The 90-day seed-0 evidence above was not
re-measured and still reflects the pre-amendment engine. The reading advice stands: ablation
conclusions come from per-family attack-level recall, not envelope AUPRC.
