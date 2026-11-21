# 010 — Trajectory monitoring: measured switches and recommended defaults

Status: **proposed (Task 4.1, 2026-11-12; re-measured after the final whole-branch review,
2026-11-16); waiting on the user.** Every trajectory feature is still behind an `EngineConfig`
switch that is off by default (T-03), and nothing here changes a default. Each recommendation
below is a proposal. If the user flips a default, that change regenerates
`tests/golden/*.sha256` and `agentfw/bench/baseline_metrics.json` in a commit of its own.

Numbers: [`docs/trajectory-benchmark.md`](../trajectory-benchmark.md), the rendered report of
`python -m agentfw.bench.trajectory --traces traces/bench --out results/trajectory-final --seeds
0,1 --workers 4`, run on committed sha `c7c123f` (clean tree), which includes the final-review
fixes T-58 to T-63, with the rows that read ledger refs (`engine`, `flow_ledger`, `provisional`,
`seq_age_discount`, `gate`, `gate+flow_ledger`, `escalate+gate+flow_ledger`, `engine-traj`,
`engine-traj-shadow`) and the attribution re-measured after ruling R28 reverted T-59's ledger
change (both seeds, both protocols; the benchmark file's header lists what moved). Every other
row and the latency table are carried over from `c7c123f` (`results/` is not committed, ruling
R24). Every number in this record comes from that report unless a line names another source. Protocol: T-53. Interpretation records: T-52
to T-64 in [`009-trajectory-clarifications.md`](009-trajectory-clarifications.md).

**What the re-run changed** (against the Task 4.1 run, code `8d4c1f1`). Every per-call and
decision number of every single-switch row and of the `gate` / `escalate` composites is
identical, and so are splice, session AUROC, the vital false-flag rate and the calibration
selection. What moved: runaway halt recall with `sessions` + vital (spin now lags one call,
T-58) and with `engine-traj`; the first-flag baseline of the `sessions` / `session_enforce` rows
(seed 0: 3.51% → 3.77%, T-61's flags) and of `sessions+vital`; `engine-traj`'s own rows by a
little (seed 0 spec §5.2 allow-all blocks / approvals 2,798 / 2,933 → 2,796 / 2,935; seed 1
oracle-reviewer Δ control FPR +0.1683 → +0.1650); and, from T-59's gate-only rule, one benign
prerequisite block on seed 0 in every row with `gate` (6 → 7 blocks; see `gate`). Latency changed
(T-60).

## The bench

- **Data.** Two seeds of the bench config: `sw1_config(s)` (20 personas, 90 days, the 8 SW1
  families, far-mode spikes, top-up) plus `loop`, `spin`, `stall` and `splice`. Seed 0 is
  `1d4c1873497b`: 177,460 envelopes, 855 attacks, 42,289 sessions (40,840 benign, 232 runaway,
  1,217 attacked). Seed 1 is `27e7e5c04f94`: 174,995 envelopes, 829 attacks, 41,949 sessions.
- **Split.** Time split with a calibration partition of 15% of the agents (3 of 20), as F10
  specifies. Per-call and decision metrics are on the test split.
- **Protocols.** Every number is given under both protocols: spec §5.2 allow-all, and the
  oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic). A result
  under the second protocol alone is never a pass.
- **What "changed" means.** Each row is one switch toggled alone on top of `engine`. Δ is the row
  minus `engine` on the same seed and protocol, with a paired session bootstrap 95% CI (n =
  1,000). The decision metrics count a call as flagged when its decision is `block` or
  `require_approval`.
- **Hints (I4).** The bench passes the dataset's `schemas.jsonl` as hints: MCP annotations and
  `has_schema`, **no `kind` hints** (`bench.dataset.load_schema_hints`). See "Measured with four
  read tools learned as writes" below: it qualifies every row marked ‡.

`engine` reference (decision benign FPR / decision control FPR, seeds 0 / 1): spec §5.2 allow-all
0.0106 / 0.0265 and 0.0125 / 0.0163; oracle-reviewer approval loop (labels act as a perfect human
reviewer → optimistic) 0.0396 / 0.0737 and 0.0466 / 0.0343.

## Per switch

Δ values are seed 0, then seed 1. Attribution is the leave-one-group-out table of `engine-traj`
(T-52). Latency is the alternated A/B `score` time on 30,000 seed-0 calls, as p50 / p99 relative
to `engine` (79 µs / 155 µs in this run, which ended at load average 4.2 / 8.2 over 1 / 5 min:
the ratios are the comparable figures). ‡ = measured with four read tools learned as writes (I4).

| switch | what it changed, spec §5.2 allow-all | what it changed, oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic) | attribution rows carried | latency p50 / p99 | recommended default |
|---|---|---|---|---|---|
| `flow_ledger` | Decisions: Δ benign FPR +0.0000 on both seeds. Per-call per-agent FPR@95: 0.578 → 0.601 and 0.682 → 0.583. | Δ benign FPR +0.0001 [0.0000, 0.0003] and +0.0006 [0.0003, 0.0010]. FPR@95 0.572 → 0.568 and 0.553 → 0.546. | not a group (the gate's input) | ×0.97 / ×0.97 | **off**, or on together with `gate` |
| `vital` | Scores and decisions identical to `engine` (T-07). Benign vital false-flag rate 3.85% and 3.63% (bar 1%, T-30). With `sessions`: runaway halt recall 0.819 → 0.944 and 0.794 → 0.877 at the same false-halt rate (0.30%); median 7 calls from onset to halt. Spin reads completed calls only since T-58 (one call of lag); with T-27's lookahead it was 0.957 and 0.918. | Identical to `engine`. False-flag rate 3.85% and 3.63%. With `sessions`: runaway halt recall 0.948 and 0.905. | 0 on every seed and protocol (never fused). Removing it gains 2 attacks on seed 0 of the oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic). | ×1.08 / ×1.02 | **off** until T-30 is decided; then on with `sessions` in shadow |
| `br_mode="score"` | Scores and decisions identical. Adds `b` to every reason. | Identical. | not a group | ×1.06 / ×1.08 | **on** (reporting only) |
| `br_mode="modulate"` ‡ | Δ benign FPR **+0.1403 [0.1365, 0.1438]** and **+0.1443 [0.1405, 0.1487]**. Δ control FPR **+0.2375** and **+0.1830**. Blocks 510 → 4,350. Precision at base rate 0.586 → 0.103. Recall gains: slow_drift **+0.667 [0.333, 1.000]** (seed 0), splice **+0.385** (seed 0). | Δ benign FPR **+0.1247** and **+0.1235**. Δ control FPR **+0.2094** and **+0.1830**. slow_drift **+0.556** (seed 0). | not a group | ×1.05 / ×1.04 | **off**: net-negative as built ‡ |
| `sessions` | Decisions identical. Shadow halts at θ_halt = 0.6: false-halt 0.29% and 0.29%, runaway recall 0.819 and 0.794. | Identical decisions. False-halt 0.28% and 0.30%. | not a group | ×1.09 / ×1.08 | **on** (shadow: records and reports only), once T-51 is decided |
| `provisional` ‡ | Δ benign FPR **−0.0011 [−0.0014, −0.0008]** and **−0.0020 [−0.0026, −0.0014]**. Δ control FPR 0.0000. Recall: magnitude_spike **−0.300 [−0.431, −0.173]** and **−0.364 [−0.500, −0.226]**; prereq_skip −0.333 (seed 0); exfil_volume −0.100 and −0.143. | Δ benign FPR **−0.0295** and **−0.0337**. Δ control FPR **−0.0472** and **−0.0180**. Recall: magnitude_spike **−0.320** and **−0.382**; splice **−0.423** and **−0.217**. Median benign θ 0.916 → 1.000 (seed 0; the T-36 drift). | 0 carried. Removing it **gains** 24 and 29 (spec §5.2 allow-all) and 24 and 30 (oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic)) attacks. | ×1.14 / ×1.15 | **off**: net-negative as built ‡ |
| `seq_age_discount` | Scores move. No metric moves beyond ±0.0001. | Δ benign FPR +0.0001 [0.0000, 0.0002] (seed 0). | not a group | ×0.99 / ×0.97 | **off** (no measured benefit) |
| `gate` ‡ | Blocks 510 → 7 and 631 → 13; approvals 575 → 1,079 and 543 → 1,161. The flagged set moves by one call: T-59's gate-only rule blocks one benign call on seed 0 whose charge id was first used in an earlier call's arguments (the T-59 residual: reviewer load, not a bypass); Δ benign FPR +0.0000 [0.0000, 0.0001]. | Δ control FPR −0.0015 [−0.0048, 0.0000] (seed 0). | 0 | ×1.06 / ×1.12 | **off** until approvals are enforced (Tier 4) |
| `escalate` ‡ | Decisions identical. Escalation rate on benign test rows 0.67% and 0.73% (bar 2%). Δ control FPR 0.0000, so it does not lower control FPR (F5). | Identical decisions. Escalation rate 1.47% and 1.67%. | not a group | ×1.04 / ×1.05 (was ×1.32 before T-60) | **off**: no decision changes without a judge |
| `session_enforce` | Δ benign FPR **+0.0027 [0.0019, 0.0037]** and **+0.0041 [0.0029, 0.0055]**. Blocks 510 → 799 and 631 → 934. | Δ benign FPR **+0.0026** and **+0.0037**. | not a group | ×1.09 / ×1.07 | **off** (enforcement is Tier 4; T-47 is pending) |
| `authority` | Blocks 510 → 485 and 631 → 588. The flagged set is unchanged (Δ 0). | Δ benign FPR −0.0000 and **−0.0014 [−0.0017, −0.0010]**. | not a group | ×1.03 / ×0.99 | **on** candidate: it never raises a decision, and the user decides |
| `engine-traj` (all on) ‡ | Δ benign FPR **+0.1081** and **+0.1061**. Δ control FPR **+0.2434** and **+0.1797**. magnitude_spike **−0.340** and **−0.455**. Precision at base rate 0.586 → 0.123. | Δ benign FPR **+0.0747** and **+0.0669**. Δ control FPR **+0.1962** and **+0.1650**. | table in the report | ×1.37 / ×1.34 (was ×1.38 / ×1.99 before T-60) | not a default |

Bold = the CI excludes 0.

- **Composite rows.** `gate+flow_ledger` behaves like `gate`: 7 and 13 blocks, Δ benign FPR
  +0.0000 under spec §5.2 allow-all. `escalate+gate+flow_ledger` restores confirmed blocks (427 and 449) with
  escalation rates of 0.66% and 0.71%, and its Δ control FPR is 0.0000. `sessions+vital` is the
  F7 shadow configuration in the `vital` row.
- **Which switches move a score** (corrected, final review m2). Under spec §5.2 allow-all only
  `flow_ledger` and `seq_age_discount` move a per-call score (and the composites and
  `engine-traj`, which contain them): every call is learned as allowed, and no other switch
  changes what the detectors learn. Under the oracle-reviewer approval loop (labels act as a
  perfect human reviewer → optimistic) every switch that changes a decision also moves later
  scores through outcome feedback (a held or blocked call's outcome decides what is learned):
  `br_mode="modulate"`, `provisional`, `gate` (seed 0), `session_enforce` and `authority`, plus
  the two above; `vital`, `br_mode="score"`, `sessions` and `escalate` do not (the report's
  "scores = engine" column). Every AUPRC CI overlaps `engine`'s on both seeds and protocols
  (seed 0, spec §5.2 allow-all: `engine` 0.659 [0.541, 0.756]).
- **Session AUROC** (attacked against benign sessions) is 0.760 on all sessions and 0.746 on the
  length-matched set for seed 0, and 0.815 / 0.785 for seed 1 (spec §5.2 allow-all). No switch
  moves it by more than 0.01. A small part of the session signal is length.
- **Splice** is reported separately (T-40). Recall at per-agent 95% thresholds is 25/26 and
  22/23. Decision recall for `engine` is 0.115 and 0.522 under spec §5.2 allow-all. Only
  `br_mode="modulate"` raises it.

## Measured with four read tools learned as writes (I4)

The rows marked ‡ (`br_mode="modulate"`, `gate`, `escalate`, `provisional`, `engine-traj`) and the
"net-negative as built" verdicts on `br_mode="modulate"` and `provisional` were measured on an
engine that sees four catalog read tools as writes:

- **Why.** The bench's hints carry annotations and `has_schema` but no `kind`, so the sequence
  detector learns `kind` from history. `salesforce.get_record`, `stripe.retrieve_charge`,
  `stripe.retrieve_customer` and `zendesk.get_ticket` take an id argument and return no item
  count, so they are learned as writes (72–93% of their calls on the CI dataset; Task 1.7, which
  asked Task 4.1 to supply kind hints before measuring modulate; 4.1 did not).
- **What a learned write changes.** Its blast radius `b` is about 0.2 higher (0.35 instead of
  0.15 on the CI dataset, Task 1.7). Under `br_mode="modulate"` that lowers θ_eff = θ·(1 − 0.6·b)
  by about 21% on those calls instead of about 9%. It also enters the `is_write` term of
  irreversibility (`destructive or (is_write and b >= 0.6)`), which decides whether `gate` and
  `escalate` run on a call and whether `provisional` may allow it; on the CI dataset those calls
  stay under 0.6 (0 of 412 irreversible, Task 1.7), and this was **not measured on the bench**.
- **Consequence.** The ‡ numbers stand as measured, but they are not yet evidence about these
  switches on an engine that knows its reads. **Prerequisite** before any ‡ switch is measured
  again for a default: supply catalog `kind` hints to the bench (and to deployments that capture
  schemas). Not implemented here.

## Latency (R25)

- **Constraint.** The global constraint is p99 < 200 µs for `Engine.score`, measured by
  `tests/perf/test_latency.py` with every trajectory switch on. The budget is not raised.
- **The every-switch case** (T-55). It times `engine-traj` with the dataset's schemas on real
  bench calls: the bench config cut to 12 days; 10,000 warm-up calls, then 10,000 timed.
- **Non-vacuity.** A separate test asserts that the timed calls exercise the paths: 1,018
  escalated, 3,782 gate outcomes, 616 non-OK session verdicts, 249 provisional (626 and 246
  before T-58 / T-61).
- **Result: passes since T-60.** F5's Python "fresh re-score" re-ran the fusion on the same store
  and envelope and always returned the fast score; it now reuses the fused score (output
  byte-identical, T-60). p99 **140.6 µs** (p50 79.9 µs) in the test run (load average 8.2 at its
  start); 137–157 µs over 7 runs of the same measurement at load average 2.5–3.0, against
  214–216 µs for the T-12 re-score alternated on the same data. Escalated calls: p50 98–102 /
  p99 164–198 µs (were 152–153 / 260–266 µs). The strict xfail is removed.
- **Margin.** The pass has little margin on a slow machine. Later the same day, on battery
  power (`pmset` power mode 1) with load average 2.8–3.9, everything ran about 1.4× slower
  (default engine p50 157–161 µs instead of 115 µs): the every-switch p99 was 197.7, 219.2,
  199.5 and 195.8 µs in 4 runs of `tests/perf/test_latency.py` (3 under budget), and the
  default-engine case failed once in the same runs (p99 184–209 µs). Under those conditions the
  timings also drifted between consecutive runs (a T-12 re-score run at p99 316–334 µs, then a
  reuse run at 340–356 µs), so they are not an A/B comparison. Like the default-engine case,
  the every-switch case needs a quiet machine on mains power.
- **Real traces (SW1 seed 0, alternated A/B, `latency_ab`).** engine-traj p99 ×1.34 of `engine`
  (208 µs against 155 µs in this loaded run; ×1.99, 224 µs against 113 µs, before T-60), and
  `escalate` alone ×1.05 (was ×1.32).
- **Optimisations** (Task 4.1, output byte-identical): `normalize` computes `arg_tokens` once,
  the blast-radius tokenizer is memoized, and the session density is computed once per call;
  `tests/core/test_latency_equivalence.py` pins them.
- **The default-engine case** passes in the same test run: p50 115.6 / p99 167.4 µs at load
  average 8.2. On this shared machine it fails under heavy load (283–307 µs at load average 14–18,
  Task 4.1), as it did before this branch.

## Calibrated thresholds (`calibrated.json`)

- **Rule.** `select_on_calibration` is applied to `engine-traj-shadow` (spec §5.2 allow-all,
  seed 0). Its result is recorded as `theta_halt_selected_cal = 0.2`, the same in the re-run.
  The warn threshold is undefined, because nothing on the grid lies below the floor and warn must
  be strictly below halt.
- **Keys.** The record uses keys that no engine reads; the `EngineConfig` defaults stay at
  0.33 / 0.6. The history line says "not for deployment: decision 010 keeps 0.33/0.6; 3-agent
  calibration partition too small" (T-54).
- **Calibration partition.** The 3 calibration agents (9,723 benign sessions) are quieter than
  the rest of the bench: 0.72% of their benign sessions reach density 0.2. So the 1% halt bar is
  already met at the bottom of the grid.
- **Whole bench.** At θ = 0.2, engine-traj-shadow false-halts 2.85% (seed 0) / 3.15% (seed 1) of
  all benign sessions, against 0.40% / 0.37% at θ = 0.6. `sessions+vital` gives the same
  selection (calibration share 0.31% at 0.2) and false-halts 1.05% of all benign sessions at
  0.2, against 0.30% at 0.6.
- **Recommendation.** Keep the defaults 0.33 / 0.6. With 3 agents the calibration partition is
  too small to stand for the fleet. The recorded value shows what the rule gives, not a value to
  deploy.

## Exit acceptance (`tests/acceptance/test_trajectory_exit.py`)

Both seeds, both protocols. A failing criterion is a strict xfail with its measured numbers (the
Tier-1 pattern). No bar, threshold or default was changed. Re-run on `c7c123f` with the bench
datasets (T-56).

| criterion | spec §5.2 allow-all | oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic) |
|---|---|---|
| F2 benign vital false-flag ≤ 1% of sessions | xfail: 3.85% / 3.63% (T-30) | xfail: 3.85% / 3.63% |
| F5 escalation lowers control FPR | xfail: unchanged (0.0265, 0.0163) | xfail: unchanged (0.0737, 0.0343) |
| F5 recall within 1 point of `engine` | pass (unchanged) | pass (unchanged) |
| F6 provisional control FPR ≤ 3% | pass: 0.0265 / 0.0163 | pass: 0.0265 / 0.0163 |
| F6 provisional control FPR below `engine` | xfail: equal to `engine` | pass: against 0.0737 / 0.0343, but magnitude_spike recall −0.32 / −0.38 |
| F7 sensitivity non-increasing, no step > 10 points | pass: max step 1.7 / 1.9 points | pass: 1.7 / 1.8 points |
| F7 first-flag false-halts above `engine-traj` | pass: 21.8% / 20.3% against 0.40% / 0.37% | pass: 21.9% / 21.2% against 0.40% / 0.38% |
| F10 removing a group changes exactly its rows | xfail: removing `provisional` gains 24 / 29 attacks (also magnitude, velocity, sequence 1–5) | xfail: `provisional` gains 24 / 30 (also magnitude 1–2, vital 2, velocity 1) |

Result of the re-run: 9 passed, 7 xfailed (strict) in 16 min (`AFW_TRAJ_TRACES=traces/bench`,
4 workers, load average 4.0 at start), the same pass / xfail pattern as Task 4.1. That run was on
`9413905`, before R28 reverted T-59's ledger change; it was not repeated after it. The rows the
criteria read are unchanged by the revert (`vital`, `escalate`, `provisional`, the attribution)
or move below the precision of the table (`engine-traj-shadow` max sensitivity step 0.0167 →
0.0168 on seed 0; first-flag false halts unchanged).

The table's T-20 totals hold by construction (`tabulate` guarantees them), so they are not
evidence and are not asserted. F10's real criterion is the literal no-gain check (T-52), and that
check fails. Each conjunctive criterion is split into one test per clause, so a failing clause
never masks a holding one (review round 1).

Two F2 bars that the final review's spin fix (T-58) can no longer meet are strict xfails with
their measured numbers, not in this bench file: `test_spin_within_w_spin` (first index 5, bar
≤ 4) and `test_spin_injection_detected_within_w_spin_events` (1 / 91 runs within 4 events of
onset, 90 at 5; 91 / 91 within 5).

## Open user decisions (not decided here)

Each row is in `009-trajectory-clarifications.md`. The numbers are from this bench (seed 0, spec
§5.2 allow-all, all splits, unless stated otherwise), re-measured on `c7c123f`. None of them
affects a default.

- **T-30: loop false flags on benign sessions.**
  - 3.85% of benign sessions (seed 0) and 3.63% (seed 1) have a full loop, spin or stall score,
    against F2's 1% bar. The numbers are identical under the oracle-reviewer approval loop (labels
    act as a perfect human reviewer → optimistic).
  - Vital never fuses, so this costs no per-call decision. With `sessions` it does lift runaway
    halt recall, from 0.819 to 0.944, at a false-halt rate of 0.30%.
  - Options: (a) contiguous-run loop counting; (b) demote task-repeat patterns; (c) make polling
    an exemption.
- **T-38: exploration budget versus tolerance band.**
  - With `provisional` alone, 161 calls in 151 sessions raise `OFF_TRACK`; 34 of those sessions
    are benign.
  - With `engine-traj` it is 822 calls in 792 sessions, 642 of them benign (1.6% of benign
    sessions).
  - On the test split: 24 / 34 `OFF_TRACK` flags with `provisional`, and 299 / 365 with
    `engine-traj` (seeds 0 / 1).
  - Options: (a) budget in calls; (b) require at least 2 unresolved entries; (c) clamp per-call
    mass.
  - Not needed if `provisional` stays off, which is the recommendation above.
- **T-47: session-overlay blocks and the F8 ceiling.**
  - With every switch on and the dataset's schemas, no call is `bare`: 174,080 calls are
    `history+schemas` and 3,380 are `schemas_only`.
  - 473 sessions halt (163 benign, 220 runaway, 90 attacked). They contain 6,934 blocks after
    their halt, 342 of them in benign sessions.
  - Of those blocks, 11 are graded `cold`. Under `authority` a cold call can only be blocked by
    the overlay or by a gate prerequisite block exempt from the ceiling, and the ceiling would
    have capped either kind. The bare-maturity case T-47 records (6 blocks on the CI dataset)
    needs a schema-less deployment.
  - Options: (i) grade the overlay by the session regime; (ii) exempt only vital-triggered halts;
    (iii) accept the overlay's exemption as a spec amendment.
  - Not needed while `session_enforce` stays off.
- **T-51: halted session records are never evicted.**
  - In the shadow configuration (`sessions+vital`), 427 of 42,289 sessions halt over the 90 days
    (121 benign, 219 runaway, 87 attacked), about 4.7 a day for 20 agents. Each record stays in
    the store for as long as the process runs.
  - Options: (a) evict halted records after a long TTL while `session_enforce` is off; (b) keep
    T-15 strict and add a cap and a halted-record metric.
  - Recommending `sessions` on in shadow (above) makes this decision necessary first.
- **`br_mode="modulate"` and `provisional` as default candidates.** The measurements above say
  no, as built, with the ‡ caveat. The follow-ups below say what would have to change before
  either is measured again.

## Follow-ups (not done here)

- **Kind hints (prerequisite, I4).** Supply catalog `kind` hints to the bench before any ‡ row is
  measured again for a default; without them 4 read tools are learned as writes (above), and
  stall fires on only 27/110 injections (T-28).
- **Recalibration on the calibration partition.** Re-select `z0`, `tau`, `prereq_rate` and the
  fusion weights there (F10). That changes Tier-1 defaults the user ruled on in decision 008, so
  it needs that decision first.
- **F9** (mined session templates): it is out of scope (T-01), and the spec says not to build it
  before F1–F8 are measured. They now are.
- **Tier-4 enforcement**: Go proxy, Redis atomicity, Cedar, live graduated responses. `gate`,
  `session_enforce` and the approval band only matter once approvals and halts are enforced. The
  Go port keeps a real re-score for F5 (its fast path takes shortcuts; T-60).
- **`br_mode="modulate"`.** The calibrator fits θ on the benign tail of fused scores, but the
  decision uses θ·(1 − 0.6·b), which sits deep in that tail. Calibrate against θ_eff (per-b
  quantiles), or cap `br_k`, before modulate is a candidate again.
- **`provisional`.** It suppresses blocks of reversible calls whose evidence is a magnitude
  spike, which drops `magnitude_spike` recall by about a third. Its executed calls also push θ
  up (T-36: median benign θ 0.916 → 1.000 under the oracle-reviewer approval loop (labels act as
  a perfect human reviewer → optimistic)). Exempt magnitude-driven candidates and censor
  provisional calls in the calibrator, then measure again.
- **The F5 judge** must catch judge exceptions (abstain) and must not be asked twice on a cache
  miss before a real judge is wired in.
- **A bigger calibration partition**, or per-agent θ, for the session thresholds.
