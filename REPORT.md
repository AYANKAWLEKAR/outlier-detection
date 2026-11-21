# Trajectory-monitoring branch report

Branch `feat/trajectory-monitoring-impl` (worktree `.superpowers/worktrees/trajectory-monitoring`), cut from `feat/tiers-0-2` and last rebased onto it at `4d8878b` (`feat/tiers-0-2` has moved on since). It implements every task of [the plan](docs/superpowers/plans/2026-11-11-trajectory-monitoring.md) for [the feature spec](docs/feature-spec-trajectory-monitoring.md): F1–F8 and F10 (Tasks 0.1–4.1), plus one fix wave from the final whole-branch review (T-58 to T-64). F9 and Tier 4 are out of scope (T-01).

**The default engine is unchanged.** Every feature sits behind an `EngineConfig` switch that defaults to off (T-03). `tests/golden/*.sha256` and `agentfw/bench/baseline_metrics.json` are byte-identical to the base, and nothing under `results/` is committed. Checks at the last change: `scripts/gate.sh` passes (ruff, format, mypy --strict; pytest 1788 passed, 1 skipped, 3 xfailed, slow tests deselected), `ci_regress` passes, and the slow files the final fixes touch pass: trajectory exit acceptance (9 passed, 7 strict xfails), the gate allow-side suite (2 passed), and the latency tests. The mains-power latency numbers (3 passed) are from `ec35ccf`; the `engine-traj` scoring path is unchanged after it (later commits touch flags of engines that are not `engine-traj`, rendering, docs, and the R28 ledger revert, which leaves the timed stream's path counts identical: 1,018 escalated, 3,782 gate outcomes, 616 non-OK verdicts, 249 provisional). On battery power the latency results were mixed (section 2).

The branch name differs from the plan's because the main checkout holds `feat/trajectory-monitoring`. That branch can be fast-forwarded to this one.

Sources. Every number below comes from decision 010 ([`010-trajectory-defaults.md`](docs/decisions/010-trajectory-defaults.md)), from [the benchmark report](docs/trajectory-benchmark.md) (one run of `python -m agentfw.bench.trajectory` on committed sha `c7c123f`, both seeds, both protocols, with the rows that read ledger refs re-measured after ruling R28; its header says which), or from the interpretation records in [`009-trajectory-clarifications.md`](docs/decisions/009-trajectory-clarifications.md) (the T-rows; not upstream's `009-a1-defaults-accepted.md`). Numbers under the oracle-reviewer approval loop are **optimistic**: labels act as a perfect human reviewer.

## 1. What was built

| Task | Feature | Switch (default) |
|---|---|---|
| 0.1 | Decision record `009-trajectory-clarifications.md`, response-content guard test | — |
| 0.2 | Calibration partition (three-way split) | `SplitSpec.calibration_frac` (0.0) |
| 0.3 | Precision at base rate, AUROC, normalizer-hash recalibration guard (`calibrated.json`) | — |
| 0.4 | `ScoreRow.decision`, leave-one-group-out attribution | — |
| 1.1 | Per-session artifact ledger, task-input seeding (F1) | — (nothing reads it by default) |
| 1.2 | Call / response fingerprints, MinHash (F2) | — |
| 1.3 | Generator: task ids, response fingerprints, edge ground truth | — |
| 1.4 | Id flow over the ledger (F1) | `flow_ledger` (off) |
| 1.5 | Vital signs: loop / spin / stall, reported, never fused (F2) | `enabled=KNOWN_DETECTORS` (v1 only) |
| 1.6 | loop / spin / stall runaway families, session labels (F2, F10) | opt-in families |
| 1.7 | MCP annotations, lexical + structural blast radius, irreversibility, regime (F3, §6.9) | `br_mode` ("off") |
| 1.8 | Sandbox proxy: tools/list schema capture, response fp, task ids | shadow only |
| 2.1 | Session flag window and density (F7 substrate) | `sessions` (off) |
| 2.2 | Provisional flags for reversible calls, `decide()` (F6) | `provisional` (off) |
| 2.3 | Age discount on transition surprise (F6) | `seq_age_discount` (off) |
| 2.4 | Splice family, length-matched set, session label column (F10) | opt-in family |
| 3.1 | Pre-execution gate for irreversible calls (F4) | `gate` (off) |
| 3.2 | Escalation with an optional judge in the approval band (F5) | `escalate` (off) |
| 3.3 | Session verdicts (warn / halt) on severe-flag density, halt metrics (F7) | `session_enforce` (off) |
| 3.4 | Verdict authority by evidence maturity, reason rendering (F8) | `authority` (off) |
| 4.1 | Trajectory benchmark CLI, exit acceptance, decision 010 (F10) | — |

The final `decide` order is: bands on θ_eff → gate (irreversible) → escalation → provisional (reversible) → ceiling → session overlay.

Each task had a fresh implementer and a fresh reviewer: one commit per task, plus fix rounds. The final whole-branch review found five important issues and six minor ones, fixed in one wave (one commit per finding, to be squashed):

| Finding | Fix | Record |
|---|---|---|
| I1: spin read the current call's own response, so the proxy (which scores before the response) never raised `SPIN` | Spin is computed over completed calls only (one call of lag); `ActionEnvelope.delta_i` removed; proxy-style scoring now equals replay in every result field and in the stored state | T-58 |
| I2: a fabricated id used first in an earlier call's arguments passed the F4 gate | The gate counts only refs from a response or the task input (or a Bloom-only id) as present. The ledger is unchanged: letting a later response ground an args-first id reopened the bypass through tools that echo their arguments (ruling R28); legitimate args-first flows are gated instead (reviewer load, not a bypass) | T-59 |
| I3: this report was stale | Rewritten from 010 and the benchmark report | T-64 |
| I4: 010 did not say that four read tools were learned as writes | Disclosed on the affected rows; kind hints listed as a prerequisite | T-64 |
| I5: F5's re-score re-ran the same computation | Escalation reuses the fused score (output byte-identical); the every-switch p99 test passes | T-60 |
| m1, m3, m5, m6 | Flags no longer depend on unrelated switches; density docstring; an overlay block over a cap renders correctly; reports mark a dirty tree | T-61 – T-63 |
| m2 | 010's "which switches move a score" corrected | T-64 |

## 2. Acceptance bars

| Bar (spec) | Result |
|---|---|
| F1: ledger recovers grammar edges, precision and recall ≥ 0.99 | **met**: 1.000 / 1.000 against grammar-provenance ground truth (T-row of Task 1.3) |
| F1: a fabricated id on a warm key scores r_flow; a task-seeded id scores 0 | **met** |
| F2: loop fires on the 2nd duplicate; loop injection detected within 1 event | **met**: 110 / 110 loop injections on the CI-size set |
| F2: spin within `w_spin` = 4 events of onset | **not met since T-58**: spin reads completed calls only, so it fires one call later: 1 / 91 spin runs within 4 events, 90 at 5, 91 / 91 within 5. Strict xfails (the unit test fires at index 5, bar ≤ 4) |
| F2: benign vital false-flag rate ≤ 1% of sessions | **not met**: 3.85% / 3.63% (bench seeds 0 / 1). Strict xfail, T-30 pending |
| F3: blast-radius ranking vs hand labels, Spearman ρ ≥ 0.8 | **met**: ρ = 0.909 |
| F4 G-a: 0 false blocks on 200 valid irreversible sessions | **met**: 0 / 200 after T-59; the stale-state regression still catches 27 / 200 |
| F5: escalation rate on benign rows ≤ 2% | **met on the bench** (`escalate` alone 0.67% / 0.73% under spec §5.2 allow-all, 1.47% / 1.67% optimistic); **not met on the CI dataset** (5.51%, strict xfail, T-42) |
| F5: escalation lowers control FPR without losing recall | **not met**: control FPR unchanged on both seeds and protocols; recall within 1 point holds |
| F6: control FPR on unusual-benign ≤ 3% and below `engine` | ≤ 3% **met**; below `engine` **not met** under spec §5.2 allow-all (equal), met only under the optimistic loop |
| F7: halt sensitivity has no cliff; first-flag baseline is worse | **met** on both seeds and protocols |
| F8: no block from cold or bare maturity (prerequisite exempt with schemas) | **met with schemas**; without schemas, session-overlay blocks on bare calls are a strict xfail pending T-47 |
| F10: removing a group changes exactly its rows | **not met**: removing `provisional` gains 24 / 29 attacks (T-52) |
| Latency: `score` p99 < 200 µs with every switch on | **met since T-60** on mains power: 140.6 µs (load average 8.2); thin margin on battery power (195.8–219.2 µs, 3 of 4 runs under budget, while the default-engine case also failed once) |

## 3. Benchmark

Two seeds of the bench config (SW1 plus loop / spin / stall / splice; seed 0: 177,460 envelopes, 855 attacks, 42,289 sessions), time split with a 15% calibration partition, each switch toggled alone on top of `engine`. Full tables: [`docs/trajectory-benchmark.md`](docs/trajectory-benchmark.md); summary and recommendations: [`010-trajectory-defaults.md`](docs/decisions/010-trajectory-defaults.md). Pairs of numbers are seed 0 / seed 1 under spec §5.2 allow-all.

`engine` reference, decision benign FPR / control FPR: 0.0106 / 0.0265 (seed 0) and 0.0125 / 0.0163 (seed 1).

**All switches together are not a good default today.** `engine-traj` raises the benign decision FPR by +0.108 / +0.106 and the control FPR by +0.243 / +0.180, and loses magnitude spikes (recall −0.34 / −0.45). Two switches cause almost all of this:

- **`br_mode="modulate"`**: benign FPR +0.140 / +0.144; blocks 510 → 4,350 (seed 0). It raises slow_drift recall (+0.667, seed 0) and splice (+0.385, seed 0). θ·(1 − 0.6·b) sits deep in the benign tail the calibrator fitted.
- **`provisional`**: lowers the benign FPR a little (−0.0011 / −0.0020) and loses attacks (magnitude_spike −0.300 / −0.364); removing it from `engine-traj` gains 24 / 29 detected attacks.

Both verdicts were measured without `kind` hints: four catalog read tools are learned as writes, which moves `b`, θ_eff under modulate, and irreversibility (010, I4). Kind hints are a prerequisite before measuring these again.

The other switches:

- **`flow_ledger`**: decisions unchanged; per-agent FPR@95 0.578 → 0.601 and 0.682 → 0.583.
- **`gate`**: blocks 510 → 7 and 631 → 13 (approvals instead); the flagged set moves by one benign call on seed 0, a legitimate flow that first used its charge id in arguments (the T-59 residual).
- **`escalate`**: decisions unchanged without a judge; 0.67% / 0.73% of benign test rows escalate.
- **`authority`**: blocks 510 → 485 and 631 → 588; it never raises a decision.
- **`session_enforce`**: benign FPR +0.0027 / +0.0041.
- **`vital`, `sessions`, `br_mode="score"`**: scores and decisions identical to `engine` (T-07, T-03).
- **`seq_age_discount`**: no metric moves beyond ±0.0001.

**Session level** (all splits). With `sessions` + vital in shadow at θ_halt = 0.6: false-halt 0.30%, runaway recall 0.944 / 0.877 (0.819 / 0.794 with `sessions` alone), median 7 calls from onset to halt. The first-flag baseline false-halts 4.2% / 4.0%. Attacked sessions halt only 7–9% of the time in every configuration: halts catch runaways, not attacks. Session AUROC (attacked vs benign) is 0.760 / 0.815, and 0.746 / 0.785 on the length-matched set. Splice decision recall is 0.115 / 0.522 for `engine`.

**Latency** (alternated A/B on seed 0, ratios to `engine`): every single switch is within ×1.15 at p99; `engine-traj` is ×1.37 / ×1.34 (p50 / p99). The `tests/perf/test_latency.py` every-switch case passes at p99 140.6 µs on mains power; on battery power it sits at the budget (010, Latency).

**Recommendations** (010; none changes a default): `br_mode="score"` on (reporting only); `sessions` on in shadow once T-51 is decided, with vital once T-30 is decided; `authority` an on candidate for the user to decide; `flow_ledger` off or together with `gate`; everything else off.

## 4. Decisions waiting on you

Each is a T-row in `009-trajectory-clarifications.md`, with numbers in 010. None affects default behaviour.

| Row | Question | Options recorded |
|---|---|---|
| **T-30** | Vital flags 3.85% of benign sessions (bar 1%), mostly task `repeat` iterations re-issuing the same call inside the loop window. | Count only contiguous repeats; demote task-repeat patterns; make polling an exemption |
| **T-38** | F6's 25% exploration budget makes one abandoned branch raise `OFF_TRACK`: 822 calls in 792 sessions (642 benign) with `engine-traj`, seed 0. | Budget in calls; require ≥ 2 unresolved; clamp per-call mass |
| **T-47** | The F8 ceiling does not cap session-overlay blocks: 11 cold-graded blocks after a halt on the bench (seed 0); bare-call blocks without schemas. | Grade the overlay by session regime; exempt vital-only halts; accept as a spec amendment |
| **T-51** | Halted sessions are never evicted (T-15), including in shadow mode: 427 of 42,289 sessions in 90 days (seed 0). | Evict after a long TTL when not enforcing; keep strict with a cap and a metric |
| 010 | Defaults: which switches, if any, to turn on. | As recommended above |

## 5. Decisions made without you

Each is recorded with its reasoning in `009-trajectory-clarifications.md`. The ones most likely to matter:

- **T-24.** Upstream's pinned trace hashes are kept as authored; tests check them on traces with `task_ids` / `fp` stripped.
- **T-32.** Bench runs load `schemas.jsonl` as hints (annotations only, no `kind`; see I4).
- **T-34.** One session gate (`Engine._book`), opened by `sessions`, `provisional` or `session_enforce`.
- **T-35 / T-41 / T-42 / T-61.** Irreversibility comes from the blast radius whenever a feature needs it or the session gate is open, even with `br_mode="off"`.
- **T-41 / T-59.** The gate counts only id-like values as missing, and as present only ids whose first sighting in the session was a response or the task input (or a Bloom-only id).
- **T-43.** The engine order is decide → flags → provisional → density → verdict → overlay; overlays read the halted state from before the call.
- **T-50.** The maturity contributor floor is min(0.3, θ_eff − cal_delta).
- **T-54.** The calibration-partition session thresholds (θ_halt 0.2, warn undefined) are recorded in `calibrated.json` but not used; the defaults stay 0.33 / 0.6.
- **T-58.** Spin is computed over completed calls only; two F2 timing bars became strict xfails with their measured numbers.
- **T-60.** The Python F5 re-score reuses the fused score; the Go port keeps a real re-score.

## 6. Known limits and follow-ups

- **Kind hints.** Without them the engine learns four read tools as writes. This qualifies the `br_mode="modulate"`, `gate`, `escalate`, `provisional` and `engine-traj` measurements, and stall fires on only 27 / 110 injections (T-28). Supplying them is the first follow-up.
- **The F5 judge** is asked twice on an update that misses the cache, and judge exceptions are not caught. Both must be fixed before a real judge is wired in.
- **The tracker's session state is never evicted** (deferred minor m7), so after the engine evicts an idle session, the tracker's ring still feeds spin on that session's next call.
- **The default-engine latency test** fails on this shared machine under heavy load, as it did before this branch.
- **Gated args-first flows** (T-59 residual). A legitimate flow that first uses an id in arguments and only later sees it in a response (a charge the user named, looked up, then refunded, without `task_ids` capture) is blocked or held on a warm key: reviewer load, not a bypass.
- Deferred minor findings (m4, m7, m8 and the per-task minors) are kept in the controller's ledger outside the repository.
