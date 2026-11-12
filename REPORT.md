# Trajectory-monitoring branch report (2026-11-12)

Branch `feat/trajectory-monitoring-impl` (worktree `.superpowers/worktrees/trajectory-monitoring`), HEAD `8d63325`, rebased on `feat/tiers-0-2` @ `a60f7cf`. It implements Tasks 0.1–3.4 of [the plan](docs/superpowers/plans/2026-11-11-trajectory-monitoring.md) from [the feature spec](docs/feature-spec-trajectory-monitoring.md) (F1–F8, F10). Task 4.1 (trajectory benchmark CLI, exit acceptance, decision 010), F9 and Tier 4 are **not** done.

**The default engine is unchanged.** Every feature sits behind an `EngineConfig` switch that defaults to off. `tests/golden/*.sha256`, `agentfw/bench/baseline_metrics.json`, `results/` and `docs/leaderboard.md` are byte-identical to `feat/tiers-0-2`. Checks: `scripts/gate.sh` passes (1718 passed, 1 xfailed), `ci_regress` passes, and the SW1 replay of `engine` reproduces the published leaderboard exactly.

The branch name differs from the plan's because the main checkout holds `feat/trajectory-monitoring` (at `2aafb11`). That branch can be fast-forwarded to this one.

## 1. What was built

| Task | Feature | Switch (default) | Commit message |
|---|---|---|---|
| 0.1 | Decision record 009, response-content guard test | — | docs: trajectory feature spec, clarifications 009; response-content guard |
| 0.2 | Calibration partition (three-way split) | `SplitSpec.calibration_frac` (0.0) | feat(bench): calibration partition |
| 0.3 | Precision at base rate, AUROC, normalizer-hash recalibration guard (`calibrated.json`) | — | feat(bench): precision at base rate, AUROC, … |
| 0.4 | `ScoreRow.decision`, leave-one-group-out attribution | — | feat(bench): decision column; attribution table |
| 1.1 | Per-session artifact ledger, task-input seeding (F1) | — (nothing reads it by default) | feat(core): per-session artifact ledger |
| 1.2 | Call / response fingerprints, MinHash (F2) | — | feat(core): call and response fingerprints |
| 1.3 | Generator: task ids, response fingerprints, edge ground truth | — | feat(synth): task-input ids, … |
| 1.4 | Id flow over the ledger (F1) | `flow_ledger` (off) | feat(core): id flow over the artifact ledger |
| 1.5 | Vital signs: loop / spin / stall, reported but never fused (F2) | `enabled=KNOWN_DETECTORS` (v1 only) | feat(core): vital-signs detector |
| 1.6 | loop / spin / stall runaway families, session labels (F2, F10) | opt-in families | feat(synth,bench): runaway families |
| 1.7 | MCP annotations, lexical + structural blast radius, irreversibility, regime (F3, §6.9) | `br_mode` ("off") | feat(core): MCP annotations, blast radius, … |
| 1.8 | Sandbox proxy: tools/list schema capture, response fp, task ids | shadow only | feat(sandbox): capture tools/list schemas, … |
| 2.1 | Session flag window and density (F7 substrate) | `sessions` (off) | feat(core): per-session flag window and density |
| 2.2 | Provisional flags for reversible calls, `decide()` (F6) | `provisional` (off) | feat(core): provisional flags |
| 2.3 | Age discount on transition surprise (F6) | `seq_age_discount` (off) | feat(core): age discount |
| 2.4 | Splice family, length-matched set, session label column (F10) | opt-in family | feat(synth,bench): splice family, … |
| 3.1 | Pre-execution gate for irreversible calls (F4) | `gate` (off) | feat(core): pre-execution gate |
| 3.2 | Escalation with an optional judge in the approval band (F5) | `escalate` (off) | feat(core): two-tier escalation |
| 3.3 | Session verdicts (warn / halt) on severe-flag density, halt metrics (F7) | `session_enforce` (off) | feat(core,bench): session verdict engine |
| 3.4 | Verdict authority by evidence maturity, reason rendering (F8) | `authority` (off) | feat(core): verdict authority |

The final `decide` order is: bands on θ_eff → gate (irreversible) → escalation → provisional (reversible) → ceiling → session overlay.

Each task had a fresh implementer and a fresh reviewer. Tasks 0.4, 1.3, 1.6, 1.7, 1.8, 2.1, 2.2, 2.4, 3.1 and 3.4 each needed one fix round, and every fix was re-reviewed. Each task is one commit.

## 2. Acceptance bars

| Bar (spec) | Result |
|---|---|
| F1: ledger recovers grammar edges, precision and recall ≥ 0.99 | **met**: 1.000 / 1.000 against grammar-provenance ground truth (independently re-derived in review) |
| F1: a fabricated id on a warm key scores r_flow; a task-seeded id scores 0 | **met** |
| F2: loop fires on the 2nd duplicate; loop injection detected within 1 event | **met**: 110/110 loop attacks; spin 91/91 within 4 calls |
| F2: benign vital false-flag rate ≤ 1% of sessions | **not met**: 3.85% on SW1 (6.00% on CI). Strict xfail; see T-30 |
| F3: blast-radius ranking vs hand labels, Spearman ρ ≥ 0.8 | **met**: ρ = 0.909 (0.54 among write tools only) |
| F4 G-a: 0 false blocks on 200 valid irreversible sessions | **met**: 0/200. The stale-state regression catches 27/200 |
| F5: escalated p99 < 50 ms without a judge | **met**: 186 µs (CI); 314 µs on escalated SW1 calls |
| F5: escalation rate on benign rows ≤ 2% | **not met on CI**: 5.51%, strict xfail (T-42). On SW1, `escalate` alone gives 0.62% (allow) / 1.96% (review, optimistic) |
| F5: escalation lowers control FPR without losing recall | **not met on SW1**: control FPR unchanged |
| F6: control FPR on unusual-benign ≤ 3% and below engine | **not met on SW1**: allow 0.0235 (= engine); review 0.047 (below engine, but over 3%) |
| F7: halt sensitivity has no cliff; first-flag baseline is worse | **met**: max step 0.5 pts; first-flag false-halt 3.9% vs 0.30% |
| F8: no block from cold or bare maturity (prerequisite exempt with schemas) | **met with schemas**. Without schemas: 6/956 session-overlay blocks on bare calls, strict xfail pending T-47 |

## 3. Benchmark (SW1, seed 0, time split)

Full tables: `results/trajectory-2026-11-12/summary.md` and `summary.json` (gitignored, in the worktree). The run took about 25 minutes for 56 replays plus attribution and latency.

Datasets:
- **D1** = `sw1_config(0)`: 174,878 envelopes, 808 attacks. With the trajectory fields stripped it equals the leaderboard dataset (`a455b10c705d`).
- **D2** = D1 plus the loop / spin / stall / splice families: 40,840 benign, 232 runaway and 1,217 attacked sessions.

Every `review` number is **optimistic**: the oracle-reviewer loop treats labels as a perfect human reviewer.

**Reference.** `engine` on D1 reproduces `docs/leaderboard.md`: AUPRC 0.736 (allow) / 0.779 (review), per-agent FPR@95 0.542 / 0.490, and the same gate cells and verdicts.

### engine vs engine-traj (every switch on), D1

| Metric | allow: engine → traj | review (optimistic): engine → traj |
|---|---|---|
| AUPRC (warm) | 0.736 → 0.736 | 0.779 → 0.802 |
| FPR@95 per-agent | 0.542 → 0.526 | 0.490 → 0.470 |
| Decision benign FPR | 0.0098 → **0.109** | 0.048 → 0.096 |
| Decision control FPR | 0.023 → **0.282** | 0.085 → 0.282 |
| Decision recall: magnitude_spike / prereq_skip / exfil_volume | 1.00 / 1.00 / 0.79 → 0.77 / 0.67 / 0.63 | 1.00 / 1.00 / 1.00 → 0.77 / 0.67 / 0.79 |
| Decision recall: slow_drift (N=11) | 0.45 → 1.00 | 0.91 → 1.00 |
| Escalation rate (benign) | 0 → 6.7% | 0 → 7.0% |
| Latency p50 / p99, alternated A/B | 62 / 107 µs → 86 / 234 µs | — |

**All switches together are not a good default today.** They cost about 10× the benign decision FPR and lose attacks. Two switches cause almost all of this; the other switches are neutral or mildly positive.

### Per switch (each alone on top of engine, D1, allow)

- **`br_mode="modulate"`: the main false-positive cost.** Benign FPR goes 0.0098 → 0.153, control FPR to 0.270, blocks 960 → 6,820. Live θ sits at about 0.9996, but θ_eff = θ·(1−0.6·b) averages 0.689 on benign rows, deep in the benign score tail. Recall does rise: slow_drift 0.45 → 1.00, exfil 0.79 → 0.84.
- **`provisional`: loses attacks.** magnitude_spike 1.00 → 0.80, prereq_skip → 0.67, exfil → 0.68. FPR barely drops (0.0098 → 0.0090). Removing it from engine-traj *raises* detected attacks from 205 to 238. The θ drift predicted in T-36 is confirmed under review: median benign θ goes 0.912 → 0.998.
- **`flow_ledger`: the only switch that improves per-call ranking.** Per-agent FPR@95 0.542 → 0.526, global 0.587 → 0.560.
- **`gate`: turns blocks into approvals.** Blocks go 960 → 14; the 14 are all PREREQ_MISSING, 4 of them benign. The flagged set is unchanged.
- **`escalate`: restores confirmed blocks.** Blocks are 763 and 0.62% of benign rows escalate. Control FPR is unchanged.
- **`authority`: blocks 960 → 884**, the capped ones becoming approvals. Flagged share is unchanged.
- **`session_enforce` (with vital): blocks 960 → 1,740** and benign FPR 0.0098 → 0.0107.
- **`vital`, `sessions`, `br_mode="score"`: scores and decisions are byte-identical to engine**, as T-03 and T-07 require. `seq_age_discount` moves 6 decisions.

### Session level (D2, allow)

- **Halts (sessions + vital, θ_halt = 0.6):** false-halt 0.30%, runaway recall 0.957, median 6 calls from onset to halt. The first-flag baseline false-halts 3.9%. With engine-traj: false-halt 0.40%, runaway recall 0.966, first-flag 21.8%. **Attacked** sessions halt only about 7% of the time in every configuration: halts catch runaways, not attacks.
- **Vital false-flag rate on benign sessions:** 3.85% (T-30).
- **Splice:** attack recall at per-agent 95% thresholds is 0.966 (28/29), but **decision** recall is only 0.17 for engine and 0.38 for engine-traj.
- **Session AUROC (attacked vs benign):** 0.760 on all sessions and 0.746 on the length-matched set, essentially the same for every switch. Some of the session signal is length, but not much.
- **Attribution (engine-traj, D1):** vital, gate and velocity carry 0 attacks alone. Novelty and sequence carry most of them. The exactness checks pass.

## 4. Decisions waiting on you

Each one is a T-row in [decision 009](docs/decisions/009-trajectory-clarifications.md). None affects default behaviour.

| Row | Question | Options recorded |
|---|---|---|
| **T-30** | Loop fires on 3.85% of benign SW1 sessions (bar 1%). The cause is task `repeat` iterations re-issuing the same call within 8 steps, which is F2's own definition of a loop. | Count only contiguous repeats; demote task-repeat patterns; rely on polling hints |
| **T-38** | F6's 25% exploration budget, applied literally, makes a single abandoned branch raise OFF_TRACK in any session of 8 or more calls. That contradicts F6's "1/8 of a severe flag". engine-traj raises 318 OFF_TRACK flags on D1. | Budget in calls; require ≥ 2 unresolved; clamp per-call mass |
| **T-47** | Session-overlay blocks fall under the F8 ceiling literally: 6 bare-call blocks without schemas. | Grade the overlay by session regime; exempt vital-only halts; accept as a spec amendment |
| **T-51** | Halted sessions are never evicted (T-15), including in shadow mode, so records grow without bound in a long-running proxy. | Evict halted sessions after a long TTL when not enforcing; keep strict and add a cap/metric in 4.1 |
| new | Should `br_mode="modulate"` and `provisional` stay candidates for a default? The benchmark above says not as built. | For decision 010 (Task 4.1) |

## 5. Decisions made without you (T-19 – T-51)

Each is recorded with its reasoning in decision 009. These are the ones most likely to matter:

- **T-24.** Upstream's pinned trace hashes (`SMALL_OWN_FILES`, `SMALL_FAR_FILES`) are kept as authored. Tests check them on traces with `task_ids` / `fp` stripped.
- **T-26.** A minimal `ToolAnnotations` was pulled forward into Task 1.5.
- **T-32.** Bench runs auto-load `schemas.jsonl`, which widens the loop window for read-only + idempotent tools.
- **T-34.** One session gate (`Engine._book`), opened by `sessions`, `provisional` or `session_enforce`.
- **T-35 / T-41 / T-42.** Irreversibility is computed whenever a feature needs it, even with `br_mode="off"`.
- **T-40.** Splice is reported separately, at splice-free thresholds. Session AUROC counts attacked sessions only.
- **T-41.** The gate counts only id-like values as missing artifacts.
- **T-43.** The engine order is decide → flags → provisional → density → verdict → overlay. Overlays read the halted state from before the call.
- **T-50 (resolved).** The maturity contributor floor is min(0.3, θ_eff − cal_delta). Per-call blocks below 0.3 graded "mature" went from 8 to 0.

## 6. Rebases onto feat/tiers-0-2

| When | Upstream picked up | Conflicts / breaks |
|---|---|---|
| After 0.4 | 8 commits (A1 docs, A1.1–A1.4) | None. `calibrated.json` restamped (A1.2 changed `detectors/base.py`) |
| Before 1.6 | 2 (A1.4 top-up, manifest census) | New upstream far-mode trace pin. Fixed with the T-24 stripped-hash check |
| After 1.8 | 7 (A1 scoring default, new engine golden, A1.5–A1.7) | `test_report.py` and `report.py` imports: kept both sides |
| After 2.4 | 8 (gate matrix, Tier-1 acceptance under both protocols) | None |
| After 3.4 | 7 (A1 leaderboard, docs, gate readers) | None |

## 7. Known limits and follow-ups

- Stall fires on only 27/110 injections. Without catalog `kind` hints, the engine learns four read tools (`get_record`, `retrieve_charge`, `retrieve_customer`, `get_ticket`) as writes. Supplying kind hints is a Task 4.1 item.
- The F5 judge is called twice on an update that misses the cache, and judge exceptions are not caught (must abstain before a real judge is wired).
- The vital detector recomputes `arg_tokens` per call; this is the main cost if latency gets tight.
- The perf p99 latency test fails on this machine under load, on the base commit too. It needs a quiet-machine run.
- All deferred minor findings are listed in the SDD ledger (`.superpowers/sdd/2026-11-11-trajectory-monitoring/progress.md`), for the final whole-branch review that runs with Task 4.1.
