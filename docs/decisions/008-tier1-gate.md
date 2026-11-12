# 008 — Tier-1 exit gate: root causes and the fix candidates

Status: **decided (Task 1.15); defaults changed by user decision (2026-11-10); scoring, benchmark
data and gate criterion changed by spec Amendment A1 (2026-11-11).** The last section,
"Amendment A1 (2026-11-11)", holds the A1 results and supersedes every one-protocol gate statement
above it. The controller's rulings are in the Decision section. At Task 1.15 no engine default and
no benchmark-protocol default changed; the user later turned on the oracle-reviewer protocol and
three engine switches (see "User decision (2026-11-10)"). After Amendment A1, on SW1 seeds 0, 1
and 2, the amended criterion passes under the oracle-reviewer approval loop (labels act as a
perfect human reviewer → optimistic) and fails under spec §5.2 allow-all. Which protocol's result
closes Tier 1 is the user's open decision (D-028). One label-semantics change was made to the
generator: `exfil_volume` now labels the call that carries the response signal.

Evidence: `evidence/gate-seed0-seed1.md`, produced by the scripts in `evidence/008-root-causes/`
and the `gate` suite of `agentfw.bench.sweeps`. Numbers marked *(manual inspection)* were read off
a replay's per-detector evidence by hand and are not produced by a committed script.

## Context

The gate is defined in spec §9 and D-008/D-009, and the brief for Task 1.15 restates it:

- attack-level recall ≥ 0.90 for each gated family;
- measured at the engine's live per-agent thresholds (`"row"`) on warm test rows;
- `magnitude_spike` is gated in the m ∈ [10, 100] bucket, and `slow_drift` is excluded;
- `top_reason_share` of the targeted detector must be ≥ 0.90;
- benign FPR must stay ≤ 2 %.

At Task 1.14 the gate failed everywhere (min recall 0.388 on seed 0).

The datasets use the SW1 config: 20 personas, 90 days, all 8 families. Seed 0 is the Task 1.14
set. Seed 1 is new: 172,805 envelopes and 611 attacks.

## 1. Why each attack is missed

Method:

- Every warm-test gate attack that the default engine misses (spec allow-all replay) is replayed
  twice more, with label-reading oracles:
  - "oraclecal": the calibrator skips attack rows;
  - "oracle": nothing learns attack rows.
- Each miss is then classified by the first rule that matches. The order is `classify.py`'s
  docstring:
  1. **(i)** label semantics: the exfil signal carrier (the call right after the read) is not
     part of the attack.
  2. **(ii)** detected once the calibrator ("oraclecal") or nothing ("oracle", velocity
     excluded) learns attack rows: an artifact of the allow-all replay.
  3. **(iv) benign-tail-saturated**: the targeted score is ≥ 0.99, but the clean threshold is
     ≥ 0.999.
  4. A per-family raw-data test (α = 0.5 %):
     - **(iii)** the attacked value is not beyond the persona's own benign p99.5;
     - otherwise **(iv)**: detector limit, or novelty ramp below 200 observations.

The table uses the labels as they stood at Task 1.14. Cells are seed 0 / seed 1.

| family (n) | hit | (i) label semantics | (ii) learns attacks: calibrator + detector state | (iii) not anomalous | (iv) detector / design limit |
|---|---|---|---|---|---|
| magnitude_spike 10-100 (13 / 8) | 7 / 5 | – | 2+1 / 1+0 | 1 / 1 | 2 / 1 |
| scope_blowup (13 / 24) | 13 / 18 | – | 0 / 6+0 | – | – |
| exfil_volume (19 / 29) | 11 / 17 | 4 / 2 | 1+3 / 0+9 | – | 0 / 1 |
| prereq_skip (15 / 18) | 14 / 18 | – | 1 / 0 | – | – |
| novel_target (49 / 36) | 19 / 15 | – | 18+6 / 12+7 | – | 6 / 2 (novelty ramp n < 200) |
| velocity_burst (50 / 52) | 35 / 33 | – | 8 / 16 | 0 / 1 | 7 / 2 (1 is a benign-saturated agent) |

**(ii) dominates: 91 of 121 misses.** The rest are (i) 6, (iii) 3 and (iv) 21. (ii) has two
mechanisms.

- **Calibrator learns attacks.** The replay allows every call (§5.2), so each agent's P² tail
  learns its attack rows. On some agents these are 0.7–13 % of rows (decision 006): velocity
  duplicates, and scope or novelty hits near 1.0. As a result, 17 / 16 of 20 agents end with
  θ ≥ 0.999 (`gate` suite, "default" trajectory). Only fused scores at about 1.0 are flagged.
  - *(Manual inspection.)* Novelty scores for missed novel targets are 0.990–0.9998, against
    thresholds of 0.9991–0.99999.
- **Detector state learns attacks.**
  - *(Manual inspection.)* helpdesk-triage-a's `__resp__.item_count` key for
    `zendesk.list_tickets` holds only 62 observations at the time of attack 00052. A session's
    last response is never learned, and this read is usually last. Exfil attack sessions append
    an outbound write after the read, so their inflated counts make up a large part of those 62.
    The IQR is 119 against a median of 9, so a 64× read scores 0.17.

**(i): a label-semantics choice, changed (TDD).** This is not a spec bug. §4.6 labels mutated
envelopes, and §2.3 / D-004 score the response `item_count` on the *next* call of the session.
`exfil_volume` labelled the read and the first outbound write. When another call sits between them
(7 of 19 gate attacks on seed 0, 5 of 29 on seed 1; `classify.py` carrier count), that call
carries the only scoreable evidence, yet it was labelled benign.

- Following §2.3, the generator now also labels the call right after the read.
- Test: `test_exfil_volume_labels_the_call_that_carries_the_response`.
- With the new labels (evidence §1, `sweep-seed0` / `sweep-seed1`):
  - seed 0: all 4 former (i) misses become (ii);
  - seed 1: of its 2 former (i) misses, 1 becomes a hit and 1 becomes (ii). Seed 1's single
    exfil (iv), data-export-a-00004, predates the change.
- Under "oracle", exfil recall goes 0.895 → 1.000 on seed 0.
- Goldens and the CI baseline are unchanged. The CI dataset has no `exfil_volume`, and
  `test_golden_engine` and `ci_regress` pass unmodified.

**(iii) and (iv) under the spec protocol** (`classify.py` listing, evidence §1):

- magnitude_spike:
  - (iii): ×22 on sales-ops-a `amount` is still exceeded by 1.7 % of its benign amounts;
    ×21 on lead-router-a by 0.8 %.
  - (iv):
    - ×12 on exec-assistant-a `due_in_days` (z = 3.13);
    - ×24 on sales-ops-b `amount` (z = 3.23);
    - ×10 on sales-ops-a `due_in_days`, seed 1 (z = 2.68).
  - The first two are above z0 = 3, but the squash maps them to only 0.55–0.58, below thresholds
    of 0.84–0.998. The third is below z0 and scores 0.40.
  - With n = 13 / 8, ≥ 0.90 allows only 1 / 0 misses.
- velocity_burst:
  - (iv): helpdesk-triage-b bursts with max windows of 21–41 calls score 0.60–0.89, against
    θ ≈ 0.90–0.92. *(Manual inspection:* the persona's window median is about 3 calls, so a
    burst needs about 46 calls to score 0.90.*)*
  - slack-notifier-a: 11 calls; 0.3 % of benign windows reach it, below α, so (iv).
  - (iii): 13 calls on helpdesk-triage-b, seed 1; 1 % of benign windows reach it.
- novel_target (iv): the ramp `min(1, n / 200)` discounts stable vocabularies with 81–178
  observations and a benign new-value rate of 0. They score 0.36–0.84.

## 2. The candidates (all opt-in; defaults unchanged) (superseded: see User decision)

| id | what | code |
|---|---|---|
| enforce | outcome BLOCKED when the engine decides BLOCK, else ALLOWED (no reviewer); nothing learns a blocked call | `replay(..., outcomes="enforce")` |
| drop | the calibrator skips calls the engine itself blocked; detectors learn all | `EngineConfig.cal_censor="drop"` |
| winsorize | the calibrator learns θ instead of the fused score for such calls, including BLOCKED ones with fused ≥ θ | `cal_censor="winsorize"`, optionally `cal_censor_min_n` (shadow first) |
| oracle-reviewer | oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic). §6.8 with each flagged call (BLOCK or REQUIRE_APPROVAL) APPROVED if benign and DENIED if an attack; unflagged calls ALLOWED | `replay(..., outcomes="review")` |
| vel attempts | refused calls still count in velocity's current window, never in its history | `vel_count_attempts=True` |
| graded prereq | `s_prereq` = the mined rate of the missing prerequisite (≈ 0.90–0.98) instead of 1 | `seq_prereq_graded=True` |
| nov50 | novelty ramp over 50 observations instead of 200 | `novelty_full_n=50` (existing constant) |
| oracle | nothing learns attack-labelled calls (reads every label) | `outcomes="oracle"`, diagnostic only |

Results on warm test rows at live thresholds. Cells are seed 0 / seed 1. Full tables, including
offcal and per-family top-1 share, are in evidence §2.

| candidate | min recall | mag 10-100 | scope | exfil | prereq | novel | velocity | benign FPR | control FPR | AUPRC | FPR95 per-agent (recall) | θ median end; agents ratcheting / saturated |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| default (spec) | 0.388 / 0.417 | 0.538 / 0.625 | 1.0 / 0.75 | 0.579 / 0.621 | 0.933 / 1.0 | 0.388 / 0.417 | 0.700 / 0.635 | 0.39 % / 0.56 % | 6.2 % / 4.9 % | 0.576 / 0.578 | 0.563 / 0.634 (0.95) | 1.000; 0 / 17 · 1.000; 0 / 16 |
| enforce | 0.923 / 0.875 | 0.923 / 0.875 | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 0.972 | 1.0 / 1.0 | **36.7 % / 52.7 %** | 48 % / 59 % | 0.210 / 0.219 | 0.523 / 0.612 | **0.431** (min 0.08); 4 / 0 · **0.331** (min 0.05); 4 / 0 |
| drop | 0.923 / 0.875 | 0.923 / 0.875 | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 0.939 / 0.972 | 1.0 / 0.981 | **37.6 % / 31.9 %** | 44 % / 51 % | 0.576 / 0.578 | 0.563 / 0.634 | **0.413** (min 0.07); 4 / 0 · 0.429; 3 / 0 |
| winsorize | 0.846 / 0.690 | 0.846 / 0.875 | 1.0 / 1.0 | 0.895 / 0.690 | 1.0 / 1.0 | 0.898 / 0.972 | 0.980 / 0.981 | **13.5 % / 13.5 %** | 37 % / 31 % | 0.576 / 0.578 | 0.563 / 0.634 | **0.600 (= θ0)**; 0 / 0 · 0.599; 0 / 0 |
| winsorize after 500 | 0.692 / 0.690 | 0.692 / 0.750 | 1.0 / 0.958 | 0.737 / 0.690 | 1.0 / 1.0 | 0.776 / 0.722 | 0.920 / 0.962 | 5.2 % / 5.3 % | 24 % / 9.3 % | 0.576 / 0.578 | 0.563 / 0.634 | 0.913; 1 / 6 · 0.877; 0 / 5 |
| enforce, winsorize after 500 | 0.923 / 0.875 | 0.923 / 0.875 | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 0.980 / 0.972 | 1.0 / 1.0 | **31.1 % / 31.2 %** | 48 % / 43 % | 0.223 / 0.243 | 0.520 / 0.599 | 0.502; 1 / 0 · 0.499; 0 / 0 |
| oracle-reviewer | 0.680 / 0.750 | 0.692 / 0.750 | 0.923 / 1.0 | 1.0 / 0.966 | 1.0 / 1.0 | 0.837 / 0.944 | 0.680 / 0.750 | 1.25 % / 1.20 % | 13.2 % / 8.7 % | 0.506 / 0.579 | 0.475 / 0.625 | 0.914; 0 / 5 · 0.901; 0 / 6 |
| oracle-reviewer + vel attempts | 0.692 / 0.750 | 0.692 / 0.750 | 0.923 / 1.0 | 1.0 / 0.966 | 1.0 / 1.0 | 0.816 / 0.944 | 0.880 / 0.962 | 1.24 % / 1.19 % | 13.2 % / 8.7 % | 0.590 / 0.638 | 0.475 / 0.624 | 0.914; 0 / 5 · 0.901; 0 / 6 |
| **oracle-reviewer + vel attempts + graded prereq + nov50** | 0.692 / 0.750 | 0.692 / 0.750 | 1.0 / 1.0 | 1.0 / 0.966 | 1.0 / 1.0 | **1.0 / 1.0** | 0.880 / 0.962 | **0.93 % / 0.89 %** | 13.8 % / 8.7 % | **0.661 / 0.686** | 0.481 / 0.625 | 0.908; 1 / 4 · 0.901; 0 / 2 |
| oracle (diagnostic) | 0.769 / 0.750 | 0.769 / 0.750 | 0.923 / 1.0 | 1.0 / 0.966 | 1.0 / 1.0 | 0.878 / 0.944 | 0.780 / 0.885 | 1.19 % / 1.18 % | 13.5 % / 8.2 % | 0.537 / 0.626 | 0.543 / 0.738 | 0.914; 0 / 5 · 0.896; 0 / 6 |

### The censoring ratchet (measured)

Label-free censoring cannot learn the benign tail above θ, because θ starts at θ0 = 0.6, which is
below every agent's benign q_0.995 (≈ 0.79–1.0 under clean learning).

- **enforce and drop truncate the calibrator's input at θ.** A truncated `q_{1−α}` always falls
  below θ, so θ ratchets down while the calibrator is still blending. Median θ ends at
  0.33–0.44, the minimum agent reaches 0.05–0.08, and 3–4 agents per seed are still falling
  during the test period. Benign FPR is 32–53 %.
- **winsorize** keeps the tail mass but pins it at the threshold that blocked it. The cold-start
  blocks at θ0 become a permanent point mass above α. Median θ freezes at 0.600 = θ0 (agent
  minimum 0.49 / 0.56), and no agent saturates. Benign FPR is 13.5 %.
- **Shadow-then-censor** (`cal_censor_min_n=500`, allow-all) starts from the polluted
  allow-all θ. Benign FPR is 5.2 %, min recall is 0.69, and 5–6 agents stay saturated. Adding
  enforce sends it back down to 0.50.
- **The oracle-reviewer loop does not ratchet.** Benign flagged calls are approved and learned,
  which is exactly §6.8 with a perfect reviewer. θ converges to the oracle's level: median 0.90–0.91, with a drift of +0.001 over the
  test period.

### Attribution (`top_reason_share` ≥ 0.90 in the brief)

These shares are the top-1 share for the recommended candidate. Under the default replay they
are also low: velocity 0.26 / 0.15.

- **velocity_burst: 0.30 / 0.22.** Sequence fires on the duplicated write: the lookup scrolls
  out of the k = 4 `prior_tools` window, so prereq scores 0.9–1.0, and so does trans.
- **exfil_volume: 0.58 / 0.64.** Sequence prereq fires on the outbound write.
- **scope_blowup: 0.85 / 1.0** and **magnitude_spike: 0.89 / 1.0**.

The targeted detector is nevertheless **among the reasons** (score > `reason_min`) of a flagged
row in 89–100 % of detected attacks, for every family and both seeds (evidence §3).

## 3. Phase-A recommendation (superseded by the Decision below)

Phase A recommended the following:

- the oracle-reviewer loop as the gate protocol;
- `vel_count_attempts`, `seq_prereq_graded` and `novelty_full_n=50` as defaults.

The controller kept the spec protocol and the spec defaults instead (GATE3 / GATE4). The
measurements stay here as the documented effect of those opt-ins.

**Opt-in effects** (seed 0 / seed 1, all under the oracle-reviewer loop; table §2):

- **`vel_count_attempts`**: velocity_burst 0.680 → 0.880 / 0.750 → 0.962.
  - Without it, a denied duplicate never enters velocity's window, so the count stalls once the
    burst is flagged at θ − δ.
  - It is a no-op under allow-all, where nothing is refused.
  - It departs from "BLOCKED never moves state" only in the current window count. The baseline
    history never sees refused calls.
- **`seq_prereq_graded`**:
  - Benign FPR falls 1.24 → 0.93 % / 1.19 → 0.89 %, measured together with nov50.
  - Saturated agents fall 5 → 3 / 6 → 2.
  - AUPRC rises 0.590 → 0.658 / 0.638 → 0.686.
  - No gate recall is lost.
- **`novelty_full_n=50`**: novel_target 0.878 → 1.000 / 0.972 → 1.000, with no measurable rise
  in benign or control FPR.

**Residual misses of the full candidate** (oracle-reviewer + all three switches;
`classify.py` run on that candidate's own replays, evidence §1 "rec-seed0" / "rec-seed1").

seed 0:

- magnitude_spike, 4 misses:
  - (ii) state-learns-attacks: support-refunds-c-00067;
  - (iii): sales-ops-a ×22;
  - (iv): exec-assistant-a ×12 `due_in_days` (z 3.13, s 0.545 vs θ 0.869) and sales-ops-b ×24
    `amount` (z 3.23, s 0.575 vs θ 0.999).
- velocity_burst, 6 misses, all (iv): five helpdesk-triage-b bursts and slack-notifier-a's 11.

seed 1:

- magnitude_spike, 2 misses:
  - (iii): lead-router-a ×21;
  - (iv): sales-ops-a ×10 `due_in_days` (z 2.68).
- velocity_burst: 1 (iii) and 1 (iv).
- exfil_volume, 1 (iv): data-export-a-00004. Its item_count of 134 is above every benign value,
  yet its response score is 0.774 against θ 0.886.

Everything else is a hit. Under the full candidate the gate still fails:

| seed | fails |
|---|---|
| 0 | magnitude_spike 0.692, velocity_burst 0.880 |
| 1 | magnitude_spike 0.750 |
| both | top-1 share: velocity_burst 0.30 / 0.22, exfil_volume 0.58 / 0.64 |

The targeted detector is among a flagged row's reasons in 0.89–1.00 of detected attacks
(evidence §3).

**Open options.** These are not rulings; they are for the user:

- magnitude_spike, one of:
  - a generator spike defined relative to the field's dispersion;
  - a magnitude scoring change;
  - accepting the spec-design bound. With n = 8–13 attacks, ±1 attack is 8–12 pp.
- velocity: a velocity-specific z0 / τ, or a raw-count (Poisson) tail score.
- attribution: gate on "targeted detector among the reasons" instead of top-1.
- control FPR: it rises 6.2 → 13.8 % / 4.9 → 8.7 % once thresholds stop saturating. Tier 3
  needs ≤ 5 %.

## Decision (controller rulings, verbatim)

- GATE3: the default benchmark protocol stays spec §5.2 allow-all. outcomes="review" stays opt-in
  and is ALWAYS reported as "oracle-reviewer approval loop (labels act as a perfect human reviewer
  → optimistic)".
- GATE4: engine defaults stay spec-faithful (novelty_full_n=200, binary s_prereq; prereq_rate 0.90
  per decision 003). vel_count_attempts / seq_prereq_graded / novelty_full_n=50 remain opt-in
  candidates for the user to decide; document their measured effect.
- GATE5: write tests/acceptance/test_tier1_exit.py (@pytest.mark.slow) encoding the brief's gate
  VERBATIM on the SW1 dataset (seed 0, 20 personas, 90 days, all 8 families; generate it inside a
  session fixture into tmp or reuse traces/ if present via an env var — must work from a clean
  checkout). The per-family recall ≥0.90 and top-reason-share ≥0.90 assertions go in a test marked
  `@pytest.mark.xfail(strict=True, reason="Tier-1 gate not met under spec protocol; see
  docs/decisions/008-tier1-gate.md")`. The remaining exit items are REAL assertions in separate
  tests: benign FPR guard ≤0.02 at "row" thresholds, ablation table present with ≥5 variants,
  cold-start curve with 4 buckets, p99 score latency < 200 µs (reuse/align with tests/perf). Also
  add a non-xfail test that records the current per-family numbers within broad tolerances so
  regressions are caught (e.g. min family recall ≥ the measured default value − 0.1).

### Consequences

- **The Tier-1 gate is not met under the spec protocol.**
  - `tests/acceptance/test_tier1_exit.py::test_tier1_exit` is a strict xfail.
  - The dataset is SW1 at 90 days; the brief's interface line said 60, and the ruling chose SW1.
  - Every other exit item is a real assertion.
  - The README states this and shows the default-protocol numbers next to the oracle-reviewer
    numbers, each clearly labelled.
- **The brief's verbatim test reads `tier1_report["extras"]`** for ablation, cold_start and drift.
  The report stores those three at its top level, so the separate real tests check them there.
- **What changed and what did not:**
  - Kept: the exfil carrier label, and every opt-in switch and replay mode, all still opt-in.
  - Unchanged: the spec protocol and the engine defaults, so the engine golden and the CI
    baseline are unchanged. (Superseded: see User decision. The pre-decision engine golden lives
    on as `tests/golden/engine-spec-allow.sha256`.)

## Spec deviations

| item | status | deviates from |
|---|---|---|
| exfil carrier label | **in effect** (generator) | §4.6 labels mutated envelopes; this follows §2.3 / D-004 |
| `outcomes="review"` (oracle-reviewer loop) | opt-in → **default benchmark protocol** by user decision (2026-11-10); `outcomes="allow"` stays selectable as the labelled comparison | §5.2 allow-all |
| `vel_count_attempts` | opt-in → **default** by user decision (2026-11-10) | "blocked calls never move state" (window count only, not the baseline) |
| `seq_prereq_graded` | opt-in → **default** by user decision (2026-11-10) | §6.6(b) s_prereq = 1 |
| `novelty_full_n = 50` | opt-in (sweep value) → **default** by user decision (2026-11-10) | §6.5: 200 |
| `cal_censor`, `outcomes="enforce"` | opt-in, rejected by the evidence (ratchet / freeze) | §6.8 / §5.2 |

## User decision (2026-11-10)

The user's words: "turn on the reviewer protocol and three engine settings (novelty ramp at 50
observations, graded prerequisite score, counting refused calls toward velocity)". This adopts the
Phase-A recommendation (§3). It supersedes GATE3 and GATE4 for the defaults only; the rulings
above stay as they were made.

**What changed** (Task U1):

- **Benchmark protocol.** `agentfw.bench.run` (and its `--outcomes` flag) now defaults to
  `outcomes="review"`: the oracle-reviewer approval loop (labels act as a perfect human reviewer →
  optimistic). Every report header and leaderboard row carries that label. Spec §5.2 allow-all
  (`--outcomes allow`) stays selectable and is labelled "spec §5.2 allow-all".
  - `replay()` itself still defaults to `"allow"`. Every bench tool now states its mode: `run`,
    its ablation table and the baselines use the run's mode; `ci_regress`, the goldens and
    `scripts/update_golden.py` use `CI_OUTCOMES = "review"`; `replay_detector`, `run_ablation` and
    `sweep` take the mode as a required argument; the sweeps CLI has `--outcomes` (default
    `"review"`).
  - The evidence keeps its meaning. The `gate` sweep suite and both evidence scripts
    (`replay_variants.py`, `shares.py`) build their config as
    `EngineConfig().replace(**{**SPEC_SWITCHES, **params})` (`SPEC_SWITCHES` in
    `agentfw.bench.sweeps`), so "default" / `{}` is still the spec engine and the commands in
    `evidence/gate-seed0-seed1.md` reproduce it. Golden `engine-spec-allow` (the spec engine under
    allow-all on the CI dataset) still equals the pre-decision engine hash `c4042bc4…`.
- **Engine defaults** (`EngineConfig`): `vel_count_attempts=True`, `seq_prereq_graded=True`,
  `novelty_full_n=50`. The tests of spec behaviour pin the spec values (`False`, `False`, `200`).
- **The sandbox proxy is unaffected.** Its shadow detector is always updated with
  `Outcome.ALLOWED` (`agentfw/sandbox/proxy_py/proxy.py`), so it never refuses a call, and
  `vel_count_attempts` is a no-op there.

**Re-measured** on SW1 seed 0 (warm test rows, live per-agent thresholds, attack level, gate
bucket; `tests/acceptance/test_tier1_exit.py`). The review column matches the §2 row
"oracle-reviewer + vel attempts + graded prereq + nov50" to the third decimal.

| family | default protocol: oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic) | top-1 share | comparison: spec §5.2 allow-all, same engine defaults | spec engine, allow-all (§2 "default") |
|---|---|---|---|---|
| magnitude_spike 10-100 | **0.692** (9/13) | 0.889 | 0.538 (7/13) | 0.538 |
| scope_blowup | 1.000 (13/13) | 0.846 | 1.000 (13/13) | 1.000 |
| exfil_volume | 1.000 (19/19) | 0.579 | 0.474 (9/19) | 0.579 |
| prereq_skip | 1.000 (15/15) | 1.000 | 0.667 (10/15) | 0.933 |
| novel_target | 1.000 (49/49) | 1.000 | 0.469 (23/49) | 0.388 |
| velocity_burst | **0.880** (44/50) | 0.295 | 0.680 (34/50) | 0.700 |
| benign FPR (live thresholds) | 0.93 % | | 0.24 % | 0.39 % |
| control FPR (live thresholds) | 13.8 % | | 6.2 % | 6.2 % |
| AUPRC (warm test) | 0.661 | | 0.612 | 0.576 |
| FPR@95 per-agent (recall 0.950) | 0.481 | | 0.555 | 0.563 |

**The Tier-1 gate is still not met.** Under the default protocol, magnitude_spike (0.692) and
velocity_burst (0.880) are below 0.90, and the top-1 share is below 0.90 for velocity_burst
(0.30), exfil_volume (0.58), scope_blowup (0.85) and magnitude_spike (0.89). `test_tier1_exit`
stays a strict xfail. The regression floor (`test_tier1_no_regression`) now pins the
review-protocol numbers above, with the same −0.1 tolerance. The benign-FPR guard (≤ 2 %) passes
at 0.93 %.

Two side effects to know about:

- **Under allow-all, the new defaults lower prereq_skip (0.933 → 0.667) and exfil_volume
  (0.579 → 0.474).** The graded prereq score (≈ 0.90–0.98 instead of 1) no longer clears the
  saturated allow-all thresholds (≥ 0.999, §1). The comparison row is therefore worse than the
  spec engine on those two families; under the review protocol it is not.
- **CI baseline.** The CI dataset's test split holds a single attack row (a `novel_target` on
  outreach-a `mail.send_email`, fused 0.997788). With one attack, AUPRC = 1 / its rank. Under the
  review protocol a benign outreach-a `mail.send_email` row scores 0.997797, just above it (under
  allow-all the same row scored 0.995614: the review loop raised it), so the attack ranks second
  and the engine's CI AUPRC is 0.5 instead of 1.0. The cause is the protocol, not the switches:
  the spec engine under review gives 0.5 too, and the new engine under allow-all gives 1.0.
  FPR@95 is unchanged (0.0084). The baseline was rewritten with `ci_regress --write-baseline`.

The SW1 leaderboard, README and writeup are not rebuilt yet. They will be rebuilt once, after the
pending engine fix in `schema_infer`, which also changes engine output.

## Amendment A1 (2026-11-11)

Spec: `docs/spec.md`, section "Amendment A1 (2026-11-11)" (A1.0 to A1.13). Implemented in plan
Tasks 1.16 to 1.26. Evidence: `evidence/a1-gate/matrix.json` and `evidence/a1-gate/matrix.md`,
written by `evidence/a1-gate/run.sh` (Task 1.23). Every gate number in this section is a cell of
`matrix.json`. Gate cells are attack-level recall on warm test rows at live per-agent thresholds,
`magnitude_spike` in bucket `10-100`, `slow_drift` excluded.

Protocols, verbatim (spec A1.5):

- spec §5.2 allow-all
- oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic)

**This section supersedes the one-protocol gate statements above it.** They stay as the record of
what was decided at the time:

- Decision, GATE5: the single strict xfail with reason "Tier-1 gate not met under spec protocol".
  `test_tier1_exit` is now parametrized by protocol. `test_tier1_exit[review]` carries no mark and
  passes. `test_tier1_exit[allow]` is a strict xfail and the only strict xfail left in
  `tests/acceptance/test_tier1_exit.py`; its reason names both protocols.
- Consequences, first bullet ("The Tier-1 gate is not met under the spec protocol"; the strict
  xfail; the README showing default-protocol numbers next to oracle-reviewer numbers). The README,
  the writeup and `docs/leaderboard.md` now show the dual-protocol table below.
- User decision (2026-11-10), the paragraph that starts "The Tier-1 gate is still not met". The
  gate criterion is now the amended one (spec A1.7). `test_tier1_no_regression` pins the seed-0
  rows of `matrix.json` under each protocol (spec §5.2 allow-all; oracle-reviewer approval loop
  (labels act as a perfect human reviewer → optimistic)), with the same −0.1 tolerance. The SW1
  leaderboard (Task 1.25), the README and the writeup (Task 1.26) are rebuilt.

### What was adopted, and from which design

Each rule was prototyped and measured before adoption (spec A1.11). The adopted design is the
reviewers' combined prototype tree (velocity A1.1, magnitude A1.2, far rule A1.4) with the
rate-based top-up, the consumed attack id and A1.3 added. The prototype's other variants (`anscombe`
and `budget` velocity modes, a negative-binomial count tail, a winsorizing fence) were not adopted
and are not in the code.

| item | spec | source | default now | before A1 (still selectable) |
|---|---|---|---|---|
| Velocity count signal: Poisson upper-tail deviate at the key's median window count, divided by the over-dispersion measured on the Anscombe scale | A1.1 | §3 "Open options": a raw-count (Poisson) tail score; the user's amendment of 2026-11-10 | `vel_count_stat = "poisson"` | `"logz"` |
| Magnitude score: tail probability on a two-piece robust log scale, read as a posterior with `tail_prior = 0.005` (= `cal_alpha`, D-023) | A1.2 | §3 "Open options": a magnitude scoring change; the user's amendment | `mag_score = "tail"` | `"z"` |
| Magnitude learns and scores numbers in BOOL-typed fields (D-024) | A1.3 | final-review ruling FR5 (not one of the user's items) | `mag_bool_numeric = True` | `False` |
| Benchmark spikes land at or beyond 5 × the persona's benign p99 (D-025), plus a magnitude-only top-up pass (D-026) | A1.4 | §3 "Open options": a generator spike defined relative to the field's dispersion; the user's amendment | `sw1_config(seed)`: `magnitude_mode = "far"`, `magnitude_far_mult = 5.0`, `magnitude_topup_rate = 0.015`; generator defaults stay `"own"` / `0.0` | `"own"`, top-up 0 |
| Gate evaluated and shown under both protocols (D-028) | A1.5 | the user's amendment | always both | one protocol |
| Attribution: targeted detector among the reasons of a flagged row (D-027); the top-1 share is still printed as "original spec §9 criterion" | A1.6, A1.7 | §3 "Open options": gate on "targeted detector among the reasons"; the user's decision of 2026-11-10 on attribution | among-the-reasons share ≥ 0.90 | top-1 share ≥ 0.90 |

`tail_prior`, `magnitude_far_mult` and `magnitude_topup_rate` are fixed by the spec. They were
not swept or changed against any seed's recall, and nothing was retuned after the measurement
below. Spec A1.4 discloses how the top-up rate was chosen: 0.015 was set once, in the draft
phase, when prototypes with a fixed count of 120 and a rate of 0.01 had already been replayed and
their recalls were known; no other rate was generated, and nothing was retuned after the
measurement. The generator defaults did not change, so the CI, tiny, validation and decision-008
datasets are byte-identical to before.

### Tier-1 gate under both protocols, SW1 seeds 0, 1 and 2

Copied from `matrix.md`. Amended engine (`EngineConfig()`), datasets `sw1_config(seed)`: seed 0
`a455b10c705d`, seed 1 `e783c301bf21`, seed 2 `62645fcfd0b8` (traces sha). Seed 2 was generated
and replayed only after seeds 0 and 1 were written to `matrix.json`. Cells: recall (n/N);
among-the-reasons share; top-1 share (original spec §9 criterion, reported and not gated).
Amended criterion (spec A1.7): per family recall ≥ 0.90 and among-the-reasons share ≥ 0.90,
benign FPR ≤ 0.02. Each row is the result under one protocol. The two rows of a seed are read
together, and neither alone is the Tier-1 gate result.

| seed | Protocol | magnitude_spike 10-100 | scope_blowup | exfil_volume | prereq_skip | novel_target | velocity_burst | benign FPR | control FPR | amended criterion |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic) | 1.000 (81/81); 1.00; 1.00 | 1.000 (13/13); 1.00; 0.85 | 1.000 (19/19); 1.00; 0.84 | 1.000 (15/15); 1.00; 1.00 | 1.000 (49/49); 1.00; 1.00 | 1.000 (50/50); 1.00; 0.40 | 0.0058 | 0.035 | PASS |
| 0 | spec §5.2 allow-all | 0.938 (76/81); 1.00; 1.00 | 1.000 (13/13); 1.00; 0.85 | 0.579 (11/19); 1.00; 1.00 | 0.533 (8/15); 1.00; 1.00 | 0.163 (8/49); 1.00; 1.00 | 0.800 (40/50); 1.00; 0.40 | 0.0014 | 0.007 | FAIL: recall:exfil_volume, recall:prereq_skip, recall:novel_target, recall:velocity_burst |
| 1 | oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic) | 1.000 (85/85); 1.00; 1.00 | 1.000 (24/24); 1.00; 1.00 | 1.000 (29/29); 1.00; 0.79 | 1.000 (18/18); 1.00; 1.00 | 1.000 (36/36); 1.00; 1.00 | 0.981 (51/52); 1.00; 0.24 | 0.0062 | 0.028 | PASS |
| 1 | spec §5.2 allow-all | 0.824 (70/85); 1.00; 0.97 | 1.000 (24/24); 1.00; 1.00 | 0.655 (19/29); 1.00; 1.00 | 0.667 (12/18); 1.00; 1.00 | 0.222 (8/36); 1.00; 1.00 | 0.808 (42/52); 1.00; 0.31 | 0.0023 | 0.013 | FAIL: recall:magnitude_spike, recall:exfil_volume, recall:prereq_skip, recall:novel_target, recall:velocity_burst |
| 2 | oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic) | 1.000 (85/85); 1.00; 0.99 | 1.000 (13/13); 1.00; 1.00 | 1.000 (38/38); 0.95; 0.82 | 1.000 (13/13); 1.00; 1.00 | 1.000 (38/38); 1.00; 1.00 | 1.000 (44/44); 1.00; 0.43 | 0.0050 | 0.050 | PASS |
| 2 | spec §5.2 allow-all | 0.965 (82/85); 1.00; 0.93 | 1.000 (13/13); 1.00; 1.00 | 0.711 (27/38); 1.00; 0.96 | 0.846 (11/13); 1.00; 1.00 | 0.211 (8/38); 1.00; 1.00 | 0.682 (30/44); 1.00; 0.37 | 0.0018 | 0.031 | FAIL: recall:exfil_volume, recall:prereq_skip, recall:novel_target, recall:velocity_burst |

On all three seeds the amended criterion passes under the oracle-reviewer approval loop (labels
act as a perfect human reviewer → optimistic) and fails under spec §5.2 allow-all. That is a pass
under one protocol next to a fail under the other. Which protocol's result closes Tier 1 is the
user's open decision (D-028; spec A1.13 item 1). The original spec §9 criterion (top-1 share
≥ 0.90) fails under both protocols on every seed (velocity_burst 0.24 to 0.43). The Tier-2 exit is
not reached: the live sandbox dataset v1 has not been built.

The composition of the `magnitude_spike` cell (spec A1.4; `matrix.json` `census`, the same
rendering as under the leaderboard's gate tables), one line per seed:

- Seed 0. Census of the magnitude_spike gate cell (spec A1.4): n 81 (warm test rows, bucket 10-100) by persona: support-refunds-c 22, lead-router-a 17, support-refunds-a 11, sales-ops-a 8, support-refunds-b 8, billing-disputes-a 4, sales-ops-b 4, crm-hygiene-a 3, exec-assistant-a 2, payouts-ops-a 1, renewals-a 1; over all 264 spikes of the dataset, median strength 48.3 and share with strength >= 10 0.970.
- Seed 1. Census of the magnitude_spike gate cell (spec A1.4): n 85 (warm test rows, bucket 10-100) by persona: lead-router-a 23, support-refunds-c 15, sales-ops-a 9, sales-ops-b 9, crm-hygiene-a 8, support-refunds-a 8, payouts-ops-a 5, support-refunds-b 4, exec-assistant-a 2, billing-disputes-a 1, renewals-a 1; over all 252 spikes of the dataset, median strength 50.5 and share with strength >= 10 0.988.
- Seed 2. Census of the magnitude_spike gate cell (spec A1.4): n 85 (warm test rows, bucket 10-100) by persona: support-refunds-c 20, support-refunds-a 16, lead-router-a 10, sales-ops-a 9, support-refunds-b 9, sales-ops-b 8, payouts-ops-a 4, billing-disputes-a 3, crm-hygiene-a 3, renewals-a 3; over all 262 spikes of the dataset, median strength 51.5 and share with strength >= 10 0.981.

### Companion rows (spec A1.5 rule 4)

The `magnitude_spike` cell above is shown with its two companions: the engine before Amendment A1
(`EngineConfig().replace(**PRE_A1_SWITCHES)`) on the same datasets, and both engines on the
sensitivity dataset (`sw1_config(0)` with `magnitude_far_mult = 1.0`, traces `3fc612ce1fdb`;
seed 0 only). The sensitivity rows are companion rows, not gate results.

| dataset | engine | Protocol | magnitude_spike 10-100 | scope_blowup | exfil_volume | prereq_skip | novel_target | velocity_burst | benign FPR | control FPR | amended criterion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SW1 seed 0 (`a455b10c705d`) | pre-A1 | oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic) | 0.975 (79/81); 1.00; 0.99 | 1.000 (13/13); 1.00; 0.85 | 1.000 (19/19); 0.89; 0.58 | 1.000 (15/15); 1.00; 1.00 | 1.000 (49/49); 1.00; 1.00 | 0.880 (44/50); 1.00; 0.30 | 0.0092 | 0.134 | FAIL: among:exfil_volume, recall:velocity_burst |
| SW1 seed 0 (`a455b10c705d`) | pre-A1 | spec §5.2 allow-all | 0.753 (61/81); 1.00; 1.00 | 1.000 (13/13); 1.00; 0.85 | 0.474 (9/19); 1.00; 1.00 | 0.667 (10/15); 1.00; 1.00 | 0.469 (23/49); 1.00; 1.00 | 0.680 (34/50); 1.00; 0.29 | 0.0025 | 0.060 | FAIL: recall:magnitude_spike, recall:exfil_volume, recall:prereq_skip, recall:novel_target, recall:velocity_burst |
| SW1 seed 1 (`e783c301bf21`) | pre-A1 | oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic) | 0.988 (84/85); 1.00; 0.98 | 1.000 (24/24); 1.00; 1.00 | 0.966 (28/29); 1.00; 0.64 | 1.000 (18/18); 1.00; 1.00 | 1.000 (36/36); 1.00; 1.00 | 0.962 (50/52); 1.00; 0.22 | 0.0087 | 0.087 | PASS |
| SW1 seed 1 (`e783c301bf21`) | pre-A1 | spec §5.2 allow-all | 0.788 (67/85); 1.00; 0.94 | 0.792 (19/24); 1.00; 1.00 | 0.586 (17/29); 1.00; 1.00 | 0.667 (12/18); 1.00; 1.00 | 0.444 (16/36); 1.00; 1.00 | 0.635 (33/52); 1.00; 0.15 | 0.0030 | 0.049 | FAIL: recall:magnitude_spike, recall:scope_blowup, recall:exfil_volume, recall:prereq_skip, recall:novel_target, recall:velocity_burst |
| SW1 seed 2 (`62645fcfd0b8`) | pre-A1 | oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic) | 0.953 (81/85); 1.00; 0.98 | 1.000 (13/13); 1.00; 1.00 | 0.895 (34/38); 0.88; 0.65 | 1.000 (13/13); 1.00; 1.00 | 1.000 (38/38); 1.00; 1.00 | 0.864 (38/44); 1.00; 0.29 | 0.0089 | 0.108 | FAIL: recall:exfil_volume, among:exfil_volume, recall:velocity_burst |
| SW1 seed 2 (`62645fcfd0b8`) | pre-A1 | spec §5.2 allow-all | 0.800 (68/85); 1.00; 0.97 | 0.615 (8/13); 1.00; 1.00 | 0.500 (19/38); 0.95; 0.95 | 0.846 (11/13); 1.00; 1.00 | 0.474 (18/38); 1.00; 1.00 | 0.591 (26/44); 1.00; 0.27 | 0.0025 | 0.062 | FAIL: recall:magnitude_spike, recall:scope_blowup, recall:exfil_volume, recall:prereq_skip, recall:novel_target, recall:velocity_burst |
| sensitivity seed 0 (`3fc612ce1fdb`) | amended (A1) | oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic) | 0.797 (51/64); 1.00; 1.00 | 1.000 (13/13); 1.00; 0.85 | 1.000 (19/19); 1.00; 0.84 | 1.000 (15/15); 1.00; 1.00 | 1.000 (49/49); 1.00; 1.00 | 1.000 (50/50); 1.00; 0.40 | 0.0060 | 0.035 | companion row (spec A1.5 rule 4), not a gate result |
| sensitivity seed 0 (`3fc612ce1fdb`) | amended (A1) | spec §5.2 allow-all | 0.656 (42/64); 1.00; 1.00 | 1.000 (13/13); 1.00; 0.85 | 0.632 (12/19); 1.00; 1.00 | 0.533 (8/15); 1.00; 1.00 | 0.163 (8/49); 1.00; 1.00 | 0.800 (40/50); 1.00; 0.38 | 0.0014 | 0.012 | companion row (spec A1.5 rule 4), not a gate result |
| sensitivity seed 0 (`3fc612ce1fdb`) | pre-A1 | oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic) | 0.688 (44/64); 1.00; 0.96 | 1.000 (13/13); 1.00; 0.85 | 1.000 (19/19); 0.84; 0.58 | 1.000 (15/15); 1.00; 1.00 | 1.000 (49/49); 1.00; 1.00 | 0.880 (44/50); 1.00; 0.30 | 0.0090 | 0.134 | companion row (spec A1.5 rule 4), not a gate result |
| sensitivity seed 0 (`3fc612ce1fdb`) | pre-A1 | spec §5.2 allow-all | 0.578 (37/64); 1.00; 1.00 | 1.000 (13/13); 1.00; 0.85 | 0.474 (9/19); 1.00; 1.00 | 0.667 (10/15); 1.00; 1.00 | 0.469 (23/49); 1.00; 1.00 | 0.680 (34/50); 1.00; 0.26 | 0.0024 | 0.062 | companion row (spec A1.5 rule 4), not a gate result |

What the companion rows show:

- **On seed 1 the pre-A1 engine also passes the amended criterion under the oracle-reviewer
  approval loop (labels act as a perfect human reviewer → optimistic).** On that seed the corrected
  data and the among-the-reasons criterion are enough; the scoring change is not needed. Under
  spec §5.2 allow-all the pre-A1 engine fails on seed 1 in six families.
- On seeds 0 and 2 the pre-A1 engine fails under the oracle-reviewer approval loop (labels act as
  a perfect human reviewer → optimistic): seed 0 on velocity_burst 0.880 (44/50) and the
  exfil_volume among-the-reasons share 0.89; seed 2 on exfil_volume 0.895 (34/38), its share 0.88,
  and velocity_burst 0.864 (38/44). These are the cells the scoring change closes.
- The pre-A1 engine already catches most far spikes under the oracle-reviewer approval loop
  (labels act as a perfect human reviewer → optimistic): 0.975 (79/81), 0.988 (84/85),
  0.953 (81/85). The magnitude cell under that protocol is decided by the generator correction
  more than by the scoring change (spec A1.12 item 4).
- On the sensitivity dataset neither engine reaches 0.90 in the magnitude cell under either
  protocol: amended 0.797 (51/64) under the oracle-reviewer approval loop (labels act as a perfect
  human reviewer → optimistic) and 0.656 (42/64) under spec §5.2 allow-all; pre-A1 0.688 (44/64)
  and 0.578 (37/64) under the same two protocols.
- The control FPR drop under the oracle-reviewer approval loop (labels act as a perfect human
  reviewer → optimistic) (pre-A1 0.134 / 0.087 / 0.108, amended 0.035 / 0.028 / 0.050 on
  seeds 0 / 1 / 2) comes from the velocity change: flat keys no longer over-score windows of 3 to
  6 calls (spec A1.11).

### Old → new record

| item | before Amendment A1 | after Amendment A1 |
|---|---|---|
| SW1 seed 0 dataset (traces sha) | `e7b3bc2a2f20` (labels `bd94a3197d82`); this legacy dataset is still what the default generator config produces | `a455b10c705d` (`sw1_config(0)`) |
| SW1 seed 0 sessions / envelopes | 42,289 / 174,878 | 42,289 / 174,878 (a spike changes an existing call) |
| SW1 seed 0 attacks | 599 | 808 |
| SW1 seed 1 attacks | 611 | 813 |
| SW1 seed 0 magnitude spikes | 55 (spec A1.11) | 264: 53 first pass, 2 dropped, 211 top-up |
| median spike strength, seed 0 | about 14 (spec A1.4) | 48.3 |
| `magnitude_spike` gate cell (warm test, bucket 10-100), seeds 0 / 1 | 13 / 8 | 81 / 85 (seed 2: 85) |
| engine golden `tests/golden/engine.sha256` | `f6573c610106` | `2223582ecd50` (Task 1.21) |
| golden `engine-spec-allow` (spec engine, allow-all) | `c4042bc466f2` | `c4042bc466f2`, unchanged by rule |
| golden `static_threshold` | `dc9f6b5927c9` | `dc9f6b5927c9`, unchanged |
| CI baseline (`agentfw/bench/baseline_metrics.json`) | engine AUPRC 0.5000, FPR@95 0.008403 | unchanged: the CI dataset is byte-identical and `ci_regress --write-baseline` gave the same numbers |
| engine defaults `vel_count_stat` / `mag_score` / `mag_bool_numeric` | `"logz"` / `"z"` / `False` | `"poisson"` / `"tail"` / `True` |
| gate criterion | recall ≥ 0.90, top-1 share ≥ 0.90, benign FPR ≤ 2 % | recall ≥ 0.90, among-the-reasons share ≥ 0.90, benign FPR ≤ 2 %; top-1 share printed as "original spec §9 criterion" |
| acceptance test | `test_tier1_exit`, one strict xfail | `test_tier1_exit[review]` passes; `test_tier1_exit[allow]` is a strict xfail |

AUPRC and other dataset-wide numbers are not compared between the old and the new SW1: the attack
mix changed, and magnitude spikes are now about a third of the attacks (spec A1.5 rule 5).

### Dataset census

From `matrix.md`. Spikes are counted over the whole dataset; the gate cell is the warm test rows in
bucket 10-100.

| dataset | traces sha | envelopes | attacks | spikes: first pass / dropped / top-up | spikes with strength ≥ 10 | share with strength ≥ 10 | median strength | gate cell n | gate cell by persona |
|---|---|---|---|---|---|---|---|---|---|
| SW1 seed 0 | `a455b10c705d` | 174,878 | 808 | 53 / 2 / 211 | 256 | 0.970 | 48.3 | 81 | support-refunds-c 22, lead-router-a 17, support-refunds-a 11, sales-ops-a 8, support-refunds-b 8, billing-disputes-a 4, sales-ops-b 4, crm-hygiene-a 3, exec-assistant-a 2, payouts-ops-a 1, renewals-a 1 |
| SW1 seed 1 | `e783c301bf21` | 172,805 | 813 | 49 / 1 / 203 | 249 | 0.988 | 50.5 | 85 | lead-router-a 23, support-refunds-c 15, sales-ops-a 9, sales-ops-b 9, crm-hygiene-a 8, support-refunds-a 8, payouts-ops-a 5, support-refunds-b 4, exec-assistant-a 2, billing-disputes-a 1, renewals-a 1 |
| SW1 seed 2 | `62645fcfd0b8` | 175,890 | 810 | 39 / 2 / 223 | 257 | 0.981 | 51.5 | 85 | support-refunds-c 20, support-refunds-a 16, lead-router-a 10, sales-ops-a 9, support-refunds-b 9, sales-ops-b 8, payouts-ops-a 4, billing-disputes-a 3, crm-hygiene-a 3, renewals-a 3 |
| sensitivity, seed 0 (`magnitude_far_mult = 1.0`) | `3fc612ce1fdb` | | | | | | | 64 | |

`Label.strength` keeps its meaning: the realized multiplier of the call's own value, in [3, 100].
The gate bucket is still multiplier 10 to 100 (D-009). Because spikes now land far outside the
benign range, 97 to 99 % of them have strength ≥ 10, so the 3-10 bucket is nearly empty.

### What is still not met

Spec A1.12 (items 1 to 14), with the numbers measured on the implementation (`matrix.json`,
`matrix.md`); item 15 comes from `docs/leaderboard.md`.

1. **The amended criterion fails under spec §5.2 allow-all on seeds 0, 1 and 2.** Seed 0:
   exfil_volume 0.579 (11/19), prereq_skip 0.533 (8/15), novel_target 0.163 (8/49), velocity_burst
   0.800 (40/50). Seed 1: those four (0.655 (19/29), 0.667 (12/18), 0.222 (8/36), 0.808 (42/52))
   and magnitude_spike 0.824 (70/85) (pre-A1 0.788 (67/85) on the same data; companion rows above).
   Seed 2: exfil_volume 0.711 (27/38), prereq_skip 0.846 (11/13), novel_target 0.211 (8/38),
   velocity_burst 0.682 (30/44). This is cause (a) of this decision, and no scoring change in
   Amendment A1 addresses it: every attack row is learned. In the seed-0 leaderboard runs the
   median live threshold over the warm test rows is ≥ 0.999 on 18 of 20 agents under spec §5.2
   allow-all and on 2 of 20 under the oracle-reviewer approval loop (labels act as a perfect human
   reviewer → optimistic) *(manual inspection: read from the runs' `scores.parquet`, which is not
   committed)*. Under the oracle-reviewer approval loop (labels act as a perfect human reviewer →
   optimistic) every gated family passes on every seed.
2. **Under spec §5.2 allow-all, recall gets worse for novel_target** on the same data: pre-A1 →
   amended 0.469 (23/49) → 0.163 (8/49) on seed 0, 0.444 (16/36) → 0.222 (8/36) on seed 1,
   0.474 (18/38) → 0.211 (8/38) on seed 2. prereq_skip on seed 0 goes 0.667 (10/15) → 0.533 (8/15).
   The mechanism (spec A1.12 item 2): bursts now score near 1 and are learned, so the thresholds of
   helpdesk-triage-b and slack-notifier-a saturate. The old number existed because the old velocity
   score failed on those agents. It must not be "fixed" by capping velocity or magnitude scores.
3. **The original spec §9 criterion (top-1 share ≥ 0.90) is not met under either protocol.** Under
   the oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic):
   velocity_burst 0.40 / 0.24 / 0.43, exfil_volume 0.84 / 0.79 / 0.82, scope_blowup 0.85 on seed 0.
   On a duplicated write, sequence reaches 1.0 on the third or fourth duplicate, before velocity
   can. The gate no longer requires top-1; it stays reported.
4. **Under the oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic)
   the magnitude cell is decided mostly by the generator correction**, and on seed 1 the pre-A1
   engine passes the amended criterion (companion rows above).
5. **On the sensitivity dataset the amended engine does not reach 0.90:** 0.797 (51/64) under the
   oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic),
   0.656 (42/64) under spec §5.2 allow-all (pre-A1 0.688 (44/64) / 0.578 (37/64)). Spec A1.12
   counted 12 of the prototype's 64 gated spikes at or below the field's largest benign value; that
   count was not repeated on the implementation's sensitivity dataset.
6. **One velocity burst is still missed under the oracle-reviewer approval loop (labels act as a
   perfect human reviewer → optimistic) on seed 1:** 0.981 (51/52). Under spec §5.2 allow-all
   seed 1 misses 10 of 52 (0.808 (42/52)).
7. **Control FPR under the oracle-reviewer approval loop (labels act as a perfect human reviewer →
   optimistic) is 3.52 % / 2.78 % / 5.01 %** on seeds 0 / 1 / 2 (under spec §5.2 allow-all 0.73 % /
   1.31 % / 3.08 %). Seed 2 is above the 5 % Tier-3 bound that spec A1.12 item 7 states for seeds 0
   and 1. Nothing was retuned (ruling A1-22).
8. **Known over-scoring cases of the magnitude tail score** (spec A1.12 item 8, measured on the
   prototype and not re-measured): power-law tails; small integer fields with tied markers; a field
   that never varied (any change of about 14 % scores 0.5); `q95`'s 5 % breakdown point under
   spec §5.2 allow-all.
9. **Velocity's over-dispersion correction is weak** (`disp` 1.0 to 1.15 on keys whose
   variance-to-mean ratio is above 2). The share of benign velocity scores ≥ 0.5 under the
   oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic) is 0.76 % /
   1.22 % / 0.92 % on seeds 0 / 1 / 2 (pre-A1 1.11 % / 1.42 % / 1.27 %), above the 0.5 % prior. The
   per-agent shares are now recorded in `matrix.md` with no threshold; the highest under the
   oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic) is
   payouts-ops-a at 3.53 % (seed 2; 3.67 % under spec §5.2 allow-all).
10. **Not measured:** `Engine.score` latency p99 on an idle machine (`latency_p99_us` is null in
    `matrix.json`; three runs of the warm-key perf test under load stayed below the 200 µs budget
    and are not pinned); seed 1 and seed 2 on the sensitivity dataset; Go parity of `erfc` and the
    cube root; the cause of the sensitivity and allow-all magnitude drop against the prototype's
    fixed-count data; the 90-day seed-0 ablation evidence of decision 007, which was measured on the
    pre-A1 engine (decision 007's dated addendum records only the `tiny_dataset` re-measurement).
11. **A BOOL-typed field in magnitude cold start** still gets only novelty's ramped fallback score.
12. **The magnitude gate cell is concentrated.** Seed 0: 81 attacks on 11 of 20 personas;
    support-refunds-c holds 22 (27 %) and the three refund personas 41 (51 %). Seed 1: 85 on 11
    personas; lead-router-a holds 23 (27 %). Seed 2: 85 on 10 personas; support-refunds-c holds 20
    (24 %) and the three refund personas 45 (53 %). A per-persona failure moves the cell by up to
    27 points.
13. **The user's item 2 is not met literally at ×10 on the widest SW1 field.** "10 to 100 times the
    usual value scores close to 1" holds at ×100 on every SW1 field, and at ×10 only on narrow
    fields (log-sd up to about 0.6). At ×10 on sales-ops-a `amount` (log-sd 0.86) the score is
    0.575 (spec A1.2).
14. **A1.3 silences novelty on BOOL-typed contradictions that are not above the learned range**
    (spec A1.12 item 14). Under `mag_bool_numeric = True` a negative number in an `up` BOOL-typed
    field (e.g. `qty = -5`) and a fraction between 0 and 1 in any BOOL-typed field now score about
    0.01 by magnitude (z = 0 under the one-sided rule for the negative number; novelty's fallback
    is silent once magnitude has a reference), where the pre-A1 engine scored 1.0 (novelty FR2
    N1). Pinned by
    `tests/core/test_engine.py::test_pinned_contradiction_not_above_the_learned_range_scores_low`.
    Keeping novelty's score for those values is a behaviour change left to the user (spec A1.13
    item 6).
15. **On the leaderboard's ranking metric the engine before Amendment A1 does better** (found on
    the implementation's leaderboard; not in spec A1.12). Per-agent FPR@95 under the
    oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic) is
    0.490 [0.405, 0.557] for the amended engine and 0.456 [0.344, 0.536] for the engine before A1
    by time, and 0.726 [0.675, 0.787] against 0.689 [0.645, 0.819] by agent; the 95 % CIs overlap.
    The engine before A1 ranks first by time on `docs/leaderboard.md`. By agent, the control FPR
    at that operating point is 0.931 against 0.822. Under spec §5.2 allow-all the amended engine
    is lower on both splits (0.542 against 0.555 by time, 0.673 against 0.679 by agent). Nothing
    was retuned in response.

### Decisions left to the user

Spec A1.13, copied verbatim:

This amendment takes a default on each point below. Each is the user's decision; the default stands until the user changes it.

1. **Which protocol's result closes Tier 1.** Default: left open; both results are always shown (A1.5) and a pass under the oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic) alone is not described as the Tier-1 gate being met.
2. **"Far" = 5 × the persona's benign p99.** Default: `magnitude_far_mult = 5.0` (A1.4).
3. **The top-up rate 0.015 and the 50-attack minimum as a dataset-validity assertion.** Default: `magnitude_topup_rate = 0.015` on SW1; a dataset with fewer than 50 attacks in the gate cell fails the acceptance test (A1.4, A1.7).
4. **The acceptance file goes green on its `review` parameter (oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic)), with its `allow` parameter (spec §5.2 allow-all) as a strict xfail.** Default: yes (plan Task 1.24).
5. **×10 on the widest SW1 field scores 0.575.** Default: accepted (A1.2, A1.12 item 13).
6. **A1.3 lowers an existing score.** After `[0, 1]·40` a `2` scored 1.0 by novelty (FR2 N1) and now scores 0.048; a `5` scores 0.335. A negative number in an `up` BOOL-typed field (e.g. `qty = -5`) and a fraction between 0 and 1 in any BOOL-typed field now score about 0.01 by magnitude (novelty's fallback is silent once magnitude has a reference) where the pre-A1 engine scored 1.0; keeping novelty's score for those values is a behaviour change (A1.12 item 14). Default: accepted (`mag_bool_numeric = True`).
7. **The new defaults leave each agent's calibrator holding old-formula scores after an upgrade (A1.8).** Default: the calibrator records are kept and the thresholds adapt over time; they are not reset.
