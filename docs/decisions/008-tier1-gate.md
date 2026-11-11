# 008 — Tier-1 exit gate: root causes and the fix candidates

Status: **decided (Task 1.15); defaults changed by user decision (2026-11-10).** The controller's
rulings are in the Decision section. At Task 1.15 no engine default and no benchmark-protocol
default changed; the user later turned on the oracle-reviewer protocol and three engine switches
(see "User decision (2026-11-10)" at the end). The Tier-1 gate is still not met. One
label-semantics change was made to the generator: `exfil_volume` now labels the call that
carries the response signal.

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
