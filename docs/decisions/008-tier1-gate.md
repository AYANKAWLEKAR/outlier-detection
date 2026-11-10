# 008 — Tier-1 exit gate: root causes and the fix candidates

Status: **DRAFT (Task 1.15 phase A).** It needs a controller ruling. No engine default and no
replay default was changed. One category-(i) bug was fixed: the generator's `exfil_volume`
labels.

Evidence for every number below: `evidence/gate-seed0-seed1.md`, with the scripts in
`evidence/008-root-causes/` and the `gate` suite of `agentfw.bench.sweeps`.

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
- Each miss is then classified by the first rule that matches, in this order (details in
  `classify.py`):
  1. **(i)** a bug against the spec;
  2. **(ii)** detected once attacks are not learned, i.e. a calibration or learning artifact of
     the allow-all replay;
  3. **(iii)** the attacked value is not beyond the persona's own benign p99.5, i.e. not
     anomalous;
  4. **(iv)** anomalous, but the spec'd detector cannot score it above a clean threshold.

The table uses the labels as they stood at Task 1.14. Cells are seed 0 / seed 1.

| family (n) | hit | (i) bug | (ii) learns attacks: calibrator + detector state | (iii) not anomalous | (iv) detector / design limit |
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
  learns its attack rows. On some agents these are 0.7–13 % of rows: velocity duplicates, and
  scope or novelty hits near 1.0. As a result, 17 / 16 of 20 agents end with θ ≥ 0.999, and
  only fused scores of exactly 1.0 are flagged.
  - Novelty scores for novel targets are 0.990–0.9998 against θ = 0.9999.
- **Detector state learns attacks.** Examples:
  - helpdesk-triage-a's `__resp__.item_count` key for `zendesk.list_tickets` holds only 62
    observations, because a session's last response is never learned and the read is usually
    last. A large share of those 62 come from exfil attack sessions, which append the outbound
    write after the read. The IQR is 119 against a median of 9, so a 64× read scores 0.17.
  - Learned novel values raise `p_unseen`, which lowers the novelty score.

**(i), one bug, now fixed (TDD).** Per §2.3 and D-004, the response `item_count` is scored on the
*next* call of the session. `exfil_volume` labelled the read and the first outbound write, but
not the call in between. In 7 of 19 (seed 0) attacks another call sits between the read and the
write. That call carries the only scoreable evidence, and it was labelled benign: one missed
attack plus one false positive.

- The generator now also labels the call right after the read.
- Test: `test_exfil_volume_labels_the_call_that_carries_the_response`.
- With the fixed labels every former (i) miss becomes (ii) (evidence §1, `sweep-seed*`). Under
  clean learning ("oracle"), exfil recall goes 0.895 → 1.000 on seed 0.
- Goldens and the CI baseline are unchanged: the CI dataset has no `exfil_volume`, and
  `test_golden_engine` and `ci_regress` pass unmodified.

**(iii) and (iv): the residue that no protocol fixes.**

- magnitude_spike:
  - Two spikes are within the persona's own range: `sales-ops-a` amount ×22 is still exceeded by
    1.7 % of its benign amounts, and lead-router-a by 0.8 %.
  - Three are anomalous but sit on wide log-normal fields: ×12 and ×10 on `due_in_days`, and
    ×24 on `amount`. Their robust z = ln m / σ_log ≈ 2.7–3.2 < z0 = 3, so they score 0.40–0.60.
  - With n = 13 / 8, ≥ 0.90 allows only 1 / 0 misses.
- velocity_burst:
  - helpdesk-triage-b's normal 10-minute windows hold 1–6 calls. Bursts of 12–41 score 0.60–0.89
    after the log1p squash, against θ ≈ 0.90. About 46 calls would be needed.
  - Two short bursts sit near the persona's own tail: 13 calls on helpdesk-triage-b (seed 1,
    1 % of benign windows reach it, so (iii)) and 10 calls on slack-notifier-a (seed 0, 0.3 %).
- novel_target: the novelty ramp `min(1, n / 200)` discounts stable vocabularies with 81–178
  observations. Those vocabularies have a benign new-value rate of 0, so the discount is the only
  reason they score 0.36–0.84.

## 2. The candidates (all opt-in; defaults unchanged)

| id | what | code |
|---|---|---|
| enforce | outcome BLOCKED when the engine decides BLOCK; nothing learns a blocked call | `replay(..., outcomes="enforce")` |
| drop | the calibrator skips calls the engine itself blocked; detectors learn all | `EngineConfig.cal_censor="drop"` |
| winsorize | the calibrator learns θ instead of the fused score for such calls, including BLOCKED ones | `cal_censor="winsorize"`, optionally `cal_censor_min_n` (shadow first) |
| review | the §6.8 feedback loop: each flagged call (BLOCK or REQUIRE_APPROVAL) is APPROVED if benign and DENIED if an attack; unflagged calls are ALLOWED | `replay(..., outcomes="review")` |
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
| review | 0.680 / 0.750 | 0.692 / 0.750 | 0.923 / 1.0 | 1.0 / 0.966 | 1.0 / 1.0 | 0.837 / 0.944 | 0.680 / 0.750 | 1.25 % / 1.20 % | 13.2 % / 8.7 % | 0.506 / 0.579 | 0.475 / 0.625 | 0.914; 0 / 5 · 0.901; 0 / 6 |
| review + vel attempts | 0.692 / 0.750 | 0.692 / 0.750 | 0.923 / 1.0 | 1.0 / 0.966 | 1.0 / 1.0 | 0.816 / 0.944 | 0.880 / 0.962 | 1.24 % / 1.19 % | 13.2 % / 8.7 % | 0.590 / 0.638 | 0.475 / 0.624 | 0.914; 0 / 5 · 0.901; 0 / 6 |
| **review + vel attempts + graded prereq + nov50** | 0.692 / 0.750 | 0.692 / 0.750 | 1.0 / 1.0 | 1.0 / 0.966 | 1.0 / 1.0 | **1.0 / 1.0** | 0.880 / 0.962 | **0.93 % / 0.89 %** | 13.8 % / 8.7 % | **0.661 / 0.686** | 0.481 / 0.625 | 0.908; 1 / 4 · 0.901; 0 / 2 |
| oracle (diagnostic) | 0.769 / 0.750 | 0.769 / 0.750 | 0.923 / 1.0 | 1.0 / 0.966 | 1.0 / 1.0 | 0.878 / 0.944 | 0.780 / 0.885 | 1.19 % / 1.18 % | 13.5 % / 8.2 % | 0.537 / 0.626 | 0.543 / 0.738 | 0.914; 0 / 5 · 0.896; 0 / 6 |

### The censoring ratchet (measured)

Label-free censoring cannot learn the benign tail above θ, because θ starts at θ0 = 0.6, which is
below every agent's benign q_0.995 (≈ 0.79–1.0 under clean learning).

- **enforce and drop truncate the calibrator's input at θ.** A truncated `q_{1−α}` always falls
  below θ, so θ ratchets down while the calibrator is still blending. Median θ ends at
  0.33–0.44, the minimum agent reaches 0.05–0.08, and 3–4 agents per seed are still falling
  during the test period. Benign FPR is 32–53 %.
- **winsorize** keeps the tail mass but pins it at the threshold that blocked it. The cold-start
  blocks at θ0 become a permanent point mass above α, so θ freezes at exactly 0.600 for 20 of
  20 agents. Benign FPR is 13.5 %.
- **Shadow-then-censor** (`cal_censor_min_n=500`, allow-all) starts from the polluted
  allow-all θ. Benign FPR is 5.2 %, min recall is 0.69, and 5–6 agents stay saturated. Adding
  enforce sends it back down to 0.50.
- **Review does not ratchet.** Benign flagged calls are approved and learned, which is exactly
  §6.8. θ converges to the oracle's level: median 0.90–0.91, with a drift of +0.001 over the
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

## 3. Recommendation (needs a ruling)

1. **Keep the exfil label fix.** It is already in this commit and is a category (i) bug.
2. **Gate replay protocol: `outcomes="review"`.** This deviates from §5.2's allow-all, which
   makes the live per-agent threshold meaningless whenever attacks exceed α of an agent's
   traffic.
   - Every label-free alternative either ratchets or freezes, and costs 5–53 % benign FPR.
   - Review is §6.8's own approval feedback with a perfect reviewer. It reads labels only for
     flagged calls; §5.4's per-agent FPR@95 already calibrates on benign labels.
   - It must be disclosed in the report, together with its load: 7.1–8.7 flagged calls per
     agent-day, 4.0–5.5 of them benign, out of ≈ 96 calls per agent-day.
3. **Make three engine switches the default.** Each is a recorded spec deviation, and each
   changes the engine golden and baseline in the same commit:
   - **`vel_count_attempts=True`.** A refused call still counts as an attempt in velocity's
     current window, but never in its baseline. Under any feedback protocol, velocity otherwise
     stalls at θ−δ because denied duplicates vanish (velocity_burst 0.680 → 0.880 / 0.750 →
     0.962). It is a no-op under allow-all. It deviates from "BLOCKED never moves state", but
     no baseline distribution moves.
   - **`seq_prereq_graded=True`.** `s_prereq` is graded instead of the binary 1 of §6.6(b).
     Benign FPR falls 1.25 → 0.93 % / 1.20 → 0.89 % (with nov50), benign-saturated agents fall
     5 → 3 / 6 → 2, AUPRC rises 0.59 → 0.66 / 0.64 → 0.69, and no gate recall is lost.
   - **`novelty_full_n=50`** instead of the spec's 200 (§6.5). novel_target goes
     0.878 → 1.000 / 0.972 → 1.000. Benign and control FPR do not rise measurably.
4. **Expected gate outcome with 2 + 3, stated honestly: still FAIL.**

   | seed | passes | fails |
   |---|---|---|
   | 0 | scope 1.0, exfil 1.0, prereq 1.0, novel 1.0, benign FPR 0.93 % | magnitude_spike 0.692 (9 / 13), velocity_burst 0.880 (44 / 50) |
   | 1 | velocity 0.962, exfil 0.966, the other three 1.0, benign FPR 0.89 % | magnitude_spike 0.750 (6 / 8) |

   In addition, the brief's `top_reason_share ≥ 0.90` fails for velocity_burst and exfil_volume
   on both seeds.

   The remaining misses are (iii) and (iv), so the brief's "do not lower the gate" collides with
   the spec'd detector design. The controller must rule on:
   - **(a) Attribution:** gate on `reason_share` (the targeted detector among the flagged row's
     reasons; 0.89–1.00) instead of top-1. This matches Tier 3's "every family yields a correct
     reason".
   - **(b) magnitude_spike 10-100.** There are 6 residual misses (4 + 2):
     - 2 are not anomalous for their persona;
     - 2 sit on agents whose benign tail still fuses to ≈ 1.0, chiefly `sequence:flow`
       (sales-ops-b, support-refunds-c);
     - the others are ×10–×24 spikes on fields with log-σ ≈ 0.75–1, below what z0 = 3 /
       τ = 0.75 in log space can flag at θ ≈ 0.9.

     The options:
     - a generator change that defines the spike relative to the field's dispersion;
     - a magnitude scoring change;
     - accepting that the family is bounded by the spec design. n = 8–13 attacks also makes ±1
       attack = 8–12 pp.
   - **(c) velocity on helpdesk-triage-b.** The options are a velocity-specific z0 / τ or a
     count-tail score, for example Poisson in raw counts instead of log1p.
5. **Control FPR rises** 6.2 → 13.8 % / 4.9 → 8.7 % once thresholds stop saturating. Tier 3's
   exit needs ≤ 5 %, so blast radius has to absorb this.

## Spec deviations implied by the recommendation

| deviation | from |
|---|---|
| review replay protocol | §5.2 "outcome = allow for every envelope" |
| `vel_count_attempts` | §2.2 and global constraints: blocked calls never move state. Only the window count moves; the baseline does not |
| `seq_prereq_graded` | §6.6(b): s_prereq = 1 |
| `novelty_full_n = 50` | §6.5: 200 |
| exfil carrier label | §4.6 labelled "mutated envelopes" only. This deviation follows §2.3 |
