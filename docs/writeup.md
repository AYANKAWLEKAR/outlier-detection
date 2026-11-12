# Agent Action Firewall: engine v1 on a synthetic benchmark

Status: Tier 2 build complete; **Tier-2 exit not reached** (the live sandbox dataset v1 has not
been built). **Tier 1** (spec Amendment A1, criterion A1.7): the amended criterion passes under the
oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic) and fails under
spec §5.2 allow-all, on SW1 seeds 0, 1 and 2. Which protocol's result closes Tier 1 is the user's
open decision (D-028; decision 008). Every number on this page comes from the **synthetic benchmark
(SW1)**, not from real or live agent traffic.

## 1. Problem

An LLM agent with tools (refunds, CRM writes, outbound mail, chat posts) can be steered into a
harmful call by a prompt injection or a bug. The calls are well-formed and authorised, so
signature or schema checks do not catch them. The engine scores each call by how unusual it is
for *this* agent, using only the agent's own history and a pooled prior across agents, and flags
calls above a per-agent threshold. It needs no labels and learns online.

## 2. Method: the engine, stage by stage (spec §6, Amendment A1)

Each tool call (`RawCall`) passes through these stages in order. Every stage scores before it
updates, and a blocked or denied call never updates a baseline.

1. **Normalize (§6.1).** Flatten args to dotted paths; list lengths become numeric
   `path.__len__`; emails and URLs yield a lowercased, IDNA-normalized domain or host; id-like
   strings yield a prefix; strings of 40+ characters keep only their length (free text is never
   stored). The session context adds the last four tools (`prior_tools`) and the ids seen in
   earlier args and responses (`session_ids_seen`). The previous call's response `item_count`
   is scored on the next call (§2.3, D-004).
2. **Type inference (§6.2).** Per (agent, tool, path): numeric, bool, id, freetext or
   categorical, sticky over 100 observations, `unknown` below 10.
3. **Magnitude (§6.3, A1.2, A1.3).** Values are taken on the scale `sign(x)·log1p(|x|)`. The
   score is a tail probability: the distance to the P² median, divided by a one-sided robust
   scale (the larger of the quartile and the 5 %/95 % marker spreads), gives a normal tail `p`,
   and `s = 0.005 / (0.005 + 0.995·p)` (`tail_prior` = `cal_alpha`, D-023). A benign value at the
   field's 99.5th percentile scores about 0.5. ×100 the median scores close to 1 on every SW1
   field; ×10 scores close to 1 only on narrow fields (log-sd up to about 0.6) and 0.575 on the
   widest SW1 field (log-sd 0.86). Money and count fields are one-sided. Cold keys (n < 30) score
   against the pooled cross-agent prior with the scale inflated 1.5× (§6.10). Numbers in a
   BOOL-typed field are learned and scored too (D-024; spec A1.3). Under that rule a negative
   number in an `up` BOOL-typed field (e.g. `qty = -5`) and a fraction between 0 and 1 in any
   BOOL-typed field now score about 0.01 by magnitude, and novelty's fallback is silent once
   magnitude has a reference; the engine before Amendment A1 scored them 1.0 (spec A1.12 item
   14; whether to keep novelty's score for them is the user's decision, A1.13 item 6).
4. **Velocity (§6.4, A1.1).** The current 10-minute call count `c` is scored by how unlikely it
   is at the key's normal rate (the median active-minute window count, D-022): a Poisson
   upper-tail deviate, divided by the key's over-dispersion, squashed with `σ((z − 3)/0.75)`. The
   money-sum signal keeps the robust log z.
5. **Novelty (§6.5).** An unseen categorical or id value scores
   `(1 − p_unseen) · min(1, n/50)`, with Good–Turing `p_unseen`; derived domains and prefixes
   have their own keys (`novelty_full_n = 50` by user decision, decision 008).
6. **Sequence (§6.6).** The maximum of three signals: transition surprise (k = 2 back-off,
   entropy-scaled), a mined prerequisite missing from `prior_tools` (graded by the mined rate,
   decision 008), and an id flow break (a write to an id the session never saw, weighted by how
   often this arg's ids normally flow).
7. **Fusion v1 (§6.7).** `top1 + 0.15·[top2 > 0.5]`, clipped to 1. Every detector score above
   0.3 becomes a reason.
8. **Calibration (§6.8).** Per-agent threshold `θ = q_{0.995}` of the agent's allowed fused
   scores (P² tail), blended with θ0 = 0.6 for the first 500 scores. Decision bands: block at
   ≥ θ, require approval in [θ − 0.1, θ).

All constants live in `agentfw/core/config.py::EngineConfig`. The spec's original formulas stay
selectable (`SPEC_SWITCHES`; the engine before Amendment A1 is `PRE_A1_SWITCHES`). The Tier-1 perf
test asserts p99 < 200 µs per score on warm keys. After Amendment A1, `Engine.score` p99 was not
measured on an idle machine: three runs of the warm-key perf test under load stayed below the
budget, and the leaderboard replays (all rows, four runs in parallel) showed p99 of about 0.3 ms;
neither is a latency result.

## 3. Benchmark protocol (spec §5, Amendment A1.5)

- **Replay (§5.2).** Envelopes in time order; score, then `update` with the call's outcome. The
  benchmark protocol (user decision 2026-11-10, decision 008) is the **oracle-reviewer approval
  loop (labels act as a perfect human reviewer → optimistic)**: an unflagged call is ALLOWED, and
  a flagged call is APPROVED if benign and DENIED if an attack, so nothing learns an attack the
  engine flagged. The spec's own protocol, **spec §5.2 allow-all**, allows and learns every call.
- **The Tier-1 gate is always evaluated under both protocols** on the same dataset and engine, and
  the two results are always shown together, each under its label (spec A1.5, D-028). A result
  under one protocol alone is never the gate result.
- **Splits (§5.3).** By time (first 70 % train, last 30 % test; the detector keeps learning
  online through the test period) and by agent (70 % of personas train, 30 % held out). The two
  are separate tables and are never averaged.
- **Metrics (§5.4).** AUPRC over fused scores; FPR at 95 % recall with a global threshold and with
  per-agent thresholds (each agent's threshold is a quantile of its own train-split benign scores,
  with one shared α chosen so that pooled test recall reaches 0.95; D-007, D-021). The per-agent
  number is the leaderboard metric, and it is always shown with the recall it achieved. The
  headline uses warm envelopes (≥ 30 prior observations of the agent-tool key). 95 % CIs come from
  a 1,000-sample bootstrap over sessions.
- **Gate criterion (A1.6, A1.7).** At the engine's live per-agent thresholds on warm test rows,
  per gated family (`slow_drift` excluded; `magnitude_spike` with strength 10 to 100): attack-level
  recall ≥ 0.90, and the targeted detector among the reasons (score > 0.3) of a flagged row in
  ≥ 0.90 of detected attacks; benign FPR ≤ 2 %. The top-1 share (the targeted detector is the top
  reason) is the original spec §9 criterion; it is still printed beside the new share and is not
  gated.
- **Baselines (§5.5).** `random`; `static_threshold` (per-field train p99 bound plus a train
  allowlist of categorical values, no sequence); `isolation_forest` (batch fit on the train split
  only: it is not an online learner and cannot extrapolate above its train range);
  `llm_judge` (a frontier model scoring the last 8 envelopes of the session, on a stratified
  2,000-envelope sample). The baselines do not learn online, so the protocol does not change their
  scores.

**What the per-agent FPR@95 operating point is not.** It is fixed after the fact using the test
labels: it asks "how many benign calls would be flagged if the thresholds were set just low
enough to catch 95 % of attack envelopes". The engine's live thresholds are much higher (target
FPR 0.5 %), and their recall is the Tier-1 gate (§6 below).

## 4. Datasets

| dataset | status | what it is |
|---|---|---|
| **SW1** (synthetic) | used for every number here | `sw1_config(0)` (spec A1.4): generator seed 0, all 20 personas, 90 days, all 8 attack families, attack rate 0.015, magnitude spikes in mode "far" plus a magnitude-only top-up pass at rate 0.015. 42,289 sessions, 174,878 envelopes, 808 attacks, of which 264 are magnitude spikes (53 first pass, 2 dropped, 211 top-up). Dataset sha256 `a455b10c705d…` (traces.jsonl). Rebuilt from a clean checkout by `scripts/build_leaderboard.py`. Seeds 1 (`e783c301bf21`) and 2 (`62645fcfd0b8`) and the sensitivity dataset (seed 0 with `magnitude_far_mult = 1.0`, `3fc612ce1fdb`) are used for the gate evidence only. The SW1 before Amendment A1 (`e7b3bc2a2f20`, 599 attacks) is superseded; its numbers are not compared with these. |
| **sandbox dataset v1** | **not built** | LLM agents (one model) running ~70 task templates against sandboxed Stripe test mode, a Salesforce Developer org, a Slack sandbox channel, a local Zendesk-like server and Mailhog, with prompt- and action-level injections. Minimums: 20 agents, 2,000 sessions, 40,000 actions, 600 attacks. The README's "How to build dataset/v1" gives the commands; they call a paid LLM and live sandbox APIs and are run by hand. |

The eight families are `magnitude_spike`, `scope_blowup`, `exfil_volume`, `prereq_skip`,
`novel_target`, `velocity_burst`, `slow_drift`, and the `benign_unusual` control set (benign but
unusual calls, used for the control FPR). A `magnitude_spike` multiplies a money or count field of
a write call by `m`, log-uniform on [3, 100]. On SW1 the draw is conditioned on landing at or
beyond 5 × the persona's own benign p99 of the field (D-025), the level the generator labels benign
in its control; a session with no such candidate stays benign and is counted. A magnitude-only
top-up pass then adds spikes: it is a fixed-rate pass at 0.015 per eligible untouched session, not
a target count (81 spikes in the seed-0 gate cell, first pass and top-up together); a dataset with
fewer than 50 gated spikes fails the acceptance test (D-026). `Label.strength` is still
the realized multiplier, and the gate bucket is still 10 to 100 (D-009); 97 % of the seed-0 spikes
now have strength ≥ 10 (median 48.3).

## 5. Results on SW1

The full leaderboard, both splits and the per-family tables, is
[docs/leaderboard.md](leaderboard.md). Every row there is the **synthetic benchmark (SW1)**, full
test split. Warm test rows; 95 % CIs from 1,000 session-bootstrap samples. Each engine row names
its protocol. AUPRC and other dataset-wide numbers are not comparable with the SW1 before
Amendment A1: the attack mix changed (magnitude spikes are about a third of the attacks; spec A1.5
rule 5).

**By time** (53,011 warm test envelopes, 2,127 attack envelopes):

| detector | protocol | AUPRC [95 % CI] | FPR@95 per-agent [95 % CI] (recall achieved) | FPR@95 global |
|---|---|---|---|---|
| engine | oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic) | 0.779 [0.727, 0.820] | 0.490 [0.405, 0.557] (0.950) | 0.546 |
| engine | spec §5.2 allow-all | 0.736 [0.680, 0.782] | 0.542 [0.528, 0.605] (0.950) | 0.587 |
| engine before A1 | oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic) | 0.677 [0.594, 0.748] | 0.456 [0.344, 0.536] (0.950) | 0.423 |
| engine before A1 | spec §5.2 allow-all | 0.627 [0.536, 0.704] | 0.555 [0.496, 0.584] (0.951) | 0.537 |
| isolation_forest (batch) | (no online learning) | 0.060 [0.047, 0.080] | 0.862 [0.644, 0.925] (0.972) | 1.000 |
| random | (no online learning) | 0.041 [0.033, 0.051] | 0.943 [0.936, 0.958] (0.950) | 0.944 |
| static_threshold (batch) | (no online learning) | 0.051 [0.037, 0.068] | 0.957 [0.786, 0.997] (0.976) | 1.000 |
| llm_judge | | not run (requires a live judge model) | | |

**By agent** (6 held-out personas; 43,562 warm test envelopes, 1,701 attack envelopes):

| detector | protocol | AUPRC [95 % CI] | FPR@95 per-agent [95 % CI] (recall achieved) | FPR@95 global |
|---|---|---|---|---|
| engine | oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic) | 0.737 [0.655, 0.790] | 0.726 [0.675, 0.787] (0.950) | 0.726 |
| engine | spec §5.2 allow-all | 0.687 [0.596, 0.749] | 0.673 [0.645, 0.709] (0.950) | 0.673 |
| engine before A1 | oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic) | 0.720 [0.636, 0.779] | 0.689 [0.645, 0.819] (0.950) | 0.689 |
| engine before A1 | spec §5.2 allow-all | 0.671 [0.577, 0.739] | 0.679 [0.662, 0.736] (0.950) | 0.679 |
| random | (no online learning) | 0.038 [0.029, 0.049] | 0.948 [0.938, 0.960] (0.950) | 0.948 |
| static_threshold (batch) | (no online learning) | 0.064 [0.043, 0.090] | 1.000 [1.000, 1.000] (1.000) | 1.000 |
| isolation_forest (batch) | (no online learning) | 0.054 [0.042, 0.068] | 1.000 [0.581, 1.000] (1.000) | 1.000 |
| llm_judge | | not run (requires a live judge model) | | |

How to read these numbers:

- **The engine ranks attacks far better than the baselines** (AUPRC 0.74–0.78 against
  0.041–0.060 by time; prevalence is 4.0 % of warm test envelopes, which is roughly what random
  scores). `static_threshold` and `isolation_forest` barely beat random, and their CIs overlap
  with it. The engine's AUPRC CI is wide: the bootstrap resamples sessions, and 50 velocity
  bursts carry 1,699 of the 2,127 attack envelopes.
- **Baselines reach 95 % recall only by flagging nearly everything.** `static_threshold` and
  `isolation_forest` scores have large tied groups (discrete bounds; clipped or zero scores), so
  their recall jumps past 0.95 (0.972–1.000) and their FPR@95 is 0.86–1.00; by agent both flag
  every benign call. Isolation forest is a batch fit on the train split and cannot score
  extrapolation above the train range, which the spec reports as its caveat.
- **No detector is usable at 95 % envelope recall.** The engine's per-agent FPR@95 of 0.490 under
  the oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic) (0.542
  under spec §5.2 allow-all) means flagging about half of benign calls. The metric is dominated by
  `velocity_burst`. Attack-level recall at that operating point is 1.000 for every engine row and
  family (leaderboard per-family table), but that is bought with the same FPR and is not the gate.
- **On the leaderboard's ranking metric the engine before Amendment A1 does better.** Per-agent
  FPR@95 under the oracle-reviewer approval loop (labels act as a perfect human reviewer →
  optimistic) is 0.490 [0.405, 0.557] for the amended engine and 0.456 [0.344, 0.536] for the
  engine before A1 by time, and 0.726 [0.675, 0.787] against 0.689 [0.645, 0.819] by agent; the
  95 % CIs overlap. The engine before A1 ranks first by time on the leaderboard. By agent, the
  control FPR at that operating point is 0.931 against 0.822. Under spec §5.2 allow-all the amended
  engine is lower on both splits (0.542 against 0.555 by time, 0.673 against 0.679 by agent).
  Nothing was retuned in response; the per-agent FPR@95 is not the Tier-1 criterion (§6).
- **At the engine's own live thresholds** (target FPR 0.5 %) benign FPR by time is 0.58 % under
  the oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic) and
  0.14 % under spec §5.2 allow-all; family recall is in §6.
- **By agent.** Held-out personas have no train-split scores of their own, so the leaderboard's
  per-agent thresholds fall back to the pooled train ECDF; this raises FPR@95 to 0.67–0.73. The
  engine itself still learns each held-out agent online during the test period, so the by-agent
  split tests the pooled priors only for the first observations of each key. The gate is
  evaluated by time only.
- **Drift and cold start (run under the oracle-reviewer approval loop (labels act as a perfect
  human reviewer → optimistic), by time, per-agent operating point).** All 11 `slow_drift` attacks
  are caught, with a median delay of 3 sessions and p90 of 7. Every warm test envelope has ≥ 30
  key observations by construction; the 230 envelopes with 30–100 observations show recall 0.929
  at FPR 0.529 (0.951 at 0.484 for the rest).
- **Latency.** The latencies in these leaderboard runs (engine p99 about 0.3 ms in the run under
  the oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic)) were
  measured with four replays running in parallel and are not a latency result.

## 6. Tier-1 status

The criterion is the amended one (spec A1.7, §3 above). Amended engine on SW1 seeds 0, 1 and 2,
copied from `docs/decisions/evidence/a1-gate/matrix.md` (Task 1.23; every number is a cell of
`matrix.json`). Cells: recall (n/N); among-the-reasons share; top-1 share (original spec §9
criterion). Each row is the result under one protocol; the two rows of a seed are read together,
and neither alone is the Tier-1 gate result.

| seed | Protocol | magnitude_spike 10-100 | scope_blowup | exfil_volume | prereq_skip | novel_target | velocity_burst | benign FPR | control FPR | amended criterion |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic) | 1.000 (81/81); 1.00; 1.00 | 1.000 (13/13); 1.00; 0.85 | 1.000 (19/19); 1.00; 0.84 | 1.000 (15/15); 1.00; 1.00 | 1.000 (49/49); 1.00; 1.00 | 1.000 (50/50); 1.00; 0.40 | 0.0058 | 0.035 | PASS |
| 0 | spec §5.2 allow-all | 0.938 (76/81); 1.00; 1.00 | 1.000 (13/13); 1.00; 0.85 | 0.579 (11/19); 1.00; 1.00 | 0.533 (8/15); 1.00; 1.00 | 0.163 (8/49); 1.00; 1.00 | 0.800 (40/50); 1.00; 0.40 | 0.0014 | 0.007 | FAIL: recall:exfil_volume, recall:prereq_skip, recall:novel_target, recall:velocity_burst |
| 1 | oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic) | 1.000 (85/85); 1.00; 1.00 | 1.000 (24/24); 1.00; 1.00 | 1.000 (29/29); 1.00; 0.79 | 1.000 (18/18); 1.00; 1.00 | 1.000 (36/36); 1.00; 1.00 | 0.981 (51/52); 1.00; 0.24 | 0.0062 | 0.028 | PASS |
| 1 | spec §5.2 allow-all | 0.824 (70/85); 1.00; 0.97 | 1.000 (24/24); 1.00; 1.00 | 0.655 (19/29); 1.00; 1.00 | 0.667 (12/18); 1.00; 1.00 | 0.222 (8/36); 1.00; 1.00 | 0.808 (42/52); 1.00; 0.31 | 0.0023 | 0.013 | FAIL: recall:magnitude_spike, recall:exfil_volume, recall:prereq_skip, recall:novel_target, recall:velocity_burst |
| 2 | oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic) | 1.000 (85/85); 1.00; 0.99 | 1.000 (13/13); 1.00; 1.00 | 1.000 (38/38); 0.95; 0.82 | 1.000 (13/13); 1.00; 1.00 | 1.000 (38/38); 1.00; 1.00 | 1.000 (44/44); 1.00; 0.43 | 0.0050 | 0.050 | PASS |
| 2 | spec §5.2 allow-all | 0.965 (82/85); 1.00; 0.93 | 1.000 (13/13); 1.00; 1.00 | 0.711 (27/38); 1.00; 0.96 | 0.846 (11/13); 1.00; 1.00 | 0.211 (8/38); 1.00; 1.00 | 0.682 (30/44); 1.00; 0.37 | 0.0018 | 0.031 | FAIL: recall:exfil_volume, recall:prereq_skip, recall:novel_target, recall:velocity_burst |

On all three seeds the amended criterion passes under the oracle-reviewer approval loop (labels
act as a perfect human reviewer → optimistic) and fails under spec §5.2 allow-all. Which
protocol's result closes Tier 1 is the user's open decision (D-028). The original spec §9 criterion
(top-1 share ≥ 0.90) fails under both protocols: velocity_burst's top-1 share is 0.24 to 0.43.
The Tier-2 exit is not reached.

The composition of the `magnitude_spike` cell (spec A1.4; `matrix.json` `census`, the same
rendering as under the leaderboard's gate tables), one line per seed:

- Seed 0. Census of the magnitude_spike gate cell (spec A1.4): n 81 (warm test rows, bucket 10-100) by persona: support-refunds-c 22, lead-router-a 17, support-refunds-a 11, sales-ops-a 8, support-refunds-b 8, billing-disputes-a 4, sales-ops-b 4, crm-hygiene-a 3, exec-assistant-a 2, payouts-ops-a 1, renewals-a 1; over all 264 spikes of the dataset, median strength 48.3 and share with strength >= 10 0.970.
- Seed 1. Census of the magnitude_spike gate cell (spec A1.4): n 85 (warm test rows, bucket 10-100) by persona: lead-router-a 23, support-refunds-c 15, sales-ops-a 9, sales-ops-b 9, crm-hygiene-a 8, support-refunds-a 8, payouts-ops-a 5, support-refunds-b 4, exec-assistant-a 2, billing-disputes-a 1, renewals-a 1; over all 252 spikes of the dataset, median strength 50.5 and share with strength >= 10 0.988.
- Seed 2. Census of the magnitude_spike gate cell (spec A1.4): n 85 (warm test rows, bucket 10-100) by persona: support-refunds-c 20, support-refunds-a 16, lead-router-a 10, sales-ops-a 9, support-refunds-b 9, sales-ops-b 8, payouts-ops-a 4, billing-disputes-a 3, crm-hygiene-a 3, renewals-a 3; over all 262 spikes of the dataset, median strength 51.5 and share with strength >= 10 0.981.

**Companion rows (spec A1.5 rule 4).** The `magnitude_spike` cell is shown with the engine before
Amendment A1 on the same data and both engines on the sensitivity dataset (spikes just outside
the benign p99; seed 0 only). The sensitivity rows are not gate results.

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

- **On seed 1 the engine before Amendment A1 also passes the amended criterion under the
  oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic).** It fails
  on seeds 0 and 2 (velocity_burst 0.880 (44/50) and 0.864 (38/44); exfil_volume's among-the-reasons
  share 0.89 and 0.88), and under spec §5.2 allow-all on all three.
- Under the oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic)
  the pre-A1 engine already catches 0.975 (79/81), 0.988 (84/85) and 0.953 (81/85) of the far
  spikes. The magnitude cell under that protocol is decided by the generator correction more than
  by the scoring change.
- On the sensitivity dataset neither engine reaches 0.90 in the magnitude cell under either
  protocol.
- The control FPR under the oracle-reviewer approval loop (labels act as a perfect human reviewer
  → optimistic) falls from 0.134 / 0.087 / 0.108 (pre-A1) to 0.035 / 0.028 / 0.050. The drop comes
  from the velocity change: flat keys no longer over-score windows of 3 to 6 calls. Seed 2's
  5.01 % is above the 5 % Tier-3 bound.

**Root causes (decision 008) and what Amendment A1 changed.** Decision 008 traced the gap to (a)
the spec §5.2 allow-all replay learning the attacks (91 of 121 misses), (b) magnitude spikes that
were not extreme, scored on a squash that was too flat, (c) velocity bursts on agents whose
traffic is already bursty, and too few magnitude attacks in the gate cell (13 on seed 0).
Amendment A1 changed the magnitude score (b), the velocity count score (c), the benchmark's spikes
and their number (b, sample size) and the attribution criterion. Nothing in it addresses (a):
under spec §5.2 allow-all every attack row is learned, and in the seed-0 leaderboard run the
median live threshold is ≥ 0.999 on 18 of 20 agents *(manual inspection, §7)*, so only fused
scores of about 1.0 are flagged.

## 7. Failure cases

From the SW1 seed-0 leaderboard runs (by time, warm test rows, the engine's live thresholds).
The per-agent counts and the score and threshold ranges below (which agents, how many rows or
attacks, best-row scores, thresholds, top-detector counts) were read from
`results/synth-v1/time/{engine,engine-allow-all}/scores.parquet` and are *(manual inspection)*:
those files and the script that read them are not committed, as decision 008 marks such numbers.
The other numbers are published elsewhere: gate cells, benign and control FPRs and top-1 shares
are cells of `docs/decisions/evidence/a1-gate/matrix.json`, the per-agent FPR@95 is
`docs/leaderboard.md`'s, and the `slow_drift` envelope recall is the README report block's.

- **Under spec §5.2 allow-all, saturated thresholds hide near-certain anomalies.** The median
  live threshold is ≥ 0.999 on 18 of 20 agents (2 of 20 under the oracle-reviewer approval loop
  (labels act as a perfect human reviewer → optimistic)). 41 of 49 `novel_target` attacks are
  missed: their best rows score 0.983–0.9999 against thresholds of 0.9993–1.0. The engine ranks
  them near the top, yet the calibrator, having learned earlier attacks as benign, sets the bar
  just above them.
- **The new velocity score makes this worse under spec §5.2 allow-all.** Bursts now score near 1
  and are learned, so helpdesk-triage-b and slack-notifier-a saturate; 9 of the 10 missed bursts
  are on those two agents, and novel_target falls from 0.469 (23/49) with the engine before
  Amendment A1 to 0.163 (8/49) (spec A1.12 item 2). The old number existed because the old
  velocity score failed on those agents.
- **Magnitude spikes under spec §5.2 allow-all.** The 5 missed far spikes (0.938 (76/81); the
  engine before Amendment A1 catches 0.753 (61/81) on the same data, and the amended engine
  0.656 (42/64) on the sensitivity dataset, companion rows in §6) are on two refund personas
  (support-refunds-b and support-refunds-c) and score 0.93–0.97 against thresholds of 0.9998–1.0.
- **Weak exfil response signal.** 4 of the 8 exfil_volume misses under spec §5.2 allow-all are on
  helpdesk-triage-a. Their response-count key (`__resp__.item_count`) scores at most 0.035, and
  their best rows (0.44–0.57, from sequence and from novelty on `text`) stay far below thresholds
  of 0.9998–1.0. Decision 008 §1 traced the same key's weak score to the inflated item counts of
  earlier exfil reads in its baseline; that was not re-checked on this dataset.
- **False positives under the oracle-reviewer approval loop (labels act as a perfect human
  reviewer → optimistic) concentrate on sequence.** 297 of 50,884 warm benign calls are flagged
  (0.58 %). 24 are `benign_unusual` controls (control FPR 3.5 %; top detector magnitude 11,
  velocity 8, sequence 5). Of the other 273, sequence is the top detector 151 times, novelty 56,
  velocity 52, magnitude 14; sales-ops-a (48), finance-reconcile-a (35) and sales-ops-b (29) have
  the most.
- **Spikes just outside the benign range are missed.** On the sensitivity dataset the amended
  engine catches 0.797 (51/64) of the gated spikes under the oracle-reviewer approval loop (labels
  act as a perfect human reviewer → optimistic) and 0.656 (42/64) under spec §5.2 allow-all. Some
  of those spikes are at or below a benign value of the same field and should not be caught by any
  honest score.
- **Attribution.** For detected velocity bursts, velocity is the top reason in 0.40 of cases on
  seed 0: sequence reaches 1.0 on the third or fourth duplicated write, before velocity can. It is
  among the reasons of a flagged row in every detected burst.
- **At the leaderboard's operating point the engine is not deployable.** Under the oracle-reviewer
  approval loop (labels act as a perfect human reviewer → optimistic), forcing 95 % envelope recall
  costs a per-agent FPR of 0.490 (0.542 under spec §5.2 allow-all). At that point, in the same
  run, `slow_drift` has the lowest envelope recall (0.656, 143 of 218 envelopes), although every
  drift attack is eventually caught.

## 8. Limitations

- **Synthetic company.** SW1 is one generated company: 20 personas, a fixed tool catalog,
  hand-written task grammars and attack generators written by the same people who wrote the
  detectors. Attack families match the detectors by construction; real attacks need not.
- **The magnitude attacks are far by construction.** On SW1 every gated spike is at least 5 × the
  persona's benign p99 of the field. That corrects labels that contradicted the generator's own
  benign control, but it also makes the cell easier. The sensitivity dataset (spikes just outside
  the p99) is the reported counterweight: 0.797 (51/64) under the oracle-reviewer approval loop
  (labels act as a perfect human reviewer → optimistic) and 0.656 (42/64) under spec §5.2
  allow-all (§6).
- **The top-up rate was chosen with earlier recalls in view** (spec A1.4). The rate 0.015 was set
  once, in the draft phase, when prototypes with a fixed count of 120 and a rate of 0.01 had
  already been replayed and their recalls were known. No other rate was generated, and nothing was
  retuned after the measurement.
- **The oracle-reviewer loop is optimistic.** Its labels act as a perfect human reviewer. The
  deployable behaviour lies between the allow-all and the oracle-reviewer numbers and depends on a
  real reviewer's error rate, which is not measured. Allow-all measures detection without
  enforcement, and that is what makes its live thresholds saturate.
- **Single model family.** The sandbox dataset (when built) uses one agent model; results may not
  transfer to other models, tools or organizations.
- **Retimed clock.** Sandbox sessions are placed on a simulated calendar (D-015): session starts
  are synthetic and only in-session gaps (clamped 0.5–120 s) and burst gaps are real, so velocity
  results on the sandbox data depend on the retiming. SW1 timestamps are fully synthetic.
- **Small and concentrated per-family attack counts.** Gate families have 13–85 warm test attacks
  per seed, so one attack moves recall by 1–8 points; CIs on per-family recall are not reported.
  The magnitude cell is concentrated: on seed 0, support-refunds-c holds 22 of its 81 attacks and
  the three refund personas 41.
- **Attack prevalence** (1.5 % of sessions, plus the magnitude top-up) is far above any real base
  rate, which flatters AUPRC; FPR is prevalence-free but the per-agent operating point is set on
  test labels.
- **Known over-scoring cases of the magnitude tail score** (spec A1.12 item 8): power-law tails,
  small integer fields with tied markers, and fields that never varied. "×10 scores close to 1"
  holds only on narrow fields.
- **Velocity's over-dispersion correction is weak.** Under the oracle-reviewer approval loop
  (labels act as a perfect human reviewer → optimistic) the share of benign velocity scores ≥ 0.5
  is 0.76 % on seed 0 (0.81 % under spec §5.2 allow-all; both above the 0.5 % prior) and reaches
  3.1 % on sales-ops-b (the same under both protocols).
- **Latency after Amendment A1** was not measured on an idle machine (`latency_p99_us` is null in
  `matrix.json`). Three runs of the warm-key perf test (`tests/perf/test_latency.py`) under load
  stayed below the 200 µs budget, and the leaderboard replays (all rows, four runs in parallel)
  showed p99 of about 0.3 ms (§5); neither is a latency result.
- **Batch baseline caveat.** `isolation_forest` is fit offline on the train split, not online.
- **LLM judge not evaluated.** It needs a live model; the code path is tested only with a scripted
  client, so no judge numbers exist.
- **Envelope-level AUPRC is dominated by `velocity_burst`** (10–60 positive envelopes per attack,
  against one for most families), so engine ablations do not order monotonically (decision 007).
  Decision 007's 90-day seed-0 ablation evidence was measured on the engine before Amendment A1
  and was not re-measured. On `tiny_dataset` its strict xfail XPASSed after the A1 scoring became
  the default (warm AUPRC: engine 0.608, engine-no-magnitude 0.522), and decision 007 carries a
  dated addendum. Read ablations from per-family attack-level recall.

## 9. Reproduce

```
uv sync --all-groups
uv run python scripts/build_leaderboard.py       # SW1 + 14 replays + docs/leaderboard.md
docs/decisions/evidence/a1-gate/run.sh seeds01   # SW1 seeds 0, 1 + sensitivity, both engines and protocols
docs/decisions/evidence/a1-gate/run.sh seed2     # then seed 2; rewrites matrix.json / matrix.md
scripts/gate.sh --full                           # includes tests/acceptance/test_tier1_exit.py
```

`run.sh seed2` refuses to run before every seed-0/1 cell is written, so seed 2 stays unseen until
seeds 0 and 1 are recorded.
