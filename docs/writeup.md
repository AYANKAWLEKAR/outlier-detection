# Agent Action Firewall: engine v1 on a synthetic benchmark

Status: Tier 2 build complete; **Tier-2 exit not reached** (the live sandbox dataset v1 has not
been built) and **Tier-1 gate not met under the spec protocol** (decision 008). Every number on
this page comes from the **synthetic benchmark (SW1)**, not from real or live agent traffic.

## 1. Problem

An LLM agent with tools (refunds, CRM writes, outbound mail, chat posts) can be steered into a
harmful call by a prompt injection or a bug. The calls are well-formed and authorised, so
signature or schema checks do not catch them. The engine scores each call by how unusual it is
for *this* agent, using only the agent's own history and a pooled prior across agents, and flags
calls above a per-agent threshold. It needs no labels and learns online.

## 2. Method: the engine, stage by stage (spec §6)

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
3. **Magnitude (§6.3).** Robust z of `sign(x)·log1p(|x|)` against P² quartiles, squashed with
   `σ((|z| − 3)/0.75)`; one-sided for money and count fields. Cold keys (n < 30) score against
   the pooled cross-agent prior with the IQR inflated 1.5× (§6.10).
4. **Velocity (§6.4).** Robust z of the current 10-minute call count (and money sum) against the
   history of active-minute window counts.
5. **Novelty (§6.5).** An unseen categorical or id value scores
   `(1 − p_unseen) · min(1, n/200)`, with Good–Turing `p_unseen`; derived domains and prefixes
   have their own keys.
6. **Sequence (§6.6).** The maximum of three signals: transition surprise (k = 2 back-off,
   entropy-scaled), a mined prerequisite missing from `prior_tools`, and an id flow break (a write
   to an id the session never saw, weighted by how often this arg's ids normally flow).
7. **Fusion v1 (§6.7).** `top1 + 0.15·[top2 > 0.5]`, clipped to 1. Every detector score above
   0.3 becomes a reason.
8. **Calibration (§6.8).** Per-agent threshold `θ = q_{0.995}` of the agent's allowed fused
   scores (P² tail), blended with θ0 = 0.6 for the first 500 scores. Decision bands: block at
   ≥ θ, require approval in [θ − 0.1, θ).

All constants are the spec's values (`agentfw/core/config.py::EngineConfig`). p99 score latency
on warm keys is under 200 µs in pure Python (Tier-1 perf test).

## 3. Benchmark protocol (spec §5)

- **Replay (§5.2).** Envelopes in time order; score, then `update(outcome=ALLOWED)` for every
  call. The harness measures detection, not enforcement, so nothing is ever blocked during replay.
- **Splits (§5.3).** By time (first 70 % train, last 30 % test; the detector keeps learning
  online through the test period) and by agent (70 % of personas train, 30 % held out). The two
  are separate tables and are never averaged.
- **Metrics (§5.4).** AUPRC over fused scores; FPR at 95 % recall with a global threshold and with
  per-agent thresholds (each agent's threshold is a quantile of its own train-split benign scores,
  with one shared α chosen so that pooled test recall reaches 0.95; D-007, D-021). The per-agent
  number is the leaderboard metric, and it is always shown with the recall it achieved. The
  headline uses warm envelopes (≥ 30 prior observations of the agent-tool key). 95 % CIs come from
  a 1,000-sample bootstrap over sessions.
- **Baselines (§5.5).** `random`; `static_threshold` (per-field train p99 bound plus a train
  allowlist of categorical values, no sequence); `isolation_forest` (batch fit on the train split
  only: it is not an online learner and cannot extrapolate above its train range);
  `llm_judge` (a frontier model scoring the last 8 envelopes of the session, on a stratified
  2,000-envelope sample).

**What the per-agent FPR@95 operating point is not.** It is fixed after the fact using the test
labels: it asks "how many benign calls would be flagged if the thresholds were set just low
enough to catch 95 % of attack envelopes". The engine's live thresholds are much higher (target
FPR 0.5 %), and their recall is the Tier-1 gate (§6 below).

## 4. Datasets

| dataset | status | what it is |
|---|---|---|
| **SW1** (synthetic) | used for every number here | Generator seed 0, all 20 personas, 90 days, all 8 attack families, attack rate 0.015: 42,289 sessions, 174,878 envelopes, 599 attacks. Dataset sha256 `e7b3bc2a2f20…` (traces.jsonl). Rebuilt from a clean checkout by `scripts/build_leaderboard.py`. |
| **sandbox dataset v1** | **not built** | LLM agents (one model) running ~70 task templates against sandboxed Stripe test mode, a Salesforce Developer org, a Slack sandbox channel, a local Zendesk-like server and Mailhog, with prompt- and action-level injections. Minimums: 20 agents, 2,000 sessions, 40,000 actions, 600 attacks. The README's "How to build dataset/v1" gives the commands; they call a paid LLM and live sandbox APIs and are run by hand. |

The eight families are `magnitude_spike` (m log-uniform on [3, 100]), `scope_blowup`,
`exfil_volume`, `prereq_skip`, `novel_target`, `velocity_burst`, `slow_drift`, and the
`benign_unusual` control set (benign but unusual calls, used for the control FPR).

## 5. Results on SW1 (spec protocol, default engine)

The full leaderboard, both splits and the per-family tables, is
[docs/leaderboard.md](leaderboard.md). Every row there is the **synthetic benchmark (SW1)**, full
test split, spec protocol. Warm test rows; 95 % CIs from 1,000 session-bootstrap samples.

**By time** (53,011 warm test envelopes, 2,062 attack envelopes):

| detector | AUPRC [95 % CI] | FPR@95 per-agent [95 % CI] (recall achieved) | FPR@95 global |
|---|---|---|---|
| engine | 0.576 [0.480, 0.658] | 0.563 [0.497, 0.583] (0.950) | 0.536 |
| isolation_forest (batch) | 0.055 [0.042, 0.074] | 0.865 [0.636, 0.925] (0.972) | 1.000 |
| random | 0.040 [0.031, 0.050] | 0.943 [0.935, 0.958] (0.950) | 0.944 |
| static_threshold (batch) | 0.048 [0.035, 0.066] | 0.950 [0.794, 0.990] (0.975) | 1.000 |
| llm_judge | not run (requires a live judge model) | | |

**By agent** (6 held-out personas; 43,562 warm test envelopes, 1,611 attack envelopes):

| detector | AUPRC [95 % CI] | FPR@95 per-agent [95 % CI] (recall achieved) | FPR@95 global |
|---|---|---|---|
| engine | 0.615 [0.504, 0.693] | 0.704 [0.661, 0.735] (0.950) | 0.704 |
| random | 0.036 [0.027, 0.046] | 0.948 [0.937, 0.960] (0.950) | 0.948 |
| static_threshold (batch) | 0.057 [0.037, 0.083] | 1.000 [1.000, 1.000] (1.000) | 1.000 |
| isolation_forest (batch) | 0.048 [0.037, 0.063] | 1.000 [0.587, 1.000] (1.000) | 1.000 |
| llm_judge | not run (requires a live judge model) | | |

How to read these numbers:

- **The engine ranks attacks far better than the baselines** (AUPRC 0.58 against 0.040–0.055 by
  time; prevalence is 3.9 % of warm test envelopes, which is roughly what random scores).
  The baselines' AUPRCs (0.036–0.057 across both splits, random included) have overlapping CIs:
  `static_threshold` and `isolation_forest` barely beat random. The
  engine's AUPRC CI is wide ([0.48, 0.66]): the bootstrap resamples sessions, and 50 velocity
  bursts carry 1,699 of the 2,062 attack envelopes.
- **Baselines reach 95 % recall only by flagging nearly everything.** `static_threshold` and
  `isolation_forest` scores have large tied groups (discrete bounds; clipped or zero scores), so
  their recall jumps past 0.95 (0.972–1.000) and their FPR@95 is 0.87–1.00; by agent both flag
  every benign call. Isolation forest is a batch
  fit on the train split and cannot score extrapolation above the train range, which the spec
  reports as its caveat.
- **No detector is usable at 95 % envelope recall.** The engine's per-agent FPR@95 of 0.563 means
  flagging more than half of benign calls. The metric is dominated by `velocity_burst` (1,699 of
  the 2,062 attack envelopes). Attack-level recall at that operating point is 0.895–1.000 per
  family (leaderboard per-family table), but that is bought with the same
  56 % FPR and is not the gate.
- **At the engine's own live thresholds** (target FPR 0.5 %) benign FPR is 0.39 % by time and
  0.73 % by agent; family recall is in §6.
- **By agent.** Held-out personas have no train-split scores of their own, so the leaderboard's
  per-agent thresholds fall back to the pooled train ECDF; this raises FPR@95 to 0.704. The engine
  itself still learns each held-out agent online during the test period, so the by-agent split
  tests the pooled priors only for the first observations of each key. At live thresholds the
  by-agent gate cells are: magnitude_spike (m ≥ 10) 0.450 (9/20), scope_blowup 0.842 (16/19),
  exfil_volume 0.800 (8/10), prereq_skip 0.929 (13/14), novel_target 0.194 (7/36),
  velocity_burst 0.944 (34/36); benign FPR 0.73 %, control FPR 5.8 %.
- **Drift and cold start (by time, per-agent operating point).** All 11 `slow_drift` attacks are
  caught, with a median delay of 3 sessions and p90 of 8. Every warm test envelope has ≥ 30 key
  observations by construction; the 230 envelopes with 30–100 observations show recall 0.982 at
  FPR 0.560, no worse than the rest.
- **Latency.** The Tier-1 perf test pins p99 < 200 µs per score on warm keys. The latencies in
  these leaderboard runs (engine p99 ≈ 1.7 ms) were measured with four replays and the test suite
  running in parallel and are not a latency result.

## 6. Tier-1 status (honest)

**The Tier-1 gate is not met under the spec protocol.** The gate (spec §9, D-008/D-009) asks for
attack-level recall ≥ 0.90 per family for the targeted detector, at the engine's *live* per-agent
thresholds, on warm test rows, with benign FPR ≤ 2 %.

Default engine, spec protocol (allow-all replay), SW1 seed 0, from this leaderboard's engine run
(identical to the Tier-1 acceptance run):

| family (gate bucket) | targeted detector | attack recall at live thresholds | top-reason share |
|---|---|---|---|
| exfil_volume | magnitude | 0.579 (11/19) | 0.727 |
| magnitude_spike (m ∈ [10, 100]) | magnitude | 0.538 (7/13) | 1.000 |
| novel_target | novelty | 0.388 (19/49) | 1.000 |
| prereq_skip | sequence | 0.933 (14/15) | 1.000 |
| scope_blowup | magnitude | 1.000 (13/13) | 0.846 |
| velocity_burst | velocity | 0.700 (35/50) | 0.257 |

Benign FPR 0.39 %, control FPR 6.2 %, minimum family recall 0.388: **gate FAIL**. Only prereq_skip
and scope_blowup pass. `tests/acceptance/test_tier1_exit.py::test_tier1_exit` encodes the gate as a
strict xfail; it will fail loudly (XPASS) if the gate ever starts to hold.

**Root causes (decision 008).** Of 121 misses on seeds 0 and 1:

- **91: the allow-all replay teaches the engine its own attacks.** The spec replay allows every
  call, so each agent's calibrator learns the attack rows into its benign tail (on some agents
  0.7–13 % of rows are attacks: velocity duplicates, scope and novelty hits near 1.0). 16–17 of
  20 agents end with θ ≥ 0.999, and only fused scores of about 1.0 are flagged. Detector state
  learns attacks too (an exfil read's inflated item counts end up in the baseline).
- **21: detector or design limits.** Magnitude spikes of ×12–×24 that are only z ≈ 3.1–3.2 on a
  wide-spread field squash to 0.55–0.58; velocity bursts of 21–41 calls on an agent whose window
  median is 3 score 0.60–0.89; the novelty ramp `min(1, n/200)` discounts stable vocabularies with
  81–178 observations.
- **6: label semantics** (the exfil response signal is carried by the call after the read; the
  generator now labels it, per §2.3).
- **3: not anomalous** (the "attack" value is inside the persona's own benign p99.5).

**Opt-in candidates, clearly labelled optimistic.** Decision 008 measured protocol and engine
switches; none is a default. The best one replays under the
**oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic)**: every
flagged call is approved if benign and denied if an attack, so the calibrator only learns benign
calls. With three opt-in engine switches (`vel_count_attempts`, `seq_prereq_graded`,
`novelty_full_n=50`) it reaches, on seed 0 / seed 1:

| family | default, spec protocol | opt-in candidate, oracle-reviewer loop (optimistic) |
|---|---|---|
| magnitude_spike (m ≥ 10) | 0.538 / 0.625 | 0.692 / 0.750 |
| scope_blowup | 1.000 / 0.750 | 1.000 / 1.000 |
| exfil_volume | 0.579 / 0.621 | 1.000 / 0.966 |
| prereq_skip | 0.933 / 1.000 | 1.000 / 1.000 |
| novel_target | 0.388 / 0.417 | 1.000 / 1.000 |
| velocity_burst | 0.700 / 0.635 | 0.880 / 0.962 |
| benign FPR | 0.39 % / 0.56 % | 0.93 % / 0.89 % |
| control FPR | 6.2 % / 4.9 % | 13.8 % / 8.7 % |

Even this optimistic candidate fails the gate (magnitude_spike 0.69 / 0.75, velocity_burst 0.88 on
seed 0), and the targeted detector is the *top* reason in only 0.30 / 0.22 of detected velocity
bursts and 0.58 / 0.64 of exfil attacks (it is *among* the reasons in 0.89–1.00).

The label-free alternatives do not fix it either (decision 008, "The censoring ratchet"; seed 0 /
seed 1):

- `enforce` and `drop` truncate the calibrator's input at θ, so θ ratchets down (median end
  0.33–0.44) and benign FPR is 36.7 / 52.7 % and 37.6 / 31.9 %.
- `winsorize` freezes θ at θ0 = 0.600, with benign FPR 13.5 / 13.5 %.
- `winsorize` after 500 scores (shadow first) starts from the saturated allow-all θ: benign FPR
  is 5.2 / 5.3 %, but min recall is 0.69 / 0.69 and 5–6 agents stay saturated. Adding `enforce`
  to it raises benign FPR to 31.1 / 31.2 %.

## 7. Failure cases

All numbers here are from the default engine on SW1 (by-time split, warm test rows, the engine's
live thresholds), read from `results/synth-v1/time/engine/scores.parquet`.

- **Saturated thresholds hide near-certain anomalies.** The median live threshold is ≥ 0.9989 on
  17 of 20 agents (1.0 on 7). Missed `novel_target` attacks score 0.9998–0.9999 against thresholds
  of 0.99995–1.0: the engine ranks them at the very top, yet the calibrator, having learned earlier
  attacks as benign under allow-all replay, sets the bar just above them. 30 of 49 warm
  `novel_target` attacks are missed this way or by the novelty ramp.
- **Velocity bursts on noisy agents.** On `slack-notifier-a` and `helpdesk-triage-b`, whose
  thresholds are lower (0.967–0.973), the closest missed bursts score 0.92–0.96. On
  helpdesk-triage-b, whose median window holds about 3 calls, a burst needs about 46 calls to
  reach 0.90 (decision 008); its bursts have 21–41.
- **Magnitude spikes on wide fields.** A ×12 change of `due_in_days` (exec-assistant-a) is
  z ≈ 3.1 on that field's spread and scores 0.54; a ×22 refund amount on sales-ops-a is still
  exceeded by 1.7 % of that persona's own benign amounts. The second is arguably not an anomaly.
- **False positives concentrate on one agent and one detector.** 200 of 50,949 warm benign calls
  are flagged (0.39 %); 88 of them are on `support-refunds-c`, 81 of those fired by the sequence
  detector, mostly on `stripe.create_refund`. Across agents the top detector of a false positive
  is sequence 136 times, velocity 38, magnitude 20, novelty 6.
- **The control set is where the FPR lives.** `benign_unusual` calls (benign by construction, but
  unusual) are 42 of the 200 false positives and give a 6.2 % control FPR at live thresholds,
  above the Tier-3 target of 5 %; 35 of the 42 are velocity.
- **At the leaderboard's operating point the engine is not deployable.** Forcing 95 % envelope
  recall costs a per-agent FPR of 0.563. Even at that point `slow_drift` has the lowest envelope
  recall (0.674, 147 of 218 envelopes), although every drift attack is eventually caught (median
  delay 3 sessions, p90 8).
- **Attribution.** For detected velocity bursts, velocity is the top reason in only 26 % of
  cases; sequence fires first, because the duplicated write's lookup scrolls out of the
  four-call `prior_tools` window.

## 8. Limitations

- **Synthetic company.** SW1 is one generated company: 20 personas, a fixed tool catalog,
  hand-written task grammars and attack generators written by the same people who wrote the
  detectors. Attack families match the detectors by construction; real attacks need not.
- **Single model family.** The sandbox dataset (when built) uses one agent model; results may not
  transfer to other models, tools or organizations.
- **Retimed clock.** Sandbox sessions are placed on a simulated calendar (D-015): session starts
  are synthetic and only in-session gaps (clamped 0.5–120 s) and burst gaps are real, so velocity
  results on the sandbox data depend on the retiming. SW1 timestamps are fully synthetic.
- **Small per-family attack counts.** Gate families have 8–52 warm test attacks per seed
  (magnitude_spike m ≥ 10: 13 and 8), so one attack moves recall by 2–12 points; CIs on per-family
  recall are wide and are not reported.
- **Attack prevalence** (1.5 % of sessions) is far above any real base rate, which flatters
  AUPRC; FPR is prevalence-free but the per-agent operating point is set on test labels.
- **Allow-all replay.** The protocol measures detection without enforcement, and that is exactly
  what makes the live thresholds saturate (§6). The deployable behaviour lies between the
  allow-all and the oracle-reviewer numbers and depends on a real reviewer's error rate, which is
  not measured.
- **Batch baseline caveat.** `isolation_forest` is fit offline on the train split, not online.
- **LLM judge not evaluated.** It needs a live model; the code path is tested only with a scripted
  client, so no judge numbers exist.
- **Envelope-level AUPRC is dominated by `velocity_burst`** (10–60 positive envelopes per attack,
  against one for most families), so engine ablations do not order monotonically (decision 007).

## 9. Reproduce

```
uv sync --all-groups
uv run python scripts/build_leaderboard.py   # SW1 + 8 replays + docs/leaderboard.md (~1 h 45 min)
scripts/gate.sh --full                              # includes tests/acceptance/test_tier1_exit.py
```
