# agentfw — Feature Spec: Trajectory-Aware Monitoring

**Source of inspiration:** Barazandeh, Swanson, Kulkarni, Mungel. *OnTrack: Real-Time Monitoring and Intervention in LLM Agent Trajectories via Streaming Structure-Aware Optimal Transport.* arXiv:2610.12375 (Cribl AI Research Lab, 8 Oct 2026). Prior work: *OTAP*, arXiv:2607.17082.

**Status:** proposal, layered on top of `agentfw-spec-v3.md`. Section numbers like §6.6 refer to v3; "Paper §4.5" refers to OnTrack.

---

## 0. Summary: what to take, what to leave

OnTrack and agentfw solve adjacent problems from opposite directions. OnTrack asks *"is this run still following a known-good plan?"* and answers with optimal transport between the agent's growing dependency graph and reference runs. agentfw asks *"is this action an outlier for this agent?"* and answers with per-field statistics. The two are complementary: OnTrack's core (L2 alignment) is the expensive, reference-dependent part; its periphery (vital signs, pre-execution gate, verdict lifecycle, evaluation discipline) is cheap, reference-free, and directly transplantable.

The paper's own results tell us how to weight the borrowing. On real SWE-agent traces, loops and stalls were caught entirely by the reference-free heuristics (Paper Table 4, every variant 30/30); the optimal-transport structure term changed the *false-alarm operating point* rather than recall (Paper §6.1, §6.3 observation 3, §7); in-distribution anomalies sat near the detection floor (Paper §6.4); and trace length was the single best early signal (Paper Table 5, k=5). The conclusion for agentfw: take the ten features below, and treat reference alignment (F9) as an optional late-tier experiment rather than a foundation.

| # | Feature | Paper source | agentfw tier |
|---|---|---|---|
| F1 | Artifact ledger and provisional dependency graph | §3 Def. 1, metadata grades; §4.4 | 1 |
| F2 | Vital-signs detector family (loop, spin, stall) | §4 architecture L1; Table 1; §6.4 | 1 |
| F3 | MCP tool annotations as the "schemas-only" regime | §3 information regimes; Fig. 2; §4.5 L3 | 1 |
| F4 | Pre-execution gate with fresh-state recompute and allow-side tests | §4.5 L3; §6.1 gate bug; Table 3 | 3–4 |
| F5 | Two-tier escalation before any intervention | §4.3 step (5); Alg. 1 line 9 | 3–4 |
| F6 | Provisional flags: grace TTL, exploration budget, age discount | §4.4 Eq. 10; §4.5 lifecycle; Prop. 3 | 2–3 |
| F7 | Session-level verdict engine (severe-flag density) | §1 S4; §6.3 abort policy, Table 6, rival policies | 3 |
| F8 | Verdict authority scaled by evidence maturity | §4.6; Fig. 2 provenance strip | 3 |
| F9 | Mined session templates (L2-lite), optional | §4.1–4.3, §4.6; Table 4; §7 | 5+ |
| F10 | Evaluation-protocol upgrades | §6.2, §6.4, §6.5, §6.6, Appendix A | 0–2 |

---

## 1. The paper in brief (the parts that matter here)

**Problem framing (Paper §1, §3).** An agent run is modeled as a growing DAG: one node per tool call, an edge when a step consumes an artifact another step produced (Def. 1). Dependency metadata comes in three grades — declared parents, logged consumed/produced identifiers, or nothing, in which case the monitor reconstructs edges by *ledger lookup*: tool schemas say which arguments are consumed artifacts, and a ledger maps each artifact identifier to the step that first surfaced it. The authors are explicit that this extraction is deterministic string matching and fails when the dependency never appears as a matchable identifier (their `sed … $(find …)` example).

**Three-layer architecture (Paper §4, "Architecture of the monitor"; Fig. 2).** L1 *vital signs* (loop, information gain, stall) need nothing and always run. L2 *transport alignment* needs reference runs. L3 *pre-execution policy gate* needs tool schemas, runs synchronously, and only for irreversible calls. Capabilities switch off as inputs disappear; nothing is reconfigured.

**Four streaming challenges (Paper §1, S1–S4).** S1: a prefix must not be penalized for reference steps it hasn't reached yet (solved by frontier-masked marginals, §4.2). S2: a new step's structural role is unknown until something consumes its output (solved by age weights and retroactive edges, §4.4). S3: per-event compute must be far below tool latency (solved by one warm-started solver iteration per event, converged solve only before intervention, §4.3). S4: an aggregate score cannot drive decisions because its meaning changes with trajectory length; decisions must come from per-step signals with a grace window (§4.5).

**Signals and lifecycle (Paper Table 1, §4.5).** Per-event signals: node leakage (unmatched transport mass), match cost, coverage velocity, loop score (near-duplicate predecessor with a retry exemption for errored steps), information gain (new call, no new state), strategy switch. A flagged new node enters EXPLORING with a TTL of 3 events; it clears if a later step consumes its artifact, else it is classified at TTL expiry; cumulative unresolved exploration above a 25% budget escalates to OFF_TRACK. The L3 gate clears an irreversible proposal only if every artifact its schema requires exists in the live graph with a producing node (plus a reference-match check when references exist); reversible actions are never blocked.

**Verdict authority (Paper §4.6).** Gold or mined references may intervene; axiomatic references built from tool contracts only warn; planner-generated sketches only inform.

**Evaluation findings that constrain design.**
- The gate bug (§6.1, Table 3): the first gate implementation read a frontier cached one event stale and blocked 50/50 valid irreversible calls while the block-side test kept passing. Fix: recompute from current state inside the gate. Lesson stated by the authors: a block-only test suite would never have found it.
- Abort policy (§6.3, Table 6): aborting at the first flag has a 99% false-stop rate; restricting to severe flags and triggering on flag *density* over a rolling 6-step window gives ~18% compute saved at ~20% balanced false stops, ~83% precision at the corpus base rate. Against rival cutoffs, flag density did not dominate the savings frontier; what it won on was threshold stability (a cosine cutoff swung from 3% to 95% false stops across a 0.15 threshold move) and actionability (a verdict with a reason supports graduated responses).
- Length confound (§6.6): once resolved/unresolved traces are length-matched, every trajectory-level signal collapses to chance. Their score is length-sensitive by construction and they frame it as the mechanism that catches runaway runs, not a defect.
- Observations belong in signals, not in the cost (§6.6 design insight): blending observation embeddings into the transport cost degraded ablation AUROC from 0.825 to 0.715 because outputs are high-variance; the same information is useful downstream as a monitoring signal.
- Calibration constant depends on the encoder (§6.5): λ had to be recalibrated 3–10× lower than the batch default and differed between synthetic and MiniLM embeddings.

---

## 2. Feature specifications

Each feature states the paper mechanism, how it maps onto agentfw's engine, the state and data-flow changes, acceptance tests, and placement.

### F1. Artifact ledger and provisional dependency graph

**Paper mechanism.** Def. 1 and the three metadata grades (§3); the ledger lookup that recovers `(producer_step, consumer_step)` edges from identifiers; retroactive edge insertion and ancestor repair (§4.4).

**Why agentfw needs it.** v3 §6.6(c) implements id flow as a per-session *set* of seen ids. That answers "was this id seen before?" but not "which step produced it, from which tool, how many steps ago, and has anything consumed it since?" Those answers are what F2 (stall), F4 (gate), and F6 (branch validation) need.

**Design.**
- Replace `session_ids_seen` with a per-session **ledger**: `artifact_id → {producer_seq, producer_tool, consumers: [seq], first_seen_source ∈ {args, response, task_input}}`.
- Edge inference on every call: for each id-typed arg value, look it up; if found, record an edge `(producer_seq → this_seq)` and append to `consumers`. Ids that first appear in a *response* are recorded as produced by this step. This is the "partial metadata" grade in the paper's terms; MCP gives us responses, so we are never in the "no metadata" grade for ids.
- **Seed the ledger with task inputs** (new). Before the first call, extract id-like tokens from the session's initiating prompt/task payload and record them as produced at `seq=0, source=task_input`. This is the agentfw analogue of the paper's *premature-valid* distinction (§4.2(c)): a write whose id was supplied by the user is not a causal inversion. Without this seed, every session whose task says "refund charge ch_123" would fire id-flow on the first write.
- Keep the v3 `r_flow` statistic but compute it against the ledger: the historical rate at which this (agent, tool, path)'s value had a producer in the ledger. Report in evidence *which* grade the producer came from.
- Provisional structure: a produced artifact that is never consumed within `h_age = 3` later calls is marked `orphan_candidate` (feeds F6, not scored on its own).
- Bound: 2,000 ledger entries per session; overflow degrades to a Bloom filter for membership and drops producer detail (log the degradation).

**State.** `sess:{session}:ledger` hash; TTL = session idle timeout. Size ≤ ~60 KB worst case.

**Acceptance.**
- Synthetic grammar with `source:` links yields the expected edge set with ≥ 99% precision/recall on benign sessions.
- A `prereq_skip` attack that fabricates an id scores `s_flow = r_flow`; the same write with the id present in the task prompt scores 0.
- Ledger replay is deterministic and `state_dump/load` round-trips it.

### F2. Vital-signs detector family (L1)

**Paper mechanism.** The reference-free L1 layer: loop score (near-duplicate predecessor on action/tool/args, retry exemption up to `r_max` for duplicates of an errored step), information gain (`1 − max cosine` to prior effect embeddings; spinning = new calls, no new state), stall (coverage velocity ≈ 0). The authors attribute all loop and stall recall on real traces to this layer (Table 4), and cost-limit traces are detected at 0.979 AUROC by flags (Table 7). The severe-flag trigger fired on 90% of cost-limit traces with ~41% of steps still ahead (§6.4).

**Why agentfw needs it.** v3 covers money/scope/novelty/order outliers. It does not cover the runaway-cost failure mode (an agent re-running the same query 200 times) or the "busy but stuck" mode. Both are tool-agnostic and zero-config, which is the whole thesis.

**Design (no embeddings; fingerprints only).**
- `call_fp = hash(tool_id, canonical(flat_args))` where `canonical` sorts keys, drops freetext fields, and *does not* normalize ids (a loop over different ids is a scan, not a loop — that is the velocity detector's job).
- `resp_fp` = 64-permutation MinHash over the shingled response text (bounded to the first 4 KB) combined with `response_meta` (item_count bucket, error flag). Computed by the proxy; the engine never sees raw response content (keeps F10's "observations in signals, not cost" discipline and the privacy posture).
- **Loop score** `s_loop`: 1 if `call_fp` equals any of the last `w_loop = 8` fingerprints in the session; else Jaccard of arg-token sets with the nearest prior call if ≥ 0.9 → 0.7. Retry exemption: if the matched prior call had `response_meta.error = true`, exempt up to `r_max = 3` consecutive duplicates, then score as a loop.
- **Information gain** `ΔI = 1 − max_{i<t} J(resp_fp_t, resp_fp_i)` (MinHash Jaccard estimate). **Spin score** `s_spin = 1` if the last `w_spin = 4` calls each had `ΔI < 0.1` and none errored.
- **Stall score** `s_stall`: coverage-velocity analogue without references. Define per-session progress as the count of *new* ledger artifacts produced (F1). `s_stall = 1` if ≥ `w_stall = 6` consecutive calls produced no new artifact and included no write-kind tool; 0.5 if the window had only reads with `ΔI < 0.3`.
- These three feed fusion as a fourth detector group (`vital`). They are *severe-flag eligible* for F7 but, by design, **never block a single call on their own** (a loop is a cost problem; blocking one iteration of it is pointless). Their enforcement path is F7's session-level `halt_session`.

**State.** Per-session ring buffers: 8 `call_fp` (8×8 B), 32 `resp_fp` MinHash sketches (32×64×4 B ≈ 8 KB), last-artifact counter. No per-agent state.

**Acceptance.**
- Synthetic loop injection (duplicate a read `r ∈ [5, 60]` times): `s_loop` fires at the 2nd duplicate (latency ≤ 1 event after onset, matching Paper Table 2 scenario C's latency 0 in spirit).
- Retry scenario (duplicate after an errored call): silence for ≤ 3 retries, loop on the 4th (Paper scenario F).
- Spin injection (varying args, identical results): `s_spin` within `w_spin` events.
- Benign sessions: loop/spin/stall false-flag rate ≤ 1% of sessions on the bench's benign split.

### F3. MCP tool annotations as the "schemas-only" regime

**Paper mechanism.** Capabilities are gated by available inputs (Fig. 2): with schemas but no references, the monitor still blocks irreversible actions whose declared prerequisites are missing (§3 "Information regimes"; §4.5 L3). Tools are classified reversible/irreversible from schema annotations; gate coverage is bounded by those annotations.

**Why agentfw needs it.** MCP already ships the schema layer the paper assumes: `tools/list` returns `inputSchema` plus `annotations` (`readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`). v3's blast-radius model (§6.9) infers write-kind structurally; it should consume these hints directly, and the proxy should treat "no annotations" as a distinct, degraded regime rather than silently inferring.

**Design.**
- Proxy captures `tools/list` per server and stores `schema:{server}:{tool}` = `{inputSchema, annotations, captured_at, hash}`. Re-captured on reconnect; a changed hash is logged (tool-definition drift is itself a signal, though out of scope for scoring here).
- Blast-radius structural source (v3 §6.9) gains annotation terms: `destructiveHint=true → +0.4`; `readOnlyHint=true → −0.4` (floor 0.05); `idempotentHint=false ∧ not readOnly → +0.1`; `openWorldHint=true → +0.1` (external side effects). Annotations are server-supplied and untrusted: **they may raise blast radius but may only lower it when corroborated** by the structural inference (tool returns item counts, no id-list args). A server that labels `delete_records` as read-only does not get a discount.
- Irreversible classification for F4: `irreversible = destructiveHint ∨ (inferred write-kind ∧ blast_radius ≥ 0.6)`.
- Regime flag on every verdict: `regime ∈ {history+schemas, schemas_only, bare}` so reasons and the dashboard say what the decision was based on, mirroring Fig. 2.

**Acceptance.**
- Bench adds annotated variants of every sandbox tool; the engine's blast-radius ranking of tools correlates (Spearman ≥ 0.8) with the hand-labeled irreversibility set from v3 §6.9.
- Adversarial annotation test: mislabeling a destructive tool as read-only does not reduce its effective blast radius below the structural inference.

### F4. Pre-execution gate with fresh-state recompute and allow-side tests

**Paper mechanism.** L3 is the only synchronous component; it evaluates the *proposed* action before the tool executes, only for irreversible calls, so worst-case latency is confined to the rare actions that warrant it (§4). The gate's first implementation read a cached frontier one event stale and blocked every valid irreversible call (§6.1, G-a); the fix recomputes from current state inside the gate. Table 3 adds a false-block sweep over 50 benign runs ending in a valid irreversible call, and the authors call out that a block-only suite would never have caught the bug.

**Why agentfw needs it.** v3 §2.2 already scores before executing, but it scores every call the same way. The paper's separation — cheap path always, strict synchronous gate only for irreversible calls — is a better latency budget and a cleaner place to hang F5's escalation.

**Design.**
- Pipeline split: `fast_path(e)` runs on every call (v3 §6.1–6.8, plus F2). If `irreversible(e)` (F3), run `gate(e)`:
  1. **Recompute, don't read cached**: re-read the session ledger and the per-key state in the same Redis transaction as the decision (v3 §7 atomicity) — never from the previous event's snapshot.
  2. Prerequisite check: every id-typed arg marked `consumed` (schema or inferred) must have a ledger producer (F1), or the gate returns `BLOCK` with `prereq_missing` naming the id and the tool that normally produces it (from mined prerequisites, v3 §6.6(b)).
  3. Distribution check: `fused ≥ θ_eff` (blast-radius-modulated, v3 §6.9) → candidate block → F5 escalation.
  4. Reversible calls skip the gate entirely and can never receive `block` from the engine (policy/Cedar can still deny). This is the paper's "reversible actions are never blocked, preserving cheap exploration."
- Latency: gate adds one Redis round trip and, on escalation only, F5's cost.

**Acceptance (both directions, per Paper Table 3).**
- G-a analogue: 200 benign sessions ending in a valid irreversible call (ids produced earlier in-session or seeded from task input) → 0 false blocks (one-sided 95% upper bound ≤ 1.5%).
- G-b analogue: missing producer → block.
- G-c analogue: schema present, distribution anomalous → escalates (F5) rather than blocking from the fast path.
- G-d analogue: `regime = bare` → gate degrades to distribution check only and the verdict says so.
- Stale-state regression test: a deliberately stale snapshot must *fail* this suite (proves the test can see the bug class).

### F5. Two-tier escalation before any intervention

**Paper mechanism.** The fast path runs one warm-started solver iteration per event; if and only if it raises a would-be intervention, the evaluator escalates to a converged solve before confirming. No intervention is ever taken on a stale linearization, and the expensive solve is paid exactly when it is cheapest relative to the tool call about to be prevented (§4.3 step 5; Alg. 1 line 9).

**Why agentfw needs it.** v3 §6.8 goes from fused score straight to a decision. An escalation tier lets us keep the fast path cheap and approximate while making the *confirmed* block as accurate as we can afford.

**Design.** On a candidate block or candidate approval from F4:
1. **Full re-score** with pooled priors (v3 §6.10) and the long velocity window, both of which the fast path may skip on cold keys.
2. **Cross-detector consistency**: recompute fusion with the Tier-5 learned fusion if available, else v1.
3. **Optional LLM-judge** (feature-flagged, off by default; ≤ 1.5 s budget): the last 8 envelopes of the session plus the structured reasons, asked for a 0–1 suspicion score. Its result is a *tie-breaker within the approval band only* — it can move `require_approval` to `allow` or `block`, never flip a confident fast-path block to allow. This keeps the paper's warning about judge circularity (§4.6, planner-generated references "may inform but never trigger an automatic halt") in force.
4. Confirmed verdict is written with `escalated = true` and both scores (fast, escalated) in the audit row.

**Acceptance.**
- Escalation rate on benign bench traffic ≤ 2% of calls; p99 escalated latency ≤ 50 ms without the judge, ≤ 1.5 s with.
- Escalation never produces a *more* permissive decision than the fast path except inside the approval band (property test).
- On the bench, escalation lowers control FPR (v3 §5.4) without reducing recall by more than 1 point per family.

### F6. Provisional flags: grace TTL, exploration budget, age discount

**Paper mechanism.** New nodes enter EXPLORING with a grace TTL of 3 events; the flag clears if a later step consumes the node's artifact; on expiry the step is classified; unresolved exploration mass above a 25% budget escalates to OFF_TRACK; dead-end branches abandoned after one or two steps land in ABANDONED at bounded cost (§4.5). Age weights ramp a node's structural influence from 0 to 1 over `h_age = 3` events unless its artifact is consumed earlier (§4.4, Eq. 10). Prop. 3 bounds the cost of a redundant leaf step by λ/m̄.

**Why agentfw needs it.** The benign-but-unusual controls in v3 §3.3 are exactly the cases this lifecycle protects: a novel read, an exploratory lookup, a one-off query. v3 scores each call independently and has no notion of "wait and see" for reversible actions.

**Design.**
- Applies to **reversible** calls only (F3). Irreversible calls go through F4 regardless.
- A reversible call with `fused ≥ θ_eff` does not produce `block`; it produces `flag = PROVISIONAL` with `ttl = 3` events and is recorded in the session's `exploration` list with its score mass.
- Clear conditions: (a) any artifact this call produced is consumed by a later call (ledger, F1) → `VALIDATED`; (b) a later call in the same tool family scores benign with the same novel value (the novelty was a vocabulary shift, not an anomaly) → `ABSORBED`.
- On TTL expiry with neither: `UNRESOLVED`; its score mass is added to the session's unresolved-exploration total. If unresolved mass exceeds `B = 25%` of the session's total scored mass (over at least 8 calls), raise the severe flag `OFF_TRACK` (consumed by F7).
- **Age discount for sequence signals** (analogue of Eq. 10): the first call of a tool this agent has never used scores its transition surprise at weight `ω = 0` and ramps to 1 over the next 3 calls, unless an irreversible call consumes its output sooner. Magnitude and novelty are not discounted — they need no lookahead, mirroring the paper's "semantic and tool costs apply immediately."
- Tolerance band: a `PROVISIONAL` flag that resolves to `UNRESOLVED` after a one-call abandoned branch contributes at most `1/8` of a severe flag to F7 density (bounded cost, no alarm).

**Acceptance.**
- Control FPR on `benign_unusual` drops measurably versus v3 without F6 (target ≤ 3%).
- Hallucinated-step analogue (a novel read on a never-seen object followed by nothing) resolves to `UNRESOLVED` at exactly TTL expiry (latency 3, matching Paper Table 2 scenario D's grace semantics).
- Age discount never suppresses a `prereq_missing` or id-flow violation on an irreversible call.

### F7. Session-level verdict engine (severe-flag density)

**Paper mechanism.** S4: an aggregate score's meaning changes with trajectory length, so decisions come from per-step signals with a short grace window, never from the aggregate (§1, §4.5). The deployed abort policy triggers on *severe* flags only (LOOP, OFF_TRACK, CAUSAL_INVERSION, BLOCK, STALLED) when their density in a rolling 6-step window exceeds a threshold; a first-flag policy had 99% false stops (§6.3). Flag density degraded gracefully across thresholds 0.2–0.7 (29%→19% false stops) where a cosine cutoff was pathologically sensitive, and it emits a verdict with a reason that supports graduated responses — replan, inject guidance, escalate to a human, abort (§6.3 "Rival abort policies").

**Why agentfw needs it.** v3's `Verdict` is per call. There is no way to say "this session is going wrong" or to stop a runaway loop, and no graceful intermediate between allow and block.

**Design.**
- New session-level state: ring buffer of the last `W = 6` per-call flag sets. Severe flags: `BLOCK` (confirmed, F5), `PREREQ_MISSING`, `ID_FLOW_VIOLATION` on an irreversible call, `LOOP`, `STALL`, `OFF_TRACK` (F6), `VELOCITY_BURST`. Non-severe: `PROVISIONAL`, `NOVEL_READ`, `COLD_START`.
- `density_t = (Σ severe flags in window) / W`, computed per call after the fast path.
- Session verdict ladder, evaluated every call: `density ≥ θ_halt` → `HALT_SESSION`; `θ_warn ≤ density < θ_halt` → `WARN_SESSION`; else `OK`. Defaults `θ_warn = 0.33`, `θ_halt = 0.6` (the paper's illustrated point; recalibrated on our bench, see F10).
- **Graduated responses**, configurable per agent: `WARN_SESSION` injects a structured notice into the next tool result (the agent can read it: "3 of the last 6 calls repeated with no new results"); `HALT_SESSION` returns an error on the next call and requires a human or policy release. In shadow mode both are recorded only.
- Per-call `Verdict` gains `session_verdict`, `density`, `severe_flags_window`. Audit stores both.
- The aggregate fused score over the session is **reported, never thresholded** — exactly the paper's rule.

**Acceptance.**
- Bench adds session-level labels (`session_label ∈ {benign, runaway, attacked}`) and reports: halt recall by session type, false-halt rate on benign sessions, median calls-from-onset to halt, and the threshold-sensitivity curve (false halts vs `θ_halt` over 0.2–0.8). Require monotone, gradual degradation — no cliff.
- First-flag policy is included as a baseline to reproduce the paper's negative result on our data.

### F8. Verdict authority scaled by evidence maturity

**Paper mechanism.** Reference provenance scales verdict authority: gold/mined → intervene; axiomatic (from tool contracts) → warn; generated → inform (§4.6; Fig. 2 strip). With no references, L2 is off and the system degrades explicitly to L1 + L3.

**Why agentfw needs it.** v3 §6.8 blends thresholds during cold start, but the *kind* of decision allowed is not tied to how much the engine knows. A block on a key with 12 observations and a pooled prior should not carry the same authority as one on 10,000 observations.

**Design.**
- Define per-decision `maturity ∈ {mature, warm, cold, bare}` from the minimum `n` across the keys that contributed ≥ 0.3 to the fused score, with `prior_used` forcing at most `warm`, and `regime = bare` forcing `bare`.
- Authority ceiling: `mature → block`; `warm → require_approval`; `cold → warn` (notice in tool result, allow); `bare → inform` (audit only). Cedar deny rules are outside this ceiling (deterministic policy always may block).
- F4's prerequisite block is exempt from the ceiling when the regime includes schemas: a missing producer for a `destructiveHint` tool is an axiomatic fact, not a statistical one — the paper's L3 keeps blocking in the schemas-only regime for the same reason.
- Every `Reason` carries `maturity` and `regime` so the rendered reason reads like: "blocked (mature: 1,140 refunds); amount at 99.97th percentile; no prior retrieve_charge for ch_… (schemas: destructive)".

**Acceptance.**
- Cold-start curve (v3 §5.4) now reports decisions by maturity; no `block` is ever emitted from a `cold` or `bare` maturity in the bench logs (property test on the audit table).

### F9. Mined session templates (L2-lite) — optional, Tier 5+

**Paper mechanism.** L2 aligns the prefix DAG against reference runs with unbalanced fused Gromov–Wasserstein, one warm-started conditional-gradient iteration per event (§4.1–4.3), frontier-masked so the agent is never punished for being early (§4.2, Eq. 5–7), multi-reference by soft-min (§4.6). Mined references (clustered successful production traces) are allowed to intervene.

**Why this is optional for agentfw.** The paper's own evidence: the structure term tracked the linear unbalanced-OT variant within 0.003–0.010 AUROC (Table 5), the mask had a null ranking effect (§6.3), and the structural term's value was established only on controlled, structurally diverse tasks (§7). agentfw's thesis is per-field distributions; references are a second axis, not the first. We also have no reference runs on day one — but we accumulate benign sessions, which are exactly the "mined references" the paper admits.

**Design sketch (do not build before F1–F8 are measured).**
- Mine templates per agent: cluster benign sessions by tool-sequence shingles (k=3) + ledger edge sets; keep the medoid of each cluster with ≥ 20 members as a template DAG. This is the paper's "mined" provenance; it may intervene (F8).
- Align online with a cheap, prefix-aware method first: a frontier-masked edit distance over the tool sequence where only template steps whose template-parents are already matched are "active" (the S1 idea without transport). Report *leakage* per step = unmatched after the grace window (F6). Only if this measurably improves the operating point on the bench do we consider porting the FGW machinery (Apache-2.0 reference code exists; see §6 below).
- Hard rule inherited from §6.6: response content never enters the alignment cost; only tool id and id-normalized args.

**Acceptance (if built).** Must lower control FPR or raise recall on `prereq_skip`/`novel_target` by a bootstrap-significant margin over F1–F8 alone, at ≤ 1 ms p99 added fast-path latency. Otherwise it stays an experiment.

### F10. Evaluation-protocol upgrades

**Paper mechanisms.** Instance-disjoint partition into calibration / reference / evaluation with λ selected on calibration only and frozen (§6.2). Length-matched pairs to expose the length confound (§6.6). In-distribution splice injection (3-step segments from real failing traces into succeeding ones) as a deliberately hard family that sat near the floor (§6.4). Precision reported at the deployment base rate, not the balanced sample (§6.3, Table 6). Bootstrap CIs at n=1,000 with the unit of resampling stated (§6.2). Secondary multi-seed protocol (Appendix A). Honest attribution of which layer carried which detection (Table 4).

**Changes to v3 §5.**
- **Three-way split**: add a *calibration* partition (by agent, 15%) on which `z0, τ` (v3 §6.3), the prerequisite threshold (§6.6b), `θ_warn/θ_halt` (F7), and fusion weights are selected and then frozen; train/test as before. Report calibration-set values alongside test metrics.
- **Length-matched control**: pair each attacked session with a benign session of the same agent within ±2 calls; report session-level metrics on the matched set. If detection collapses on matched pairs, the engine is detecting length, and the writeup says so.
- **New attack family `splice`**: transplant a 3-call benign segment from a *different agent's* session (same tool vocabulary) into a session. Expected to be hard; it tests whether per-agent baselines carry cross-agent signal. Report it separately; do not fold into the headline.
- **Precision at base rate**: every operating point reports precision at the benchmark's attack base rate (≈1.5% of sessions) in addition to balanced metrics.
- **Attribution table**: for every detected attack, record which detector group (magnitude / novelty / sequence / vital / gate) carried it; publish the table. Removing a group must change exactly the rows it carried.
- **Recalibration discipline** (from §6.5): any change to the normalizer's transform (`log1p`, canonicalization, MinHash shingling) invalidates calibrated constants; CI fails if `report.json` carries constants calibrated against a different normalizer hash.
- **Response content stays out of the scoring core** (from §6.6): a CI grep forbids `response_text` in `core/detectors/`; only `response_meta` and `resp_fp` may cross.

---

## 3. Changes to existing v3 sections (diff list)

| v3 section | Change |
|---|---|
| §2.2 online path | Split into fast path (all calls) + gate (irreversible only) + escalation (candidate interventions only); add session verdict after fast path |
| §3 data model | `ActionEnvelope.session_ids_seen` → `ledger_refs` (list of `{id, producer_seq, source}`); add `call_fp`, `resp_fp`, `schema_hash`, `annotations`, `regime`; `Verdict` adds `session_verdict`, `density`, `maturity`, `escalated`, `fast_score`, `escalated_score`; new `Flag` enum with severity |
| §4.6 attack injector | Add `loop`, `spin`, `stall`, `splice`; add session-level labels |
| §5.3 splits | Add calibration partition; add length-matched set |
| §5.4 metrics | Add session-level metrics, precision at base rate, attribution table, threshold-sensitivity curve |
| §6.1 normalize | Ledger construction and task-input seeding; fingerprints computed by proxy |
| §6.6(c) id flow | Rewritten over the ledger with producer provenance |
| §6.8 calibration | Decision bands replaced by authority ceiling (F8) + bands |
| §6.9 blast radius | Annotation terms; irreversible classification exported to F4 |
| §7 state | `sess:` gains ledger, fingerprint rings, flag window, exploration list |
| §8 testing | Allow-side gate suite; stale-state regression; no-response-content grep |
| §9 tiers | F1–F3 into Tier 1; F6 into Tier 2–3; F4/F5/F7/F8 into Tier 3–4; F9 Tier 5+; F10 spread over 0–2 |

---

## 4. Risks introduced by these features

- **Ledger extraction misses** (the paper's binding constraint, §7 "Extraction"): dependencies that never surface as a matchable id (a shell command resolving its own target) are invisible. Mitigation: `r_flow` is learned per field, so agents whose writes routinely lack producers get low id-flow weight; `maturity` prevents blocking on thin evidence.
- **Fingerprint false loops**: idempotent polling (status checks) looks like a loop. Mitigation: `idempotentHint=true ∧ readOnlyHint=true` tools get `w_loop` raised to 32 and loop demoted to non-severe.
- **Session halts on long-running benign agents**: density over a 6-call window is length-invariant by design, but agents with legitimately repetitive phases will warn. Mitigation: per-agent `θ_halt` calibrated like per-agent thresholds; shadow mode default.
- **Annotation trust**: server-supplied hints are an attack surface. Mitigation in F3 (hints may raise, may lower only when corroborated).

---

## 5. Placement in the tiered plan

| Tier | Adds from this spec |
|---|---|
| 0 | F10 three-way split, attribution table scaffolding, response-content CI rule |
| 1 | F1 ledger + task-input seeding; F2 vital signs; F3 annotations capture + blast-radius terms; F10 loop/spin/stall families |
| 2 | F6 provisional flags; F10 splice family, length-matched set, session labels |
| 3 | F4 gate + allow-side suite; F5 escalation (no judge); F7 session verdicts in shadow; F8 authority ceiling |
| 4 | F4/F5/F7 in the Go proxy; graduated responses; judge behind a flag |
| 5+ | F9 mined templates experiment |

---

## 6. License and IP analysis

Not legal advice; a short consultation with a lawyer before commercial launch is cheap and worth it. The facts as they stand:

**The paper's text and figures: CC BY-NC-ND 4.0.** The arXiv abstract page for 2610.12375 shows the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 license. That license governs the *expression* — the prose, equations as typeset, figures, tables. It permits copying and redistribution with attribution for non-commercial purposes and forbids adapted versions and commercial use *of the paper itself*. For agentfw that means: don't paste the paper's text, figures, or tables into product docs, marketing, or a commercial README; paraphrase and cite instead (this spec does that). The license does not restrict building a product on the ideas, because copyright does not cover ideas, methods, or algorithms — it covers their expression. Independently writing code that implements a method described in a CC BY-NC-ND paper is not a derivative work of the paper.

**The reference code: Apache License 2.0.** Cribl released `ontrack.py` (pure NumPy) and the fault-injection tests on Hugging Face under `cribl-ai/ONTRACK`, license Apache 2.0; the OTAP code (`cribl-ai/OTAP`) is also Apache 2.0. Apache 2.0 permits commercial use, modification, and redistribution, requires keeping the license and copyright notices and a NOTICE file if one exists, and — importantly — includes an express patent license from each contributor covering claims that their contribution necessarily infringes (Apache 2.0 §3). That grant covers the code as contributed, not the ideas in general, and it terminates if you sue the licensor over patents in that work. If you port any of their code into the Go proxy or use their test scenarios, carry the attribution. If you implement from the paper's description without copying code, you owe nothing under Apache but should still cite.

**Patents are the real question, and they are independent of both licenses.** A company lab publishing does not mean it did not file. US law gives inventors a one-year grace period after their own publication to file, and applications publish 18 months after filing, so a filing on OnTrack or OTAP might not be visible until 2027–2028. A quick web search found nothing, but a web search is not a patent search. Do a proper one on Google Patents / USPTO for assignee "Cribl" and the four authors' names, and repeat it in a year. Two things reduce exposure regardless: agentfw's core (per-field streaming statistics, Good–Turing novelty, blast-radius-modulated thresholds) is a different method from OT alignment and would not read on claims about transport; and the parts this spec borrows (loop/stall heuristics, schema-based prerequisite gates, flag-density stopping rules) are generic and have prior art the paper itself cites, which makes them hard to claim narrowly. If a patent surfaces that reads on F9's alignment, F9 is the one feature to drop or design around.

**Trademark.** Don't name anything "OnTrack" or "OTAP". Cribl is an operating company with brand counsel.

**Norms.** Cite the paper in the README, the benchmark writeup, and the docstrings of F1/F2/F6/F7. Beyond courtesy, it strengthens the independent-development record and makes the product more credible to the exact audience you want (researchers and security engineers who read arXiv).

**Bottom line.** Yes, it is valid to take inspiration from this paper and build a product. The ideas are free to implement; the code is free to use with attribution; the prose and figures are not free to reproduce; patents are unknown and worth a real search. Everything in this spec is designed as an independent implementation of the *mechanisms*, in agentfw's own data model, with the paper cited as prior art.

---

## 7. Citation

Barazandeh, B., Swanson, C., Kulkarni, C., & Mungel, N. (2026). *OnTrack: Real-Time Monitoring and Intervention in LLM Agent Trajectories via Streaming Structure-Aware Optimal Transport.* arXiv:2610.12375. https://arxiv.org/abs/2610.12375 — paper CC BY-NC-ND 4.0; reference code Apache-2.0 at https://huggingface.co/cribl-ai/ONTRACK.

Barazandeh, B., Majumdar, S., & Michailidis, G. (2026). *OTAP: Structure-Aware Optimal Transport for Evaluating Planning and Execution in Agent Trajectories.* arXiv:2607.17082. Reference code Apache-2.0 at https://huggingface.co/cribl-ai/OTAP.
