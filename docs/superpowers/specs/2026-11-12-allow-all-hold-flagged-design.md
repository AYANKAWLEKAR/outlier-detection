# Design: hold what the engine flags (spec Amendment A2), for the allow-all gate

Date: 2026-11-12 (updated 2026-11-16). Status: draft for the user's approval. Approach picked by the
user: **C**.

## Intent and why

Under spec §5.2 allow-all every call is `ALLOWED`, so the engine learns the attacks it lets through.
That is also how the engine runs in a real shadow deployment with no reviewer: the sandbox proxy's
shadow detector updates every call with `Outcome.ALLOWED` (`agentfw/sandbox/proxy_py/proxy.py`). The
amended Tier-1 criterion (spec A1.7) passes on SW1 seeds 0-2 under the oracle-reviewer approval loop
(labels act as a perfect human reviewer → optimistic) and fails under spec §5.2 allow-all on all three.
The user wants the allow-all result fixed with a real detection change: the engine should stop treating
its own flags as normal traffic when nobody has confirmed them.

## What the measurement says (recon on SW1 seed 0, current engine, spec §5.2 allow-all)

Evidence: `.superpowers/sdd/allow-all-recon/report.md` (git-ignored scratch; the numbers that the
design relies on are re-measured by the plan and committed as evidence).

- 71 attack misses. **64 come from the calibrator**: each agent's threshold θ (q_{1-α}, α = 0.5 %) learns
  attack rows and ends ≥ 0.999 on 19 of 20 agents (2 of 20 under the oracle-reviewer loop).
  **7 come from detector state** learning attack values: the `__resp__.item_count` reference on
  helpdesk-triage-a (4 exfil misses) and the `amount` reference on support-refunds-c (3 magnitude misses).
- Velocity bursts drive the saturation: 171 bursts, 10-60 rows each (median 34), 80 % of all attack rows;
  82 % of burst rows sit at or above their agent's benign q_0.995. A label-reading calibrator that skips
  only burst rows leaves 2 of 20 agents saturated.
- Label-free de-duplication is not enough on its own: one sample per session leaves 17 of 20 saturated
  (attack sessions are 3.45 % of sessions, about 7 × α); one sample per burst leaves 6 of 20 saturated and
  still fails novel_target (42/49), exfil_volume (15/19) and prereq_skip (9/15). On 6 agents the
  non-burst attacks alone exceed α.
- Decision 008 §2 measured label-free censoring before: never learning flagged calls (`drop`, `enforce`)
  ratchets θ down to 0.05-0.44 (benign FPR 32-53 %); learning them at θ (`winsorize`) freezes θ at
  θ0 = 0.6 (13.5 %). Both started censoring at cold start, when θ0 = 0.6 sits below every agent's benign
  tail.

## Goals

1. Under spec §5.2 allow-all, the amended criterion (spec A1.7, unchanged: per-family attack recall
   ≥ 0.90, targeted detector among the reasons ≥ 0.90 of detected attacks, benign FPR ≤ 2 %) passes on
   SW1 seeds 0, 1, 2 and a new held-out seed 3.
2. Results under the oracle-reviewer approval loop are **bit-identical** to today (same score hashes).
3. Real engine behaviour: no label is read by the engine; the mechanism also runs in the sandbox proxy.

## Non-goals

- Changing the gate criterion, α, θ0, any A1 constant, the generator, or the datasets.
- Fixing control FPR (reported, not targeted), the A1.12 open items, or the Tier-2 exit.
- Replaying held calls into detector state later (no time travel; see Design).

## Anti-goals (a failure even if the gate passes)

- Any constant chosen by looking at seed 1, 2 or 3 results. Constants are fixed on seed 0 only, then the
  other seeds are measured once and reported as they come out.
- A benign FPR or control FPR jump hidden behind the pass: both are always reported next to the gate,
  under both protocols, with the pre-A2 engine as a companion row.
- A pass that comes from the engine never learning new benign behaviour (unbounded adaptation lag).
  The plan measures how long a new benign value keeps getting flagged.

## Constraints

- Spec A1.5 rules hold (both protocols always shown; magnitude cell only with its companions; never
  the bare words "gate PASS"; never "the Tier-1 gate met"). The verbatim protocol labels are used.
- Old behaviour stays selectable: `SPEC_SWITCHES` and `PRE_A1_SWITCHES` gain the switch in its off
  position, so `engine-spec-allow` (`c4042bc4…`) and the pre-A1 golden (`f6573c61…`) stay pinned.
- Determinism, mypy --strict, ruff, TDD; no absolute paths in committed docs (public repo).
- Bounded memory per agent (the engine targets many agents).

## Approaches considered

| | what | trade-off | verdict |
|---|---|---|---|
| A | Hold every call the engine flags under `ALLOWED`; learn it only once the same pattern has been flagged in several separate sessions on different days | one rule covers calibrator and detector state; risk: ratchet if held benign tail never comes back, adaptation lag, attacks that recur across sessions get learned | part of C |
| B | Narrow: one calibrator sample per burst, plus guards on the two detector references | smallest; measured short (6/20 agents still saturated; novel, exfil, prereq fail) | rejected alone |
| **C** | **A, with B's burst collapse inside the recurrence count: all flagged calls of one session with the same pattern count as one occurrence** | most complete; most work; same risks as A | **picked by the user** |

## Design

### When a call is held

A call is **held** when all of these are true:

- `EngineConfig.learn_flagged == "hold"` (new field; values `"all"` = today's behaviour, `"hold"`);
- the outcome is `ALLOWED` (an `APPROVED` call is a reviewer's verdict and is learned as today;
  `BLOCKED` / `DENIED` are not learned, as today);
- the call was scored by `score()` just before (the engine's per-envelope cache holds its fused score
  and θ) and that decision was flagged (`BLOCK` or `REQUIRE_APPROVAL`, the same set the
  oracle-reviewer loop uses). A call that reaches `update()` without a cached score (sampled replays in
  `agentfw/bench/run.py`, a proxy cache miss) is learned as today. *(Controller's call after the
  builder check: this keeps the oracle-reviewer results bit-identical even in sampled replays.)*
- the agent's calibrator is warm: `n_scored(agent) ≥ hold_warm_n`. Before that, everything is learned as
  today (cold start; this is what avoids the decision-008 ratchet, which began censoring at θ0).

Under the oracle-reviewer loop no call is ever `ALLOWED` and flagged at once, so that protocol's results
cannot change (goal 2; pinned by a test that compares score hashes).

### What a held call does and does not move

- **Not learned:** detector baselines (magnitude references, novelty vocabularies and counts, velocity
  history, sequence transition and prerequisite statistics, the response `item_count` reference), pooled
  priors, type inference, and the calibrator.
- **Still moves:** velocity's current-window count (`observe_refused`, as for refused calls under
  `vel_count_attempts`), because the call happened. Session context (`prior_tools`, `prev_response`) is
  carried on the envelope by the normalizer, not in engine state, so it is unaffected.

### Recurrence: when a held pattern becomes learnable

- **Pattern key** of a flagged call: `(tool_id, detector, path, value_key)` of its top reason, where
  `value_key` is the categorical value for novelty, a log₂ bucket of the value for magnitude, a log₂
  bucket of the current window count for velocity (so a 34-call burst never shares a key with the
  benign 3-call windows), and for sequence the first missing prerequisite or, on a transition reason,
  the previous tool. *(Controller's call after the builder check: an empty velocity key would let every
  burst on a tool share one key and get established after a few bursts.)* Exact encoding: builder's call
  within this shape; it must not include anything from the labels.
- Per agent, a bounded table maps pattern key → the distinct session ids and distinct days on which it
  was flagged. **All flagged calls of one session with the same key count once** (the burst collapse
  from B).
- A key is **established** once it has been flagged in ≥ `hold_min_sessions` distinct sessions on
  ≥ `hold_min_days` distinct days. From then on a flagged `ALLOWED` call with that key updates detector
  state normally, but **the calibrator still takes at most one sample per (session, key)**, so a burst
  never reaches the threshold as 34 samples even after its key is established. Earlier held calls are
  not replayed. *(Controller's call after the builder check: the recon showed one sample per burst is
  needed to keep θ off 0.999.)*
- **What the calibrator does with a held call** is one grid dimension: `hold_cal = "skip"` (the call is
  not observed) or `hold_cal = "theta"` (observed once per (session, key) as censored tail mass at the
  current θ, decision 008's winsorize, but only after warm-up). Decision 008 measured that skipping
  ratchets θ down and that winsorize-from-cold-start freezes θ at θ0; starting either after warm-up is
  untested, so the seed-0 grid measures both.
- The table is bounded per agent (`hold_max_keys`, least recently seen evicted) and is part of the
  engine's serialised state, so a restarted engine keeps it.

### Configuration (new `EngineConfig` fields)

`learn_flagged: Literal["all", "hold"]`, `hold_cal: Literal["skip", "theta"]`, `hold_warm_n: int`,
`hold_min_sessions: int`, `hold_min_days: int`, `hold_max_keys: int`. Values are fixed by the plan's
seed-0 measurement step (a small grid on seed 0 only), then frozen. The selection rule is written down
before the grid runs: among points that pass the criterion under spec §5.2 allow-all and meet the
adaptation-lag limit, the lowest benign FPR wins, ties to the shortest lag.

**If no grid point passes on seed 0** *(controller's call after the builder check)*: the mechanism may
be revised on seed 0 at most twice. Each revision is a change to the mechanism (not a constant tweak
around the same point), is written into the plan's ledger with the seed-0 numbers that motivated it,
and is followed by the same grid and the same rule. Seeds 1-3 stay unmeasured throughout. If the third
grid still has no passing point, the work stops there: the best point by minimum family recall with
benign FPR ≤ 2 % is reported on all seeds, the default stays `"all"`, and the spec §5.2 allow-all xfail
stays.

**Default.** It becomes `"hold"` only if seeds 1, 2 and 3 pass under spec §5.2 allow-all, the
oracle-reviewer results are bit-identical, and the adaptation-lag limit holds; otherwise it stays
`"all"` and the result is reported as is.

### Adaptation lag

SW1 personas have no benign drift, so lag is measured by a separate probe (in `agentfw/bench`, outside
the benchmark datasets): on a seed-0 replay, starting at a fixed warm day, one agent begins using a new
benign value (a new recipient, a new amount level, a new tool rate) in a fraction of its sessions every
day. Lag = days from first use until the value stops being flagged. It is reported per detector next to
every gate result and is part of the go/no-go.

**Limit (user's decision, 2026-11-16): at most 3 days.** A grid point whose lag exceeds 3 days on any
probe is not eligible, and the default does not become `"hold"` unless the limit holds.

## Spec and evidence

- **Spec Amendment A2** (dated section appended to `docs/spec.md`, original text untouched): amends §6.8
  (what is learned) with the rule above, lists the new fields, and states what stays unchanged.
- **Evidence** `docs/decisions/evidence/a2-gate/`: seeds 0-3, both protocols, three engines (A2, A1 as
  the pre-A2 companion, pre-A1), the sensitivity dataset, benign and control FPR, per-agent θ, held-call
  counts, and the adaptation-lag measurement. Seed 3 is generated and measured only after the constants
  are frozen.
- **Docs:** decision 010 (A2 results), README, writeup, leaderboard rebuild, the acceptance test's
  `allow` parameter loses its strict xfail only if it passes.

## Decided by the user

- Approach C (2026-11-12).
- A1.13 defaults accepted, each an ablation candidate (decision 009).
- Adaptation-lag limit: at most 3 days (2026-11-16).

## Controller's calls (after the builder check)

- Velocity keys carry a log₂ window-count bucket; sequence keys carry the missing prerequisite or the
  previous tool.
- An established key still gives the calibrator at most one sample per (session, key).
- `hold_cal` ("skip" / "theta", both only after warm-up) is a seed-0 grid dimension.
- Calls without a cached score are learned as today.
- At most two mechanism revisions on seed 0 if nothing passes; then stop and report.
- Lag is measured by a separate probe outside the benchmark datasets.

## Left to the builder

- The exact encoding of the pattern key per detector.
- The grid values for the seed-0 step and the table's eviction details.
- Test structure, file layout and task order.
