# agentfw: Agent Action Firewall

Anomaly detection for AI agent tool calls. Packages: `core` (detectors), `synth` (synthetic traces),
`bench` (evaluation), `sandbox` (live MCP proxy and fake backends).

Run the green gate with `scripts/gate.sh` (add `--full` to include slow tests).

## How to build dataset/v1

The public dataset is packaged from sandbox run directories. The live collection calls a real LLM
and sandbox APIs, so it is run by hand (not in CI). Full text, including the dataset card template,
is in [docs/dataset_card.md](docs/dataset_card.md).

```
# 1. seed: the first live run seeds the sandbox org idempotently (id_map.json); no separate command
# 2. run in resumable chunks (same --out; finished sessions are skipped), up to 2,200 sessions
for n in 200 500 1000 1500 2000 2200; do
  uv run python -m agentfw.sandbox.run --sessions $n --seed 0 --out runs/v1 \
    --action-rate 0.15 --prompt-rate 0.10
done
# 3. package (only ledger status "completed" sessions are included)
uv run python -m agentfw.sandbox.package build --runs runs/v1 --out dataset/v1 --seed 0
# 4. validate (>= 20 agents, 2,000 sessions, 40,000 actions, 600 attacks)
uv run python -m agentfw.sandbox.package validate dataset/v1
```

Offline check of the same commands: add `--fake` to the run, and pass lowered `--min-*` values to
`validate`.

Live-run notes (read before the collection):

* The Salesforce org must be a Developer Edition or a sandbox. The seeder and the live backend
  check `Organization` first and refuse anything else.
* Developer Edition orgs ship with sample Contacts on real email domains. These trip the
  reserved-domain check (D-016) and packaging refuses. Delete them before the run, or keep only
  records whose `Description` starts with `afw-seed:`.
* Stripe refunds persist across runs and retries. A failed session that is retried may already
  have refunded its charge, so the retry's refund check can fail.
* The Mailhog listing returns 50 messages by default. With `max_concurrency=4`, concurrent sessions
  can satisfy each other's fixed-recipient mail checks.
* The two points above only affect the ledger's `checks` health numbers, not traces or labels.
* The Salesforce budget reservation over-counts while sessions are running (each running session
  holds its full estimate). It errs on the conservative side.
* `package build` / `validate` also scan for the literal values of every sandbox secret env var
  that is set (`LLM_API_KEY`, `SF_PASSWORD`, ...), even when the rest of the config is missing.
  Values shorter than 8 characters (e.g. `LLM_API_KEY=EMPTY`) are skipped, with a note.

### Benchmark dataset/v1 (Tier-2 exit)

Once `dataset/v1` validates, run every detector on the full by-time test split into
`results/v1/<name>/`, run `engine` and `llm_judge` on the same stratified 2,000-envelope sample,
and run the by-agent split into a separate directory (it is a separate table, never averaged with
by-time):

```
for d in engine random static_threshold isolation_forest; do
  uv run python -m agentfw.bench.run --traces dataset/v1 --detector $d --out results/v1/$d
  uv run python -m agentfw.bench.run --traces dataset/v1 --detector $d --split agent \
    --out results/v1-agent/$d
done
uv run python -m agentfw.bench.run --traces dataset/v1 --detector engine \
  --sample stratified:2000 --out results/v1/engine-sample
# live judge model; needs JUDGE_BASE_URL and JUDGE_API_KEY in the environment
uv run python -m agentfw.bench.run --traces dataset/v1 --detector llm_judge --judge-model <model> \
  --sample stratified:2000 --out results/v1/llm_judge-sample
uv run python -m agentfw.bench.leaderboard --label "sandbox dataset v1" \
  --out docs/leaderboard-v1.md results/v1/*/report.json
# by-agent split: a separate table, never merged with by-time
uv run python -m agentfw.bench.leaderboard --label "sandbox dataset v1" \
  --out docs/leaderboard-v1-agent.md results/v1-agent/*/report.json
uv run pytest -m slow tests/acceptance/test_tier2_exit.py   # skips while dataset/v1 is absent
```

## Leaderboard and writeup

[docs/leaderboard.md](docs/leaderboard.md) ranks the engine and the baselines on the **synthetic
benchmark (SW1)**, by time and by agent, with bootstrap CIs. SW1 is `sw1_config(0)` after Amendment
A1 (spec A1.4): seed 0, 20 personas, 90 days, all 8 families, magnitude spikes at or beyond 5 × the
persona's benign p99 of the field, plus a magnitude-only top-up pass (dataset `a455b10c705d`).
Regenerate it with `uv run python scripts/build_leaderboard.py` (`--jobs 4` by default; `--reuse`
keeps finished runs). [docs/writeup.md](docs/writeup.md) describes the method, the datasets, the
results with CIs, the failure cases and the limitations.

**Tier-2 exit status: not reached.** The live sandbox dataset v1 has not been built (it needs a
real LLM and sandbox accounts), so the Tier-2 numbers are not published yet;
`tests/acceptance/test_tier2_exit.py` skips until it exists. The SW1 leaderboard is a synthetic
stand-in, not the exit.

## License

No license has been chosen yet. A license must be chosen before this repository or the dataset is
published.

## Tier 1 status

Tier 1 is evaluated with the amended criterion of spec Amendment A1 (A1.7): per gated family
(`slow_drift` excluded, `magnitude_spike` in bucket 10-100) attack-level recall ≥ 0.90 and
among-the-reasons share ≥ 0.90 at the engine's live per-agent thresholds on warm test rows, and
benign FPR ≤ 2 %. It is always evaluated under two protocols and shown as two labelled results
(spec A1.5, D-028):

- spec §5.2 allow-all
- oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic)

**Status.** On SW1 seeds 0, 1 and 2 the amended criterion passes under the oracle-reviewer approval
loop (labels act as a perfect human reviewer → optimistic) and fails under spec §5.2 allow-all
(seed 0: exfil_volume, prereq_skip, novel_target and velocity_burst). That is a pass under one
protocol next to a fail under the other. Which protocol's result closes Tier 1 is the user's open
decision (D-028). The original spec §9 criterion (top-1 share ≥ 0.90) is not met under either
protocol. The Tier-2 exit is not reached. In `tests/acceptance/test_tier1_exit.py`,
`test_tier1_exit[review]` passes and `test_tier1_exit[allow]` is a strict xfail.

Details, seeds 1 and 2, and what is still not met (including a seed-2 control FPR of 5.01 % under
the oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic), above the
5 % Tier-3 bound; 3.08 % under spec §5.2 allow-all) are in
[docs/decisions/008-tier1-gate.md](docs/decisions/008-tier1-gate.md), section "Amendment A1
(2026-11-11)". Every number in the table below is a cell of
[`docs/decisions/evidence/a1-gate/matrix.json`](docs/decisions/evidence/a1-gate/matrix.json).

## Tier-1 gate under both protocols, SW1 seed 0

Cells: recall (n/N); among-the-reasons share; top-1 share (original spec §9 criterion, reported and
not gated). Each row is the result under one protocol; the two rows of an engine are read
together, and neither alone is the Tier-1 gate result. The `magnitude_spike` cell is shown with
its two companions (spec A1.5 rule 4): the engine before Amendment A1
(`EngineConfig().replace(**PRE_A1_SWITCHES)`) on the same data, and both engines on the sensitivity
dataset (`magnitude_far_mult = 1.0`). The sensitivity rows are not gate results. On seed 1 the
pre-A1 engine also passes the amended criterion under the oracle-reviewer approval loop (labels act
as a perfect human reviewer → optimistic) and fails it under spec §5.2 allow-all (decision 008).

The composition of the SW1 seed-0 `magnitude_spike` cell (spec A1.4; `matrix.json` `census`, the
same line as under the leaderboard's gate tables): Census of the magnitude_spike gate cell (spec A1.4): n 81 (warm test rows, bucket 10-100) by persona: support-refunds-c 22, lead-router-a 17, support-refunds-a 11, sales-ops-a 8, support-refunds-b 8, billing-disputes-a 4, sales-ops-b 4, crm-hygiene-a 3, exec-assistant-a 2, payouts-ops-a 1, renewals-a 1; over all 264 spikes of the dataset, median strength 48.3 and share with strength >= 10 0.970.

| dataset | engine | Protocol | magnitude_spike 10-100 | scope_blowup | exfil_volume | prereq_skip | novel_target | velocity_burst | benign FPR | control FPR | amended criterion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SW1 seed 0 (`a455b10c705d`) | amended (A1) | oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic) | 1.000 (81/81); 1.00; 1.00 | 1.000 (13/13); 1.00; 0.85 | 1.000 (19/19); 1.00; 0.84 | 1.000 (15/15); 1.00; 1.00 | 1.000 (49/49); 1.00; 1.00 | 1.000 (50/50); 1.00; 0.40 | 0.0058 | 0.035 | PASS |
| SW1 seed 0 (`a455b10c705d`) | amended (A1) | spec §5.2 allow-all | 0.938 (76/81); 1.00; 1.00 | 1.000 (13/13); 1.00; 0.85 | 0.579 (11/19); 1.00; 1.00 | 0.533 (8/15); 1.00; 1.00 | 0.163 (8/49); 1.00; 1.00 | 0.800 (40/50); 1.00; 0.40 | 0.0014 | 0.007 | FAIL: recall:exfil_volume, recall:prereq_skip, recall:novel_target, recall:velocity_burst |
| SW1 seed 0 (`a455b10c705d`) | pre-A1 | oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic) | 0.975 (79/81); 1.00; 0.99 | 1.000 (13/13); 1.00; 0.85 | 1.000 (19/19); 0.89; 0.58 | 1.000 (15/15); 1.00; 1.00 | 1.000 (49/49); 1.00; 1.00 | 0.880 (44/50); 1.00; 0.30 | 0.0092 | 0.134 | FAIL: among:exfil_volume, recall:velocity_burst |
| SW1 seed 0 (`a455b10c705d`) | pre-A1 | spec §5.2 allow-all | 0.753 (61/81); 1.00; 1.00 | 1.000 (13/13); 1.00; 0.85 | 0.474 (9/19); 1.00; 1.00 | 0.667 (10/15); 1.00; 1.00 | 0.469 (23/49); 1.00; 1.00 | 0.680 (34/50); 1.00; 0.29 | 0.0025 | 0.060 | FAIL: recall:magnitude_spike, recall:exfil_volume, recall:prereq_skip, recall:novel_target, recall:velocity_burst |
| sensitivity seed 0 (`3fc612ce1fdb`) | amended (A1) | oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic) | 0.797 (51/64); 1.00; 1.00 | 1.000 (13/13); 1.00; 0.85 | 1.000 (19/19); 1.00; 0.84 | 1.000 (15/15); 1.00; 1.00 | 1.000 (49/49); 1.00; 1.00 | 1.000 (50/50); 1.00; 0.40 | 0.0060 | 0.035 | companion row (spec A1.5 rule 4), not a gate result |
| sensitivity seed 0 (`3fc612ce1fdb`) | amended (A1) | spec §5.2 allow-all | 0.656 (42/64); 1.00; 1.00 | 1.000 (13/13); 1.00; 0.85 | 0.632 (12/19); 1.00; 1.00 | 0.533 (8/15); 1.00; 1.00 | 0.163 (8/49); 1.00; 1.00 | 0.800 (40/50); 1.00; 0.38 | 0.0014 | 0.012 | companion row (spec A1.5 rule 4), not a gate result |
| sensitivity seed 0 (`3fc612ce1fdb`) | pre-A1 | oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic) | 0.688 (44/64); 1.00; 0.96 | 1.000 (13/13); 1.00; 0.85 | 1.000 (19/19); 0.84; 0.58 | 1.000 (15/15); 1.00; 1.00 | 1.000 (49/49); 1.00; 1.00 | 0.880 (44/50); 1.00; 0.30 | 0.0090 | 0.134 | companion row (spec A1.5 rule 4), not a gate result |
| sensitivity seed 0 (`3fc612ce1fdb`) | pre-A1 | spec §5.2 allow-all | 0.578 (37/64); 1.00; 1.00 | 1.000 (13/13); 1.00; 0.85 | 0.474 (9/19); 1.00; 1.00 | 0.667 (10/15); 1.00; 1.00 | 0.469 (23/49); 1.00; 1.00 | 0.680 (34/50); 1.00; 0.26 | 0.0024 | 0.062 | companion row (spec A1.5 rule 4), not a gate result |

## Benchmark report

The block below is the leaderboard's `engine` run (split `time`), one protocol only: the
oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic). Its spec §5.2
allow-all counterpart is the second row of the table above. Its "Per-family recall" table is
measured at the per-agent FPR@95 operating point (benign FPR about 0.49), so its recall is not the
gate; the gate cells are in its "Tier-1 gate preview" section, at the live thresholds. Its latency
line was measured during the loaded leaderboard build and is not a latency result; `Engine.score`
p99 after Amendment A1 has not been measured on an idle machine.

<!-- report:start -->
### Benchmark report

Detector `engine`, split `time`, seed 0, git `d2de57fce3`, dataset `a455b10c705d`.

Protocol: oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic).

#### Headline

| Metric | All rows | Warm only (headline) |
|---|---|---|
| AUPRC | 0.779 | 0.779 |
| AUPRC 95% CI | [0.727, 0.820] | [0.727, 0.820] |
| FPR@95% recall (global) | 0.546 | 0.546 |
| FPR@95% recall (per-agent) | 0.484 | 0.490 |
| per-agent recall achieved | 0.950 | 0.950 |
| FPR@95 per-agent 95% CI | [0.407, 0.562] | [0.405, 0.557] |
| test rows | 53011 | 53011 |
| test attack rows | 2127 | 2127 |

Per-agent FPR is only meaningful at its achieved recall (shown above).

#### Per-family recall

Thresholds: `per_agent`. Cells are recall (detected/total).

| Family | Bucket | Envelope recall | Attack recall |
|---|---|---|---|
| exfil_volume | 10-30 | 1.000 (11/11) | 1.000 (4/4) |
| exfil_volume | 30-100 | 0.882 (30/34) | 1.000 (15/15) |
| exfil_volume | all | 0.911 (41/45) | 1.000 (19/19) |
| magnitude_spike | 3-10 | 1.000 (3/3) | 1.000 (3/3) |
| magnitude_spike | 10-100 | 1.000 (81/81) | 1.000 (81/81) |
| magnitude_spike | all | 1.000 (84/84) | 1.000 (84/84) |
| novel_target | all | 1.000 (49/49) | 1.000 (49/49) |
| prereq_skip | all | 1.000 (19/19) | 1.000 (15/15) |
| scope_blowup | 100-1000 | 1.000 (5/5) | 1.000 (5/5) |
| scope_blowup | 1000-10000 | 1.000 (8/8) | 1.000 (8/8) |
| scope_blowup | all | 1.000 (13/13) | 1.000 (13/13) |
| slow_drift | 0.03-0.06 | 0.417 (10/24) | 1.000 (2/2) |
| slow_drift | 0.06-0.1 | 0.686 (133/194) | 1.000 (9/9) |
| slow_drift | all | 0.656 (143/218) | 1.000 (11/11) |
| velocity_burst | 10-30 | 0.988 (338/342) | 1.000 (18/18) |
| velocity_burst | 30-60 | 0.983 (1334/1357) | 1.000 (32/32) |
| velocity_burst | all | 0.984 (1672/1699) | 1.000 (50/50) |

Control (benign-unusual) FPR: 0.587 all, 0.592 warm.

#### Tier-1 gate preview

Protocol: oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic).

Warm test rows, live per-agent thresholds, attack level (slow_drift excluded). Amended criterion (spec A1.7): per family attack recall >= 0.9 and among-the-reasons share >= 0.9, benign FPR <= 0.02. Among-the-reasons share = detected attacks with the targeted detector among the reasons of a flagged row. Top-1 share is reported beside it and not gated.

| Family | Bucket | Detector | Attack recall | Among-the-reasons share | Top-1 share (original spec §9 criterion) |
|---|---|---|---|---|---|
| exfil_volume | all | magnitude | 1.000 (19/19) | 1.000 | 0.842 |
| magnitude_spike | 10-100 | magnitude | 1.000 (81/81) | 1.000 | 1.000 |
| novel_target | all | novelty | 1.000 (49/49) | 1.000 | 1.000 |
| prereq_skip | all | sequence | 1.000 (15/15) | 1.000 | 1.000 |
| scope_blowup | all | magnitude | 1.000 (13/13) | 1.000 | 0.846 |
| velocity_burst | all | velocity | 1.000 (50/50) | 1.000 | 0.400 |

Benign FPR 0.0058, control FPR 0.035, min recall 1.000.

amended criterion under oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic): PASS (one protocol only; this is not the Tier-1 gate result, see the dual-protocol table)

#### Ablations

No ablations in this run.

#### Cold start

| n_key_obs | Recall | FPR | Attacks | Benign |
|---|---|---|---|---|
| [0, 10) | n/a | n/a | 0 | 0 |
| [10, 30) | n/a | n/a | 0 | 0 |
| [30, 100) | 0.929 | 0.529 | 56 | 174 |
| [100, inf) | 0.951 | 0.484 | 2071 | 50710 |

#### Drift

Thresholds: `per_agent`. Delay in sessions from onset to the first flagged envelope of each `slow_drift` attack; misses count as k.

| Median delay | p90 delay | Miss rate | Attacks |
|---|---|---|---|
| 3.0 | 7.0 | 0.000 | 11 |

Latency: p50 59.9 us, p99 301.9 us; detector state 9582049 bytes.
<!-- report:end -->
