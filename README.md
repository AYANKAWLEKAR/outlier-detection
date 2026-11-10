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
uv run pytest -m slow tests/acceptance/test_tier2_exit.py   # skips while dataset/v1 is absent
```

## Leaderboard and writeup

[docs/leaderboard.md](docs/leaderboard.md) ranks the engine and the baselines on the **synthetic
benchmark (SW1)** (seed 0, 20 personas, 90 days, all 8 families), by time and by agent, with
bootstrap CIs. Regenerate it with `uv run python scripts/build_leaderboard.py` (about 1 h 45 min
with 4 processes, mostly the 1,000-sample bootstraps and the isolation forest; `--reuse` keeps
finished runs). [docs/writeup.md](docs/writeup.md) describes the method, the
datasets, the results with CIs, the failure cases and the limitations.

**Tier-2 exit status: not reached.** The live sandbox dataset v1 has not been built (it needs a
real LLM and sandbox accounts), so the Tier-2 numbers are not published yet;
`tests/acceptance/test_tier2_exit.py` skips until it exists. The SW1 leaderboard is a synthetic
stand-in, not the exit.

## License

No license has been chosen yet. A license must be chosen before this repository or the dataset is
published.

## Tier 1 status

**The Tier-1 exit gate is not met** (per-family attack recall ≥ 0.90 for the targeted detector).
See [docs/decisions/008-tier1-gate.md](docs/decisions/008-tier1-gate.md); the gate is encoded as a
strict xfail in `tests/acceptance/test_tier1_exit.py`. Numbers are on the SW1 dataset (seed 0,
20 personas, 90 days, all 8 families): attack-level recall on warm test rows at the engine's live
per-agent thresholds, with seed 1 in parentheses.

| family | default engine, spec protocol (allow-all replay, §5.2) | opt-in candidate under the oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic) |
|---|---|---|
| magnitude_spike (m ≥ 10) | 0.538 (0.625) | 0.692 (0.750) |
| scope_blowup | 1.000 (0.750) | 1.000 (1.000) |
| exfil_volume | 0.579 (0.621) | 1.000 (0.966) |
| prereq_skip | 0.933 (1.000) | 1.000 (1.000) |
| novel_target | 0.388 (0.417) | 1.000 (1.000) |
| velocity_burst | 0.700 (0.635) | 0.880 (0.962) |
| benign FPR | 0.39 % (0.56 %) | 0.93 % (0.89 %) |

The candidate column is not the benchmark protocol and its engine switches
(`vel_count_attempts`, `seq_prereq_graded`, `novelty_full_n=50`) are not defaults. Both are
documented for a decision in 008. The report below is the default engine under the spec protocol.
In that report, the "Per-family recall" table is measured at the per-agent FPR@95 operating
point, where benign FPR is about 0.56, so its near-perfect recall is not the gate. The gate's
numbers are in its "Tier-1 gate preview" section, at the live thresholds.

## Benchmark report

<!-- report:start -->
### Benchmark report

Detector `engine`, split `time`, seed 0, git `bbed32f4d6`, dataset `e7b3bc2a2f20`.

#### Headline

| Metric | All rows | Warm only (headline) |
|---|---|---|
| AUPRC | 0.576 | 0.576 |
| AUPRC 95% CI | [0.480, 0.658] | [0.480, 0.658] |
| FPR@95% recall (global) | 0.536 | 0.536 |
| FPR@95% recall (per-agent) | 0.564 | 0.563 |
| per-agent recall achieved | 0.950 | 0.950 |
| FPR@95 per-agent 95% CI | [0.486, 0.582] | [0.497, 0.583] |
| test rows | 53011 | 53011 |
| test attack rows | 2062 | 2062 |

Per-agent FPR is only meaningful at its achieved recall (shown above).

#### Per-family recall

Thresholds: `per_agent`. Cells are recall (detected/total).

| Family | Bucket | Envelope recall | Attack recall |
|---|---|---|---|
| exfil_volume | 10-30 | 1.000 (11/11) | 1.000 (4/4) |
| exfil_volume | 30-100 | 0.912 (31/34) | 1.000 (15/15) |
| exfil_volume | all | 0.933 (42/45) | 1.000 (19/19) |
| magnitude_spike | 3-10 | 0.667 (4/6) | 0.667 (4/6) |
| magnitude_spike | 10-100 | 1.000 (13/13) | 1.000 (13/13) |
| magnitude_spike | all | 0.895 (17/19) | 0.895 (17/19) |
| novel_target | all | 1.000 (49/49) | 1.000 (49/49) |
| prereq_skip | all | 1.000 (19/19) | 1.000 (15/15) |
| scope_blowup | 100-1000 | 1.000 (5/5) | 1.000 (5/5) |
| scope_blowup | 1000-10000 | 1.000 (8/8) | 1.000 (8/8) |
| scope_blowup | all | 1.000 (13/13) | 1.000 (13/13) |
| slow_drift | 0.03-0.06 | 0.583 (14/24) | 1.000 (2/2) |
| slow_drift | 0.06-0.1 | 0.686 (133/194) | 1.000 (9/9) |
| slow_drift | all | 0.674 (147/218) | 1.000 (11/11) |
| velocity_burst | 10-30 | 0.974 (333/342) | 1.000 (18/18) |
| velocity_burst | 30-60 | 0.987 (1339/1357) | 1.000 (32/32) |
| velocity_burst | all | 0.984 (1672/1699) | 1.000 (50/50) |

Control (benign-unusual) FPR: 0.698 all, 0.698 warm.

#### Tier-1 gate preview

Warm test rows, live per-agent thresholds, attack level; gate = recall >= 0.9 per family (slow_drift excluded). Share = top-reason share of the targeted detector among detected attacks.

| Family | Bucket | Detector | Attack recall | Top-reason share |
|---|---|---|---|---|
| exfil_volume | all | magnitude | 0.579 (11/19) | 0.727 |
| magnitude_spike | 10-100 | magnitude | 0.538 (7/13) | 1.000 |
| novel_target | all | novelty | 0.388 (19/49) | 1.000 |
| prereq_skip | all | sequence | 0.933 (14/15) | 1.000 |
| scope_blowup | all | magnitude | 1.000 (13/13) | 0.846 |
| velocity_burst | all | velocity | 0.700 (35/50) | 0.257 |

Benign FPR 0.0039, control FPR 0.062; min recall 0.388: gate FAIL.

#### Ablations

Warm test rows. FPR@95 per-agent with the per-agent recall it achieved; family recall is attack level at live thresholds.

| Variant | AUPRC | FPR@95 per-agent | per-agent recall | exfil_volume | magnitude_spike | novel_target | prereq_skip | scope_blowup | slow_drift | velocity_burst | State bytes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| engine | 0.576 | 0.563 | 0.950 | 0.579 | 0.368 | 0.388 | 0.933 | 1.000 | 0.455 | 0.700 | 11267000 |
| engine-no-magnitude | 0.567 | 0.559 | 0.950 | 0.158 | 0.000 | 0.388 | 0.933 | 0.154 | 0.364 | 0.700 | 11211629 |
| engine-no-novelty | 0.620 | 0.476 | 0.950 | 0.579 | 0.316 | 0.000 | 0.933 | 1.000 | 0.455 | 0.700 | 3167322 |
| engine-no-velocity | 0.330 | 0.987 | 0.950 | 0.632 | 0.053 | 0.408 | 0.933 | 1.000 | 0.273 | 0.520 | 11220248 |
| engine-no-sequence | 0.595 | 0.451 | 0.950 | 0.526 | 0.368 | 0.551 | 0.000 | 0.923 | 0.273 | 0.460 | 11203907 |

#### Cold start

| n_key_obs | Recall | FPR | Attacks | Benign |
|---|---|---|---|---|
| [0, 10) | n/a | n/a | 0 | 0 |
| [10, 30) | n/a | n/a | 0 | 0 |
| [30, 100) | 0.982 | 0.560 | 55 | 175 |
| [100, inf) | 0.949 | 0.564 | 2007 | 50774 |

#### Drift

Thresholds: `per_agent`. Delay in sessions from onset to the first flagged envelope of each `slow_drift` attack; misses count as k.

| Median delay | p90 delay | Miss rate | Attacks |
|---|---|---|---|
| 3.0 | 8.0 | 0.000 | 11 |

Latency: p50 47.7 us, p99 86.8 us; detector state 11267000 bytes.
<!-- report:end -->
