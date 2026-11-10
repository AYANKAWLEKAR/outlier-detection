# Agent Action Firewall benchmark dataset {{version}}

{{fake_banner}}
Labeled tool-call traces from LLM agents running against sandboxed (or fake) Stripe, Salesforce,
Slack, Zendesk and mail backends, with injected attacks. Produced by
`python -m agentfw.sandbox.package` from `agentfw.sandbox.run` run directories.

## Contents

| | |
|---|---|
| agents | {{agents}} |
| sessions | {{sessions}} |
| actions (envelopes) | {{actions}} |
| attacks (distinct `attack_id`) | {{attacks}} |
| attack envelopes | {{attack_envelopes}} |
| agent model | {{model}} |
| packaging seed | {{seed}} |

Attack envelopes by family:

| family | envelopes |
|---|---|
{{by_family_rows}}

Attack envelopes by injection level:

| level | envelopes |
|---|---|
{{by_injection_level_rows}}

## Files

* `traces.jsonl` / `traces.parquet`: one `RawCall` per envelope, sorted by
  `(ts_unix_ms, session_id, seq_no)`. In Parquet, `args` and `response_meta` are JSON strings.
* `labels.jsonl` / `labels.parquet`: exactly one `Label` per envelope (`benign` or `attack`).
* `manifest.json`: counts, run metadata and a sha256 per file.

Load with `agentfw.bench.dataset.load_dir`, which reads JSONL or Parquet and returns the same
objects either way.

## Timeline

Sandbox runs compress weeks of agent work into hours, so timestamps are **simulated** (D-015):
sessions are placed on a calendar by a Poisson process over working hours (UTC), in-session gaps
are kept but clamped to 0.5-120 s, and `velocity_burst` gaps are kept exactly.

## Safety and privacy

* All data is synthetic. Every email address uses an RFC 2606 reserved domain (`.example`,
  `.test`) (D-016); `validate_dataset` fails on any other domain.
* The files are scanned for credential patterns (Stripe and OpenAI-style keys, Slack
  tokens, authorization headers, Salesforce session ids); any hit fails validation. The manifest records the model id only.
* Agent transcripts (prompts and model output) are **not** part of the public dataset and are never
  copied by the packager.

## Limitations

* One agent model and one synthetic company: results may not transfer to other models, tools or
  organizations.
* Attack prevalence is set by `--action-rate` / `--prompt-rate`, not drawn from natural traffic;
  it is far above real-world base rates.
* Attack labels are planted or matched ground truth (what the injector inserted or the payload
  matcher recognized), not observed harm; a planted attack the agent ignored is still labeled where
  it was injected.
* `failed` and `budget_truncated` sessions are excluded, a survivorship bias toward sessions that
  ran cleanly.
* The calendar is retimed from a fixed start date; only in-session gaps (clamped to 0.5-120 s) and
  `velocity_burst` gaps are real, session starts are simulated.

## How to build dataset/v1

The build calls a real LLM and real sandbox APIs and costs money, so it is run by hand, not in CI.
Configure the sandbox environment first (see `agentfw/sandbox/config.py`: LLM endpoint and key,
Stripe `sk_test_` key, Salesforce Developer Edition org, Slack sandbox channel, Mailhog).

1. **Seed.** The first live run seeds the sandbox org idempotently and caches ids in
   `runs/v1/id_map.json`; there is no separate seed command.
2. **Run in resumable chunks.** The plan is prefix-stable and the run directory is resumable, so
   raise `--sessions` on the same `--out` until it reaches 2,200 (finished sessions are skipped):

   ```
   for n in 200 500 1000 1500 2000 2200; do
     uv run python -m agentfw.sandbox.run --sessions $n --seed 0 --out runs/v1 \
       --action-rate 0.15 --prompt-rate 0.10
   done
   ```

   Check `runs/v1/summary.json` (`stopped_reason` says which budget stopped a chunk; rerun later).
   If the counts below fall short, schedule more sessions or raise the attack rates and repackage;
   do not lower the minimums.
3. **Package.** Only `completed` ledger sessions are included:

   ```
   uv run python -m agentfw.sandbox.package build --runs runs/v1 --out dataset/v1 --seed 0
   ```

4. **Validate.** Requires at least 20 agents, 2,000 sessions, 40,000 actions and 600 attacks:

   ```
   uv run python -m agentfw.sandbox.package validate dataset/v1
   ```

   It exits 0 and prints `"ok": true` when the dataset is valid.

To check the commands offline, run a fake-mode run (`--fake`) and pass lowered minimums to
`validate` (`--min-agents`, `--min-sessions`, `--min-actions`, `--min-attacks`).
