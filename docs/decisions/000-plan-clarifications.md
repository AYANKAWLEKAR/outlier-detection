# Plan clarifications

Spec clarifications this plan commits to. Open decisions 1-6 from spec section 10 get their own records later.

D-021 to D-028 come with spec Amendment A1 (2026-11-11). D-021 was cited in the code before it was in this
table; it was backfilled with Amendment A1 (plan Task 1.26).

| ID | Clarification |
|---|---|
| D-001 | "Parts 1 and 2" = Tiers 1–2; Tier 0 is included as a prerequisite. |
| D-002 | Inferred `field_types` live in the engine's `TypeInference` (per-agent state) and are reported in `DetectorResult.field_types`. The normalizer has no engine state, so `ActionEnvelope` does not carry them. |
| D-003 | `ResponseMeta` gains `ids: tuple[str, ...]` (≤ 256 id-like strings). Id flow needs response-produced ids (a ticket's charge id). |
| D-004 | `ActionEnvelope` gains `prev_response: PrevResponse | None` (previous call's tool and `item_count`). This implements §2.3's "score on the next call". |
| D-005 | Strings ≥ 40 chars become derived numeric `path.__strlen__`; the content is dropped from `flat_args`. |
| D-006 | `Label` gains `strength: float | None` and `step: int | None` (strength buckets; drift onset). |
| D-007 | FPR@95%: the operating point is the **highest** threshold whose recall ≥ 0.95. A literal "lowest threshold" maximizes FPR. |
| D-008 | Recall is computed both per envelope and per attack (`attack_id` detected if any of its envelopes crosses the threshold). Multi-envelope families gate on per-attack recall. |
| D-009 | Tier 1 recall gate excludes `slow_drift` and uses `magnitude_spike` with m ≥ 10. `m` is sampled log-uniform on [3, 100]. |
| D-010 | Velocity history is sampled only over *active* minutes (window non-empty), and the 24 h window is stored sparsely. Task 1.14 confirms this as Open Decision 4. |
| D-011 | Novelty values evicted from the 10k LRU go to a lazily allocated count-min sketch, which is the sketch's only consumer. |
| D-012 | Cold-key novelty scored against the pooled prior is multiplied by 0.67 (≈ 1/1.5), mirroring magnitude's IQR × 1.5 leniency. |
| D-013 | Pooled prior keys are `p:m:{tool}:{path}` and `p:n:{tool}:{path}` (two record kinds under §7's `p:`). |
| D-014 | Action-level injected (mutated) calls are never forwarded to real APIs; the proxy returns the fake backend's result for that call. |
| D-015 | Sandbox sessions are retimed onto a simulated timeline at packaging; in-session gaps are kept (clamped 0.5–120 s); burst gaps are kept exactly. |
| D-016 | The synthetic company and attack payloads use only RFC 2606 reserved domains (`.example`, `.test`). |
| D-017 | Dependencies the spec implies but doesn't list: `pyyaml`, `scikit-learn` (isolation forest), `openai` (OpenAI-compatible client). |
| D-018 | `zendesk` in the sandbox is a local SQLite-backed MCP server, not a real API. |
| D-019 | Golden hashes are taken over canonical JSONL (`envelope_id`, score rounded to 1e-9), not Parquet bytes (Parquet metadata varies by pyarrow version). |
| D-020 | `velocity_burst` window `w` is uniform on [60, 300] s (the spec gives only `r`). |
| D-021 | Per-agent operating point: the smallest α (highest thresholds) whose pooled test recall ≥ the target. The plan's "largest" would always pick the maximum α and the maximum FPR (the same slip D-007 fixed). |
| D-022 | Velocity "normal rate" is the median of the key's historical window count (already in state, robust to learned bursts). The money-sum signal keeps the §6.4 rule so both signals share one scale. |
| D-023 | `tail_prior` equals `cal_alpha` by definition (0.005). It is a separate field so a `cal_alpha` sweep moves thresholds only. It is not tuned. |
| D-024 | In a BOOL-typed field magnitude learns values that `try_number` parses as finite (Python bools are not numbers) and scores only values that contradict BOOL. Novelty scores the contradiction only when magnitude has no usable reference. Pooled priors still learn NUMERIC-typed paths only. |
| D-025 | "Far outside the normal range" = at or beyond `magnitude_far_mult` (5) × the persona's own benign p99 of the field, taken before injection; 5 × p99 is the level of the benign-labelled control. A first-pass session with no feasible candidate stays benign, consumes its attack id and is counted in the manifest. |
| D-026 | The top-up is a second, magnitude-only pass on the stream `default_rng([seed, stable_hash("attacks:magnitude_topup")])`, one draw per eligible untouched session. On SW1 its rate equals `attack_rate` (0.015). The benchmark's magnitude gate cell must hold ≥ 50 attacks. |
| D-027 | Among-the-reasons share: per detected attack of a gated family in its gate bucket, attributed when some **flagged** row carries a targeted-detector score `> reason_min`. `top_reason_share` is unchanged (family-wide) and is printed beside it as "original spec §9 criterion". |
| D-028 | The Tier-1 gate is evaluated under both protocols and always shown as two labelled results. Which protocol's result closes Tier 1 is the user's open decision; no document may say "Tier-1 gate met" on the review result alone. |
