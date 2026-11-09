# Plan clarifications

Spec clarifications this plan commits to. Open decisions 1-6 from spec section 10 get their own records later.

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
