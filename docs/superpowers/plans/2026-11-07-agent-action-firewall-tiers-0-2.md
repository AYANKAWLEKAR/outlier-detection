# Agent Action Firewall (Tiers 0–2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the statistical action-firewall engine, its synthetic trace generator and evaluation harness (Tier 0 foundation + Tier 1 engine), then the real-API sandbox that produces and publishes the public benchmark dataset with baselines (Tier 2).

**Architecture:** One Python package `agentfw` with four subpackages: `core` (normalizer + online detectors + fusion + calibration, pure Python + numpy, no scipy), `synth` (seeded, deterministic trace generator + attack injector), `bench` (chronological score-before-update replay, metrics, baselines, reports, CI regression), and `sandbox` (MCP tool servers on real sandboxed APIs, a pass-through MCP proxy that logs envelopes, a ReAct agent loop over an OpenAI-compatible endpoint, injection, dataset packaging). Every detector exposes separate `score()` / `update()` calls; the harness always scores before it updates.

**Tech Stack:** Python 3.12, uv, ruff, mypy --strict, pydantic v2, dataclasses(slots), orjson, numpy, xxhash, pyarrow, PyYAML; pytest + hypothesis + pytest-benchmark; scipy/pandas in tests and reporting only; Tier 2 adds `mcp`, `openai`, `stripe`, `simple-salesforce`, `slack_sdk`, `scikit-learn`, Docker (Mailhog).

**Spec:** `docs/spec.md` (Technical Spec v3). Section references below (§N.N) point into it.

---

## Scope and interpretation

"Parts 1 and 2" is read as **Tier 1 (engine) and Tier 2 (sandbox + public benchmark)** from spec §9. Both sit on Tier 0 (data model, normalizer, generator, harness), and nothing exists yet in the repo, so Tier 0 is Phase 0 of this plan. Tiers 3–5 (blast radius, Go proxy, dashboard) are out of scope. Where the code needs a seam for them (adapter hints, verdicts, shadow scoring), the plan adds the seam and nothing more.

## Feasibility assessment

**Verdict: feasible with the stated stack.** Phases 0–1 are local, deterministic and low risk. Phase 2 carries most of the risk: it depends on external accounts, an LLM that does not behave the same way every run, and dataset-size targets that the spec's own task shapes don't reach.

**Timeline (one engineer, full time).** Spec totals 4.5–5.5 weeks; this plan estimates **6.5–8 weeks**.

| Phase | Spec | This plan | Why the delta |
|---|---|---|---|
| 0 — Tier 0 foundation | 4–5 d | 5–6 d | 12 tasks; generator validation (§4.7) and golden hashing are fiddly. |
| 1 — Tier 1 engine | 1.5 wk | 2–2.5 wk | 4 detectors + calibration + priors + 17 extra personas + decision sweeps. |
| 2 — Tier 2 sandbox | 2–3 wk | 3–4 wk | Account setup lead time, 70+ task templates, LLM nondeterminism, run time against API rate limits. |

**Cash cost.** Phases 0–1: $0. Phase 2 (estimates; verify current provider prices before running): agent runs ≈ 2,000–2,500 sessions × ~20 tool-call turns × ~5k-token average context ≈ 200–250M input tokens. At hosted 70B-class prices of roughly $0.2–1.2 per million input tokens, that is **≈ $50–300**, or ~2× that with development reruns. LLM-judge baseline: 2k envelopes × ~4k tokens ≈ 8M tokens on a frontier model, **≈ $25–40**. Stripe test mode, a Salesforce Developer Edition org, a free Slack workspace and Mailhog: $0.

**Risks and mitigations**

| # | Risk | Impact | Mitigation in this plan |
|---|---|---|---|
| R1 | 40k actions / 2k sessions = 20 calls per session; single-ticket tasks produce 4–8 | Misses the Tier 2 exit target | Agents work a *queue* of 3–5 items per session (`AgentDef.batch_size`, Task 2.6); runner can add sessions until the target is met (Task 2.8). |
| R2 | Prompt-level injections succeed only when the model complies (often a minority of attempts) | < 600 attacks | Action-level injection fills the quota; `injection_level` is reported separately (Tasks 2.7, 2.9). |
| R3 | Salesforce Developer Edition: ~15k API calls / 24 h, ~5 MB data storage (~2.5k records) | Throttling; `scope_blowup` to 10k ids can't execute | Seeded company ≤ 2,000 SF records; persisted call budget (12k/day default); mass updates via sObject Collections (200/call); mutated calls never forwarded (D-014). |
| R4 | Stripe refunds can't exceed the charge amount; test mode is rate-limited | Mutated magnitude calls fail | Mutated calls are never forwarded; the proxy returns a fake-backend result (D-014). Concurrency capped at 4. |
| R5 | Provider function-calling names disallow `.` (`^[a-zA-Z0-9_-]{1,64}$`) | Tool calls rejected | `to_fn_name` / `from_fn_name` mapping `stripe.create_refund ↔ stripe__create_refund` (Task 2.5). |
| R6 | Mailhog is unmaintained | Docker image churn | It still works; the HTTP assertion client lives in one module, so Mailpit can replace it (SMTP compatible). |
| R7 | With persona σ = 0.7, a ×3–×10 spike is inside natural variance (z ≈ ln m / 0.7) | `magnitude_spike` can't reach 90% across the whole m range | Tier 1 gate uses the m ≥ 10 strength bucket; [3,10) is reported (D-009). |
| R8 | `slow_drift` targets the drift detector, which is Tier 5 | Tier 1 "≥ 90% per family" can't hold for it | Excluded from the Tier 1 gate; delay and miss rate are reported (D-009). |
| R9 | `velocity_burst` / `exfil_volume` / `slow_drift` span several envelopes; early ones are unflaggable by design | Envelope-level recall is capped well below 90% | Gate on attack-level recall (any envelope of the `attack_id` flagged); envelope-level also reported (D-008). |
| R10 | p99 < 200 µs in pure Python is tight | Tier 1 gate fails | No pydantic in the hot path, slots dataclasses, precompiled regexes; perf test from Task 1.11 onward catches regressions early. |
| R11 | Sandbox wall-clock timestamps compress weeks of behaviour into hours | Velocity baselines unrealistic | Packaging retimes sessions onto a simulated timeline, preserving in-session gaps (D-015). |
| R12 | Personas use `zendesk.*`, but §1.2 lists no Zendesk server | Missing tool server | Local SQLite-backed `zendesk` MCP server (D-018). |

## Spec clarifications this plan commits to

Task 0.1 copies this table into `docs/decisions/000-plan-clarifications.md`. Open decisions 1–6 from §10 get their own records in Task 1.14.

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

## Global Constraints

- Python **3.12** exactly (`requires-python = ">=3.12,<3.13"`); env and lockfile via **uv**; **ruff** (lint + format) and **mypy --strict** must pass on `agentfw/` and `tests/`.
- **pydantic v2** only for data crossing a boundary (file, wire): `RawCall`, `ResponseMeta`, `Label`, synth/sandbox config. Inside the hot path: `@dataclass(slots=True)` (frozen where immutable). **orjson** for all JSON I/O, with `OPT_SORT_KEYS` wherever bytes are hashed.
- `agentfw.core` must **never import scipy, pandas, sklearn or pydantic-heavy validation** in the scoring path (enforced by `tests/test_no_scipy.py`, Task 1.11).
- **Score before update, never the reverse.** `score()` must not mutate state (enforced by idempotence and no-leak tests). `update()` is applied only for outcomes `ALLOWED` / `APPROVED`; `BLOCKED` / `DENIED` never move a baseline.
- Freetext content (strings ≥ 40 chars) is **never stored in detector state**.
- Determinism: same seed ⇒ byte-identical `traces.jsonl` / `labels.jsonl`. Per-persona RNG = `np.random.default_rng([seed, stable_hash(agent_id)])`; attack RNG = `np.random.default_rng([seed, stable_hash("attacks")])`. `stable_hash` uses xxhash64, never Python's `hash()`.
- All tuned constants live in `agentfw/core/config.py::EngineConfig` with the spec's values: `k_prior_tools=4`, `max_session_ids=2000`, `freetext_min_len=40`, `type_min_obs=10`, `type_sticky_window=100`, `reservoir_size=64`, `mag_probs=(0.05,0.25,0.5,0.75,0.95)`, `ewma_alpha=0.02`, `mag_cold_n=30`, `z0=3.0`, `tau=0.75`, `mag_eps=0.05`, `prior_iqr_inflate=1.5`, `prior_novelty_discount=0.67`, `priors_mode="all"`, `vel_short_min=10`, `vel_long_min=1440`, `vel_eps=0.25`, `vel_cold_n=30`, `novelty_lru=10_000`, `novelty_full_n=200`, `cms_width=2048`, `cms_depth=4`, `trans_min_ctx=20`, `trans_smoothing=0.5`, `prereq_min_n=50`, `prereq_rate=0.95`, `flow_min_obs=20`, `fusion_bonus=0.15`, `fusion_bonus_gate=0.5`, `reason_min=0.3`, `cal_alpha=0.005`, `cal_theta0=0.6`, `cal_blend_n=500`, `cal_delta=0.1`.
- Bench constants: `warmup_min_obs=30`, `train_frac=0.7`, `recall_target=0.95`, `bootstrap_n=1000`, `attack_rate=0.015`, `warmup_sessions=30`, CI regression fails if AUPRC drops > `0.01` or FPR@95 rises > `0.002` (0.2 pp).
- **Sandbox safety:** Stripe key must start with `sk_test_` (refuse otherwise); Slack posts only to the one configured sandbox channel; email only via Mailhog SMTP on localhost/compose; Salesforce only the Developer Edition org; no real personal data (D-016). Mutated calls are never forwarded (D-014).
- **Green gate.** `scripts/gate.sh` runs `uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest -q -m "not slow and not live"`. `scripts/gate.sh --full` drops the `slow` exclusion. **No task starts until the previous task's gate is green.**
- Test markers: `slow` (acceptance and perf, > 30 s), `live` (needs sandbox credentials or network; skipped when env vars are absent).

## Review Focus

Inputs the spec implies but no feature test naturally exercises. Each one has a pinned test in the task named.

1. **Messy argument shapes** (null values, empty lists, lists of dicts, 50-level nesting, non-string dict keys, bools mixed with numbers). Expected: the normalizer never raises; deep subtrees are truncated at depth 16 with a `__truncated__` marker. Pinned by Task 0.3 `test_messy_args_do_not_crash`.
2. **Numbers that aren't clean floats** (`"37.50"` strings, `NaN`, `±inf`, negatives, `True` in a numeric field). Expected: numeric strings count as numeric; NaN is skipped (no score, no update); ±inf scores 1.0 and is never pushed; negatives go through `sign·log1p`. Pinned by Task 1.3 `test_numeric_strings_infer_numeric` and Task 1.4 `test_non_finite_values`.
3. **Brand-new or degenerate keys** (n = 0, a constant field with IQR = 0, an agent with one tool so the vocabulary V = 1). Expected: every score is finite and in [0, 1], with no division by zero. Pinned by Task 1.11 `test_scores_always_finite_in_unit_interval` (hypothesis).
4. **Case and Unicode variants of addresses** (`Bob@ACME.example` vs `bob@acme.example`, IDN `bücher.example`). Expected: derived domains and hosts are lowercased and IDNA-normalized, so no false novelty. Pinned by Task 0.3 `test_domain_case_and_idn_normalized` and Task 1.5 `test_case_variant_domain_not_novel`.
5. **Downstream tool failure** (server raises, times out, or returns non-JSON). Expected: the proxy returns `isError`, records the envelope with `response_meta.error=True`, and keeps serving the session. Pinned by Task 2.4 `test_downstream_failure_recorded_and_survivable`.

---

## File structure

```
pyproject.toml  uv.lock  README.md  docker-compose.yml  .env.example  scripts/gate.sh  scripts/update_golden.py
.github/workflows/ci.yml
agentfw/
  core/
    envelope.py        data model (RawCall, ActionEnvelope, Label, DetectorScore, Reason, DetectorResult, Verdict, enums)
    ids.py             deterministic ULIDs
    patterns.py        id/email/url/number recognizers
    normalize.py       flatten, derived fields, SessionTracker, normalize()
    config.py          EngineConfig (all constants)
    hints.py           ToolHint (adapter seam for Tier 3)
    state.py           Record registry, DictStore, key builders, dump/load
    schema_infer.py    TypeInference, TypeView
    stats/p2.py        P2Quantiles, p2_tail
    stats/ewma.py      Ewma
    sketch/bloom.py    BloomFilter
    sketch/cms.py      CountMinSketch
    detectors/base.py  SubDetector protocol, squash, signed_log1p, iter_fields
    detectors/{magnitude,novelty,velocity,sequence}.py
    fusion.py  calibration.py  priors.py  engine.py
  synth/
    schema.py load.py dists.py grammar.py sampler.py attacks.py generate.py
    catalog/*.yaml personas/*.yaml tasks/*.yaml
  bench/
    protocol.py dataset.py replay.py metrics.py report.py registry.py run.py ci_regress.py
    ablation.py sweeps.py sampling.py leaderboard.py baseline_metrics.json
    baselines/{random_,static_threshold,isolation_forest,llm_judge}.py
  sandbox/
    config.py llm.py agent.py injector.py run.py retime.py package.py checks.py
    seed/{company,stripe_seed,salesforce_seed,zendesk_seed}.py
    tools/{common,stripe_server,salesforce_server,mail_server,slack_server,zendesk_server}.py
    proxy_py/{proxy,response_meta,sinks,__main__}.py
    tasks/schema.py tasks/*.yaml agents/*.yaml payloads/*.yaml
tests/  (mirrors agentfw/; plus tests/factories.py, tests/acceptance/, tests/perf/, tests/golden/)
docs/   spec.md writeup.md leaderboard.md dataset_card.md decisions/
```

---

# Phase 0 — Tier 0 foundation

### Task 0.1: Repo scaffold and green gate

**Files:**
- Create: `pyproject.toml`, `agentfw/__init__.py`, `agentfw/{core,synth,bench,sandbox}/__init__.py`, `tests/__init__.py`, `tests/test_smoke.py`, `scripts/gate.sh`, `.gitignore`, `.github/workflows/ci.yml`, `README.md`, `docs/decisions/000-plan-clarifications.md`

**Interfaces:**
- Produces: `agentfw.__version__ = "0.1.0"`; `scripts/gate.sh [--full]`; pytest markers `slow`, `live`.

- [ ] **Step 1: `git init`, then write the failing test** `tests/test_smoke.py`

```python
def test_package_importable_with_version() -> None:
    import agentfw
    assert agentfw.__version__ == "0.1.0"
```

- [ ] **Step 2: Run `uv run pytest tests/test_smoke.py -v`.** Expected: FAIL (no project or package yet).
- [ ] **Step 3: Create the project.** `pyproject.toml`: `requires-python = ">=3.12,<3.13"`. Runtime deps: `pydantic>=2.9`, `orjson`, `numpy`, `xxhash`, `pyarrow`, `pyyaml`. Dev group: `pytest`, `hypothesis`, `pytest-benchmark`, `ruff`, `mypy`, `scipy`, `pandas`, `types-PyYAML`. Optional group `sandbox`: `mcp`, `openai`, `stripe`, `simple-salesforce`, `slack_sdk`, `scikit-learn`. Settings: `[tool.mypy] strict = true`, `files = ["agentfw", "tests"]`, `plugins = ["pydantic.mypy"]`; `[tool.pytest.ini_options] markers = ["slow", "live"]`; ruff `line-length = 100`, select `E,F,I,UP,B,SIM`. `gate.sh` as in Global Constraints. CI workflow runs `uv sync --all-groups` then `scripts/gate.sh`. Copy the "Spec clarifications" table into `000-plan-clarifications.md`.
- [ ] **Step 4: Run `uv sync --all-groups && ./scripts/gate.sh`.** Expected: exit 0, 1 passed.
- [ ] **Step 5: Commit** `git add -A && git commit -m "chore: scaffold agentfw package, tooling and green gate"`

### Task 0.2: Data model and deterministic ids

**Files:**
- Create: `agentfw/core/envelope.py`, `agentfw/core/ids.py`, `tests/factories.py`, `tests/core/test_envelope.py`, `tests/core/test_ids.py`

**Interfaces:**
- Produces (`envelope.py`):
  - `Scalar = str | int | float | bool | None`; `JsonValue` (recursive JSON type alias)
  - `FieldType(StrEnum)`: `NUMERIC, BOOL, CATEGORICAL, ID, FREETEXT, UNKNOWN`
  - `Outcome(StrEnum)`: `ALLOWED, BLOCKED, APPROVED, DENIED`; `Decision(StrEnum)`: `ALLOW, BLOCK, REQUIRE_APPROVAL`
  - `ResponseMeta(BaseModel, frozen)`: `item_count: int | None = None`, `bytes: int | None = None`, `error: bool = False`, `ids: tuple[str, ...] = ()` (max 256)
  - `RawCall(BaseModel, frozen)`: `envelope_id: str`, `schema_version: int = 1`, `session_id`, `agent_id`, `principal_id: str`, `seq_no: int ≥ 0`, `ts_unix_ms: int`, `tool_id: str` (pattern `^[a-z0-9_]+\.[a-z0-9_]+$`), `args: dict[str, JsonValue]`, `response_meta: ResponseMeta | None = None`
  - `PrevResponse(slots, frozen)`: `tool_id: str`, `item_count: int | None`
  - `IdSnapshot(slots, frozen)`: `exact: frozenset[str]`, `bloom: BloomFilter | None`; `__contains__(v: str) -> bool`. `BloomFilter` is added in Task 0.3; declare it under `TYPE_CHECKING` until then.
  - `ActionEnvelope(slots, frozen)`: every `RawCall` field plus `flat_args: Mapping[str, tuple[Scalar, ...]]`, `derived: Mapping[str, tuple[Scalar, ...]]`, `prior_tools: tuple[str, ...]`, `session_ids_seen: IdSnapshot`, `prev_response: PrevResponse | None`, `blast_radius: float | None = None`
  - `Label(BaseModel, frozen)`: `envelope_id`, `label: Literal["benign","attack"]`, `attack_family: str | None`, `injection_level: Literal["prompt","action"] | None`, `attack_id: str | None`, `control_type: Literal["benign_unusual"] | None`, `strength: float | None`, `step: int | None`. Validator: attack ⇒ family and attack_id set and control_type None.
  - `DetectorScore(slots, frozen)`: `detector: str`, `field: str | None`, `score: float`, `evidence: Mapping[str, object]`
  - `BaselineSummary(slots, frozen)`: `median: float`, `iqr: float`, `n: int`
  - `Reason(slots, frozen)`: `detector`, `field`, `observed: Scalar`, `baseline_summary: BaselineSummary | None`, `percentile: float | None`, `blast_radius: float | None`, `prereq_missing: str | None`, `prior_used: bool = False`
  - `DetectorResult(slots, frozen)`: `fused_score: float`, `per_detector: tuple[DetectorScore, ...]`, `reasons: tuple[Reason, ...]`, `latency_us: float`, `threshold: float | None = None`, `decision: Decision | None = None`, `field_types: Mapping[str, FieldType] = {}`
  - `Verdict(slots, frozen)` (§3 fields; used from Tier 4)
- Produces (`ids.py`): `ulid_from(ts_ms: int, rand80: int) -> str` (Crockford base32, 26 chars); `new_ulid(ts_ms: int, rng: np.random.Generator) -> str`
- Produces (`tests/factories.py`): `make_raw(**overrides) -> RawCall` with defaults `session_id="s1"`, `agent_id="a1"`, `principal_id="p1"`, `seq_no=0`, `ts_unix_ms=1_700_000_000_000`, `tool_id="stripe.create_refund"`, `args={"amount": 37.5}`

- [ ] **Step 1: Write failing tests**

```python
def test_rawcall_json_roundtrip() -> None:
    raw = make_raw(args={"amount": 37.5, "to": ["x@acme.example"]},
                   response_meta=ResponseMeta(item_count=3, ids=("ch_ABCDEFGH12",)))
    assert RawCall.model_validate_json(raw.model_dump_json()) == raw
    assert raw.schema_version == 1

def test_rawcall_rejects_unnamespaced_tool() -> None:
    with pytest.raises(ValidationError):
        make_raw(tool_id="create_refund")

def test_response_meta_ids_bounded() -> None:
    with pytest.raises(ValidationError):
        ResponseMeta(ids=tuple(f"ch_{i:010d}" for i in range(257)))

def test_label_integrity_rules() -> None:
    Label(envelope_id="e", label="benign", control_type="benign_unusual")
    with pytest.raises(ValidationError):
        Label(envelope_id="e", label="attack")                       # no family / attack_id
    with pytest.raises(ValidationError):
        Label(envelope_id="e", label="attack", attack_family="prereq_skip",
              attack_id="A1", control_type="benign_unusual")

def test_ulid_shape_order_determinism() -> None:
    rng = np.random.default_rng(0)
    a, b = new_ulid(1_000, rng), new_ulid(2_000, rng)
    assert len(a) == 26 and a < b
    assert new_ulid(5, np.random.default_rng(7)) == new_ulid(5, np.random.default_rng(7))
```

- [ ] **Step 2: Run `uv run pytest tests/core -v`.** Expected: FAIL (ImportError).
- [ ] **Step 3: Implement** `envelope.py` and `ids.py` with the signatures above. `ActionEnvelope` is a frozen slots dataclass; assigning to it raises `FrozenInstanceError`.
- [ ] **Step 4: Run `./scripts/gate.sh`.** Expected: green.
- [ ] **Step 5: Commit** `git commit -am "feat(core): envelope data model and deterministic ULIDs"`

### Task 0.3: Patterns, Bloom filter, normalizer, session tracker

**Files:**
- Create: `agentfw/core/patterns.py`, `agentfw/core/sketch/bloom.py`, `agentfw/core/normalize.py`, `tests/core/test_patterns.py`, `tests/core/test_bloom.py`, `tests/core/test_normalize.py`
- Modify: `tests/factories.py` (add `make_env`)

**Interfaces:**
- Consumes: `RawCall`, `ActionEnvelope`, `IdSnapshot`, `PrevResponse` (Task 0.2)
- Produces:
  - `patterns.is_id_like(s: str) -> bool`: prefix + ≥ 8 alnum (`^[A-Za-z]{1,6}_[A-Za-z0-9]{8,}$`), UUID, email, Salesforce 15/18-char id (`^[0-9a-zA-Z]{15}([0-9a-zA-Z]{3})?$` with a leading 3-char key prefix), `T-` ticket style `^[A-Z]{1,4}-\d{6,}$`
  - `patterns.id_prefix(s) -> str | None` (`"ch_"`, `"acct_"`, SF first 3 chars, `None` for UUID/email); `email_domain(s) -> str | None` (lowercased, IDNA); `url_host(s) -> str | None` (lowercased, IDNA); `try_number(v: object) -> float | None` (accepts int/float/numeric strings, rejects bool)
  - `BloomFilter(capacity: int, fp_rate: float = 1e-3)`: `add(v: str)`, `__contains__`, `copy()`, `to_dict()`, `from_dict()`; xxhash64 double hashing
  - `normalize.flatten(args) -> tuple[dict[str, tuple[Scalar, ...]], dict[str, tuple[Scalar, ...]]]` returning `(flat_args, derived)`
  - `SessionTracker(k: int = 4, max_ids: int = 2000)`: `snapshot(session_id) -> tuple[tuple[str, ...], IdSnapshot, PrevResponse | None]`; `observe(e: ActionEnvelope) -> None` (records `tool_id`, id-like arg values, `response_meta.ids`, and `prev_response`)
  - `normalize(raw: RawCall, tracker: SessionTracker) -> ActionEnvelope` (reads the tracker and **does not** observe)
  - `tests/factories.make_env(raw=None, tracker=None, **raw_overrides) -> ActionEnvelope`

Derived-field rules (§6.1, D-005): lists give `path.__len__` and elements at `path[]`. Email gives `path.__domain__`, URL gives `path.__host__`, id-like strings give `path.__prefix__`, strings ≥ 40 chars give `path.__strlen__` with the string removed. Dicts nested beyond depth 16 are dropped and `path.__truncated__ = (1,)` is emitted.

- [ ] **Step 1: Write failing tests** (excerpt; include all)

```python
def test_flatten_nested_lists_and_lengths() -> None:
    flat, d = flatten({"a": {"b": 1}, "to": ["X@Acme.example", "y@acme.example"], "empty": []})
    assert flat["a.b"] == (1,)
    assert flat["to[]"] == ("X@Acme.example", "y@acme.example")
    assert d["to.__len__"] == (2,) and d["empty.__len__"] == (0,)
    assert d["to[].__domain__"] == ("acme.example", "acme.example")

def test_long_strings_become_length_only() -> None:
    flat, d = flatten({"body": "x" * 120})
    assert "body" not in flat and d["body.__strlen__"] == (120,)

def test_id_prefix_and_url_host() -> None:
    _, d = flatten({"c": "ch_3Nabc12345XYZ", "sf": "003Dn00000ABCDEFG", "u": "https://Files.Example.com/x?y=1"})
    assert d["c.__prefix__"] == ("ch_",) and d["sf.__prefix__"] == ("003",)
    assert d["u.__host__"] == ("files.example.com",)

def test_domain_case_and_idn_normalized() -> None:
    _, d = flatten({"to": "Bob@ACME.example", "cc": "a@BÜCHER.example"})
    assert d["to.__domain__"] == ("acme.example",)
    assert d["cc.__domain__"] == ("xn--bcher-kva.example",)

def test_messy_args_do_not_crash() -> None:
    deep: dict[str, object] = {}
    cur = deep
    for _ in range(50):
        cur["n"] = {}; cur = cur["n"]  # type: ignore[assignment]
    flat, d = flatten({"none": None, "flag": True, "items": [{"id": "ch_AAAAAAAAAA"}, {"id": None}],
                       "deep": deep, 5: "intkey"})  # type: ignore[dict-item]
    assert flat["items[].id"] == ("ch_AAAAAAAAAA", None)
    assert any(k.endswith("__truncated__") for k in d)
    assert flat["5"] == ("intkey",)

def test_tracker_is_causal_and_bounded() -> None:
    tr = SessionTracker(k=4, max_ids=2000)
    e0 = normalize(make_raw(tool_id="zendesk.get_ticket", args={"ticket_id": "T-00000001"},
                            response_meta=ResponseMeta(item_count=1, ids=("ch_AAAAAAAAAA",))), tr)
    assert e0.prior_tools == () and e0.prev_response is None
    tr.observe(e0)
    e1 = normalize(make_raw(seq_no=1, tool_id="stripe.retrieve_charge", args={"charge_id": "ch_AAAAAAAAAA"}), tr)
    assert e1.prior_tools == ("zendesk.get_ticket",)
    assert "ch_AAAAAAAAAA" in e1.session_ids_seen
    assert e1.prev_response == PrevResponse("zendesk.get_ticket", 1)
    for i in range(2, 8):
        tr.observe(normalize(make_raw(seq_no=i, tool_id=f"x.t{i}"), tr))
    assert normalize(make_raw(seq_no=8), tr).prior_tools == ("x.t4", "x.t5", "x.t6", "x.t7")

def test_ids_overflow_to_bloom_and_sessions_isolated() -> None:
    tr = SessionTracker(max_ids=10)
    for i in range(50):
        tr.observe(make_env(make_raw(seq_no=i, args={"id": f"ch_{i:010d}"}), tr))
    snap = normalize(make_raw(seq_no=50), tr).session_ids_seen
    assert all(f"ch_{i:010d}" in snap for i in range(50)) and len(snap.exact) == 10
    assert sum(f"zz_{i:010d}" in snap for i in range(10_000)) < 100
    assert "ch_0000000001" not in normalize(make_raw(session_id="s2"), tr).session_ids_seen
```

`test_bloom.py`: no false negatives for 5k adds; false-positive rate < 0.5% at capacity; `from_dict(to_dict())` behaves identically.

- [ ] **Step 2: Run `uv run pytest tests/core -v`.** Expected: FAIL.
- [ ] **Step 3: Implement** the three modules. The tracker keeps per session a `deque(maxlen=k)` of tools, a `set` of ids (on overflow the set is frozen into `exact` and further ids go into a `BloomFilter(capacity=8*max_ids)`), and the last `PrevResponse`. `snapshot` returns copies so envelopes stay immutable.
- [ ] **Step 4: Run `./scripts/gate.sh`.** Expected: green.
- [ ] **Step 5: Commit** `git commit -am "feat(core): normalizer with derived fields and causal session tracker"`

### Task 0.4: Synth schema, tool catalog, 3 personas, task grammars

**Files:**
- Create: `agentfw/synth/schema.py`, `agentfw/synth/load.py`, `agentfw/synth/catalog/{stripe,salesforce,mail,slack,zendesk}.yaml`, `agentfw/synth/personas/{support-refunds-a,sales-ops-a,outreach-a}.yaml`, `agentfw/synth/tasks/{support,sales,outreach}.yaml`, `tests/synth/test_schema.py`

**Interfaces:**
- Produces (pydantic models, `extra="forbid"`):
  - `DistSpec`: `family: Literal["lognormal","gamma","poisson","zipf","categorical","constant","uniform"]` plus params `mu, sigma, shape, scale, lam, min, max, vocab_size, s, vocab: list[str], weights: list[float], dirichlet_alpha: float = 50.0, value, round_cents: bool`
  - `ArgSpec`: `type: Literal["numeric","categorical","id","email","freetext","list_of_id","list_of_email","freetext_map","bool"]`, `prefix: str | None`, `source: str | None` (`"external"` or `"<server.tool>.<field>"`), `dist: DistSpec | None`, `shape: Literal["money","count"] | None`, `vocab: list[str] | None`, `len_dist: DistSpec | None`
  - `ResponseSpec`: `item_count: DistSpec | None`, `produces_ids: list[str] = []`
  - `ToolSpec`: `kind: Literal["read","write"]`, `args: dict[str, ArgSpec]`, `produces: list[str]`, `response: ResponseSpec`
  - `Catalog = dict[str, ToolSpec]` keyed by `server.tool`
  - `StepSpec`: `tool: str`, `as_: str | None` (YAML `as`), `args: dict[str, str]` (bindings), `skippable: bool = False`, `optional_prob: float = 1.0`
  - `TaskSpec`: `name`, `steps: list[StepSpec]`, `repeat: DistSpec | None` (items per session)
  - `Branches`: `skip_lookup: float = 0.05`, `retry: float = 0.03`, `abandon: float = 0.02`, `arg_dropout: float = 0.02`
  - `Persona`: `agent_id`, `tools: list[str]`, `session_rate: DistSpec`, `tasks: list[TaskWeight]`, `fields: dict[str, DistSpec]`, `drift: DriftSpec | None`, `noise` → `Branches`, `id_pools: dict[str, IdPoolSpec(prefix, size, s)]`, `id_flow_rate: float = 1.0`, `working_hours: tuple[int, int] = (8, 18)`, `quarter_end_days: list[int] = []`
  - `load.load_catalog(dir) -> Catalog`, `load_personas(dir) -> list[Persona]`, `load_tasks(dir) -> dict[str, TaskSpec]`, `validate_world(catalog, personas, tasks) -> None` (raises `WorldError` with all problems listed)

Binding grammar for `StepSpec.args` values: `ext:<pool>` (Zipf draw from a persona id pool), `<alias>.<field>` (value produced by an earlier step), `sample` (persona `fields["<tool>.<arg>"]` override, else the catalog `dist`), `frac:<alias>.<field>:<lo>:<hi>` (uniform fraction of an earlier numeric value). Catalog must include at least: `stripe.{retrieve_charge, list_charges, create_refund, retrieve_customer, create_payout}`, `salesforce.{run_soql, get_record, update_record, mass_update, create_task}`, `mail.send_email`, `slack.{post_message, read_channel}`, `zendesk.{get_ticket, list_tickets, add_comment, update_status}`. Refund `reason` vocab includes `manager_approved` with weight 0.01 (needed by the control family in Task 1.12). Every email domain is `.example`/`.test` (D-016).

- [ ] **Step 1: Write failing tests**

```python
def test_world_loads_and_is_consistent() -> None:
    cat, ps, ts = load_catalog(CAT), load_personas(PER), load_tasks(TASKS)
    validate_world(cat, ps, ts)                                   # no WorldError
    assert {p.agent_id for p in ps} == {"support-refunds-a", "sales-ops-a", "outreach-a"}

def test_spec_example_values_preserved() -> None:
    p = {p.agent_id: p for p in load_personas(PER)}["support-refunds-a"]
    amt = p.fields["stripe.create_refund.amount"]
    assert (amt.family, amt.mu, amt.sigma) == ("lognormal", 3.6, 0.7)
    assert p.session_rate.lam == 40

def test_validate_world_reports_unknown_tool_and_bad_source(tmp_path: Path) -> None:
    ...  # write a persona using "stripe.nope" and a source "stripe.retrieve_charge.missing"
    with pytest.raises(WorldError, match="stripe.nope") as ei: validate_world(...)
    assert "missing" in str(ei.value)
```

- [ ] **Step 2: Run the tests.** Expected: FAIL.
- [ ] **Step 3: Implement** the models, loaders and YAML. `small_refund` follows §4.4: `zendesk.get_ticket(ticket_id=ext:tickets)` as `ticket` → `stripe.retrieve_charge(charge_id=ticket.charge_id)` with `skippable: true` → `stripe.create_refund(charge_id=ticket.charge_id, amount=sample, reason=sample)` (ruling R1: refund amounts follow the persona override so §4.7's KS check holds; `frac:` stays available for other tasks) → `mail.send_email(to=ticket.customer_email, subject=sample, body=sample)`.
- [ ] **Step 4: Run `./scripts/gate.sh`.** Expected: green.
- [ ] **Step 5: Commit** `git commit -am "feat(synth): catalog, persona and task-grammar schema with 3 personas"`

### Task 0.5: Distributions and grammar walker

**Files:**
- Create: `agentfw/synth/dists.py`, `agentfw/synth/grammar.py`, `tests/synth/test_dists.py`, `tests/synth/test_grammar.py`

**Interfaces:**
- Consumes: Task 0.4 models
- Produces:
  - `dists.sample(spec: DistSpec, rng: Generator) -> float | int | str`. Lognormal draws are made in log space. `categorical` draws a Dirichlet(`dirichlet_alpha·weights`) once per persona (cached on the `IdPool`/persona context) so rare values exist. `round_cents` rounds to 2 dp. `min`/`max` clip.
  - `IdPool(prefix: str, size: int, s: float, rng: Generator)`: `draw() -> str` (Zipf over a fixed pool, so hot ids recur); `fabricate(rng) -> str` (same prefix, guaranteed not in the pool)
  - `PlannedCall(slots)`: `tool_id: str`, `args: dict[str, JsonValue]`, `response_meta: ResponseMeta | None`, `gap_s: float`
  - `grammar.walk(task: TaskSpec, persona: Persona, catalog: Catalog, rng: Generator, pools: dict[str, IdPool]) -> list[PlannedCall]`. Produced values of a skipped step are still generated, so later bindings resolve. Retry duplicates a call with identical args. Abandon truncates after a uniformly chosen step. Arg dropout removes a non-id arg.

- [ ] **Step 1: Write failing tests**

```python
def test_lognormal_median() -> None:
    rng = np.random.default_rng(0)
    xs = [sample(DistSpec(family="lognormal", mu=3.6, sigma=0.7), rng) for _ in range(10_000)]
    assert abs(np.median(xs) / np.exp(3.6) - 1) < 0.05

def test_money_rounded_and_categorical_has_rare_values() -> None: ...

def test_small_refund_order_and_id_flow(world) -> None:
    calls = walk(world.tasks["small_refund"], world.persona("support-refunds-a"), world.catalog,
                 np.random.default_rng(1), world.pools("support-refunds-a"))
    tools = [c.tool_id for c in calls]
    assert tools[0] == "zendesk.get_ticket" and "stripe.create_refund" in tools
    refund = next(c for c in calls if c.tool_id == "stripe.create_refund")
    assert refund.args["charge_id"] in calls[0].response_meta.ids          # id flows from ticket

def test_skip_lookup_rate_matches_branch(world) -> None:
    rng = np.random.default_rng(2)
    skipped = sum("stripe.retrieve_charge" not in [c.tool_id for c in walk(..., rng, ...)]
                  for _ in range(10_000))
    assert abs(skipped / 10_000 - 0.05) < 0.01

def test_retry_duplicates_args_and_abandon_truncates(world) -> None: ...
```

- [ ] **Step 2: Run the tests.** Expected: FAIL.
- [ ] **Step 3: Implement** `sample`, `IdPool` and `walk`.
- [ ] **Step 4: Run `./scripts/gate.sh`.** Expected: green.
- [ ] **Step 5: Commit** `git commit -am "feat(synth): distributions, id pools and task-grammar walker"`

### Task 0.6: Sampler, generate CLI, byte-identical determinism

**Files:**
- Create: `agentfw/synth/sampler.py`, `agentfw/synth/generate.py`, `tests/synth/test_sampler.py`, `tests/synth/test_generate.py`

**Interfaces:**
- Produces:
  - `sampler.stable_hash(s: str) -> int` (xxhash64)
  - `sampler.sample_persona(persona, catalog, tasks, *, days: int, global_seed: int, start_ms: int, max_sessions: int | None = None) -> list[list[RawCall]]` (sessions, in time order). Session starts come from a Poisson process inside `working_hours`. In-session gaps are lognormal seconds. `seq_no` runs 0..n−1. `envelope_id = new_ulid(ts, rng)`. `principal_id = f"{agent_id}-principal"`.
  - `generate.GenerateConfig(BaseModel)` (ruling R3: `max_sessions` is a **total** cap split evenly across personas in sorted `agent_id` order, remainder to the first): `seed: int`, `personas_dir: Path`, `tasks_dir: Path`, `catalog_dir: Path`, `days: int`, `start_ms: int = 1_767_225_600_000`, `max_sessions: int | None`, `attack_rate: float = 0.015`, `warmup_sessions: int = 30`, `families: list[str]`, `control_rate: float = 0.005`
  - `generate.Manifest(BaseModel)`: `config`, `files: dict[str, str]` (sha256), `counts: dict[str, int]`
  - `generate.generate(cfg, out_dir: Path) -> Manifest` writes `traces.jsonl` (one `RawCall` per line, orjson `OPT_SORT_KEYS`, sorted by `(ts, session_id, seq_no)`), `labels.jsonl` (one `Label` per envelope) and `manifest.json`
  - CLI: `python -m agentfw.synth.generate --seed 0 --days 30 --out traces/dev [--personas DIR] [--max-sessions N] [--families a,b]`

- [ ] **Step 1: Write failing tests**

```python
def test_same_seed_byte_identical(tmp_path: Path) -> None:
    m1 = generate(cfg(seed=0, days=3), tmp_path / "a")
    m2 = generate(cfg(seed=0, days=3), tmp_path / "b")
    assert m1.files == m2.files
    assert generate(cfg(seed=1, days=3), tmp_path / "c").files != m1.files

def test_sessions_well_formed(tmp_path: Path) -> None:
    raws = load_traces(...)                     # small helper reading traces.jsonl
    # ids unique; seq_no contiguous per session; ts non-decreasing within session;
    # session starts inside working hours; each envelope has exactly one label row

def test_session_rate_mean(world) -> None:   # 30 days ⇒ mean sessions/day within 10% of lam
    ...
```

- [ ] **Step 2: Run the tests.** Expected: FAIL.
- [ ] **Step 3: Implement** the sampler and CLI. With `families=[]`, every label is benign. Attacks are wired in Task 0.7.
- [ ] **Step 4: Run `./scripts/gate.sh`.** Expected: green.
- [ ] **Step 5: Commit** `git commit -am "feat(synth): deterministic sampler and generate CLI"`

### Task 0.7: Attack injector (3 families) and schedule

**Files:**
- Create: `agentfw/synth/attacks.py`, `tests/synth/test_attacks.py`
- Modify: `agentfw/synth/generate.py` (call `inject`)

**Interfaces:**
- Produces:
  - `AttackContext(slots)`: `catalog`, `persona`, `pools`, `attack_id: str`, `persona_sessions: Sequence[list[RawCall]]`, `session_index: int`
  - `AttackResult(slots)`: `session: list[RawCall]`, `labels: list[Label]` (attack labels only; the caller fills benign)
  - `AttackFn = Callable[[list[RawCall], Generator, AttackContext], AttackResult | None]` (None = not applicable to this session)
  - `ATTACKS: dict[str, AttackFn]`, `eligible(family: str, persona: Persona, catalog: Catalog) -> bool`
  - Single-call mutators, reused by the sandbox in Task 2.7: `mutate_scale(raw, path, factor) -> RawCall`, `mutate_replace(raw, path, value) -> RawCall`, `fabricate_id(prefix, rng) -> str`, `novel_value(kind: Literal["email","categorical","id"], rng, prefix=None) -> str` (emails at `x-<rand>.test`)
  - `inject(sessions_by_persona: dict[str, list[list[RawCall]]], cfg: GenerateConfig, rng: Generator, ctx_factory) -> tuple[dict[str, list[list[RawCall]]], list[Label]]`

Families in this task: `magnitude_spike` (scale a money/count field of a write by `m`, log-uniform [3,100], `strength=m`), `prereq_skip` (remove every read in the session that precedes the target write and replace the write's id args with `fabricate_id`), `novel_target` (replace an email/stable categorical/destination id in a write with `novel_value`). Schedule: each eligible persona session is attacked with probability `attack_rate`, never within its first `warmup_sessions`. The family is chosen uniformly among the eligible ones. `attack_id = f"atk-{agent}-{n:05d}"`.

- [ ] **Step 1: Write failing tests**

```python
def test_attacks_are_pure_functions(world, benign_session) -> None:
    before = copy.deepcopy(benign_session)
    ATTACKS["magnitude_spike"](benign_session, np.random.default_rng(0), ctx())
    assert benign_session == before

def test_magnitude_spike_scales_and_labels(world, benign_session) -> None:
    r = ATTACKS["magnitude_spike"](benign_session, np.random.default_rng(0), ctx())
    assert r and len(r.labels) == 1 and 3 <= r.labels[0].strength <= 100

def test_prereq_skip_removes_lookup_and_fabricates(world, benign_session) -> None:
    r = ATTACKS["prereq_skip"](...)
    write = next(c for c in r.session if c.envelope_id == r.labels[0].envelope_id)
    earlier_ids = {i for c in r.session if c.seq_no < write.seq_no
                   for i in (c.response_meta.ids if c.response_meta else ())}
    assert write.args["charge_id"] not in earlier_ids
    assert "stripe.retrieve_charge" not in [c.tool_id for c in r.session if c.seq_no < write.seq_no]

def test_schedule_rate_and_warmup(tmp_path: Path) -> None:
    # 3 personas × 60 days: attacked fraction of eligible sessions in [0.010, 0.020];
    # no attacked session among the first 30 sessions of any persona

def test_attack_stream_does_not_perturb_benign(tmp_path: Path) -> None:
    # sessions untouched by attacks are byte-identical between attack_rate=0 and attack_rate=0.015
```

- [ ] **Step 2: Run the tests.** Expected: FAIL.
- [ ] **Step 3: Implement** the families, the mutators and `inject`. Renumber `seq_no` after a deletion.
- [ ] **Step 4: Run `./scripts/gate.sh`.** Expected: green.
- [ ] **Step 5: Commit** `git commit -am "feat(synth): attack injector with magnitude_spike, prereq_skip, novel_target"`

### Task 0.8: Harness protocol and chronological replay

**Files:**
- Create: `agentfw/bench/protocol.py`, `agentfw/bench/dataset.py`, `agentfw/bench/replay.py`, `tests/bench/test_replay.py`

**Interfaces:**
- Produces:
  - `protocol.Detector(Protocol)`: `reset() -> None`, `score(e: ActionEnvelope) -> DetectorResult`, `update(e: ActionEnvelope, outcome: Outcome) -> None`, `state_bytes() -> int`
  - `protocol.BatchFit(Protocol)`: `fit(train: Sequence[ActionEnvelope]) -> None`
  - `dataset.load_traces(path) -> list[RawCall]`, `load_labels(path) -> dict[str, Label]`, `load_dir(dir) -> tuple[list[RawCall], dict[str, Label], Manifest]`
  - `replay.SplitSpec(kind: Literal["time","agent"] = "time", train_frac: float = 0.7, seed: int = 0)`
  - `replay.assign_split(raws, spec) -> dict[str, Literal["train","test"]]`. Time split: `cut = t_min + train_frac·(t_max − t_min)`. Agent split: sorted agent ids, `rng.permutation`, first 70% train.
  - `replay.ScoreRow(slots)`: `envelope_id, session_id, agent_id, tool_id, ts_unix_ms, split, fused_score, per_detector_json: str, label, attack_family, attack_id, injection_level, strength, step, control_type, is_warm, n_key_obs, latency_us, threshold`
  - `replay.replay(raws, labels, detector, split: SplitSpec, *, warmup_min_obs: int = 30, sample_ids: set[str] | None = None, strict: bool = False) -> list[ScoreRow]`. If `strict`, raises `LeakError` when `state_bytes()` changes across `score()`. Envelopes outside `sample_ids` are still updated but get `fused_score = nan`.
  - `replay.write_score_table(rows, path)` / `read_score_table(path)` (Parquet); `canonical_hash(rows) -> str` (D-019)

- [ ] **Step 1: Write failing tests**

```python
class Spy:
    def __init__(self) -> None: self.calls: list[str] = []; self.seen_flat: list[bool] = []
    def reset(self) -> None: ...
    def score(self, e): self.calls.append(f"score:{e.envelope_id}"); self.seen_flat.append(bool(e.flat_args)); return DetectorResult(0.0, (), (), 0.0)
    def update(self, e, o): assert o is Outcome.ALLOWED; self.calls.append(f"update:{e.envelope_id}")
    def state_bytes(self) -> int: return 0

def test_score_strictly_before_update_in_ts_order(raws3) -> None:
    spy = Spy(); replay(list(reversed(raws3)), {}, spy, SplitSpec())
    ids = [r.envelope_id for r in sorted(raws3, key=lambda r: (r.ts_unix_ms, r.session_id, r.seq_no))]
    assert spy.calls == [x for i in ids for x in (f"score:{i}", f"update:{i}")]
    assert all(spy.seen_flat)

def test_tie_order_deterministic() -> None: ...       # equal ts across sessions → (session_id, seq_no)
def test_is_warm_flips_after_30_prior_obs() -> None:   # 31st call of (agent, tool) is warm; n_key_obs = 30
def test_batchfit_receives_train_only() -> None: ...
def test_strict_mode_catches_leaky_score() -> None: ...
def test_score_table_parquet_roundtrip_and_hash(tmp_path) -> None: ...
```

- [ ] **Step 2: Run the tests.** Expected: FAIL.
- [ ] **Step 3: Implement** the modules. Keep a per-`(agent, tool)` observation counter inside `replay` (detector-agnostic). Call `BatchFit.fit` before the loop if the detector implements it. Time `score()` with `perf_counter_ns`.
- [ ] **Step 4: Run `./scripts/gate.sh`.** Expected: green.
- [ ] **Step 5: Commit** `git commit -am "feat(bench): detector protocol and score-before-update replay"`

### Task 0.9: Metrics and harness self-tests

**Files:**
- Create: `agentfw/bench/metrics.py`, `tests/bench/test_metrics.py`

**Interfaces:**
- Consumes: `ScoreRow`
- Produces (numpy only):
  - `auprc(scores: NDArray[np.float64], y: NDArray[np.bool_]) -> float` (average precision with tie groups)
  - `fpr_at_recall(scores, y, recall: float = 0.95) -> tuple[float, float]` returning `(threshold, fpr)` (D-007)
  - `per_agent_fpr_at_recall(rows, recall=0.95) -> PerAgentResult(alpha, fpr, recall, thresholds)`. Each agent's threshold is the `1−α` quantile of its **train-split benign** scores, computed exactly per row from the train ECDF (rulings D-021/M1: pick the **smallest** α whose test recall ≥ target; unreachable → FPR NaN; `Metrics.per_agent_recall` records the recall achieved).
  - `Thresholds = float | Mapping[str, float] | Literal["row"]` (`"row"` = each row's own `ScoreRow.threshold`, i.e. the engine's live per-agent threshold)
  - `recall_by_family(rows, thresholds: Thresholds, level: Literal["envelope","attack"]) -> dict[str, dict[str, RecallCell]]` with `STRENGTH_BUCKETS = {"magnitude_spike": (3, 10, 100), "scope_blowup": (100, 1000, 10000), "velocity_burst": (10, 30, 60), "slow_drift": (0.03, 0.06, 0.10), "exfil_volume": (10, 30, 100)}` plus an `"all"` bucket. Bucket keys are strings `"3-10"`, `"10-100"`, `"all"`.
  - `benign_fpr(rows, thresholds: Thresholds) -> float`; `top_reason_share(rows, family: str, detector: str, thresholds: Thresholds) -> float` (share of detected attacks of `family` whose highest per-detector score came from `detector`, read from `per_detector_json`)
  - `control_fpr(rows, thresholds) -> float`; `cold_start_curve(rows, thresholds) -> list[BucketCell]` with buckets `(0,10),(10,30),(30,100),(100,inf)` on `n_key_obs`
  - `drift_delay(rows, thresholds) -> DriftDelay(median, p90, miss_rate, n)`. Sessions are counted by `Label.step`; an undetected attack counts as k.
  - `latency_summary(rows) -> tuple[float, float]` (p50, p99 µs)
  - `bootstrap_ci(rows, metric: Callable[[Sequence[ScoreRow]], float], n: int = 1000, seed: int = 0, level: float = 0.95) -> tuple[float, float]`, resampling **sessions**
  - `Metrics(slots)`: `auprc, auprc_ci, thr95_global, fpr95_global, recall_at_global_thr, fpr95_per_agent, fpr95_per_agent_ci, per_agent_alpha, per_family_envelope, per_family_attack, control_fpr, cold_start, drift, latency_p50_us, latency_p99_us, n_rows, n_attack_rows`; `compute_all(rows, *, warm_only: bool) -> Metrics` (test split only)

- [ ] **Step 1: Write failing tests**

```python
def test_perfect_scores() -> None:
    y = np.array([0, 0, 1, 0, 1], bool); s = y.astype(float)
    assert auprc(s, y) == 1.0 and fpr_at_recall(s, y)[1] == 0.0

def test_random_scores_auprc_near_base_rate() -> None:
    rng = np.random.default_rng(0); y = rng.random(20_000) < 0.05
    assert abs(auprc(rng.random(20_000), y) - y.mean()) < 0.01

def test_all_tied_scores_equal_base_rate() -> None:
    y = np.array([1, 0, 0, 0], bool); assert auprc(np.zeros(4), y) == 0.25

def test_operating_point_is_highest_threshold_meeting_recall() -> None:
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.1]); y = np.array([1, 1, 0, 1, 0], bool)
    assert fpr_at_recall(s, y, 0.95) == (0.6, 0.5)

def test_per_agent_beats_global_on_scaled_agents() -> None:
    # agent A benign ~U(0,0.2) attacks 0.3; agent B benign ~U(0.5,0.7) attacks 0.8
    # per-agent FPR == 0.0 while global FPR > 0.4

def test_bootstrap_ci_contains_point_and_resamples_sessions() -> None: ...
def test_attack_level_recall_counts_any_envelope() -> None: ...
def test_drift_delay_counts_sessions_and_misses() -> None: ...
```

- [ ] **Step 2: Run the tests.** Expected: FAIL.
- [ ] **Step 3: Implement** `metrics.py`.
- [ ] **Step 4: Run `./scripts/gate.sh`.** Expected: green.
- [ ] **Step 5: Commit** `git commit -am "feat(bench): metrics with per-agent thresholds and session bootstrap"`

### Task 0.10: Baselines — random and static_threshold

**Files:**
- Create: `agentfw/bench/baselines/__init__.py`, `random_.py`, `static_threshold.py`, `agentfw/bench/registry.py`, `tests/bench/test_baselines.py`

**Interfaces:**
- Produces:
  - `RandomDetector(seed: int = 0)`: score = uniform from `stable_hash(f"{seed}:{envelope_id}")`, so repeated calls agree
  - `StaticThreshold(quantile: float = 0.99)` implementing `Detector` and `BatchFit`. On numeric paths (flat+derived values where `try_number` succeeds and the value is not id-like) it uses a per-(tool, path) upper bound from train. Categorical paths (non-id strings, plus derived `__domain__`/`__host__`/`__prefix__`) use a per-(tool, path) train allowlist. Score = 1.0 on any violation, else 0.0. `update` is a no-op.
  - `registry.DETECTORS: dict[str, Callable[..., Detector]]`, `make_detector(name: str, **kw) -> Detector`

- [ ] **Step 1: Write failing tests**: random is deterministic per id and in [0,1), and its scores have mean ≈ 0.5 over 10k. Static: fit on amounts 10..100 → amount 1000 scores 1.0 and amount 50 scores 0.0; an unseen `reason` scores 1.0; `update` leaves `state_bytes()` unchanged.
- [ ] **Step 2: Run the tests.** Expected: FAIL.
- [ ] **Step 3: Implement** both baselines and the registry (`"random"`, `"static_threshold"`).
- [ ] **Step 4: Run `./scripts/gate.sh`.** Expected: green.
- [ ] **Step 5: Commit** `git commit -am "feat(bench): random and static_threshold baselines"`

### Task 0.11: Report and run CLI

**Files:**
- Create: `agentfw/bench/report.py`, `agentfw/bench/run.py`, `tests/bench/test_report.py`, `tests/conftest.py` (session fixture `tiny_dataset`: seed 0, 3 personas, 5 days, 3 families, `attack_rate=0.15` per ruling R2)

**Interfaces:**
- Produces:
  - `RunMeta(BaseModel)`: `git_sha` (`git rev-parse HEAD` or `"nogit"`), `seed`, `dataset_sha` (from manifest), `detector`, `detector_config: dict`, `split: SplitSpec`, `created_unix`
  - `build_report(all_: Metrics, warm: Metrics, meta: RunMeta, extras: Mapping[str, object] = {}) -> dict[str, object]` with top-level keys `headline` (`auprc`, `auprc_ci`, `fpr95_global`, `fpr95_per_agent`, `fpr95_per_agent_ci`, `warm_only`), `per_family`, `control_fpr`, `cold_start`, `drift`, `latency`, `state_bytes`, `meta`, `extras`
  - `render_markdown(report) -> str`; `update_readme(readme: Path, md: str) -> None` (replaces text between `<!-- report:start -->` and `<!-- report:end -->`)
  - `run(traces_dir: Path, detector: str, split: SplitSpec, out_dir: Path, **detector_kw) -> Path` writes `scores.parquet`, `report.json`, `report.md`
  - CLI `python -m agentfw.bench.run --traces DIR --detector NAME [--split time|agent] --out DIR [--update-readme]`

- [ ] **Step 1: Write failing tests**: `run()` on `tiny_dataset` with `static_threshold` creates all three files. The report has every top-level key above. The markdown contains `| Family |` and `FPR@95`. `update_readme` is idempotent (running it twice gives the same file).
- [ ] **Step 2: Run the tests.** Expected: FAIL.
- [ ] **Step 3: Implement** the report and CLI. Add the README markers.
- [ ] **Step 4: Run `./scripts/gate.sh`.** Expected: green.
- [ ] **Step 5: Commit** `git commit -am "feat(bench): report.json/report.md and run CLI"`

### Task 0.12: Generator validation (§4.7), CI regression, golden — Tier 0 exit

**Files:**
- Create: `tests/synth/test_generator_validation.py`, `agentfw/bench/ci_regress.py`, `agentfw/bench/baseline_metrics.json`, `scripts/update_golden.py`, `tests/golden/static_threshold.sha256`, `tests/bench/test_ci_regress.py`

**Interfaces:**
- Produces:
  - `ci_regress.CI_DATASET = GenerateConfig(seed=0, 3 personas, days=60, max_sessions=200, attack_rate=0.15, families=[current families])` (ruling R2: 1.5% would leave ~2 attacks in 200 sessions)
  - `ci_regress.compare(current: Mapping[str, float], baseline: Mapping[str, float]) -> list[str]` (failure messages; AUPRC drop > 0.01 or FPR@95 per-agent rise > 0.002)
  - `ci_regress.main() -> int`: runs every detector named in `baseline_metrics.json`, prints its report, returns 1 on any failure. `--write-baseline` rewrites the file.
  - `scripts/update_golden.py --detector NAME` rewrites `tests/golden/<name>.sha256`

- [ ] **Step 1: Write failing tests**

```python
def test_non_triviality_static_threshold(validation_dataset) -> None:   # 3 personas, 60 days
    rows = replay(*validation_dataset, make_detector("static_threshold"), SplitSpec())
    m = compute_all(rows, warm_only=False)
    assert not (m.recall_at_global_thr >= 0.95 and m.fpr95_global < 0.01)

def test_amount_distribution_matches_persona(world) -> None:
    amts = sampled_field("support-refunds-a", "stripe.create_refund.amount", n=10_000)  # benign, pre-frac
    assert scipy.stats.kstest(amts, "lognorm", args=(0.7, 0, np.exp(3.6))).pvalue > 0.01

def test_prereq_base_rate_matches_grammar(validation_dataset) -> None:
    # fraction of create_refund with retrieve_charge in prior_tools within 0.01 of 1 - skip_lookup

def test_label_integrity(validation_dataset) -> None:
    # every attack_id has ≥1 attack label; no label is attack with control_type set; 1 label per envelope

def test_compare_boundaries() -> None:
    base = {"auprc": 0.80, "fpr95": 0.010}
    assert compare({"auprc": 0.790, "fpr95": 0.012}, base) == []
    assert len(compare({"auprc": 0.789, "fpr95": 0.0121}, base)) == 2

def test_golden_static_threshold() -> None:
    rows = replay(*ci_dataset(), make_detector("static_threshold"), SplitSpec())
    assert canonical_hash(rows) == Path("tests/golden/static_threshold.sha256").read_text().strip()
```

- [ ] **Step 2: Run the tests.** Expected: FAIL.
- [ ] **Step 3: Implement** `ci_regress.py` and `update_golden.py`. Generate the baseline and golden files with them. Add `uv run python -m agentfw.bench.ci_regress` to the CI workflow.
- [ ] **Step 4: Run `./scripts/gate.sh --full` and `uv run python -m agentfw.bench.ci_regress`.** Expected: green, exit 0. **Tier 0 exit:** the harness runs end to end, the generator passes §4.7, and static_threshold does not reach 95%/1%.
- [ ] **Step 5: Commit** `git commit -am "test: generator validation, CI regression and golden hash (Tier 0 exit)"`

---

# Phase 1 — Tier 1 engine

### Task 1.1: EngineConfig, P² quantiles, EWMA

**Files:**
- Create: `agentfw/core/config.py`, `agentfw/core/hints.py`, `agentfw/core/stats/{__init__,p2,ewma}.py`, `tests/core/test_p2.py`, `tests/core/test_ewma.py`

**Interfaces:**
- Produces:
  - `EngineConfig` (frozen slots dataclass with every constant in Global Constraints), `EngineConfig.replace(**kw) -> EngineConfig`
  - `ToolHint(slots, frozen)`: `kind: Literal["read","write"] | None = None`, `field_hints: Mapping[str, Literal["money","count"]] = {}`
  - `P2Quantiles(probs: tuple[float, ...])`: `add(x: float) -> None`, `quantile(p: float) -> float`, `percentile_of(x: float) -> float`, `n: int`, `to_dict() / from_dict()`, `__eq__`
  - `p2_tail(p: float) -> P2Quantiles` (markers `(p/2, p, (1+p)/2)`)
  - `Ewma(alpha: float)`: `add(x)`, `value: float | None`, `to_dict() / from_dict()`

Algorithm: Jain & Chlamtac P² generalized to arbitrary marker probabilities. Markers sit at `(0, *probs, 1)` with desired-position increments `dn'_i = p_i`. Use the piecewise-parabolic height update, falling back to linear when the parabolic estimate breaks marker monotonicity. Until `len(probs)+2` samples have arrived, keep a sorted buffer and return exact quantiles. `quantile(p)` and `percentile_of(x)` interpolate linearly between adjacent markers.

- [ ] **Step 1: Write failing tests**

```python
def test_p2_converges_on_lognormal_logspace() -> None:
    rng = np.random.default_rng(0); est = P2Quantiles((0.05, 0.25, 0.5, 0.75, 0.95))
    for y in rng.normal(3.6, 0.7, 1000): est.add(float(y))
    for p, tol in [(0.25, 0.1), (0.5, 0.08), (0.75, 0.1), (0.05, 0.2), (0.95, 0.2)]:
        assert abs(est.quantile(p) - scipy.stats.norm.ppf(p, 3.6, 0.7)) < tol

def test_p2_exact_below_marker_count() -> None:
    est = P2Quantiles((0.5,)); [est.add(x) for x in (3.0, 1.0)]
    assert est.quantile(0.5) == 2.0

def test_p2_tail_995_on_uniform() -> None:
    rng = np.random.default_rng(1); est = p2_tail(0.995)
    for x in rng.random(100_000): est.add(float(x))
    assert abs(est.quantile(0.995) - 0.995) < 0.003

@given(st.lists(st.floats(-1e6, 1e6), min_size=1, max_size=500))
def test_p2_markers_monotone_and_serializable(xs: list[float]) -> None:
    est = P2Quantiles((0.25, 0.5, 0.75)); [est.add(x) for x in xs]
    qs = [est.quantile(p) for p in (0.25, 0.5, 0.75)]
    assert qs == sorted(qs) and P2Quantiles.from_dict(est.to_dict()) == est

def test_p2_constant_memory() -> None:   # len(orjson.dumps(to_dict())) identical after 1e3 and 1e5 adds
```

- [ ] **Step 2: Run the tests.** Expected: FAIL.
- [ ] **Step 3: Implement** the modules.
- [ ] **Step 4: Run `./scripts/gate.sh`.** Expected: green.
- [ ] **Step 5: Commit** `git commit -am "feat(core): EngineConfig and P2 streaming quantiles"`

### Task 1.2: State store, records, snapshots

**Files:**
- Create: `agentfw/core/state.py`, `tests/core/test_state.py`

**Interfaces:**
- Produces:
  - `Record(Protocol)`: `KIND: ClassVar[str]`, `to_dict() -> dict[str, object]`, `from_dict(d) -> Self`
  - `register_record(cls) -> cls` (decorator; registry keyed by `KIND`)
  - `StateStore(Protocol)`: `get(key: str) -> Record | None`, `put(key: str, rec: Record) -> None`, `items() -> Iterator[tuple[str, Record]]`, `nbytes() -> int`, `dump(path: Path) -> None`, `load(path: Path) -> None`, `clear() -> None`
  - `DictStore` (implementation). `nbytes` is the sum of `len(orjson.dumps(rec.to_dict()))`, maintained incrementally on `put`. `dump` writes JSONL rows `{"key", "kind", "data"}`.
  - Key builders: `key_type(a, t, p)`→`t:`, `key_mag`→`m:`, `key_vel(a, t, p=None)`→`v:`, `key_nov`→`n:`, `key_trans(a)`→`s:{a}:trans`, `key_prereq(a, t)`→`s:{a}:prereq:{t}`, `key_flow`→`f:`, `key_cal(a)`→`c:{a}`, `key_kind(a, t)`→`k:{a}:{t}`, `key_prior_mag(t, p)`→`p:m:`, `key_prior_nov(t, p)`→`p:n:`. Each segment escapes `%`→`%25` and `:`→`%3A`.

- [ ] **Step 1: Write failing tests**: key escaping round-trips (a path containing `:`); hypothesis property `load(dump(S)) == S` over a toy `@register_record` type with random fields; `nbytes` equals the recomputed sum after random puts and overwrites; `clear()` resets `nbytes` to 0.
- [ ] **Step 2: Run the tests.** Expected: FAIL.
- [ ] **Step 3: Implement** `state.py`.
- [ ] **Step 4: Run `./scripts/gate.sh`.** Expected: green.
- [ ] **Step 5: Commit** `git commit -am "feat(core): keyed state store with JSONL snapshots"`

### Task 1.3: Type inference

**Files:**
- Create: `agentfw/core/schema_infer.py`, `tests/core/test_schema_infer.py`

**Interfaces:**
- Consumes: `DictStore`, `EngineConfig`, `patterns.*`
- Produces:
  - `TypeRecord` (`KIND="type"`): `n_numeric, n_bool, n_idlike, n_total, sum_len`, `distinct: set[str]` (capped at 1024, plus `distinct_overflow`), `reservoir: list[str]` (64, Algorithm R seeded from `stable_hash(key)`), `window: deque[tuple[bool, bool, bool, int, str]]` (maxlen 100), `type: FieldType`, `confidence: float`, `changes: int`
  - `TypeChange(slots)`: `key`, `old: FieldType`, `new: FieldType`, `n_total`
  - `TypeInference(store, cfg)`: `type_of(agent, tool, path) -> FieldType` (pure); `observe(agent, tool, path, values: tuple[Scalar, ...]) -> TypeChange | None`
  - `TypeView(inference, agent, tool)`: `__call__(path) -> FieldType`. Derived suffixes are fixed by construction: `__len__`/`__strlen__`/`__truncated__` → NUMERIC; `__domain__`/`__host__`/`__prefix__` → CATEGORICAL.

Rules (§6.2), evaluated over cumulative counts for the first assignment (at `n_total ≥ 10`) and over the last 100 observations for any change:

| Type | Rule |
|---|---|
| bool | values ⊆ {true, false, 0, 1} |
| numeric | ≥ 95% parse via `try_number` and are not id-like |
| id | ≥ 90% id-like, or (`n ≥ 50` and distinct/n ≥ 0.9 and mean length ≤ 40) |
| freetext | mean length > 40 and distinct/n > 0.9 |
| categorical | otherwise |

Stickiness: a change happens only when the 100-observation window is full and its rule result differs from the current type.

- [ ] **Step 1: Write failing tests**

```python
def test_unknown_until_ten_obs(ti) -> None:
    for i in range(9): ti.observe("a", "t", "amount", (float(i),))
    assert ti.type_of("a", "t", "amount") is FieldType.UNKNOWN
    ti.observe("a", "t", "amount", (9.0,)); assert ti.type_of("a", "t", "amount") is FieldType.NUMERIC

def test_numeric_strings_infer_numeric(ti) -> None: ...      # "37.50" ×20 → NUMERIC
def test_ids_emails_categoricals_bools(ti) -> None: ...      # ch_… → ID; emails → ID; {usd,eur,gbp} → CATEGORICAL; True/False → BOOL
def test_type_is_sticky_then_changes(ti) -> None:
    for i in range(200): ti.observe("a", "t", "f", (float(i),))
    for _ in range(99): assert ti.observe("a", "t", "f", ("x",)) is None
    ch = ti.observe("a", "t", "f", ("x",))
    assert ch and ch.new is FieldType.CATEGORICAL
def test_per_agent_independent_and_type_of_is_pure(ti, store) -> None: ...
```

- [ ] **Step 2: Run the tests.** Expected: FAIL.
- [ ] **Step 3: Implement** `schema_infer.py`.
- [ ] **Step 4: Run `./scripts/gate.sh`.** Expected: green.
- [ ] **Step 5: Commit** `git commit -am "feat(core): sticky per-agent type inference"`

### Task 1.4: Magnitude detector (incl. response metadata)

**Files:**
- Create: `agentfw/core/detectors/{__init__,base,magnitude}.py`, `tests/core/test_magnitude.py`

**Interfaces:**
- Consumes: `P2Quantiles`, `Ewma`, `TypeView`, `DictStore`, `ToolHint`
- Produces:
  - `base.SubDetector(Protocol)`: `name: str`; `score(e: ActionEnvelope, types: TypeView) -> list[DetectorScore]`; `update(e: ActionEnvelope, types: TypeView) -> None` (called only for allowed outcomes)
  - `base.squash(z: float, z0: float, tau: float) -> float` = `1/(1+exp(-(z-z0)/tau))`; `base.signed_log1p(x: float) -> float`; `base.iter_fields(e) -> Iterator[tuple[str, tuple[Scalar, ...]]]` (flat then derived)
  - `magnitude.direction_for(path: str, hints: Mapping[str, str]) -> Literal["up","both"]` (`up` if the hint is money/count, or the last segment matches `(?i)amount|count|qty|total|len`)
  - `MagRecord` (`KIND="mag"`): `q: P2Quantiles` over `mag_probs`, `n`, `ewma`, `ewma2`, `direction`
  - `MagnitudeDetector(store, cfg, priors: "PooledPriors | None" = None, hints: Mapping[str, ToolHint] = {})`
  - Response field: when `e.prev_response` has an `item_count`, score key `m:{agent}:{prev_tool}:__resp__.item_count`, field name `"__resp__.item_count"`, evidence `{"tool": prev_tool}`. Update pushes the current envelope's own `response_meta.item_count` into `m:{agent}:{tool}:__resp__.item_count`.

Scoring (§6.3). For each NUMERIC path, take the max over its values. `y = signed_log1p(x)`, `z = (y − q50) / max((q75−q25)/1.349, mag_eps)`. With `direction=up`, `z = max(z, 0)`; otherwise `z = |z|`. Score `s = squash(z, z0, tau)`. Evidence: `percentile`, `median_raw`, `iqr_raw`, `n`, `direction`, `prior_used`. If `n < mag_cold_n`, use the prior (Task 1.10); without a prior, emit `s = 0.0` with `evidence["cold"] = True`. NaN is skipped. ±inf scores 1.0 with `evidence["non_finite"] = True` and is never pushed.

- [ ] **Step 1: Write failing tests**

```python
def test_squash_anchor_points() -> None:
    assert squash(3.0, 3.0, 0.75) == 0.5 and abs(squash(5.25, 3.0, 0.75) - 0.95) < 1e-3

def test_warm_spike_and_normal(trained_mag) -> None:      # 500 lognormal(3.6,0.7) refund amounts
    assert score_amount(trained_mag, 37.0) < 0.1
    assert score_amount(trained_mag, 37.0 * 20) > 0.9
    assert score_amount(trained_mag, 37.0 / 20) < 0.05     # direction=up ignores small

def test_direction_both_flags_small(trained_both) -> None: ...   # field "latitude"
@given(st.floats(0, 1e7), st.floats(0, 1e7))
def test_monotone_in_x_for_direction_up(trained_mag, a: float, b: float) -> None:
    lo, hi = sorted((a, b)); assert score_amount(trained_mag, lo) <= score_amount(trained_mag, hi)
def test_score_is_idempotent_and_pure(trained_mag, store) -> None: ...
def test_cold_without_prior_scores_zero() -> None: ...
def test_non_finite_values(trained_mag) -> None: ...   # nan → no score; inf → 1.0 and n unchanged after update
def test_constant_field_is_finite() -> None: ...       # 100×10.0 then 10.0 → s<0.05; 20.0 → s>0.9
def test_response_item_count_scored_on_next_call() -> None: ...  # item_count 50× typical → field "__resp__.item_count" s>0.9
```

- [ ] **Step 2: Run the tests.** Expected: FAIL.
- [ ] **Step 3: Implement** `base.py` and `magnitude.py`.
- [ ] **Step 4: Run `./scripts/gate.sh`.** Expected: green.
- [ ] **Step 5: Commit** `git commit -am "feat(core): magnitude detector with robust z and response-meta scoring"`

### Task 1.5: Novelty detector and count-min sketch

**Files:**
- Create: `agentfw/core/sketch/cms.py`, `agentfw/core/detectors/novelty.py`, `tests/core/test_cms.py`, `tests/core/test_novelty.py`

**Interfaces:**
- Produces:
  - `CountMinSketch(width=2048, depth=4)`: numpy `uint32` table; `add(v: str, c: int = 1)`, `estimate(v) -> int`, `to_dict()` (base64 of bytes) / `from_dict()`
  - `NoveltyRecord` (`KIND="nov"`): `seen: OrderedDict[str, int]` (LRU ≤ `novelty_lru`), `n_total`, `n_singletons`, `n_distinct`, `evicted: CountMinSketch | None` (allocated on the first eviction, D-011)
  - `NoveltyDetector(store, cfg, priors=None)`: scores CATEGORICAL and ID paths plus derived `__domain__`/`__host__`/`__prefix__`. A value is "seen" if it is in the LRU or `evicted.estimate(v) > 0`. Seen ⇒ `s = 0`. Unseen ⇒ `s = (1 − n_singletons/n_total) · min(1, n_total/novelty_full_n)`. Take the max per path. Evidence: `p_unseen`, `n_total`, `n_distinct`, `value_hash`. Raw values are not put in evidence when the type is ID.

- [ ] **Step 1: Write failing tests**

```python
def test_second_sighting_is_not_novel(nov) -> None: ...
def test_stable_vocab_exact_formula(nov) -> None:
    feed(nov, "currency", ["usd", "eur", "gbp"] * 100)            # n=300, no singletons
    assert score(nov, "currency", "jpy") == pytest.approx(1.0)
def test_half_warm_formula(nov) -> None:
    feed(nov, "currency", ["usd", "eur"] * 50)                    # n=100
    assert score(nov, "currency", "jpy") == pytest.approx(0.5)
def test_singleton_heavy_field_scores_low(nov) -> None:           # all-unique customer ids → s < 0.05
def test_new_address_known_domain_vs_new_domain(nov) -> None: ... # to[] ≈0, __domain__ 0; new domain high
def test_case_variant_domain_not_novel(nov) -> None: ...         # "Bob@ACME.example" after "acme.example"
def test_lru_bound_and_evicted_still_seen(nov) -> None: ...      # 20k distinct ⇒ len(seen) ≤ 10_000; evicted value → 0
def test_cms_overestimates_never_under() -> None: ...
```

- [ ] **Step 2: Run the tests.** Expected: FAIL.
- [ ] **Step 3: Implement** `cms.py` and `novelty.py`. When an evicted value reappears, re-insert it with count `estimate + 1` and decrement `n_singletons` if the estimate was 1.
- [ ] **Step 4: Run `./scripts/gate.sh`.** Expected: green.
- [ ] **Step 5: Commit** `git commit -am "feat(core): Good-Turing novelty detector with LRU and CMS spill"`

### Task 1.6: Velocity detector

**Files:**
- Create: `agentfw/core/detectors/velocity.py`, `tests/core/test_velocity.py`

**Interfaces:**
- Produces:
  - `VelocityRecord` (`KIND="vel"`): `short: list[tuple[int, float]]` (ring of `vel_short_min` per-minute `(count, sum)`), `long: dict[int, tuple[int, float]]` (sparse, pruned to `vel_long_min` minutes; D-010), `last_minute: int`, `hist_count: P2Quantiles((0.25, 0.5, 0.75))` and `hist_sum` over `log1p` of the window totals
  - `VelocityDetector(store, cfg, hints={})`: keys `v:{agent}:{tool}` (count) and `v:{agent}:{tool}:{path}` for money fields (`direction_for == "up"` and name matches `amount|total`, or hint money)

Clock: `minute = e.ts_unix_ms // 60_000`. Advancing from `last_minute` to `minute` rolls the ring and samples the window total into history once per elapsed minute, but **only while the window is non-empty**. At most `vel_short_min + 1` iterations are needed after idle. Score: `y = log1p(window_count_including_this_call)`, robust `z` against `hist_count` with floor `vel_eps`, one-sided, squashed with `z0`/`tau`. Do the same for sum and score the max. Cold (`hist.n < vel_cold_n`) ⇒ 0. Evidence: `window_count`, `baseline_median`, `n_hist`. `score` computes the virtual advance without mutating.

- [ ] **Step 1: Write failing tests**: one call every 5 min over 3 working days, then 40 median-sized calls within 2 min → the 15th burst call scores > 0.95; normal cadence scores < 0.2; a 12 h idle gap adds ≤ 11 history samples; score is idempotent; `len(long) ≤ 1441`; `state_bytes` per key < 30 KB after a simulated week.
- [ ] **Step 2: Run the tests.** Expected: FAIL.
- [ ] **Step 3: Implement** `velocity.py`.
- [ ] **Step 4: Run `./scripts/gate.sh`.** Expected: green.
- [ ] **Step 5: Commit** `git commit -am "feat(core): sliding-window velocity detector"`

### Task 1.7: Sequence detector — transitions and mined prerequisites

**Files:**
- Create: `agentfw/core/detectors/sequence.py`, `tests/core/test_sequence_trans.py`

**Interfaces:**
- Produces:
  - `TransRecord` (`KIND="trans"`): `c2: dict[str, dict[str, int]]` (context key `"a|b"`), `c1`, `c0: dict[str, int]`, `tools: set[str]`
  - `PrereqRecord` (`KIND="prereq"`): `n_y: int`, `cnt: dict[str, int]`, `cached: tuple[int, frozenset[str]]` (n_y at compute time, set)
  - `SequenceDetector(store, cfg, hints={})` with internal `_s_trans(e) -> tuple[float, dict]`, `_s_prereq(e) -> tuple[float, dict]`; public `score` returns one `DetectorScore(detector="sequence", field=None, score=max(parts), evidence={"fired": name, ...})`

Transitions (§6.6a): context = last two of `prior_tools`, padded with `"<s>"`. Use k = 2 if that context total ≥ `trans_min_ctx`, else k = 1 if its total ≥ 20, else k = 0. `V = len(tools) + 1`. `P = (c + 0.5)/(total + 0.5·V)`. `H` = entropy of the smoothed context distribution, `H_max = ln V`. `s_trans = (1 − P)·(1 − H/H_max)`, and 0 if `V ≤ 2`. Prerequisites (§6.6b): `n_y ≥ prereq_min_n` and `cnt[X]/n_y ≥ prereq_rate` ⇒ X is a prerequisite. `s_prereq = 1` if any prerequisite is absent from `prior_tools`, with evidence `prereq_missing`.

- [ ] **Step 1: Write failing tests**

```python
def test_skip_in_deterministic_agent_scores_high(seq) -> None:   # 100 sessions A→B→C; A→C > 0.8
def test_surprise_discounted_in_high_entropy_agent(seq) -> None:  # uniform over 8 tools → < 0.2
def test_backoff_to_k1_when_context_sparse(seq) -> None: ...      # evidence["k"] == 1
def test_prereq_appears_exactly_at_50th_obs(seq) -> None:
    for i in range(49): obs_y_with_x(seq)
    assert prereqs(seq, "Y") == frozenset()
    obs_y_with_x(seq); assert prereqs(seq, "Y") == frozenset({"X"})
def test_prereq_not_mined_at_94_percent(seq) -> None: ...
def test_missing_prereq_scores_one_with_evidence(seq) -> None: ...
```

- [ ] **Step 2: Run the tests.** Expected: FAIL.
- [ ] **Step 3: Implement** the transition and prerequisite parts.
- [ ] **Step 4: Run `./scripts/gate.sh`.** Expected: green.
- [ ] **Step 5: Commit** `git commit -am "feat(core): sequence transitions and mined prerequisites"`

### Task 1.8: Sequence detector — id flow and write-kind inference

**Files:**
- Modify: `agentfw/core/detectors/sequence.py`
- Create: `tests/core/test_sequence_flow.py`

**Interfaces:**
- Produces:
  - `ToolKindRecord` (`KIND="toolkind"`, key `k:{agent}:{tool}`): `has_item_count: bool`, `has_id_arg: bool`
  - `FlowRecord` (`KIND="flow"`): `present: int`, `total: int`
  - `SequenceDetector.is_write(agent, tool) -> bool` = `hints[tool].kind == "write"` if a hint exists, else `not has_item_count and has_id_arg` (learned from **past** updates only)
  - `_s_flow(e, types) -> tuple[float, dict]`. For a write tool, for each ID-typed flat value absent from `e.session_ids_seen`: `s_flow = present/total` if `total ≥ flow_min_obs`, else 0. Take the max. Evidence: `missing_id_path`, `r_flow`. Update: for each ID-typed value, `total += 1`, and `present += 1` if the value is in `session_ids_seen`.

- [ ] **Step 1: Write failing tests**

```python
def test_flow_score_equals_historical_rate(seq) -> None:   # 19/20 present → fabricated id scores 0.95
def test_external_id_writer_scores_zero(seq) -> None: ...   # r_flow = 0 → 0.0
def test_reads_never_flow_scored(seq) -> None: ...          # tool with item_count history
def test_dummy_lookup_of_different_id_still_fires(seq) -> None:
    # §8 evasion: retrieve_charge(ch_OTHER) then create_refund(ch_FAKE) → sequence score ≥ 0.9, fired == "flow"
```

- [ ] **Step 2: Run the tests.** Expected: FAIL.
- [ ] **Step 3: Implement** id flow and write-kind inference. The sequence score becomes `max(s_trans, s_prereq, s_flow)`.
- [ ] **Step 4: Run `./scripts/gate.sh`.** Expected: green.
- [ ] **Step 5: Commit** `git commit -am "feat(core): id-flow signal and write-kind inference"`

### Task 1.9: Fusion v1 and per-agent calibration

**Files:**
- Create: `agentfw/core/fusion.py`, `agentfw/core/calibration.py`, `tests/core/test_fusion.py`, `tests/core/test_calibration.py`

**Interfaces:**
- Produces:
  - `FusionResult(slots)`: `fused: float`, `reasons: tuple[Reason, ...]`
  - `fuse(scores: Sequence[DetectorScore], cfg: EngineConfig) -> FusionResult`: `top1 + fusion_bonus·[top2 > fusion_bonus_gate]`, clipped to 1. Each score > `reason_min` becomes a Reason, ordered by descending score.
  - `reason_from(s: DetectorScore) -> Reason` (maps evidence keys `median_raw`, `iqr_raw`, `n`, `percentile`, `prereq_missing`, `prior_used`, `observed`)
  - `CalRecord` (`KIND="cal"`): `q: P2Quantiles` (= `p2_tail(1 − cal_alpha)`), `n_scored`
  - `Calibrator(store, cfg)`: `threshold(agent) -> float` (`θ0` at n = 0, else `w·q + (1−w)·θ0` with `w = min(1, n/cal_blend_n)`); `decide(agent, fused) -> Decision`; `observe(agent, fused, outcome)` (only `ALLOWED`/`APPROVED`)

- [ ] **Step 1: Write failing tests**

```python
@pytest.mark.parametrize("scores,expected", [
    ([0.9, 0.6], 1.0), ([0.9, 0.5], 0.9), ([0.7, 0.6, 0.6], 0.85), ([], 0.0), ([0.2], 0.2)])
def test_fusion_v1(scores, expected) -> None:
    assert fuse([ds(s) for s in scores], EngineConfig()).fused == pytest.approx(expected)

def test_reasons_only_above_03_and_ordered() -> None: ...
def test_threshold_blend(cal) -> None:
    assert cal.threshold("a") == 0.6
    for _ in range(250): cal.observe("a", 0.1, Outcome.ALLOWED)
    assert cal.threshold("a") == pytest.approx(0.5 * 0.1 + 0.5 * 0.6, abs=1e-6)
def test_decision_bands(cal) -> None: ...   # ≥θ BLOCK; [θ−0.1, θ) REQUIRE_APPROVAL; else ALLOW
def test_blocked_and_denied_not_learned_approved_is(cal) -> None: ...
```

- [ ] **Step 2: Run the tests.** Expected: FAIL.
- [ ] **Step 3: Implement** fusion and calibration.
- [ ] **Step 4: Run `./scripts/gate.sh`.** Expected: green.
- [ ] **Step 5: Commit** `git commit -am "feat(core): fusion v1 and per-agent threshold calibration"`

### Task 1.10: Pooled cross-agent priors

**Files:**
- Create: `agentfw/core/priors.py`, `tests/core/test_priors.py`
- Modify: `agentfw/core/detectors/magnitude.py`, `agentfw/core/detectors/novelty.py` (consult priors on cold keys)

**Interfaces:**
- Produces: `PooledPriors(store, cfg)`: `observe_numeric(tool, path, y)`, `observe_categorical(tool, path, v)`, `magnitude_ref(tool, path, direction) -> MagRecord | None` (n ≥ `mag_cold_n`; `None` if `priors_mode="off"`, or if `"money_count_only"` and direction ≠ `up`), `novelty_ref(tool, path) -> NoveltyRecord | None`
- Behaviour: magnitude on a cold key uses the prior's quantiles with IQR × `prior_iqr_inflate`. Novelty on a cold key (`n_total < mag_cold_n`) scores against the pooled record × `prior_novelty_discount` (D-012). Both set `prior_used=True`.

- [ ] **Step 1: Write failing tests**: a new agent's first refund of 37 scores < 0.1 with `prior_used`; ×50 scores > 0.9 with `prior_used`; the z with the prior equals the z computed from manually inflated IQR; `priors_mode="off"` ⇒ cold score 0; `"money_count_only"` ⇒ a `latitude` field gets no prior.
- [ ] **Step 2: Run the tests.** Expected: FAIL.
- [ ] **Step 3: Implement** `priors.py` and the detector hooks.
- [ ] **Step 4: Run `./scripts/gate.sh`.** Expected: green.
- [ ] **Step 5: Commit** `git commit -am "feat(core): pooled cross-agent priors for cold keys"`

### Task 1.11: Engine composition, invariants, golden, latency

**Files:**
- Create: `agentfw/core/engine.py`, `tests/core/test_engine.py`, `tests/core/test_engine_props.py`, `tests/test_no_scipy.py`, `tests/perf/test_latency.py`, `tests/golden/engine.sha256`
- Modify: `agentfw/bench/registry.py` (add `engine` and ablations), `agentfw/bench/baseline_metrics.json` (add `engine`)

**Interfaces:**
- Produces:
  - `ALL_DETECTORS = frozenset({"magnitude", "novelty", "velocity", "sequence"})`
  - `Engine(cfg: EngineConfig = EngineConfig(), store: StateStore | None = None, enabled: frozenset[str] = ALL_DETECTORS, hints: Mapping[str, ToolHint] = {})` implementing `Detector`. `score` runs the enabled sub-detectors with a `TypeView`, then `fuse`, `Calibrator.threshold`/`decide`, and returns `DetectorResult` with `field_types` filled. It caches `(envelope_id, fused)`. `update(e, outcome)` does nothing unless the outcome is `ALLOWED`/`APPROVED`. Otherwise, in order: sub-detector updates, priors observe, calibrator observe (with the cached fused, or a fresh score if none is cached), and finally type observe.
  - Registry names: `engine`, `engine-no-{magnitude,novelty,velocity,sequence}`, `engine-only-{magnitude,novelty,velocity,sequence}`, `engine-no-priors`

- [ ] **Step 1: Write failing tests**

```python
@given(envelope_streams())
def test_score_idempotent_and_no_leak(stream) -> None:
    eng = Engine(); [ (eng.score(e), eng.update(e, Outcome.ALLOWED)) for e in stream[:-1] ]
    before = eng.state_bytes(); a = eng.score(stream[-1]); b = eng.score(stream[-1])
    assert strip_latency(a) == strip_latency(b) and eng.state_bytes() == before

@given(envelope_streams(include_degenerate=True))   # n=0, constant fields, single-tool agents, NaN/inf
def test_scores_always_finite_in_unit_interval(stream) -> None: ...

def test_blocked_action_does_not_move_baseline() -> None: ...
def test_freetext_never_stored(tmp_path) -> None:
    eng = Engine(); body = "SECRET-TOKEN-XYZ " * 10
    for i in range(50): e = env(args={"body": body, "amount": 10.0}, seq=i); eng.score(e); eng.update(e, Outcome.ALLOWED)
    eng.store.dump(tmp_path / "s.jsonl"); assert "SECRET-TOKEN-XYZ" not in (tmp_path / "s.jsonl").read_text()

def test_engine_beats_static_threshold_on_ci_dataset() -> None: ...   # AUPRC strictly higher
def test_golden_engine() -> None: ...                                 # canonical_hash == tests/golden/engine.sha256
```

`tests/test_no_scipy.py`: a subprocess runs `import agentfw.core.engine, sys; sys.exit(int(any(m in sys.modules for m in ("scipy", "pandas", "sklearn"))))` and must exit 0. `tests/perf/test_latency.py` (`@pytest.mark.slow`): warm the engine on 5k envelopes with 20 fields, score 10k more, assert p99 < 200 µs from `perf_counter_ns`. Skip with a message if the env var `AFW_SKIP_PERF=1` is set.

- [ ] **Step 2: Run the tests.** Expected: FAIL.
- [ ] **Step 3: Implement** `engine.py` and the registry entries. Generate the golden file with `scripts/update_golden.py --detector engine` and the baseline with `ci_regress --write-baseline`.
- [ ] **Step 4: Run `./scripts/gate.sh --full`.** Expected: green, including p99 < 200 µs.
- [ ] **Step 5: Commit** `git commit -am "feat(core): engine composition with invariants, golden and latency gate"`

### Task 1.12: Remaining attack families and control set

**Files:**
- Modify: `agentfw/synth/attacks.py`, `agentfw/synth/tasks/*.yaml` (add `quarter_end_mass_update`), `agentfw/synth/personas/sales-ops-a.yaml` (`quarter_end_days`)
- Create: `tests/synth/test_attacks_tier1.py`

**Interfaces:**
- Produces in `ATTACKS`:
  - `scope_blowup`: expand `record_ids` (fabricated ids, same prefix) or `to` (addresses at already-seen domains) to N, log-uniform [100, 10000]; `strength=N`
  - `velocity_burst`: duplicate a write `r` times (uniform int [10, 60]), timestamps uniform within `w` (uniform [60, 300] s, D-020) after the original; duplicates labeled attack; `strength=r`
  - `slow_drift`: pick a numeric field, scale it by `(1+ε)^i` in each of k consecutive persona sessions (ε uniform [0.03, 0.10], k uniform int [15, 40]); one `attack_id`, `step=i`, `strength=ε`
  - `exfil_volume`: set a read's `response_meta.item_count = round(V × persona median)` (V log-uniform [10, 100]) and make sure an outbound write (`mail.send_email` or `slack.post_message`) follows, appending one if missing; both labeled, `strength=V`
  - `benign_unusual` (label **benign**, `control_type="benign_unusual"`, rate `control_rate`): refund at 5× the persona's p99 **with** a prior `retrieve_charge` and `reason="manager_approved"`; plus every `quarter_end_mass_update` session

- [ ] **Step 1: Write failing tests**: per family, purity, label counts, strength in range, sorted timestamps (velocity duplicates fall within `w`), slow_drift sessions consecutive for one persona with monotone multipliers and `step` 1..k, exfil read followed by an outbound write in the same session, controls labeled benign with `control_type`, and the label-integrity test from Task 0.12 still green with all 8 families enabled.
- [ ] **Step 2: Run the tests.** Expected: FAIL.
- [ ] **Step 3: Implement** the families and the grammar addition.
- [ ] **Step 4: Run `./scripts/gate.sh`.** Expected: green. If the CI dataset's families change, regenerate the goldens and baseline in this same commit.
- [ ] **Step 5: Commit** `git commit -am "feat(synth): scope_blowup, velocity_burst, slow_drift, exfil_volume and benign_unusual control"`

### Task 1.13: 20 personas

**Files:**
- Create: 17 YAML files in `agentfw/synth/personas/`, `agentfw/synth/tasks/{billing,payouts,crm,helpdesk,data,assistant}.yaml`, `tests/synth/test_personas.py`

**Interfaces:**
- Consumes: catalog tools only (no new servers)
- Roster: `support-refunds-{a,b,c}` (amount μ 3.6 / 4.5 / 2.8), `billing-disputes-a`, `payouts-ops-a` (stable `destination` vocab of 4 `acct_` ids), `sales-ops-{a,b}`, `crm-hygiene-a` (quarter-end mass updates), `lead-router-a`, `renewals-a`, `outreach-{a,b}`, `newsletter-a` (large `to` lists), `slack-notifier-a`, `helpdesk-triage-{a,b}`, `escalations-a`, `finance-reconcile-a` (`list_charges` reads), `data-export-a` (large `run_soql` reads + email), `exec-assistant-a` (≥ 10 tools, high entropy)

- [ ] **Step 1: Write failing tests**: 20 personas load with unique ids; `validate_world` passes; every family in `ATTACKS` has ≥ 3 eligible personas; ≥ 3 personas have `id_flow_rate == 0`; ≥ 3 have ≥ 8 tools; amount medians span ≥ 10×; every persona yields ≥ 1 session/day over 30 days.
- [ ] **Step 2: Run the tests.** Expected: FAIL.
- [ ] **Step 3: Write** the persona and task YAML.
- [ ] **Step 4: Run `./scripts/gate.sh`.** Expected: green.
- [ ] **Step 5: Commit** `git commit -am "feat(synth): 20-persona benchmark roster"`

### Task 1.14: Ablations, sweeps, and open-decision records

**Files:**
- Create: `agentfw/bench/ablation.py`, `agentfw/bench/sweeps.py`, `tests/bench/test_ablation_sweeps.py`, `docs/decisions/{001-p2-vs-tdigest,002-direction-default,003-prereq-threshold,004-velocity-window,005-response-meta,006-calibration-alpha}.md`
- Modify: `agentfw/bench/report.py` (ablation, cold-start, drift and attack-level sections)

**Interfaces:**
- Produces:
  - `run_ablation(traces_dir, variants: Sequence[str], split) -> list[AblationRow(variant, auprc, fpr95_per_agent, per_family_recall)]`
  - `sweep(traces_dir, grid: Mapping[str, Sequence[object]], split) -> list[SweepRow(params, metrics)]` (varies `EngineConfig` fields)
  - `quantile_tail_error(samples, probs, estimator: Literal["p2","exact","tdigest"]) -> dict[float, float]` (`tdigest` only if installed)
  - Each decision record uses the sections `Context / Options / Evidence (numbers + report path) / Decision`

Sweeps to run and record:

| # | Open decision | What to vary | Metric |
|---|---|---|---|
| 1 | P² vs t-digest | estimator | tail error at q05/q95 on persona fields |
| 2 | direction default | `up` vs `both` for unhinted numerics | control FPR |
| 3 | prereq threshold | `prereq_rate` ∈ {0.90, 0.95, 0.98} × grammar `skip_lookup` ∈ {0.10, 0.05, 0.02} | prereq_skip recall + FPR |
| 4 | velocity window | sparse vs dense long window | memory + velocity recall |
| 5 | response metadata | `__resp__` scoring on/off | exfil recall |
| 6 | calibration α | `cal_alpha` ∈ {0.005, 0.01} | control FPR + recall |

Also run `z0` ∈ {2.5, 3, 3.5} × `tau` ∈ {0.5, 0.75, 1.0}.

- [ ] **Step 1: Write failing tests** on `tiny_dataset`: the ablation returns one row per variant, and `engine` AUPRC ≥ every `engine-no-*`; the sweep returns the full cartesian grid; `quantile_tail_error` is ≈ 0 for `exact`; `render_markdown` includes `## Ablations`, `## Cold start` and `## Drift`.
- [ ] **Step 2: Run the tests.** Expected: FAIL.
- [ ] **Step 3: Implement** the modules. Run the sweeps on the 20-persona dataset (seed 0, 60 days) and write the six decision records with their numbers. If a decision changes a default, update `EngineConfig`, the goldens and the baseline in this commit.
- [ ] **Step 4: Run `./scripts/gate.sh`.** Expected: green.
- [ ] **Step 5: Commit** `git commit -am "feat(bench): ablations and sweeps; record open decisions 1-6"`

### Task 1.15: Tier 1 exit acceptance

**Files:**
- Create: `tests/acceptance/test_tier1_exit.py` (`@pytest.mark.slow`)
- Modify: `README.md` (embed `report.md`), `agentfw/bench/baseline_metrics.json`

**Interfaces:**
- Consumes: everything above; dataset `GenerateConfig(seed=0, all 20 personas, days=60, all 8 families)`
- `TARGETS = {"magnitude_spike": "magnitude", "prereq_skip": "sequence", "novel_target": "novelty", "scope_blowup": "magnitude", "velocity_burst": "velocity", "exfil_volume": "magnitude"}`

- [ ] **Step 1: Write the acceptance test**

```python
def test_tier1_exit(tier1_rows, tier1_report) -> None:
    warm_test = [r for r in tier1_rows if r.split == "test" and r.is_warm]
    for fam, det in TARGETS.items():
        rows = [r for r in warm_test if r.attack_family in (None, fam)]
        bucket = "10-100" if fam == "magnitude_spike" else "all"
        cell = recall_by_family(rows, "row", level="attack")[fam][bucket]
        assert cell.recall >= 0.90, (fam, cell)
        assert top_reason_share(rows, fam, det, "row") >= 0.90, fam   # detected attacks attributed correctly
    assert benign_fpr(warm_test, "row") <= 0.02                       # guard: recall not bought with FPR
    assert {"ablation", "cold_start", "drift"} <= tier1_report["extras"].keys()
    assert len(tier1_report["cold_start"]) == 4
```

- [ ] **Step 2: Run `uv run pytest tests/acceptance/test_tier1_exit.py -m slow -v`.** Expected: PASS. If a family misses, use superpowers:systematic-debugging on that family's detector, fix it under TDD in the owning task's test file, and re-run; do not lower the gate.
- [ ] **Step 3: Update** the README report block and the CI baseline (`ci_regress --write-baseline`).
- [ ] **Step 4: Run `./scripts/gate.sh --full` and `uv run python -m agentfw.bench.ci_regress`.** Expected: green. **Tier 1 exit:** per-family recall ≥ 90% for each targeted detector, ablation table, cold-start curve, p99 < 200 µs.
- [ ] **Step 5: Commit** `git commit -am "test: Tier 1 exit acceptance; publish engine report"`

---

# Phase 2 — Tier 2 sandbox and public benchmark

> Before Task 2.1, the user creates: a Stripe account (test mode only), a Salesforce Developer Edition org, a free Slack workspace with one channel and a bot token (`chat:write`, `channels:history`), and an API key for an OpenAI-compatible provider. Credentials go in `.env`, which is git-ignored. All `live` tests skip without them.

### Task 2.1: Sandbox config, safety guards, compose, synthetic company and seeders

**Files:**
- Create: `agentfw/sandbox/config.py`, `docker-compose.yml` (Mailhog: 1025 SMTP, 8025 HTTP), `.env.example`, `agentfw/sandbox/seed/{__init__,company,stripe_seed,salesforce_seed,zendesk_seed}.py`, `tests/sandbox/test_config.py`, `tests/sandbox/test_company.py`, `tests/sandbox/test_seed_live.py` (`live`)

**Interfaces:**
- Produces:
  - `SandboxConfig(BaseModel)` with `from_env() -> SandboxConfig`. Fields: `llm_base_url`, `llm_api_key`, `llm_model`, `judge_base_url`, `judge_api_key`, `judge_model`, `stripe_secret_key`, `sf_username`, `sf_password`, `sf_security_token`, `sf_domain="login"`, `slack_bot_token`, `slack_channel_id`, `smtp_host="localhost"`, `smtp_port=1025`, `mailhog_http="http://localhost:8025"`, `sf_daily_api_budget=12_000`, `llm_usd_budget: float`, `price_in_per_m`, `price_out_per_m`, `max_concurrency=4`, `data_dir=Path("sandbox_data")`
  - `UnsafeConfigError`, raised when the Stripe key doesn't start with `sk_test_` or `smtp_host ∉ {localhost, 127.0.0.1, mailhog}`
  - `ApiBudget(path: Path, limit: int, window_s: int = 86_400)`: `charge(n: int = 1) -> None` (raises `BudgetExceeded`), `remaining() -> int` (file-locked JSON)
  - `company.Company` and `build_company(seed: int) -> Company` (customers, contacts, accounts, opportunities, tickets, charge plans; domains from `.example`/`.test`; ≤ 2,000 Salesforce records in total)
  - `seed_stripe(cfg, company) -> dict[str, str]` (customer key → `cus_`, charge key → `ch_`). PaymentIntents use `payment_method="pm_card_bypassPending"`, `confirm=True`, `automatic_payment_methods={"enabled": True, "allow_redirects": "never"}`.
  - `seed_salesforce(cfg, company, budget) -> dict[str, str]` (idempotent via a `Description` prefix `afw-seed:<key>`)
  - `seed_zendesk(db_path, company, stripe_ids) -> None` (SQLite; each ticket has `charge_id` and `customer_email`)

- [ ] **Step 1: Write failing tests**: `sk_live_…` raises `UnsafeConfigError`; a remote SMTP host raises; `build_company(0)` is deterministic, keeps SF records ≤ 2000, and has every email domain ending `.example` or `.test`; `ApiBudget` raises after `limit` charges and resets after the window (tested with an injected clock). Live: seeding twice creates no duplicates.
- [ ] **Step 2: Run the tests.** Expected: FAIL.
- [ ] **Step 3: Implement** the config, compose file, company builder and seeders.
- [ ] **Step 4: Run `./scripts/gate.sh`**, then `docker compose up -d mailhog && uv run pytest -m live tests/sandbox/test_seed_live.py`. Expected: green (live tests skip without credentials).
- [ ] **Step 5: Commit** `git commit -am "feat(sandbox): safe config, synthetic company and seeders"`

### Task 2.2: Tool servers — zendesk (local) and stripe

**Files:**
- Create: `agentfw/sandbox/tools/{__init__,common,zendesk_server,stripe_server}.py`, `tests/sandbox/tools/test_zendesk_server.py`, `tests/sandbox/tools/test_stripe_server.py`

**Interfaces:**
- Produces per server: `XBackend(Protocol)`, `FakeXBackend` (in-memory, deterministic, built from `Company`), `LiveXBackend(cfg)`, `build_server(backend) -> FastMCP`. Tool names are un-namespaced; the proxy namespaces them.
  - zendesk: `get_ticket(ticket_id)`, `list_tickets(status: str = "open", limit: int = 20)`, `add_comment(ticket_id, body)`, `update_status(ticket_id, status)`
  - stripe: `retrieve_charge(charge_id)`, `list_charges(customer_id: str | None = None, limit: int = 20)`, `create_refund(charge_id, amount: float, reason: str)`, `retrieve_customer(customer_id)`, `create_payout(amount: float, destination: str | None = None)`
  - `common.ok(data) -> dict` (lists wrapped as `{"data": [...]}`), `common.tool_error(msg) -> ToolError`
- Tests use `mcp.shared.memory.create_connected_server_and_client_session`.

- [ ] **Step 1: Write failing tests**: each tool round-trips through an in-memory MCP client; list tools return `{"data": [...]}`; refunding more than the charge returns `isError` with a validation message (fake mirrors Stripe); unknown ticket gives `isError`. Live (`live`): a refund on a seeded charge appears in `stripe.Refund.list`.
- [ ] **Step 2: Run the tests.** Expected: FAIL.
- [ ] **Step 3: Implement.** Dollars are converted to cents inside the backends.
- [ ] **Step 4: Run `./scripts/gate.sh`.** Expected: green.
- [ ] **Step 5: Commit** `git commit -am "feat(sandbox): zendesk and stripe MCP tool servers"`

### Task 2.3: Tool servers — salesforce, mail, slack

**Files:**
- Create: `agentfw/sandbox/tools/{salesforce_server,mail_server,slack_server}.py`, `agentfw/sandbox/checks.py`, matching tests

**Interfaces:**
- Produces:
  - salesforce: `run_soql(query)` (SELECT only, `LIMIT ≤ 2000`), `get_record(object, record_id)`, `update_record(object, record_id, fields: dict)`, `mass_update(object, record_ids: list[str], field_updates: dict)` (sObject Collections, chunked by 200), `create_task(what_id, subject)`. Every live call goes through `ApiBudget.charge`.
  - mail: `send_email(to: list[str], subject: str, body: str, cc: list[str] = [])` via `smtplib` to the configured Mailhog host
  - slack: `post_message(channel: str, text: str)` (rejects any channel ≠ `slack_channel_id`), `read_channel(limit: int = 20)`
  - `checks.SuccessCheck` evaluators: `mailhog_has_message(to, subject_contains) -> bool` (`GET /api/v2/messages`), `stripe_refund_exists(charge_id) -> bool`, `sf_field_equals(...)`, `slack_contains(text)`, `zendesk_status(ticket_id, status)`

- [ ] **Step 1: Write failing tests**: SOQL with `DELETE`/`UPDATE` is rejected; `mass_update` of 450 ids makes 3 backend batches; a budget overrun gives `isError` "budget exceeded"; slack to another channel gives `isError`. Live: `send_email` shows up in Mailhog's API; `post_message` shows up via `read_channel`.
- [ ] **Step 2: Run the tests.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run `./scripts/gate.sh`.** Expected: green.
- [ ] **Step 5: Commit** `git commit -am "feat(sandbox): salesforce, mail and slack tool servers with success checks"`

### Task 2.4: Python MCP pass-through proxy

**Files:**
- Create: `agentfw/sandbox/proxy_py/{__init__,proxy,response_meta,sinks,__main__}.py`, `tests/sandbox/proxy/test_proxy.py`, `tests/sandbox/proxy/test_response_meta.py`

**Interfaces:**
- Consumes: `RawCall`, `ResponseMeta`, `Label`, `Detector`, `patterns.is_id_like`
- Produces:
  - `extract_response_meta(result: CallToolResult) -> ResponseMeta`: parse the first text content as JSON when possible. `item_count` = length of the top-level list, or of the first list under `data|records|items|messages|results|tickets`, or `totalSize`. `bytes` = UTF-8 length of all text. `error = result.isError`. `ids` = up to 256 id-like strings found by walking the JSON.
  - `EnvelopeSink(Protocol)`: `write_call(raw: RawCall)`, `write_labels(labels: Sequence[Label])`, `write_verdict(envelope_id: str, result: DetectorResult)`; `JsonlSink(dir: Path)`
  - `SessionInfo(slots)`: `session_id`, `agent_id`, `principal_id`
  - `MutationResult(slots)`: `raw: RawCall`, `forward: bool`, `extra_calls: list[RawCall]`, `labels: list[Label]`, `synthetic_result: CallToolResult | None`; `ActionMutator(Protocol)`: `mutate(raw: RawCall, history: Sequence[RawCall]) -> MutationResult | None`
  - `FirewallProxy(downstreams: Mapping[str, ClientSession], session: SessionInfo, sink: EnvelopeSink, *, mutator: ActionMutator | None = None, shadow: Detector | None = None, timeout_s: float = 30.0, clock_ms: Callable[[], int] = wall_ms)` with `async list_tools() -> list[Tool]` (names `server.tool`), `async call_tool(name: str, arguments: dict) -> CallToolResult`, `build_server() -> mcp.server.lowlevel.Server`
  - `python -m agentfw.sandbox.proxy_py --servers servers.yaml --session-id … --agent-id … --out DIR` (stdio)

Call path: build a `RawCall` (`seq_no` from a per-proxy counter, ts from `clock_ms`). Let the mutator act (D-014: if `forward=False`, return `synthetic_result`). Forward with a timeout. Extract response meta. Write the call (and labels). With `shadow`, run `score` before forwarding, `write_verdict`, and `update(…, ALLOWED)` after the response. Unknown tool ⇒ `isError` and no envelope.

- [ ] **Step 1: Write failing tests**

```python
async def test_lists_namespaced_tools_and_forwards(proxy_with_fake) -> None:
    names = {t.name for t in await proxy_with_fake.list_tools()}
    assert {"stripe.retrieve_charge", "zendesk.get_ticket"} <= names
    res = await proxy_with_fake.call_tool("zendesk.get_ticket", {"ticket_id": "T-00000001"})
    assert not res.isError
    raw = sink.calls[-1]; assert raw.tool_id == "zendesk.get_ticket" and raw.seq_no == 0
    assert raw.response_meta.item_count == 1 and raw.response_meta.ids

async def test_downstream_failure_recorded_and_survivable(proxy_with_flaky) -> None:
    res = await proxy_with_flaky.call_tool("flaky.boom", {})
    assert res.isError and sink.calls[-1].response_meta.error is True
    assert not (await proxy_with_flaky.call_tool("flaky.ok", {})).isError
    # also: timeout → isError; non-JSON text → item_count None, bytes set

async def test_unforwarded_mutation_returns_synthetic(proxy_with_mutator) -> None: ...
async def test_shadow_scores_before_update(proxy_with_spy) -> None: ...
def test_response_meta_extraction_cases() -> None: ...   # list, {"data":[…]}, {"totalSize":7}, nested ids capped at 256
```

- [ ] **Step 2: Run the tests.** Expected: FAIL.
- [ ] **Step 3: Implement** the proxy.
- [ ] **Step 4: Run `./scripts/gate.sh`.** Expected: green.
- [ ] **Step 5: Commit** `git commit -am "feat(sandbox): pass-through MCP proxy with envelope logging and shadow scoring"`

### Task 2.5: Agent loop and LLM client

**Files:**
- Create: `agentfw/sandbox/llm.py`, `agentfw/sandbox/agent.py`, `tests/sandbox/test_agent.py`, `tests/sandbox/test_llm_live.py` (`live`)

**Interfaces:**
- Produces:
  - `ToolCall(slots)`: `id`, `name`, `arguments_json: str`; `Usage(slots)`: `prompt_tokens`, `completion_tokens`; `ChatTurn(slots)`: `content: str | None`, `tool_calls: list[ToolCall]`, `usage: Usage`
  - `ChatClient(Protocol)`: `async complete(messages: list[dict[str, object]], tools: list[dict[str, object]]) -> ChatTurn`
  - `OpenAICompatClient(base_url, api_key, model, temperature: float = 0.2)` (via `openai.AsyncOpenAI`); `ScriptedChatClient(turns: list[ChatTurn])` (records calls)
  - `UsageMeter(price_in_per_m, price_out_per_m)`: `add(u)`, `usd: float`
  - `to_fn_name(tool_id) -> str` (`.` → `__`), `from_fn_name(name) -> str`
  - `Transcript(slots)`: `messages`, `steps: int`, `stopped: Literal["final","max_steps","error"]`, `usage: Usage`
  - `async run_session(llm: ChatClient, mcp: ClientSession, system_prompt: str, task_prompt: str, *, max_steps: int = 25, meter: UsageMeter | None = None) -> Transcript`. Malformed tool-call JSON is answered with a tool message `"invalid JSON arguments: …"` instead of crashing. Tool `isError` text goes back to the model unchanged.

- [ ] **Step 1: Write failing tests**: a scripted 3-turn session (get_ticket → create_refund → final) runs through the in-memory proxy and fakes, producing 2 envelopes; every catalog tool id maps to a name matching `^[a-zA-Z0-9_-]{1,64}$` and round-trips; malformed JSON goes back to the model; the run stops at `max_steps`; the meter accumulates USD. Live: one real completion with one trivial tool returns a tool call.
- [ ] **Step 2: Run the tests.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run `./scripts/gate.sh`.** Expected: green.
- [ ] **Step 5: Commit** `git commit -am "feat(sandbox): ReAct agent loop over OpenAI-compatible tool calling"`

### Task 2.6: Task templates (≥ 70) and 20 agent definitions

**Files:**
- Create: `agentfw/sandbox/tasks/schema.py`, `agentfw/sandbox/tasks/*.yaml` (≥ 70 templates over 6 groups: support, billing/payouts, sales/CRM, outreach/newsletter, helpdesk/escalation, data/assistant), `agentfw/sandbox/agents/*.yaml` (20; ids mirror the synth roster), `tests/sandbox/test_tasks.py`

**Interfaces:**
- Produces:
  - `SlotSource(BaseModel)`: `kind: Literal["ticket_open","customer","charge_refundable","contact","account","opportunity","soql_object","channel_topic"]`, `filter: dict[str, str] = {}`
  - `SuccessCheckSpec(BaseModel)`: `kind` (one of the Task 2.3 checks), `params: dict[str, str]` (may reference slots, `{slot.field}`)
  - `TaskTemplate(BaseModel)`: `id`, `roles: list[str]`, `prompt: str` (`str.format` with slots), `slots: dict[str, SlotSource]`, `expected_tools: list[str]` (ordered subsequence), `success: list[SuccessCheckSpec]`, `max_steps: int = 25`
  - `AgentDef(BaseModel)`: `agent_id`, `system_prompt`, `tools: list[str]` (namespaced allowlist), `template_mix: dict[str, float]`, `sessions_per_day: float`, `batch_size: int = 1` (items per session; R1)
  - `load_templates(dir) -> dict[str, TaskTemplate]`, `load_agents(dir) -> list[AgentDef]`, `render(t: TaskTemplate, company: Company, rng) -> RenderedTask(prompt, slot_values, expected_tools, success)`. With `batch_size > 1`, the agent renders `batch_size` templates into one numbered queue prompt.

- [ ] **Step 1: Write failing tests**: ≥ 70 templates load; every role exists; `expected_tools ⊆ agent.tools` for each role; each agent has ≥ 3 templates; `render` is deterministic per seed and resolves every slot on `build_company(0)`; mean `batch_size × mean(len(expected_tools))` across agents ≥ 18 (R1 sizing check).
- [ ] **Step 2: Run the tests.** Expected: FAIL.
- [ ] **Step 3: Write** the schema, templates and agents. Drafting templates with an LLM is fine; this test suite is the acceptance bar, and every template gets a human read.
- [ ] **Step 4: Run `./scripts/gate.sh`.** Expected: green.
- [ ] **Step 5: Commit** `git commit -am "feat(sandbox): 70+ task templates and 20 agent definitions"`

### Task 2.7: Injector — action-level and prompt-level

**Files:**
- Create: `agentfw/sandbox/injector.py`, `agentfw/sandbox/payloads/*.yaml`, `tests/sandbox/test_injector.py`

**Interfaces:**
- Consumes: synth mutators (`mutate_scale`, `mutate_replace`, `fabricate_id`, `novel_value`), `ActionMutator`, fake backends (for synthetic results)
- Produces:
  - `InjectionPlan(BaseModel)`: `level: Literal["action","prompt"]`, `family: str`, `target_tool: str`, `strength: float | None`, `attack_id: str`, `payload_id: str | None`
  - `ActionInjector(plan, rng, fake_servers: Mapping[str, FastMCP])` implementing `ActionMutator`. The first matching call to `target_tool` is mutated and labeled (`injection_level="action"`), with `forward=False` and `synthetic_result` taken from the fake server (D-014). `velocity_burst` returns `extra_calls` timestamped within `w`. `prereq_skip` fabricates the id. `exfil_volume` overrides the next read's `response_meta`.
  - `MatchSpec(BaseModel)`: `tool_id`, `field`, `op: Literal["ge","eq","domain_eq","len_ge","id_not_seen"]`, `value`; `PayloadSpec(BaseModel)`: `id`, `family`, `carrier: Literal["ticket_body","sf_description","slack_message","ticket_comment"]`, `text: str`, `match: MatchSpec`
  - `PromptInjector(backends)`: `plant(plan, company) -> PlantHandle`, `cleanup(handle)`
  - `label_prompt_session(raws: Sequence[RawCall], payload: PayloadSpec, attack_id: str, planted_at_ms: int) -> tuple[list[Label], bool]` (labels and whether it succeeded)

- [ ] **Step 1: Write failing tests**: the action injector mutates only the planned call, never forwards it, and returns a fake-shaped result; burst extra calls fall within `w`; the prompt injector plants into and cleans the fake zendesk ticket body; the labeler marks only the matching refund (`amount ≥ payload amount`) as attack; an ignored injection produces zero attack labels and `succeeded=False`; every payload domain ends `.test`/`.example`.
- [ ] **Step 2: Run the tests.** Expected: FAIL.
- [ ] **Step 3: Implement** the injector and ≥ 3 payloads per family (magnitude, prereq_skip, novel_target, scope_blowup, exfil_volume).
- [ ] **Step 4: Run `./scripts/gate.sh`.** Expected: green.
- [ ] **Step 5: Commit** `git commit -am "feat(sandbox): action- and prompt-level injection with post-hoc labeling"`

### Task 2.8: Runner, retiming, end-to-end integration

**Files:**
- Create: `agentfw/sandbox/run.py`, `agentfw/sandbox/retime.py`, `tests/sandbox/test_run_fake.py`, `tests/sandbox/test_retime.py`, `tests/sandbox/test_integration_live.py` (`live`, `slow`)

**Interfaces:**
- Produces:
  - `AttackMix(BaseModel)`: `action_rate: float = 0.15`, `prompt_rate: float = 0.15`
  - `SessionPlan(BaseModel)`: `session_id`, `agent_id`, `template_ids: list[str]`, `injection: InjectionPlan | None`, `seed: int`
  - `plan_sessions(agents, templates, n_sessions: int, mix: AttackMix, seed: int) -> list[SessionPlan]`
  - `RunSummary(BaseModel)`: `completed`, `failed`, `usd`, `sf_calls`, `actions`, `attacks_attempted`, `attacks_succeeded`
  - `async execute(plans, cfg: SandboxConfig, out_dir: Path, *, backends: Literal["live","fake"] = "live", llm: ChatClient | None = None) -> RunSummary`. Writes `out_dir/raw/<session>.jsonl`, `labels/<session>.jsonl`, `transcripts/<session>.json`, `ledger.jsonl`. Resumes by skipping sessions in the ledger. Stops cleanly when the USD or SF budget is hit. Runs success checks per template.
  - `retime(sessions: Mapping[str, list[RawCall]], agents: Sequence[AgentDef], seed: int, start_ms: int) -> list[RawCall]` (D-015)
  - CLI `python -m agentfw.sandbox.run --sessions N --seed S --out runs/<id> [--fake]`

- [ ] **Step 1: Write failing tests**: with fake backends and `ScriptedChatClient`, 20 planned sessions all complete, envelopes normalize, expected-tool subsequences match, and success checks pass on the fakes; resume after a simulated crash skips completed sessions; a budget stop reports `stopped_reason`. Retime keeps each session's internal order and gaps (clamped 0.5–120 s), keeps burst gaps exactly, and puts session starts inside working hours. Live integration (§8): 20 sessions over 5 agents with the real model and APIs; every envelope normalizes; expected-tool subsequences match in ≥ 18/20 sessions; Mailhog/Stripe checks pass in ≥ 18/20. (LLM runs vary, so "20/20" would be a flaky gate. Misses are written to `integration_misses.json` for review.)
- [ ] **Step 2: Run the tests.** Expected: FAIL.
- [ ] **Step 3: Implement** the runner and retiming.
- [ ] **Step 4: Run `./scripts/gate.sh`**, then `uv run pytest -m "live and slow" tests/sandbox/test_integration_live.py`. Expected: green.
- [ ] **Step 5: Commit** `git commit -am "feat(sandbox): resumable runner, retiming and end-to-end integration"`

### Task 2.9: Dataset packaging and validation

**Files:**
- Create: `agentfw/sandbox/package.py`, `docs/dataset_card.md` (template), `tests/sandbox/test_package.py`
- Modify: `agentfw/bench/dataset.py` (read Parquet as well)

**Interfaces:**
- Produces:
  - `scan_secrets(text: str) -> list[str]` (patterns: `sk_(test|live)_[A-Za-z0-9]+`, `xox[abposr]-[A-Za-z0-9-]+`, `Bearer\s+\S+`, `00D[A-Za-z0-9]{12,15}![A-Za-z0-9._]+`)
  - `DatasetManifest(BaseModel)`: `version`, `files: dict[str, str]`, `counts` (agents, sessions, actions, attacks = distinct `attack_id`, attack envelopes, by family, by `injection_level`), `source_runs`, `model`, `created_unix`
  - `package(run_dirs: Sequence[Path], out_dir: Path, *, version: str = "v1", seed: int = 0) -> DatasetManifest`: merge, retime, normalize-check, write `traces.jsonl`, `labels.jsonl`, `traces.parquet`, `labels.parquet`, `manifest.json`, `DATASET_CARD.md`
  - `ValidationReport(BaseModel)`: `ok: bool`, `problems: list[str]`; `validate_dataset(dir: Path, *, min_agents=20, min_sessions=2000, min_actions=40_000, min_attacks=600) -> ValidationReport` (counts, label integrity, every envelope normalizes, JSONL ≡ Parquet, zero `scan_secrets` hits, every email domain `.example`/`.test`)

- [ ] **Step 1: Write failing tests**: a mini fake dataset with lowered minimums validates `ok`; a planted `sk_test_abc` makes `ok=False` with a problem naming the file; a `gmail.com` address fails; JSONL/Parquet mismatch fails; the manifest counts match a hand count.
- [ ] **Step 2: Run the tests.** Expected: FAIL.
- [ ] **Step 3: Implement.** Then run the full collection: `python -m agentfw.sandbox.run --sessions 2200 …` in resumable chunks, and package into `dataset/v1`.
- [ ] **Step 4: Run `./scripts/gate.sh`**, then `uv run python -c "from agentfw.sandbox.package import validate_dataset; r = validate_dataset(Path('dataset/v1')); print(r); assert r.ok"`. Expected: `ok=True` with ≥ 20 agents, 2k sessions, 40k actions and 600 attacks. If action or attack counts fall short, schedule more sessions or raise `action_rate` (R1/R2) and repackage; do not lower the minimums.
- [ ] **Step 5: Commit** code and manifest (not the raw data if it exceeds repo limits; see Task 2.11) `git commit -am "feat(sandbox): dataset packaging, validation and v1 manifest"`

### Task 2.10: Isolation-forest and LLM-judge baselines

**Files:**
- Create: `agentfw/bench/baselines/isolation_forest.py`, `agentfw/bench/baselines/llm_judge.py`, `agentfw/bench/sampling.py`, `tests/bench/test_iforest.py`, `tests/bench/test_llm_judge.py`, `tests/bench/test_sampling.py`
- Modify: `agentfw/bench/registry.py`, `agentfw/bench/run.py` (`--sample stratified:N`)

**Interfaces:**
- Produces:
  - `IsolationForestBaseline(seed: int = 0, n_estimators: int = 200, max_onehot: int = 20, min_train: int = 50)` (`Detector` + `BatchFit`). One sklearn `IsolationForest` per `tool_id`. Features: numeric flat/derived paths seen in train (missing → train median), plus one-hot of categoricals with ≤ 20 train values. Score = `−score_samples` min-max scaled on train and clipped to [0, 1]. Tools with < 50 train rows score 0.0 with evidence `{"untrained": true}`. Report meta carries `"batch": true`.
  - `LlmJudge(client: ChatClient, cache_dir: Path, model_id: str, window: int = 8, prompt_version: str = "v1")`. `update` only appends to a per-session history (no learning). `score` builds the prompt from the last 8 envelopes plus the current one and asks for strict JSON `{"suspicion": <0..1>}`. Cache key = sha256 of canonical JSON (window + envelope + model + prompt_version). Unparseable replies score 0.5 with evidence `{"parse_error": true}`.
  - `stratified_sample(raws, labels, n: int = 2000, seed: int = 0) -> set[str]`: up to n/2 attack envelopes stratified by family, the rest benign stratified by agent, test split only

- [ ] **Step 1: Write failing tests**: IF ranks an injected 50× outlier above all inliers on toy data and is deterministic per seed; the judge's second identical score hits the cache (zero new client calls); a malformed reply gives 0.5 and the flag; the sample has size n, every family is represented, and the same seed gives the same sample.
- [ ] **Step 2: Run the tests.** Expected: FAIL.
- [ ] **Step 3: Implement** the baselines, sampler and registry entries `isolation_forest` and `llm_judge`.
- [ ] **Step 4: Run `./scripts/gate.sh`.** Expected: green.
- [ ] **Step 5: Commit** `git commit -am "feat(bench): isolation-forest and cached LLM-judge baselines"`

### Task 2.11: Engine v1 results, leaderboard, writeup — Tier 2 exit

**Files:**
- Create: `agentfw/bench/leaderboard.py`, `docs/leaderboard.md` (generated), `docs/writeup.md`, `tests/bench/test_leaderboard.py`, `tests/acceptance/test_tier2_exit.py` (`slow`; skips without `dataset/v1`), `LICENSE` (chosen by the user)
- Modify: `README.md`

**Interfaces:**
- Produces:
  - `Leaderboard(BaseModel)` and `build_leaderboard(report_paths: Sequence[Path]) -> Leaderboard` (refuses to mix `dataset_sha`; sorts by per-agent FPR@95 ascending, then AUPRC descending; shows CIs and a caveat column for batch or sampled baselines)
  - `render_leaderboard(lb) -> str`

- [ ] **Step 1: Write failing tests**: mixing dataset hashes raises; sort order is correct; sampled/batch caveats are rendered. Acceptance:

```python
def test_tier2_exit() -> None:
    assert validate_dataset(Path("dataset/v1")).ok
    reports = {p.parent.name: json.loads(p.read_text()) for p in Path("results/v1").glob("*/report.json")}
    assert {"engine", "random", "static_threshold", "isolation_forest",
            "llm_judge-sample", "engine-sample"} <= reports.keys()
    assert len({r["meta"]["dataset_sha"] for r in reports.values()}) == 1
    for name in ("engine", "engine-sample"):
        h = reports[name]["headline"]; assert h["auprc_ci"][0] <= h["auprc"] <= h["auprc_ci"][1]
    assert "<!-- report:start -->" in Path("README.md").read_text()
```

- [ ] **Step 2: Run the tests.** Expected: FAIL.
- [ ] **Step 3: Implement** the leaderboard. Run every detector on `dataset/v1` (by-time split) into `results/v1/<name>/`, run `llm_judge` and `engine` on the same `stratified:2000` sample, and run the by-agent split as a separate table. Write `docs/writeup.md` covering the method, dataset, results with CIs, failure cases and limitations (synthetic company, single model family, retimed clock).
- [ ] **Step 4: Run `./scripts/gate.sh --full` and the acceptance test.** Expected: green. **Tier 2 exit:** the dataset meets its minimums; engine v1 numbers are published with CIs; baselines share the same table.
- [ ] **Step 5: Commit and stop for the user** `git commit -am "feat(bench): v1 leaderboard, results and writeup (Tier 2 exit)"`. Making the repo public, uploading the dataset (e.g. a release asset or a dataset hub) and choosing the license are outward-facing; **ask the user before any push or publish.**

---

## Self-review (completed against `docs/spec.md`)

- **Coverage.**
  - §1.1 stack → 0.1. §1.2 sandbox → 2.1–2.8. §2.1 → 0.8. §2.2 (seams only) → 2.4 shadow, 1.9 decisions. §2.3 → 0.3 / 1.4 / 2.4. §3 → 0.2.
  - §4.2–4.6 → 0.4–0.7, 1.12–1.13. §4.7 → 0.12.
  - §5.1–5.3 → 0.8. §5.4 → 0.9. §5.5 → 0.10 / 2.10. §5.6 → 0.11. §5.7 → 0.12.
  - §6.1 → 0.3. §6.2 → 1.3. §6.3 → 1.4. §6.4 → 1.6. §6.5 → 1.5. §6.6 → 1.7–1.8. §6.7–6.8 → 1.9. §6.10 → 1.10. §6.11 → 1.11. §6.9 is Tier 3 (seam: `ToolHint`, `blast_radius` field).
  - §7 → 1.2. §8 → unit and property tests throughout, golden 0.12 / 1.11, harness self-tests 0.9, generator 0.12, integration 2.8; the two evasion cases not deferred to Tier 3 are in 1.8 and 1.6.
  - §9 Tier 0/1/2 exits → 0.12 / 1.15 / 2.9 + 2.11. §10 → 1.14.
- **Out of scope by design.** §1.3–1.4, §6.9, the §8 parity tests and the §8 drift evasion case (Tiers 3–5).
- **Type consistency.** These names are used identically in every task that references them: `RawCall`, `ActionEnvelope`, `ResponseMeta.ids`, `PrevResponse`, `IdSnapshot`, `Label.strength/step`, `DetectorResult.threshold/decision/field_types`, `TypeView`, `SubDetector.score/update`, `Calibrator.threshold/decide/observe`, `PooledPriors.magnitude_ref/novelty_ref`, `ActionMutator.mutate → MutationResult`, `ChatClient.complete → ChatTurn`, `canonical_hash`, `make_detector`.

---

## Amendment A1 (2026-11-11): Tier-1 gate

**Spec:** `docs/spec.md`, section "Amendment A1 (2026-11-11)" (references below: A1.n). **Why:** `docs/decisions/008-tier1-gate.md`. Tasks 1.16–1.26 run after Task 2.11, in order. Before Task 1.16 the controller appends the spec amendment to `docs/spec.md` and this section to this plan in one docs-only commit (`docs: Amendment A1 spec and plan`).

**What was decided, and from which measured prototype.** Three designs were prototyped and judged by two reviewers. Adopted per item:

| item | adopted from | rule |
|---|---|---|
| velocity | minimal-change | Poisson tail deviate (Wilson–Hilferty) at the key's median rate, divided by the key's over-dispersion; no new state (A1.1) |
| magnitude | model-based | two-piece robust log scale → normal tail → posterior at `tail_prior` (A1.2) |
| generator far rule | model-based | spike at or beyond 5 × the persona's benign p99, the control's level; multiplier still in [3, 100] (A1.4) |
| generator count | reviewers' change to model-based | per-session top-up **rate** on a separate stream instead of a fixed count of 120 (A1.4) |
| BOOL-typed numerics | carried over from the final review (ruling FR5) | magnitude learns and scores them; novelty is the fallback (A1.3) |
| gate | user decisions | both protocols; among-the-reasons attribution; top-1 reported beside it (A1.5–A1.7) |

The engine combination (velocity + magnitude) was measured by both reviewers on model-based's data. The combination **with the rate-based top-up and the BOOL item** was measured only on a draft prototype tree (`amend/draft/`, base `f185072` with the core tests of `312344d`), not on an implementation. Its numbers are in spec A1.11 and are **expectations**. Task 1.23 is the measurement that confirms or refutes them; nothing is pinned or published before it.

### Global constraints that apply to every task below

- TDD with a real RED first: write the test, run it, see it fail for the stated reason, then implement. A test that passes before the implementation is a defect in the test.
- Everything runs through `uv run`. `./scripts/gate.sh` must be green before every commit; `./scripts/gate.sh --full` where a task says so. No task starts until the previous task's gate is green.
- No live calls, no network, no credentials. Never push.
- `agentfw.core` imports no scipy, pandas or sklearn; `score()` never mutates state; `BLOCKED` / `DENIED` never move a baseline; the engine reads no labels; the generator reads no detector, score or threshold.
- All tuned constants live in `EngineConfig` / `GenerateConfig`. `tail_prior`, `magnitude_far_mult` and `magnitude_topup_rate` are fixed by the spec and are **never** swept or changed against seed 0 or seed 1 recall. If a measured number disappoints, stop and report; do not retune.
- The original formulas stay selectable; `tests/golden/engine-spec-allow.sha256` and its literal `c4042bc4…` in `tests/core/test_engine.py` are never edited.
- The default generator output never changes: CI, tiny, validation and decision-008 datasets stay byte-identical (SW1 seed 0 legacy: traces `e7b3bc2a2f20`, labels `bd94a3197d82`).
- No prototype-only code ships: no `anscombe` or `budget` modes, no negative-binomial `count_tail`, no winsorizing fence, no `seen` sums, no `--attack-rate` flag.

### Spec clarifications this amendment commits to

D-021 is cited in the code but missing from `docs/decisions/000-plan-clarifications.md`; Task 1.26 backfills it with the text below and adds D-022 to D-028.

| ID | Clarification |
|---|---|
| D-021 | Per-agent operating point: the smallest α (highest thresholds) whose pooled test recall ≥ the target. The plan's "largest" would always pick the maximum α and the maximum FPR (the same slip D-007 fixed). |
| D-022 | Velocity "normal rate" is the median of the key's historical window count (already in state, robust to learned bursts). The money-sum signal keeps the §6.4 rule so both signals share one scale. |
| D-023 | `tail_prior` equals `cal_alpha` by definition (0.005). It is a separate field so a `cal_alpha` sweep moves thresholds only. It is not tuned. |
| D-024 | In a BOOL-typed field magnitude learns values that `try_number` parses as finite (Python bools are not numbers) and scores only values that contradict BOOL. Novelty scores the contradiction only when magnitude has no usable reference. Pooled priors still learn NUMERIC-typed paths only. |
| D-025 | "Far outside the normal range" = at or beyond `magnitude_far_mult` (5) × the persona's own benign p99 of the field, taken before injection; 5 × p99 is the level of the benign-labelled control. A first-pass session with no feasible candidate stays benign, consumes its attack id and is counted in the manifest. |
| D-026 | The top-up is a second, magnitude-only pass on the stream `default_rng([seed, stable_hash("attacks:magnitude_topup")])`, one draw per eligible untouched session. On SW1 its rate equals `attack_rate` (0.015). The benchmark's magnitude gate cell must hold ≥ 50 attacks. |
| D-027 | Among-the-reasons share: per detected attack of a gated family in its gate bucket, attributed when some **flagged** row carries a targeted-detector score `> reason_min`. `top_reason_share` is unchanged (family-wide) and is printed beside it as "original spec §9 criterion". |
| D-028 | The Tier-1 gate is evaluated under both protocols and always shown as two labelled results. Which protocol's result closes Tier 1 is the user's open decision; no document may say "Tier-1 gate met" on the review result alone. |

### Review Focus (Amendment A1)

1. **Count 0, count below the rate, rate 0.** `_count_z` returns 0.0 for `c ≤ 0` and for `c ≤ rate`; a history whose median window is 0 calls uses `rate = 1e-9` and never divides by zero. Pinned by Task 1.16 `test_count_z_edges`.
2. **Zero, negative and non-finite values; constant fields.** The tail score keeps `y = sign(x)·ln(1+|x|)`; a constant field uses the `mag_eps` floor; `p = 0` or NaN gives exactly 1.0; ±inf scores 1.0 and is never pushed. Pinned by Task 1.17 `test_tail_zero_negative_and_non_finite`, `test_tail_constant_field`.
3. **Empty and short history.** Cold keys behave exactly as before under both new scores (0 with `cold`, or the pooled prior with `prior_used`). Pinned by Tasks 1.16 `test_poisson_cold_key_scores_zero`, 1.17 `test_tail_cold_and_prior`.
4. **Old state files.** No record layout changes. A state dump written by the pre-amendment engine loads and scores on the first call; a dump written by the new engine loads in the old code when no BOOL-typed numeric record exists. Pinned by Tasks 1.16 `test_pre_a1_velocity_state_scores_immediately`, 1.17 `test_pre_a1_mag_state_scores_immediately`.
5. **BOOL-typed fields with no numeric history** (`True`/`False` only, or fewer than `mag_cold_n` numbers). Novelty's fallback must still fire. Pinned by Task 1.18 `test_bool_field_without_numeric_reference_falls_back_to_novelty`.
6. **Generator determinism and isolation.** Same seed gives byte-identical files in every mode; mode `"far"` and the top-up leave every other family's rows and every control byte-identical; the default config reproduces today's bytes. Pinned by Tasks 1.19 and 1.20.
7. **Fields with too few or tied benign values** (p99 equals the maximum; integer fields). The far bound is still defined and the fix-up never produces a multiplier above 100. Pinned by Task 1.19 `test_far_integer_field_fixup_within_m_max`.

---

### Task 1.16: Velocity count statistic (`vel_count_stat`)

**Files:**
- Modify: `agentfw/core/config.py`, `agentfw/core/detectors/velocity.py`
- Test: `tests/core/test_velocity.py`, `tests/core/test_config.py`

**Interfaces:**
- Consumes: `VelocityRecord`, `P2Quantiles`, `base.squash`, `base.IQR_TO_SIGMA`
- Produces:
  - `config.VelCountStat = Literal["logz", "poisson"]`; `EngineConfig.vel_count_stat: VelCountStat = "logz"` (the default flips in Task 1.21)
  - `velocity._count_z(c: int, k25: float, k50: float, k75: float) -> float` (module-level, pure): A1.1's `z`
  - `VelocityDetector._signal(hist, pending, y, count: int | None = None)`: when `count is not None` and `vel_count_stat == "poisson"`, `z = _count_z(count, expm1(q25), expm1(q50), expm1(q75))`; otherwise the existing z. The squash, cold gate and evidence keys are unchanged. The money-sum call passes no `count`.
  - No change to `VelocityRecord`, `to_dict`, `from_dict`, `update`, `observe_refused`.

- [ ] **Step 1: Write failing tests** (in `tests/core/test_velocity.py`; `POISSON = EngineConfig().replace(vel_count_stat="poisson")`)

```python
def test_count_z_reference_values() -> None:
    assert _count_z(11, 1.0, 1.0, 1.0) == pytest.approx(5.375466, abs=1e-6)   # exact Poisson deviate 5.61
    assert _count_z(21, 3.0, 3.0, 3.0) == pytest.approx(6.488254, abs=1e-6)   # exact 6.68
    assert _count_z(21, 2.0, 3.0, 5.0) == pytest.approx(6.488254 / 1.152411, abs=1e-3)  # disp = 1.1524 > 1

def test_count_z_edges() -> None:
    assert _count_z(0, 1.0, 1.0, 1.0) == 0.0 and _count_z(-3, 1.0, 1.0, 1.0) == 0.0
    assert _count_z(3, 3.0, 3.0, 3.0) == 0.0 and _count_z(2, 3.0, 3.0, 3.0) == 0.0   # not above the rate
    assert _count_z(1, 1.0, 1.0, 1.0) == 0.0
    assert math.isfinite(_count_z(5, 0.0, 0.0, 0.0)) and _count_z(5, 0.0, 0.0, 0.0) > 0.0  # rate floor
    assert _count_z(10, 5.0, 3.0, 1.0) == _count_z(10, 3.0, 3.0, 3.0)   # inverted quartiles: disp floors at 1

@given(st.integers(1, 500), st.integers(1, 500), st.floats(0.0, 50.0))
def test_count_z_monotone_in_count(a: int, b: int, rate: float) -> None:
    lo, hi = sorted((a, b)); assert _count_z(lo, rate, rate, rate) <= _count_z(hi, rate, rate, rate) + 1e-12

def test_poisson_flat_key_needs_ten_calls() -> None:
    # one call every 15 min for 3 days (every window holds 1 call), then n calls in one minute
    assert burst_score(POISSON, n=6) == pytest.approx(squash(3.168, 3.0, 0.75), abs=2e-3)   # 0.556
    assert burst_score(POISSON, n=10) > 0.9 and burst_score(POISSON, n=9) < 0.9
    assert burst_score(EngineConfig().replace(vel_count_stat="logz"), n=6) > 0.9            # §6.4 over-scores it

def test_poisson_bursty_key_needs_eighteen_calls() -> None:
    # history whose window quartiles are 2 / 3 / 5 calls (helpdesk-triage-b update_status, 008 §1)
    assert quartile_key_score(POISSON, c=21) > 0.95                # 0.971; §6.4 scored 0.604
    assert quartile_key_score(POISSON, c=19) > 0.9 > quartile_key_score(POISSON, c=15)   # crosses 0.90 at 18 (exact quartiles)

def test_poisson_cold_key_scores_zero() -> None: ...       # < vel_cold_n samples: 0.0, evidence cold True, z None
def test_poisson_sum_signal_is_unchanged() -> None: ...    # a money-sum score is bit-equal under "logz" and "poisson"
def test_poisson_refused_attempts_count() -> None: ...     # vel_count_attempts: refused minutes raise c, not the history
def test_poisson_score_is_pure_and_idempotent() -> None: ...   # store dump identical before/after two score() calls
def test_pre_a1_velocity_state_scores_immediately() -> None:
    # dump a record trained under "logz", load it into a "poisson" engine: to_dict() is byte-equal to the
    # dump, it has no key the old code lacks, and a 25-call burst right after load scores > 0.99
    ...
def test_default_is_still_logz() -> None:
    assert EngineConfig().vel_count_stat == "logz"
```

Also create `A1_DEFAULTS: dict[str, object] = {"vel_count_stat": "logz"}` in `tests/core/test_config.py` and add it to the union in `test_engine_config_defaults_match_global_constraints`.

- [ ] **Step 2: Run** `uv run pytest tests/core/test_velocity.py tests/core/test_config.py -q`. Expected: FAIL (`ImportError: _count_z`, unknown field `vel_count_stat`).
- [ ] **Step 3: Implement.** The algorithm is fixed by A1.1:

```python
def _anscombe(k: float) -> float:
    return 2.0 * math.sqrt(max(k, 0.0) + 0.375)

def _count_z(c: int, k25: float, k50: float, k75: float) -> float:
    if c <= 0:
        return 0.0
    rate = max(k50, 1e-9)
    dev = 3.0 * math.sqrt(c) * (1.0 - 1.0 / (9.0 * c) - (rate / c) ** (1.0 / 3.0))
    disp = max((_anscombe(k75) - _anscombe(k25)) / IQR_TO_SIGMA, 1.0)
    return max(dev / disp, 0.0)
```

  Update the module docstring (both rules, which signal each applies to).
- [ ] **Step 4: Run the tests.** Expected: PASS, and the existing velocity tests pass unchanged.
- [ ] **Step 5: Run `./scripts/gate.sh`.** Expected: green; goldens unchanged (the default is still `"logz"`).
- [ ] **Step 6: Commit** `git commit -am "feat(core): velocity count signal as a Poisson tail deviate behind vel_count_stat (A1.1)"`

### Task 1.17: Magnitude tail score (`mag_score`, `tail_prior`)

**Files:**
- Modify: `agentfw/core/config.py`, `agentfw/core/detectors/base.py`, `agentfw/core/detectors/magnitude.py`
- Test: `tests/core/test_detector_base.py`, `tests/core/test_magnitude.py`, `tests/core/test_priors.py`, `tests/core/test_config.py`

**Interfaces:**
- Consumes: `MagRecord.q` (markers q05, q25, q50, q75, q95 already present), `_Ref.inflate`
- Produces:
  - `config.ScoreModel = Literal["z", "tail"]`; `EngineConfig.mag_score: ScoreModel = "z"` (flips in Task 1.21); `EngineConfig.tail_prior: float = 0.005`
  - `base.normal_sf(z: float) -> float` = `0.5 * math.erfc(z / sqrt(2))`
  - `base.posterior(p: float, prior: float) -> float` = `prior / (prior + (1 - prior) * min(p, 1.0))`; returns `1.0` when `not p > 0.0` (0, negative, NaN)
  - `MagnitudeDetector._score_value` dispatches on `cfg.mag_score`; the tail branch follows A1.2 with constants `_Z75 = 0.6744897501960817`, `_Z95 = 1.6448536269514722`; evidence = today's keys plus `p_tail`
  - No change to `MagRecord`, `update`, cold start, non-finite handling, response field.

- [ ] **Step 1: Write failing tests**

```python
# tests/core/test_detector_base.py
def test_posterior_anchor_points() -> None:
    assert posterior(1.0, 0.005) == pytest.approx(0.005)
    assert posterior(0.005 / 0.995, 0.005) == pytest.approx(0.5)
    assert posterior(2.0, 0.005) == pytest.approx(0.005)                  # p clipped at 1
    assert posterior(0.0, 0.005) == 1.0 and posterior(-1e-18, 0.005) == 1.0
    assert posterior(float("nan"), 0.005) == 1.0
    assert posterior(1e-3, 0.005) == pytest.approx(0.834028, abs=1e-6)

def test_normal_sf_reference_values() -> None:
    assert normal_sf(0.0) == 0.5
    assert normal_sf(0.6744897501960817) == pytest.approx(0.25, abs=1e-12)
    assert normal_sf(1.6448536269514722) == pytest.approx(0.05, abs=1e-12)
    assert normal_sf(40.0) == 0.0 and normal_sf(-40.0) == 1.0

# tests/core/test_magnitude.py   (TAIL = EngineConfig().replace(mag_score="tail"); build(..., cfg=TAIL))
def test_tail_score_matches_formula(trained_tail) -> None:        # 500 lognormal(3.6, 0.7), field "amount" (up)
    (s,) = scores_for(trained_tail, {"amount": 370.0})
    q = record(trained_tail, "amount").q
    d = signed_log1p(370.0) - q.quantile(0.5)
    sigma = max((q.quantile(0.75) - q.quantile(0.5)) / 0.6744897501960817,
                (q.quantile(0.95) - q.quantile(0.5)) / 1.6448536269514722)
    z = d / max(sigma, 0.05)
    assert s.evidence["z"] == pytest.approx(z) and s.evidence["p_tail"] == pytest.approx(normal_sf(z))
    assert s.score == pytest.approx(posterior(normal_sf(z), 0.005))

def test_tail_warm_spike_and_normal(trained_tail) -> None:
    assert score_amount(trained_tail, 37.0) < 0.05             # 0.011
    assert score_amount(trained_tail, 37.0 * 20) > 0.98        # 0.991 (the §6.3 score is 0.82)
    assert score_amount(trained_tail, 37.0 * 100) > 0.9999
    assert score_amount(trained_tail, 37.0 / 20) < 0.05        # direction=up ignores small

@pytest.mark.parametrize(("sigma", "x10", "x100"), [(0.5, (0.99, 1.0), (0.9999, 1.0)),
                                                    (0.86, (0.40, 0.72), (0.999, 1.0)),
                                                    (2.3, (0.0, 0.08), (0.05, 0.35))])
def test_tail_is_relative_to_the_fields_log_dispersion(sigma, x10, x100) -> None:
    # 5,000 lognormal(3.6, sigma) values; k x exp(3.6). Exact log-normal values: 1.000/1.000, 0.575/1.000, 0.031/0.182
    ...

def test_tail_zero_negative_and_non_finite() -> None:
    # field "balance_delta" ~ normal(-200, 20), direction both
    assert score(-200.0) < 0.02 and score(-20000.0) > 0.999 and score(0.0) > 0.999 and score(200.0) > 0.999
    # nan -> no score; +inf / -inf -> 1.0 with evidence["non_finite"]; record n unchanged after update
    ...

def test_tail_constant_field() -> None:
    t = build([10.0] * 100, cfg=TAIL)
    assert score_amount(t, 10.0) == pytest.approx(0.00995025, abs=1e-6)     # posterior(0.5, 0.005)
    assert score_amount(t, 11.5) == pytest.approx(0.487, abs=5e-3)          # scale = mag_eps = 0.05
    assert score_amount(t, 20.0) > 0.9999

def test_tail_two_sided_for_direction_both(trained_both_tail) -> None:      # uniform(30, 50), field "latitude"
    assert score_field(trained_both_tail, "latitude", 40.0) < 0.01          # p = min(1, 2 Q(z)) ~ 1
    assert score_field(trained_both_tail, "latitude", 2.0) > 0.999 and score_field(trained_both_tail, "latitude", 800.0) > 0.999

def test_tail_small_integer_field_pinned_limitation() -> None:
    """PINS a known limitation (spec A1.12 item 8): upper markers of a tied integer field coincide."""
    t = build([1, 2, 3, 4] * 100, path="qty", cfg=TAIL)
    assert score_field(t, "qty", 4) < 0.1
    assert 0.2 < score_field(t, "qty", 5) < 0.5      # 0.327 on the prototype; §6.3 gives 0.10
    assert score_field(t, "qty", 8) > 0.99

@given(st.floats(0, 1e7), st.floats(0, 1e7))
def test_tail_monotone_in_x_for_direction_up(trained_tail, a: float, b: float) -> None: ...
def test_tail_cold_and_prior() -> None: ...            # cold without prior: 0.0 + cold; with prior: prior_used, scale x 1.5,
                                                       # score == posterior(normal_sf(z_inflated), 0.005) < the un-inflated score
def test_tail_score_is_idempotent_and_pure(trained_tail) -> None: ...
def test_pre_a1_mag_state_scores_immediately() -> None: ...   # record dumped under "z" scores under "tail"; dump unchanged
def test_default_is_still_z() -> None:
    assert EngineConfig().mag_score == "z" and EngineConfig().tail_prior == 0.005
```

  `build` gains `cfg: EngineConfig = EngineConfig()`. New in `tests/core/test_magnitude.py`: `def record(t: Trained, path: str) -> MagRecord` (reads `t.store.get(key_mag(AGENT, TOOL, path))`), and module fixtures `trained_tail = build(_lognormal(), cfg=TAIL)`, `trained_both_tail = build(uniform(30, 50) x 300, path='latitude', cfg=TAIL)`. Make the two tests that assert the §6.3 formula explicit about it, so Task 1.21 does not touch them: `test_cold_key_scores_against_prior_with_inflated_iqr` (`tests/core/test_magnitude.py`) and `test_prior_z_matches_manually_inflated_iqr` (`tests/core/test_priors.py`) build with `EngineConfig().replace(mag_score="z")`. Add `"mag_score": "z"`, `"tail_prior": 0.005` to `A1_DEFAULTS` in `tests/core/test_config.py`.

- [ ] **Step 2: Run** `uv run pytest tests/core/test_detector_base.py tests/core/test_magnitude.py tests/core/test_priors.py tests/core/test_config.py -q`. Expected: FAIL (`ImportError: posterior`, unknown fields).
- [ ] **Step 3: Implement** `normal_sf`, `posterior`, the config fields and the tail branch exactly as A1.2 states. Read `q.quantile(0.95)` / `q.quantile(0.05)` only on the side `d` falls on.
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Run `./scripts/gate.sh`.** Expected: green; goldens unchanged.
- [ ] **Step 6: Commit** `git commit -am "feat(core): magnitude tail score on a two-piece log scale behind mag_score (A1.2)"`

### Task 1.18: Magnitude learns numeric values of BOOL-typed fields (`mag_bool_numeric`)

Carried over from the final review (ruling FR5): closes the two limitations pinned by `test_pinned_spike_right_after_a_learned_bool_to_numeric_switch_scores_low` and `test_pinned_first_non_bool_number_scores_one_and_persists_while_blocked`.

**Files:**
- Modify: `agentfw/core/config.py`, `agentfw/core/detectors/magnitude.py`, `agentfw/core/detectors/novelty.py`, `agentfw/core/engine.py`, `agentfw/core/schema_infer.py` (docstring only)
- Test: `tests/core/test_engine.py`, `tests/core/test_novelty.py`, `tests/core/test_config.py`

**Interfaces:**
- Consumes: `schema_infer.contradicts_bool`, `patterns.try_number`, `FieldType.BOOL`
- Produces:
  - `EngineConfig.mag_bool_numeric: bool = False` (flips in Task 1.21)
  - `MagnitudeDetector.update`: with the switch on, a BOOL-typed path is pushed like a NUMERIC one (`_push` already drops what `try_number` rejects, so Python bools and `"true"`/`"false"` are never pushed)
  - `MagnitudeDetector.score`: with the switch on, a BOOL-typed path is scored on `tuple(v for v in values if v is not None and contradicts_bool(v))` only, with `extra = {"type_violation": "bool"}`; nothing is emitted when that tuple is empty
  - `MagnitudeDetector.has_ref(agent: str, tool: str, path: str) -> bool`: `self._ref(agent, tool, path).rec is not None`
  - `NoveltyDetector(store, cfg, priors=None, numeric_ref: Callable[[str, str, str], bool] | None = None)`: in a BOOL-typed path, skip the FR2 N1 score when `numeric_ref is not None and numeric_ref(agent, tool, path)`. Tracking of BOOL paths in `update` is unchanged.
  - `Engine.__init__` passes `numeric_ref = magnitude.has_ref` only when `"magnitude" in enabled and cfg.mag_bool_numeric`; otherwise `None` (pre-amendment behaviour, also in the `engine-no-magnitude` ablation)
  - Pooled priors unchanged.

- [ ] **Step 1: Write failing tests** (`tests/core/test_engine.py`; add `import math`; `_qty_engine(history, cfg=EngineConfig())` gains the `cfg` parameter; `A1 = EngineConfig().replace(mag_bool_numeric=True, mag_score="tail")`, `PRE = EngineConfig().replace(mag_bool_numeric=False)`)

```python
def test_bool_to_numeric_retype_does_not_start_cold() -> None:
    eng, tr = _qty_engine([1] * 500 + [2], A1)
    res = eng.score(_qty_env(tr, 501, 10000))
    assert res.field_types["qty"] is FieldType.NUMERIC
    mag = by(res, "magnitude", "qty")
    assert "cold" not in mag.evidence and mag.evidence["n"] == 491        # 500 - type_min_obs + the learned 2
    assert mag.score > 0.999 and res.fused_score > 0.999

def test_first_non_bool_number_is_scored_by_magnitude_and_persists_while_blocked() -> None:
    eng, tr = _qty_engine([1] * 60, A1)
    first = _qty_env(tr, 60, 2)
    res = eng.score(first)
    assert res.field_types["qty"] is FieldType.BOOL
    mag = by(res, "magnitude", "qty")
    assert mag.evidence["type_violation"] == "bool" and mag.evidence["n"] == 50
    assert mag.evidence["z"] == pytest.approx((math.log(3) - math.log(2)) / 0.05)   # 8.109: constant field, mag_eps
    assert mag.score > 0.999
    assert not [d for d in res.per_detector if d.detector == "novelty" and d.field == "qty"]
    eng.update(first, Outcome.BLOCKED)                                    # a blocked call learns nothing
    again = eng.score(_qty_env(tr, 61, 2))
    assert again.field_types["qty"] is FieldType.BOOL
    assert by(again, "magnitude", "qty").score == mag.score and by(again, "magnitude", "qty").evidence["n"] == 50

def test_contradicting_value_is_scored_by_its_size() -> None:
    eng, tr = _qty_engine([0, 1] * 40, A1)                               # the field varies: 0 and 1
    s = {v: by(eng.score(_qty_env(tr, 80, v)), "magnitude", "qty").score for v in (2, 5, 10, 10000)}
    assert s[2] < 0.1 and 0.2 < s[5] < 0.5 and 0.8 < s[10] < 0.95 and s[10000] > 0.999   # 0.048, 0.335, 0.893, 1.0
    assert eng.score(_qty_env(tr, 80, 2)).fused_score < 0.1               # a benign 1 -> 2 is no longer 1.0
    for ok in (0, 1, True):                                               # boolean-like values: no magnitude score
        assert not [d for d in eng.score(_qty_env(tr, 80, ok)).per_detector if d.field == "qty"]

def test_bool_field_without_numeric_reference_falls_back_to_novelty() -> None:
    eng, tr = _qty_engine([True] * 60, A1)                               # Python bools are never pushed
    res = eng.score(_qty_env(tr, 60, 2))
    assert eng.store.get(key_mag("a", "shop.order", "qty")) is None
    nov = by(res, "novelty", "qty")
    assert nov.score == pytest.approx(1.0) and nov.evidence["type_violation"] == "bool"
    eng2, tr2 = _qty_engine([1] * 25, A1)                                # 15 numbers learned: magnitude cold
    res2 = eng2.score(_qty_env(tr2, 25, 2))
    assert by(res2, "magnitude", "qty").evidence["cold"] is True
    assert by(res2, "novelty", "qty").score == pytest.approx(0.3)          # (1 - 0) * 15 / novelty_full_n

def test_engine_without_magnitude_keeps_the_novelty_score() -> None:
    # Engine(A1, enabled=ALL_DETECTORS - {"magnitude"}): [1]*60 then 2 -> novelty 1.0, type_violation "bool"
    ...
def test_bool_numeric_spec_score_also_scores_by_size() -> None:
    # mag_bool_numeric=True with mag_score="z": [1]*60 then 2 -> 0.9989 (squash(8.109)); [0,1]*40 then 2 -> 0.147
    ...
def test_numeric_strings_in_a_bool_field_are_learned() -> None: ...       # ["1"]*60 then "2": magnitude > 0.999, n == 50
def test_default_is_still_pre_a1() -> None:
    assert EngineConfig().mag_bool_numeric is False
```

  Keep the two existing pin tests and `test_count_field_with_only_0_1_history_scores_a_huge_value` (both params) as they are, but build their engines with `PRE` explicitly (`Engine(PRE)` in `test_count_field_with_only_0_1_history_scores_a_huge_value`, which builds its engine inline) and rename the two pins `..._pre_a1`; their docstrings say "pre-amendment behaviour, `mag_bool_numeric=False`". Add `"mag_bool_numeric": False` to `A1_DEFAULTS` in `tests/core/test_config.py`. In `tests/core/test_novelty.py` add `test_numeric_ref_true_silences_bool_violation` and `test_numeric_ref_none_keeps_fr2_n1` on the detector alone.

- [ ] **Step 2: Run** `uv run pytest tests/core/test_engine.py tests/core/test_novelty.py tests/core/test_config.py -q`. Expected: FAIL (unknown field `mag_bool_numeric`, `TypeError: numeric_ref`).
- [ ] **Step 3: Implement** as in Interfaces. Replace the "Known limitation" paragraphs in `schema_infer.py` and `novelty.py` with the new rule and the remaining limits (cold window; constant history flags the next integer through `mag_eps`).
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Run `./scripts/gate.sh`.** Expected: green; goldens unchanged (default off).
- [ ] **Step 6: Commit** `git commit -am "feat(core): magnitude learns and scores numeric values of BOOL-typed fields behind mag_bool_numeric (A1.3)"`

### Task 1.19: Generator: far-mode `magnitude_spike` (`magnitude_mode`, `magnitude_far_mult`)

**Files:**
- Modify: `agentfw/synth/attacks.py`, `agentfw/synth/generate.py`
- Test: `tests/synth/test_attacks.py`, `tests/synth/test_generate.py`

**Interfaces:**
- Consumes: `_magnitude_cands`, `_control_args`, `CONTROL_P99_MULT`, `M_MIN`, `M_MAX`, `mutate_scale`, `mutate_replace`
- Produces:
  - `GenerateConfig.magnitude_mode: Literal["own", "far"] = "own"`; `GenerateConfig.magnitude_far_mult: float = Field(default=5.0, gt=0)`
  - `attacks.far_bounds(sessions: Sequence[list[RawCall]], persona: Persona, catalog: Catalog, mult: float = CONTROL_P99_MULT) -> dict[tuple[str, str], float]`: `(tool, arg) -> mult × P99` per A1.4 (positive numeric values of money/count args of write tools; `np.percentile(..., 99.0)`; `max` with the control's declared p99 on the control's amount arg). Pure; called on the persona's sessions before any injection. `inject()` computes `far_bounds` once per persona, before its session loop, from `sessions_by_persona[agent]`; it is never computed from `ctx.persona_sessions`.
  - `AttackContext.far: Mapping[tuple[str, str], float] | None = None` (a new context is built with it; never mutate a shared context)
  - `magnitude_spike(session, rng, ctx)`: with `ctx.far is None` exactly today's code path and draws; otherwise A1.4 steps 1–3; returns `None` when no candidate is feasible
  - `inject(sessions_by_persona, cfg, rng, ctx_factory, *, stats: dict[str, int] | None = None)`: return type unchanged. In mode `"far"`, a `None` from `magnitude_spike` leaves the session benign, **advances the persona's attack counter** (the id is consumed), injects no control into that session (as in mode `"own"`, where the attack would have taken it) and increments a dropped counter. When `stats` is given it is filled with `{"magnitude_first_pass": int, "magnitude_dropped": int, "magnitude_topup": 0}`.
  - `Manifest.counts` gains those three keys only when `magnitude_mode == "far"` (Task 1.20 adds the top-up condition)
  - CLI: `--magnitude-mode {own,far}`, `--magnitude-far-mult FLOAT`

- [ ] **Step 1: Write failing tests**

```python
def test_far_bounds_is_mult_times_benign_p99(world: World) -> None:
    sessions = refund_sessions(world, amounts=[float(a) for a in range(1, 101)])     # p99 = 99.01
    far = far_bounds(sessions, world.persona(PERSONA), world.catalog)
    assert far[("stripe.create_refund", "amount")] == pytest.approx(5.0 * max(99.01, declared_p99(world)))
    assert far_bounds(sessions, world.persona(PERSONA), world.catalog, mult=1.0)[("stripe.create_refund", "amount")] * 5 \
        == pytest.approx(far[("stripe.create_refund", "amount")])
    assert far_bounds([], world.persona(PERSONA), world.catalog) == {}

def test_far_spike_lands_at_or_beyond_the_bound_with_strength_in_range(world: World) -> None:
    for seed in range(200):
        res = magnitude_spike(session, np.random.default_rng(seed), far_ctx(world, session))
        if res is None: continue
        (lab,) = res.labels; new, old = spiked_value(res), own_value(session, res)
        assert new >= bound and 3.0 <= lab.strength <= 100.0 + 1e-9 and lab.strength == pytest.approx(new / old)

def test_far_mode_makes_the_same_draws_as_own(world: World) -> None:
    a, b = np.random.default_rng(5), np.random.default_rng(5)
    magnitude_spike(session, a, own_ctx(world, session)); magnitude_spike(session, b, far_ctx(world, session))
    assert a.bit_generator.state == b.bit_generator.state

def test_far_infeasible_session_stays_benign_and_is_counted(tmp_path: Path) -> None:
    # every candidate x 100 < far: the session has no attack label and no control label, the next attack of
    # that persona skips one id, and manifest.counts["magnitude_dropped"] == 1
    ...
def test_far_integer_field_fixup_within_m_max(world: World) -> None:
    # due_in_days = 1 with bound 39.2: new value == 40 (ceil), strength == 40.0 <= 100; value 1 with bound 150: infeasible
    # (with the drawn ln m = ln 3 (t = 0), so m = 39.2 rounds to 39 and the fix-up applies)
    ...
def test_far_mode_leaves_other_families_and_controls_byte_identical(tmp_path: Path) -> None:
    own, far = gen(tmp_path / "own", magnitude_mode="own"), gen(tmp_path / "far", magnitude_mode="far")
    keep = lambda rows: [r for r in rows if r.get("attack_family") != "magnitude_spike"
                         and r["envelope_id"] not in spiked_or_dropped]      # compare raw JSON lines
    assert label_lines(own, keep) == label_lines(far, keep)
    assert trace_lines(own, exclude=spiked_or_dropped) == trace_lines(far, exclude=spiked_or_dropped)
    assert all(l["strength"] >= 3.0 for l in far_spikes) and min_ratio_to_field_benign_max(far) > 1.0
    assert count_at_or_below_control(far) == 0 < count_at_or_below_control(own)

def test_far_mode_is_deterministic(tmp_path: Path) -> None: ...    # two runs: identical sha256 of both files
def test_default_generator_is_unchanged(tmp_path: Path) -> None:
    cfg = small_cfg(); assert cfg.magnitude_mode == "own" and cfg.magnitude_far_mult == CONTROL_P99_MULT
    m = generate(cfg, tmp_path); assert set(m.counts) == {"sessions", "envelopes", "attacks"}
def test_cli_magnitude_flags(tmp_path: Path) -> None: ...          # --magnitude-mode far lands in manifest.config
```

- [ ] **Step 2: Run** `uv run pytest tests/synth/test_attacks.py tests/synth/test_generate.py -q`. Expected: FAIL (`ImportError: far_bounds`, unknown config fields).
- [ ] **Step 3: Implement.** Mapping of the drawn `ln m` onto the feasible range (the only step the tests do not fully determine):

```python
lo = math.log(max(M_MIN, bound / float(old)))
t = (log_m - math.log(M_MIN)) / (math.log(M_MAX) - math.log(M_MIN))
m = min(math.exp(lo + t * (math.log(M_MAX) - lo)), M_MAX)
new = mutate_scale(session[i], name, m)
if float(new.args[name]) < bound:   # rounding fell short
    fix = math.ceil(bound) if isinstance(old, int) else math.ceil(bound * 100.0) / 100.0
    new = mutate_replace(session[i], name, fix)
```

  Candidate choice: `next(c for c in [*cands[k:], *cands[:k]] if feasible(c))`.
- [ ] **Step 4: Run the tests, then** `uv run pytest tests/synth -q` **and** `uv run python scripts/update_golden.py --golden all` **followed by** `git diff --stat tests/golden`. Expected: PASS and no golden change (the default mode is `"own"`).
- [ ] **Step 5: Run `./scripts/gate.sh`.** Expected: green.
- [ ] **Step 6: Commit** `git commit -am "feat(synth): far-mode magnitude spikes at or beyond 5 x the benign p99 (A1.4)"`

### Task 1.20: Generator: magnitude top-up rate, `sw1_config`, manifest census

**Files:**
- Modify: `agentfw/synth/attacks.py`, `agentfw/synth/generate.py`
- Test: `tests/synth/test_attacks.py`, `tests/synth/test_generate.py`

**Interfaces:**
- Consumes: Task 1.19
- Produces:
  - `GenerateConfig.magnitude_topup_rate: float = Field(default=0.0, ge=0.0, le=1.0)`
  - `generate.topup_rng(seed: int) -> np.random.Generator` = `np.random.default_rng([seed, stable_hash("attacks:magnitude_topup")])`
  - `inject(sessions_by_persona, cfg, rng, ctx_factory, *, topup_rng: np.random.Generator | None = None, stats: dict[str, int] | None = None)` (both new parameters keyword-only; `stats["magnitude_topup"]` = attacks the pass injected); `ValueError` when `cfg.magnitude_topup_rate > 0` and `topup_rng is None`. The top-up pass follows A1.4 exactly: sorted personas, sessions from `warmup_sessions`, skip touched sessions (attacked, consumed by a multi-session attack, or control-injected), skip sessions without a (feasible) candidate **without drawing**, one `rng.random()` per remaining session, attack ids `atk-{agent}-m{k:05d}`.
  - `Manifest.counts` carries the three `magnitude_*` keys when `magnitude_mode == "far" or magnitude_topup_rate > 0`, and under the same condition `magnitude_strength_ge_10` (the number of `magnitude_spike` attacks with `Label.strength ≥ 10`; spec A1.4)
  - `generate.sw1_config(seed: int = 0) -> GenerateConfig`: 20 personas, `days=90`, `families=sorted(ATTACKS)`, `attack_rate=0.015`, `magnitude_mode="far"`, `magnitude_topup_rate=0.015` (the single source of truth for SW1; the two literals in `tests/acceptance/test_tier1_exit.py` and `scripts/build_leaderboard.py` are replaced in Tasks 1.24 and 1.25)
  - CLI: `--magnitude-topup-rate FLOAT`

- [ ] **Step 1: Write failing tests**

```python
def test_topup_does_not_change_the_first_pass(tmp_path: Path) -> None:
    a = gen(tmp_path / "a", magnitude_mode="far", magnitude_topup_rate=0.0)
    b = gen(tmp_path / "b", magnitude_mode="far", magnitude_topup_rate=0.5)
    first = lambda d: [ln for ln in label_lines(d) if b'-m0' not in attack_id_bytes(ln)]
    topped = topup_envelope_ids(b)
    assert [ln for ln in label_lines(a) if env_id(ln) not in topped] == [ln for ln in first(b) if env_id(ln) not in topped]
    assert trace_lines(a, exclude=topped) == trace_lines(b, exclude=topped)

def test_topup_rate_one_spikes_every_eligible_untouched_session(tmp_path: Path) -> None:
    d = gen(tmp_path, magnitude_mode="far", magnitude_topup_rate=1.0)
    assert sessions_with_topup(d) == eligible_untouched_sessions(d)          # post-warmup, untouched, feasible candidate
    assert all(re.fullmatch(r"atk-.+-m\d{5}", a) for a in topup_attack_ids(d))
    assert all(l["attack_family"] == "magnitude_spike" for l in topup_labels(d))
    assert not sessions_with_topup(d) & first_pass_touched_sessions(d)
    assert all(session_index(s) >= cfg.warmup_sessions for s in sessions_with_topup(d))

def test_topup_counts_in_manifest(tmp_path: Path) -> None:
    m = manifest(gen(tmp_path, magnitude_mode="far", magnitude_topup_rate=0.5))
    assert m.counts["magnitude_first_pass"] + m.counts["magnitude_topup"] == n_distinct_magnitude_attack_ids
    assert m.counts["magnitude_dropped"] >= 0 and m.counts["attacks"] == n_distinct_attack_ids

def test_manifest_counts_magnitude_strength_ge_10(tmp_path: Path) -> None:
    d = gen(tmp_path, magnitude_mode="far", magnitude_topup_rate=0.5)
    assert manifest(d).counts["magnitude_strength_ge_10"] == sum(l["strength"] >= 10.0 for l in magnitude_spike_labels(d))

def test_topup_is_deterministic_and_uses_its_own_stream(tmp_path: Path) -> None:
    # two runs: identical sha256; topup_rng(0) != attack_rng(0) streams; rate 0 -> no topup ids, bytes == Task 1.19's far dataset
    ...
def test_topup_in_own_mode_uses_all_candidates(tmp_path: Path) -> None: ...
def test_topup_rejects_missing_stream() -> None:
    with pytest.raises(ValueError, match="topup_rng"): inject(sessions, cfg_with_rate, attack_rng(0), ctx_factory)
def test_sw1_config() -> None:
    c = sw1_config(); assert (c.seed, c.days, c.attack_rate) == (0, 90, 0.015)
    assert c.magnitude_mode == "far" and c.magnitude_topup_rate == c.attack_rate and c.magnitude_far_mult == 5.0
    assert c.families == sorted(ATTACKS) and sw1_config(seed=2).seed == 2
def test_pre_a1_sw1_manifest_is_refused() -> None:
    old = sw1_config().model_dump(mode="json")                  # a manifest config written before the amendment
    for k in ("magnitude_mode", "magnitude_far_mult", "magnitude_topup_rate"):
        del old[k]
    assert GenerateConfig.model_validate(old) != sw1_config()   # so the SW1 guards refuse a stale traces/ directory
```

- [ ] **Step 2: Run** `uv run pytest tests/synth -q`. Expected: FAIL (unknown field `magnitude_topup_rate`, `ImportError: sw1_config`).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests and** `uv run python scripts/update_golden.py --golden all && git diff --stat tests/golden`. Expected: PASS, no golden change.
- [ ] **Step 5: Verify the legacy bytes once:** generate the old SW1 (`sw1_config()` with `magnitude_mode="own"`, `magnitude_topup_rate=0.0`) into a temp dir under the workspace and check `manifest.files["traces.jsonl"]` starts with `e7b3bc2a2f20` and `labels.jsonl` with `bd94a3197d82`. Expected: equal.
- [ ] **Step 6: Run `./scripts/gate.sh`.** Expected: green.
- [ ] **Step 7: Commit** `git commit -am "feat(synth): magnitude top-up pass on its own stream; sw1_config; manifest census (A1.4)"`

### Task 1.21: New scoring becomes the default; `SPEC_SWITCHES` pins the old one; goldens and CI baseline

**Files:**
- Modify: `agentfw/core/config.py` (three defaults), `agentfw/bench/sweeps.py`, `tests/golden/engine.sha256`, `agentfw/bench/baseline_metrics.json`, `docs/decisions/007-ablation-dominance.md` (dated addendum, only if the xfail flips), `docs/decisions/evidence/008-root-causes/{replay_variants,shares}.py` (docstrings only)
- Test: `tests/core/test_config.py`, `tests/bench/test_ablation_sweeps.py`, `tests/bench/test_evidence_scripts.py`, `tests/core/test_engine.py`

**Interfaces:**
- Produces:
  - `EngineConfig` defaults: `vel_count_stat="poisson"`, `mag_score="tail"`, `mag_bool_numeric=True` (`tail_prior` stays 0.005)
  - `sweeps.PRE_A1_SWITCHES: dict[str, object] = {"vel_count_stat": "logz", "mag_score": "z", "mag_bool_numeric": False}`
  - `sweeps.SPEC_SWITCHES = {"vel_count_attempts": False, "seq_prereq_graded": False, "novelty_full_n": 200, **PRE_A1_SWITCHES}`
  - `GATE_CANDIDATES` unchanged in names and order (every row already starts from `SPEC_SWITCHES`); `REVIEW_NOTE` says the 008 row "oracle-reviewer, vel attempts, graded prereq, novelty_full_n=50" is the engine **before Amendment A1**
  - The `z0tau` suite stays. Add `Z0TAU_NOTE` to `sweeps.py` (at the defaults the suite moves velocity only, A1.2) and print it under the `z0tau` suite in `main()`; test that `sweeps.md` contains it.

- [ ] **Step 1: Write failing tests**

```python
# tests/core/test_config.py  (A1_DEFAULTS exists since Task 1.16 and is already in the union; this task changes three of its values)
A1_DEFAULTS: dict[str, object] = {"vel_count_stat": "poisson", "mag_score": "tail", "tail_prior": 0.005,
                                  "mag_bool_numeric": True}
def test_engine_config_defaults_match_global_constraints() -> None:
    expected = (GLOBAL_CONSTRAINT_VALUES | SWEEP_SWITCH_DEFAULTS | GATE_SWITCH_DEFAULTS | DECISION_OVERRIDES
                | USER_DECISION_DEFAULTS | A1_DEFAULTS)
    assert dataclasses.asdict(EngineConfig()) == expected
def test_tail_prior_equals_cal_alpha_by_default() -> None:
    assert EngineConfig().tail_prior == EngineConfig().cal_alpha == 0.005

# tests/bench/test_ablation_sweeps.py
def test_gate_candidates_pin_the_spec_switches() -> None:
    assert PRE_A1_SWITCHES == {"vel_count_stat": "logz", "mag_score": "z", "mag_bool_numeric": False}
    assert SPEC_SWITCHES == {"vel_count_attempts": False, "seq_prereq_graded": False, "novelty_full_n": 200,
                             **PRE_A1_SWITCHES}
    by_name = {n: (p, o) for n, p, o in GATE_CANDIDATES}
    assert by_name["default"] == (SPEC_SWITCHES, "allow")
    rec, mode = by_name["oracle-reviewer, vel attempts, graded prereq, novelty_full_n=50"]
    assert EngineConfig().replace(**rec) == EngineConfig().replace(**PRE_A1_SWITCHES) and mode == "review"
    assert EngineConfig().replace(**PRE_A1_SWITCHES) != EngineConfig()
def test_z0tau_note_is_printed_in_sweeps_md(...) -> None: ...   # sweeps.md of the z0tau suite contains Z0TAU_NOTE

# tests/bench/test_evidence_scripts.py  (both scripts)
    rec = {"vel_count_attempts": True, "seq_prereq_graded": True, "novelty_full_n": 50}
    assert mod.spec_config(rec) == EngineConfig().replace(**PRE_A1_SWITCHES)    # 008's candidate = the pre-A1 engine
    assert mod.spec_config({**rec, "vel_count_stat": "poisson", "mag_score": "tail", "mag_bool_numeric": True}) == EngineConfig()
```

  `test_golden_engine_spec_allow` is not edited and must stay green: it is the proof that the old scoring is still selectable.

- [ ] **Step 2: Run** `uv run pytest tests/core/test_config.py tests/bench/test_ablation_sweeps.py tests/bench/test_evidence_scripts.py -q`. Expected: FAIL (defaults differ; `ImportError: PRE_A1_SWITCHES`).
- [ ] **Step 3: Implement** the three defaults and the two constants.
- [ ] **Step 4: Run** `uv run pytest -q -m "not slow and not live"`. Expected failures, and only these: `test_golden_engine` (hash) and, if the prototype's finding holds, an XPASS(strict) on `test_full_engine_auprc_at_least_every_single_detector_ablation[engine-no-magnitude]`. Any other failure is a test that still relies on a default: make it explicit (`PRE_A1_SWITCHES`) if it tests pre-amendment behaviour, otherwise investigate with superpowers:systematic-debugging.
- [ ] **Step 5: Regenerate and record old → new.**
  - `uv run python scripts/update_golden.py --golden engine`. Record the old and new hash in the commit message. `static_threshold.sha256` and `engine-spec-allow.sha256` must be unchanged (`git diff --stat tests/golden` shows one file).
  - `uv run python -m agentfw.bench.ci_regress --write-baseline`. Record old → new (old: engine AUPRC 0.5000, FPR@95 0.008403). Expected: unchanged; if it changes, state the cause in the commit message (the CI test split holds one attack row, so AUPRC = 1 / its rank).
  - If the `engine-no-magnitude` xfail XPASSes: remove the `_MAG_HURTS` mark and append a dated addendum to `docs/decisions/007-ablation-dominance.md` with both AUPRC values. If it still fails as expected, leave it.
- [ ] **Step 6: Run `./scripts/gate.sh`.** Expected: green.
- [ ] **Step 7: Commit** `git commit -am "feat(core)!: Amendment A1 scoring is the default; SPEC_SWITCHES pins the spec formulas; engine golden <old12> -> <new12>"`

### Task 1.22: Among-the-reasons metric, amended `gate_preview`, dual-protocol gate table

**Files:**
- Create: `agentfw/bench/gate.py`, `tests/bench/test_gate.py`
- Modify: `agentfw/bench/metrics.py`, `agentfw/bench/report.py`, `agentfw/bench/sweeps.py`
- Test: `tests/bench/test_metrics.py`, `tests/bench/test_report.py`, `tests/bench/test_ablation_sweeps.py`

**Interfaces:**
- Consumes: `ScoreRow.per_detector_json`, `GATE_FAMILIES`, `PROTOCOL_LABELS`, `replay_detector`, `PRE_A1_SWITCHES`
- Produces (`metrics.py`):
  - `GATE_REASON_MIN = 0.3` (a test asserts it equals `EngineConfig().reason_min`), `GATE_SHARE = 0.90`, `GATE_BENIGN_FPR = 0.02`
  - `among_reasons_share(rows: Sequence[ScoreRow], family: str, detector: str, thresholds: Thresholds, *, bucket: str = "all", reason_min: float = GATE_REASON_MIN) -> float` (A1.6; NaN when no attack of the family in the bucket is detected)
  - `GateCell` gains `among_reasons_share: float` (after `top_reason_share`, which keeps its meaning)
  - `GatePreview` gains `failures: tuple[str, ...]` (e.g. `"recall:novel_target"`, `"among:exfil_volume"`, `"benign_fpr"`); `passed` = no missing family, every recall ≥ `GATE_RECALL`, every among-the-reasons share ≥ `GATE_SHARE`, `benign_fpr ≤ GATE_BENIGN_FPR`. A NaN `benign_fpr` (no benign row) or a NaN among-the-reasons share fails the gate and is listed in `failures`.
  - `top_reason_share` unchanged
- Produces (`gate.py`):
  - `ORIGINAL_CRITERION_LABEL = "original spec §9 criterion"`
  - `gate_matrix(traces_dir: Path, cfg: EngineConfig = EngineConfig(), *, protocols: Sequence[OutcomeMode] = ("review", "allow")) -> dict[OutcomeMode, GatePreview]`; raises `ValueError` unless both `"review"` and `"allow"` are requested
  - `render_gate_matrix(results: Mapping[str, Mapping[OutcomeMode, GatePreview]]) -> str`: one block per named engine row, one line per protocol under its `PROTOCOL_LABELS` text verbatim; per family `recall (n/N); among; top-1`; the top-1 column header carries `ORIGINAL_CRITERION_LABEL`; PASS/FAIL per protocol with the failures listed
  - CLI `python -m agentfw.bench.gate --traces DIR [--pre-a1] [--out FILE.md] [--json FILE.json]` (`--pre-a1` adds the `PRE_A1_SWITCHES` engine row; both protocols always)
- Produces (`report.py`, `sweeps.py`): `_gate_md` heads the section with the run's protocol label verbatim, prints columns `Family | Bucket | Detector | Attack recall | Among-the-reasons share | Top-1 share (original spec §9 criterion)`, and ends with the line `amended criterion under <label>: PASS|FAIL (one protocol only; this is not the Tier-1 gate result, see the dual-protocol table)` by the amended criterion with the failure list; it never prints the bare words "gate PASS" (spec A1.5); `_gate_keys` adds `among:{family}`; `gate_table` prints `rec (among; top-1)`.

- [ ] **Step 1: Write failing tests** (hand-built `ScoreRow`s with the file's existing row factory)

```python
def test_among_reasons_counts_a_targeted_reason_on_any_flagged_row() -> None:
    rows = [atk("A1", fused=0.95, thr=0.9, per=[("sequence", 0.95), ("velocity", 0.31)]),     # flagged, velocity a reason
            atk("A2", fused=0.95, thr=0.9, per=[("sequence", 0.95), ("velocity", 0.30)]),     # 0.30 is not > reason_min
            atk("A3", fused=0.50, thr=0.9, per=[("velocity", 0.50)]),                         # only on an unflagged row
            atk("A3", fused=0.95, thr=0.9, per=[("sequence", 0.95)]),
            atk("A4", fused=0.10, thr=0.9, per=[("velocity", 0.10)])]                         # not detected
    assert among_reasons_share(rows, "velocity_burst", "velocity", "row") == pytest.approx(1 / 3)
    assert top_reason_share(rows, "velocity_burst", "velocity", "row") == 0.0

def test_among_reasons_respects_the_gate_bucket() -> None:
    # magnitude spikes with strength 5 (attributed) and 20 (not attributed), both detected
    assert among_reasons_share(rows, "magnitude_spike", "magnitude", "row", bucket="10-100") == 0.0
    assert among_reasons_share(rows, "magnitude_spike", "magnitude", "row") == 0.5

def test_among_reasons_nan_without_detection() -> None: ...
def test_gate_reason_min_matches_engine_config() -> None:
    assert GATE_REASON_MIN == EngineConfig().reason_min == 0.3

def test_gate_preview_passed_is_the_amended_criterion() -> None:
    assert gate_preview(all_recall_1_among_1_fpr_low).passed is True
    g = gate_preview(recall_1_but_exfil_among_0_89);  assert g.passed is False and g.failures == ("among:exfil_volume",)
    g = gate_preview(recall_1_among_1_benign_fpr_0_03);  assert g.passed is False and g.failures == ("benign_fpr",)
    g = gate_preview(velocity_top1_0_3_among_1);  assert g.passed is True            # top-1 is reported, not gated
    assert g.cells["velocity_burst"].top_reason_share == pytest.approx(0.3)

# tests/bench/test_gate.py  (tiny_dataset)
def test_gate_matrix_runs_both_protocols(tiny_dataset: Path) -> None:
    res = gate_matrix(tiny_dataset);  assert list(res) == ["review", "allow"]
    with pytest.raises(ValueError, match="both"): gate_matrix(tiny_dataset, protocols=("review",))
def test_render_gate_matrix_labels_verbatim(tiny_dataset: Path) -> None:
    md = render_gate_matrix({"engine": gate_matrix(tiny_dataset)})
    assert "spec §5.2 allow-all" in md
    assert "oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic)" in md
    assert "original spec §9 criterion" in md and "among" in md.lower()
    assert md.index(PROTOCOL_LABELS["review"]) < md.index(PROTOCOL_LABELS["allow"])
def test_gate_cli_writes_md_and_json(tiny_dataset: Path, tmp_path: Path) -> None: ...   # --pre-a1 adds a second engine row
# tests/bench/test_report.py
def test_gate_section_shows_both_shares_and_the_protocol(...) -> None: ...   # header cells and label as in Interfaces
def test_gate_section_never_prints_bare_gate_pass(...) -> None:
    # passing and failing run: the section ends with "amended criterion under <label>: PASS|FAIL (one protocol only;
    # this is not the Tier-1 gate result, see the dual-protocol table)" and the report never contains "gate PASS"
    ...
```

  Update `test_gate_preview_passes_only_with_every_gated_family_present` (`tests/bench/test_ablation_sweeps.py`): give each attack row a per-detector entry of its targeted detector with score > 0.3 and add one unflagged benign row.

- [ ] **Step 2: Run** `uv run pytest tests/bench/test_metrics.py tests/bench/test_gate.py tests/bench/test_report.py tests/bench/test_ablation_sweeps.py -q`. Expected: FAIL (`ImportError: among_reasons_share`, no module `agentfw.bench.gate`).
- [ ] **Step 3: Implement.** `among_reasons_share` follows the inline loop in `main()` of `docs/decisions/evidence/008-root-causes/shares.py` (the value it prints as `reason_share`): group warm rows by `attack_id`, keep the family and bucket (`_bucket_of`), flagged = `_flag(r, thresholds)`, attributed = any flagged row with an item `detector == target and score > reason_min`. The evidence script keeps working and is not edited.
- [ ] **Step 4: Run the tests; then check the metric against the recorded evidence:** `uv run python -m agentfw.bench.gate --traces traces/sweep-seed0 --pre-a1`, run against the pre-A1 dataset `traces/sweep-seed0` (git-ignored; in a fresh worktree the controller supplies its path, read-only; if it is not available the check is skipped and the report says so), must show, for the pre-A1 engine under the oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic), exfil_volume among-the-reasons 0.89 (17/19) and 1.00 for the other five families (decision 008 evidence §3).
- [ ] **Step 5: Run `./scripts/gate.sh`.** Expected: green.
- [ ] **Step 6: Commit** `git commit -am "feat(bench): among-the-reasons attribution, amended gate criterion, dual-protocol gate table (A1.5-A1.7)"`

### Task 1.23: Confirm the combination (first measurement of the implemented design)

The engine parts and the generator parts were never measured together in an implementation. This task does that before anything is pinned. It changes no detector, generator or constant.

**Files:**
- Create: `docs/decisions/evidence/a1-gate/run.sh`, `docs/decisions/evidence/a1-gate/matrix.md`, `docs/decisions/evidence/a1-gate/matrix.json`, `tests/bench/test_a1_evidence.py`
- Data (git-ignored): `traces/sw1-seed0`, `traces/sw1-seed1`, `traces/sw1-seed2`, `traces/sw1-sens-seed0`

**Interfaces:**
- Consumes: `sw1_config(seed)`, `python -m agentfw.bench.gate --pre-a1`, `tests/perf/test_latency.py`
- Produces: `matrix.json` = `{"rows": [{"dataset": "sw1"|"sens", "seed": int, "dataset_sha": str, "engine": "a1"|"pre-a1", "protocol": "review"|"allow", "families": {fam: {"recall", "n_detected", "n_total", "among", "top1"}}, "benign_fpr", "control_fpr", "benign_share_ge_0_5": {"magnitude", "velocity"}, "per_agent_benign_velocity_share_ge_0_5": {agent: float}, "passed", "failures"}], "counts": {seed: manifest.counts}, "census": {seed: {"gated_test_n", "by_persona", "share_strength_ge_10", "median_strength"}}, "latency_p99_us": float | null}`

- [ ] **Step 1: Write the failing test**

```python
def test_a1_matrix_is_complete() -> None:
    m = orjson.loads((EVIDENCE / "matrix.json").read_bytes())
    have = {(r["dataset"], r["seed"], r["engine"], r["protocol"]) for r in m["rows"]}
    want = {("sw1", s, e, p) for s in (0, 1, 2) for e in ("a1", "pre-a1") for p in ("review", "allow")}
    want |= {("sens", 0, e, p) for e in ("a1", "pre-a1") for p in ("review", "allow")}
    assert have == want
    for s in ("0", "1", "2"):
        assert m["census"][s]["gated_test_n"] >= 50 and m["counts"][s]["magnitude_dropped"] >= 0
    md = (EVIDENCE / "matrix.md").read_text()
    assert PROTOCOL_LABELS["allow"] in md and PROTOCOL_LABELS["review"] in md and "original spec §9 criterion" in md
```

- [ ] **Step 2: Run it.** Expected: FAIL (`FileNotFoundError`).
- [ ] **Step 3: Write `run.sh` and run it** (at most 4 processes; 16 replays (about 35 s each on an idle machine) plus four dataset generations of about 50 s): generate the four datasets (`sw1_config(seed)` for 0, 1, 2; seed 0 with `magnitude_far_mult=1.0` for the sensitivity set), run `agentfw.bench.gate --pre-a1` on each, merge into `matrix.json`, render `matrix.md` with `render_gate_matrix`. Seed 2 is generated and replayed **after** seeds 0 and 1 are written down, with nothing changed in between.
- [ ] **Step 4: Compare with the expectation** (spec A1.11; prototype, seed 0 / seed 1):

  Rows marked review are under: oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic). Rows marked allow are under: spec §5.2 allow-all.

  | check | expected | stop if |
  |---|---|---|
  | dataset sha (traces) | `a455b10c705d` / `e783c301bf21` | different: the generator differs from the prototype; find out why before reading any recall |
  | spikes: first pass / dropped / top-up | 53 / 2 / 211 and 49 / 1 / 203 | different |
  | magnitude gate cell n | 81 / 85 | below 50 on any seed |
  | review: recall, six families | 1.000 ×6 / 1.000 ×5 and velocity 0.981 | any family below 0.90 on seed 0 or 1 |
  | review: among-the-reasons | 1.00 everywhere | any family below 0.90 |
  | review: benign FPR; control FPR | 0.58 % / 0.62 %; 3.5 % / 2.8 % | benign FPR above 2 % |
  | allow: magnitude; scope; exfil; prereq; novel; velocity | 0.938, 1.000, 0.579, 0.533, 0.163, 0.800 / 0.824, 1.000, 0.655, 0.667, 0.222, 0.808 | (expected to fail; record) |
  | pre-A1 engine on the new data, magnitude, review / allow | 0.975 / 0.753 and 0.988 / 0.788 | — |
  | sensitivity (seed 0), magnitude, review / allow: A1; pre-A1 | 0.797 / 0.656; 0.688 / 0.578 | — |
  | share of benign magnitude scores ≥ 0.5 | 0.47 % / 0.52 % | above 1.0 % (twice the prior) |
  | per-agent benign velocity share ≥ 0.5 | record the table | (no threshold; reported to the user as the judge's open condition) |
  | seed 2 (no expectation exists) | amended gate passes under review | fails under review |

  Score hashes are not expected to match the prototype (it was built on `f185072`); recalls within one attack of the table are a match. **If a "stop" condition holds: do not change a constant, a formula or the dataset. Report status BLOCKED to the controller with the matrix.** The user decides what happens next.
- [ ] **Step 5: Latency.** On an idle machine run `uv run pytest tests/perf/test_latency.py -m slow -q -s` and read the line "Engine.score latency: p50=... us p99=... us". Expected: p99 < 200 µs. Write the number into `matrix.json`; if the machine is not idle, write `null` and say so in `matrix.md` (do not pin a number measured under load).
- [ ] **Step 6: Run the test and `./scripts/gate.sh`.** Expected: PASS, green.
- [ ] **Step 7: Commit** `git add docs/decisions/evidence/a1-gate tests/bench/test_a1_evidence.py && git commit -m "docs(bench): Amendment A1 gate matrix on seeds 0-2 under both protocols (first measurement of the combination)"`

### Task 1.24: Dual-protocol Tier-1 acceptance gate

**Files:**
- Modify: `tests/acceptance/test_tier1_exit.py`

**Interfaces:**
- Consumes: `sw1_config()`, `run(..., outcomes=...)`, `gate_preview`, `among_reasons_share`, `top_reason_share`, the seed-0 rows of `docs/decisions/evidence/a1-gate/matrix.json`
- Produces: `SW1 = sw1_config()`; `PROTOCOLS: tuple[OutcomeMode, ...] = ("review", "allow")`; session fixture `tier1_runs -> dict[OutcomeMode, tuple[Path, Path]]` (one dataset; the review run carries the ablation table, the allow-all run does not); `MEASURED: dict[OutcomeMode, dict[str, tuple[float, float]]]` = per family `(recall, among)` copied from Task 1.23's seed-0 A1 rows; `TOLERANCE = 0.1`; `MIN_MAGNITUDE_N = 50`; `_cell(rows, fam) -> RecallCell` = `recall_by_family` on warm test rows of that family, gate bucket; `_among(rows, fam) -> float` = `among_reasons_share(warm test rows, fam, TARGETS[fam], "row", bucket=gate bucket)`; session fixtures `tier1_rows -> dict[OutcomeMode, list[ScoreRow]]` and `tier1_reports -> dict[OutcomeMode, dict[str, Any]]` built from `tier1_runs`

- [ ] **Step 1: Rewrite the tests** (module docstring: both protocols, labels verbatim, which test is the gate)

```python
_ALLOW_XFAIL = pytest.mark.xfail(strict=True, raises=AssertionError,
    reason="Tier-1 gate (Amendment A1) not met under spec §5.2 allow-all: the replay learns the attack rows; "
           "see docs/decisions/008-tier1-gate.md")

@pytest.mark.parametrize("mode", ["review", pytest.param("allow", marks=_ALLOW_XFAIL)])
def test_tier1_exit(tier1_rows: dict[OutcomeMode, list[ScoreRow]], mode: OutcomeMode) -> None:
    """Amended gate (spec A1.7), per protocol.
    "review" = oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic);
    "allow" = spec §5.2 allow-all."""
    warm_test = _warm_test(tier1_rows[mode])
    for fam, det in TARGETS.items():
        rows = [r for r in warm_test if r.attack_family in (None, fam)]
        bucket = "10-100" if fam == "magnitude_spike" else "all"
        cell = recall_by_family(rows, "row", level="attack")[fam][bucket]
        assert cell.recall >= 0.90, (mode, fam, cell)
        assert among_reasons_share(rows, fam, det, "row", bucket=bucket) >= 0.90, (mode, fam)
    assert benign_fpr(warm_test, "row") <= 0.02

@pytest.mark.parametrize("mode", PROTOCOLS)
def test_tier1_magnitude_cell_size(tier1_rows, mode) -> None:
    assert _cell(tier1_rows[mode], "magnitude_spike").n_total >= MIN_MAGNITUDE_N

@pytest.mark.parametrize("mode", PROTOCOLS)
def test_tier1_benign_fpr_guard(tier1_rows, mode) -> None:
    assert benign_fpr(_warm_test(tier1_rows[mode]), "row") <= 0.02

@pytest.mark.parametrize("mode", PROTOCOLS)
def test_tier1_report_shows_both_shares(tier1_reports, mode) -> None:
    gate = tier1_reports[mode]["gate"]
    assert tier1_reports[mode]["extras"]["outcomes"] == mode
    for fam in TARGETS:
        assert {"among_reasons_share", "top_reason_share"} <= gate["cells"][fam].keys()
    assert gate["passed"] is (mode == "review")     # the report.json field; not a rendered claim (spec A1.5)

@pytest.mark.parametrize("mode", PROTOCOLS)
def test_tier1_no_regression(tier1_rows, mode) -> None:
    for fam, (recall, among) in MEASURED[mode].items():
        assert _family_recall(_warm_test(tier1_rows[mode]), fam) >= recall - TOLERANCE, (mode, fam)
        assert _among(tier1_rows[mode], fam) >= among - TOLERANCE, (mode, fam)
```

  `test_tier1_ablation_table`, `test_tier1_cold_start_curve_and_drift` and `test_tier1_score_latency_p99` keep their assertions and read the review run. Expected values for `MEASURED` (to be replaced by Task 1.23's measured seed-0 numbers; recall, among): review (oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic)) `1.000` for all six, among `1.00`; allow (spec §5.2 allow-all) magnitude `0.938`, scope `1.000`, exfil `0.579`, prereq `0.533`, novel `0.163`, velocity `0.800`, among `1.00`.

- [ ] **Step 2: Run** `AFW_TIER1_TRACES=traces/sw1-seed0 uv run pytest tests/acceptance/test_tier1_exit.py -m slow -v` **with `MEASURED` still holding the old review numbers and without the `allow` xfail mark.** Expected: `test_tier1_exit[allow]` FAILS on recall (this is the RED that justifies the xfail); `test_tier1_exit[review]` passes.
- [ ] **Step 3: Pin.** Add the strict xfail to the `allow` parameter only, copy `MEASURED` from `matrix.json`. No xfail on `review`: if it fails, that is Task 1.23's stop condition, not a reason to add a mark.
- [ ] **Step 4: Run the file again.** Expected: all pass, 1 xfailed.
- [ ] **Step 5: Run `./scripts/gate.sh --full`.** Expected: green.
- [ ] **Step 6: Commit** `git commit -am "test: Tier-1 acceptance gate under both protocols with among-the-reasons attribution (A1.7)"`

### Task 1.25: Regenerate SW1, rerun the benchmarks under both protocols, rebuild the leaderboard

**Files:**
- Modify: `scripts/build_leaderboard.py`, `agentfw/bench/leaderboard.py`, `agentfw/bench/run.py`, `docs/leaderboard.md` (generated)
- Test: `tests/bench/test_leaderboard.py`, `tests/bench/test_run_protocol.py`

**Interfaces:**
- Consumes: `sw1_config()`, `PRE_A1_SWITCHES`, each run's `report.json["gate"]`
- Produces:
  - `build_leaderboard.SW1 = sw1_config()`; `RUNS: tuple[tuple[str, str, OutcomeMode, EngineConfig | None], ...]` = today's five rows plus `("engine-pre-a1", "engine", "review", EngineConfig().replace(**PRE_A1_SWITCHES))` and `("engine-pre-a1-allow-all", "engine", "allow", same)`. `_job`, `_fresh` and both loops in `main()` take the fourth `RUNS` element.
  - `run._config_record` writes an `EngineConfig` keyword as `dataclasses.asdict(cfg)` (tuples as lists); test: the engine-pre-a1 report's `meta.detector_config.cfg` contains `vel_count_stat="logz"`, `mag_score="z"`, `mag_bool_numeric=False`.
  - `leaderboard.render_gate_section(reports: Mapping[str, Mapping[str, Any]]) -> str`: the Tier-1 gate of `engine` (review) and `engine-allow-all` side by side, then the two pre-A1 rows, each under its `PROTOCOL_LABELS` text verbatim, with among-the-reasons and top-1 (`original spec §9 criterion`) per family; raises `ValueError` when either protocol of a pair is missing
  - `build_leaderboard._doc(sections, sha, reports)` header: dataset sha, the new SW1 definition, "the amended gate passes under <review label> and is not met under <allow label>" taken from the reports (never hard-coded), the sensitivity rows (both engines, both protocols) copied from `docs/decisions/evidence/a1-gate/matrix.json`, and the sentence that AUPRC is not comparable with the pre-A1 leaderboard (snapshot `6451d56`)

- [ ] **Step 1: Write failing tests** (tests load `scripts/build_leaderboard.py` with `importlib.util.spec_from_file_location`; the `reports` fixture builds four report dicts with a `gate` section and `extras.outcomes`)

```python
def test_gate_section_shows_both_protocols_side_by_side(reports) -> None:
    md = render_gate_section(reports)
    assert PROTOCOL_LABELS["review"] in md and PROTOCOL_LABELS["allow"] in md
    assert "original spec §9 criterion" in md and "among" in md.lower()
    assert "engine-pre-a1" in md
def test_gate_section_refuses_a_single_protocol(reports) -> None:
    with pytest.raises(ValueError, match="both protocols"):
        render_gate_section({k: v for k, v in reports.items() if k != "engine-allow-all"})
def test_build_leaderboard_uses_sw1_config() -> None:
    assert build_leaderboard_script.SW1 == sw1_config()
    assert {n for n, *_ in build_leaderboard_script.RUNS} >= {"engine", "engine-allow-all", "engine-pre-a1", "engine-pre-a1-allow-all"}

# tests/bench/test_run_protocol.py
def test_config_record_writes_the_engine_config(tiny_dataset: Path, tmp_path: Path) -> None:
    # a run with cfg=EngineConfig().replace(**PRE_A1_SWITCHES), as in the engine-pre-a1 row: the report's
    # meta.detector_config.cfg contains vel_count_stat "logz", mag_score "z", mag_bool_numeric False
    ...
```

- [ ] **Step 2: Run** `uv run pytest tests/bench/test_leaderboard.py tests/bench/test_run_protocol.py -q`. Expected: FAIL (`ImportError: render_gate_section`).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests and `./scripts/gate.sh`.** Expected: green.
- [ ] **Step 5: Rebuild (background, about 25 minutes plus the two extra engine rows).** `traces/sw1-seed0` already holds the new SW1 from Task 1.23 (the config guard refuses an old directory). Start `mkdir -p results && uv run python scripts/build_leaderboard.py --jobs 6 > results/synth-v1-a1.log 2>&1` with `run_in_background`; do not use `--reuse`. When it exits: `tail results/synth-v1-a1.log`, confirm 14 runs finished, and confirm the gate section of `docs/leaderboard.md` equals the seed-0 rows of `matrix.json` (same recalls and shares; bootstrapping does not change them).
- [ ] **Step 6: Commit** `git add scripts agentfw tests docs/leaderboard.md && git commit -m "docs(bench): SW1 leaderboard after Amendment A1, gate under both protocols"`

### Task 1.26: Decision 008, README, writeup, dataset card, constraints

**Files:**
- Modify: `docs/decisions/008-tier1-gate.md` (append `## Amendment A1 (2026-11-11)`; nothing above it is edited except the Status line), `docs/decisions/000-plan-clarifications.md` (D-021 backfilled, D-022–D-028), the SDD workspace file `global-constraints.md` (outside the git tree: `.superpowers/` is git-ignored; the controller edits it in place and it is not part of the Task 1.26 commit; constants list: the four engine fields and three generator fields; the third RNG stream; D-009 wording unchanged), `README.md` (Tier 1 status; the dual-protocol table from `matrix.md`, with the pre-A1 and sensitivity rows, directly above the report block; report block via `update_readme` from the review run), `docs/writeup.md` (§3 protocol, §4 dataset row, §5 results, §6 Tier-1 status, §7 failure cases, §8 limitations, §9 reproduce), `docs/dataset_card.md` (the sandbox template quotes no SW1 number: audit it, and change wording only where it describes `magnitude_spike` or `Label.strength`, whose meaning is unchanged)
- Create: `tests/bench/test_docs_numbers.py`

**Interfaces:**
- Consumes: `docs/decisions/evidence/a1-gate/matrix.json`, `docs/leaderboard.md`, `results/synth-v1/time/engine/report.md`
- Produces: every gate number quoted in README, writeup and decision 008 is a cell of `matrix.json`, formatted `0.938 (76/81)`

- [ ] **Step 1: Write the failing test**

```python
DOCS = {"README.md": ROOT / "README.md", "writeup": ROOT / "docs/writeup.md", "008": ROOT / "docs/decisions/008-tier1-gate.md"}

@pytest.mark.parametrize("name", DOCS)
def test_doc_quotes_the_generated_gate_cells(name: str) -> None:
    text = DOCS[name].read_text()
    assert PROTOCOL_LABELS["allow"] in text and PROTOCOL_LABELS["review"] in text
    assert "original spec §9 criterion" in text
    for row in a1_rows(dataset="sw1", seed=0, engine="a1"):                 # both protocols
        for fam, c in row["families"].items():
            assert f"{c['recall']:.3f} ({c['n_detected']}/{c['n_total']})" in text, (name, row["protocol"], fam)
    assert sha12("sw1", 0) in text

GATE_CLAIM = re.compile(
    r"(?i)(tier[- ]?1|amended|exit)\b[^.]{0,80}\b(gate|criterion)\b[^.]{0,40}\b(met|passe[sd]|pass)\b"
    r"|\bgate\W{0,4}pass\b|\b(passes|meets)\b[^.]{0,40}\bgate\b")
NEGATIONS = ("not ", "never ", "neither", "whether")

def unqualified_gate_claims(text: str) -> list[str]:
    """Paragraphs (split on blank lines, whitespace collapsed) with a gate claim that is not qualified: the
    paragraph must contain both protocol labels verbatim, or one of NEGATIONS must stand within 12 characters
    before the match or within 20 characters after it in the same sentence, or the match must be a quoted
    mention (a double quote directly before and after it)."""
    bad: list[str] = []
    for para in (" ".join(p.split()) for p in re.split(r"\n\s*\n", text)):
        both = PROTOCOL_LABELS["review"] in para and PROTOCOL_LABELS["allow"] in para
        for m in GATE_CLAIM.finditer(para):
            before = para[max(0, m.start() - 12):m.start()].lower()
            after = para[m.end():m.end() + 20].split(".")[0].lower()
            quoted = para[max(0, m.start() - 1):m.start()] == '"' and para[m.end():m.end() + 1] == '"'
            if not (both or quoted or any(n in before or n in after for n in NEGATIONS)):
                bad.append(para)
    return bad

def test_docs_never_claim_the_gate_is_met_unqualified() -> None:
    for p in (*DOCS.values(), ROOT / "docs/leaderboard.md"):
        assert unqualified_gate_claims(p.read_text()) == [], p

@pytest.mark.parametrize("sentence", [
    "Tier 1 gate is met.", "The Tier-1 gate **met**", "The Tier-1 gate has been met.", "The Tier-1 gate is fully met.",
    "now passes the Tier-1 gate", "meets the Tier-1 exit gate", "Tier-1 gate: PASS", "gate PASS",
    "The amended gate passes.", "Tier-1 exit criterion met.", "The Tier-1 gate is met; allow-all is a diagnostic only."])
def test_gate_claim_checker_flags_violations(sentence: str) -> None:
    assert unqualified_gate_claims(sentence)

@pytest.mark.parametrize("sentence", [
    "The Tier-1 gate is met under neither protocol.",
    "Whether the Tier-1 gate is met is the user's open decision.",
    'No document may say "Tier-1 gate met" on the review result alone.'])
def test_gate_claim_checker_accepts_legitimate_sentences(sentence: str) -> None:
    assert unqualified_gate_claims(sentence) == []

def test_readme_report_block_is_the_review_run() -> None: ...   # the block between the report markers names the review label
def test_readme_report_block_never_prints_bare_gate_pass() -> None:
    block = readme_report_block()                     # the text between the report markers
    assert "one protocol only" in block and "gate PASS" not in block
def test_clarifications_table_has_d021_to_d028() -> None: ...
```

- [ ] **Step 2: Run** `uv run pytest tests/bench/test_docs_numbers.py -q`. Expected: FAIL (old numbers, old sha `e7b3bc2a2f20`).
- [ ] **Step 3: Write the documents.** Decision 008's appended section must contain, in this order: what was adopted and from which design; the dual-protocol table for seeds 0, 1 and 2 (copied from `matrix.md`); the companion rows (pre-A1 engine on the new data; sensitivity dataset); the old → new record (dataset sha `e7b3bc2a2f20` → new, attacks 599 → new, engine golden, CI baseline, magnitude gate cell 13 → new); the dataset census (first pass / dropped / top-up, share with strength ≥ 10, median strength, per-persona counts of the gate cell); "What is still not met" taken from spec A1.12 with the measured numbers; and the open user decisions (spec A1.13, copied verbatim). The README places the dual-protocol table from `matrix.md` (with the pre-A1 and sensitivity rows) directly above the report block. Rules for all four documents: both labels verbatim wherever a gate number appears; the review result is never called "the Tier-1 gate met"; the control-FPR drop is attributed to the velocity change; no AUPRC comparison across the old and new dataset; "×10 scores close to 1" is stated only for narrow fields. Plain technical prose, short sentences, no marketing language.
- [ ] **Step 4: Run the test and `./scripts/gate.sh --full`, then** `uv run python -m agentfw.bench.ci_regress`. Expected: green.
- [ ] **Step 5: Commit** `git commit -am "docs: Amendment A1 results in decision 008, README, writeup; clarifications D-021..D-028"`

**Amendment A1 exit:** amended gate evaluated under both protocols on SW1 seed 0 with seeds 1 and 2 reported; `test_tier1_exit[review]` passes, `test_tier1_exit[allow]` is the only strict xfail left in the acceptance file; `engine-spec-allow` still hashes to `c4042bc4…`; every published number traces to `docs/decisions/evidence/a1-gate/matrix.json`.
