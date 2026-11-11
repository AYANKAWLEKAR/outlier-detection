# Trajectory-Aware Monitoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add session-level trajectory monitoring to agentfw — an artifact ledger, reference-free vital signs (loop / spin / stall), MCP tool annotations and blast radius, a pre-execution gate for irreversible calls, escalation, provisional flags, session verdicts, verdict authority by evidence maturity — plus the evaluation upgrades needed to measure them honestly.

**Architecture:** Session context stays where it already lives: the normalizer's `SessionTracker` grows an artifact ledger and a ring of recent-call fingerprints and hands them to the engine on the envelope (causal, read-only). The engine gains a `vital` sub-detector (stateless, reads the envelope) and a decision layer (`agentfw/core/trajectory/`) that runs after fusion: blast radius → threshold modulation → gate (irreversible only) → escalation → provisional lifecycle (reversible only) → authority ceiling → per-session flag window and verdict. Session records live in the engine's `StateStore` under `sess:`. **Every behaviour change sits behind an `EngineConfig` switch whose default is the current (spec v3 + decision 008) behaviour**, so goldens do not move; the last task measures every switch and writes a decision record for the user to set defaults.

**Tech Stack:** Python 3.12, uv, ruff, mypy --strict, pydantic v2 (boundary types only), slots dataclasses, orjson, numpy, xxhash; pytest + hypothesis; scipy only in tests. Sandbox side: `mcp` (`ToolAnnotations`, `tools/list`).

**Spec:** `docs/feature-spec-trajectory-monitoring.md` (feature spec, "F1"–"F10"), layered on `docs/spec.md` (v3, "§N.N"). Paper sections ("Paper §4.5") are cited, never quoted.

---

## Scope and interpretation

Branch `feat/trajectory-monitoring`, cut from `feat/tiers-0-2` at `2aafb11`. Tiers 0–2 exist; Tier 3 (blast radius, §6.9) and Tier 4 (Go proxy, Redis, enforcement) do not. This plan builds everything the feature spec places in Tiers 0–3 in Python, plus the Python-proxy parts of F3 and F2. Task 0.1 copies the table below into `docs/decisions/009-trajectory-clarifications.md`.

| ID | Clarification |
|---|---|
| T-01 | **In scope:** F1–F8 and F10 in `agentfw/core`, `agentfw/bench`, `agentfw/synth` and the Python sandbox proxy. **Out of scope:** F9 (the spec says not before F1–F8 are measured); everything Tier 4 (Go proxy, Redis atomicity, Cedar, live graduated responses). The Python proxy is shadow-only, so `WARN_SESSION` / `HALT_SESSION` are recorded, never acted on, there. |
| T-02 | F3/F4/F8 need blast radius and `θ_eff` (§6.9), which do not exist yet. Task 1.7 builds §6.9's **lexical + structural** sources plus the F3 annotation terms. The learned LightGBM source stays Tier 3+ work. |
| T-03 | **Switches default to current behaviour.** Each feature adds an `EngineConfig` field (or makes `vital` opt-in via `enabled`) whose default reproduces today's scores *and* decisions. `tests/golden/*.sha256` must not change in any task. Task 4.1 measures each switch and writes `docs/decisions/010-trajectory-defaults.md`; defaults flip only on a user decision (same pattern as decision 008). |
| T-04 | `ActionEnvelope.session_ids_seen` stays (membership, including the Bloom overflow) and `ledger_refs` is **added** next to it. The v3 id-flow path keeps reading `session_ids_seen` unless `flow_ledger` is on. |
| T-05 | Task-input ids travel as `RawCall.task_ids` on a session's **first** call. An MCP proxy never sees the prompt; the sandbox runner passes them through `SessionInfo`. Synth sets them to the session's externally sourced (`source: external`) id values. |
| T-06 | `regime` has four values, not three: `history+schemas`, `history_only`, `schemas_only`, `bare`. With three, every decision on today's (schema-less) benchmark would be `bare` and F8 would make the whole engine inform-only. "History" = the (agent, tool) key has ≥ `mag_cold_n` observations; "schemas" = the tool has a captured schema. |
| T-07 | `vital` scores are reported in `per_detector` and reasons but **excluded from the fused max**. That is how "never block a single call on their own" (F2) is enforced, and it keeps the fused score — and so the goldens — unchanged when `vital` is enabled. |
| T-08 | Loop score: the first exact duplicate in the window scores 0.7, the second and later score 1.0; a non-exact near-duplicate (arg-token Jaccard ≥ 0.9) scores 0.7. The `LOOP` flag needs `s_loop = 1.0`. This reconciles F2's formula with its own acceptance test ("fires at the 2nd duplicate") and with the synth grammar's benign single retries. |
| T-09 | `SPIN` is a severe flag (F2 says loop/spin/stall are severe-flag eligible; F7's list omits it). `density` = Σ over the window's calls of that call's **highest** flag weight, divided by `W`, so it stays in [0, 1]. |
| T-10 | `loop` / `spin` / `stall` are attack families labelled `attack` but listed in `RUNAWAY_FAMILIES`: they are opt-in in `GenerateConfig.families`, excluded from every per-call metric (neither positive nor negative), and scored only by session metrics. `splice` is opt-in too and reported separately. |
| T-11 | The gate's prerequisite block (F4 step 2) fires only when the (agent, tool, path) is warm (`FlowRecord.total ≥ flow_min_obs`) and its historical presence rate `r_flow ≥ gate_flow_rate` (0.90). A cold key escalates instead. An id known only through the Bloom overflow counts as present. Otherwise, personas with benign fabricated ids (`id_flow_rate < 1`) and any un-seeded sandbox session would be false-blocked. |
| T-12 | Python escalation (F5) = a fresh, cache-free re-score plus an optional injected judge. The Python fast path skips nothing today, so "full re-score with pooled priors and the long window" is already the fast path; fast-path shortcuts belong to the Go port. |
| T-13 | Attribution (F10) is leave-one-group-out: an attack is "carried" by group G when it is detected with all groups and undetected without G; detected attacks that no single removal un-detects are `redundant`. That makes "removing a group changes exactly the rows it carried" true by construction instead of approximately. |
| T-14 | Maturity cut points: `cold` n < 30 (`mat_warm_n`), `warm` 30 ≤ n < 300, `mature` n ≥ 300 (`mat_mature_n`). A cold-maturity decision becomes `ALLOW` with `notice="warn"`; a `bare` one becomes `ALLOW` with `notice="inform"`. `Decision` gains no new members. |
| T-15 | A halted session is never idle-evicted; it stays halted until `Engine.release_session`. Other session records are evicted after `sess_idle_ms` without a call. |
| T-16 | The orphan-candidate marker (F1) is not stored by the tracker. A provisional call whose artifact nobody consumes within the TTL (F6) is the same fact, and the engine already sees consumption through `ledger_refs`. |
| T-17 | Blast-radius numbers the v3 spec leaves open: lexical = 0.6 for an irreversible verb in the tool name, +0.2 for a scope noun, +0.2 when a `money`/`count` field hint is present (clip to [0, 1]); structural starts at 0.3, +0.4 write, + min(0.3, log10(len)/4) for the longest id list, −0.3 read (clip); `b = 0.5·lexical + 0.5·structural`, then the F3 annotation terms, floor 0.05. |
| T-18 | Response fingerprints in synth are computed with the same `resp_fingerprint` the proxy uses, over the generated payload's sorted-key JSON. |

**Interaction with Amendment A1** (velocity tail probability, log-scale magnitude, generator spikes; ordered 2026-11-10, not yet on this branch): no task here touches magnitude or velocity scoring. Both edit `agentfw/synth/attacks.py` and `generate.py`; whichever lands second rebases. If A1 lands first, Task 4.1 runs on the A1 engine.

## Global Constraints

- Python **3.12**; `scripts/gate.sh` (ruff check, ruff format --check, mypy --strict, pytest) green at the end of every task; CI's `python -m agentfw.bench.ci_regress` green.
- No scipy in `agentfw/` (`tests/test_no_scipy.py`); numpy only.
- `score()` is pure: `replay(..., strict=True)` must never raise `LeakError` with any switch on.
- `tests/golden/*.sha256` and `agentfw/bench/baseline_metrics.json` are unchanged by every task (T-03). A task that needs to change one stops and asks.
- Every new switch defaults to current behaviour (T-03). New state records register with `register_record` and round-trip through `DictStore.dump/load`.
- Response content never enters `agentfw/core/detectors/` or `agentfw/core/trajectory/`: only `ResponseMeta` fields (`item_count`, `bytes`, `error`, `ids`, `fp`) cross.
- Raw ids never appear in reasons or evidence (existing rule): use paths, hashes and producer tools.
- Bounds: ledger ≤ 2,000 exact entries per session, Bloom beyond; recent-call ring 32; loop window 8 (32 for read-only + idempotent tools); MinHash 64 permutations over the first 4 KB of response text.
- `tests/perf/test_latency.py` p99 bound still holds with every trajectory switch on (Task 4.1 adds that run).
- Cite OnTrack (arXiv:2610.12375) in the module docstrings of the ledger, vital, provisional and session-verdict code. Paraphrase only; never paste the paper's prose, figures or tables. Nothing is named "OnTrack" or "OTAP".
- Reviewer-protocol (`outcomes="review"`) numbers are always labelled optimistic (`PROTOCOL_LABELS`).

## Review Focus

1. **Interleaved sessions in replay.** Replay orders by timestamp across sessions, so two sessions with the same tool sequence interleave call by call. Ledger, recent-call ring, loop windows and `sess:` records must never cross sessions. Pinned in Tasks 1.1 and 1.5.
2. **An id inside the same call that "produces" it.** A value first seen in this call's own args must not count as its own producer, and an id that is in `task_ids` and later comes back in a response keeps `source="task_input"`. Pinned in Task 1.1.
3. **Errored or fingerprint-less responses.** `response_meta` is `None`, or has `error=True` and `fp=None`: ΔI is undefined, so the call must neither count as "no new information" (spin) nor crash. Pinned in Task 1.5.
4. **Ledger overflow.** Past 2,000 ids, membership comes from the Bloom filter and there is no producer. The gate must treat a Bloom-only id as present (no `prereq_missing` block). Pinned in Task 3.1.
5. **Idle eviction mid-session.** A session record evicted for idleness that gets another call starts a fresh window (`OK`), and a halted session is never evicted (T-15). Pinned in Task 3.3.

---

## File structure

| Path | Responsibility |
|---|---|
| `agentfw/core/envelope.py` (modify) | `RawCall.task_ids`, `ResponseMeta.fp`, `LedgerRef`, `RecentCall`, new `ActionEnvelope` fields, `TrajectoryInfo` on `DetectorResult`, `Reason.maturity/regime` |
| `agentfw/core/fingerprint.py` (new) | `call_fingerprint`, `arg_tokens`, `resp_fingerprint` (numpy MinHash), `minhash_jaccard` |
| `agentfw/core/ledger.py` (new) | `ArtifactLedger` per session: producers, consumers, Bloom overflow, dump/load |
| `agentfw/core/normalize.py` (modify) | `SessionTracker` owns ledgers + recent-call rings; `normalize` fills `ledger_refs`, `call_fp`, `recent`, `task_ids` |
| `agentfw/core/hints.py` (modify) | `ToolAnnotations`; `ToolHint.annotations`, `ToolHint.has_schema` |
| `agentfw/core/schemas.py` (new) | `ToolSchema`, `load_schemas`, `hints_from_schemas` |
| `agentfw/core/blast.py` (new) | lexical / structural / annotation blast radius, `irreversible`, `regime_of` |
| `agentfw/core/detectors/vital.py` (new) | `VitalDetector`: `s_loop`, `s_spin`, `s_stall` |
| `agentfw/core/detectors/sequence.py` (modify) | ledger-backed id flow, producer counts, age discount, evidence `n` |
| `agentfw/core/trajectory/flags.py` (new) | `Flag`, weights, `derive_flags` |
| `agentfw/core/trajectory/session.py` (new) | `SessionRecord`, `SessionBook` (window, density, verdict, eviction, release) |
| `agentfw/core/trajectory/provisional.py` (new) | `ProvEntry`, lifecycle advance |
| `agentfw/core/trajectory/gate.py` (new) | `gate()` for irreversible calls |
| `agentfw/core/trajectory/escalate.py` (new) | `Judge` protocol, `escalate()` |
| `agentfw/core/trajectory/authority.py` (new) | `maturity_of`, `apply_ceiling` |
| `agentfw/core/trajectory/decide.py` (new) | `decide()`: the post-fusion pipeline the engine calls |
| `agentfw/core/render.py` (new) | `render_reason` (F8 rendered text) |
| `agentfw/core/engine.py`, `config.py`, `state.py` (modify) | wiring, switches, `key_sess` |
| `agentfw/synth/*` (modify) | annotations in catalog, `task_ids`, response `fp`, edge ground truth, `loop`/`spin`/`stall`/`splice`, `schemas.jsonl` |
| `agentfw/bench/replay.py` (modify) | `calibration` split, new `ScoreRow` columns |
| `agentfw/bench/metrics.py` (modify) | runaway exclusion, precision, AUROC |
| `agentfw/bench/session_metrics.py` (new) | session labels, decision metrics, halt metrics, sensitivity, first-flag baseline, length-matched set |
| `agentfw/bench/attribution.py` (new) | leave-one-group-out attribution table |
| `agentfw/bench/calibrated.py` + `calibrated.json` (new) | normalizer hash and the frozen calibrated constants |
| `agentfw/bench/trajectory.py` (new) | Task 4.1 measurement CLI |
| `agentfw/sandbox/proxy_py/*`, `tools/*`, `run.py`, `package.py` (modify) | tools/list capture, response `fp`, annotations, `task_ids` |

---

## Phase 0 — evaluation scaffolding (F10, Tier 0 items)

### Task 0.1: Spec in repo, clarifications record, response-content guard

**Files:**
- Create: `docs/decisions/009-trajectory-clarifications.md`, `tests/test_core_purity.py`
- Already copied: `docs/feature-spec-trajectory-monitoring.md`

- [ ] **Step 1: Write the failing test** `tests/test_core_purity.py::test_no_response_content_in_scoring_core`: walk `agentfw/core/detectors/*.py` and `agentfw/core/trajectory/*.py` (the second dir may not exist yet; skip missing dirs), assert no file contains the tokens `response_text`, `CallToolResult`, `.content` or `TextContent`. Add `test_guard_sees_a_violation`, which writes a temp file containing `response_text` and calls the same scanner function `scan_for_response_content(paths) -> list[str]` (defined in the test module), asserting one hit.
- [ ] **Step 2: Run it** — `uv run pytest tests/test_core_purity.py -v`. Expected: both PASS (guard test proves the scanner can see a violation; the tree is clean today).
- [ ] **Step 3: Write `009-trajectory-clarifications.md`** with the T-01…T-18 table verbatim from this plan, the A1 note, and a "Paper citation" line (title, arXiv id, licence note: paraphrase only).
- [ ] **Step 4: Commit** — `git add docs/feature-spec-trajectory-monitoring.md docs/decisions/009-trajectory-clarifications.md tests/test_core_purity.py && git commit -m "docs: trajectory feature spec, clarifications 009; response-content guard"`

### Task 0.2: Calibration partition (three-way split)

**Files:**
- Modify: `agentfw/bench/replay.py` (`SplitSpec`, `assign_split`, `Split`), `agentfw/bench/metrics.py` (`compute_all`, `_valid` callers), `agentfw/bench/report.py` (show calibration counts)
- Test: `tests/bench/test_replay.py`, `tests/bench/test_metrics.py`

**Interfaces:**
- Produces: `SplitSpec.calibration_frac: float = 0.0`; `Split = Literal["train", "test", "calibration"]`; `assign_split` marks every envelope of the calibration agents `"calibration"`; `calibration_rows(rows: Sequence[ScoreRow]) -> list[ScoreRow]` in `metrics.py`.

- [ ] **Step 1: Write failing tests**

```python
def test_calibration_partition_is_by_agent_and_disjoint():
    raws = [make_raw(agent_id=f"a{i % 20}", session_id=f"s{i}", ts_unix_ms=1_700_000_000_000 + i)
            for i in range(400)]
    a = assign_split(raws, SplitSpec(kind="time", calibration_frac=0.15, seed=3))
    cal_agents = {r.agent_id for r in raws if a[r.envelope_id] == "calibration"}
    assert len(cal_agents) == 3                       # round(0.15 * 20)
    assert all(a[r.envelope_id] == "calibration" for r in raws if r.agent_id in cal_agents)
    assert {a[r.envelope_id] for r in raws if r.agent_id not in cal_agents} == {"train", "test"}

def test_zero_calibration_frac_is_the_old_split():
    assert assign_split(raws, SplitSpec()) == assign_split(raws, SplitSpec(calibration_frac=0.0))

def test_compute_all_ignores_calibration_rows():
    # rows built so that adding calibration rows with extreme scores changes nothing
    assert compute_all(rows + cal_rows, warm_only=False, n_boot=0) == compute_all(rows, warm_only=False, n_boot=0)
```

- [ ] **Step 2: Run** `uv run pytest tests/bench/test_replay.py tests/bench/test_metrics.py -k calibration -v` — Expected: FAIL (`calibration_frac` unknown).
- [ ] **Step 3: Implement.** Calibration agents are drawn first, with `np.random.default_rng(seed)` over the sorted agent list, `n = round(calibration_frac · n_agents)`; the time/agent split then runs on the remaining envelopes only (time cut computed over them). `compute_all` drops `split == "calibration"` rows before anything else. The report prints `n_calibration_rows` / `calibration_agents`.
- [ ] **Step 4: Run** the same tests plus `uv run pytest tests/bench -q` and `uv run python -m agentfw.bench.ci_regress` — Expected: PASS, goldens unchanged.
- [ ] **Step 5: Commit** — `feat(bench): calibration partition (F10 three-way split)`

### Task 0.3: Precision at base rate, AUROC, recalibration guard

**Files:**
- Modify: `agentfw/bench/metrics.py` (`Metrics`, `compute_all`)
- Create: `agentfw/bench/calibrated.py`, `agentfw/bench/calibrated.json`
- Test: `tests/bench/test_metrics.py`, `tests/bench/test_calibrated.py`

**Interfaces:**
- Produces: `precision_at(recall: float, fpr: float, base_rate: float) -> float`; `auroc(scores: FloatArray, y: BoolArray) -> float`; `Metrics.precision_global: float`, `Metrics.attack_base_rate: float` (test-set envelope prevalence); `NORMALIZER_FILES: tuple[str, ...]` (repo-relative paths of `core/normalize.py`, `core/patterns.py`, `core/detectors/base.py`); `normalizer_hash() -> str` (sha256 over those files' bytes, in order); `load_calibrated() -> CalibratedConstants` (pydantic: `normalizer_hash: str`, `constants: dict[str, float]`, `history: list[str]`); CLI `python -m agentfw.bench.calibrated --restamp --reason TEXT`.

- [ ] **Step 1: Write failing tests**

```python
def test_precision_at_base_rate():
    assert precision_at(0.9, 0.01, 0.015) == pytest.approx(0.9*0.015 / (0.9*0.015 + 0.01*0.985))
    assert math.isnan(precision_at(0.0, 0.0, 0.015))

def test_auroc_perfect_and_random():
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([False, False, True, True])) == 1.0
    assert auroc(np.array([0.5]*4), np.array([False, True]*2)) == 0.5

def test_calibrated_constants_match_normalizer():
    assert load_calibrated().normalizer_hash == normalizer_hash(), (
        "normalizer changed: recalibrate, or restamp with --reason if no constant depends on it")
```

- [ ] **Step 2: Run** — Expected: FAIL (missing names / file).
- [ ] **Step 3: Implement.** `calibrated.json` is seeded from today's `EngineConfig` values that were tuned on the bench: `z0`, `tau`, `prereq_rate`, `cal_alpha`, `novelty_full_n`, and the current hash; `history` = `["2026-11-11 initial stamp: tiers 0-2 constants"]`. `--restamp` rewrites the hash and appends `"<date> <reason>"`. `report.json` gains `normalizer_hash`. `auroc` uses the same tie handling as `auprc` (rank-based, ties averaged).
- [ ] **Step 4: Run** `uv run pytest tests/bench -q` — Expected: PASS.
- [ ] **Step 5: Commit** — `feat(bench): precision at base rate, AUROC, normalizer-hash recalibration guard (F10)`

### Task 0.4: Decision column and attribution table

**Files:**
- Modify: `agentfw/bench/replay.py` (`ScoreRow.decision`)
- Create: `agentfw/bench/attribution.py`
- Test: `tests/bench/test_attribution.py`

**Interfaces:**
- Consumes: `ablation.replay_detector`, `registry.make_detector`.
- Produces: `ScoreRow.decision: str | None = None` (last field; `res.decision.value` or `None`); `AttributionGroup(name: str, drop_detector: str | None = None, cfg_off: Mapping[str, object] = {})` (frozen dataclass); `DEFAULT_GROUPS: tuple[AttributionGroup, ...]` = one group per name in `ALL_DETECTORS`; `attribution_table(raws, labels, *, groups, cfg, outcomes, split) -> dict[str, dict[str, int]]` (family → group name or `"redundant"` → attacks carried); `detected_attacks(rows) -> set[str]` (attack ids with any row whose decision is `block` or `require_approval`).

- [ ] **Step 1: Write failing tests**

```python
def test_attribution_is_leave_one_group_out():
    raws, labels = ci_dataset()                       # agentfw.bench.ci_regress
    table = attribution_table(raws, labels, groups=DEFAULT_GROUPS, cfg=EngineConfig(),
                              outcomes="review", split=SplitSpec())
    full = detected_attacks(replay_rows(raws, labels, "engine"))
    assert sum(sum(v.values()) for v in table.values()) == len(full)
    for g in DEFAULT_GROUPS:                          # T-13: exactness by construction
        without = detected_attacks(replay_rows(raws, labels, f"engine-no-{g.name}"))
        carried = {a for a in full - without}
        assert sum(v.get(g.name, 0) for v in table.values()) == len(carried)
```

Here `replay_rows(raws, labels, name)` is a test helper wrapping `replay(..., outcomes="review")`.

- [ ] **Step 2: Run** `uv run pytest tests/bench/test_attribution.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement.** One full replay plus one per group (`parallel_map` from `ablation.py`). An attack carried by two groups at once (undetected without either) counts under each; document that. A group with `cfg_off` replays with `cfg.replace(**cfg_off)` (used later for `gate`, `provisional`).
- [ ] **Step 4: Run** `uv run pytest tests/bench -q` — Expected: PASS.
- [ ] **Step 5: Commit** — `feat(bench): decision column; leave-one-group-out attribution table (F10)`

---

## Phase 1 — session substrate (F1, F2, F3; Tier 1)

### Task 1.1: Artifact ledger and task-input seeding

**Files:**
- Create: `agentfw/core/ledger.py`
- Modify: `agentfw/core/envelope.py`, `agentfw/core/normalize.py`
- Test: `tests/core/test_ledger.py`, `tests/core/test_normalize.py`

**Interfaces:**
- Produces:
  - `RawCall.task_ids: tuple[str, ...] = Field(default=(), max_length=64)`.
  - `LedgerRef(artifact_id: str, path: str, producer_seq: int, producer_tool: str | None, source: Literal["args", "response", "task_input"])` (frozen slots).
  - `ActionEnvelope.ledger_refs: tuple[LedgerRef, ...] = ()`, `ActionEnvelope.task_ids: tuple[str, ...] = ()` (new fields with defaults, after `blast_radius`).
  - `ArtifactLedger(max_entries: int = 2000)` with `lookup(v: str) -> LedgerEntry | None`, `__contains__(v) -> bool` (exact or Bloom), `seed_task(ids: Iterable[str]) -> None`, `record(e: ActionEnvelope) -> None`, `to_dict()` / `from_dict()`, `degraded: bool`. `LedgerEntry(producer_seq: int, producer_tool: str | None, source: str, consumers: list[int])`.
  - `SessionTracker.ledger(session_id) -> ArtifactLedger | None`; `SessionTracker.dump(path)` / `SessionTracker.load(path)` (JSONL, sorted, orjson `OPT_SORT_KEYS`).

- [ ] **Step 1: Write failing tests**

```python
def test_response_id_becomes_producer_and_later_arg_is_a_ref():
    t = SessionTracker()
    read = make_env(make_raw(seq_no=0, tool_id="stripe.retrieve_charge", args={"charge_id": "ch_aaaaaaaa11"},
                             response_meta=ResponseMeta(ids=("re_bbbbbbbb22",))), t)
    t.observe(read)
    w = make_env(make_raw(seq_no=1, tool_id="stripe.create_refund", args={"refund_id": "re_bbbbbbbb22"}), t)
    assert w.ledger_refs == (LedgerRef("re_bbbbbbbb22", "refund_id", 0, "stripe.retrieve_charge", "response"),)

def test_task_ids_seed_before_the_first_call():
    t = SessionTracker()
    e = make_env(make_raw(seq_no=0, args={"charge_id": "ch_aaaaaaaa11"}, task_ids=("ch_aaaaaaaa11",)), t)
    assert e.ledger_refs[0].source == "task_input" and e.ledger_refs[0].producer_seq == 0

def test_task_input_source_survives_later_response():          # Review Focus 2
    ...  # task id, later returned in a response: lookup().source stays "task_input"

def test_own_args_are_not_their_own_producer():                # Review Focus 2
    e = make_env(make_raw(seq_no=0, args={"a": "ch_aaaaaaaa11", "b": "ch_aaaaaaaa11"}), SessionTracker())
    assert e.ledger_refs == ()

def test_interleaved_sessions_do_not_share_ledgers():          # Review Focus 1
    ...  # s1 produces re_x; s2's write of re_x has no ref

def test_overflow_degrades_to_bloom_membership():
    led = ArtifactLedger(max_entries=4)
    ...  # 6 ids recorded: the last two are `in led`, lookup() is None, led.degraded is True

def test_tracker_dump_load_round_trip(tmp_path): ...           # replay a 30-call session; dump; load; identical refs
```

Every id literal in these tests must satisfy `patterns.is_id_like`; copy the id style already used in `tests/core/test_normalize.py` if the ones above do not.

- [ ] **Step 2: Run** `uv run pytest tests/core/test_ledger.py tests/core/test_normalize.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement.** `normalize` computes `ledger_refs` for every id-like (`is_id_like`) string in `flat_args`, in sorted path order, from the tracker's ledger *plus* this raw call's own `task_ids` (so the first call sees its seeds). `SessionTracker.observe` seeds `task_ids` (source `task_input`, seq 0, tool `None`) before recording the call; then arg ids first seen become entries with source `args` and producer = this call, and response ids first seen become entries with source `response`; existing entries get `consumers.append(seq_no)`. The first sighting wins (an entry is never overwritten). Bloom overflow reuses `BloomFilter(capacity=8 * max_entries)` and logs once per session (`logging.getLogger(__name__).warning`). `session_ids_seen` is untouched (T-04). Module docstring cites the paper (Def. 1, ledger lookup).
- [ ] **Step 4: Run** the tests, then `uv run pytest -q -m "not slow and not live"` and `uv run python -m agentfw.bench.ci_regress` — Expected: PASS; goldens unchanged (nothing reads the new fields yet).
- [ ] **Step 5: Commit** — `feat(core): per-session artifact ledger with task-input seeding (F1)`

### Task 1.2: Fingerprints

**Files:**
- Create: `agentfw/core/fingerprint.py`
- Modify: `agentfw/core/envelope.py` (`ResponseMeta.fp`), `agentfw/bench/calibrated.py` (`NORMALIZER_FILES` += `core/fingerprint.py`), `agentfw/bench/calibrated.json` (restamp)
- Test: `tests/core/test_fingerprint.py`

**Interfaces:**
- Produces: `MINHASH_K = 64`, `SHINGLE_WORDS = 3`, `MAX_RESP_BYTES = 4096`; `ResponseMeta.fp: tuple[int, ...] | None = None` (validator: `None` or exactly 64 ints in `[0, 2**61 - 1)`); `call_fingerprint(tool_id: str, flat_args: Mapping[str, tuple[Scalar, ...]]) -> int` (xxh64 of sorted-key orjson of `[tool_id, sorted(flat_args.items())]`; ids are *not* normalized); `arg_tokens(flat_args) -> frozenset[int]` (xxh32 of `f"{path}={value}"`); `resp_fingerprint(text: str, item_count: int | None, error: bool) -> tuple[int, ...] | None` (`None` when the shingle set is empty); `minhash_jaccard(a: tuple[int, ...], b: tuple[int, ...]) -> float`.

- [ ] **Step 1: Write failing tests**

```python
def test_call_fp_ignores_key_order_but_not_ids():
    assert call_fingerprint("t.x", {"a": (1,), "b": ("u",)}) == call_fingerprint("t.x", {"b": ("u",), "a": (1,)})
    assert call_fingerprint("t.x", {"id": ("ch_aaaaaaaa11",)}) != call_fingerprint("t.x", {"id": ("ch_aaaaaaaa22",)})

def test_minhash_estimates_jaccard():
    a = resp_fingerprint(" ".join(f"w{i}" for i in range(400)), 10, False)
    b = resp_fingerprint(" ".join(f"w{i}" for i in range(200, 600)), 10, False)
    assert minhash_jaccard(a, a) == 1.0
    assert minhash_jaccard(a, b) == pytest.approx(1/3, abs=0.15)

def test_only_first_4kb_counts_and_meta_tokens_matter():
    base = "x " * 5000
    assert resp_fingerprint(base, 1, False) == resp_fingerprint(base + "tail", 1, False)
    assert resp_fingerprint("same text here", 1, False) != resp_fingerprint("same text here", 1, True)

def test_fingerprint_is_deterministic_across_processes(): ...   # subprocess recomputes; equal tuples
```

- [ ] **Step 2: Run** `uv run pytest tests/core/test_fingerprint.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement.** Shingles = word 3-grams of the UTF-8-safe first 4,096 bytes, lowercased and whitespace-split, plus the tokens `#ic:<floor(log2(1 + item_count))>` (when not `None`) and `#err:<0|1>`; each shingle is hashed with `xxh64_intdigest`. MinHash is numpy universal hashing `h_i(x) = (a_i · x + b_i) mod (2**61 − 1)` with 64 `(a_i, b_i)` drawn once from `np.random.default_rng(0x5EED)` and frozen as a module constant (overflow-safe: split `x` into 30-bit halves or use Python ints via `np.frompyfunc`; pick one and test against a pure-Python reference on 50 random inputs).
- [ ] **Step 4: Restamp** — `uv run python -m agentfw.bench.calibrated --restamp --reason "add fingerprint.py; no calibrated constant depends on it yet"`; run `uv run pytest tests/bench/test_calibrated.py tests/core/test_fingerprint.py -v`. Expected: PASS.
- [ ] **Step 5: Commit** — `feat(core): call and response fingerprints (F2)`

### Task 1.3: Generator — task ids, response fingerprints, edge ground truth

**Files:**
- Modify: `agentfw/synth/grammar.py` (`PlannedCall`, `_build_args`, `_meta`), `agentfw/synth/sampler.py` (first call's `task_ids`)
- Test: `tests/synth/test_grammar.py`, `tests/synth/test_ledger_edges.py`

**Interfaces:**
- Consumes: `resp_fingerprint` (Task 1.2), `SessionTracker` ledger (Task 1.1).
- Produces: `PlannedCall.external_ids: tuple[str, ...] = ()` (id values of args whose `ArgSpec.source == "external"`); `PlannedCall.sources: tuple[tuple[str, int], ...] = ()` (`(arg path, index in this walk's emitted calls of the producing call)`, only when the producer was emitted); synth `ResponseMeta.fp` = `resp_fingerprint(orjson.dumps(produced, option=OPT_SORT_KEYS).decode(), item_count, False)`; a session's first `RawCall.task_ids` = its calls' `external_ids` deduplicated in order, cut to 64.

- [ ] **Step 1: Write failing tests**

```python
def test_external_ids_seed_first_call(sampled_session):
    first = sampled_session[0]
    externals = [i for c in sampled_session for i in planned(c).external_ids]
    assert first.task_ids == tuple(dict.fromkeys(externals))[:64]
    assert all(c.task_ids == () for c in sampled_session[1:])

def test_ledger_recovers_grammar_edges():                      # F1 acceptance
    # 300 benign walks over all personas (id_flow_rate forced to 1.0), seq-numbered, run through
    # SessionTracker; expected edges from PlannedCall.sources, found edges from ledger_refs
    p, r = edge_precision_recall(...)
    assert p >= 0.99 and r >= 0.99

def test_no_extra_rng_draws():
    # same seed: every RawCall field except task_ids and response_meta.fp is byte-identical to a
    # generation with fingerprints/task ids stripped
```

- [ ] **Step 2: Run** `uv run pytest tests/synth -k "external_ids or ledger_recovers or extra_rng" -v` — Expected: FAIL.
- [ ] **Step 3: Implement.** `sources` is filled where `_resolve` binds an alias whose producing step was emitted (track alias → emitted index). Nothing draws from `rng`. `edge_precision_recall(walks) -> tuple[float, float]` is a test helper in `tests/synth/test_ledger_edges.py`.
- [ ] **Step 4: Run** `uv run pytest tests/synth -q`, then `uv run python -m agentfw.bench.ci_regress` — Expected: PASS; the engine golden is unchanged (no detector reads `task_ids` / `fp` yet). If the generator determinism test pins trace bytes, update its expected hash in this commit and say so in the message.
- [ ] **Step 5: Commit** — `feat(synth): task-input ids, response fingerprints, ledger edge ground truth (F1, F2)`

### Task 1.4: Id flow over the ledger

**Files:**
- Modify: `agentfw/core/config.py`, `agentfw/core/detectors/sequence.py` (`_s_flow`, `_update_flow`, `FlowRecord`)
- Test: `tests/core/test_sequence_flow.py`

**Interfaces:**
- Consumes: `ActionEnvelope.ledger_refs`.
- Produces: `EngineConfig.flow_ledger: bool = False`; `FlowRecord.producers: dict[str, int]` (producer tool → count, from `ledger_refs`; `from_dict` defaults to `{}` for old snapshots); `SequenceDetector.usual_producer(agent, tool, path) -> str | None` (most frequent producer tool, ties by name); flow evidence gains `ref_source` (`"args" | "response" | "task_input" | None`) and `n` (= `FlowRecord.total`).

- [ ] **Step 1: Write failing tests**

```python
def test_prereq_skip_fabricated_id_scores_r_flow():            # F1 acceptance
    eng = warm_flow_engine(cfg=EngineConfig(flow_ledger=True), present_rate=0.95)   # 40 looked-up writes
    s = flow_score(eng, args={"charge_id": "ch_fabricated1"}, task_ids=())
    assert s == pytest.approx(0.95)

def test_same_write_with_task_seeded_id_scores_zero():         # F1 acceptance
    eng = warm_flow_engine(cfg=EngineConfig(flow_ledger=True), present_rate=0.95)
    assert flow_score(eng, args={"charge_id": "ch_fromtask01"}, task_ids=("ch_fromtask01",)) == 0.0

def test_switch_off_keeps_v3_behaviour():
    assert flow_score(warm_flow_engine(EngineConfig(), 0.95), args=..., task_ids=("ch_fromtask01",)) == pytest.approx(0.95)
```

`warm_flow_engine` / `flow_score` are helpers in the test module (session per write: optional lookup, then write).

- [ ] **Step 2: Run** — Expected: FAIL.
- [ ] **Step 3: Implement.** With `flow_ledger`, "present" means the value has a `LedgerRef` in this envelope *or* is in `session_ids_seen` (Bloom-only ids, Review Focus 4); `r_flow` history counts presence the same way. Without it, nothing changes.
- [ ] **Step 4: Run** `uv run pytest tests/core -q && uv run python -m agentfw.bench.ci_regress` — Expected: PASS, goldens unchanged.
- [ ] **Step 5: Commit** — `feat(core): id flow over the artifact ledger, behind flow_ledger (F1)`

### Task 1.5: Vital-signs detector

**Files:**
- Modify: `agentfw/core/envelope.py` (`RecentCall`, `ActionEnvelope.call_fp`, `ActionEnvelope.recent`), `agentfw/core/normalize.py`, `agentfw/core/engine.py`, `agentfw/core/config.py`, `agentfw/bench/registry.py`, `tests/factories.py`
- Create: `agentfw/core/detectors/vital.py`
- Test: `tests/core/test_vital.py`

**Interfaces:**
- Consumes: fingerprints (1.2), ledger (1.1), `SequenceDetector.is_write`.
- Produces:
  - `RecentCall(seq_no: int, tool_id: str, call_fp: int, arg_tokens: frozenset[int], error: bool, delta_i: float | None, new_artifacts: int)`; `ActionEnvelope.call_fp: int = 0`, `ActionEnvelope.recent: tuple[RecentCall, ...] = ()` (oldest first, ≤ 32, this session only). `delta_i` = `1 − max MinHash-Jaccard` against earlier fingerprints in the ring, `None` when the call had no `fp`; `new_artifacts` = ledger entries this call created.
  - `VITAL = "vital"`; `KNOWN_DETECTORS = ALL_DETECTORS | {VITAL}` (`ALL_DETECTORS` keeps meaning the v1 four). `Engine(enabled=...)` validates against `KNOWN_DETECTORS`.
  - `VitalDetector(cfg, hints, is_write: Callable[[str, str], bool])`, `name = "vital"`, emitting three `DetectorScore`s with `field` `"__loop__"`, `"__spin__"`, `"__stall__"`; `update` is a no-op.
  - Config: `vital_w_loop: int = 8`, `vital_w_loop_polling: int = 32`, `vital_r_max: int = 3`, `vital_jaccard: float = 0.9`, `vital_w_spin: int = 4`, `vital_spin_di: float = 0.1`, `vital_w_stall: int = 6`, `vital_stall_di: float = 0.3`.
  - Registry: `"engine-vital"` = v1 + vital.
  - `tests/factories.py`: `run_session(engine, raws, tracker=None, outcome=Outcome.ALLOWED) -> list[DetectorResult]` (normalize, score, update, observe, in order).

- [ ] **Step 1: Write failing tests**

```python
def test_loop_fires_at_second_duplicate():                     # F2 acceptance, T-08
    raws = [read_call(seq) for seq in range(6)]                 # identical read, identical response
    loop = [vital(r, "__loop__") for r in run_session(Engine(enabled=KNOWN_DETECTORS), raws)]
    assert loop[:3] == [0.0, 0.7, 1.0]

def test_retry_exemption_after_error():                        # Paper scenario F analogue
    raws = [read_call(0, error=True)] + [read_call(i, error=True) for i in range(1, 5)]
    loop = [...]
    assert loop[1:4] == [0.0, 0.0, 0.0] and loop[4] == 1.0

def test_near_duplicate_scores_0_7(): ...                      # one of 12 arg tokens differs
def test_polling_tools_use_the_long_window(): ...              # readOnly+idempotent hint: dup 9 calls back still counts
def test_spin_within_w_spin():                                 # varying args, identical fp
    assert first_index(spin_scores, 1.0) <= 4
def test_stall_six_reads_no_new_artifacts():
    assert stall_scores[6] == 1.0 and stall_scores[5] == 0.0
def test_stall_half_when_reads_low_gain(): ...                 # 0.5 branch
def test_errored_or_fp_less_calls_never_count_as_spin():       # Review Focus 3
    ...  # 6 calls with response_meta=None or fp=None: spin stays 0, no exception
def test_interleaved_sessions_keep_separate_rings():           # Review Focus 1
def test_vital_never_moves_the_fused_score():                  # T-07
    for v1, vt in zip(run_session(Engine(), raws), run_session(Engine(enabled=KNOWN_DETECTORS), raws)):
        assert v1.fused_score == vt.fused_score and v1.decision == vt.decision
```

- [ ] **Step 2: Run** `uv run pytest tests/core/test_vital.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement.** The tracker keeps a per-session `deque(maxlen=32)` of `RecentCall` plus the matching fingerprints; `normalize` copies the deque to a tuple and sets `call_fp`. Loop: count exact `call_fp` matches among the last `w` entries (`w` = `vital_w_loop_polling` when the tool's annotations say `readOnlyHint ∧ idempotentHint`, else `vital_w_loop`); when the most recent match errored, skip scoring while the run of consecutive errored duplicates is ≤ `vital_r_max`. Spin = 1 when the last `vital_w_spin` entries all have `delta_i is not None`, `delta_i < vital_spin_di` and `not error`. Stall over the last `vital_w_stall` entries (0 when fewer exist): 1.0 when every entry has `new_artifacts == 0` and none is a write (`is_write`); otherwise 0.5 when every entry is a read with `delta_i is not None` and `delta_i < vital_stall_di` (busy re-reading, even if an id trickles in); otherwise 0. The engine fuses only non-vital scores (T-07) and appends vital scores and their reasons (score > `reason_min`) after. Docstring cites the paper's L1 layer.
- [ ] **Step 4: Run** `uv run pytest tests/core -q && uv run pytest tests/perf -q -m slow && uv run python -m agentfw.bench.ci_regress` — Expected: PASS; goldens unchanged.
- [ ] **Step 5: Commit** — `feat(core): vital-signs detector (loop, spin, stall), opt-in (F2)`

### Task 1.6: Runaway families and session labels

**Files:**
- Modify: `agentfw/synth/attacks.py` (`loop`, `spin`, `stall`, `ATTACKS`, `_CANDS`), `agentfw/synth/generate.py` (defaults unchanged), `agentfw/bench/metrics.py` (`_valid`)
- Create: `agentfw/bench/session_metrics.py` (first part)
- Test: `tests/synth/test_attacks_runaway.py`, `tests/bench/test_session_metrics.py`

**Interfaces:**
- Produces: `RUNAWAY_FAMILIES = frozenset({"loop", "spin", "stall"})` (in `attacks.py`); strength `r` (repeats) log-uniform on [5, 60] for all three, `Label.strength = r`; `SessionLabel = Literal["benign", "runaway", "attacked"]`; `session_labels(rows: Sequence[ScoreRow]) -> dict[str, SessionLabel]` (attacked beats runaway beats benign); `vital_false_flag_rate(rows) -> float` (share of benign sessions with any call whose `__loop__ == 1` or `__spin__ == 1` or `__stall__ == 1` in `per_detector_json`).
- Family semantics: `loop` inserts `r` copies of a read right after it (same args, same `response_meta`, 0.5–3 s gaps); `spin` inserts `r` reads of the same tool with fresh id args from the persona pool and the original call's `response_meta` (same `fp`, no new ids); `stall` inserts `r` reads (persona tools, kind read) whose `response_meta.ids` are a subset of ids already seen in the session and whose `fp` is the earlier read's `fp`. Every inserted call is labelled `attack` with its family and the session's `attack_id`.

- [ ] **Step 1: Write failing tests**

```python
def test_runaway_families_are_opt_in():
    assert not RUNAWAY_FAMILIES & set(GenerateConfig().families)

def test_runaway_rows_excluded_from_per_call_metrics():
    assert compute_all(rows_with_loop, warm_only=False, n_boot=0) == compute_all(rows_without, ...)

def test_session_labels_priority(): ...
def test_loop_injection_detected_within_one_event():          # F2 acceptance, end to end
    # generate CI-sized traces with families=("loop",), replay engine-vital: for every loop attack,
    # the first call with __loop__ == 1.0 is at most 2 calls after the original read
def test_benign_vital_false_flag_rate_at_most_1pct():          # F2 acceptance
    assert vital_false_flag_rate(benign_rows) <= 0.01
```

- [ ] **Step 2: Run** `uv run pytest tests/synth/test_attacks_runaway.py tests/bench/test_session_metrics.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement** the three families with the existing helpers (`_renumber`, `_label`, `burst_duplicates` for spacing). `_valid` in `metrics.py` drops rows whose `attack_family in RUNAWAY_FAMILIES`.
- [ ] **Step 4: Run** `uv run pytest tests/synth tests/bench -q && uv run python -m agentfw.bench.ci_regress` — Expected: PASS. If the benign false-flag rate is above 1%, stop and report the offending pattern (likely grammar `retry` + `abandon`); do not loosen the bar.
- [ ] **Step 5: Commit** — `feat(synth,bench): loop/spin/stall runaway families and session labels (F2, F10)`

### Task 1.7: Annotations, blast radius, irreversibility, regime

**Files:**
- Modify: `agentfw/core/hints.py`, `agentfw/core/config.py`, `agentfw/core/engine.py`, `agentfw/synth/schema.py` (`ToolSpec.annotations`), `agentfw/synth/catalog/*.yaml`, `agentfw/synth/generate.py` (write `schemas.jsonl`), `agentfw/bench/registry.py` / `run.py` (load `schemas.jsonl` next to traces when present)
- Create: `agentfw/core/schemas.py`, `agentfw/core/blast.py`, `agentfw/bench/data/irreversibility_labels.yaml`
- Test: `tests/core/test_blast.py`, `tests/core/test_schemas.py`

**Interfaces:**
- Produces:
  - `ToolAnnotations(read_only: bool | None = None, destructive: bool | None = None, idempotent: bool | None = None, open_world: bool | None = None)` (frozen slots, in `hints.py`; core never imports `mcp`); `ToolHint.annotations: ToolAnnotations | None = None`; `ToolHint.has_schema: bool = False`.
  - `ToolSchema(tool_id: str, input_schema: dict[str, JsonValue], annotations: ToolAnnotations, captured_at: int, hash: str)`; `load_schemas(path: Path) -> dict[str, ToolSchema]` (last row per tool wins); `hints_from_schemas(schemas, base: Mapping[str, ToolHint]) -> dict[str, ToolHint]` (keeps `kind`/`field_hints`, sets `annotations`, `has_schema=True`).
  - `blast.py`: `IRREVERSIBLE_VERBS`, `SCOPE_NOUNS` (from §6.9); `lexical(tool_id, flat_paths, field_hints) -> float`; `structural(is_write: bool, id_list_len: int, has_item_count: bool) -> float`; `blast_radius(e: ActionEnvelope, hint: ToolHint | None, is_write: bool, has_item_count: bool) -> float` (T-17 numbers, then annotation terms: destructive +0.4; read-only −0.4 **only when corroborated** by `structural` seeing a read with no id-list arg; `idempotent is False ∧ not read_only` +0.1; `open_world` +0.1; floor 0.05, cap 1); `tool_blast_radius(tool_id, kind, annotations, field_hints) -> float` (static, for the Spearman test); `irreversible(b: float, hint, is_write) -> bool` = `destructive ∨ (is_write ∧ b ≥ irreversible_b)`; `Regime = Literal["history+schemas", "history_only", "schemas_only", "bare"]`; `regime_of(has_schema: bool, n_tool_obs: int, cold_n: int) -> Regime` (T-06).
  - Config: `br_mode: Literal["off", "score", "modulate"] = "off"`, `br_k: float = 0.6`, `irreversible_b: float = 0.6`. `"score"` computes `b` and puts it on reasons and `TrajectoryInfo`; `"modulate"` also sets `θ_eff = θ · (1 − br_k · b)` for the decision.
  - `TrajectoryInfo` (frozen slots, in `envelope.py`) is introduced here with `regime`, `blast_radius: float | None`, `irreversible: bool`, `threshold_eff: float`; later tasks append fields with defaults. `DetectorResult.trajectory: TrajectoryInfo | None = None`; the engine always fills it.
  - `SequenceDetector.n_tool(agent, tool) -> int` (= `TransRecord.c0[tool]`, 0 if absent).

- [ ] **Step 1: Write failing tests**

```python
def test_tool_ranking_matches_hand_labels():                   # F3 acceptance
    labels = yaml.safe_load(LABELS.read_text())               # tool_id -> 0..3, every catalog tool
    b = {t: tool_blast_radius(t, spec.kind, ann(spec), hints(spec)) for t, spec in catalog.items()}
    rho = scipy.stats.spearmanr([labels[t] for t in b], [b[t] for t in b]).statistic
    assert rho >= 0.8

def test_mislabelled_read_only_gets_no_discount():             # F3 adversarial acceptance
    honest = blast_radius(delete_env, ToolHint(kind="write"), is_write=True, has_item_count=False)
    lying = blast_radius(delete_env, ToolHint(kind="write", annotations=ToolAnnotations(read_only=True)),
                         is_write=True, has_item_count=False)
    assert lying >= honest

def test_destructive_hint_raises_and_marks_irreversible(): ...
def test_regime_four_values(): ...                             # T-06 table
def test_br_off_leaves_decisions_unchanged(): ...              # Engine(), schemas loaded: same decisions as without
def test_modulate_lowers_threshold_for_writes():
    r = Engine(EngineConfig(br_mode="modulate"), hints=h).score(refund_env)
    assert r.trajectory.threshold_eff == pytest.approx(r.threshold * (1 - 0.6 * r.trajectory.blast_radius))
def test_load_schemas_last_row_wins(tmp_path): ...
```

- [ ] **Step 2: Run** `uv run pytest tests/core/test_blast.py tests/core/test_schemas.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement.** Annotate every catalog tool in the YAML (`annotations: {read_only, destructive, idempotent, open_world}`), consistent with its `kind` and real-world semantics. Hand labels in `irreversibility_labels.yaml` use 0 = read, 1 = reversible write, 2 = costly to reverse, 3 = irreversible (money moved, message sent, record deleted), with one comment line per tool giving the reason. `generate` writes `schemas.jsonl` (one row per catalog tool, `input_schema` derived from `ArgSpec` types, `captured_at` = generation epoch). Bench run auto-loads `<traces>/schemas.jsonl` into the engine's hints; with `br_mode="off"` this changes nothing.
- [ ] **Step 4: Run** `uv run pytest -q -m "not slow and not live" && uv run python -m agentfw.bench.ci_regress` — Expected: PASS; goldens unchanged.
- [ ] **Step 5: Commit** — `feat(core): MCP annotations, lexical+structural blast radius, irreversibility, regime (F3, §6.9)`

### Task 1.8: Sandbox proxy — schema capture, response fingerprints, task ids

**Files:**
- Modify: `agentfw/sandbox/proxy_py/proxy.py` (`list_tools`, `SessionInfo`, first-call stamping), `agentfw/sandbox/proxy_py/response_meta.py`, `agentfw/sandbox/proxy_py/sinks.py` (`write_schema`), `agentfw/sandbox/tools/*_server.py` (annotations), `agentfw/sandbox/run.py` (task ids), `agentfw/sandbox/package.py` (`schemas.jsonl` in `_FILES`, manifest, validation)
- Test: `tests/sandbox/proxy/test_proxy.py`, `tests/sandbox/proxy/test_response_meta.py`, `tests/sandbox/test_package.py`, `tests/sandbox/tools/test_*_server.py`

**Interfaces:**
- Consumes: `resp_fingerprint`, `ToolSchema` row shape (Task 1.7).
- Produces: `SessionInfo.task_ids: tuple[str, ...] = ()`; `EnvelopeSink.write_schema(row: dict[str, JsonValue]) -> None`; `extract_response_meta` sets `fp` from the joined text blocks; a re-captured tool whose schema hash changed logs `WARNING tool-definition drift: <tool_id>` and writes the new row.

- [ ] **Step 1: Write failing tests** — `test_list_tools_writes_one_schema_row_per_tool`; `test_schema_drift_logged` (downstream returns a changed `inputSchema` on reconnect: `caplog` has the drift line, two rows written); `test_first_call_carries_task_ids_only`; `test_response_meta_fp_matches_core_function`; `test_every_server_tool_is_annotated` (each `list_tools()` entry has `annotations` with `readOnlyHint` and `destructiveHint` set); `test_package_includes_and_validates_schemas`.
- [ ] **Step 2: Run** `uv run pytest tests/sandbox -q -m "not live"` — Expected: FAIL.
- [ ] **Step 3: Implement.** Server annotations use `mcp.types.ToolAnnotations` and must agree with the synth catalog's annotations for the same tool ids (a test compares them). The runner fills `task_ids` with the id-like values (`is_id_like`) of the task template's resolved inputs.
- [ ] **Step 4: Run** `uv run pytest tests/sandbox -q -m "not live"` — Expected: PASS.
- [ ] **Step 5: Commit** — `feat(sandbox): capture tools/list schemas, response fingerprints, task ids (F2, F3, F1)`

---

## Phase 2 — provisional lifecycle and harder evaluation (F6, F10; Tier 2)

### Task 2.1: Session records, flags and the flag window

**Files:**
- Create: `agentfw/core/trajectory/__init__.py`, `agentfw/core/trajectory/flags.py`, `agentfw/core/trajectory/session.py`
- Modify: `agentfw/core/state.py` (`key_sess`), `agentfw/core/engine.py` (session recording on **every** outcome), `agentfw/core/config.py`, `agentfw/core/envelope.py` (`TrajectoryInfo` += `flags`, `density`), `agentfw/bench/replay.py` (`ScoreRow.flags: str = ""`, `ScoreRow.density: float | None = None`)
- Test: `tests/core/test_flags.py`, `tests/core/test_session_book.py`

**Interfaces:**
- Produces:
  - `Flag(StrEnum)`: `BLOCK, PREREQ_MISSING, ID_FLOW_VIOLATION, LOOP, SPIN, STALL, OFF_TRACK, VELOCITY_BURST, PROVISIONAL, UNRESOLVED, NOVEL_READ, COLD_START`; `SEVERE: frozenset[Flag]` = the first eight; `flag_weight(f: Flag, cfg) -> float` (severe 1.0, `UNRESOLVED` → `prov_unresolved_weight`, else 0).
  - `derive_flags(per: Sequence[DetectorScore], decision: Decision, irreversible: bool, hint: ToolHint | None, cfg: EngineConfig) -> set[Flag]`: `BLOCK` ⇐ decision; `PREREQ_MISSING` ⇐ sequence `fired == "prereq"` and score ≥ `flag_min`; `ID_FLOW_VIOLATION` ⇐ irreversible ∧ sequence evidence `missing_id_path` is not `None` ∧ evidence `r_flow ≥ flag_min`; `LOOP` ⇐ `__loop__ == 1.0` unless the hint is read-only ∧ idempotent; `SPIN`/`STALL` ⇐ score == 1.0; `VELOCITY_BURST` ⇐ velocity ≥ `flag_min`; `NOVEL_READ` ⇐ novelty ≥ `flag_min` ∧ not irreversible. (`OFF_TRACK`, `PROVISIONAL`, `UNRESOLVED`, `COLD_START` come from later tasks.)
  - `key_sess(session_id) -> str` = `"sess:" + _esc(session_id)`; `SessionRecord` (`KIND = "sess"`): `last_ts: int`, `n_calls: int`, `window: list[tuple[int, list[str]]]` (seq, sorted flag names; last `session_window`), `verdict: str = "ok"`, `halted: bool = False`, plus fields later tasks add with defaults.
  - `SessionBook(store, cfg)`: `peek(session_id) -> SessionRecord | None` (pure); `density_with(rec, flags) -> float` (T-09, window including this call); `record(e, flags) -> None` (append, trim, `last_ts`, `n_calls`); `maybe_evict(now_ms) -> int` (every `sess_sweep_every` records, drop non-halted records idle > `sess_idle_ms`; returns how many).
  - Config: `session_window: int = 6`, `flag_min: float = 0.5`, `sess_idle_ms: int = 1_800_000`, `sess_sweep_every: int = 4096`, `prov_unresolved_weight: float = 0.125` (used from 2.2).
  - Engine: `score` computes flags and `density` purely; `update` calls `SessionBook.record` first, for every outcome (blocked calls must enter the window), then the existing logic. `score` caches `(envelope_id, fused, theta, flags)`.

- [ ] **Step 1: Write failing tests** — `test_density_counts_highest_weight_per_call` (a call with `LOOP` and `BLOCK` adds 1, not 2); `test_window_is_six_calls`; `test_blocked_calls_enter_the_window` (outcome `BLOCKED` → `peek().window` grew); `test_score_is_pure_with_sessions` (hypothesis: `score` twice → identical, `state_bytes` unchanged); `test_record_round_trips_dump_load`; `test_interleaved_sessions_separate_windows` (Review Focus 1); `test_polling_tool_loop_is_not_severe`.
- [ ] **Step 2: Run** `uv run pytest tests/core/test_flags.py tests/core/test_session_book.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement** as specified. Replay fills `flags` (comma-joined sorted names) and `density` from `res.trajectory`.
- [ ] **Step 4: Run** `uv run pytest -q -m "not slow and not live" && uv run python -m agentfw.bench.ci_regress` — Expected: PASS; goldens unchanged (flags never change a decision yet).
- [ ] **Step 5: Commit** — `feat(core): per-session flag window and density (F7 substrate)`

### Task 2.2: Provisional lifecycle and decision metrics

**Files:**
- Create: `agentfw/core/trajectory/provisional.py`, `agentfw/core/trajectory/decide.py`
- Modify: `agentfw/core/trajectory/session.py` (`SessionRecord.exploration`, `total_mass`, `unresolved_mass`, `off_track_raised`), `agentfw/core/engine.py` (call `decide`), `agentfw/core/config.py`, `agentfw/bench/session_metrics.py`
- Test: `tests/core/test_provisional.py`, `tests/bench/test_session_metrics.py`

**Interfaces:**
- Produces:
  - `ProvEntry(seq_no: int, ttl: int, mass: float, server: str, field: str | None, value_hash: str | None, state: Literal["provisional", "validated", "absorbed", "unresolved"])` (`value_hash` from novelty evidence `value_hash` when novelty fired).
  - `advance(rec: SessionRecord, e: ActionEnvelope, fused: float, theta_eff: float, per: Sequence[DetectorScore], cfg) -> set[Flag]` — pure on a *copy*; the engine applies it to the stored record in `update`. Order per call: (a) for each open entry, `validated` if any `ledger_refs[i].producer_seq == entry.seq_no`; else `absorbed` if `e.tool_id` shares the entry's server, the same `field` carries the same `value_hash`, and `fused < theta_eff`; (b) decrement `ttl` of still-open entries; at 0 → `unresolved`, add its mass to `unresolved_mass`, emit `UNRESOLVED`; (c) `total_mass += fused`; if `n_calls + 1 ≥ prov_min_calls` and `unresolved_mass / total_mass > prov_budget` and not `off_track_raised` → emit `OFF_TRACK`, set `off_track_raised` (cleared when the ratio falls back below budget).
  - `decide(...)` in `decide.py` (grows in later tasks): from `fused`, `theta`, `cfg.cal_delta`, blast radius / `θ_eff` (1.7), irreversibility → `(Decision, TrajectoryInfo)`. With `provisional=True`, a reversible call whose base decision is `BLOCK` becomes `ALLOW` and opens a `ProvEntry` (`ttl = prov_ttl`, `mass = fused`) with flag `PROVISIONAL`.
  - Config: `provisional: bool = False`, `prov_ttl: int = 3`, `prov_budget: float = 0.25`, `prov_min_calls: int = 8`.
  - `session_metrics.py`: `decision_flagged(r: ScoreRow) -> bool` (`decision in {"block", "require_approval"}`); `decision_recall_by_family(rows) -> dict[str, float]` (attack level); `decision_control_fpr(rows) -> float`; `decision_benign_fpr(rows) -> float`.

- [ ] **Step 1: Write failing tests**

```python
def test_hallucinated_read_resolves_unresolved_at_ttl():       # F6 acceptance, latency 3
    flags = per_call_flags(run_session(prov_engine(), [novel_read()] + [benign_read()] * 3))
    assert Flag.PROVISIONAL in flags[0]
    assert [Flag.UNRESOLVED in f for f in flags[1:]] == [False, False, True]

def test_consumed_artifact_validates(): ...                    # a later call's ledger_ref points at it
def test_vocabulary_shift_absorbs(): ...                       # same novel value, same server, benign score
def test_off_track_once_over_budget(): ...                     # 8+ calls, unresolved > 25 % of mass: exactly one OFF_TRACK
def test_one_call_abandoned_branch_costs_one_eighth():
    assert density_after_one_unresolved == pytest.approx(0.125 / 6)
def test_irreversible_calls_never_provisional(): ...
def test_switch_off_identical_decisions(): ...
def test_decision_metrics_on_handmade_rows(): ...
```

- [ ] **Step 2: Run** `uv run pytest tests/core/test_provisional.py tests/bench/test_session_metrics.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement.** The engine replaces `self.calibrator.decide(agent, fused)` with `decide(...)`; with every switch off it returns exactly `calibrator.decide`'s decision (a test asserts this over the CI dataset). Module docstring cites the paper's §4.5 lifecycle.
- [ ] **Step 4: Run** `uv run pytest -q -m "not slow and not live" && uv run python -m agentfw.bench.ci_regress` — Expected: PASS; goldens unchanged.
- [ ] **Step 5: Commit** — `feat(core): provisional flags for reversible calls (grace TTL, exploration budget) (F6)`

### Task 2.3: Age discount for transition surprise

**Files:**
- Modify: `agentfw/core/detectors/sequence.py`, `agentfw/core/config.py`
- Test: `tests/core/test_sequence_trans.py`

**Interfaces:**
- Produces: `EngineConfig.seq_age_discount: bool = False`, `EngineConfig.age_h: int = 3`; `ValidatedRecord` (`KIND = "validated"`, key `s:{agent}:validated`, `tools: set[str]`) — a tool joins when an *irreversible* call's `ledger_refs` name it as producer (needs `TrajectoryInfo.irreversible`; the engine passes `irreversible` to `SequenceDetector.update` through a new keyword `irreversible: bool = False`); `s_trans` is multiplied by `ω = 1.0 if tool in validated else min(1.0, c0[tool] / age_h)`; evidence gains `age_weight`.

- [ ] **Step 1: Write failing tests** — `test_first_use_of_a_new_tool_has_zero_transition_weight`; `test_ramp_over_three_calls` (ω = 0, 1/3, 2/3, 1); `test_consumed_by_irreversible_lifts_discount_early`; `test_discount_never_touches_prereq_or_flow` (F6 acceptance: `prereq` and `flow` parts identical with the switch on and off); `test_magnitude_and_novelty_unaffected`.
- [ ] **Step 2: Run** `uv run pytest tests/core/test_sequence_trans.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest tests/core -q && uv run python -m agentfw.bench.ci_regress` — Expected: PASS; goldens unchanged.
- [ ] **Step 5: Commit** — `feat(core): age discount for transition surprise on new tools (F6)`

### Task 2.4: Splice family, length-matched set, session label column

**Files:**
- Modify: `agentfw/synth/attacks.py` (`splice`, `AttackContext.donors`), `agentfw/synth/generate.py` (`ctx_factory` supplies donors), `agentfw/bench/replay.py` (`ScoreRow.session_label: str | None = None`, filled after replay from `session_labels`), `agentfw/bench/session_metrics.py`
- Test: `tests/synth/test_attacks_splice.py`, `tests/bench/test_session_metrics.py`

**Interfaces:**
- Produces: `AttackContext.donors: Sequence[tuple[str, list[RawCall]]] = ()` (sessions of *other* personas sharing ≥ 3 tools with this persona, earliest first, at most 50); `splice` (opt-in, T-10): pick a donor session and a 3-call window whose tools are all in this persona's `tools`; insert copies (fresh `envelope_id`s from the attack RNG, this session's ids, `seq_no` renumbered, timestamps evenly between the neighbours) at a uniform position; label the three `attack`/`splice`. `length_matched(rows, tol: int = 2, seed: int = 0) -> list[ScoreRow]` (each attacked session paired with an unused benign session of the same agent whose call count differs by ≤ `tol`; unpaired sessions dropped); `session_scores(rows) -> dict[str, float]` (max fused per session); `session_auroc(rows) -> float`.

- [ ] **Step 1: Write failing tests** — `test_splice_transplants_three_calls_from_another_agent`; `test_splice_is_opt_in`; `test_length_matched_pairs_within_tol` (every pair's lengths differ ≤ 2, no session reused); `test_session_auroc_on_handmade_rows`; `test_session_label_column_filled`.
- [ ] **Step 2: Run** — Expected: FAIL.
- [ ] **Step 3: Implement.** Draws only from the attack RNG (persona streams unchanged; the existing determinism test still passes).
- [ ] **Step 4: Run** `uv run pytest tests/synth tests/bench -q && uv run python -m agentfw.bench.ci_regress` — Expected: PASS.
- [ ] **Step 5: Commit** — `feat(synth,bench): splice family, length-matched set, session label column (F10)`

---

## Phase 3 — decisions (F4, F5, F7, F8; Tier 3)

### Task 3.1: Pre-execution gate with fresh-state recompute

**Files:**
- Create: `agentfw/core/trajectory/gate.py`
- Modify: `agentfw/core/trajectory/decide.py`, `agentfw/core/config.py`, `agentfw/core/envelope.py` (`TrajectoryInfo.gate`)
- Test: `tests/core/test_gate.py`, `tests/acceptance/test_gate_allow_side.py` (slow)

**Interfaces:**
- Consumes: ledger refs (1.1), `FlowRecord` + `usual_producer` (1.4), `irreversible` (1.7).
- Produces: `GateOutcome(action: Literal["pass", "block", "escalate"], prereq_missing: str | None, missing_path: str | None)`; `gate(e, types, seq: SequenceDetector, fused: float, theta_eff: float, cfg) -> GateOutcome`, called from `decide` only when `cfg.gate` and the call is irreversible. Steps: (1) everything is read from `e` and the store at call time — no cached snapshot; (2) for each ID-typed flat value with no `LedgerRef` and not in `session_ids_seen`: warm (`total ≥ flow_min_obs`) with `r_flow ≥ gate_flow_rate` → `block` with `prereq_missing = usual_producer(...)` and `missing_path`; cold → `escalate` (T-11); (3) `fused ≥ theta_eff` → `escalate`. With `gate=True`, a reversible call is capped at `REQUIRE_APPROVAL` (never `BLOCK` from the engine). A gate block sets flag `PREREQ_MISSING` and decision `BLOCK`. `regime` without schemas (`history_only` / `bare`) skips step 2 (G-d).
- Config: `gate: bool = False`, `gate_flow_rate: float = 0.90`.

- [ ] **Step 1: Write failing tests**

```python
def test_g_a_no_false_blocks_on_valid_irreversible_calls():   # F4 acceptance (slow)
    # 200 benign synth sessions (annotated catalog, id_flow_rate=1.0) each ending in a valid
    # irreversible call whose ids were produced in-session or seeded from task input
    blocks = [r for r in last_call_results if r.decision is Decision.BLOCK]
    assert blocks == []

def test_g_b_missing_producer_blocks():
    r = gate_engine().score(refund_with_fabricated_charge)
    assert r.decision is Decision.BLOCK and r.trajectory.gate.prereq_missing == "stripe.retrieve_charge"

def test_g_c_anomalous_distribution_escalates_not_blocks(): ...
def test_g_d_no_schemas_distribution_only(): ...               # regime history_only: never a prereq block
def test_reversible_never_blocked(): ...
def test_bloom_only_id_is_present():                          # Review Focus 4
def test_stale_snapshot_fails_g_a():                          # stale-state regression
    # StaleTracker(SessionTracker) returns the snapshot from before the previous observe();
    # running the G-a generator through it must produce >= 1 false block
    assert count_blocks(run_g_a(tracker_cls=StaleTracker)) >= 1
```

`StaleTracker` and `run_g_a(tracker_cls)` live in `tests/acceptance/test_gate_allow_side.py`.

- [ ] **Step 2: Run** `uv run pytest tests/core/test_gate.py -v && uv run pytest tests/acceptance/test_gate_allow_side.py -v -m slow` — Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** both again, then `uv run pytest -q -m "not slow and not live" && uv run python -m agentfw.bench.ci_regress` — Expected: PASS; goldens unchanged.
- [ ] **Step 5: Commit** — `feat(core): pre-execution gate for irreversible calls with allow-side suite (F4)`

### Task 3.2: Escalation

**Files:**
- Create: `agentfw/core/trajectory/escalate.py`
- Modify: `agentfw/core/trajectory/decide.py`, `agentfw/core/engine.py` (`Engine(..., judge: Judge | None = None)`), `agentfw/core/config.py`, `agentfw/core/envelope.py` (`TrajectoryInfo.escalated`, `fast_score`, `escalated_score`), `agentfw/bench/replay.py` (`ScoreRow.escalated: bool = False`)
- Test: `tests/core/test_escalate.py`

**Interfaces:**
- Produces: `class Judge(Protocol): def __call__(self, e: ActionEnvelope, reasons: Sequence[Reason]) -> float | None: ...` (`None` = abstain / timed out; the implementation owns its session history and its 1.5 s budget); `escalate(e, fast_fused, fast_decision, rescore: Callable[[], float], theta_eff, cfg, judge) -> tuple[Decision, float]`. Triggered by a gate `escalate` or a candidate `BLOCK` / `REQUIRE_APPROVAL` on an irreversible call. `rescore()` recomputes the fused score from the store without the engine cache. The judge is consulted only when the escalated decision is `REQUIRE_APPROVAL`: `≥ judge_block` → `BLOCK`, `≤ judge_allow` → `ALLOW`, else unchanged. A confident fast-path `BLOCK` is never relaxed.
- Config: `escalate: bool = False`, `judge_block: float = 0.8`, `judge_allow: float = 0.2`.

- [ ] **Step 1: Write failing tests** — hypothesis property `test_escalation_never_more_permissive_outside_band` (random fused, θ, judge outputs: escalated decision ≥ fast decision in the order ALLOW < REQUIRE_APPROVAL < BLOCK unless the fast decision was `REQUIRE_APPROVAL`); `test_judge_cannot_flip_confident_block`; `test_judge_none_means_unchanged`; `test_audit_fields_filled`; `test_escalation_rate_on_benign_ci_dataset_at_most_2pct` (slow; `escalated` share of benign rows with `gate`+`escalate` on); `test_escalated_p99_latency_under_50ms_without_judge` (slow).
- [ ] **Step 2: Run** `uv run pytest tests/core/test_escalate.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest -q -m "not live" && uv run python -m agentfw.bench.ci_regress` — Expected: PASS.
- [ ] **Step 5: Commit** — `feat(core): two-tier escalation with optional judge in the approval band (F5)`

### Task 3.3: Session verdicts

**Files:**
- Modify: `agentfw/core/trajectory/session.py` (verdict ladder, halt, release), `agentfw/core/engine.py` (`release_session`), `agentfw/core/config.py`, `agentfw/core/envelope.py` (`TrajectoryInfo.session_verdict`), `agentfw/bench/replay.py` (`ScoreRow.session_verdict: str | None = None`), `agentfw/bench/session_metrics.py`
- Test: `tests/core/test_session_verdict.py`, `tests/bench/test_session_metrics.py`

**Interfaces:**
- Produces: `SessionVerdict(StrEnum)`: `OK = "ok"`, `WARN = "warn_session"`, `HALT = "halt_session"`; ladder per call on the density including this call: `≥ theta_halt` → `HALT` (sticky: `halted = True`), `≥ theta_warn` → `WARN`, else `OK`; a halted session reports `HALT` on every later call. With `session_enforce=True`, a call in a session already halted *before* it gets decision `BLOCK` and a `Reason(detector="session", ...)`. `Engine.release_session(session_id) -> None` clears `halted`, `verdict` and the window. The fused session aggregate is reported in `TrajectoryInfo` (`session_mass`) and never thresholded.
  - Config: `theta_warn: float = 0.33`, `theta_halt: float = 0.6`, `session_enforce: bool = False`.
  - `session_metrics.py`: `HaltMetrics(recall_by_label: dict[str, float], false_halt_rate: float, median_calls_from_onset: float, precision_at_base_rate: float)`; `halt_metrics(rows, theta_halt: float, base_rate: float = 0.015) -> HaltMetrics` — recomputed offline from the `density` column (a session halts at its first call with `density ≥ theta_halt`; shadow replay makes this exact); onset = first attack-labelled call in the session; `halt_sensitivity(rows, thetas=np.arange(0.2, 0.81, 0.05)) -> list[tuple[float, float, float]]` (θ, false-halt rate, runaway recall); `first_flag_halts(rows) -> HaltMetrics` (halt at the first severe flag — the paper's negative baseline).

- [ ] **Step 1: Write failing tests** — `test_ladder_thresholds`; `test_halt_is_sticky_until_release`; `test_enforce_blocks_after_halt_only_when_switched_on`; `test_halted_session_never_evicted` and `test_evicted_idle_session_restarts_ok` (Review Focus 5); `test_aggregate_never_thresholded` (a session with high `session_mass` but no severe flags stays `OK`); `test_halt_metrics_on_handmade_rows`; `test_sensitivity_curve_is_monotone_non_increasing_in_false_halts`; `test_first_flag_baseline_halts_earlier_and_more`.
- [ ] **Step 2: Run** `uv run pytest tests/core/test_session_verdict.py tests/bench/test_session_metrics.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement.** Module docstring cites the paper (§1 S4, §6.3).
- [ ] **Step 4: Run** `uv run pytest -q -m "not slow and not live" && uv run python -m agentfw.bench.ci_regress` — Expected: PASS; goldens unchanged (`session_enforce` off).
- [ ] **Step 5: Commit** — `feat(core,bench): session verdict engine on severe-flag density, halt metrics (F7)`

### Task 3.4: Maturity and the authority ceiling

**Files:**
- Create: `agentfw/core/trajectory/authority.py`, `agentfw/core/render.py`
- Modify: `agentfw/core/trajectory/decide.py`, `agentfw/core/detectors/sequence.py` (evidence `n` on every part), `agentfw/core/fusion.py` (`Reason.maturity`, `Reason.regime`, `Reason.blast_radius`), `agentfw/core/envelope.py` (`Reason.maturity: str | None = None`, `Reason.regime: str | None = None`; `TrajectoryInfo.maturity`, `capped_from`, `notice`), `agentfw/core/config.py`, `agentfw/bench/replay.py` (`ScoreRow.maturity: str | None = None`, `ScoreRow.regime: str | None = None`)
- Test: `tests/core/test_authority.py`, `tests/core/test_render.py`

**Interfaces:**
- Produces: `Maturity = Literal["mature", "warm", "cold", "bare"]`; `evidence_n(s: DetectorScore) -> int | None` (first of evidence `n`, `n_total`, `n_hist`); `maturity_of(per, fused, regime, cfg) -> Maturity` — minimum `n` over the scores that contributed ≥ 0.3 to the fused score under fusion v1: the top non-vital score when it is ≥ 0.3, plus the second when it earned the bonus (`> fusion_bonus_gate`); no contributor (fused < 0.3) → `mature` (nothing to cap); a contributor without `n` counts as `cold`; `prior_used` caps at `warm`; `regime == "bare"` forces `bare`; `apply_ceiling(decision, maturity, gate: GateOutcome | None, regime) -> tuple[Decision, Decision | None, Literal["warn", "inform"] | None]` (T-14 table; a gate `block` is exempt when `regime` has schemas; returns new decision, the pre-cap decision if capped, and the notice); `render_reason(r: Reason, info: TrajectoryInfo) -> str`, e.g. `"blocked (mature: 1,140 refunds); amount at 99.97th percentile; no prior stripe.retrieve_charge (schemas: destructive)"`.
  - Config: `authority: bool = False`, `mat_warm_n: int = 30`, `mat_mature_n: int = 300`. With `authority=True`, maturity `cold` also adds flag `COLD_START`.

- [ ] **Step 1: Write failing tests** — `test_ceiling_table` (four rows); `test_prior_used_caps_at_warm`; `test_gate_block_exempt_with_schemas_only`; `test_no_block_from_cold_or_bare_in_audit` (F8 acceptance; slow: replay the CI dataset with every switch on, assert no row has `decision == "block"` with `maturity in {"cold", "bare"}` unless `flags` contains `PREREQ_MISSING` and `regime` contains `schemas`); `test_every_reason_carries_maturity_and_regime`; `test_render_reason_examples` (three golden strings: magnitude, prereq, novelty — none contains a raw id).
- [ ] **Step 2: Run** `uv run pytest tests/core/test_authority.py tests/core/test_render.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement.** `decide` order is final here: bands on `θ_eff` → gate (irreversible) → escalation → provisional (reversible) → ceiling → session overlay (`session_enforce`). Write that order into `decide.py`'s docstring.
- [ ] **Step 4: Run** `uv run pytest -q -m "not live" && uv run python -m agentfw.bench.ci_regress` — Expected: PASS; goldens unchanged.
- [ ] **Step 5: Commit** — `feat(core): verdict authority scaled by evidence maturity; reason rendering (F8)`

---

## Phase 4 — measurement and defaults

### Task 4.1: Trajectory benchmark, exit acceptance, defaults decision record

**Files:**
- Create: `agentfw/bench/trajectory.py`, `tests/acceptance/test_trajectory_exit.py`, `docs/decisions/010-trajectory-defaults.md`, `results/trajectory/` (report outputs)
- Modify: `agentfw/bench/registry.py` (`"engine-traj"`: every switch on, `br_mode="modulate"`, `enabled=KNOWN_DETECTORS`), `agentfw/bench/calibrated.json` (θ values), `tests/perf/test_latency.py` (an `engine-traj` case), `README.md` (one status paragraph)

**Interfaces:**
- Consumes: everything above.
- Produces: `python -m agentfw.bench.trajectory --traces DIR --out DIR [--seeds 0,1] [--workers N]` writing `trajectory_report.json` / `.md` with: a feature on/off table (one row per switch toggled alone on top of `engine`, plus `engine-traj`) with per-call metrics, decision metrics and control FPR; the attribution table with groups `magnitude, novelty, velocity, sequence, vital, gate (cfg_off={"gate": False}), provisional (cfg_off={"provisional": False})`; halt metrics, the sensitivity curve and the first-flag baseline; `splice` recall (separate section); session AUROC on all sessions and on the length-matched set; precision at base rate for every operating point; `select_on_calibration(rows) -> tuple[float, float]` — `theta_halt` = smallest value on the 0.05 grid in [0.2, 0.8] with benign false-halt ≤ 1% on calibration sessions, `theta_warn` = smallest grid value with benign warn rate ≤ 5%, capped below `theta_halt`; both written to `calibrated.json` with the normalizer hash. Every table carries its `PROTOCOL_LABELS` label; reviewer-protocol numbers are marked optimistic.

- [ ] **Step 1: Write the acceptance test** `tests/acceptance/test_trajectory_exit.py` (marked `slow`; criteria that fail get `pytest.mark.xfail(strict=True, reason=...)`, the Tier-1 pattern), one test per feature-spec acceptance not already pinned by a unit test: F2 benign vital false-flag ≤ 1% of sessions on the full bench; F5 escalation lowers decision control FPR versus `engine` without reducing any family's attack-level decision recall by more than 1 point; F6 decision control FPR on `benign_unusual` ≤ 3% with `provisional` and below `engine`'s; F7 halt-sensitivity false halts non-increasing in θ with no step > 10 points between neighbouring θ ("no cliff"), and the first-flag baseline's false-halt rate above `engine-traj`'s; F10 attribution exactness on the full bench. Run on the generator's default bench config plus `families` extended with `loop, spin, stall, splice`, `calibration_frac=0.15`, seeds 0 and 1.
- [ ] **Step 2: Run** `uv run pytest tests/acceptance/test_trajectory_exit.py -m slow -v` — Expected: some tests fail; mark exactly those `xfail(strict=True)` with the measured number in `reason`. Do not change thresholds or defaults to make one pass.
- [ ] **Step 3: Run the benchmark** — `uv run python -m agentfw.bench.trajectory --traces traces/bench --out results/trajectory --seeds 0,1 --workers 8`, under both `outcomes="allow"` (spec) and `"review"` (labelled optimistic). Commit the reports.
- [ ] **Step 4: Write `010-trajectory-defaults.md`.** Per switch: what it changed (with CIs), the attribution rows it carried, the latency cost, and a recommended default. List as **follow-ups, not done here:** re-selecting `z0`, `tau`, `prereq_rate` and fusion weights on the calibration partition (this changes Tier-1 defaults the user ruled on in decision 008), F9, and Tier-4 enforcement. Defaults flip only after the user decides; if they do, that change regenerates the goldens and `baseline_metrics.json` in its own commit.
- [ ] **Step 5: Verify** — `scripts/gate.sh && uv run python -m agentfw.bench.ci_regress && uv run pytest tests/perf -m slow -q` — Expected: PASS (the `engine-traj` latency case included).
- [ ] **Step 6: Commit** — `feat(bench): trajectory benchmark, exit acceptance, defaults decision record (F10)`

---

## Self-review (against `docs/feature-spec-trajectory-monitoring.md`)

**Coverage.** F1 → 1.1, 1.3, 1.4 (orphan marker folded into F6, T-16). F2 → 1.2, 1.5, 1.6, 1.8. F3 → 1.7, 1.8. F4 → 3.1. F5 → 3.2. F6 → 2.2, 2.3. F7 → 2.1, 3.3. F8 → 3.4. F9 → out of scope (T-01), recorded as a follow-up in 4.1. F10: three-way split 0.2; precision at base rate 0.3; response-content rule 0.1; recalibration discipline 0.3 + 1.2; attribution 0.4 + 4.1; runaway families 1.6; splice, length-matched set, session labels 2.4; session metrics 3.3; Tier-4 parts of F4/F5/F7 → out of scope (T-01). v3 diff list (feature spec §3): §2.2 → 3.4 `decide` order; §3 data model → 1.1, 1.2, 1.5, 1.7, 2.1–3.4; §4.6 → 1.6, 2.4; §5.3/5.4 → 0.2, 0.3, 3.3; §6.1 → 1.1, 1.5; §6.6(c) → 1.4; §6.8 → 3.4; §6.9 → 1.7; §7 → 2.1; §8 → 0.1, 3.1; §9 → phases. Risks (feature spec §4): extraction misses → T-11 + 1.4; polling false loops → 1.5, 2.1; long benign sessions → per-agent θ is a 010 follow-up, shadow default kept; annotation trust → 1.7 adversarial test.

**Type consistency.** `LedgerRef`, `RecentCall`, `TrajectoryInfo`, `Flag`, `SessionRecord`, `ProvEntry`, `GateOutcome`, `Judge`, `Maturity`, `Regime` are each defined once (Interfaces) and referenced by the same names afterwards. `ALL_DETECTORS` keeps its v1 meaning; `KNOWN_DETECTORS` adds `vital`.

**Defaults.** Every task's verification step re-runs `ci_regress` with goldens unchanged; the only planned golden change is a user-approved default flip after Task 4.1.
