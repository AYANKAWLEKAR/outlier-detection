"""Runaway families (Task 1.6, F2/F10): ``loop``, ``spin``, ``stall``; and the R10 ``fp`` fix."""

from __future__ import annotations

import copy
import statistics
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np
import orjson
import pytest

from agentfw.bench.ci_regress import CI_DATASET
from agentfw.bench.dataset import load_dir
from agentfw.bench.registry import make_detector
from agentfw.bench.replay import ScoreRow, SplitSpec, replay
from agentfw.core.config import EngineConfig
from agentfw.core.envelope import RawCall
from agentfw.core.fingerprint import resp_fingerprint
from agentfw.core.patterns import is_id_like
from agentfw.synth.attacks import (
    ATTACKS,
    RUN_GAP_MAX_MS,
    RUN_GAP_MIN_MS,
    RUN_MAX,
    RUN_MIN,
    RUNAWAY_ATTACKS,
    RUNAWAY_FAMILIES,
    AttackContext,
    AttackResult,
    applicable,
    eligible,
)
from agentfw.synth.generate import GenerateConfig, generate, sw1_config
from agentfw.synth.sampler import sample_persona

from .conftest import World

START = 1_767_225_600_000
Ctx = Callable[..., AttackContext]


def _sessions(world: World, agent: str, days: int = 6, seed: int = 0) -> list[list[RawCall]]:
    return sample_persona(
        world.persona(agent), world.catalog, world.tasks, days=days, global_seed=seed,
        start_ms=START,
    )  # fmt: skip


@pytest.fixture
def make_ctx(world: World) -> Ctx:
    def make(agent: str) -> AttackContext:
        return AttackContext(
            world.catalog, world.persona(agent), world.pools(agent), f"atk-{agent}-00000", [], 0
        )

    return make


def _session_with(world: World, agent: str, tool: str, min_len: int = 3) -> list[RawCall]:
    """The first sampled session of ``agent`` with a ``tool`` call and at least ``min_len`` calls,
    whose ``tool`` call is not its last."""
    for s in _sessions(world, agent):
        idx = [i for i, c in enumerate(s) if c.tool_id == tool]
        if len(s) >= min_len and idx and idx[0] < len(s) - 1:
            return s
    raise AssertionError(f"no {tool} session for {agent}")


def _seq_ok(session: list[RawCall]) -> None:
    assert [c.seq_no for c in session] == list(range(len(session)))
    ts = [c.ts_unix_ms for c in session]
    assert ts == sorted(ts)
    assert len({c.envelope_id for c in session}) == len(session)


def _run(s: list[RawCall], r: AttackResult) -> tuple[int, list[RawCall]]:
    """(index of the call the run follows, the inserted calls), checking contiguity."""
    labelled = {lab.envelope_id for lab in r.labels}
    pos = [i for i, c in enumerate(r.session) if c.envelope_id in labelled]
    assert pos == list(range(pos[0], pos[0] + len(pos)))  # one contiguous run
    return pos[0] - 1, [r.session[i] for i in pos]


def _check_common(family: str, s: list[RawCall], r: AttackResult) -> tuple[int, list[RawCall]]:
    _seq_ok(r.session)
    i, run = _run(s, r)
    n = len(run)
    assert RUN_MIN <= n <= RUN_MAX and len(r.session) == len(s) + n
    assert all(lab.label == "attack" and lab.attack_family == family for lab in r.labels)
    assert {lab.strength for lab in r.labels} == {float(n)}
    assert {lab.attack_id for lab in r.labels} == {"atk-" + s[0].agent_id + "-00000"}
    assert all(c.task_ids == () for c in run)
    assert all(c.session_id == s[0].session_id for c in run)
    # 0.5-3 s gaps from the call the run follows; the original calls keep their order and gaps.
    ts = [r.session[i].ts_unix_ms, *(c.ts_unix_ms for c in run)]
    assert all(RUN_GAP_MIN_MS <= b - a <= RUN_GAP_MAX_MS for a, b in zip(ts, ts[1:], strict=False))
    kept = [c for c in r.session if c.envelope_id not in {x.envelope_id for x in run}]
    assert [c.envelope_id for c in kept] == [c.envelope_id for c in s]
    assert [c.model_copy(update={"seq_no": 0, "ts_unix_ms": 0}) for c in kept] == [
        c.model_copy(update={"seq_no": 0, "ts_unix_ms": 0}) for c in s
    ]
    gaps_after = [b.ts_unix_ms - a.ts_unix_ms for a, b in zip(kept, kept[1:], strict=False)]
    gaps_before = [b.ts_unix_ms - a.ts_unix_ms for a, b in zip(s, s[1:], strict=False)]
    shift = run[-1].ts_unix_ms - r.session[i].ts_unix_ms
    expect = [g + (shift if k == i else 0) for k, g in enumerate(gaps_before)]
    assert gaps_after == expect
    return i, run


# --- registry and opt-in --------------------------------------------------------------------------


def test_runaway_families_are_opt_in() -> None:
    assert frozenset({"loop", "spin", "stall"}) == RUNAWAY_FAMILIES == frozenset(RUNAWAY_ATTACKS)
    assert not RUNAWAY_FAMILIES & set(GenerateConfig.model_fields["families"].default)
    assert not RUNAWAY_FAMILIES & set(ATTACKS)  # sorted(ATTACKS) is SW1's family list (T-28)
    assert not RUNAWAY_FAMILIES & set(sw1_config().families)
    assert not RUNAWAY_FAMILIES & set(CI_DATASET.families)


@pytest.mark.parametrize("family", sorted(RUNAWAY_FAMILIES))
def test_every_runaway_family_has_three_eligible_personas(world: World, family: str) -> None:
    n = [p.agent_id for p in world.personas.values() if eligible(family, p, world.catalog)]
    assert len(n) >= 3, (family, n)


def test_runaway_attacks_are_pure_and_match_applicable(world: World, make_ctx: Ctx) -> None:
    for agent in ("sales-ops-a", "support-refunds-a", "exec-assistant-a"):
        for s in _sessions(world, agent)[:15]:
            for name, fn in RUNAWAY_ATTACKS.items():
                before = copy.deepcopy(s)
                got = fn(s, np.random.default_rng(0), make_ctx(agent))
                assert s == before, name
                assert applicable(name, s, make_ctx(agent)) == (got is not None), (name, agent)


def test_writes_only_session_is_not_applicable(world: World, make_ctx: Ctx) -> None:
    s = _sessions(world, "support-refunds-a")[0]
    writes = [c for c in s if world.catalog[c.tool_id].kind == "write"]
    assert writes
    for name in RUNAWAY_FAMILIES:
        assert not applicable(name, writes, make_ctx("support-refunds-a"))


def test_repeats_are_log_uniform_on_5_60(world: World, make_ctx: Ctx) -> None:
    s = _session_with(world, "sales-ops-a", "salesforce.run_soql")
    rs = []
    for seed in range(300):
        r = RUNAWAY_ATTACKS["loop"](s, np.random.default_rng(seed), make_ctx("sales-ops-a"))
        assert r is not None
        rs.append(len(r.labels))
    assert min(rs) >= RUN_MIN and max(rs) <= RUN_MAX
    assert 13 <= statistics.median(rs) <= 22  # log-uniform median sqrt(5 * 60) = 17.3
    assert sum(r <= 10 for r in rs) > sum(r >= 50 for r in rs)  # mass skews low


# --- the three families ---------------------------------------------------------------------------


def test_loop_inserts_identical_copies_right_after_a_read(world: World, make_ctx: Ctx) -> None:
    s = _session_with(world, "sales-ops-a", "salesforce.run_soql")
    for seed in range(20):
        r = RUNAWAY_ATTACKS["loop"](s, np.random.default_rng(seed), make_ctx("sales-ops-a"))
        assert r is not None
        i, run = _check_common("loop", s, r)
        orig = r.session[i]
        assert world.catalog[orig.tool_id].kind == "read"
        for c in run:
            assert (c.tool_id, c.args, c.response_meta) == (
                orig.tool_id,
                orig.args,
                orig.response_meta,
            )
        again = RUNAWAY_ATTACKS["loop"](s, np.random.default_rng(seed), make_ctx("sales-ops-a"))
        assert again == r  # deterministic


def test_spin_varies_pool_ids_and_keeps_the_response(world: World, make_ctx: Ctx) -> None:
    agent = "support-refunds-a"
    s = _session_with(world, agent, "zendesk.get_ticket")
    pool = set(world.pools(agent)["tickets"].ids)
    for seed in range(20):
        r = RUNAWAY_ATTACKS["spin"](s, np.random.default_rng(seed), make_ctx(agent))
        assert r is not None
        i, run = _check_common("spin", s, r)
        orig = r.session[i]
        assert orig.tool_id == "zendesk.get_ticket"
        assert orig.response_meta is not None and orig.response_meta.fp is not None
        for c in run:
            assert c.tool_id == orig.tool_id and c.response_meta == orig.response_meta
            assert c.args["ticket_id"] in pool
            assert {k: v for k, v in c.args.items() if k != "ticket_id"} == {
                k: v for k, v in orig.args.items() if k != "ticket_id"
            }
        assert len({c.args["ticket_id"] for c in run}) > len(run) // 2  # fresh, not one id


def test_stall_rereads_only_what_the_session_has_seen(world: World, make_ctx: Ctx) -> None:
    checked = varied = 0
    for agent in sorted(world.personas):
        persona = world.persona(agent)
        reads = {t for t in persona.tools if world.catalog[t].kind == "read"}
        for s in _sessions(world, agent)[:20]:
            r = RUNAWAY_ATTACKS["stall"](s, np.random.default_rng(len(s)), make_ctx(agent))
            if r is None:
                continue
            checked += 1
            i, run = _check_common("stall", s, r)
            earlier = r.session[i]
            assert earlier.tool_id in reads and earlier.response_meta is not None
            seen: set[str] = set()
            for c in r.session[: i + 1]:
                seen |= {v for v in _strings(c.args) if is_id_like(v)}
                seen |= set(c.response_meta.ids if c.response_meta else ())
            for c in run:
                assert c.tool_id in reads
                assert c.response_meta == earlier.response_meta  # ids subset of seen, same fp
                assert set(c.response_meta.ids) <= seen
                assert {v for v in _strings(c.args) if is_id_like(v)} <= seen  # no new artifact
                # the copied response keeps the shape of the tool it lands on
                has_count = world.catalog[c.tool_id].response.item_count is not None
                assert (c.response_meta.item_count is not None) == has_count, (agent, c.tool_id)
            varied += len({(c.tool_id, orjson.dumps(c.args)) for c in run}) > 1
    assert checked >= 100
    assert varied >= checked // 2  # mostly a mix of re-reads, not one call repeated


def _strings(v: object) -> list[str]:
    if isinstance(v, str):
        return [v]
    if isinstance(v, list):
        return [x for e in v for x in _strings(e)]
    if isinstance(v, dict):
        return [x for e in v.values() for x in _strings(e)]
    return []


# --- R10: response_meta.fp stays consistent where an attack rewrites the response ---------------


def _digest(world: World) -> list[RawCall]:
    for s in _sessions(world, "outreach-a"):
        if [c.tool_id for c in s][:2] == ["slack.read_channel", "slack.post_message"]:
            return s
    raise AssertionError("no digest session")


def test_exfil_volume_drops_the_stale_fp_of_the_inflated_read(world: World, make_ctx: Ctx) -> None:
    s = _digest(world)
    assert s[0].response_meta is not None and s[0].response_meta.fp is not None
    r = ATTACKS["exfil_volume"](s, np.random.default_rng(0), make_ctx("outreach-a"))
    assert r is not None
    read = next(c for c in r.session if c.envelope_id == r.labels[0].envelope_id)
    assert read.response_meta is not None and read.response_meta.fp is None


def test_exfil_volume_fingerprints_the_appended_write(world: World, make_ctx: Ctx) -> None:
    s = [c for c in _digest(world) if c.tool_id == "slack.read_channel"]
    r = ATTACKS["exfil_volume"](s, np.random.default_rng(4), make_ctx("outreach-a"))
    assert r is not None and len(r.session) == len(s) + 1
    meta = r.session[-1].response_meta
    assert meta is not None and meta.fp is not None
    blob = orjson.dumps(list(meta.ids))
    assert meta.bytes == 64 + len(blob)
    assert meta.fp == resp_fingerprint(blob.decode(), None, False)


# --- end to end: generate, replay engine-vital ----------------------------------------------------


def _vital(r: ScoreRow, field: str) -> float:
    return next(
        float(it["score"])
        for it in orjson.loads(r.per_detector_json)
        if it["detector"] == "vital" and it["field"] == field
    )


def _replayed(tmp_path: Path, family: str) -> list[ScoreRow]:
    """The CI dataset's size and personas with every post-warm-up session attacked by
    ``family``, replayed through ``engine-vital``."""
    cfg = CI_DATASET.model_copy(update={"families": [family], "attack_rate": 1.0})
    generate(cfg, tmp_path)
    raws, labels, _ = load_dir(tmp_path)
    return replay(raws, labels, make_detector("engine-vital"), SplitSpec())


def _attack_runs(rows: list[ScoreRow]) -> list[tuple[list[ScoreRow], int, int]]:
    """Per attack: (its session's rows in order, index of the original call, run length)."""
    by_sess: dict[str, list[ScoreRow]] = defaultdict(list)
    for r in rows:  # replay order is (ts, session, seq) order
        by_sess[r.session_id].append(r)
    out = []
    for atk in sorted({r.attack_id for r in rows if r.attack_id is not None}):
        sess = by_sess[next(r.session_id for r in rows if r.attack_id == atk)]
        pos = [k for k, r in enumerate(sess) if r.attack_id == atk]
        assert pos == list(range(pos[0], pos[0] + len(pos))) and pos[0] >= 1
        out.append((sess, pos[0] - 1, len(pos)))
    return out


def _first(sess: list[ScoreRow], start: int, field: str) -> int | None:
    return next((k for k in range(start, len(sess)) if _vital(sess[k], field) == 1.0), None)


def test_loop_injection_detected_within_one_event(tmp_path: Path) -> None:
    """F2 acceptance: LOOP (1.0) fires at the 2nd duplicate, so at most 2 calls after the read."""
    runs = _attack_runs(_replayed(tmp_path, "loop"))
    assert len(runs) >= 100
    for sess, orig, _ in runs:
        hit = _first(sess, orig, "__loop__")
        assert hit is not None and hit - orig <= 2, (sess[orig].session_id, orig, hit)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="T-58 (final review I1): spin reads completed calls only (a proxy scores a call before "
    "its response), so it fires one call later: measured 1/91 runs within w_spin = 4 events of "
    "the onset, 90 at 5 (91/91 within 4 when spin read the current call's response, T-27). "
    "Bar kept; the lag-one bound is pinned below.",
)
def test_spin_injection_detected_within_w_spin_events(tmp_path: Path) -> None:
    """F2 acceptance: SPIN within ``vital_w_spin`` events of the onset."""
    w = EngineConfig().vital_w_spin
    runs = _attack_runs(_replayed(tmp_path, "spin"))
    assert len(runs) >= 80
    for sess, orig, _ in runs:
        hit = _first(sess, orig, "__spin__")
        assert hit is not None and hit - orig <= w, (sess[orig].session_id, orig, hit)


def test_spin_injection_detected_within_w_spin_plus_one_events(tmp_path: Path) -> None:
    """T-58: SPIN over completed calls detects every spin run, at most one call after the
    ``vital_w_spin`` bound (91/91 on this set)."""
    w = EngineConfig().vital_w_spin
    runs = _attack_runs(_replayed(tmp_path, "spin"))
    assert len(runs) >= 80
    for sess, orig, _ in runs:
        hit = _first(sess, orig, "__spin__")
        assert hit is not None and hit - orig <= w + 1, (sess[orig].session_id, orig, hit)


def _learned_read(world: World, tool: str) -> bool:
    """Without tool hints the engine learns a tool as a write when it takes an id arg and its
    responses carry no item count (``SequenceDetector.is_write``): ``salesforce.get_record`` is a
    catalog read that the engine sees as a write, and STALL never counts a window holding one."""
    spec = world.catalog[tool]
    has_id = any(a.type in ("id", "list_of_id") for a in spec.args.values())
    return spec.response.item_count is not None or not has_id


def test_stall_injection_detected_once_the_window_is_inside_the_run(
    world: World, tmp_path: Path
) -> None:
    """STALL scores the ``vital_w_stall`` calls before the current one (T-27): every call whose
    window lies inside the run, on tools the engine sees as reads, scores 1.0."""
    w = EngineConfig().vital_w_stall
    runs = _attack_runs(_replayed(tmp_path, "stall"))
    assert len(runs) >= 100
    checked = 0
    for sess, orig, n in runs:
        for k in range(orig + 1 + w, min(orig + n + 2, len(sess))):
            window = sess[k - w : k]
            if all(_learned_read(world, r.tool_id) for r in window):
                checked += 1
                assert _vital(sess[k], "__stall__") == 1.0, (sess[orig].session_id, orig, k)
    assert checked >= 200
