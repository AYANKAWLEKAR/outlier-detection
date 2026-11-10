from __future__ import annotations

import math
from pathlib import Path

import pytest

from agentfw.bench.protocol import BatchFit
from agentfw.bench.replay import (
    LeakError,
    SplitSpec,
    assign_split,
    canonical_hash,
    read_score_table,
    replay,
    write_score_table,
)
from agentfw.core.envelope import (
    ActionEnvelope,
    DetectorResult,
    DetectorScore,
    Label,
    Outcome,
    RawCall,
)
from tests.factories import make_raw


class Spy:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.seen_flat: list[bool] = []

    def reset(self) -> None: ...

    def score(self, e: ActionEnvelope) -> DetectorResult:
        self.calls.append(f"score:{e.envelope_id}")
        self.seen_flat.append(bool(e.flat_args))
        return DetectorResult(0.0, (), (), 0.0)

    def update(self, e: ActionEnvelope, outcome: Outcome) -> None:
        assert outcome is Outcome.ALLOWED
        self.calls.append(f"update:{e.envelope_id}")

    def state_bytes(self) -> int:
        return 0


@pytest.fixture
def raws3() -> list[RawCall]:
    return [
        make_raw(session_id="s1", seq_no=0, ts_unix_ms=1_000),
        make_raw(session_id="s1", seq_no=1, ts_unix_ms=2_000),
        make_raw(session_id="s2", seq_no=0, ts_unix_ms=1_500),
    ]


def test_score_strictly_before_update_in_ts_order(raws3: list[RawCall]) -> None:
    spy = Spy()
    replay(list(reversed(raws3)), {}, spy, SplitSpec())
    ids = [
        r.envelope_id for r in sorted(raws3, key=lambda r: (r.ts_unix_ms, r.session_id, r.seq_no))
    ]
    assert spy.calls == [x for i in ids for x in (f"score:{i}", f"update:{i}")]
    assert all(spy.seen_flat)


def test_tie_order_deterministic() -> None:
    raws = [
        make_raw(session_id="sB", seq_no=0, ts_unix_ms=5),
        make_raw(session_id="sA", seq_no=1, ts_unix_ms=5),
        make_raw(session_id="sA", seq_no=0, ts_unix_ms=5),
    ]
    rows = replay(raws, {}, Spy(), SplitSpec())
    assert [r.envelope_id for r in rows] == [
        raws[2].envelope_id,
        raws[1].envelope_id,
        raws[0].envelope_id,
    ]


def test_is_warm_flips_after_30_prior_obs() -> None:
    raws = [make_raw(seq_no=i, ts_unix_ms=1_000 + i) for i in range(32)]
    rows = replay(raws, {}, Spy(), SplitSpec())
    assert [r.n_key_obs for r in rows][:3] == [0, 1, 2]
    assert rows[29].n_key_obs == 29 and not rows[29].is_warm
    assert rows[30].n_key_obs == 30 and rows[30].is_warm


def test_labels_and_missing_labels() -> None:
    a = make_raw(seq_no=0, ts_unix_ms=1)
    b = make_raw(seq_no=1, ts_unix_ms=2)
    labels = {
        a.envelope_id: Label(
            envelope_id=a.envelope_id,
            label="attack",
            attack_family="x",
            attack_id="atk1",
            injection_level="action",
            strength=2.0,
            step=1,
        )
    }
    rows = replay([a, b], labels, Spy(), SplitSpec())
    assert rows[0].label == "attack" and rows[0].attack_id == "atk1" and rows[0].strength == 2.0
    assert rows[1].label == "benign" and rows[1].attack_family is None and rows[1].step is None


class Fitter(Spy):
    def __init__(self) -> None:
        super().__init__()
        self.fit_ids: list[str] = []
        self.fit_before_score = False

    def fit(self, train: object) -> None:
        self.fit_before_score = not self.calls
        self.fit_ids = [e.envelope_id for e in train]  # type: ignore[attr-defined]


def test_batchfit_receives_train_only() -> None:
    raws = [make_raw(seq_no=i, ts_unix_ms=1_000 * i) for i in range(10)]
    spec = SplitSpec(kind="time", train_frac=0.7)
    split = assign_split(raws, spec)
    f = Fitter()
    assert isinstance(f, BatchFit)
    rows = replay(raws, {}, f, spec)
    assert f.fit_before_score
    assert set(f.fit_ids) == {i for i, s in split.items() if s == "train"}
    assert 0 < len(f.fit_ids) < 10
    assert {r.split for r in rows} == {"train", "test"}


def test_time_split_cut() -> None:
    raws = [make_raw(seq_no=i, ts_unix_ms=i * 10) for i in range(11)]  # 0..100
    split = assign_split(raws, SplitSpec(kind="time", train_frac=0.7))
    train = [r for r in raws if split[r.envelope_id] == "train"]
    assert max(r.ts_unix_ms for r in train) <= 70
    assert len(train) == 8


def test_agent_split_deterministic_by_agent() -> None:
    raws = [make_raw(agent_id=f"a{i}", session_id=f"s{i}", ts_unix_ms=i) for i in range(10)] + [
        make_raw(agent_id="a0", session_id="s0", seq_no=1, ts_unix_ms=99)
    ]
    spec = SplitSpec(kind="agent", seed=3)
    s1, s2 = assign_split(raws, spec), assign_split(raws, spec)
    assert s1 == s2
    by_agent: dict[str, set[str]] = {}
    for r in raws:
        by_agent.setdefault(r.agent_id, set()).add(s1[r.envelope_id])
    assert all(len(v) == 1 for v in by_agent.values())
    assert sum(next(iter(v)) == "train" for v in by_agent.values()) == 7


class Leaky(Spy):
    def __init__(self) -> None:
        super().__init__()
        self.n = 0

    def score(self, e: ActionEnvelope) -> DetectorResult:
        self.n += 1
        return super().score(e)

    def state_bytes(self) -> int:
        return self.n


def test_strict_mode_catches_leaky_score(raws3: list[RawCall]) -> None:
    with pytest.raises(LeakError):
        replay(raws3, {}, Leaky(), SplitSpec(), strict=True)
    replay(raws3, {}, Leaky(), SplitSpec(), strict=False)


def test_sample_ids_updates_all_scores_subset(raws3: list[RawCall]) -> None:
    spy = Spy()
    keep = {raws3[0].envelope_id}
    rows = replay(raws3, {}, spy, SplitSpec(), sample_ids=keep)
    assert len([c for c in spy.calls if c.startswith("update")]) == 3
    by_id = {r.envelope_id: r for r in rows}
    assert not math.isnan(by_id[raws3[0].envelope_id].fused_score)
    assert math.isnan(by_id[raws3[1].envelope_id].fused_score)


class Rich(Spy):
    def score(self, e: ActionEnvelope) -> DetectorResult:
        ds = DetectorScore("mag", "amount", 0.123456789123, {"k": 1})
        return DetectorResult(0.5, (ds,), (), 12.0, threshold=0.4)


def test_score_table_parquet_roundtrip_and_hash(raws3: list[RawCall], tmp_path: Path) -> None:
    rows = replay(raws3, {}, Rich(), SplitSpec())
    rows_nan = replay(raws3, {}, Rich(), SplitSpec(), sample_ids=set())
    p = tmp_path / "scores.parquet"
    write_score_table(rows, p)
    back = read_score_table(p)
    assert back == rows
    assert canonical_hash(back) == canonical_hash(rows)
    assert rows[0].threshold == 0.4 and rows[0].latency_us >= 0
    assert '"detector":"mag"' in rows[0].per_detector_json
    h = canonical_hash(rows)
    assert len(h) == 64 and h != canonical_hash(rows_nan)
    write_score_table(rows_nan, p)
    assert all(math.isnan(r.fused_score) for r in read_score_table(p))


def test_load_dir_roundtrip_and_replay(tmp_path: Path) -> None:
    from agentfw.bench.dataset import load_dir
    from agentfw.synth.generate import GenerateConfig, generate
    from agentfw.synth.load import CATALOG_DIR, PERSONAS_DIR, TASKS_DIR, TIER0_PERSONA_IDS

    cfg = GenerateConfig(
        seed=1,
        personas_dir=PERSONAS_DIR,
        persona_ids=list(TIER0_PERSONA_IDS),
        tasks_dir=TASKS_DIR,
        catalog_dir=CATALOG_DIR,
        days=1,
        max_sessions=6,
    )
    man = generate(cfg, tmp_path)
    raws, labels, manifest = load_dir(tmp_path)
    assert manifest == man
    assert len(raws) > 0
    assert set(labels) == {r.envelope_id for r in raws}
    rows = replay(raws, labels, Spy(), SplitSpec())
    assert len(rows) == len(raws)
