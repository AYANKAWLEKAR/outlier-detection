from __future__ import annotations

from agentfw.bench.attribution import (
    DEFAULT_GROUPS,
    REDUNDANT,
    attribution_table,
    detected_attacks,
)
from agentfw.bench.ci_regress import ci_dataset
from agentfw.bench.registry import make_detector
from agentfw.bench.replay import ScoreRow, SplitSpec, replay
from agentfw.core.config import EngineConfig
from agentfw.core.engine import ALL_DETECTORS
from agentfw.core.envelope import Label, RawCall


def replay_rows(raws: list[RawCall], labels: dict[str, Label], name: str) -> list[ScoreRow]:
    return replay(raws, labels, make_detector(name), SplitSpec(), outcomes="review")


def test_default_groups_cover_all_detectors() -> None:
    assert {g.name for g in DEFAULT_GROUPS} == set(ALL_DETECTORS)


def test_attribution_is_leave_one_group_out() -> None:
    raws, labels = ci_dataset()
    table = attribution_table(
        raws,
        labels,
        groups=DEFAULT_GROUPS,
        cfg=EngineConfig(),
        outcomes="review",
        split=SplitSpec(),
    )
    full = detected_attacks(replay_rows(raws, labels, "engine"))
    assert full
    union: set[str] = set()
    for g in DEFAULT_GROUPS:  # T-13: exactness by construction
        without = detected_attacks(replay_rows(raws, labels, f"engine-no-{g.name}"))
        carried = full - without
        union |= carried
        assert sum(v.get(g.name, 0) for v in table.values()) == len(carried)
    redundant = sum(v.get(REDUNDANT, 0) for v in table.values())
    assert redundant + len(union) == len(full)  # T-20
    assert sum(sum(v.values()) for v in table.values()) >= len(full)


def test_decision_column_values() -> None:
    raws, labels = ci_dataset()
    rows = replay_rows(raws, labels, "engine")
    assert {r.decision for r in rows} <= {None, "allow", "block", "require_approval"}
    assert any(r.decision == "allow" for r in rows)


def test_attributed_attacks_have_test_split_rows() -> None:
    raws, labels = ci_dataset()
    rows = replay_rows(raws, labels, "engine")
    test_ids = {r.attack_id for r in rows if r.label == "attack" and r.split == "test"}
    assert detected_attacks(rows) <= test_ids
    assert detected_attacks(rows, splits=frozenset({"train", "test"})) >= detected_attacks(rows)
