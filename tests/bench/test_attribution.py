from __future__ import annotations

from pathlib import Path

from agentfw.bench.attribution import (
    DEFAULT_GROUPS,
    REDUNDANT,
    attribution_table,
    detected_attacks,
)
from agentfw.bench.ci_regress import CI_DATASET, ci_dataset
from agentfw.bench.dataset import load_dir
from agentfw.bench.registry import make_detector
from agentfw.bench.replay import ScoreRow, SplitSpec, replay
from agentfw.core.config import EngineConfig
from agentfw.core.engine import ALL_DETECTORS
from agentfw.core.envelope import Label, RawCall
from agentfw.synth.attacks import RUNAWAY_FAMILIES
from agentfw.synth.generate import generate


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


def test_runaway_attacks_are_never_attributed(tmp_path: Path) -> None:
    """T-10: runaway rows are in no per-call metric, so attribution (which matches test-split
    recall, T-20) neither detects nor tabulates them, even when the engine flags them."""
    families = [*CI_DATASET.families, *sorted(RUNAWAY_FAMILIES)]
    # attack rate 0.5: enough runaway rows that the engine flags some on the test split
    generate(CI_DATASET.model_copy(update={"families": families, "attack_rate": 0.5}), tmp_path)
    raws, labels, _ = load_dir(tmp_path)
    rows = replay_rows(raws, labels, "engine")
    flagged_runaway = {
        r.attack_id
        for r in rows
        if r.attack_family in RUNAWAY_FAMILIES
        and r.split == "test"
        and r.decision in ("block", "require_approval")
    }
    assert flagged_runaway  # the engine does flag some: the exclusion is not vacuous
    detected = detected_attacks(rows)
    assert not detected & flagged_runaway
    runaway_ids = {
        lab.attack_id for lab in labels.values() if lab.attack_family in RUNAWAY_FAMILIES
    }
    assert not detected_attacks(rows, splits=frozenset({"train", "test"})) & runaway_ids
    table = attribution_table(
        raws,
        labels,
        groups=DEFAULT_GROUPS,
        cfg=EngineConfig(),
        outcomes="review",
        split=SplitSpec(),
    )
    assert not RUNAWAY_FAMILIES & set(table)
    assert "unknown" not in table


def test_tool_hints_pickle_for_parallel_attribution() -> None:
    import pickle

    from agentfw.core.hints import ToolAnnotations, ToolHint

    h = ToolHint(kind="write", field_hints={"amount": "money"}, annotations=ToolAnnotations(destructive=True), has_schema=True)
    back = pickle.loads(pickle.dumps(h))
    assert back == h
    assert dict(back.field_hints) == {"amount": "money"}
    try:
        back.field_hints["x"] = "count"  # type: ignore[index]
    except TypeError:
        pass
    else:
        raise AssertionError("field_hints must stay read-only after unpickling")
