"""Session-level labels and metrics (Task 1.6, F2/F10)."""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any

import orjson
import pytest

from agentfw.bench.ci_regress import CI_DATASET
from agentfw.bench.dataset import load_dir
from agentfw.bench.metrics import compute_all
from agentfw.bench.registry import make_detector
from agentfw.bench.replay import ScoreRow, SplitSpec, replay
from agentfw.bench.session_metrics import session_labels, vital_false_flag_rate
from agentfw.synth.attacks import RUNAWAY_FAMILIES
from agentfw.synth.generate import generate

_ids = itertools.count()


def _vital_json(loop: float = 0.0, spin: float = 0.0, stall: float = 0.0) -> str:
    items = [
        {"detector": "magnitude", "field": "amount", "score": 1.0, "evidence": {}},
        *(
            {"detector": "vital", "field": f, "score": s, "evidence": {}}
            for f, s in (("__loop__", loop), ("__spin__", spin), ("__stall__", stall))
        ),
    ]
    return orjson.dumps(items).decode()


def _row(session: str, **kw: Any) -> ScoreRow:
    i = next(_ids)
    base: dict[str, Any] = {
        "envelope_id": f"e{i:06d}",
        "session_id": session,
        "agent_id": "A",
        "tool_id": "zendesk.get_ticket",
        "ts_unix_ms": 1_000 + i,
        "split": "test",
        "fused_score": 0.0,
        "per_detector_json": _vital_json(),
        "label": "benign",
        "attack_family": None,
        "attack_id": None,
        "injection_level": None,
        "strength": None,
        "step": None,
        "control_type": None,
        "is_warm": True,
        "n_key_obs": 100,
        "latency_us": 10.0,
        "threshold": None,
    }
    base.update(kw)
    return ScoreRow(**base)


def _atk(session: str, family: str, **kw: Any) -> ScoreRow:
    return _row(session, label="attack", attack_family=family, attack_id=f"atk-{session}", **kw)


def test_session_labels_priority() -> None:
    rows = [
        _row("plain"),
        _row("ctl", control_type="benign_unusual"),
        _row("run"),
        _atk("run", "loop"),
        _atk("run", "stall"),
        _row("mixed"),
        _atk("mixed", "spin"),
        _atk("mixed", "magnitude_spike"),
        _atk("hit", "novel_target"),
        _row("hit"),
    ]
    assert session_labels(rows) == {
        "plain": "benign",
        "ctl": "benign",  # a control is benign
        "run": "runaway",
        "mixed": "attacked",  # attacked beats runaway
        "hit": "attacked",
    }
    assert session_labels([]) == {}


def test_vital_false_flag_rate_counts_benign_sessions_with_a_full_vital_score() -> None:
    rows = [
        _row("b1", per_detector_json=_vital_json(loop=1.0)),
        _row("b1"),
        _row("b2", per_detector_json=_vital_json(loop=0.7, stall=0.5)),  # below 1: not a flag
        _row("b3", per_detector_json=_vital_json(spin=1.0)),
        _row("b4", per_detector_json="[]"),
        _row("b5", control_type="benign_unusual", per_detector_json=_vital_json(stall=1.0)),
        # not benign sessions: outside the denominator whatever they score
        _atk("r1", "loop", per_detector_json=_vital_json(loop=1.0)),
        _row("a1", per_detector_json=_vital_json(loop=1.0)),
        _atk("a1", "magnitude_spike"),
    ]
    assert vital_false_flag_rate(rows) == pytest.approx(3 / 5)
    assert math.isnan(vital_false_flag_rate([_atk("x", "loop")]))
    assert math.isnan(vital_false_flag_rate([]))


def _replayed(tmp_path: Path, families: list[str], attack_rate: float = 0.15) -> list[ScoreRow]:
    cfg = CI_DATASET.model_copy(update={"families": families, "attack_rate": attack_rate})
    generate(cfg, tmp_path)
    raws, labels, _ = load_dir(tmp_path)
    return replay(raws, labels, make_detector("engine-vital"), SplitSpec())


def test_runaway_rows_excluded_from_per_call_metrics(tmp_path: Path) -> None:
    families = [*CI_DATASET.families, *sorted(RUNAWAY_FAMILIES)]
    rows = _replayed(tmp_path, families, attack_rate=0.5)  # every runaway family present
    runaway = [r for r in rows if r.attack_family in RUNAWAY_FAMILIES]
    others = [r for r in rows if r.attack_family not in RUNAWAY_FAMILIES]
    assert {r.attack_family for r in runaway} == RUNAWAY_FAMILIES
    assert any(r.label == "attack" for r in others)
    with_runaway = compute_all(rows, warm_only=False, n_boot=0)
    without = compute_all(others, warm_only=False, n_boot=0)
    assert repr(with_runaway) == repr(without)  # repr: NaN fields compare equal
    assert not RUNAWAY_FAMILIES & set(with_runaway.per_family_attack)
    labels = session_labels(rows)
    assert {labels[r.session_id] for r in runaway} <= {"runaway", "attacked"}
    assert "runaway" in labels.values()


@pytest.fixture(scope="module")
def benign_rows(tmp_path_factory: pytest.TempPathFactory) -> list[ScoreRow]:
    return _replayed(tmp_path_factory.mktemp("benign"), [])


def test_benign_dataset_sessions_are_all_benign(benign_rows: list[ScoreRow]) -> None:
    assert set(session_labels(benign_rows).values()) == {"benign"}
    assert len(session_labels(benign_rows)) == 200


# Measured at Task 1.6 (T-30): 12 of 200 sessions (6.00%) on this dataset, all LOOP, all
# sales-ops-a; full benign SW1 3.64% (1,538 / 42,289). The bar stays 1%.
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "measured 6.00% (12/200 sessions) on CI_DATASET with families=[] "
        "(full benign SW1: 3.64%, 1,538/42,289): loop on task `repeat` iterations (same call "
        "re-issued within w_loop=8); F2 semantics vs bar pending user decision"
    ),
)
def test_benign_vital_false_flag_rate_at_most_1pct(benign_rows: list[ScoreRow]) -> None:
    """F2 acceptance: loop/spin/stall flag at most 1% of benign sessions."""
    assert vital_false_flag_rate(benign_rows) <= 0.01
