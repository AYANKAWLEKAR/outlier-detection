"""Per-call flags (F7 substrate, Task 2.1): ``derive_flags``, ``SEVERE`` and ``flag_weight``."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from agentfw.core.config import EngineConfig
from agentfw.core.envelope import Decision, DetectorScore
from agentfw.core.hints import ToolAnnotations, ToolHint
from agentfw.core.trajectory.flags import SEVERE, Flag, derive_flags, flag_weight

CFG = EngineConfig()
POLLING = ToolHint(annotations=ToolAnnotations(read_only=True, idempotent=True))


def seq(score: float, **ev: object) -> DetectorScore:
    base: dict[str, object] = {"fired": None, "missing_id_path": None, "r_flow": 0.0, "n": 0}
    return DetectorScore("sequence", None, score, base | ev)


def vital(field: str, score: float) -> DetectorScore:
    return DetectorScore("vital", field, score, {})


def ds(detector: str, score: float, field: str | None = None) -> DetectorScore:
    ev: Mapping[str, object] = {}
    return DetectorScore(detector, field, score, ev)


def flags(
    *per: DetectorScore,
    decision: Decision = Decision.ALLOW,
    irreversible: bool = False,
    hint: ToolHint | None = None,
) -> set[Flag]:
    return derive_flags(per, decision, irreversible, hint, CFG)


def test_severe_is_the_first_eight() -> None:
    assert list(Flag)[:8] == sorted(SEVERE, key=list(Flag).index)
    assert {
        Flag.BLOCK,
        Flag.PREREQ_MISSING,
        Flag.ID_FLOW_VIOLATION,
        Flag.LOOP,
        Flag.SPIN,
        Flag.STALL,
        Flag.OFF_TRACK,
        Flag.VELOCITY_BURST,
    } == SEVERE
    assert len(Flag) == 12
    assert all(f.value == f.name for f in Flag)  # stored and reported by name


def test_flag_weight() -> None:
    cfg = CFG.replace(prov_unresolved_weight=0.25)
    assert all(flag_weight(f, cfg) == 1.0 for f in SEVERE)
    assert flag_weight(Flag.UNRESOLVED, cfg) == 0.25
    assert flag_weight(Flag.UNRESOLVED, CFG) == 0.125
    for f in (Flag.PROVISIONAL, Flag.NOVEL_READ, Flag.COLD_START):
        assert flag_weight(f, cfg) == 0.0


def test_no_scores_no_flags() -> None:
    assert flags() == set()
    assert flags(seq(0.0), ds("velocity", 0.1), ds("novelty", 0.2, "x")) == set()


def test_block_from_the_decision_only() -> None:
    assert flags(decision=Decision.BLOCK) == {Flag.BLOCK}
    assert flags(decision=Decision.REQUIRE_APPROVAL) == set()


def test_prereq_missing_needs_fired_prereq_and_flag_min() -> None:
    assert flags(seq(0.9, fired="prereq")) == {Flag.PREREQ_MISSING}
    assert flags(seq(0.5, fired="prereq")) == {Flag.PREREQ_MISSING}  # >= flag_min
    assert flags(seq(0.49, fired="prereq")) == set()
    assert flags(seq(0.9, fired="trans")) == set()


def test_id_flow_violation_only_on_irreversible() -> None:
    s = seq(0.8, fired="flow", missing_id_path="charge", r_flow=0.8)
    assert flags(s, irreversible=True) == {Flag.ID_FLOW_VIOLATION}
    assert flags(s, irreversible=False) == set()
    low = seq(0.4, fired="flow", missing_id_path="charge", r_flow=0.4)
    assert flags(low, irreversible=True) == set()
    # r_flow without a missing path (nothing missing) is not a violation
    assert flags(seq(0.0, r_flow=0.9), irreversible=True) == set()
    # the flow evidence counts even when another part fired higher
    both = seq(1.0, fired="trans", missing_id_path="charge", r_flow=0.6)
    assert flags(both, irreversible=True) == {Flag.ID_FLOW_VIOLATION}


def test_vital_flags_need_a_full_score() -> None:
    assert flags(vital("__loop__", 1.0)) == {Flag.LOOP}
    assert flags(vital("__loop__", 0.7)) == set()
    assert flags(vital("__spin__", 1.0)) == {Flag.SPIN}
    assert flags(vital("__stall__", 1.0)) == {Flag.STALL}
    assert flags(vital("__stall__", 0.5)) == set()


def test_loop_exempt_on_read_only_idempotent_tools() -> None:
    assert flags(vital("__loop__", 1.0), hint=POLLING) == set()
    only_ro = ToolHint(annotations=ToolAnnotations(read_only=True))
    assert flags(vital("__loop__", 1.0), hint=only_ro) == {Flag.LOOP}
    assert flags(vital("__loop__", 1.0), hint=ToolHint(kind="read")) == {Flag.LOOP}
    # the exemption is LOOP's only: spin and stall still flag on a polling tool
    assert flags(vital("__spin__", 1.0), hint=POLLING) == {Flag.SPIN}


def test_velocity_burst() -> None:
    assert flags(ds("velocity", 0.5, "__count__")) == {Flag.VELOCITY_BURST}
    assert flags(ds("velocity", 0.49)) == set()


def test_novel_read_only_when_reversible() -> None:
    per = (ds("novelty", 0.1, "a"), ds("novelty", 0.7, "b"))
    assert flags(*per) == {Flag.NOVEL_READ}
    assert flags(*per, irreversible=True) == set()


@pytest.mark.parametrize("flag_min", [0.3, 0.8])
def test_flag_min_is_configurable(flag_min: float) -> None:
    cfg = CFG.replace(flag_min=flag_min)
    got = derive_flags([ds("velocity", 0.6)], Decision.ALLOW, False, None, cfg)
    assert (Flag.VELOCITY_BURST in got) == (flag_min <= 0.6)


def test_several_flags_at_once() -> None:
    per = (vital("__loop__", 1.0), ds("velocity", 0.9), seq(1.0, fired="prereq"))
    assert flags(*per, decision=Decision.BLOCK) == {
        Flag.BLOCK,
        Flag.LOOP,
        Flag.VELOCITY_BURST,
        Flag.PREREQ_MISSING,
    }
