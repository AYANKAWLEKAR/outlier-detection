from __future__ import annotations

import dataclasses
from typing import Literal, get_args

import pytest

from agentfw.core.config import EngineConfig, ScoreModel, VelCountStat
from agentfw.core.hints import ToolHint

# Verbatim from global-constraints.md ("All tuned constants live in ... EngineConfig").
GLOBAL_CONSTRAINT_VALUES: dict[str, object] = {
    "k_prior_tools": 4,
    "max_session_ids": 2000,
    "freetext_min_len": 40,
    "type_min_obs": 10,
    "type_sticky_window": 100,
    "reservoir_size": 64,
    "mag_probs": (0.05, 0.25, 0.5, 0.75, 0.95),
    "ewma_alpha": 0.02,
    "mag_cold_n": 30,
    "z0": 3.0,
    "tau": 0.75,
    "mag_eps": 0.05,
    "prior_iqr_inflate": 1.5,
    "prior_novelty_discount": 0.67,
    "priors_mode": "all",
    "vel_short_min": 10,
    "vel_long_min": 1440,
    "vel_eps": 0.25,
    "vel_cold_n": 30,
    "novelty_lru": 10_000,
    "novelty_full_n": 200,
    "cms_width": 2048,
    "cms_depth": 4,
    "trans_min_ctx": 20,
    "trans_smoothing": 0.5,
    "prereq_min_n": 50,
    "prereq_rate": 0.95,
    "flow_min_obs": 20,
    "fusion_bonus": 0.15,
    "fusion_bonus_gate": 0.5,
    "reason_min": 0.3,
    "cal_alpha": 0.005,
    "cal_theta0": 0.6,
    "cal_blend_n": 500,
    "cal_delta": 0.1,
}

# Task 1.14 sweep switches (Open Decisions 2 and 5, velocity sum signal): defaults keep the
# spec's behaviour.
SWEEP_SWITCH_DEFAULTS: dict[str, object] = {
    "mag_direction_default": "both",
    "mag_resp_signal": True,
    "vel_sum_signal": True,
}


# Task 1.15 switches (docs/decisions/008-tier1-gate.md). The censoring switches stay opt-in and
# keep the spec's behaviour; the other three are defaults by user decision (below).
GATE_SWITCH_DEFAULTS: dict[str, object] = {
    "cal_censor": "none",
    "cal_censor_min_n": 0,
}

# User decision 2026-11-10 (decision 008, "User decision"): three engine switches on by default.
# Spec values: vel_count_attempts=False, seq_prereq_graded=False (§6.6(b)), novelty_full_n=200
# (§6.5); tests of spec behaviour pin those values explicitly.
USER_DECISION_DEFAULTS: dict[str, object] = {
    "vel_count_attempts": True,
    "seq_prereq_graded": True,
    "novelty_full_n": 50,
}


# Defaults changed by an open-decision record (Task 1.14), overriding the spec value above.
DECISION_OVERRIDES: dict[str, object] = {
    "prereq_rate": 0.90,  # docs/decisions/003-prereq-threshold.md
}

# Amendment A1 (spec A1.9): the amended scoring is the default (Task 1.21). Pre-amendment values:
# vel_count_stat="logz" (§6.4), mag_score="z" (§6.3), mag_bool_numeric=False; tests of the old
# behaviour pin them explicitly (``agentfw.bench.sweeps.PRE_A1_SWITCHES``). ``tail_prior``
# equals ``cal_alpha`` by definition (D-023).
A1_DEFAULTS: dict[str, object] = {
    "vel_count_stat": "poisson",
    "mag_score": "tail",
    "tail_prior": 0.005,
    "mag_bool_numeric": True,
}


def test_engine_config_defaults_match_global_constraints() -> None:
    expected = (
        GLOBAL_CONSTRAINT_VALUES
        | SWEEP_SWITCH_DEFAULTS
        | GATE_SWITCH_DEFAULTS
        | DECISION_OVERRIDES
        | USER_DECISION_DEFAULTS
        | A1_DEFAULTS
    )
    assert dataclasses.asdict(EngineConfig()) == expected


def test_tail_prior_equals_cal_alpha_by_default() -> None:
    assert EngineConfig().tail_prior == EngineConfig().cal_alpha == 0.005


def test_vel_count_stat_values() -> None:
    assert get_args(VelCountStat) == ("logz", "poisson")
    assert EngineConfig().replace(vel_count_stat="poisson").vel_count_stat == "poisson"


def test_score_model_values() -> None:
    assert get_args(ScoreModel) == ("z", "tail")
    cfg = EngineConfig()
    assert cfg.replace(mag_score="tail").mag_score == "tail"
    assert cfg.tail_prior == cfg.cal_alpha  # equal by definition (A1.2), a field of its own


def test_user_decision_engine_defaults() -> None:
    """Decision 008, user decision 2026-11-10: attempts count, graded prereq, novelty ramp 50."""
    cfg = EngineConfig()
    assert cfg.vel_count_attempts is True
    assert cfg.seq_prereq_graded is True
    assert cfg.novelty_full_n == 50


def test_engine_config_frozen_slots_and_replace() -> None:
    cfg = EngineConfig()
    assert not hasattr(cfg, "__dict__")
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.z0 = 4.0  # type: ignore[misc]
    other = cfg.replace(z0=4.0, priors_mode="off")
    assert other.z0 == 4.0 and other.priors_mode == "off"
    assert cfg.z0 == 3.0 and cfg == EngineConfig()
    with pytest.raises(TypeError):
        cfg.replace(not_a_field=1)


def test_tool_hint_defaults_and_immutability() -> None:
    h = ToolHint()
    assert h.kind is None and dict(h.field_hints) == {}
    src: dict[str, Literal["money", "count"]] = {"amount": "money"}
    h2 = ToolHint(kind="write", field_hints=src)
    src["qty"] = "count"
    assert dict(h2.field_hints) == {"amount": "money"}
    with pytest.raises(dataclasses.FrozenInstanceError):
        h2.kind = "read"  # type: ignore[misc]
    with pytest.raises(TypeError):
        h2.field_hints["x"] = "count"  # type: ignore[index]
