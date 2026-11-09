from __future__ import annotations

import dataclasses
from typing import Literal

import pytest

from agentfw.core.config import EngineConfig
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


def test_engine_config_defaults_match_global_constraints() -> None:
    assert dataclasses.asdict(EngineConfig()) == GLOBAL_CONSTRAINT_VALUES


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
