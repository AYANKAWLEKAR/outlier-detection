"""Decision 008's evidence scripts replay the spec engine by default (fix round 1 of Task U1).

The engine defaults changed by user decision (2026-11-10) and by Amendment A1 (Task 1.21), so the
scripts build their config as ``EngineConfig().replace(**{**SPEC_SWITCHES, **params})``: no
PARAMS still means the spec engine, and 008's candidates keep the pre-amendment scoring.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from agentfw.bench.sweeps import PRE_A1_SWITCHES, SPEC_SWITCHES
from agentfw.core.config import EngineConfig

EVIDENCE = Path(__file__).resolve().parents[2] / "docs/decisions/evidence/008-root-causes"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"_evidence_{name}", EVIDENCE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # importing must not run the script
    return mod


@pytest.mark.parametrize("name", ["replay_variants", "shares"])
def test_evidence_script_base_is_the_spec_engine(name: str) -> None:
    mod = _load(name)
    spec_cfg = EngineConfig().replace(**SPEC_SWITCHES)
    assert mod.spec_config({}) == spec_cfg
    assert spec_cfg != EngineConfig()  # the defaults moved; the evidence base did not
    rec = {"vel_count_attempts": True, "seq_prereq_graded": True, "novelty_full_n": 50}
    # 008's recommended candidate = the engine before Amendment A1 (A1.9) ...
    assert mod.spec_config(rec) == EngineConfig().replace(**PRE_A1_SWITCHES)
    # ... and with the amended scoring on top, today's default.
    a1 = {"vel_count_stat": "poisson", "mag_score": "tail", "mag_bool_numeric": True}
    assert mod.spec_config({**rec, **a1}) == EngineConfig()
    assert mod.spec_config({"z0": 2.5}) == spec_cfg.replace(z0=2.5)
