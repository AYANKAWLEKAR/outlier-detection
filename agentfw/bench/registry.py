"""Detector registry: name -> factory."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agentfw.bench.baselines.isolation_forest import IsolationForestBaseline
from agentfw.bench.baselines.llm_judge import LlmJudge
from agentfw.bench.baselines.random_ import RandomDetector
from agentfw.bench.baselines.static_threshold import StaticThreshold
from agentfw.bench.protocol import Detector
from agentfw.core.config import EngineConfig
from agentfw.core.engine import ALL_DETECTORS, DETECTOR_ORDER, Engine

__all__ = ["DETECTORS", "make_detector"]


def _engine(
    enabled: frozenset[str] = ALL_DETECTORS, *, priors: bool = True
) -> Callable[..., Engine]:
    """Engine factory for one ablation; ``priors=False`` forces ``priors_mode="off"``."""

    def make(cfg: EngineConfig | None = None, **kw: Any) -> Engine:
        cfg = EngineConfig() if cfg is None else cfg
        if not priors:
            cfg = cfg.replace(priors_mode="off")
        return Engine(cfg, enabled=enabled, **kw)

    return make


DETECTORS: dict[str, Callable[..., Detector]] = {
    "random": RandomDetector,
    "static_threshold": StaticThreshold,
    "isolation_forest": IsolationForestBaseline,
    "llm_judge": LlmJudge,
    "engine": _engine(),
    **{f"engine-no-{d}": _engine(ALL_DETECTORS - {d}) for d in DETECTOR_ORDER},
    **{f"engine-only-{d}": _engine(frozenset({d})) for d in DETECTOR_ORDER},
    "engine-no-priors": _engine(priors=False),
}


def make_detector(name: str, **kw: Any) -> Detector:
    try:
        factory = DETECTORS[name]
    except KeyError:
        raise KeyError(f"unknown detector {name!r}; available: {sorted(DETECTORS)}") from None
    return factory(**kw)
