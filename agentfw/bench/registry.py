"""Detector registry: name -> factory."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from agentfw.bench.baselines.isolation_forest import IsolationForestBaseline
from agentfw.bench.baselines.llm_judge import LlmJudge
from agentfw.bench.baselines.random_ import RandomDetector
from agentfw.bench.baselines.static_threshold import StaticThreshold
from agentfw.bench.dataset import load_schema_hints
from agentfw.bench.protocol import Detector
from agentfw.core.config import EngineConfig
from agentfw.core.engine import ALL_DETECTORS, DETECTOR_ORDER, KNOWN_DETECTORS, Engine

__all__ = ["DETECTORS", "ENGINE_VARIANTS", "make_bench_detector", "make_detector"]


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
    # v1 + the vital signs (F2). Vital scores are never fused (T-07), so this variant's fused
    # scores equal "engine"'s; no engine-no-vital / engine-only-vital ablations.
    "engine-vital": _engine(KNOWN_DETECTORS),
}


#: Registry names built by ``Engine`` (they take ``hints=``).
ENGINE_VARIANTS = frozenset(n for n in DETECTORS if n == "engine" or n.startswith("engine-"))


def make_detector(name: str, **kw: Any) -> Detector:
    try:
        factory = DETECTORS[name]
    except KeyError:
        raise KeyError(f"unknown detector {name!r}; available: {sorted(DETECTORS)}") from None
    return factory(**kw)


def make_bench_detector(name: str, traces_dir: str | Path, **kw: Any) -> Detector:
    """``make_detector`` for a replay over ``traces_dir``: an engine variant also gets the
    dataset's captured schemas (``schemas.jsonl``, F3) as hints unless ``hints`` is given. With
    the default config (``br_mode="off"``) they change no fused score or decision."""
    if name in ENGINE_VARIANTS and "hints" not in kw:
        hints = load_schema_hints(traces_dir)
        if hints:
            kw["hints"] = hints
    return make_detector(name, **kw)
