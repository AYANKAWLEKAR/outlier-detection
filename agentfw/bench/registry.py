"""Detector registry: name -> factory."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agentfw.bench.baselines.random_ import RandomDetector
from agentfw.bench.baselines.static_threshold import StaticThreshold
from agentfw.bench.protocol import Detector

__all__ = ["DETECTORS", "make_detector"]

DETECTORS: dict[str, Callable[..., Detector]] = {
    "random": RandomDetector,
    "static_threshold": StaticThreshold,
}


def make_detector(name: str, **kw: Any) -> Detector:
    try:
        factory = DETECTORS[name]
    except KeyError:
        raise KeyError(f"unknown detector {name!r}; available: {sorted(DETECTORS)}") from None
    return factory(**kw)
