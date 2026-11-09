"""Baseline detectors the benchmark compares against."""

from __future__ import annotations

from agentfw.bench.baselines.random_ import RandomDetector
from agentfw.bench.baselines.static_threshold import StaticThreshold

__all__ = ["RandomDetector", "StaticThreshold"]
