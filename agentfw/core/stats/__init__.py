"""Streaming statistics used by the detectors (pure Python, no scipy)."""

from agentfw.core.stats.ewma import Ewma
from agentfw.core.stats.p2 import P2Quantiles, p2_tail

__all__ = ["Ewma", "P2Quantiles", "p2_tail"]
