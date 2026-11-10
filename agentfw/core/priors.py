"""Pooled cross-agent priors for cold keys (spec §6.10).

``PooledPriors`` keeps one magnitude record per ``(tool, path)`` (key ``p:m:{tool}:{path}``) and
one novelty record per ``(tool, path)`` (key ``p:n:{tool}:{path}``, D-013), fed by every agent's
allowed calls. It satisfies ``PriorSource``; the detectors decide *when* a key is cold and apply
the IQR inflation / novelty discount (D-012).

Records reuse ``MagRecord.push_y`` and ``NoveltyRecord.push`` so the P²/EWMA and LRU/singleton
bookkeeping lives in one place.

``priors_mode`` gating (``EngineConfig.priors_mode``):
  * ``"all"``: refs are served for every field.
  * ``"money_count_only"``: ``magnitude_ref`` only serves ``direction == "up"`` fields; novelty
    has no direction, so ``novelty_ref`` returns ``None``.
  * ``"off"``: refs return ``None`` and observations are not stored.
"""

from __future__ import annotations

from agentfw.core.config import EngineConfig
from agentfw.core.detectors.magnitude import MagRecord, direction_for
from agentfw.core.detectors.novelty import NoveltyRecord
from agentfw.core.state import StateStore, key_prior_mag, key_prior_nov

__all__ = ["PooledPriors"]


class PooledPriors:
    def __init__(self, store: StateStore, cfg: EngineConfig) -> None:
        self._store = store
        self._cfg = cfg

    def observe_numeric(self, tool: str, path: str, y: float) -> None:
        """Fold one numeric observation; ``y`` is the ALREADY transformed ``signed_log1p(x)``."""
        if self._cfg.priors_mode == "off":
            return
        key = key_prior_mag(tool, path)
        rec = self._store.get(key)
        if not isinstance(rec, MagRecord):
            rec = MagRecord.new(self._cfg, direction_for(path, {}))
            self._store.put(key, rec)
        rec.push_y(y)

    def observe_categorical(self, tool: str, path: str, v: str) -> None:
        """Fold one categorical/ID value (its string form)."""
        if self._cfg.priors_mode == "off":
            return
        key = key_prior_nov(tool, path)
        rec = self._store.get(key)
        if not isinstance(rec, NoveltyRecord):
            rec = NoveltyRecord()
            self._store.put(key, rec)
        rec.push(v, self._cfg)

    def magnitude_ref(self, tool: str, path: str, direction: str) -> MagRecord | None:
        mode = self._cfg.priors_mode
        if mode == "off" or (mode == "money_count_only" and direction != "up"):
            return None
        rec = self._store.get(key_prior_mag(tool, path))
        if isinstance(rec, MagRecord) and rec.n >= self._cfg.mag_cold_n:
            return rec
        return None

    def novelty_ref(self, tool: str, path: str) -> NoveltyRecord | None:
        if self._cfg.priors_mode != "all":
            return None
        rec = self._store.get(key_prior_nov(tool, path))
        return rec if isinstance(rec, NoveltyRecord) else None
