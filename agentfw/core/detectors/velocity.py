"""Velocity detector: sliding-window call counts and money sums per (agent, tool) (spec §6.4).

Keys: ``v:{agent}:{tool}`` counts every call; ``v:{agent}:{tool}:{path}`` additionally sums a
money field, for flat paths whose hint is ``money``, or (no hint) whose last segment has
``amount``/``total`` at its start or after an ``_`` (``amount``, ``total_amount``,
``items[].amount``; not ``account``, ``discount``, ``subtotal``); such names always have
``direction_for == "up"``. A ``count`` hint makes a path not money.
A money field contributes the sum of its finite values (``try_number``; NaN/±inf/non-numbers
skipped); a call with no finite value does not touch that money key. With
``vel_sum_signal=False`` (Task 1.14 sweep) no money key is scored or kept: count signal only.

Clock: envelope time only, ``minute = ts_unix_ms // 60_000``. A timestamp earlier than the
record's ``last_minute`` (out of order) is treated as ``last_minute``: it never rewinds the clock.
Advancing from ``last_minute`` to ``minute`` samples each completed window total
(``log1p(count)`` into the count key's ``hist_count``, ``signed_log1p(sum)`` into a money key's
``hist_sum``; each record keeps only the history it scores) and rolls the
ring, one step per elapsed minute, but only while the window is non-empty (D-010: history is
over *active* minutes). After at most ``vel_short_min`` samples the window is empty and the
rest of the gap is skipped, so any idle gap costs O(``vel_short_min``). The window total is
kept running (minus each evicted bucket) rather than re-summed every step.

Score (one ``DetectorScore`` per envelope): ``y = log1p(window_count + 1)`` (this call
included), ``z = max((y - q50) / max(IQR / 1.349, vel_eps), 0)`` against ``hist_count``,
``s = squash(z, z0, tau)``; likewise for each money key with ``y = signed_log1p(window_sum +
this call's sum)`` against ``hist_sum``. The highest wins (ties keep the count): ``field`` is
``None`` for the count signal and the money path for a sum. A signal whose history has fewer
than ``vel_cold_n`` samples scores 0. ``score`` performs the advance virtually on a copy of the
ring and of the history (only when samples are pending); it never mutates state.

Evidence: ``signal`` (``count``/``sum``), ``window_count`` (calls in the window, this one
included), ``baseline_median`` (raw units, ``None`` without history), ``n_hist``, ``z`` (``None``
when cold), ``cold``; a sum adds ``window_sum``.

Refused attempts (``vel_count_attempts``, decision 008; default on by user decision 2026-11-10,
off = spec): the engine reports each blocked / denied call through ``observe_refused``; the count
key keeps its minute in ``refused`` (only while it is inside the short window, and only for a key
that already exists). ``score`` adds
the refused attempts of the current window to ``window_count``; nothing else (the ring, the
history, money sums) ever sees them, so the baseline never moves toward a refused call.

The 24 h window ``long`` is kept sparse (D-010): ``{minute: (count, sum)}`` for active minutes
only, pruned to the last ``vel_long_min`` minutes (emptied when ``vel_long_min <= 0``). It is
maintained for state/reporting (Open Decision 4) and is not scored in v1, matching §6.4. Sums
are clamped to finite floats so records stay JSON-finite.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar, Self

from agentfw.core.config import EngineConfig
from agentfw.core.detectors.base import (
    IQR_TO_SIGMA,
    clamp_finite,
    signed_expm1,
    signed_log1p,
    squash,
)
from agentfw.core.envelope import ActionEnvelope, DetectorScore
from agentfw.core.hints import ToolHint
from agentfw.core.patterns import try_number
from agentfw.core.schema_infer import TypeView
from agentfw.core.state import StateStore, key_vel, register_record
from agentfw.core.stats.p2 import P2Quantiles

__all__ = ["NAME", "VelocityDetector", "VelocityRecord"]

NAME = "velocity"
MS_PER_MIN = 60_000

Bucket = tuple[int, float]  # (count, sum)

_EMPTY: Bucket = (0, 0.0)
_HIST_PROBS = (0.25, 0.5, 0.75)
_MONEY_NAME = re.compile(r"(?i)(^|_)(amount|total)")


def _roll(ring: list[Bucket], last: int, to: int, with_sum: bool) -> tuple[list[Bucket], Bucket]:
    """Advance ``ring`` (in place) from minute ``last`` to ``to``.

    Returns the sampled window totals (one per completed minute while the window is non-empty,
    so at most ``len(ring)``) and the window total after the advance. The total is kept as a
    running count/sum, minus each evicted bucket; the sum is recomputed only when it becomes
    non-finite (overflowing buckets). ``with_sum=False`` (count key) skips sums: totals are 0.0.
    """
    w = len(ring)
    cnt = sum(c for c, _ in ring)
    tot = sum(s for _, s in ring) if with_sum else 0.0
    samples: list[Bucket] = []
    m = last
    while m < to and cnt > 0:  # an empty window stays empty: the rest of the gap is skipped
        samples.append((cnt, clamp_finite(tot)))
        m += 1
        i = m % w
        c, s = ring[i]
        if c:
            ring[i] = _EMPTY
            cnt -= c
            if with_sum:
                tot = tot - s if cnt else 0.0
                if not math.isfinite(tot):
                    tot = sum(s for _, s in ring)
    return samples, (cnt, clamp_finite(tot))


@register_record
@dataclass(slots=True)
class VelocityRecord:
    """Exactly one history is kept: ``hist_count`` on the count key, ``hist_sum`` on a money
    key (whose bucket counts still gate sampling to active minutes). Count-key sums are always
    0.0 and are not serialized."""

    KIND: ClassVar[str] = "vel"
    short: list[Bucket]  # ring indexed by minute % len(short)
    long: dict[int, Bucket]  # sparse, ascending minutes, pruned to vel_long_min
    last_minute: int
    hist_count: P2Quantiles | None  # count key only
    hist_sum: P2Quantiles | None  # money keys only
    refused: list[int] = field(default_factory=list)  # minutes of refused attempts (opt-in)

    @classmethod
    def new(cls, cfg: EngineConfig, minute: int, money: bool) -> Self:
        return cls(
            short=[_EMPTY] * cfg.vel_short_min,
            long={},
            last_minute=minute,
            hist_count=None if money else P2Quantiles(_HIST_PROBS),
            hist_sum=P2Quantiles(_HIST_PROBS) if money else None,
        )

    @property
    def money(self) -> bool:
        return self.hist_sum is not None

    def to_dict(self) -> dict[str, object]:
        mins = list(self.long)
        d: dict[str, object] = {
            "short_count": [c for c, _ in self.short],
            "last_minute": self.last_minute,
            # Parallel arrays with minutes as offsets back from last_minute (compact JSON).
            "long_ago": [self.last_minute - m for m in mins],
            "long_count": [self.long[m][0] for m in mins],
            "hist_count": None if self.hist_count is None else self.hist_count.to_dict(),
            "hist_sum": None if self.hist_sum is None else self.hist_sum.to_dict(),
        }
        if self.money:
            d["short_sum"] = [s for _, s in self.short]
            d["long_sum"] = [self.long[m][1] for m in mins]
        if self.refused:
            d["refused"] = list(self.refused)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        hc, hs = d["hist_count"], d["hist_sum"]
        money = hs is not None
        short_count = [int(c) for c in d["short_count"]]
        last = int(d["last_minute"])
        ago = [int(a) for a in d["long_ago"]]
        long_count = [int(c) for c in d["long_count"]]
        ss, ls = d.get("short_sum"), d.get("long_sum")
        if (hc is None) != money or (ss is None) == money or (ls is None) == money:
            raise ValueError("VelocityRecord needs exactly one history, and sums iff money")
        short_sum = [float(s) for s in ss] if ss is not None else [0.0] * len(short_count)
        long_sum = [float(s) for s in ls] if ls is not None else [0.0] * len(long_count)
        ok = (
            len(short_count) == len(short_sum) > 0
            and len(ago) == len(long_count) == len(long_sum)
            and all(a >= 0 for a in ago)
            and all(a > b for a, b in zip(ago, ago[1:], strict=False))  # ascending minutes
            and all(c >= 0 for c in short_count)
            and all(c > 0 for c in long_count)
            and all(math.isfinite(s) for s in [*short_sum, *long_sum])
        )
        if not ok:
            raise ValueError("inconsistent VelocityRecord state")
        return cls(
            short=list(zip(short_count, short_sum, strict=True)),
            long={last - a: (c, s) for a, c, s in zip(ago, long_count, long_sum, strict=True)},
            last_minute=last,
            hist_count=None if hc is None else P2Quantiles.from_dict(hc),
            hist_sum=None if hs is None else P2Quantiles.from_dict(hs),
            refused=[int(m) for m in d.get("refused", [])],
        )


@dataclass(slots=True, frozen=True)
class _Signal:
    score: float
    evidence: dict[str, object]


class VelocityDetector:
    name: str = NAME

    def __init__(
        self,
        store: StateStore,
        cfg: EngineConfig,
        hints: Mapping[str, ToolHint] = MappingProxyType({}),
    ) -> None:
        self._store = store
        self._cfg = cfg
        self._hints = hints

    # -- helpers -----------------------------------------------------------------------------

    def _record(self, key: str) -> VelocityRecord | None:
        rec = self._store.get(key)
        return rec if isinstance(rec, VelocityRecord) else None

    def _money_sums(self, e: ActionEnvelope) -> list[tuple[str, float]]:
        """``(path, finite sum of its values)`` for each money path carrying a finite value."""
        if not self._cfg.vel_sum_signal:
            return []
        h = self._hints.get(e.tool_id)
        hints: Mapping[str, str] = h.field_hints if h is not None else {}
        out: list[tuple[str, float]] = []
        for path, values in e.flat_args.items():
            hint = hints.get(path)
            if hint is None:  # implies direction_for(path) == "up" (same anchored keywords)
                money = _MONEY_NAME.search(path.rsplit(".", 1)[-1]) is not None
            else:
                money = hint == "money"
            if not money:
                continue
            xs = [x for x in map(try_number, values) if x is not None and math.isfinite(x)]
            if xs:
                out.append((path, clamp_finite(sum(xs))))
        return out

    def _virtual(self, rec: VelocityRecord, minute: int) -> tuple[Bucket, list[Bucket]]:
        """Window total and pending history samples after advancing to ``minute``; pure."""
        samples, total = _roll(
            list(rec.short), rec.last_minute, max(minute, rec.last_minute), rec.money
        )
        return total, samples

    def _signal(self, hist: P2Quantiles | None, pending: list[float], y: float) -> _Signal:
        """Robust one-sided z of ``y`` against ``hist`` plus ``pending`` samples; pure."""
        cfg = self._cfg
        n = (hist.n if hist is not None else 0) + len(pending)
        if hist is None or n == 0:
            return _Signal(0.0, {"baseline_median": None, "n_hist": 0, "z": None, "cold": True})
        if pending:
            hist = P2Quantiles.from_dict(hist.to_dict())  # work on a copy
            for v in pending:
                hist.add(v)
        q25, q50, q75 = (hist.quantile(p) for p in _HIST_PROBS)
        ev: dict[str, object] = {"baseline_median": signed_expm1(q50), "n_hist": n}
        if n < cfg.vel_cold_n:
            return _Signal(0.0, {**ev, "z": None, "cold": True})
        z = max((y - q50) / max((q75 - q25) / IQR_TO_SIGMA, cfg.vel_eps), 0.0)
        return _Signal(squash(z, cfg.z0, cfg.tau), {**ev, "z": z, "cold": False})

    # -- SubDetector ----------------------------------------------------------------------------

    def score(self, e: ActionEnvelope, types: TypeView) -> list[DetectorScore]:
        agent, tool = e.agent_id, e.tool_id
        minute = e.ts_unix_ms // MS_PER_MIN
        rec = self._record(key_vel(agent, tool))
        count, samples = 1, list[Bucket]()
        if rec is not None:
            (count, _), samples = self._virtual(rec, minute)
            count += 1
            if rec.refused:
                lo = max(minute, rec.last_minute) - self._cfg.vel_short_min
                count += sum(1 for m in rec.refused if m > lo)
        hist = rec.hist_count if rec is not None else None
        sig = self._signal(hist, [math.log1p(c) for c, _ in samples], math.log1p(count))
        best_field: str | None = None
        best = _Signal(sig.score, {"signal": "count", "window_count": count, **sig.evidence})
        for path, x in self._money_sums(e):
            mrec = self._record(key_vel(agent, tool, path))
            if mrec is None:
                total, pending = x, []
            else:
                (_, total), msamples = self._virtual(mrec, minute)
                total = clamp_finite(total + x)
                pending = [signed_log1p(s) for _, s in msamples]
            hist_sum = mrec.hist_sum if mrec is not None else None
            msig = self._signal(hist_sum, pending, signed_log1p(total))
            if msig.score > best.score:
                best_field = path
                best = _Signal(
                    msig.score,
                    {"signal": "sum", "window_count": count, "window_sum": total, **msig.evidence},
                )
        return [DetectorScore(NAME, best_field, best.score, best.evidence)]

    def _commit(self, key: str, minute: int, x: float, money: bool) -> None:
        rec = self._record(key)
        if rec is None:
            rec = VelocityRecord.new(self._cfg, minute, money)
            self._store.put(key, rec)
        to = max(minute, rec.last_minute)
        samples, _ = _roll(rec.short, rec.last_minute, to, rec.money)
        for c, s in samples:
            if rec.hist_count is not None:
                rec.hist_count.add(math.log1p(c))
            if rec.hist_sum is not None:
                rec.hist_sum.add(signed_log1p(s))
        i = to % len(rec.short)
        c, s = rec.short[i]
        rec.short[i] = (c + 1, clamp_finite(s + x))
        rec.last_minute = to
        c, s = rec.long.get(to, _EMPTY)
        rec.long[to] = (c + 1, clamp_finite(s + x))
        cutoff = to - self._cfg.vel_long_min
        while rec.long and (oldest := next(iter(rec.long))) <= cutoff:  # keys ascend
            del rec.long[oldest]
        if rec.refused:
            self._prune_refused(rec, to)

    def _prune_refused(self, rec: VelocityRecord, minute: int) -> None:
        lo = minute - self._cfg.vel_short_min
        rec.refused = [m for m in rec.refused if m > lo]

    def observe_refused(self, e: ActionEnvelope) -> None:
        """Count a refused call as an attempt in its short window (``vel_count_attempts``)."""
        if not self._cfg.vel_count_attempts:
            return
        rec = self._record(key_vel(e.agent_id, e.tool_id))
        if rec is None:
            return
        minute = max(e.ts_unix_ms // MS_PER_MIN, rec.last_minute)
        rec.refused.append(minute)
        self._prune_refused(rec, minute)

    def update(self, e: ActionEnvelope, types: TypeView) -> None:
        agent, tool = e.agent_id, e.tool_id
        minute = e.ts_unix_ms // MS_PER_MIN
        self._commit(key_vel(agent, tool), minute, 0.0, money=False)
        for path, x in self._money_sums(e):
            self._commit(key_vel(agent, tool, path), minute, x, money=True)
