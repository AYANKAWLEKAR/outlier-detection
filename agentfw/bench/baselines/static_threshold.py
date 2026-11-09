"""Static-threshold baseline (spec 5.5).

Per (tool, numeric path): fixed upper bound = ``quantile`` of all train values. Per (tool,
categorical path): allowlist of train values. Nothing on sequence. Paths never seen in train
for a tool have no bound/allowlist and therefore never violate. Score is 1.0 on any violation,
else 0.0; ``update`` is a no-op (the fit is frozen).
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence

import numpy as np
import orjson

from agentfw.core.envelope import (
    ActionEnvelope,
    BaselineSummary,
    DetectorResult,
    DetectorScore,
    Outcome,
    Reason,
    Scalar,
)
from agentfw.core.patterns import is_id_like, try_number

__all__ = ["StaticThreshold"]

_NAME = "static_threshold"
_Key = tuple[str, str]


def _values(e: ActionEnvelope) -> Iterator[tuple[str, Scalar]]:
    for src in (e.flat_args, e.derived):
        for path, vals in src.items():
            for v in vals:
                yield path, v


def _numeric(v: Scalar) -> float | None:
    if isinstance(v, str) and is_id_like(v):
        return None
    return try_number(v)


def _categorical(v: Scalar) -> str | None:
    if isinstance(v, str) and try_number(v) is None and not is_id_like(v):
        return v
    return None


class StaticThreshold:
    def __init__(self, quantile: float = 0.99) -> None:
        self._q = quantile
        self._bounds: dict[_Key, float] = {}
        self._allow: dict[_Key, frozenset[str]] = {}

    def reset(self) -> None:
        self._bounds = {}
        self._allow = {}

    def fit(self, train: Sequence[ActionEnvelope]) -> None:
        nums: dict[_Key, list[float]] = {}
        cats: dict[_Key, set[str]] = {}
        for e in train:
            for path, v in _values(e):
                key = (e.tool_id, path)
                if (x := _numeric(v)) is not None:
                    if np.isfinite(x):
                        nums.setdefault(key, []).append(x)
                elif (c := _categorical(v)) is not None:
                    cats.setdefault(key, set()).add(c)
        self._bounds = {k: float(np.quantile(vs, self._q)) for k, vs in nums.items()}
        self._allow = {k: frozenset(vs) for k, vs in cats.items()}

    def score(self, e: ActionEnvelope) -> DetectorResult:
        t0 = time.perf_counter()
        scores: list[DetectorScore] = []
        reasons: list[Reason] = []
        for path, v in _values(e):
            key = (e.tool_id, path)
            if (x := _numeric(v)) is not None:
                bound = self._bounds.get(key)
                if bound is not None and x > bound:
                    ev = {"kind": "above_bound", "bound": bound}
                    scores.append(DetectorScore(_NAME, path, 1.0, ev))
                    reasons.append(
                        Reason(_NAME, path, v, BaselineSummary(bound, 0.0, 0), None, None, None)
                    )
            elif (c := _categorical(v)) is not None:
                allow = self._allow.get(key)
                if allow is not None and c not in allow:
                    ev = {"kind": "not_in_allowlist", "allowlist_size": len(allow)}
                    scores.append(DetectorScore(_NAME, path, 1.0, ev))
                    reasons.append(Reason(_NAME, path, v, None, None, None, None))
        return DetectorResult(
            fused_score=1.0 if scores else 0.0,
            per_detector=tuple(scores),
            reasons=tuple(reasons),
            latency_us=(time.perf_counter() - t0) * 1e6,
        )

    def update(self, e: ActionEnvelope, outcome: Outcome) -> None:
        return None

    def state_bytes(self) -> int:
        payload = {
            "bounds": {f"{t}\x00{p}": b for (t, p), b in sorted(self._bounds.items())},
            "allow": {f"{t}\x00{p}": sorted(a) for (t, p), a in sorted(self._allow.items())},
        }
        return len(orjson.dumps(payload, option=orjson.OPT_SORT_KEYS))
