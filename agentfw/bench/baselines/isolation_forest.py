"""Per-tool isolation-forest baseline (spec 5.5).

One ``sklearn`` IsolationForest per ``tool_id``, fit on the train split. Features: numeric
flat/derived paths seen in train (missing -> train median) and one-hot columns for categorical
paths with at most ``max_onehot`` train values. The score is ``-score_samples`` min-max scaled on
the train scores and clipped to [0, 1]. Tools with fewer than ``min_train`` train rows score 0.0.
``update`` is a no-op (frozen fit). sklearn is allowed here only (never in ``agentfw.core``).
"""

from __future__ import annotations

import pickle
import time
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest

from agentfw.core.envelope import (
    ActionEnvelope,
    DetectorResult,
    DetectorScore,
    Outcome,
    Scalar,
)
from agentfw.core.patterns import is_id_like, try_number

__all__ = ["IsolationForestBaseline"]

_NAME = "isolation_forest"
_CLIP = 1e30


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


def _row(e: ActionEnvelope) -> tuple[dict[str, float], dict[str, str]]:
    """First numeric / first categorical value per path (NaN numerics are skipped)."""
    nums: dict[str, float] = {}
    cats: dict[str, str] = {}
    for path, v in _values(e):
        if (x := _numeric(v)) is not None:
            if not np.isnan(x):
                nums.setdefault(path, x)
        elif (c := _categorical(v)) is not None:
            cats.setdefault(path, c)
    return nums, cats


@dataclass(slots=True)
class _ToolModel:
    num_paths: list[str]
    medians: list[float]
    cat_cols: list[tuple[str, str]]
    forest: IsolationForest
    lo: float
    hi: float

    def matrix(self, e: ActionEnvelope) -> np.ndarray:
        nums, cats = _row(e)
        x = [nums.get(p, m) for p, m in zip(self.num_paths, self.medians, strict=True)]
        x += [1.0 if cats.get(p) == v else 0.0 for p, v in self.cat_cols]
        return np.clip(np.array([x], dtype=np.float64), -_CLIP, _CLIP)

    def scale(self, raw: float) -> float:
        span = self.hi - self.lo
        s = (raw - self.lo) / span if span > 0 else (1.0 if raw > self.hi else 0.0)
        return float(min(1.0, max(0.0, s)))


class IsolationForestBaseline:
    def __init__(
        self,
        seed: int = 0,
        n_estimators: int = 200,
        max_onehot: int = 20,
        min_train: int = 50,
    ) -> None:
        self._seed = seed
        self._n_estimators = n_estimators
        self._max_onehot = max_onehot
        self._min_train = min_train
        self._models: dict[str, _ToolModel] = {}
        self._state_bytes = 0

    def reset(self) -> None:
        self._models = {}
        self._state_bytes = 0

    def fit(self, train: Sequence[ActionEnvelope]) -> None:
        by_tool: dict[str, list[ActionEnvelope]] = defaultdict(list)
        for e in train:
            by_tool[e.tool_id].append(e)
        self._models = {}
        for tool in sorted(by_tool):
            if len(by_tool[tool]) >= self._min_train and (m := self._fit_tool(by_tool[tool])):
                self._models[tool] = m
        self._state_bytes = sum(len(pickle.dumps(m.forest)) for m in self._models.values())

    def _fit_tool(self, envs: Sequence[ActionEnvelope]) -> _ToolModel | None:
        rows = [_row(e) for e in envs]
        num_vals: dict[str, list[float]] = defaultdict(list)
        cat_vals: dict[str, set[str]] = defaultdict(set)
        for nums, cats in rows:
            for p, x in nums.items():
                num_vals[p].append(x)
            for p, c in cats.items():
                cat_vals[p].add(c)
        num_paths = sorted(num_vals)
        # a path that is mostly categorical is categorical, even if a stray value parses as number
        cat_cols = [
            (p, v)
            for p in sorted(cat_vals)
            if len(cat_vals[p]) <= self._max_onehot
            for v in sorted(cat_vals[p])
        ]
        if not num_paths and not cat_cols:
            return None
        medians = [float(np.median(np.clip(num_vals[p], -_CLIP, _CLIP))) for p in num_paths]
        forest = IsolationForest(n_estimators=self._n_estimators, random_state=self._seed)
        model = _ToolModel(num_paths, medians, cat_cols, forest, 0.0, 0.0)
        mat = np.vstack([model.matrix(e) for e in envs])
        forest.fit(mat)
        train_raw = -forest.score_samples(mat)
        model.lo, model.hi = float(train_raw.min()), float(train_raw.max())
        return model

    def score(self, e: ActionEnvelope) -> DetectorResult:
        t0 = time.perf_counter()
        model = self._models.get(e.tool_id)
        if model is None:
            det = DetectorScore(_NAME, None, 0.0, {"untrained": True})
            fused = 0.0
        else:
            anomaly = float(-model.forest.score_samples(model.matrix(e))[0])
            fused = model.scale(anomaly)
            det = DetectorScore(_NAME, None, fused, {"raw": anomaly})
        return DetectorResult(
            fused_score=fused,
            per_detector=(det,),
            reasons=(),
            latency_us=(time.perf_counter() - t0) * 1e6,
        )

    def update(self, e: ActionEnvelope, outcome: Outcome) -> None:
        return None

    def state_bytes(self) -> int:
        return self._state_bytes
