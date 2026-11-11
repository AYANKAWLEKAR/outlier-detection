"""Magnitude detector: robust z of ``sign(x)*log1p(|x|)`` per (agent, tool, path) (spec §6.3,
amended by A1.2).

Scoring, per NUMERIC path (flat then derived): every value is parsed with ``try_number``;
NaN and non-numbers are skipped; any ``±inf`` scores 1.0 with ``evidence["non_finite"]``.
Otherwise each finite value ``x`` gets ``y = signed_log1p(x)`` and is scored by the rule
``mag_score`` selects; the path's score is the maximum over its values (for ``up`` that is the
score of the largest value).

- ``"z"`` (spec §6.3): ``z = (y - q50) / max((q75 - q25) / 1.349, mag_eps)``, one-sided
  (``max(z, 0)``) for ``direction == "up"`` and ``|z|`` for ``"both"``, and
  ``s = squash(z, z0, tau)``.
- ``"tail"`` (spec A1.2): ``d = y - q50`` over a robust scale taken on the side ``d`` falls on,
  ``sigma = max((q75 - q50) / 0.6745, (q95 - q50) / 1.6449)`` for ``d >= 0`` and
  ``max((q50 - q25) / 0.6745, (q50 - q05) / 1.6449)`` below,
  ``scale = max(sigma * inflate, mag_eps)`` (``inflate`` is 1 on a warm record and
  ``prior_iqr_inflate`` on a pooled prior; the ``mag_eps`` floor itself is never inflated);
  ``up``: ``z = max(d / scale, 0)``, ``p = Q(z)``; ``both``: ``z = |d| / scale``,
  ``p = min(1, 2 Q(z))`` with ``Q`` the standard normal upper tail; and
  ``s = posterior(p, tail_prior)`` (exactly 1.0 when ``p`` is 0). Evidence gains ``p_tail``.

Both rules read the same ``MagRecord`` markers: no state differs between them, so a record
trained under either scores under the other. Any other ``mag_score`` is rejected when the
detector is built (``ValueError``), so a mistyped value never silently selects the other rule.
So is a ``tail_prior`` outside the open interval (0, 1) (NaN included), under either rule: it is
a prior probability, degenerate at 0 and 1, and beyond them the tail score leaves [0, 1].

Cold keys (``n < mag_cold_n``) score against the pooled prior (§6.10) with its scale inflated by
``prior_iqr_inflate`` and ``prior_used=True``; with no usable prior they score 0.0 with
``evidence["cold"] = True``.

Response metadata (§2.3): the previous call's ``item_count`` (``e.prev_response``) is scored on
the next call under field ``__resp__.item_count`` against
``m:{agent}:{prev_tool}:__resp__.item_count``, and ``update`` pushes that *same* ``prev_response``
count into the same key. Score and update of the response signal therefore happen on one
envelope (call k+1), so a blocked k+1 never moves the response baseline ("score before
update"). The envelope's own ``response_meta.item_count`` is never pushed directly; as a
consequence a session's last response is never learned (accepted trade-off). With
``mag_resp_signal=False`` (Open Decision 5 sweep) the response signal is neither scored nor learned.

Every score's evidence carries ``n`` (0 when no record exists), ``direction`` and
``prior_used``, including cold and non-finite scores.

``score`` only reads the store; ``update`` pushes finite values only (never NaN/±inf).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Self, get_args

from agentfw.core.config import Direction, EngineConfig, ScoreModel
from agentfw.core.detectors.base import (
    IQR_TO_SIGMA,
    PriorSource,
    clamp_finite,
    iter_fields,
    normal_sf,
    posterior,
    signed_expm1,
    signed_log1p,
    squash,
)
from agentfw.core.envelope import ActionEnvelope, DetectorScore, FieldType, Scalar
from agentfw.core.hints import ToolHint
from agentfw.core.patterns import try_number
from agentfw.core.schema_infer import TypeView
from agentfw.core.state import StateStore, key_mag, register_record
from agentfw.core.stats.ewma import Ewma
from agentfw.core.stats.p2 import P2Quantiles

__all__ = ["RESP_FIELD", "Direction", "MagRecord", "MagnitudeDetector", "direction_for"]

RESP_FIELD = "__resp__.item_count"
NAME = "magnitude"

_UP_NAME = re.compile(r"(?i)(^|_)(amount|count|qty|total|len|strlen)")
_UP_HINTS = frozenset({"money", "count"})
_SCORE_MODELS: tuple[ScoreModel, ...] = get_args(ScoreModel)
# Standard normal quantiles: the half-normal sigma implied by the distance from the median to
# the quartile (0.75) and to the 5 % / 95 % marker (spec A1.2).
_Z75 = 0.6744897501960817
_Z95 = 1.6448536269514722


def direction_for(path: str, hints: Mapping[str, str], default: Direction = "both") -> Direction:
    """``up`` for money/count-hinted fields, or when a keyword (amount, count, qty, total, len)
    (or strlen) starts the last path segment or follows an ``_`` in it (``item_count``, ``__len__``,
    ``total_amount``; not ``account``, ``discount``, ``calendar``); else ``default``
    (``EngineConfig.mag_direction_default``, Open Decision 2)."""
    if hints.get(path) in _UP_HINTS:
        return "up"
    return "up" if _UP_NAME.search(path.rsplit(".", 1)[-1]) else default


@register_record
@dataclass(slots=True)
class MagRecord:
    KIND: ClassVar[str] = "mag"
    q: P2Quantiles
    ewma: Ewma
    ewma2: Ewma
    direction: Direction = "both"
    n: int = 0

    @classmethod
    def new(cls, cfg: EngineConfig, direction: Direction) -> Self:
        return cls(
            q=P2Quantiles(cfg.mag_probs),
            ewma=Ewma(cfg.ewma_alpha),
            ewma2=Ewma(cfg.ewma_alpha),
            direction=direction,
        )

    def push(self, x: float) -> None:
        """Fold one raw value in; non-finite values are ignored."""
        if not math.isfinite(x):
            return
        self.push_y(signed_log1p(x))

    def push_y(self, y: float) -> None:
        """Fold one already-transformed value ``y = signed_log1p(x)`` in; non-finite is ignored."""
        if not math.isfinite(y):
            return
        self.q.add(y)
        self.ewma.add(y)
        self.ewma2.add(y * y)
        self.n += 1

    def to_dict(self) -> dict[str, object]:
        return {
            "q": self.q.to_dict(),
            "ewma": self.ewma.to_dict(),
            "ewma2": self.ewma2.to_dict(),
            "direction": self.direction,
            "n": self.n,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        direction = d["direction"]
        if direction not in ("up", "both"):
            raise ValueError(f"bad direction {direction!r}")
        return cls(
            q=P2Quantiles.from_dict(d["q"]),
            ewma=Ewma.from_dict(d["ewma"]),
            ewma2=Ewma.from_dict(d["ewma2"]),
            direction=direction,
            n=int(d["n"]),
        )


@dataclass(slots=True, frozen=True)
class _Ref:
    """What a value is scored against: a warm per-agent record or a pooled prior."""

    rec: MagRecord | None  # None: cold with no usable prior
    direction: Direction
    prior_used: bool
    n: int  # observations behind the ref (0 when no record exists)
    inflate: float = 1.0


class MagnitudeDetector:
    name: str = NAME

    def __init__(
        self,
        store: StateStore,
        cfg: EngineConfig,
        priors: PriorSource | None = None,
        hints: Mapping[str, ToolHint] = MappingProxyType({}),
    ) -> None:
        if cfg.mag_score not in _SCORE_MODELS:
            raise ValueError(f"unknown mag_score {cfg.mag_score!r}; known: {_SCORE_MODELS}")
        if not 0.0 < cfg.tail_prior < 1.0:  # also rejects NaN
            raise ValueError(f"tail_prior must be in (0, 1), got {cfg.tail_prior!r}")
        self._store = store
        self._cfg = cfg
        self._priors = priors
        self._hints = hints

    # -- helpers -----------------------------------------------------------------------------

    def _field_hints(self, tool: str) -> Mapping[str, str]:
        h = self._hints.get(tool)
        return h.field_hints if h is not None else {}

    def _record(self, key: str) -> MagRecord | None:
        rec = self._store.get(key)
        return rec if isinstance(rec, MagRecord) else None

    def _ref(self, agent: str, tool: str, path: str) -> _Ref:
        rec = self._record(key_mag(agent, tool, path))
        if rec is not None:
            direction = rec.direction
        else:
            direction = direction_for(
                path, self._field_hints(tool), self._cfg.mag_direction_default
            )
        n = rec.n if rec is not None else 0
        if rec is not None and n >= self._cfg.mag_cold_n:
            return _Ref(rec, direction, prior_used=False, n=n)
        if self._priors is not None:
            prior = self._priors.magnitude_ref(tool, path, direction)
            if prior is not None and prior.n >= self._cfg.mag_cold_n:
                inflate = self._cfg.prior_iqr_inflate
                return _Ref(prior, direction, prior_used=True, n=prior.n, inflate=inflate)
        return _Ref(None, direction, prior_used=False, n=n)

    @staticmethod
    def _base_evidence(ref: _Ref) -> dict[str, object]:
        return {"n": ref.n, "direction": ref.direction, "prior_used": ref.prior_used}

    def _score_value(self, ref: _Ref, x: float) -> tuple[float, dict[str, object]]:
        """Score one finite value against ``ref``; pure."""
        rec = ref.rec
        if rec is None:
            return 0.0, {"cold": True, **self._base_evidence(ref)}
        cfg = self._cfg
        q = rec.q
        y = signed_log1p(x)
        q25, q50, q75 = q.quantile(0.25), q.quantile(0.5), q.quantile(0.75)
        tail: dict[str, object] = {}
        if cfg.mag_score == "tail":  # spec A1.2
            d = y - q50
            if d >= 0.0:
                sigma = max((q75 - q50) / _Z75, (q.quantile(0.95) - q50) / _Z95)
            else:
                sigma = max((q50 - q25) / _Z75, (q50 - q.quantile(0.05)) / _Z95)
            scale = max(sigma * ref.inflate, cfg.mag_eps)
            if ref.direction == "up":
                z = max(d / scale, 0.0)
                p = normal_sf(z)
            else:
                z = abs(d) / scale
                p = min(1.0, 2.0 * normal_sf(z))
            s = posterior(p, cfg.tail_prior)
            tail["p_tail"] = p
        else:  # spec §6.3
            scale = max((q75 - q25) / IQR_TO_SIGMA * ref.inflate, cfg.mag_eps)
            z = (y - q50) / scale
            z = max(z, 0.0) if ref.direction == "up" else abs(z)
            s = squash(z, cfg.z0, cfg.tau)
        return s, {
            "observed": x,
            "z": z,
            **tail,
            "percentile": q.percentile_of(y),
            "median_raw": signed_expm1(q50),
            "iqr_raw": clamp_finite(signed_expm1(q75) - signed_expm1(q25)),
            **self._base_evidence(ref),
        }

    def _score_values(
        self, ref: _Ref, values: tuple[Scalar, ...], field_name: str, extra: Mapping[str, object]
    ) -> DetectorScore | None:
        xs: list[float] = []
        for v in values:
            x = try_number(v)
            if x is None or math.isnan(x):
                continue
            if math.isinf(x):
                ev = {"non_finite": True, **self._base_evidence(ref), **extra}
                return DetectorScore(NAME, field_name, 1.0, ev)
            xs.append(x)
        if not xs:
            return None
        best_s, best_ev = -1.0, {}
        for x in xs:
            s, ev = self._score_value(ref, x)
            if s > best_s:
                best_s, best_ev = s, ev
        return DetectorScore(NAME, field_name, best_s, {**best_ev, **extra})

    # -- SubDetector ----------------------------------------------------------------------------

    def score(self, e: ActionEnvelope, types: TypeView) -> list[DetectorScore]:
        out: list[DetectorScore] = []
        agent, tool = e.agent_id, e.tool_id
        for path, values in iter_fields(e):
            if types(path) is not FieldType.NUMERIC:
                continue
            ds = self._score_values(self._ref(agent, tool, path), values, path, {})
            if ds is not None:
                out.append(ds)
        prev = e.prev_response
        if self._cfg.mag_resp_signal and prev is not None and prev.item_count is not None:
            ref = self._ref(agent, prev.tool_id, RESP_FIELD)
            ds = self._score_values(ref, (prev.item_count,), RESP_FIELD, {"tool": prev.tool_id})
            if ds is not None:
                out.append(ds)
        return out

    def _push(self, agent: str, tool: str, path: str, values: tuple[Scalar, ...]) -> None:
        xs = [x for x in map(try_number, values) if x is not None and math.isfinite(x)]
        if not xs:
            return
        key = key_mag(agent, tool, path)
        rec = self._record(key)
        if rec is None:
            cfg = self._cfg
            rec = MagRecord.new(
                cfg, direction_for(path, self._field_hints(tool), cfg.mag_direction_default)
            )
            self._store.put(key, rec)
        for x in xs:
            rec.push(x)

    def update(self, e: ActionEnvelope, types: TypeView) -> None:
        agent, tool = e.agent_id, e.tool_id
        for path, values in iter_fields(e):
            if types(path) is FieldType.NUMERIC:
                self._push(agent, tool, path, values)
        prev = e.prev_response  # same value score() saw; never this call's own response
        if self._cfg.mag_resp_signal and prev is not None and prev.item_count is not None:
            self._push(agent, prev.tool_id, RESP_FIELD, (prev.item_count,))
