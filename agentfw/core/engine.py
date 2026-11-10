"""The engine (spec §6): every sub-detector composed behind the harness ``Detector`` protocol.

``score(e)`` (pure): look up the type of every flat and derived path once (``TypeView`` as of
*before* this envelope), run the enabled sub-detectors, fuse (§6.7), and read the agent's
calibrated threshold and decision band (§6.8). The result carries ``field_types`` for every
path. The engine remembers ``(envelope_id, fused)`` of the last score; that cache is not
detector state (it is not in the store and never counted by ``state_bytes``).

``update(e, outcome)``: nothing moves unless the outcome is ``ALLOWED`` / ``APPROVED`` (a blocked
or denied action never moves a baseline; two opt-in exceptions (Task 1.15): with
``cal_censor="winsorize"`` a ``BLOCKED`` call adds the threshold that blocked it to the calibrator,
and with ``vel_count_attempts`` a refused call counts in velocity's current window -- neither
moves a baseline distribution). Otherwise, in order:

1. the fused score and threshold that decided ``e``: the cached ones when they are ``e``'s, else
   a fresh pure score (and the current threshold) computed *before* anything below mutates state;
2. sub-detector updates (with the same pre-envelope types as ``score``);
3. pooled priors (§6.10): every finite NUMERIC value as ``signed_log1p(x)`` (and the previous
   response's ``item_count`` under magnitude's ``__resp__.item_count``, unless
   ``mag_resp_signal`` is off) when magnitude is enabled;
   every CATEGORICAL / ID value (derived ``__domain__`` / ``__host__`` / ``__prefix__`` included)
   under novelty's key ``value_key(v)`` when novelty is enabled. ``PooledPriors`` applies
   ``priors_mode``;
4. the calibrator, with the fused score from step 1 -- unless ``cal_censor`` says otherwise for a
   call the engine itself blocked (fused >= threshold): ``"drop"`` skips it, ``"winsorize"``
   feeds the threshold instead, both only once the agent's calibrator holds
   ``cal_censor_min_n`` scores (Task 1.15, ``docs/decisions/008-tier1-gate.md``);
5. type inference, last, over the flat paths (derived paths are typed by construction).

Freetext never reaches state: the normalizer already replaced strings of ``freetext_min_len``+
characters by a derived ``__strlen__`` length.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from time import perf_counter_ns
from types import MappingProxyType

from agentfw.core.calibration import Calibrator
from agentfw.core.config import CAL_CENSOR_MODES, EngineConfig
from agentfw.core.detectors.base import SubDetector, signed_log1p, value_key
from agentfw.core.detectors.magnitude import RESP_FIELD, MagnitudeDetector
from agentfw.core.detectors.novelty import NoveltyDetector
from agentfw.core.detectors.sequence import SequenceDetector
from agentfw.core.detectors.velocity import VelocityDetector
from agentfw.core.envelope import (
    ActionEnvelope,
    DetectorResult,
    DetectorScore,
    FieldType,
    Outcome,
)
from agentfw.core.fusion import FusionResult, fuse
from agentfw.core.hints import ToolHint
from agentfw.core.patterns import try_number
from agentfw.core.priors import PooledPriors
from agentfw.core.schema_infer import TypeInference, TypeView
from agentfw.core.state import DictStore, StateStore

__all__ = ["ALL_DETECTORS", "DETECTOR_ORDER", "Engine"]

DETECTOR_ORDER = ("magnitude", "novelty", "velocity", "sequence")
ALL_DETECTORS = frozenset(DETECTOR_ORDER)

_LEARN = frozenset({Outcome.ALLOWED, Outcome.APPROVED})
_CATEGORICAL = frozenset({FieldType.CATEGORICAL, FieldType.ID})
_NO_HINTS: Mapping[str, ToolHint] = MappingProxyType({})


def _finite_y(v: object) -> float | None:
    """``signed_log1p(x)`` of a finite numeric value, else ``None`` (NaN/±inf/non-numbers)."""
    x = try_number(v)
    return signed_log1p(x) if x is not None and math.isfinite(x) else None


class _CallTypes(TypeView):
    """A ``TypeView`` pinned to the types looked up once for one call (other paths fall through)."""

    def __init__(
        self, inference: TypeInference, agent: str, tool: str, types: dict[str, FieldType]
    ) -> None:
        super().__init__(inference, agent, tool)
        self.types = types

    def __call__(self, path: str) -> FieldType:
        t = self.types.get(path)
        return t if t is not None else TypeView.__call__(self, path)


class Engine:
    def __init__(
        self,
        cfg: EngineConfig = EngineConfig(),  # noqa: B008 - frozen dataclass, safe as a default
        store: StateStore | None = None,
        enabled: frozenset[str] = ALL_DETECTORS,
        hints: Mapping[str, ToolHint] = _NO_HINTS,
    ) -> None:
        unknown = set(enabled) - ALL_DETECTORS
        if unknown:
            raise ValueError(f"unknown detectors {sorted(unknown)}; known: {DETECTOR_ORDER}")
        if cfg.cal_censor not in CAL_CENSOR_MODES:
            raise ValueError(f"unknown cal_censor {cfg.cal_censor!r}; known: {CAL_CENSOR_MODES}")
        self.cfg = cfg
        self.store: StateStore = DictStore() if store is None else store
        self.enabled = frozenset(enabled)
        self.hints = hints
        self.inference = TypeInference(self.store, cfg)
        self.priors = PooledPriors(self.store, cfg)
        self.calibrator = Calibrator(self.store, cfg)
        self._feed_numeric_priors = "magnitude" in self.enabled and cfg.priors_mode != "off"
        self._feed_categorical_priors = "novelty" in self.enabled and cfg.priors_mode != "off"
        built: dict[str, SubDetector] = {
            "magnitude": MagnitudeDetector(self.store, cfg, self.priors, hints),
            "novelty": NoveltyDetector(self.store, cfg, self.priors),
            "velocity": VelocityDetector(self.store, cfg, hints),
            "sequence": SequenceDetector(self.store, cfg, hints),
        }
        self._detectors = tuple(built[n] for n in DETECTOR_ORDER if n in self.enabled)
        vel = built["velocity"]
        self._velocity = (
            vel if isinstance(vel, VelocityDetector) and "velocity" in self.enabled else None
        )
        self._cache: tuple[str, float, float] | None = None  # (envelope_id, fused, threshold)

    # -- Detector protocol ---------------------------------------------------------------------

    def reset(self) -> None:
        self.store.clear()
        self._cache = None

    def score(self, e: ActionEnvelope) -> DetectorResult:
        t0 = perf_counter_ns()
        view = self._view(e)
        per, fr = self._fuse(e, view)
        agent = e.agent_id
        theta = self.calibrator.threshold(agent)
        decision = self.calibrator.decide(agent, fr.fused)
        self._cache = (e.envelope_id, fr.fused, theta)
        return DetectorResult(
            fused_score=fr.fused,
            per_detector=per,
            reasons=fr.reasons,
            latency_us=(perf_counter_ns() - t0) / 1000.0,
            threshold=theta,
            decision=decision,
            field_types=view.types,
        )

    def update(self, e: ActionEnvelope, outcome: Outcome) -> None:
        cached = self._cache
        fused: float | None = None
        theta = math.nan
        if cached is not None and cached[0] == e.envelope_id:
            self._cache = None
            _, fused, theta = cached
        censor = self.cfg.cal_censor
        if censor != "none" and self.calibrator.n_scored(e.agent_id) < self.cfg.cal_censor_min_n:
            censor = "none"  # still in shadow: learn like the spec
        if outcome not in _LEARN:
            if self._velocity is not None:
                self._velocity.observe_refused(e)  # no-op unless vel_count_attempts
            if outcome is Outcome.BLOCKED and censor == "winsorize":
                if fused is None:
                    theta = self.calibrator.threshold(e.agent_id)
                self.calibrator.observe_censored(e.agent_id, theta)
            return
        view = self._view(e)
        if fused is None:
            fused = self._fuse(e, view)[1].fused  # pure: before any state below moves
            theta = self.calibrator.threshold(e.agent_id)
        for d in self._detectors:
            d.update(e, view)
        self._observe_priors(e, view)
        if censor == "none" or fused < theta:
            self.calibrator.observe(e.agent_id, fused, outcome)
        elif censor == "winsorize":
            self.calibrator.observe(e.agent_id, theta, outcome)
        agent, tool = e.agent_id, e.tool_id
        for path, values in e.flat_args.items():
            self.inference.observe(agent, tool, path, values)

    def state_bytes(self) -> int:
        return self.store.nbytes()

    # -- internals ------------------------------------------------------------------------------

    def _view(self, e: ActionEnvelope) -> _CallTypes:
        base = TypeView(self.inference, e.agent_id, e.tool_id)
        types = {path: base(path) for path in e.flat_args}
        for path in e.derived:
            types[path] = base(path)
        return _CallTypes(self.inference, e.agent_id, e.tool_id, types)

    def _fuse(
        self, e: ActionEnvelope, view: TypeView
    ) -> tuple[tuple[DetectorScore, ...], FusionResult]:
        per: list[DetectorScore] = []
        for d in self._detectors:
            per.extend(d.score(e, view))
        return tuple(per), fuse(per, self.cfg)

    def _observe_priors(self, e: ActionEnvelope, view: _CallTypes) -> None:
        num, cat = self._feed_numeric_priors, self._feed_categorical_priors
        if not (num or cat):
            return
        priors, tool, max_len = self.priors, e.tool_id, self.cfg.freetext_min_len
        for src in (e.flat_args, e.derived):
            for path, values in src.items():
                ft = view.types[path]
                if num and ft is FieldType.NUMERIC:
                    for v in values:
                        if (y := _finite_y(v)) is not None:
                            priors.observe_numeric(tool, path, y)
                elif cat and ft in _CATEGORICAL:
                    for v in values:
                        if v is not None:
                            priors.observe_categorical(tool, path, value_key(v, max_len))
        prev = e.prev_response  # magnitude learns the same value under the same field
        if not (num and self.cfg.mag_resp_signal) or prev is None:
            return
        if (y := _finite_y(prev.item_count)) is not None:
            priors.observe_numeric(prev.tool_id, RESP_FIELD, y)
