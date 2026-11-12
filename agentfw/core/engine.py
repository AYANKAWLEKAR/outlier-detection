"""The engine (spec §6): every sub-detector composed behind the harness ``Detector`` protocol.

``score(e)`` (pure): look up the type of every flat and derived path once (``TypeView`` as of
*before* this envelope), run the enabled sub-detectors, fuse (§6.7), and read the agent's
calibrated threshold and decision band (§6.8). The result carries ``field_types`` for every
path. The engine remembers ``(envelope_id, fused)`` of the last score; that cache is not
detector state (it is not in the store and never counted by ``state_bytes``).

``update(e, outcome)``: nothing moves unless the outcome is ``ALLOWED`` / ``APPROVED`` (a blocked
or denied action never moves a baseline; two exceptions (Task 1.15): with the opt-in
``cal_censor="winsorize"`` a ``BLOCKED`` call the engine itself would block (fused >= threshold)
adds that threshold to the calibrator, and with ``vel_count_attempts`` (default on by user
decision, decision 008; off = spec) a refused call counts in velocity's current window -- neither
moves a baseline distribution). Otherwise, in order:

1. the fused score and threshold that decided ``e``: the cached ones when they are ``e``'s, else
   a fresh pure score (and the current threshold) computed *before* anything below mutates state;
2. sub-detector updates (with the same pre-envelope types as ``score``);
3. pooled priors (§6.10): every finite value of a NUMERIC-typed path (never a BOOL-typed one,
   whatever ``mag_bool_numeric`` says: spec A1.3) as ``signed_log1p(x)`` (and the previous
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

Vital signs (F2, Task 1.5) are opt-in: ``enabled`` defaults to the v1 four (``ALL_DETECTORS``)
and accepts ``"vital"`` too (``KNOWN_DETECTORS``). Vital scores are never fused (T-07): fusion
runs over the v1 scores only, then the vital scores are appended to ``per_detector`` and their
reasons (score > ``reason_min``, highest first) after the v1 reasons. So the fused score, the
decision and every piece of state are the same with and without ``"vital"``.

Trajectory (F3, Task 1.7): every result carries a ``TrajectoryInfo``: the regime (T-06: per-key
history = the sequence detector's unigram count of the (agent, tool) >= ``mag_cold_n``; schemas =
the tool's hint ``has_schema``), the blast radius ``b`` (``core.blast``; ``None`` while
``br_mode="off"``), irreversibility and the threshold the decision used. ``br_mode="score"`` also
sets ``Reason.blast_radius = b`` on every reason; ``"modulate"`` decides against
``θ_eff = θ · (1 − br_k · b)``. ``DetectorResult.threshold``, the calibrator and every piece of
state keep the agent's own θ, so modulation moves decisions only. ``b`` reads the sequence
detector's kind records (write kind, item counts) and never writes, so ``score`` stays pure.

Session flags (F7 substrate, Task 2.1): ``score`` also derives the call's flags
(``core.trajectory.flags``) into ``TrajectoryInfo.flags`` and caches them with the fused score.
Everything else about sessions sits behind one gate, ``Engine._book`` (T-34, ruling R15): a
``SessionBook`` when ``sessions`` -- or, later, any feature that needs session records -- is on,
else None (no ``sess:`` record is ever written, ``density`` is None and the session settings are
not validated). With it, ``score`` reads the session's flag window into
``TrajectoryInfo.density`` (T-09; pure, ``SessionBook.peek``), and ``update`` first records the
call's flags in that window -- for **every** outcome, before the early return above, since a
blocked call is still a step of the session -- with the cached flags when they are ``e``'s, else
a fresh pure evaluation computed before anything moves (its fused score and threshold then serve
step 1). Session records are never read by a detector, so scores, decisions and every other
record are the same with and without the gate.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from time import perf_counter_ns
from types import MappingProxyType

from agentfw.core.blast import blast_radius, irreversible, regime_of
from agentfw.core.calibration import Calibrator
from agentfw.core.config import BR_MODES, CAL_CENSOR_MODES, EngineConfig
from agentfw.core.detectors.base import SubDetector, signed_log1p, value_key
from agentfw.core.detectors.magnitude import RESP_FIELD, MagnitudeDetector
from agentfw.core.detectors.novelty import NoveltyDetector
from agentfw.core.detectors.sequence import SequenceDetector
from agentfw.core.detectors.velocity import VelocityDetector
from agentfw.core.detectors.vital import VitalDetector
from agentfw.core.envelope import (
    ActionEnvelope,
    Decision,
    DetectorResult,
    DetectorScore,
    FieldType,
    Outcome,
    Reason,
    TrajectoryInfo,
)
from agentfw.core.fusion import FusionResult, fuse
from agentfw.core.hints import ToolHint
from agentfw.core.patterns import try_number
from agentfw.core.priors import PooledPriors
from agentfw.core.schema_infer import TypeInference, TypeView
from agentfw.core.state import DictStore, StateStore
from agentfw.core.trajectory.flags import Flag, derive_flags
from agentfw.core.trajectory.session import SessionBook

__all__ = ["ALL_DETECTORS", "DETECTOR_ORDER", "KNOWN_DETECTORS", "VITAL", "Engine"]

DETECTOR_ORDER = ("magnitude", "novelty", "velocity", "sequence")  # the fused (v1) detectors
ALL_DETECTORS = frozenset(DETECTOR_ORDER)
VITAL = "vital"
KNOWN_DETECTORS = ALL_DETECTORS | {VITAL}

_LEARN = frozenset({Outcome.ALLOWED, Outcome.APPROVED})
_CATEGORICAL = frozenset({FieldType.CATEGORICAL, FieldType.ID})
_NO_HINTS: Mapping[str, ToolHint] = MappingProxyType({})


def _finite_y(v: object) -> float | None:
    """``signed_log1p(x)`` of a finite numeric value, else ``None`` (NaN/±inf/non-numbers)."""
    x = try_number(v)
    return signed_log1p(x) if x is not None and math.isfinite(x) else None


def _with_b(reasons: tuple[Reason, ...], b: float | None) -> tuple[Reason, ...]:
    """Every reason with ``blast_radius=b`` (unchanged while ``b`` is None)."""
    if b is None:
        return reasons
    return tuple(dataclasses.replace(r, blast_radius=b) for r in reasons)


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
        unknown = set(enabled) - KNOWN_DETECTORS
        if unknown:
            known = (*DETECTOR_ORDER, VITAL)
            raise ValueError(f"unknown detectors {sorted(unknown)}; known: {known}")
        if cfg.cal_censor not in CAL_CENSOR_MODES:
            raise ValueError(f"unknown cal_censor {cfg.cal_censor!r}; known: {CAL_CENSOR_MODES}")
        if cfg.br_mode not in BR_MODES:
            raise ValueError(f"unknown br_mode {cfg.br_mode!r}; known: {BR_MODES}")
        if not 0.0 <= cfg.br_k <= 1.0:  # keeps θ_eff >= 0
            raise ValueError(f"br_k must be in [0, 1], got {cfg.br_k}")
        self.cfg = cfg
        self.store: StateStore = DictStore() if store is None else store
        self.enabled = frozenset(enabled)
        self.hints = hints
        self.inference = TypeInference(self.store, cfg)
        self.priors = PooledPriors(self.store, cfg)
        self.calibrator = Calibrator(self.store, cfg)
        self._feed_numeric_priors = "magnitude" in self.enabled and cfg.priors_mode != "off"
        self._feed_categorical_priors = "novelty" in self.enabled and cfg.priors_mode != "off"
        magnitude = MagnitudeDetector(self.store, cfg, self.priors, hints)
        # spec A1.3: novelty leaves a BOOL-contradicting number to magnitude wherever magnitude
        # has a reference for the field -- only when magnitude runs and the switch is on.
        numeric_ref = (
            magnitude.has_ref if "magnitude" in self.enabled and cfg.mag_bool_numeric else None
        )
        sequence = SequenceDetector(self.store, cfg, hints)
        self._sequence = sequence  # kind records and counts for the trajectory (read-only)
        built: dict[str, SubDetector] = {
            "magnitude": magnitude,
            "novelty": NoveltyDetector(self.store, cfg, self.priors, numeric_ref),
            "velocity": VelocityDetector(self.store, cfg, hints),
            "sequence": sequence,
        }
        self._detectors = tuple(built[n] for n in DETECTOR_ORDER if n in self.enabled)
        # Reported, never fused (T-07); stateless, so it has no update to run.
        self._vital = (
            VitalDetector(cfg, hints, sequence.is_write) if VITAL in self.enabled else None
        )
        vel = built["velocity"]
        self._velocity = (
            vel if isinstance(vel, VelocityDetector) and "velocity" in self.enabled else None
        )
        # The one session gate (T-34, ruling R15): ``_book`` is None exactly when no feature needs
        # session records; recording (``update``) and density (``_evaluate``) both test it. A later
        # feature that needs the window ORs its switch into this expression.
        sessions_on = cfg.sessions
        self._book: SessionBook | None = SessionBook(self.store, cfg) if sessions_on else None
        # (envelope_id, fused, threshold, flags) of the last score
        self._cache: tuple[str, float, float, frozenset[Flag]] | None = None

    # -- Detector protocol ---------------------------------------------------------------------

    def reset(self) -> None:
        self.store.clear()
        if self._book is not None:
            self._book.reset()
        self._cache = None

    def score(self, e: ActionEnvelope) -> DetectorResult:
        t0 = perf_counter_ns()
        view, per, fr, theta, traj, decision = self._evaluate(e)
        self._cache = (e.envelope_id, fr.fused, theta, traj.flags)
        return DetectorResult(
            fused_score=fr.fused,
            per_detector=per,
            reasons=_with_b(fr.reasons, traj.blast_radius),
            latency_us=(perf_counter_ns() - t0) / 1000.0,
            threshold=theta,
            decision=decision,
            field_types=view.types,
            trajectory=traj,
        )

    def update(self, e: ActionEnvelope, outcome: Outcome) -> None:
        cached = self._cache
        fused: float | None = None
        theta = math.nan
        flags: frozenset[Flag] | None = None
        if cached is not None and cached[0] == e.envelope_id:
            self._cache = None
            _, fused, theta, flags = cached
        book = self._book
        if book is not None:  # first, for every outcome (blocked calls enter the window)
            if flags is None:  # pure: nothing below has moved yet
                _, _, fr, theta, traj, _ = self._evaluate(e)
                fused, flags = fr.fused, traj.flags
            book.record(e, flags)
            book.maybe_evict(e.ts_unix_ms)
        censor = self.cfg.cal_censor
        if censor != "none" and self.calibrator.n_scored(e.agent_id) < self.cfg.cal_censor_min_n:
            censor = "none"  # still in shadow: learn like the spec
        if outcome not in _LEARN:
            if self._velocity is not None:
                self._velocity.observe_refused(e)  # no-op unless vel_count_attempts
            if outcome is Outcome.BLOCKED and censor == "winsorize":
                if fused is None:  # pure score and current threshold, nothing has moved yet
                    fused = self._fuse(e, self._view(e))[1].fused
                    theta = self.calibrator.threshold(e.agent_id)
                if fused >= theta:  # only the engine's own blocks are tail mass
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

    def _evaluate(
        self, e: ActionEnvelope
    ) -> tuple[
        _CallTypes, tuple[DetectorScore, ...], FusionResult, float, TrajectoryInfo, Decision
    ]:
        """Pure: types, scores, fusion, the agent's θ, the trajectory (flags and density
        included) and the decision."""
        view = self._view(e)
        per, fr = self._fuse(e, view)
        agent = e.agent_id
        theta = self.calibrator.threshold(agent)
        traj = self._trajectory(e, theta)
        decision = self.calibrator.decide(agent, fr.fused, traj.threshold_eff)
        hint = self.hints.get(e.tool_id)
        flags = frozenset(derive_flags(per, decision, traj.irreversible, hint, self.cfg))
        book = self._book
        density = None if book is None else book.density_with(book.peek(e.session_id), flags)
        traj = dataclasses.replace(traj, flags=flags, density=density)
        return view, per, fr, theta, traj, decision

    def _trajectory(self, e: ActionEnvelope, theta: float) -> TrajectoryInfo:
        cfg, seq = self.cfg, self._sequence
        agent, tool = e.agent_id, e.tool_id
        hint = self.hints.get(tool)
        has_schema = hint is not None and hint.has_schema
        regime = regime_of(has_schema, seq.n_tool(agent, tool), cfg.mag_cold_n)
        if cfg.br_mode == "off":
            return TrajectoryInfo(regime, None, irreversible(None, hint, False), theta)
        is_write = seq.is_write(agent, tool)
        b = blast_radius(e, hint, is_write, seq.has_item_count(agent, tool))
        theta_eff = theta * (1.0 - cfg.br_k * b) if cfg.br_mode == "modulate" else theta
        return TrajectoryInfo(
            regime, b, irreversible(b, hint, is_write, cfg.irreversible_b), theta_eff
        )

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
        fr = fuse(per, self.cfg)
        if self._vital is None:
            return tuple(per), fr
        vital = self._vital.score(e, view)
        reasons = fr.reasons + fuse(vital, self.cfg).reasons  # vital reasons only, fused unused
        return (*per, *vital), FusionResult(fr.fused, reasons)

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
