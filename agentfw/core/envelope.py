"""Data model: raw calls, normalized envelopes, labels, detector results and verdicts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

if TYPE_CHECKING:
    from agentfw.core.sketch.bloom import BloomFilter
    from agentfw.core.trajectory.flags import Flag
    from agentfw.core.trajectory.gate import GateOutcome
    from agentfw.core.trajectory.session import SessionVerdict

__all__ = [
    "ActionEnvelope",
    "BaselineSummary",
    "Decision",
    "DetectorResult",
    "DetectorScore",
    "FieldType",
    "IdSnapshot",
    "JsonValue",
    "Label",
    "LedgerRef",
    "Outcome",
    "PrevResponse",
    "RawCall",
    "Reason",
    "RecentCall",
    "Regime",
    "ResponseMeta",
    "Scalar",
    "TrajectoryInfo",
    "Verdict",
]

Scalar = str | int | float | bool | None
#: What a decision was based on (F3, T-06): per-key history (>= ``mag_cold_n`` observations of the
#: (agent, tool)) and/or a captured tool schema.
Regime = Literal["history+schemas", "history_only", "schemas_only", "bare"]


class FieldType(StrEnum):
    NUMERIC = "numeric"
    BOOL = "bool"
    CATEGORICAL = "categorical"
    ID = "id"
    FREETEXT = "freetext"
    UNKNOWN = "unknown"


class Outcome(StrEnum):
    ALLOWED = "allowed"
    BLOCKED = "blocked"
    APPROVED = "approved"
    DENIED = "denied"


class Decision(StrEnum):
    ALLOW = "allow"
    BLOCK = "block"
    REQUIRE_APPROVAL = "require_approval"


class ResponseMeta(BaseModel):
    model_config = ConfigDict(frozen=True)

    item_count: int | None = None
    bytes: int | None = None
    error: bool = False
    ids: tuple[str, ...] = Field(default=(), max_length=256)
    # MinHash signature of the response text (core.fingerprint); omitted while None so existing
    # traces and manifest hashes keep their bytes.
    fp: tuple[int, ...] | None = None

    @field_validator("fp")
    @classmethod
    def _check_fp(cls, v: tuple[int, ...] | None) -> tuple[int, ...] | None:
        if v is not None and (len(v) != 64 or any(not 0 <= x < (1 << 61) - 1 for x in v)):
            raise ValueError("fp must be None or 64 ints in [0, 2**61 - 1)")
        return v

    @model_serializer(mode="wrap")
    def _omit_none_fp(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        out: dict[str, Any] = handler(self)
        if self.fp is None:
            out.pop("fp", None)
        return out


class RawCall(BaseModel):
    model_config = ConfigDict(frozen=True)

    envelope_id: str
    schema_version: int = 1
    session_id: str
    agent_id: str
    principal_id: str
    seq_no: int = Field(ge=0)
    ts_unix_ms: int
    tool_id: str = Field(pattern=r"^[a-z0-9_]+\.[a-z0-9_]+$")
    args: dict[str, JsonValue]
    response_meta: ResponseMeta | None = None
    # Ids the task itself supplied (set on a session's first call). Omitted from serialized
    # output while empty so existing traces and manifest hashes keep their bytes.
    task_ids: tuple[str, ...] = Field(default=(), max_length=64)

    @model_serializer(mode="wrap")
    def _omit_empty_task_ids(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        out: dict[str, Any] = handler(self)
        if not self.task_ids:
            out.pop("task_ids", None)
        return out


@dataclass(slots=True, frozen=True)
class LedgerRef:
    """An id used in a call's args that the session ledger already knows (see core.ledger)."""

    artifact_id: str
    path: str
    producer_seq: int
    producer_tool: str | None
    source: Literal["args", "response", "task_input"]


@dataclass(slots=True, frozen=True)
class PrevResponse:
    tool_id: str
    item_count: int | None


@dataclass(slots=True, frozen=True)
class RecentCall:
    """One completed call in a session's recent-call ring (vital signs, F2).

    ``delta_i`` is the call's information gain, ``1 - max`` MinHash-Jaccard of its response
    fingerprint against the earlier fingerprints in the ring (1.0 when there are none), or
    ``None`` when the call had no fingerprint. ``new_artifacts`` counts the ledger entries the
    call created.
    """

    seq_no: int
    tool_id: str
    call_fp: int
    arg_tokens: frozenset[int]
    error: bool
    delta_i: float | None
    new_artifacts: int


@dataclass(slots=True, frozen=True)
class IdSnapshot:
    exact: frozenset[str]
    bloom: BloomFilter | None

    def __contains__(self, v: str) -> bool:
        if v in self.exact:
            return True
        return self.bloom is not None and v in self.bloom


@dataclass(slots=True, frozen=True)
class ActionEnvelope:
    envelope_id: str
    schema_version: int
    session_id: str
    agent_id: str
    principal_id: str
    seq_no: int
    ts_unix_ms: int
    tool_id: str
    args: dict[str, JsonValue]
    response_meta: ResponseMeta | None
    flat_args: Mapping[str, tuple[Scalar, ...]]
    derived: Mapping[str, tuple[Scalar, ...]]
    prior_tools: tuple[str, ...]
    session_ids_seen: IdSnapshot
    prev_response: PrevResponse | None
    blast_radius: float | None = None
    ledger_refs: tuple[LedgerRef, ...] = ()
    task_ids: tuple[str, ...] = ()
    # Vital signs (F2): this call's fingerprint, the session's completed calls before it (oldest
    # first, at most 32) and this call's own information gain against them (see RecentCall).
    call_fp: int = 0
    recent: tuple[RecentCall, ...] = ()
    delta_i: float | None = None


class Label(BaseModel):
    model_config = ConfigDict(frozen=True)

    envelope_id: str
    label: Literal["benign", "attack"]
    attack_family: str | None = None
    injection_level: Literal["prompt", "action"] | None = None
    attack_id: str | None = None
    control_type: Literal["benign_unusual"] | None = None
    strength: float | None = None
    step: int | None = None

    @model_validator(mode="after")
    def _check_integrity(self) -> Self:
        if self.label == "attack":
            if self.attack_family is None or self.attack_id is None:
                raise ValueError("attack labels require attack_family and attack_id")
            if self.control_type is not None:
                raise ValueError("attack labels must not carry a control_type")
        return self


@dataclass(slots=True, frozen=True)
class DetectorScore:
    detector: str
    field: str | None
    score: float
    evidence: Mapping[str, object]


@dataclass(slots=True, frozen=True)
class BaselineSummary:
    median: float
    iqr: float
    n: int


@dataclass(slots=True, frozen=True)
class Reason:
    detector: str
    field: str | None
    observed: Scalar
    baseline_summary: BaselineSummary | None
    percentile: float | None
    blast_radius: float | None
    prereq_missing: str | None
    prior_used: bool = False


@dataclass(slots=True, frozen=True)
class TrajectoryInfo:
    """Trajectory facts the engine attaches to every result (F3; later tasks append fields with
    defaults). ``blast_radius`` is ``None`` while ``br_mode="off"``; ``threshold_eff`` is the
    threshold the decision used (the agent's own unless ``br_mode="modulate"``)."""

    regime: Regime
    blast_radius: float | None
    irreversible: bool
    threshold_eff: float
    #: This call's flags (``core.trajectory.flags``) and its session's flag density including
    #: this call (T-09; ``None`` unless ``EngineConfig.sessions``).
    flags: frozenset[Flag] = frozenset()
    density: float | None = None
    #: The pre-execution gate's outcome (F4): set only when ``EngineConfig.gate`` is on and the
    #: call is irreversible.
    gate: GateOutcome | None = None
    #: Two-tier escalation (F5): ``escalated`` is True when the escalation stage ran on this call
    #: (``EngineConfig.escalate``, T-42); then ``fast_score`` is the fused score of the fast path
    #: and ``escalated_score`` the fresh re-score the decision rests on (both ``None`` otherwise).
    escalated: bool = False
    fast_score: float | None = None
    escalated_score: float | None = None
    #: Session verdict (F7, Task 3.3, T-43) of this call on ``density`` (sticky ``HALT``) and the
    #: session's fused mass including this call -- the aggregate, reported and never
    #: thresholded. Both ``None`` while the session gate is shut (like ``density``).
    session_verdict: SessionVerdict | None = None
    session_mass: float | None = None


@dataclass(slots=True, frozen=True)
class DetectorResult:
    fused_score: float
    per_detector: tuple[DetectorScore, ...]
    reasons: tuple[Reason, ...]
    latency_us: float
    threshold: float | None = None
    decision: Decision | None = None
    field_types: Mapping[str, FieldType] = field(default_factory=dict)
    trajectory: TrajectoryInfo | None = None  # always set by the engine


@dataclass(slots=True, frozen=True)
class Verdict:
    envelope_id: str
    decision: Decision
    fused_score: float
    threshold_used: float
    detector_scores: tuple[DetectorScore, ...]
    reasons: tuple[Reason, ...]
    policy_match: str | None
    mode: Literal["shadow", "enforce"]
    latency_us: float
