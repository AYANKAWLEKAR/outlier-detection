"""Data model: raw calls, normalized envelopes, labels, detector results and verdicts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

if TYPE_CHECKING:
    from agentfw.core.sketch.bloom import BloomFilter

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
    "Outcome",
    "PrevResponse",
    "RawCall",
    "Reason",
    "ResponseMeta",
    "Scalar",
    "Verdict",
]

Scalar = str | int | float | bool | None


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


@dataclass(slots=True, frozen=True)
class PrevResponse:
    tool_id: str
    item_count: int | None


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
class DetectorResult:
    fused_score: float
    per_detector: tuple[DetectorScore, ...]
    reasons: tuple[Reason, ...]
    latency_us: float
    threshold: float | None = None
    decision: Decision | None = None
    field_types: Mapping[str, FieldType] = field(default_factory=dict)


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
