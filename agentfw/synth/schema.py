"""Declarative "world" models (catalog, personas, task grammars) the trace generator samples from.

Conventions the walker (Task 0.5) relies on:

* ``ToolSpec.produces`` lists produced fields as ``entity.field`` (spec 4.2).
  ``ToolSpec.produced`` maps each of those names to an ArgSpec saying how to generate its value.
* A step binding ``<alias>.<field>`` (or a catalog source ``<server.tool>.<field>``) names the
  produced entry whose key ends in ``.<field>``; the entity part is documentation only.
* ``ResponseSpec.produces_ids`` lists produced entries that are ids to expose in
  ``response_meta.ids``.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

Family = Literal["lognormal", "gamma", "poisson", "zipf", "categorical", "constant", "uniform"]
ArgType = Literal[
    "numeric",
    "categorical",
    "id",
    "email",
    "freetext",
    "list_of_id",
    "list_of_email",
    "freetext_map",
    "bool",
]

_REQUIRED: dict[str, tuple[str, ...]] = {
    "lognormal": ("mu", "sigma"),
    "gamma": ("shape", "scale"),
    "poisson": ("lam",),
    "zipf": ("vocab_size", "s"),
    "categorical": ("vocab",),
    "constant": ("value",),
    "uniform": ("min", "max"),
}


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class DistSpec(_Model):
    family: Family
    mu: float | None = None
    sigma: float | None = None
    shape: float | None = None
    scale: float | None = None
    lam: float | None = None
    min: float | None = None
    max: float | None = None
    vocab_size: int | None = None
    s: float | None = None
    vocab: list[str] | None = None
    weights: list[float] | None = None
    dirichlet_alpha: float = 50.0
    value: float | str | bool | None = None
    round_cents: bool = False

    @model_validator(mode="after")
    def _check_params(self) -> Self:
        missing = [k for k in _REQUIRED[self.family] if getattr(self, k) is None]
        if missing:
            raise ValueError(f"{self.family} dist requires {missing}")
        if self.weights is not None and (
            self.vocab is None or len(self.weights) != len(self.vocab)
        ):
            raise ValueError("weights must match vocab length")
        return self


class ArgSpec(_Model):
    type: ArgType
    prefix: str | None = None
    source: str | None = None  # "external" or "<server.tool>.<field>"
    dist: DistSpec | None = None
    shape: Literal["money", "count"] | None = None
    vocab: list[str] | None = None
    len_dist: DistSpec | None = None


class ResponseSpec(_Model):
    item_count: DistSpec | None = None
    produces_ids: list[str] = Field(default_factory=list)


class ToolSpec(_Model):
    kind: Literal["read", "write"]
    args: dict[str, ArgSpec]
    produces: list[str] = Field(default_factory=list)
    # Generation spec for every entry of ``produces`` (same keys).
    produced: dict[str, ArgSpec] = Field(default_factory=dict)
    response: ResponseSpec = Field(default_factory=ResponseSpec)


Catalog = dict[str, ToolSpec]


class StepSpec(_Model):
    tool: str
    as_: str | None = Field(default=None, alias="as")
    args: dict[str, str] = Field(default_factory=dict)
    skippable: bool = False
    optional_prob: float = 1.0


class TaskSpec(_Model):
    name: str
    steps: list[StepSpec]
    repeat: DistSpec | None = None  # items per session


class Branches(_Model):
    skip_lookup: float = 0.05
    retry: float = 0.03
    abandon: float = 0.02
    arg_dropout: float = 0.02


class TaskWeight(_Model):
    name: str
    weight: float


class DriftSpec(_Model):
    field: str
    rate_per_session: float


class IdPoolSpec(_Model):
    prefix: str
    size: int
    s: float = 1.1


class Persona(_Model):
    agent_id: str
    tools: list[str]
    session_rate: DistSpec
    tasks: list[TaskWeight]
    fields: dict[str, DistSpec] = Field(default_factory=dict)
    drift: DriftSpec | None = None
    noise: Branches = Field(default_factory=Branches)
    id_pools: dict[str, IdPoolSpec] = Field(default_factory=dict)
    id_flow_rate: float = 1.0
    working_hours: tuple[int, int] = (8, 18)
    quarter_end_days: list[int] = Field(default_factory=list)
