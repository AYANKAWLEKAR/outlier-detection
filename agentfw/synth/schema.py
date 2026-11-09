"""Declarative "world" models (catalog, personas, task grammars) the trace generator samples from.

Conventions the walker (Task 0.5) relies on:

* ``ToolSpec.produces`` lists produced fields as ``entity.field`` (spec 4.2).
  ``ToolSpec.produced`` maps each of those names to an ArgSpec saying how to generate its value.
* A step binding ``<alias>.<field>`` (or a catalog source ``<server.tool>.<field>``) names the
  produced entry whose key ends in ``.<field>``; the entity part is documentation only.
* ``ResponseSpec.produces_ids`` lists produced entries that are ids to expose in
  ``response_meta.ids``.
* ``ResponseSpec.item_count`` is the single source of truth for a response's list size: the walker
  draws it once per call and reuses it as the length of every produced list field (those fields
  carry no ``len_dist`` of their own; validation rejects one that does).
* A produced ArgSpec with ``from_arg`` echoes the value of that argument of the same call.
* Persona ``fields`` precedence. A key is ``<tool>.<arg>`` or ``<tool>.<arg>.<derived>``
  with derived in ``__domain__``, ``__len__``, ``__host__``, ``__prefix__``.
  Every key must be reachable:
  - a ``sample`` binding of that (tool, arg) in one of the persona's tasks uses the override in
    preference to the catalog dist;
  - a ``<tool>.<arg>.__domain__`` override (at most one per persona) defines the domain distribution
    of ALL emails generated for that persona (default: the catalog email dist);
  - a ``<tool>.<arg>.__len__`` override on a ``list_of_email`` arg bound to a scalar source means
    the list is ``[bound value]`` plus ``len - 1`` extra persona-generated emails (``len`` drawn
    from the override).
* Benign generated email domains follow ``<word><k>.example`` (``ArgSpec.domain_suffix``);
  attack and novel values use ``.test`` (RFC 2606 only, D-016).
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

Family = Literal[
    "lognormal", "gamma", "poisson", "zipf", "categorical", "constant", "uniform", "bernoulli"
]
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
    "bernoulli": ("p",),
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
    p: float | None = None
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
    from_arg: str | None = None  # produced only: echo this call's argument
    domain_suffix: str | None = None  # email types: ".example" or ".test"


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
