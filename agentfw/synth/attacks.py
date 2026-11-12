"""Attack injector (spec 4.6): pure functions turning a benign session into a labeled attack.

Each family is an ``AttackFn(session, rng, ctx) -> AttackResult | None`` (``None`` = the family
does not apply to this session). Inputs are never mutated: ``RawCall`` is frozen and every mutated
call is a ``model_copy`` with deep-copied args. Surviving calls keep their ``envelope_id``.

Families (seven attacks plus the ``benign_unusual`` control, below): ``magnitude_spike`` (scale a
money/count field of a write by ``m``, log-uniform on [3, 100], ``Label.strength = m``),
``prereq_skip`` (drop every read before a write whose id args
flow from those reads and fabricate those ids) and ``novel_target`` (swap an email / stable
categorical / external destination id of a write for a never-seen value; emails use ``.test``,
D-016).

``scope_blowup`` grows a write's ``record_ids`` / ``to`` list to N (log-uniform [100, 10000]; new
ids share the prefix, new addresses reuse the list's domains). ``velocity_burst`` duplicates a write
r in [10, 60] times at uniform offsets within w in [60, 300] s (D-020); only the duplicates are
labeled. ``slow_drift`` scales one money/count field by ``(1+eps)^i`` in each of k consecutive
persona sessions (eps in [0.03, 0.10], k in [15, 40] clamped to the sessions that remain; the
attack applies only if at least 15 remain); ``Label.step`` is 1-BASED (``i`` for the i-th drifted
session, the unit ``bench.metrics.drift_delay`` counts) and ``strength = eps``. ``exfil_volume``
sets a read's ``item_count`` to ``round(V * median)`` (V log-uniform [10, 100]) and labels the
read, the call right after it (which carries ``prev_response``, §2.3 / D-004) and the first
outbound write (mail / slack) after it, appending one at the end of the session if missing.
The "persona median" is the median of the tool's catalog ``response.item_count`` dist (personas
do not override response dists): lognormal ``exp(mu)``, poisson ``lam`` (approximation),
gamma Wilson-Hilferty ``k*theta*(1-1/(9k))**3`` (approximation), uniform midpoint, constant value.
``benign_unusual`` is the control, not an attack: it is NOT drawn by the attack choice, it fires
on its own draw at ``control_rate`` and labels the touched call BENIGN with
``control_type="benign_unusual"``: a refund of 5x the persona's p99 amount (lognormal
``exp(mu + 2.326 sigma)``) with ``reason="manager_approved"`` and a prior lookup that produced its
charge id. (Quarter-end mass-update sessions are scheduled by the sampler and labeled in generate.)

A field is "money/count" when its catalog ``ArgSpec.shape`` is ``money``/``count`` or it is a
numeric arg named ``amount``/``count``/``qty``/``total``.

``magnitude_mode="far"`` (Amendment A1.4): ``magnitude_spike`` makes the same two draws but lands
the value at or beyond the field's far bound, ``magnitude_far_mult`` x the persona's own benign
p99 (``far_bounds``; 5 x p99 is the level of the benign-labelled control). A session where no
candidate can get there within ``M_MAX`` stays benign, consumes its attack id and is counted.

``inject`` draws only from the attack RNG it is handed, so persona streams, and therefore every
non-attacked session, are unaffected by the attack rate.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import orjson

from agentfw.core.envelope import JsonValue, Label, RawCall, ResponseMeta
from agentfw.core.ids import new_ulid
from agentfw.core.patterns import email_domain, is_id_like
from agentfw.synth.dists import IdPool, new_id
from agentfw.synth.grammar import plan_args
from agentfw.synth.schema import ArgSpec, Catalog, DistSpec, Persona

if TYPE_CHECKING:
    from agentfw.synth.generate import GenerateConfig

__all__ = [
    "ATTACKS",
    "AttackContext",
    "AttackFn",
    "AttackResult",
    "BURST_MAX",
    "BURST_MIN",
    "BURST_W_MAX_S",
    "BURST_W_MIN_S",
    "CONTROL_P99_MULT",
    "M_MAX",
    "M_MIN",
    "SCOPE_MAX",
    "SCOPE_MIN",
    "V_MAX",
    "V_MIN",
    "applicable",
    "blow_up_list",
    "burst_duplicates",
    "eligible",
    "fabricate_id",
    "far_bounds",
    "inject",
    "mutate_replace",
    "mutate_scale",
    "novel_value",
]

M_MIN, M_MAX = 3.0, 100.0
SCOPE_MIN, SCOPE_MAX = 100, 10_000
BURST_MIN, BURST_MAX = 10, 60
BURST_W_MIN_S, BURST_W_MAX_S = 60.0, 300.0  # D-020
EPS_MIN, EPS_MAX = 0.03, 0.10
K_MIN, K_MAX = 15, 40
V_MIN, V_MAX = 10.0, 100.0
CONTROL = "benign_unusual"
CONTROL_P99_MULT = 5.0
_Z99 = 2.326
_OUTBOUND = ("mail.send_email", "slack.post_message")
_MONEY_COUNT_NAMES = frozenset({"amount", "count", "qty", "total"})
_RAND_ALPHABET = np.array(list("abcdefghijklmnopqrstuvwxyz0123456789"))


@dataclass(slots=True)
class AttackContext:
    catalog: Catalog
    persona: Persona
    pools: dict[str, IdPool]
    attack_id: str
    persona_sessions: Sequence[list[RawCall]]
    session_index: int
    # ``magnitude_mode="far"`` (A1.4): ``(tool, arg) -> far bound`` of this persona's money/count
    # write fields (``far_bounds``); ``None`` = the spec 4.6 construction (mode "own").
    far: Mapping[tuple[str, str], float] | None = None


@dataclass(slots=True)
class AttackResult:
    session: list[RawCall]
    labels: list[Label]  # attack (or control) labels only; the caller fills in benign
    # Replacements for persona sessions ``index+1 .. index+len(extra_sessions)`` (multi-session
    # attacks such as slow_drift); ``session`` replaces the session at ``index``.
    extra_sessions: list[list[RawCall]] = field(default_factory=list)


AttackFn = Callable[[list[RawCall], np.random.Generator, AttackContext], AttackResult | None]
# ``(agent_id, attack_id, persona_sessions, session_index) -> context``. ``inject`` calls it once
# per attack or control it tries. In mode "far" with ``magnitude_spike`` enabled it also calls it
# once per persona before that persona's first session, with ``attack_id=""`` and index 0, only
# to read the persona and catalog: a factory must not count on one call per attack.
CtxFactory = Callable[[str, str, Sequence[list[RawCall]], int], AttackContext]


# --- single-call mutators (reused by the sandbox, Task 2.7) -------------------------------------


def _walk_to_parent(args: dict[str, Any], path: str) -> tuple[Any, str | int]:
    parts = path.split(".")
    node: Any = args
    for p in parts[:-1]:
        node = node[int(p)] if isinstance(node, list) else node[p]
    last = parts[-1]
    return node, (int(last) if isinstance(node, list) else last)


def _with_args(raw: RawCall, args: dict[str, JsonValue]) -> RawCall:
    return raw.model_copy(update={"args": args})


def mutate_replace(raw: RawCall, path: str, value: JsonValue) -> RawCall:
    """Copy of ``raw`` with the arg at dotted ``path`` (list indices as digits) set to ``value``."""
    args: dict[str, Any] = copy.deepcopy(raw.args)
    parent, key = _walk_to_parent(args, path)
    parent[key] = value
    return _with_args(raw, args)


def _scaled(v: int | float, factor: float) -> int | float:
    """``v x factor`` as a spike writes it: a whole number for an int field, cents otherwise."""
    return int(round(v * factor)) if isinstance(v, int) else round(v * factor, 2)


def mutate_scale(raw: RawCall, path: str, factor: float) -> RawCall:
    """Copy of ``raw`` with the numeric arg at ``path`` multiplied by ``factor``."""
    args: dict[str, Any] = copy.deepcopy(raw.args)
    parent, key = _walk_to_parent(args, path)
    v = parent[key]
    if isinstance(v, bool) or not isinstance(v, int | float):
        raise TypeError(f"{raw.tool_id}.{path} is not numeric: {v!r}")
    parent[key] = _scaled(v, factor)
    return _with_args(raw, args)


def fabricate_id(prefix: str, rng: np.random.Generator) -> str:
    """An id-like string with ``prefix`` that is (with overwhelming probability) unseen."""
    return new_id(prefix, rng)


def _rand(rng: np.random.Generator, n: int = 8) -> str:
    return "".join(_RAND_ALPHABET[rng.integers(0, len(_RAND_ALPHABET), size=n)])


def novel_value(
    kind: Literal["email", "categorical", "id"], rng: np.random.Generator, prefix: str | None = None
) -> str:
    """A never-seen value: email at ``x-<rand>.test`` (D-016), ``x-<rand>`` or a fabricated id."""
    if kind == "email":
        return f"user{int(rng.integers(0, 1000))}@x-{_rand(rng)}.test"
    if kind == "categorical":
        return f"x-{_rand(rng)}"
    return fabricate_id(prefix or "nx_", rng)


# --- catalog introspection ----------------------------------------------------------------------


def _is_money_count(name: str, a: ArgSpec) -> bool:
    return a.shape in ("money", "count") or (a.type == "numeric" and name in _MONEY_COUNT_NAMES)


def _vocab(a: ArgSpec) -> list[str] | None:
    return a.vocab or (a.dist.vocab if a.dist is not None else None)


def _writes(persona: Persona, catalog: Catalog) -> list[str]:
    return [t for t in persona.tools if t in catalog and catalog[t].kind == "write"]


def _flows_from_read(a: ArgSpec, catalog: Catalog) -> bool:
    if a.type != "id" or a.source is None or a.source == "external":
        return False
    src = catalog.get(a.source.rsplit(".", 1)[0])
    return src is not None and src.kind == "read"


def _novel_kind(a: ArgSpec) -> str | None:
    if a.type in ("email", "list_of_email"):
        return "email"
    if a.type == "categorical" and _vocab(a):
        return "categorical"
    if a.type == "id" and a.source == "external" and a.prefix:
        return "id"
    return None


def _any_write_arg(
    test: Callable[[ArgSpec, Catalog, str], bool],
) -> Callable[[Persona, Catalog], bool]:
    return lambda persona, catalog: any(
        test(a, catalog, n) for t in _writes(persona, catalog) for n, a in catalog[t].args.items()
    )


def _is_list_scope(a: ArgSpec) -> bool:
    return (a.type == "list_of_id" and a.prefix is not None) or a.type == "list_of_email"


def _median(d: DistSpec) -> float | None:
    """Median (or a documented approximation) of a count dist; ``None`` if not computable."""
    m: float | None
    if d.family == "lognormal" and d.mu is not None:
        m = math.exp(d.mu)
    elif d.family == "poisson" and d.lam is not None:
        m = d.lam  # approximation: median ~ lam
    elif d.family == "gamma" and d.shape is not None and d.scale is not None:
        k = d.shape  # Wilson-Hilferty approximation
        m = k * d.scale * (1 - 1 / (9 * k)) ** 3
    elif d.family == "uniform" and d.min is not None and d.max is not None:
        m = (d.min + d.max) / 2
    elif (
        d.family == "constant"
        and isinstance(d.value, int | float)
        and not isinstance(d.value, bool)
    ):
        m = float(d.value)
    else:
        return None
    if d.min is not None:
        m = max(m, d.min)
    if d.max is not None:
        m = min(m, d.max)
    return m if m > 0 else None


def _item_median(tool: str, catalog: Catalog) -> float | None:
    spec = catalog.get(tool)
    if spec is None or spec.kind != "read" or spec.response.item_count is None:
        return None
    return _median(spec.response.item_count)


def _outbound_tools(persona: Persona, catalog: Catalog) -> list[str]:
    return [t for t in _OUTBOUND if t in _writes(persona, catalog)]


def _p99(persona: Persona, tool: str, name: str, a: ArgSpec) -> float | None:
    d = persona.fields.get(f"{tool}.{name}") or a.dist
    if d is None or d.family != "lognormal" or d.mu is None or d.sigma is None:
        return None
    return math.exp(d.mu + _Z99 * d.sigma)


def _control_args(tool: str, persona: Persona, catalog: Catalog) -> tuple[str, float, str] | None:
    """``(amount arg, persona p99, reason arg)`` if ``tool`` is a refund-like write, else None."""
    spec = catalog.get(tool)
    if spec is None or spec.kind != "write":
        return None
    reason = next(
        (
            n
            for n, a in spec.args.items()
            if a.type == "categorical" and "manager_approved" in (_vocab(a) or [])
        ),
        None,
    )
    for n, a in spec.args.items():
        if reason is not None and _is_money_count(n, a) and (p := _p99(persona, tool, n, a)):
            return n, p, reason
    return None


def _eligible_control(persona: Persona, catalog: Catalog) -> bool:
    return any(
        _control_args(t, persona, catalog) is not None
        and any(_flows_from_read(a, catalog) for a in catalog[t].args.values())
        for t in _writes(persona, catalog)
    )


def _eligible_exfil(persona: Persona, catalog: Catalog) -> bool:
    return bool(_outbound_tools(persona, catalog)) and any(
        _item_median(t, catalog) is not None for t in persona.tools if t in catalog
    )


_ELIGIBLE: dict[str, Callable[[Persona, Catalog], bool]] = {
    "magnitude_spike": _any_write_arg(lambda a, cat, name: _is_money_count(name, a)),
    "prereq_skip": _any_write_arg(lambda a, cat, name: _flows_from_read(a, cat)),
    "novel_target": _any_write_arg(lambda a, cat, name: _novel_kind(a) is not None),
    "scope_blowup": _any_write_arg(lambda a, cat, name: _is_list_scope(a)),
    "velocity_burst": lambda persona, catalog: bool(_writes(persona, catalog)),
    "slow_drift": _any_write_arg(lambda a, cat, name: _is_money_count(name, a)),
    "exfil_volume": _eligible_exfil,
    CONTROL: _eligible_control,
}


def eligible(family: str, persona: Persona, catalog: Catalog) -> bool:
    """True iff the persona has a write tool this family can mutate (``KeyError`` if unknown)."""
    return _ELIGIBLE[family](persona, catalog)


# --- families -----------------------------------------------------------------------------------
# Each family has a pure candidate finder (no RNG) shared by ``applicable`` and the mutation.


def _is_num(v: object) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool)


def _label(
    ctx: AttackContext,
    family: str,
    env: str,
    strength: float | None = None,
    step: int | None = None,
) -> Label:
    return Label(
        envelope_id=env,
        label="attack",
        attack_family=family,
        injection_level="action",
        attack_id=ctx.attack_id,
        strength=strength,
        step=step,
    )


def _renumber(calls: list[RawCall]) -> list[RawCall]:
    return [c.model_copy(update={"seq_no": k}) for k, c in enumerate(calls)]


def _replace_call(session: list[RawCall], i: int, new: RawCall) -> list[RawCall]:
    return [*session[:i], new, *session[i + 1 :]]


def _kind(ctx: AttackContext, c: RawCall) -> str | None:
    spec = ctx.catalog.get(c.tool_id)
    return spec.kind if spec is not None else None


def _magnitude_cands(session: list[RawCall], ctx: AttackContext) -> list[tuple[int, str]]:
    return [
        (i, name)
        for i, c in enumerate(session)
        if _kind(ctx, c) == "write"
        for name, a in ctx.catalog[c.tool_id].args.items()
        if _is_money_count(name, a) and _is_num(c.args.get(name)) and float(c.args[name]) > 0  # type: ignore[arg-type]
    ]


def far_bounds(
    sessions: Sequence[list[RawCall]],
    persona: Persona,
    catalog: Catalog,
    mult: float = CONTROL_P99_MULT,
) -> dict[tuple[str, str], float]:
    """``(tool, arg) -> mult x P99`` of each money/count arg of a write tool (A1.4), from one
    persona's sessions BEFORE any injection. ``P99`` is the 99th percentile (linear interpolation)
    of the arg's positive numeric values; on the arg the ``benign_unusual`` control mutates it is
    at least the persona's declared p99, the value the control uses. Pure: reads the persona's own
    benign values and the control's constant, nothing else."""
    values: dict[tuple[str, str], list[float]] = {}
    for session in sessions:
        for c in session:
            spec = catalog.get(c.tool_id)
            if spec is None or spec.kind != "write":
                continue
            for name, a in spec.args.items():
                v = c.args.get(name)
                if _is_money_count(name, a) and _is_num(v) and float(v) > 0:  # type: ignore[arg-type]
                    values.setdefault((c.tool_id, name), []).append(float(v))  # type: ignore[arg-type]
    out: dict[tuple[str, str], float] = {}
    for (tool, name), xs in values.items():
        p99 = float(np.percentile(xs, 99.0))
        control = _control_args(tool, persona, catalog)
        if control is not None and control[0] == name:
            p99 = max(p99, control[1])
        out[(tool, name)] = mult * p99
    return out


def _far_feasible(session: list[RawCall], ctx: AttackContext) -> list[tuple[int, str]]:
    """Spike candidates whose own value x ``M_MAX`` reaches the field's far bound (A1.4 step 1).

    "Value x ``M_MAX``" is the value the largest multiplier really writes (``_scaled``: whole
    numbers or cents), not the raw float product: ``4.11 * 100.0`` is ``411.00000000000006``, but
    no spike of 4.11 within ``M_MAX`` lands above 411.0. A bound in that one-ulp gap is out of
    reach, so the candidate is infeasible (and ``0.29 * 100.0``, one ulp BELOW 29, reaches 29.0)."""
    far = ctx.far or {}
    return [
        (i, name)
        for i, name in _magnitude_cands(session, ctx)
        if (bound := far.get((session[i].tool_id, name))) is not None
        and _scaled(session[i].args[name], M_MAX) >= bound  # type: ignore[arg-type]
    ]


def _far_fix(old: int | float, bound: float) -> int | float:
    """The lowest value of ``old``'s kind at or beyond ``bound`` (A1.4 step 3): ``ceil(bound)`` for
    an int field, ``bound`` rounded up to cents otherwise. For a feasible candidate it is never
    above ``_scaled(old, M_MAX)``, which is itself such a value at or beyond the bound."""
    if isinstance(old, int):
        return math.ceil(bound)
    fix = math.ceil(bound * 100.0) / 100.0
    # ``bound * 100.0`` can round DOWN to a whole number of cents (a bound one ulp above a cent
    # value); the quotient is then that cent value, one ulp short of the bound.
    return fix if fix >= bound else round(fix + 0.01, 2)


def magnitude_spike(
    session: list[RawCall], rng: np.random.Generator, ctx: AttackContext
) -> AttackResult | None:
    """Mode "own" (``ctx.far is None``, spec 4.6): multiply the picked call's own value by ``m``,
    log-uniform on ``[M_MIN, M_MAX]``. Mode "far" (A1.4): the same two draws in the same order;
    the spiked call is the first feasible candidate in cyclic order from the picked one, and the
    drawn ``ln m`` is mapped onto ``[max(M_MIN, bound / value), M_MAX]`` so the value lands at or
    beyond the far bound. ``None`` (after both draws) when no candidate is feasible.
    ``Label.strength`` is the realized multiplier in both modes."""
    cands = _magnitude_cands(session, ctx)
    if not cands:
        return None
    k = int(rng.integers(len(cands)))
    log_m = rng.uniform(np.log(M_MIN), np.log(M_MAX))
    if ctx.far is None:
        i, name = cands[k]
        m = float(np.clip(np.exp(log_m), M_MIN, M_MAX))
        new = mutate_scale(session[i], name, m)
    else:
        feasible = set(_far_feasible(session, ctx))
        pick = next((c for c in [*cands[k:], *cands[:k]] if c in feasible), None)
        if pick is None:
            return None
        i, name = pick
        old = session[i].args[name]
        bound = ctx.far[(session[i].tool_id, name)]
        lo = math.log(max(M_MIN, bound / float(old)))  # type: ignore[arg-type]
        t = (log_m - math.log(M_MIN)) / (math.log(M_MAX) - math.log(M_MIN))
        m = min(math.exp(lo + t * (math.log(M_MAX) - lo)), M_MAX)
        new = mutate_scale(session[i], name, m)
        if float(new.args[name]) < bound:  # type: ignore[arg-type]  # rounding fell short
            new = mutate_replace(session[i], name, _far_fix(old, bound))  # type: ignore[arg-type]
    old_v, new_v = float(session[i].args[name]), float(new.args[name])  # type: ignore[arg-type]
    return AttackResult(
        _replace_call(session, i, new),
        [_label(ctx, "magnitude_spike", new.envelope_id, new_v / old_v)],  # realized multiplier
    )


def _str_values(v: object) -> Iterator[str]:
    if isinstance(v, str):
        yield v
    elif isinstance(v, list):
        for x in v:
            yield from _str_values(x)
    elif isinstance(v, dict):
        for x in v.values():
            yield from _str_values(x)


def _prereq_cands(session: list[RawCall], ctx: AttackContext) -> list[tuple[int, list[str]]]:
    cands: list[tuple[int, list[str]]] = []
    for i, c in enumerate(session):
        if _kind(ctx, c) != "write":
            continue
        read_ids = {
            x
            for e in session[:i]
            if _kind(ctx, e) == "read"
            for x in (e.response_meta.ids if e.response_meta else ())
        }
        names = [
            n
            for n, a in ctx.catalog[c.tool_id].args.items()
            if _flows_from_read(a, ctx.catalog) and c.args.get(n) in read_ids
        ]
        if names:
            cands.append((i, names))
    return cands


def _known_before(calls: list[RawCall], upto: int) -> set[str]:
    known: set[str] = set()
    for e in calls[:upto]:
        known.update(_str_values(e.args))
        if e.response_meta:
            known.update(e.response_meta.ids)
    return known


def prereq_skip(
    session: list[RawCall], rng: np.random.Generator, ctx: AttackContext
) -> AttackResult | None:
    cands = _prereq_cands(session, ctx)
    if not cands:
        return None
    i, names = cands[int(rng.integers(len(cands)))]
    target = session[i]
    spec = ctx.catalog[target.tool_id]
    for n in names:
        target = mutate_replace(target, n, fabricate_id(spec.args[n].prefix or "nx_", rng))
    kept: list[RawCall] = []
    for j, c in enumerate(session):
        if j < i and _kind(ctx, c) == "read":
            continue
        kept.append(target if j == i else c)
    out = [c.model_copy(update={"seq_no": k}) for k, c in enumerate(kept)]
    t = next(k for k, c in enumerate(out) if c.envelope_id == target.envelope_id)
    labels = [_label(ctx, "prereq_skip", target.envelope_id)]
    # A later call is part of the attack if one of its id-like args was produced by an earlier call
    # of the ORIGINAL session and no longer is after the deletion. Values that were never visible
    # in the benign session (e.g. an email that did not flow from a visible read) do not count.
    orig_index = {c.envelope_id: j for j, c in enumerate(session)}
    for k in range(t + 1, len(out)):
        was = _known_before(session, orig_index[out[k].envelope_id])
        now = _known_before(out, k)
        if any(is_id_like(v) and v in was and v not in now for v in _str_values(out[k].args)):
            labels.append(_label(ctx, "prereq_skip", out[k].envelope_id))
    return AttackResult(out, labels)


def _novel_cands(session: list[RawCall], ctx: AttackContext) -> list[tuple[int, str, str]]:
    """``(call index, arg name, kind)``; a list_of_email candidate is resolved to an index later."""
    cands: list[tuple[int, str, str]] = []
    for i, c in enumerate(session):
        if _kind(ctx, c) != "write":
            continue
        for name, a in ctx.catalog[c.tool_id].args.items():
            kind, v = _novel_kind(a), c.args.get(name)
            if kind is None or v is None:
                continue
            if a.type == "list_of_email":
                if isinstance(v, list) and v:
                    cands.append((i, name, kind))
            elif isinstance(v, str):
                cands.append((i, name, kind))
    return cands


def novel_target(
    session: list[RawCall], rng: np.random.Generator, ctx: AttackContext
) -> AttackResult | None:
    cands = _novel_cands(session, ctx)
    if not cands:
        return None
    i, name, kind = cands[int(rng.integers(len(cands)))]
    a = ctx.catalog[session[i].tool_id].args[name]
    path = name
    if a.type == "list_of_email":
        n = len(session[i].args[name])  # type: ignore[arg-type]
        path = f"{name}.{int(rng.integers(n))}"
    value = novel_value(kind, rng, a.prefix)  # type: ignore[arg-type]
    while kind == "categorical" and value in (_vocab(a) or []):
        value = novel_value("categorical", rng)  # pragma: no cover - x- prefix never collides
    new = mutate_replace(session[i], path, value)
    return AttackResult(
        _replace_call(session, i, new), [_label(ctx, "novel_target", new.envelope_id)]
    )


def blow_up_list(
    orig: Sequence[str], ids: bool, n: int, prefix: str, rng: np.random.Generator
) -> list[JsonValue]:
    """``orig`` grown to ``n`` entries: fabricated ids with ``prefix`` (``ids``), else addresses
    reusing ``orig``'s email domains (D-016). Shared by ``scope_blowup`` and the sandbox
    injector."""
    m = n - len(orig)
    extra: list[str]
    if ids:
        extra = [fabricate_id(prefix, rng) for _ in range(m)]
    else:
        domains = sorted({d for x in orig if (d := email_domain(x))})
        users = rng.choice(1_000_000, size=m, replace=False)
        dom = rng.integers(len(domains), size=m)
        extra = [f"user{int(u):06d}@{domains[int(d)]}" for u, d in zip(users, dom, strict=True)]
    return [*orig, *extra]


def burst_duplicates(orig: RawCall, r: int, w_ms: int, rng: np.random.Generator) -> list[RawCall]:
    """``r`` copies of ``orig`` at sorted uniform offsets in (0, ``w_ms``] ms. Shared by
    ``velocity_burst`` and the sandbox injector."""
    offsets = np.sort(rng.integers(1, w_ms + 1, size=r))
    dups = []
    for off in offsets:
        ts = orig.ts_unix_ms + int(off)
        dups.append(
            orig.model_copy(
                update={
                    "envelope_id": new_ulid(ts, rng),
                    "ts_unix_ms": ts,
                    "args": copy.deepcopy(orig.args),
                }
            )
        )
    return dups


def _scope_cands(session: list[RawCall], ctx: AttackContext) -> list[tuple[int, str]]:
    cands: list[tuple[int, str]] = []
    for i, c in enumerate(session):
        if _kind(ctx, c) != "write":
            continue
        for name, a in ctx.catalog[c.tool_id].args.items():
            v = c.args.get(name)
            if not _is_list_scope(a) or not isinstance(v, list) or not v:
                continue
            if not all(isinstance(x, str) for x in v):
                continue
            if a.type == "list_of_email" and not any(email_domain(str(x)) for x in v):
                continue
            cands.append((i, name))
    return cands


def scope_blowup(
    session: list[RawCall], rng: np.random.Generator, ctx: AttackContext
) -> AttackResult | None:
    cands = _scope_cands(session, ctx)
    if not cands:
        return None
    i, name = cands[int(rng.integers(len(cands)))]
    a = ctx.catalog[session[i].tool_id].args[name]
    orig = [str(x) for x in session[i].args[name]]  # type: ignore[union-attr]
    n = int(round(np.exp(rng.uniform(np.log(SCOPE_MIN), np.log(SCOPE_MAX)))))
    n = max(n, len(orig) + 1)
    new = mutate_replace(
        session[i], name, blow_up_list(orig, a.type == "list_of_id", n, a.prefix or "nx_", rng)
    )
    return AttackResult(
        _replace_call(session, i, new), [_label(ctx, "scope_blowup", new.envelope_id, float(n))]
    )


def _velocity_cands(session: list[RawCall], ctx: AttackContext) -> list[int]:
    return [i for i, c in enumerate(session) if _kind(ctx, c) == "write"]


def velocity_burst(
    session: list[RawCall], rng: np.random.Generator, ctx: AttackContext
) -> AttackResult | None:
    cands = _velocity_cands(session, ctx)
    if not cands:
        return None
    orig = session[cands[int(rng.integers(len(cands)))]]
    r = int(rng.integers(BURST_MIN, BURST_MAX + 1))
    w_ms = int(round(rng.uniform(BURST_W_MIN_S, BURST_W_MAX_S) * 1000))
    dups = burst_duplicates(orig, r, w_ms, rng)
    merged = _renumber(sorted([*session, *dups], key=lambda c: c.ts_unix_ms))  # stable: dups last
    return AttackResult(
        merged, [_label(ctx, "velocity_burst", d.envelope_id, float(r)) for d in dups]
    )


def _drift_cands(session: list[RawCall], ctx: AttackContext) -> list[tuple[int, str]]:
    if ctx.session_index + K_MIN > len(ctx.persona_sessions):
        return []
    return _magnitude_cands(session, ctx)


def slow_drift(
    session: list[RawCall], rng: np.random.Generator, ctx: AttackContext
) -> AttackResult | None:
    cands = _drift_cands(session, ctx)
    if not cands:
        return None
    i, name = cands[int(rng.integers(len(cands)))]
    tool = session[i].tool_id
    eps = float(rng.uniform(EPS_MIN, EPS_MAX))
    remaining = len(ctx.persona_sessions) - ctx.session_index
    k = int(rng.integers(K_MIN, min(K_MAX, remaining) + 1))
    window = [session, *ctx.persona_sessions[ctx.session_index + 1 : ctx.session_index + k]]
    out: list[list[RawCall]] = []
    labels: list[Label] = []
    for step, s in enumerate(window, start=1):  # Label.step is 1-based
        factor = (1 + eps) ** step
        new_s = list(s)
        for j, c in enumerate(s):
            if c.tool_id == tool and _is_num(c.args.get(name)) and float(c.args[name]) > 0:  # type: ignore[arg-type]
                new_s[j] = mutate_scale(c, name, factor)
                labels.append(_label(ctx, "slow_drift", c.envelope_id, eps, step))
        out.append(new_s)
    return AttackResult(out[0], labels, out[1:])


def _exfil_cands(session: list[RawCall], ctx: AttackContext) -> list[int]:
    can_append = bool(_outbound_tools(ctx.persona, ctx.catalog))
    return [
        i
        for i, c in enumerate(session)
        if c.response_meta is not None
        and c.response_meta.item_count is not None
        and _item_median(c.tool_id, ctx.catalog) is not None
        and (can_append or _next_outbound(session, i, ctx) is not None)
    ]


def _next_outbound(session: list[RawCall], i: int, ctx: AttackContext) -> int | None:
    return next((j for j in range(i + 1, len(session)) if session[j].tool_id in _OUTBOUND), None)


def exfil_volume(
    session: list[RawCall], rng: np.random.Generator, ctx: AttackContext
) -> AttackResult | None:
    cands = _exfil_cands(session, ctx)
    if not cands:
        return None
    i = cands[int(rng.integers(len(cands)))]
    v = float(np.clip(np.exp(rng.uniform(np.log(V_MIN), np.log(V_MAX))), V_MIN, V_MAX))
    read = session[i]
    assert read.response_meta is not None
    median = _item_median(read.tool_id, ctx.catalog)
    assert median is not None
    meta = read.response_meta.model_copy(update={"item_count": round(v * median)})
    out = _replace_call(session, i, read.model_copy(update={"response_meta": meta}))
    j = _next_outbound(out, i, ctx)
    if j is None:
        tools = _outbound_tools(ctx.persona, ctx.catalog)
        tool = tools[int(rng.integers(len(tools)))]
        spec = ctx.catalog[tool]
        ts = out[-1].ts_unix_ms + int(rng.integers(500, 3000))
        args = plan_args(tool, ctx.persona, ctx.catalog, rng, ctx.pools)
        produced = {k: spec.produced[k] for k in spec.response.produces_ids}
        ids = tuple(new_id(pa.prefix or "id_", rng) for pa in produced.values())
        size = 64 + len(orjson.dumps(list(ids)))
        out.append(
            read.model_copy(
                update={
                    "envelope_id": new_ulid(ts, rng),
                    "seq_no": len(out),
                    "ts_unix_ms": ts,
                    "tool_id": tool,
                    "args": args,
                    "response_meta": ResponseMeta(bytes=size, ids=ids),
                }
            )
        )
        j = len(out) - 1
    # §2.3 / D-004: the inflated item_count is scored on the call right after the read, so that
    # call is part of the attack even when it is not the outbound write (Task 1.15, decision 008).
    hit = [i, *([i + 1] if i + 1 < j else []), j]
    return AttackResult(out, [_label(ctx, "exfil_volume", out[k].envelope_id, v) for k in hit])


def _control_cands(session: list[RawCall], ctx: AttackContext) -> list[tuple[int, str]]:
    """Refund-like writes whose flowed id arg came from a prior call to its SOURCE tool (e.g.
    ``stripe.retrieve_charge``), not merely from some earlier read that exposed the id."""
    cands: list[tuple[int, str]] = []
    for i, names in _prereq_cands(session, ctx):
        spec = ctx.catalog[session[i].tool_id]
        if _control_args(session[i].tool_id, ctx.persona, ctx.catalog) is None:
            continue
        before = {c.tool_id for c in session[:i]}
        if any(
            (src := spec.args[n].source) is not None and src.rsplit(".", 1)[0] in before
            for n in names
        ):
            cands.append((i, ""))
    return cands


def benign_unusual(
    session: list[RawCall], rng: np.random.Generator, ctx: AttackContext
) -> AttackResult | None:
    cands = _control_cands(session, ctx)
    if not cands:
        return None
    i, _ = cands[int(rng.integers(len(cands)))]
    c = session[i]
    amount, p99, reason = _control_args(c.tool_id, ctx.persona, ctx.catalog) or ("", 0.0, "")
    new = mutate_replace(c, amount, round(CONTROL_P99_MULT * p99, 2))
    new = mutate_replace(new, reason, "manager_approved")
    lab = Label(envelope_id=new.envelope_id, label="benign", control_type="benign_unusual")
    return AttackResult(_replace_call(session, i, new), [lab])


ATTACKS: dict[str, AttackFn] = {
    "magnitude_spike": magnitude_spike,
    "prereq_skip": prereq_skip,
    "novel_target": novel_target,
    "scope_blowup": scope_blowup,
    "velocity_burst": velocity_burst,
    "slow_drift": slow_drift,
    "exfil_volume": exfil_volume,
    CONTROL: benign_unusual,
}

_CANDS: dict[str, Callable[[list[RawCall], AttackContext], Sequence[object]]] = {
    "magnitude_spike": _magnitude_cands,
    "prereq_skip": _prereq_cands,
    "novel_target": _novel_cands,
    "scope_blowup": _scope_cands,
    "velocity_burst": _velocity_cands,
    "slow_drift": _drift_cands,
    "exfil_volume": _exfil_cands,
    CONTROL: _control_cands,
}


def applicable(family: str, session: list[RawCall], ctx: AttackContext) -> bool:
    """Pure check: would ``family`` find something to mutate in ``session``?"""
    return bool(_CANDS[family](session, ctx))


# --- schedule -----------------------------------------------------------------------------------


def inject(
    sessions_by_persona: dict[str, list[list[RawCall]]],
    cfg: GenerateConfig,
    rng: np.random.Generator,
    ctx_factory: CtxFactory,
    *,
    stats: dict[str, int] | None = None,
) -> tuple[dict[str, list[list[RawCall]]], list[Label]]:
    """Attack persona sessions with probability ``cfg.attack_rate`` (after warmup), and inject
    ``benign_unusual`` controls at ``cfg.control_rate`` when that family is enabled.

    ``ctx_factory(agent_id, attack_id, persona_sessions, session_index)`` builds the context (it
    knows the catalog, personas and pools). Personas are visited in sorted ``agent_id`` order and
    sessions in time order; only ``rng`` (the attack stream) is ever drawn from. The family is
    uniform over the families applicable to THAT session (persona-eligible and with something to
    mutate); a session where none applies stays benign. ``benign_unusual`` is never chosen as an
    attack: its draw is independent of attack selection (a session is a control candidate only if it
    was not attacked), and with it disabled no control draw is made (streams are unchanged).

    ``cfg.magnitude_mode == "far"`` (A1.4) with ``magnitude_spike`` enabled: each persona's far
    bounds are computed once, from its untouched sessions (one extra ``ctx_factory`` call per
    persona, with an empty attack id, gives the persona and catalog), and handed to the attack on
    a new context. A chosen ``magnitude_spike`` with no feasible candidate leaves the session
    benign: its attack id is consumed and the session gets no control, exactly as if the spike had
    taken it, so the attack stream makes the draws it makes in mode "own" and every other attack
    and control is unchanged. Only that case is a dropped spike; a ``None`` from any other family
    or mode (none returns one once ``applicable`` found a candidate) falls through to the control.
    ``stats``, when given, receives ``magnitude_first_pass`` (spikes injected),
    ``magnitude_dropped`` (those left benign) and ``magnitude_topup`` (0).
    """
    families = list(dict.fromkeys(cfg.families))
    unknown = sorted(set(families) - set(ATTACKS))
    if unknown:
        raise ValueError(f"unknown attack family: {unknown}")
    controls = CONTROL in families
    attack_fams = [f for f in families if f != CONTROL]
    far_spikes = cfg.magnitude_mode == "far" and "magnitude_spike" in attack_fams
    out: dict[str, list[list[RawCall]]] = {}
    labels: list[Label] = []
    n_dropped = 0
    for agent in sorted(sessions_by_persona):
        sessions = list(sessions_by_persona[agent])
        far: dict[tuple[str, str], float] | None = None
        if far_spikes:  # from the untouched sessions, before any injection
            base = ctx_factory(agent, "", sessions, 0)
            far = far_bounds(
                sessions_by_persona[agent], base.persona, base.catalog, cfg.magnitude_far_mult
            )
        n_attacks = n_controls = 0
        idx = cfg.warmup_sessions
        while idx < len(sessions):
            # Draw order per visited session: u_attack, then (only if the control is enabled)
            # u_control, both unconditionally; then the attack's own draws (family, params) or the
            # control's draws. A multi-session attack skips the sessions it consumed.
            u_attack = rng.random()
            u_control = rng.random() if controls else 1.0
            if u_attack < cfg.attack_rate:
                ctx = ctx_factory(agent, f"atk-{agent}-{n_attacks:05d}", sessions, idx)
                if far is not None:
                    ctx = replace(ctx, far=far)  # a new context: the factory's may be shared
                fams = [
                    f
                    for f in attack_fams
                    if eligible(f, ctx.persona, ctx.catalog) and applicable(f, sessions[idx], ctx)
                ]
                if fams:
                    family = fams[int(rng.integers(len(fams)))]
                    res = ATTACKS[family](sessions[idx], rng, ctx)
                    # applicable() found a candidate, so ``None`` is a far-mode spike with no
                    # feasible one; no other family or mode returns it here.
                    if res is not None:
                        sessions[idx] = res.session
                        for j, extra in enumerate(res.extra_sessions, start=1):
                            sessions[idx + j] = extra
                        labels += res.labels
                        n_attacks += 1
                        idx += 1 + len(res.extra_sessions)
                        continue
                    if far_spikes and family == "magnitude_spike":
                        # Dropped spike (A1.4 step 4): the session stays benign and takes no
                        # control, and its attack id is consumed, so later attacks keep their
                        # mode-"own" ids.
                        n_attacks += 1
                        n_dropped += 1
                        idx += 1
                        continue
            if u_control < cfg.control_rate:
                ctx = ctx_factory(agent, f"ctl-{agent}-{n_controls:05d}", sessions, idx)
                if eligible(CONTROL, ctx.persona, ctx.catalog) and applicable(
                    CONTROL, sessions[idx], ctx
                ):
                    res = ATTACKS[CONTROL](sessions[idx], rng, ctx)
                    if res is not None:
                        sessions[idx] = res.session
                        labels += res.labels
                        n_controls += 1
            idx += 1
        out[agent] = sessions
    if stats is not None:
        spikes = {lab.attack_id for lab in labels if lab.attack_family == "magnitude_spike"}
        stats.update(
            magnitude_first_pass=len(spikes), magnitude_dropped=n_dropped, magnitude_topup=0
        )
    return out, labels
