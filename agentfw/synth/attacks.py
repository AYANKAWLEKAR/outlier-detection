"""Attack injector (spec 4.6): pure functions turning a benign session into a labeled attack.

Each family is an ``AttackFn(session, rng, ctx) -> AttackResult | None`` (``None`` = the family
does not apply to this session). Inputs are never mutated: ``RawCall`` is frozen and every mutated
call is a ``model_copy`` with deep-copied args. Surviving calls keep their ``envelope_id``.

Families here: ``magnitude_spike`` (scale a money/count field of a write by ``m``, log-uniform on
[3, 100], ``Label.strength = m``), ``prereq_skip`` (drop every read before a write whose id args
flow from those reads and fabricate those ids) and ``novel_target`` (swap an email / stable
categorical / external destination id of a write for a never-seen value; emails use ``.test``,
D-016).

A field is "money/count" when its catalog ``ArgSpec.shape`` is ``money``/``count`` or it is a
numeric arg named ``amount``/``count``/``qty``/``total``.

``inject`` draws only from the attack RNG it is handed, so persona streams, and therefore every
non-attacked session, are unaffected by the attack rate.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from agentfw.core.envelope import JsonValue, Label, RawCall
from agentfw.core.patterns import is_id_like
from agentfw.synth.dists import IdPool, new_id
from agentfw.synth.schema import ArgSpec, Catalog, Persona

if TYPE_CHECKING:
    from agentfw.synth.generate import GenerateConfig

__all__ = [
    "ATTACKS",
    "AttackContext",
    "AttackFn",
    "AttackResult",
    "applicable",
    "eligible",
    "fabricate_id",
    "inject",
    "mutate_replace",
    "mutate_scale",
    "novel_value",
]

M_MIN, M_MAX = 3.0, 100.0
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


@dataclass(slots=True)
class AttackResult:
    session: list[RawCall]
    labels: list[Label]  # attack labels only; the caller fills in benign


AttackFn = Callable[[list[RawCall], np.random.Generator, AttackContext], AttackResult | None]
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


def mutate_scale(raw: RawCall, path: str, factor: float) -> RawCall:
    """Copy of ``raw`` with the numeric arg at ``path`` multiplied by ``factor``."""
    args: dict[str, Any] = copy.deepcopy(raw.args)
    parent, key = _walk_to_parent(args, path)
    v = parent[key]
    if isinstance(v, bool) or not isinstance(v, int | float):
        raise TypeError(f"{raw.tool_id}.{path} is not numeric: {v!r}")
    parent[key] = int(round(v * factor)) if isinstance(v, int) else round(v * factor, 2)
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


_ELIGIBLE: dict[str, Callable[[ArgSpec, Catalog, str], bool]] = {
    "magnitude_spike": lambda a, cat, name: _is_money_count(name, a),
    "prereq_skip": lambda a, cat, name: _flows_from_read(a, cat),
    "novel_target": lambda a, cat, name: _novel_kind(a) is not None,
}


def eligible(family: str, persona: Persona, catalog: Catalog) -> bool:
    """True iff the persona has a write tool this family can mutate (``KeyError`` if unknown)."""
    test = _ELIGIBLE[family]
    return any(
        test(a, catalog, n) for t in _writes(persona, catalog) for n, a in catalog[t].args.items()
    )


# --- families -----------------------------------------------------------------------------------
# Each family has a pure candidate finder (no RNG) shared by ``applicable`` and the mutation.


def _is_num(v: object) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool)


def _label(ctx: AttackContext, family: str, env: str, strength: float | None = None) -> Label:
    return Label(
        envelope_id=env,
        label="attack",
        attack_family=family,
        injection_level="action",
        attack_id=ctx.attack_id,
        strength=strength,
    )


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


def magnitude_spike(
    session: list[RawCall], rng: np.random.Generator, ctx: AttackContext
) -> AttackResult | None:
    cands = _magnitude_cands(session, ctx)
    if not cands:
        return None
    i, name = cands[int(rng.integers(len(cands)))]
    m = float(np.clip(np.exp(rng.uniform(np.log(M_MIN), np.log(M_MAX))), M_MIN, M_MAX))
    new = mutate_scale(session[i], name, m)
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


ATTACKS: dict[str, AttackFn] = {
    "magnitude_spike": magnitude_spike,
    "prereq_skip": prereq_skip,
    "novel_target": novel_target,
}

_CANDS: dict[str, Callable[[list[RawCall], AttackContext], Sequence[object]]] = {
    "magnitude_spike": _magnitude_cands,
    "prereq_skip": _prereq_cands,
    "novel_target": _novel_cands,
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
) -> tuple[dict[str, list[list[RawCall]]], list[Label]]:
    """Attack persona sessions with probability ``cfg.attack_rate`` (after warmup).

    ``ctx_factory(agent_id, attack_id, persona_sessions, session_index)`` builds the context (it
    knows the catalog, personas and pools). Personas are visited in sorted ``agent_id`` order and
    sessions in time order; only ``rng`` (the attack stream) is ever drawn from. The family is
    uniform over the families applicable to THAT session (persona-eligible and with something to
    mutate); a session where none applies stays benign.
    """
    families = list(dict.fromkeys(cfg.families))
    unknown = sorted(set(families) - set(ATTACKS))
    if unknown:
        raise ValueError(f"unknown attack family: {unknown}")
    out: dict[str, list[list[RawCall]]] = {}
    labels: list[Label] = []
    for agent in sorted(sessions_by_persona):
        sessions = list(sessions_by_persona[agent])
        n_attacks = 0
        for idx in range(cfg.warmup_sessions, len(sessions)):
            if rng.random() >= cfg.attack_rate:
                continue
            ctx = ctx_factory(agent, f"atk-{agent}-{n_attacks:05d}", sessions, idx)
            fams = [
                f
                for f in families
                if eligible(f, ctx.persona, ctx.catalog) and applicable(f, sessions[idx], ctx)
            ]
            if not fams:
                continue
            res = ATTACKS[fams[int(rng.integers(len(fams)))]](sessions[idx], rng, ctx)
            if res is not None:  # always true: applicable() guaranteed a candidate
                sessions[idx] = res.session
                labels += res.labels
                n_attacks += 1
        out[agent] = sessions
    return out, labels
