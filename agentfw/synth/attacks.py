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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from agentfw.core.envelope import JsonValue, Label, RawCall
from agentfw.synth.dists import IdPool, new_id
from agentfw.synth.schema import ArgSpec, Catalog, Persona

if TYPE_CHECKING:
    from agentfw.synth.generate import GenerateConfig

__all__ = [
    "ATTACKS",
    "AttackContext",
    "AttackFn",
    "AttackResult",
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


def magnitude_spike(
    session: list[RawCall], rng: np.random.Generator, ctx: AttackContext
) -> AttackResult | None:
    cands = [
        (i, name)
        for i, c in enumerate(session)
        if c.tool_id in ctx.catalog and ctx.catalog[c.tool_id].kind == "write"
        for name, a in ctx.catalog[c.tool_id].args.items()
        if _is_money_count(name, a) and _is_num(c.args.get(name)) and float(c.args[name]) > 0  # type: ignore[arg-type]
    ]
    if not cands:
        return None
    i, name = cands[int(rng.integers(len(cands)))]
    m = float(np.clip(np.exp(rng.uniform(np.log(M_MIN), np.log(M_MAX))), M_MIN, M_MAX))
    new = mutate_scale(session[i], name, m)
    return AttackResult(
        _replace_call(session, i, new), [_label(ctx, "magnitude_spike", new.envelope_id, m)]
    )


def prereq_skip(
    session: list[RawCall], rng: np.random.Generator, ctx: AttackContext
) -> AttackResult | None:
    cands: list[tuple[int, list[str]]] = []
    for i, c in enumerate(session):
        spec = ctx.catalog.get(c.tool_id)
        if spec is None or spec.kind != "write":
            continue
        read_ids = {
            x
            for e in session[:i]
            if ctx.catalog.get(e.tool_id) is not None and ctx.catalog[e.tool_id].kind == "read"
            for x in (e.response_meta.ids if e.response_meta else ())
        }
        names = [
            n
            for n, a in spec.args.items()
            if _flows_from_read(a, ctx.catalog) and c.args.get(n) in read_ids
        ]
        if names:
            cands.append((i, names))
    if not cands:
        return None
    i, names = cands[int(rng.integers(len(cands)))]
    target = session[i]
    spec = ctx.catalog[target.tool_id]
    for n in names:
        target = mutate_replace(target, n, fabricate_id(spec.args[n].prefix or "nx_", rng))
    kept: list[RawCall] = []
    for j, c in enumerate(session):
        if (
            j < i
            and ctx.catalog.get(c.tool_id) is not None
            and ctx.catalog[c.tool_id].kind == "read"
        ):
            continue
        kept.append(target if j == i else c)
    out = [c.model_copy(update={"seq_no": k}) for k, c in enumerate(kept)]
    return AttackResult(out, [_label(ctx, "prereq_skip", target.envelope_id)])


def novel_target(
    session: list[RawCall], rng: np.random.Generator, ctx: AttackContext
) -> AttackResult | None:
    cands: list[tuple[int, str, str]] = []
    for i, c in enumerate(session):
        spec = ctx.catalog.get(c.tool_id)
        if spec is None or spec.kind != "write":
            continue
        for name, a in spec.args.items():
            kind, v = _novel_kind(a), c.args.get(name)
            if kind is None or v is None:
                continue
            if a.type == "list_of_email":
                if isinstance(v, list) and v:
                    cands.append((i, f"{name}.{int(rng.integers(len(v)))}", kind))
            elif isinstance(v, str):
                cands.append((i, name, kind))
    if not cands:
        return None
    i, path, kind = cands[int(rng.integers(len(cands)))]
    a = ctx.catalog[session[i].tool_id].args[path.split(".")[0]]
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


# --- schedule -----------------------------------------------------------------------------------


def inject(
    sessions_by_persona: dict[str, list[list[RawCall]]],
    cfg: GenerateConfig,
    rng: np.random.Generator,
    ctx_factory: CtxFactory,
) -> tuple[dict[str, list[list[RawCall]]], list[Label]]:
    """Attack eligible persona sessions with probability ``cfg.attack_rate`` (after warmup).

    ``ctx_factory(agent_id, attack_id, persona_sessions, session_index)`` builds the context (it
    knows the catalog, personas and pools). Personas are visited in sorted ``agent_id`` order and
    sessions in time order; only ``rng`` (the attack stream) is ever drawn from. The family is
    uniform over the persona's eligible families; if it does not apply to the drawn session the
    remaining ones are tried in random order, and the session stays benign if none applies.
    """
    unknown = sorted(set(cfg.families) - set(ATTACKS))
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
            probe = ctx_factory(agent, "", sessions, idx)
            fams = [f for f in cfg.families if eligible(f, probe.persona, probe.catalog)]
            for k in rng.permutation(len(fams)):
                ctx = ctx_factory(agent, f"atk-{agent}-{n_attacks:05d}", sessions, idx)
                res = ATTACKS[fams[int(k)]](sessions[idx], rng, ctx)
                if res is not None:
                    sessions[idx] = res.session
                    labels += res.labels
                    n_attacks += 1
                    break
        out[agent] = sessions
    return out, labels
