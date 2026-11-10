"""Task-grammar walker: persona + task -> ordered list of planned tool calls (spec 4.4-4.5).

Behaviour summary (see ``agentfw.synth.schema`` for the generation rules this implements):

* Steps run in order; a task with ``repeat`` runs its step list ``max(1, sample(repeat))`` times,
  with fresh aliases per repetition.
* A step that is skipped (``skippable`` + ``skip_lookup``, or ``optional_prob`` not taken) emits no
  call but its produced values are still generated, so later bindings resolve.
* ``retry`` appends one duplicate of the emitted call (deep-copied identical args, identical
  ``response_meta``). ``abandon`` (probability per walk) stops after a uniformly chosen step.
* ``arg_dropout`` is a per-argument probability: each non-id argument (type other than ``id`` /
  ``list_of_id``) is dropped independently. Produced values come from the full argument set.
* ``response_meta.bytes`` is ``64 + len(orjson(produced payload))`` (a fixed header plus the
  payload we generated), so it is deterministic and grows with ``item_count``.
  ``response_meta.ids`` is the concatenation of the ``produces_ids`` fields, truncated to 256;
  ``item_count`` is only set for tools with ``response.item_count`` and is never truncated.
* ``gap_s`` is 0 for the first call, then lognormal (median ~2.7 s); packaging retimes (D-015).
* Categorical Dirichlet weights are cached in ``weights_cache``, a dict the caller owns and should
  keep per persona so repeated walks share weights. Omitted: a fresh dict per walk.
* ``id_flow_rate`` is the probability a bound id value is honoured; otherwise a fabricated id
  (not in the matching pool) is used.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import orjson

from agentfw.core.envelope import JsonValue, ResponseMeta
from agentfw.synth.dists import IdPool, Sampled, WeightsCache, new_id, sample
from agentfw.synth.schema import ArgSpec, Catalog, DistSpec, Persona, TaskSpec, ToolSpec

__all__ = ["PlannedCall", "plan_args", "walk"]

_GAP = DistSpec(family="lognormal", mu=1.0, sigma=0.8)
_DOMAIN_WORDS = ("acme", "globex", "initech", "umbrella", "hooli", "stark", "wayne", "wonka")
_TEXT_WORDS = (
    "please", "review", "the", "account", "order", "update", "customer", "request", "follow",
    "up", "invoice", "status", "shipping", "refund", "issue", "thanks", "team", "regarding",
    "details", "confirm", "pending", "resolved", "contact", "meeting", "quarter", "renewal",
)  # fmt: skip
_ID_TYPES = ("id", "list_of_id")
_MAX_IDS = 256
_BYTES_HEADER = 64


@dataclass(slots=True)
class PlannedCall:
    tool_id: str
    args: dict[str, JsonValue]
    response_meta: ResponseMeta | None
    gap_s: float


@dataclass(slots=True)
class _Ctx:
    rng: np.random.Generator
    persona: Persona
    catalog: Catalog
    pools: dict[str, IdPool]
    cache: WeightsCache = field(default_factory=dict)


def _num(v: Sampled) -> float:
    return float(v)


def _email(ctx: _Ctx, a: ArgSpec) -> str:
    key = next((k for k in ctx.persona.fields if k.endswith(".__domain__")), None)
    dist = (
        ctx.persona.fields[key] if key else a.dist or DistSpec(family="zipf", vocab_size=12, s=1.3)
    )
    k = sample(dist, ctx.rng, cache=ctx.cache, key=key)
    suffix = a.domain_suffix or ".example"
    if isinstance(k, int) and not isinstance(k, bool):
        domain = f"{_DOMAIN_WORDS[k % len(_DOMAIN_WORDS)]}{k}{suffix}"
    else:
        domain = f"{k}{suffix}"
    return f"user{int(ctx.rng.integers(0, 1_000_000)):06d}@{domain}"


def _text(ctx: _Ctx, a: ArgSpec) -> str:
    assert a.len_dist is not None
    n = max(1, round(_num(sample(a.len_dist, ctx.rng))))
    idx = ctx.rng.integers(0, len(_TEXT_WORDS), size=n // 3 + 2)
    return " ".join(_TEXT_WORDS[i] for i in idx)[:n].ljust(n, "x")


def _list_len(ctx: _Ctx, a: ArgSpec, override: DistSpec | None, item_count: int | None) -> int:
    if item_count is not None:
        return item_count
    dist = override or a.len_dist
    return 1 if dist is None else max(1, round(_num(sample(dist, ctx.rng))))


def _len_override(ctx: _Ctx, tool: str, arg: str) -> DistSpec | None:
    return ctx.persona.fields.get(f"{tool}.{arg}.__len__")


def _gen(
    ctx: _Ctx,
    key: str,
    a: ArgSpec,
    *,
    dist: DistSpec | None = None,
    len_dist: DistSpec | None = None,
    item_count: int | None = None,
) -> JsonValue:
    rng = ctx.rng
    t = a.type
    if t == "numeric":
        d = dist or a.dist
        assert d is not None
        v = _num(sample(d, rng, cache=ctx.cache, key=key))
        if a.shape == "count":
            return round(v)
        return round(v, 2) if a.shape == "money" else v
    if t == "categorical":
        d = dist or a.dist or DistSpec(family="categorical", vocab=a.vocab)
        return str(sample(d, rng, cache=ctx.cache, key=key))
    if t == "id":
        return new_id(a.prefix or "id_", rng)
    if t == "email":
        return _email(ctx, a)
    if t == "freetext":
        return _text(ctx, a)
    if t == "freetext_map":
        return {"note": _text(ctx, a)}
    if t == "bool":
        assert a.dist is not None
        return bool(sample(a.dist, rng))
    n = _list_len(ctx, a, len_dist, item_count)
    if t == "list_of_id":
        return [new_id(a.prefix or "id_", rng) for _ in range(n)]
    return [_email(ctx, a) for _ in range(n)]  # list_of_email


def _resolve(aliases: dict[str, dict[str, JsonValue]], ref: str) -> JsonValue:
    alias, _, fld = ref.partition(".")
    return aliases[alias][fld]


def _fabricated(ctx: _Ctx, prefix: str | None) -> str:
    pool = next((p for p in ctx.pools.values() if p.prefix == prefix), None)
    return pool.fabricate(ctx.rng) if pool else new_id(prefix or "id_", ctx.rng)


def _adapt(ctx: _Ctx, tool: str, arg: str, a: ArgSpec, val: JsonValue) -> JsonValue:
    """Fit a bound (produced) value to the argument's declared type."""
    if a.type == "list_of_email" and isinstance(val, str):
        n = _list_len(ctx, a, _len_override(ctx, tool, arg), None)
        return [val, *(_email(ctx, a) for _ in range(n - 1))]  # F1: [bound] + extras
    if a.type == "list_of_id" and isinstance(val, str):
        return [val]
    if (
        a.type == "id"
        and isinstance(val, str)
        and ctx.persona.id_flow_rate < 1.0
        and ctx.rng.random() >= ctx.persona.id_flow_rate
    ):
        return _fabricated(ctx, a.prefix)
    return val


def _build_args(
    ctx: _Ctx, tool: str, spec: ToolSpec, bindings: dict[str, str],
    aliases: dict[str, dict[str, JsonValue]],
) -> dict[str, JsonValue]:  # fmt: skip
    args: dict[str, JsonValue] = {}
    for name, a in spec.args.items():
        b = bindings.get(name, "sample")
        if b == "sample":
            args[name] = _gen(
                ctx,
                f"{tool}.{name}",
                a,
                dist=ctx.persona.fields.get(f"{tool}.{name}"),
                len_dist=_len_override(ctx, tool, name),
            )
        elif b.startswith("ext:"):
            args[name] = ctx.pools[b[4:]].draw()
        elif b.startswith("frac:"):
            _, ref, lo, hi = b.split(":")
            base = _resolve(aliases, ref)
            assert isinstance(base, int | float)
            args[name] = round(float(base) * float(ctx.rng.uniform(float(lo), float(hi))), 2)
        else:
            args[name] = _adapt(ctx, tool, name, a, _resolve(aliases, b))
    return args


def _produce(
    ctx: _Ctx, tool: str, spec: ToolSpec, args: dict[str, JsonValue]
) -> tuple[dict[str, JsonValue], int | None]:
    item_count: int | None = None
    if spec.response.item_count is not None:
        item_count = max(0, round(_num(sample(spec.response.item_count, ctx.rng))))
    produced: dict[str, JsonValue] = {}
    for key, pa in spec.produced.items():
        if pa.from_arg is not None:
            src = args[pa.from_arg]
            produced[key] = list(src) if isinstance(src, list) else src
        else:
            produced[key] = _gen(ctx, f"{tool}:{key}", pa, dist=None, item_count=item_count)
    return produced, item_count


def _meta(spec: ToolSpec, produced: dict[str, JsonValue], item_count: int | None) -> ResponseMeta:
    ids: list[str] = []
    for key in spec.response.produces_ids:
        v = produced[key]
        if isinstance(v, str):
            ids.append(v)
        elif isinstance(v, list):
            ids.extend(x for x in v if isinstance(x, str))
    size = _BYTES_HEADER + len(orjson.dumps(produced, option=orjson.OPT_SORT_KEYS))
    return ResponseMeta(item_count=item_count, bytes=size, error=False, ids=tuple(ids[:_MAX_IDS]))


def _gap(ctx: _Ctx, first: bool) -> float:
    return 0.0 if first else max(1e-3, _num(sample(_GAP, ctx.rng)))


def walk(
    task: TaskSpec,
    persona: Persona,
    catalog: Catalog,
    rng: np.random.Generator,
    pools: dict[str, IdPool],
    *,
    weights_cache: WeightsCache | None = None,
) -> list[PlannedCall]:
    ctx = _Ctx(rng, persona, catalog, pools, {} if weights_cache is None else weights_cache)
    noise = persona.noise
    reps = 1 if task.repeat is None else max(1, round(_num(sample(task.repeat, rng))))
    n_steps = reps * len(task.steps)
    abandon_after = (
        int(rng.integers(n_steps)) if noise.abandon > 0 and rng.random() < noise.abandon else None
    )
    calls: list[PlannedCall] = []
    idx = 0
    for _ in range(reps):
        aliases: dict[str, dict[str, JsonValue]] = {}
        for step in task.steps:
            spec = catalog[step.tool]
            emit = not (step.optional_prob < 1.0 and rng.random() >= step.optional_prob)
            if step.skippable and noise.skip_lookup > 0 and rng.random() < noise.skip_lookup:
                emit = False
            args = _build_args(ctx, step.tool, spec, step.args, aliases)
            produced, item_count = _produce(ctx, step.tool, spec, args)
            if step.as_:
                by_field: dict[str, JsonValue] = {}
                for key, val in produced.items():
                    by_field.setdefault(key.split(".", 1)[1], val)
                aliases[step.as_] = by_field
            if emit:
                if noise.arg_dropout > 0:
                    args = {
                        k: v
                        for k, v in args.items()
                        if spec.args[k].type in _ID_TYPES or rng.random() >= noise.arg_dropout
                    }
                meta = _meta(spec, produced, item_count)
                calls.append(PlannedCall(step.tool, args, meta, _gap(ctx, not calls)))
                if noise.retry > 0 and rng.random() < noise.retry:
                    calls.append(
                        PlannedCall(step.tool, copy.deepcopy(args), meta, _gap(ctx, False))
                    )
            if abandon_after == idx:
                return calls
            idx += 1
    return calls


def plan_args(
    tool: str,
    persona: Persona,
    catalog: Catalog,
    rng: np.random.Generator,
    pools: dict[str, IdPool],
) -> dict[str, JsonValue]:
    """Persona-flavoured args for one standalone ``tool`` call: every arg sampled, no bindings."""
    return _build_args(_Ctx(rng, persona, catalog, pools), tool, catalog[tool], {}, {})
