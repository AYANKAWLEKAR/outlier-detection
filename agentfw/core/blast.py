"""Blast radius ``b ∈ [0.05, 1]`` (spec §6.9, lexical + structural sources), the F3 annotation
terms, irreversibility and the information regime (T-06). Numbers: T-17.

* **Lexical** (tool and field names): 0.6 when a tool-name token is an irreversible verb
  (:data:`IRREVERSIBLE_VERBS`); +0.2 when a tool-name or field-name token is a scope noun
  (:data:`SCOPE_NOUNS`); +0.2 when a field of the call carries a ``money`` / ``count`` hint;
  clipped to [0, 1]. The tool name is the part after the server prefix (``stripe.create_refund``
  -> ``create``, ``refund``), split on non-alphanumerics and camelCase; a token also matches with
  one trailing ``s`` removed (``refunds``).
* **Structural**: 0.3, +0.4 for a write, + ``min(0.3, log10(n) / 4)`` for the longest list of
  id-like values among the args (``n`` values), −0.3 for a read with response item counts;
  clipped to [0, 1].
* ``b = 0.5 · lexical + 0.5 · structural``, then the MCP annotation terms (F3): ``destructive``
  +0.4; ``read_only`` −0.4 **only when corroborated** (the structural inference sees a read with
  item counts and no id-list argument), because annotations are server-supplied and untrusted:
  they may raise b and may lower it only with that evidence; ``idempotent is False`` +0.1 unless
  the tool is a corroborated read; ``open_world`` +0.1. Finally floor 0.05, cap 1.

Everything here is a pure function of its arguments; the engine supplies ``is_write`` and
``has_item_count`` from the sequence detector's (read-only) kind records.
"""

from __future__ import annotations

import functools
import math
import re
from collections.abc import Iterable, Mapping

from agentfw.core.envelope import ActionEnvelope, Regime
from agentfw.core.hints import FieldHint, ToolAnnotations, ToolHint
from agentfw.core.patterns import is_id_like

__all__ = [
    "B_FLOOR",
    "IRREVERSIBLE_B",
    "IRREVERSIBLE_VERBS",
    "SCOPE_NOUNS",
    "Regime",
    "blast_radius",
    "id_list_len",
    "irreversible",
    "lexical",
    "regime_of",
    "structural",
    "tool_blast_radius",
]

IRREVERSIBLE_VERBS = frozenset(
    {"refund", "payout", "transfer", "delete", "drop", "void", "cancel", "send", "post", "publish"}
)
SCOPE_NOUNS = frozenset({"bulk", "mass", "all", "batch"})

LEX_VERB = 0.6
LEX_SCOPE = 0.2
LEX_HINT = 0.2
ST_BASE = 0.3
ST_WRITE = 0.4
ST_ID_LIST_MAX = 0.3
ST_READ = 0.3
W_LEXICAL = 0.5
ANN_DESTRUCTIVE = 0.4
ANN_READ_ONLY = 0.4
ANN_NOT_IDEMPOTENT = 0.1
ANN_OPEN_WORLD = 0.1
B_FLOOR = 0.05
IRREVERSIBLE_B = 0.6  # EngineConfig.irreversible_b's default

_SPLIT = re.compile(r"[^A-Za-z0-9]+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_EMPTY_HINTS: Mapping[str, FieldHint] = {}


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return min(hi, max(lo, x))


@functools.lru_cache(maxsize=4096)
def _tokens(name: str) -> frozenset[str]:
    """Name tokens; memoized (a pure function of the name; tool and path names repeat on every
    call, so this keeps the lexical source off the score path's critical cost; Task 4.1)."""
    out: set[str] = set()
    for part in _SPLIT.split(_CAMEL.sub("_", name)):
        if part:
            t = part.lower()
            out.add(t)
            if len(t) > 1 and t.endswith("s"):
                out.add(t[:-1])
    return frozenset(out)


def lexical(tool_id: str, flat_paths: Iterable[str], field_hints: Mapping[str, FieldHint]) -> float:
    """§6.9 lexical source over the tool name, this call's flat paths and their field hints."""
    paths = tuple(flat_paths)
    tool = _tokens(tool_id.split(".", 1)[-1])
    fields: set[str] = set()
    for p in paths:
        fields |= _tokens(p)
    s = LEX_VERB if tool & IRREVERSIBLE_VERBS else 0.0
    if (tool | fields) & SCOPE_NOUNS:
        s += LEX_SCOPE
    if any(field_hints.get(p) in ("money", "count") for p in paths):
        s += LEX_HINT
    return _clip(s)


def structural(is_write: bool, id_list_len: int, has_item_count: bool) -> float:
    """§6.9 structural source: write kind, the longest id list, a read with item counts."""
    s = ST_BASE
    if is_write:
        s += ST_WRITE
    elif has_item_count:
        s -= ST_READ
    if id_list_len > 0:
        s += min(ST_ID_LIST_MAX, math.log10(id_list_len) / 4.0)
    return _clip(s)


def id_list_len(e: ActionEnvelope) -> int:
    """Length of the longest list argument, counting its id-like string values (0 if none)."""
    best = 0
    for path, values in e.flat_args.items():
        if path.endswith("[]"):
            best = max(best, sum(1 for v in values if isinstance(v, str) and is_id_like(v)))
    return best


def _with_annotations(b: float, ann: ToolAnnotations | None, corroborated_read: bool) -> float:
    if ann is not None:
        trusted_read_only = ann.read_only is True and corroborated_read
        if ann.destructive is True:
            b += ANN_DESTRUCTIVE
        if trusted_read_only:
            b -= ANN_READ_ONLY
        if ann.idempotent is False and not trusted_read_only:
            b += ANN_NOT_IDEMPOTENT
        if ann.open_world is True:
            b += ANN_OPEN_WORLD
    return _clip(b, B_FLOOR, 1.0)


def _combine(
    lex: float, is_write: bool, n_ids: int, has_item_count: bool, ann: ToolAnnotations | None
) -> float:
    b = W_LEXICAL * lex + (1.0 - W_LEXICAL) * structural(is_write, n_ids, has_item_count)
    corroborated = not is_write and has_item_count and n_ids == 0
    return _with_annotations(b, ann, corroborated)


def blast_radius(
    e: ActionEnvelope, hint: ToolHint | None, is_write: bool, has_item_count: bool
) -> float:
    """``b`` of one call (T-17, then the F3 annotation terms)."""
    fh = _EMPTY_HINTS if hint is None else hint.field_hints
    lex = lexical(e.tool_id, e.flat_args, fh)
    return _combine(
        lex, is_write, id_list_len(e), has_item_count, None if hint is None else hint.annotations
    )


def tool_blast_radius(
    tool_id: str,
    kind: str | None,
    annotations: ToolAnnotations | None,
    field_hints: Mapping[str, FieldHint],
) -> float:
    """Static ``b`` of a tool from its declaration alone (the F3 ranking test): every hinted field
    present, no id list, and a declared read taken as the structural read (item counts)."""
    lex = lexical(tool_id, field_hints, field_hints)
    return _combine(lex, kind == "write", 0, kind == "read", annotations)


def irreversible(
    b: float | None, hint: ToolHint | None, is_write: bool, irreversible_b: float = IRREVERSIBLE_B
) -> bool:
    """``destructive ∨ (is_write ∧ b ≥ irreversible_b)``; with no ``b`` (``br_mode="off"``) only
    the destructive annotation counts."""
    ann = None if hint is None else hint.annotations
    if ann is not None and ann.destructive is True:
        return True
    return b is not None and is_write and b >= irreversible_b


def regime_of(has_schema: bool, n_tool_obs: int, cold_n: int) -> Regime:
    """T-06: "history" = the (agent, tool) has ``>= cold_n`` observations; "schemas" = the tool
    has a captured schema."""
    history = n_tool_obs >= cold_n
    if history:
        return "history+schemas" if has_schema else "history_only"
    return "schemas_only" if has_schema else "bare"
