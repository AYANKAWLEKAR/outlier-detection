"""Bounded response metadata (spec 2.3, D-003) from an MCP ``CallToolResult``.

* ``item_count``: the first text block is parsed as JSON. A top-level list counts its length; an
  object counts the first list under ``data | records | items | messages | results | tickets`` (in
  that order), else an integer ``totalSize``, else 1 (a single record such as a ticket). JSON
  scalars, non-JSON text, empty results and error results carry no count (``None``).
* ``bytes``: UTF-8 length of every text block.
* ``error``: the result's ``is_error`` flag (``isError`` on the wire).
* ``ids``: up to 256 distinct id-like string *values* (``patterns.is_id_like``), in document order.
"""

from __future__ import annotations

from collections.abc import Iterator

import orjson
from mcp.types import CallToolResult, TextContent

from agentfw.core.envelope import ResponseMeta
from agentfw.core.patterns import is_id_like

__all__ = ["LIST_KEYS", "MAX_IDS", "extract_response_meta"]

LIST_KEYS = ("data", "records", "items", "messages", "results", "tickets")
MAX_IDS = 256
_NOT_JSON = object()


def _item_count(doc: object) -> int | None:
    if isinstance(doc, list):
        return len(doc)
    if not isinstance(doc, dict):
        return None
    for key in LIST_KEYS:
        v = doc.get(key)
        if isinstance(v, list):
            return len(v)
    total = doc.get("totalSize")
    if isinstance(total, int) and not isinstance(total, bool):
        return total
    return 1


def _strings(doc: object) -> Iterator[str]:
    """String values in document order (iterative, so deep documents cannot overflow)."""
    stack = [doc]
    while stack:
        v = stack.pop()
        if isinstance(v, str):
            yield v
        elif isinstance(v, dict):
            stack.extend(reversed(list(v.values())))
        elif isinstance(v, list):
            stack.extend(reversed(v))


def _ids(doc: object) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for s in _strings(doc):
        if s not in seen and is_id_like(s):
            seen[s] = None
            if len(seen) == MAX_IDS:
                break
    return tuple(seen)


def extract_response_meta(result: CallToolResult) -> ResponseMeta:
    texts = [c.text for c in result.content if isinstance(c, TextContent)]
    doc: object = _NOT_JSON
    if texts:
        try:
            doc = orjson.loads(texts[0])
        except orjson.JSONDecodeError:
            doc = _NOT_JSON
    parsed = doc is not _NOT_JSON
    return ResponseMeta(
        item_count=_item_count(doc) if parsed and not result.is_error else None,
        bytes=sum(len(t.encode()) for t in texts),
        error=bool(result.is_error),
        ids=_ids(doc) if parsed else (),
    )
