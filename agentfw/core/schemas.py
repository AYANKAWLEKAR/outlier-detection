"""Captured tool schemas (F3): the MCP ``tools/list`` layer as plain data.

One ``schemas.jsonl`` row per capture: ``tool_id``, ``input_schema`` (JSON Schema),
``annotations`` (MCP names: ``readOnlyHint``, ``destructiveHint``, ``idempotentHint``,
``openWorldHint``; an undeclared hint is omitted), ``captured_at`` (ms) and ``hash``
(:func:`schema_hash` of the schema and annotations). A re-capture appends a row, so the last row
of a tool wins. :func:`hints_from_schemas` lays the annotations over the adapter hints; core never
imports ``mcp``.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson

from agentfw.core.envelope import JsonValue
from agentfw.core.hints import ToolAnnotations, ToolHint

__all__ = [
    "ANNOTATION_KEYS",
    "ToolSchema",
    "hints_from_schemas",
    "load_schemas",
    "schema_hash",
]

#: ``ToolAnnotations`` field -> MCP annotation name.
ANNOTATION_KEYS: dict[str, str] = {
    "read_only": "readOnlyHint",
    "destructive": "destructiveHint",
    "idempotent": "idempotentHint",
    "open_world": "openWorldHint",
}
_FIELD_OF = {v: k for k, v in ANNOTATION_KEYS.items()}
_REQUIRED_KEYS = ("tool_id", "input_schema", "captured_at", "hash")  # annotations: optional


def schema_hash(input_schema: Mapping[str, Any], annotations: Mapping[str, bool]) -> str:
    """sha256 of the sorted-key JSON of ``{"annotations", "input_schema"}`` (MCP names)."""
    blob = {"annotations": dict(annotations), "input_schema": dict(input_schema)}
    return hashlib.sha256(orjson.dumps(blob, option=orjson.OPT_SORT_KEYS)).hexdigest()


def _annotations_row(a: ToolAnnotations) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for f, key in ANNOTATION_KEYS.items():
        v = getattr(a, f)
        if v is not None:
            out[key] = v
    return out


def _annotations_from_row(raw: object) -> ToolAnnotations:
    if not isinstance(raw, dict):
        raise ValueError(f"annotations must be an object, got {raw!r}")
    kw: dict[str, bool] = {}
    for key, v in raw.items():
        if key not in _FIELD_OF:
            raise ValueError(f"unknown annotation {key!r}; known: {sorted(_FIELD_OF)}")
        if not isinstance(v, bool):
            raise ValueError(f"annotation {key!r} must be a bool, got {v!r}")
        kw[_FIELD_OF[key]] = v
    return ToolAnnotations(**kw)


@dataclass(frozen=True, slots=True)
class ToolSchema:
    tool_id: str
    input_schema: dict[str, JsonValue]
    annotations: ToolAnnotations
    captured_at: int
    hash: str

    @classmethod
    def capture(
        cls,
        tool_id: str,
        input_schema: dict[str, JsonValue],
        annotations: ToolAnnotations,
        captured_at: int,
    ) -> ToolSchema:
        """A schema as captured now: ``hash`` computed from the schema and annotations."""
        schema = copy.deepcopy(input_schema)  # the hash must keep describing the stored content
        h = schema_hash(schema, _annotations_row(annotations))
        return cls(tool_id, schema, annotations, captured_at, h)

    def to_row(self) -> dict[str, JsonValue]:
        annotations: dict[str, JsonValue] = dict(_annotations_row(self.annotations))
        return {
            "tool_id": self.tool_id,
            "input_schema": copy.deepcopy(self.input_schema),
            "annotations": annotations,
            "captured_at": self.captured_at,
            "hash": self.hash,
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> ToolSchema:
        missing = [k for k in _REQUIRED_KEYS if k not in row]
        if missing:
            raise ValueError(f"schema row is missing {missing}")
        tool, schema, at, h = row["tool_id"], row["input_schema"], row["captured_at"], row["hash"]
        if not isinstance(tool, str) or not isinstance(schema, dict) or not isinstance(h, str):
            raise ValueError(f"malformed schema row for {tool!r}")
        if isinstance(at, bool) or not isinstance(at, int):
            raise ValueError(f"captured_at must be an int, got {at!r}")
        return cls(
            tool, copy.deepcopy(schema), _annotations_from_row(row.get("annotations", {})), at, h
        )


def load_schemas(path: Path) -> dict[str, ToolSchema]:
    """``tool_id -> ToolSchema`` from a ``schemas.jsonl``; the last row of a tool wins."""
    out: dict[str, ToolSchema] = {}
    for line in Path(path).read_bytes().splitlines():
        if line.strip():
            s = ToolSchema.from_row(orjson.loads(line))
            out[s.tool_id] = s
    return out


def hints_from_schemas(
    schemas: Mapping[str, ToolSchema], base: Mapping[str, ToolHint]
) -> dict[str, ToolHint]:
    """``base`` with every schema'd tool's ``annotations`` set and ``has_schema=True``; its
    ``kind`` / ``field_hints`` stay the base hint's (adapters own those)."""
    out = dict(base)
    for tool, s in schemas.items():
        b = base.get(tool)
        out[tool] = ToolHint(
            kind=None if b is None else b.kind,
            field_hints={} if b is None else b.field_hints,
            annotations=s.annotations,
            has_schema=True,
        )
    return out
