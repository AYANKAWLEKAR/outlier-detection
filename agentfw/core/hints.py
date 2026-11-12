"""Per-tool adapter hints (the Tier 3 adapter seam)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

__all__ = ["FieldHint", "ToolAnnotations", "ToolHint"]

FieldHint = Literal["money", "count"]


@dataclass(frozen=True, slots=True)
class ToolAnnotations:
    """A tool's MCP behaviour annotations (``readOnlyHint``, ``destructiveHint``,
    ``idempotentHint``, ``openWorldHint``); ``None`` = not declared. Plain data: core never
    imports ``mcp``."""

    read_only: bool | None = None
    destructive: bool | None = None
    idempotent: bool | None = None
    open_world: bool | None = None


@dataclass(frozen=True, slots=True)
class ToolHint:
    """``kind`` marks a tool read/write; ``field_hints`` maps flat paths to money/count;
    ``annotations`` carries the tool's declared MCP annotations, if any; ``has_schema`` says a
    captured schema (``core.schemas``) stands behind them (the "schemas" half of the regime, T-06).

    ``field_hints`` is copied into a read-only mapping, so a hint cannot change after
    construction (nor through the caller's original dict).
    """

    kind: Literal["read", "write"] | None = None
    field_hints: Mapping[str, FieldHint] = field(default_factory=dict)
    annotations: ToolAnnotations | None = None
    has_schema: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "field_hints", MappingProxyType(dict(self.field_hints)))
