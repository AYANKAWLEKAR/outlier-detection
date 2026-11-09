"""Per-tool adapter hints (the Tier 3 adapter seam)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

__all__ = ["FieldHint", "ToolHint"]

FieldHint = Literal["money", "count"]


@dataclass(frozen=True, slots=True)
class ToolHint:
    """``kind`` marks a tool read/write; ``field_hints`` maps flat paths to money/count.

    ``field_hints`` is copied into a read-only mapping, so a hint cannot change after
    construction (nor through the caller's original dict).
    """

    kind: Literal["read", "write"] | None = None
    field_hints: Mapping[str, FieldHint] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "field_hints", MappingProxyType(dict(self.field_hints)))
