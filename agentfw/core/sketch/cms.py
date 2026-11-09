"""Count-min sketch over a numpy ``uint32`` table (never under-estimates)."""

from __future__ import annotations

import base64
from typing import Any, Self

import numpy as np
import xxhash

__all__ = ["CountMinSketch"]

_U32_MAX = 2**32 - 1


class CountMinSketch:
    __slots__ = ("_table", "depth", "width")

    def __init__(self, width: int = 2048, depth: int = 4) -> None:
        if width < 1 or depth < 1:
            raise ValueError("width and depth must be >= 1")
        self.width = width
        self.depth = depth
        self._table = np.zeros((depth, width), dtype=np.uint32)

    def _cols(self, v: str) -> list[int]:
        data = v.encode("utf-8", "surrogatepass")
        return [xxhash.xxh64_intdigest(data, seed=r) % self.width for r in range(self.depth)]

    def add(self, v: str, c: int = 1) -> None:
        if c < 0:
            raise ValueError("count must be >= 0")
        for r, col in enumerate(self._cols(v)):
            self._table[r, col] = min(_U32_MAX, int(self._table[r, col]) + c)  # saturate

    def estimate(self, v: str) -> int:
        return int(min(self._table[r, col] for r, col in enumerate(self._cols(v))))

    def to_dict(self) -> dict[str, Any]:
        # little-endian so the encoding is platform independent
        raw = self._table.astype("<u4", copy=False).tobytes()
        return {"width": self.width, "depth": self.depth, "table": base64.b64encode(raw).decode()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        cms = cls(int(d["width"]), int(d["depth"]))
        raw = base64.b64decode(d["table"])
        if len(raw) != cms.width * cms.depth * 4:
            raise ValueError("count-min table size does not match width*depth")
        cms._table = (
            np.frombuffer(raw, dtype="<u4").astype(np.uint32).reshape(cms.depth, cms.width).copy()
        )
        return cms
