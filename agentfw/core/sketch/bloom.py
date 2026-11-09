"""Bloom filter with xxhash64 double hashing."""

from __future__ import annotations

import base64
import math
from typing import Any, Self

import xxhash

__all__ = ["BloomFilter"]


class BloomFilter:
    __slots__ = ("_bits", "_k", "_m", "capacity", "fp_rate")

    def __init__(self, capacity: int, fp_rate: float = 1e-3) -> None:
        if capacity < 1 or not 0.0 < fp_rate < 1.0:
            raise ValueError("capacity must be >= 1 and fp_rate in (0, 1)")
        self.capacity = capacity
        self.fp_rate = fp_rate
        self._m = max(8, math.ceil(-capacity * math.log(fp_rate) / (math.log(2) ** 2)))
        self._k = max(1, round(self._m / capacity * math.log(2)))
        self._bits = bytearray((self._m + 7) // 8)

    def _positions(self, v: str) -> list[int]:
        data = v.encode("utf-8", "surrogatepass")
        h1 = xxhash.xxh64_intdigest(data)
        h2 = xxhash.xxh64_intdigest(data, seed=1) | 1
        return [(h1 + i * h2) % self._m for i in range(self._k)]

    def add(self, v: str) -> None:
        for p in self._positions(v):
            self._bits[p >> 3] |= 1 << (p & 7)

    def __contains__(self, v: str) -> bool:
        return all(self._bits[p >> 3] & (1 << (p & 7)) for p in self._positions(v))

    def copy(self) -> BloomFilter:
        other = BloomFilter(self.capacity, self.fp_rate)
        other._bits = bytearray(self._bits)
        return other

    def to_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "fp_rate": self.fp_rate,
            "bits": base64.b64encode(bytes(self._bits)).decode("ascii"),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        bf = cls(int(d["capacity"]), float(d["fp_rate"]))
        bits = bytearray(base64.b64decode(d["bits"]))
        if len(bits) != len(bf._bits):
            raise ValueError("bloom bit array size does not match capacity/fp_rate")
        bf._bits = bits
        return bf
