"""Deterministic ULID generation (Crockford base32, 48-bit ms timestamp + 80-bit randomness)."""

from __future__ import annotations

import numpy as np

_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_TS_BITS = 48
_RAND_BITS = 80


def ulid_from(ts_ms: int, rand80: int) -> str:
    """Encode a timestamp and 80 random bits as a 26-char ULID."""
    if not 0 <= ts_ms < (1 << _TS_BITS):
        raise ValueError(f"ts_ms out of 48-bit range: {ts_ms}")
    if not 0 <= rand80 < (1 << _RAND_BITS):
        raise ValueError(f"rand80 out of 80-bit range: {rand80}")
    value = (ts_ms << _RAND_BITS) | rand80
    chars = []
    for _ in range(26):
        chars.append(_ALPHABET[value & 31])
        value >>= 5
    return "".join(reversed(chars))


def new_ulid(ts_ms: int, rng: np.random.Generator) -> str:
    """ULID at ``ts_ms`` with randomness drawn from ``rng`` (deterministic for a seeded rng)."""
    hi = int(rng.integers(0, 1 << 40))
    lo = int(rng.integers(0, 1 << 40))
    return ulid_from(ts_ms, (hi << 40) | lo)
