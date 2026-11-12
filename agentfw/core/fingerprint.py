"""Call and response fingerprints for the session substrate.

``call_fingerprint`` identifies a (tool, flat args) pair exactly; ``arg_tokens`` gives a set
view of the arguments for overlap tests; ``resp_fingerprint`` is a 64-permutation MinHash over
word shingles of the first 4 KB of response text plus coarse metadata tokens. The synth
generator and the sandbox proxy both call ``resp_fingerprint`` so their ``ResponseMeta.fp``
values agree. Everything is deterministic across processes (xxhash, a frozen numpy seed).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import orjson
import xxhash

from agentfw.core.envelope import Scalar

__all__ = [
    "MAX_RESP_BYTES",
    "MINHASH_K",
    "SHINGLE_WORDS",
    "arg_tokens",
    "call_fingerprint",
    "minhash_jaccard",
    "resp_fingerprint",
]

MINHASH_K = 64
SHINGLE_WORDS = 3
MAX_RESP_BYTES = 4096
_P = (1 << 61) - 1

_rng = np.random.default_rng(0x5EED)
# Frozen universal-hash coefficients; a_i in [1, p), b_i in [0, p). Python ints (exact).
_A: tuple[int, ...] = tuple(int(v) for v in _rng.integers(1, _P, size=MINHASH_K, dtype=np.int64))
_B: tuple[int, ...] = tuple(int(v) for v in _rng.integers(0, _P, size=MINHASH_K, dtype=np.int64))
# Object arrays keep the 122-bit products exact (no uint64 overflow).
_A_OBJ = np.array(_A, dtype=object)
_B_OBJ = np.array(_B, dtype=object)


_INT_LO, _INT_HI = -(1 << 63), 1 << 64  # what orjson serializes


def _safe(v: Scalar) -> Scalar:
    """``v``, or ``"#int:<hex>"`` for an int beyond 64 bits (orjson refuses those and ``str``
    refuses ints past the digit limit; hex has no limit). Every other value is unchanged."""
    if isinstance(v, int) and not isinstance(v, bool) and not _INT_LO <= v < _INT_HI:
        return f"#int:{v:x}"
    return v


def _utf8_safe(v: object) -> object:
    """A string with lone surrogates (not UTF-8, so not JSON) as ``"#str:<hex of its bytes>"``."""
    if isinstance(v, str):
        try:
            v.encode()
        except UnicodeEncodeError:
            return "#str:" + v.encode("utf-8", "surrogatepass").hex()
    return v


def call_fingerprint(tool_id: str, flat_args: Mapping[str, tuple[Scalar, ...]]) -> int:
    """xxh64 of the sorted-key JSON of ``[tool_id, sorted(flat_args.items())]``; ids kept raw.

    Ints beyond 64 bits and strings with lone surrogates (which JSON cannot carry) are replaced
    by tagged hex forms, so any flattened call has a fingerprint.
    """
    items = sorted((k, [_safe(x) for x in v]) for k, v in flat_args.items())
    try:
        raw = orjson.dumps([tool_id, items], option=orjson.OPT_SORT_KEYS)
    except orjson.JSONEncodeError:  # lone surrogates somewhere: rare, so only then re-encode
        safe = [(_utf8_safe(k), [_utf8_safe(x) for x in v]) for k, v in items]
        raw = orjson.dumps([_utf8_safe(tool_id), safe], option=orjson.OPT_SORT_KEYS)
    return xxhash.xxh64_intdigest(raw)


def arg_tokens(flat_args: Mapping[str, tuple[Scalar, ...]]) -> frozenset[int]:
    """xxh32 of ``"path=value"`` for every flattened argument value."""
    return frozenset(
        xxhash.xxh32_intdigest(f"{path}={_safe(value)}".encode("utf-8", "surrogatepass"))
        for path, vals in flat_args.items()
        for value in vals
    )


def _shingles(text: str, item_count: int | None, error: bool) -> set[str] | None:
    clipped = text.encode("utf-8")[:MAX_RESP_BYTES].decode("utf-8", errors="ignore")
    words = clipped.lower().split()
    if not words:
        return None
    n = SHINGLE_WORDS
    if len(words) < n:
        grams: Iterable[str] = [" ".join(words)]
    else:
        grams = (" ".join(words[i : i + n]) for i in range(len(words) - n + 1))
    out = set(grams)
    if item_count is not None:
        out.add(f"#ic:{(1 + max(item_count, 0)).bit_length() - 1}")
    out.add(f"#err:{int(error)}")
    return out


def _minhash(xs: list[int]) -> tuple[int, ...]:
    x = np.array([v % _P for v in xs], dtype=object)
    sig = [int(((a * x + b) % _P).min()) for a, b in zip(_A_OBJ, _B_OBJ, strict=True)]
    return tuple(sig)


def resp_fingerprint(text: str, item_count: int | None, error: bool) -> tuple[int, ...] | None:
    """MinHash signature of a response, or ``None`` when the text has no words."""
    sh = _shingles(text, item_count, error)
    if sh is None:
        return None
    return _minhash([xxhash.xxh64_intdigest(s.encode()) for s in sh])


def minhash_jaccard(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    """Estimated Jaccard similarity: fraction of agreeing signature slots."""
    if len(a) != len(b) or not a:
        raise ValueError("signatures must be non-empty and equal length")
    return sum(x == y for x, y in zip(a, b, strict=True)) / len(a)
