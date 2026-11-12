from __future__ import annotations

import random
import subprocess
import sys

import orjson
import pytest
from pydantic import ValidationError

from agentfw.core import fingerprint as fpm
from agentfw.core.envelope import ResponseMeta
from agentfw.core.fingerprint import (
    MINHASH_K,
    arg_tokens,
    call_fingerprint,
    minhash_jaccard,
    resp_fingerprint,
)

P = (1 << 61) - 1


def test_call_fp_ignores_key_order_but_not_ids() -> None:
    assert call_fingerprint("t.x", {"a": (1,), "b": ("u",)}) == call_fingerprint(
        "t.x", {"b": ("u",), "a": (1,)}
    )
    assert call_fingerprint("t.x", {"id": ("ch_aaaaaaaa11",)}) != call_fingerprint(
        "t.x", {"id": ("ch_aaaaaaaa22",)}
    )
    assert call_fingerprint("t.x", {}) != call_fingerprint("t.y", {})


def test_arg_tokens() -> None:
    toks = arg_tokens({"a": (1, "x"), "b": (None,)})
    assert len(toks) == 3 and all(isinstance(t, int) for t in toks)
    assert arg_tokens({}) == frozenset()


def test_minhash_estimates_jaccard() -> None:
    a = resp_fingerprint(" ".join(f"w{i}" for i in range(400)), 10, False)
    b = resp_fingerprint(" ".join(f"w{i}" for i in range(200, 600)), 10, False)
    assert a is not None and b is not None and len(a) == MINHASH_K
    assert minhash_jaccard(a, a) == 1.0
    assert minhash_jaccard(a, b) == pytest.approx(1 / 3, abs=0.15)


def test_only_first_4kb_counts_and_meta_tokens_matter() -> None:
    base = "x " * 5000
    assert resp_fingerprint(base, 1, False) == resp_fingerprint(base + "tail", 1, False)
    assert resp_fingerprint("same text here", 1, False) != resp_fingerprint(
        "same text here", 1, True
    )
    assert resp_fingerprint("same text here", 1, False) != resp_fingerprint(
        "same text here", 100, False
    )


def test_split_multibyte_char_is_dropped_not_fatal() -> None:
    text = "a " * 2047 + "é" * 10  # the 4096-byte cut lands inside a 2-byte char
    assert resp_fingerprint(text, None, False) is not None


def test_empty_text_has_no_fingerprint_and_short_text_does() -> None:
    assert resp_fingerprint("", 3, False) is None
    assert resp_fingerprint("   \n", None, True) is None
    assert resp_fingerprint("ok", None, False) is not None


def test_minhash_matches_pure_python_big_int_reference() -> None:
    rnd = random.Random(7)
    for _ in range(50):
        xs = [rnd.getrandbits(64) for _ in range(rnd.randint(1, 40))]
        ref = tuple(
            min((a * (x % P) + b) % P for x in xs) for a, b in zip(fpm._A, fpm._B, strict=True)
        )
        assert fpm._minhash(xs) == ref
        assert all(0 <= v < P for v in ref)


def test_fingerprint_is_deterministic_across_processes() -> None:
    code = (
        "from agentfw.core.fingerprint import resp_fingerprint as r, call_fingerprint as c;"
        "import json;"
        "print(json.dumps([list(r('alpha beta gamma delta', 7, True)), c('t.x', {'a': (1,)})]))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    sig, cfp = orjson.loads(out.stdout)
    assert tuple(sig) == resp_fingerprint("alpha beta gamma delta", 7, True)
    assert cfp == call_fingerprint("t.x", {"a": (1,)})


def test_response_meta_fp_default_omitted_and_round_trips() -> None:
    assert "fp" not in orjson.loads(ResponseMeta(item_count=1).model_dump_json())
    assert "fp" not in ResponseMeta().model_dump(mode="json")
    sig = resp_fingerprint("alpha beta gamma delta", 1, False)
    m = ResponseMeta(item_count=1, fp=sig)
    assert ResponseMeta.model_validate_json(m.model_dump_json()).fp == sig
    assert ResponseMeta.model_validate(m.model_dump(mode="json")) == m


@pytest.mark.parametrize("bad", [(1,) * 63, (1,) * 65, (-1,) + (1,) * 63, (P,) + (1,) * 63])
def test_response_meta_fp_validator_rejects(bad: tuple[int, ...]) -> None:
    with pytest.raises(ValidationError):
        ResponseMeta(fp=bad)
