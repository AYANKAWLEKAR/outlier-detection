from __future__ import annotations

import numpy as np
import orjson
import pytest

from agentfw.core.sketch.cms import CountMinSketch


def test_empty_estimates_zero() -> None:
    assert CountMinSketch().estimate("x") == 0


def test_add_and_estimate_exact_when_sparse() -> None:
    cms = CountMinSketch()
    cms.add("a")
    cms.add("a", 4)
    cms.add("b", 2)
    assert cms.estimate("a") == 5
    assert cms.estimate("b") == 2


def test_cms_overestimates_never_under() -> None:
    rng = np.random.default_rng(7)
    cms = CountMinSketch(width=64, depth=4)  # tiny: force collisions
    truth: dict[str, int] = {}
    for _ in range(5000):
        v = f"v{int(rng.integers(0, 2000))}"
        truth[v] = truth.get(v, 0) + 1
        cms.add(v)
    assert all(cms.estimate(v) >= c for v, c in truth.items())
    assert any(cms.estimate(v) > c for v, c in truth.items())  # collisions really happen


def test_saturates_instead_of_wrapping() -> None:
    cms = CountMinSketch(width=8, depth=2)
    cms.add("a", 2**32 - 1)
    cms.add("a", 10)
    assert cms.estimate("a") == 2**32 - 1


def test_negative_count_rejected() -> None:
    with pytest.raises(ValueError):
        CountMinSketch().add("a", -1)


def test_roundtrip_is_deterministic_and_json_safe() -> None:
    cms = CountMinSketch(width=128, depth=3)
    for i in range(50):
        cms.add(f"k{i}", i + 1)
    d = cms.to_dict()
    assert orjson.dumps(d, option=orjson.OPT_SORT_KEYS) == orjson.dumps(
        CountMinSketch.from_dict(d).to_dict(), option=orjson.OPT_SORT_KEYS
    )
    back = CountMinSketch.from_dict(orjson.loads(orjson.dumps(d)))
    assert all(back.estimate(f"k{i}") == cms.estimate(f"k{i}") for i in range(50))


def test_from_dict_rejects_wrong_size() -> None:
    d = CountMinSketch(width=8, depth=2).to_dict()
    d["width"] = 16
    with pytest.raises(ValueError):
        CountMinSketch.from_dict(d)


def test_bad_dims_rejected() -> None:
    with pytest.raises(ValueError):
        CountMinSketch(width=0)
