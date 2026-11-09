from __future__ import annotations

from agentfw.core.sketch.bloom import BloomFilter


def test_no_false_negatives() -> None:
    bf = BloomFilter(capacity=5000)
    for i in range(5000):
        bf.add(f"id_{i}")
    assert all(f"id_{i}" in bf for i in range(5000))


def test_false_positive_rate_at_capacity() -> None:
    bf = BloomFilter(capacity=5000)
    for i in range(5000):
        bf.add(f"id_{i}")
    fp = sum(f"other_{i}" in bf for i in range(20_000))
    assert fp / 20_000 < 0.005


def test_dict_round_trip_and_copy() -> None:
    bf = BloomFilter(capacity=1000)
    for i in range(1000):
        bf.add(f"id_{i}")
    clone = BloomFilter.from_dict(bf.to_dict())
    cp = bf.copy()
    bf.add("late")
    assert "late" not in cp
    probes = [f"id_{i}" for i in range(1000)] + [f"x_{i}" for i in range(2000)]
    assert [p in cp for p in probes] == [p in clone for p in probes]
