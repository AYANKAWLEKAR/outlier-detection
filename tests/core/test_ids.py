import numpy as np
import pytest

from agentfw.core.ids import new_ulid, ulid_from


def test_ulid_shape_order_determinism() -> None:
    rng = np.random.default_rng(0)
    a, b = new_ulid(1_000, rng), new_ulid(2_000, rng)
    assert len(a) == 26 and a < b
    assert new_ulid(5, np.random.default_rng(7)) == new_ulid(5, np.random.default_rng(7))


def test_ulid_from_known_values() -> None:
    assert ulid_from(0, 0) == "0" * 26
    assert ulid_from(1, 0) == "0000000001" + "0" * 16
    assert ulid_from(0, 31) == "0" * 25 + "Z"
    assert ulid_from((1 << 48) - 1, (1 << 80) - 1) == "7" + "Z" * 25


def test_ulid_from_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        ulid_from(-1, 0)
    with pytest.raises(ValueError):
        ulid_from(1 << 48, 0)
    with pytest.raises(ValueError):
        ulid_from(0, 1 << 80)
