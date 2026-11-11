from __future__ import annotations

import math
import sys

import pytest

from agentfw.core.detectors import base, magnitude, velocity
from agentfw.core.detectors.base import (
    IQR_TO_SIGMA,
    clamp_finite,
    normal_sf,
    posterior,
    signed_expm1,
    signed_log1p,
    value_key,
)

MAX = sys.float_info.max


@pytest.mark.parametrize("x", [0.0, 1.0, -1.0, 37.5, -1234.56, 1e300])
def test_signed_expm1_inverts_signed_log1p(x: float) -> None:
    assert signed_expm1(signed_log1p(x)) == pytest.approx(x, rel=1e-12)


def test_signed_expm1_saturates_instead_of_overflowing() -> None:
    assert signed_expm1(1e6) == MAX
    assert signed_expm1(-1e6) == -MAX


def test_clamp_finite() -> None:
    assert clamp_finite(math.inf) == MAX
    assert clamp_finite(-math.inf) == -MAX
    assert clamp_finite(2.5) == 2.5


def test_iqr_to_sigma() -> None:
    assert IQR_TO_SIGMA == 1.349


def test_detectors_share_the_base_helpers() -> None:
    """No private copies of the shared helpers in the detectors (review finding)."""
    for mod in (magnitude, velocity):
        for dup in ("_clamp", "_unlog", "_IQR_TO_SIGMA", "_MAX"):
            assert not hasattr(mod, dup), f"{mod.__name__}.{dup}"
    assert base.IQR_TO_SIGMA == IQR_TO_SIGMA


def test_value_key_short_values_are_str() -> None:
    assert value_key("usd", 40) == "usd"
    assert value_key(True, 40) == "True"
    assert value_key(12, 40) == "12"


def test_value_key_hashes_long_and_unstringifiable_values() -> None:
    long_host = "a" * 40 + ".example"
    k = value_key(long_host, 40)
    assert k.startswith("#") and len(k) < 40 and k == value_key(long_host, 40)
    big = 10**5000  # str() refuses: beyond the interpreter's str-digit limit
    assert value_key(big, 40) == value_key(10**5000, 40) != value_key(big + 1, 40)
    assert value_key(10**400, 40).startswith("#")


# --- Amendment A1.2: tail probability and its posterior ---------------------------------------


def test_posterior_anchor_points() -> None:
    assert posterior(1.0, 0.005) == pytest.approx(0.005)
    assert posterior(0.005 / 0.995, 0.005) == pytest.approx(0.5)
    assert posterior(2.0, 0.005) == pytest.approx(0.005)  # p clipped at 1
    assert posterior(0.0, 0.005) == 1.0 and posterior(-1e-18, 0.005) == 1.0
    assert posterior(float("nan"), 0.005) == 1.0
    assert posterior(1e-3, 0.005) == pytest.approx(0.834028, abs=1e-6)


def test_posterior_each_factor_ten_in_p_multiplies_the_odds_by_ten() -> None:
    def odds(p: float) -> float:
        s = posterior(p, 0.005)
        return s / (1.0 - s)

    assert odds(1e-3) / odds(1e-2) == pytest.approx(10.0)
    assert odds(1e-6) / odds(1e-5) == pytest.approx(10.0)


def test_normal_sf_reference_values() -> None:
    assert normal_sf(0.0) == 0.5
    assert normal_sf(0.6744897501960817) == pytest.approx(0.25, abs=1e-12)
    assert normal_sf(1.6448536269514722) == pytest.approx(0.05, abs=1e-12)
    assert normal_sf(40.0) == 0.0 and normal_sf(-40.0) == 1.0
