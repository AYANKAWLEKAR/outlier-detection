from __future__ import annotations

import math
import sys

import pytest

from agentfw.core.detectors import base, magnitude, velocity
from agentfw.core.detectors.base import IQR_TO_SIGMA, clamp_finite, signed_expm1, signed_log1p

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
