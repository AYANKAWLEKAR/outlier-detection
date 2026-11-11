from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from agentfw.core.config import EngineConfig
from agentfw.core.detectors.base import IQR_TO_SIGMA, signed_log1p, squash
from agentfw.core.detectors.magnitude import MagnitudeDetector, MagRecord
from agentfw.core.detectors.novelty import NoveltyDetector, NoveltyRecord
from agentfw.core.envelope import DetectorScore, FieldType
from agentfw.core.priors import PooledPriors
from agentfw.core.schema_infer import TypeInference, TypeView
from agentfw.core.state import DictStore, key_prior_mag, key_prior_nov
from tests.factories import make_env

TOOL = "stripe.create_refund"
NEW = "new_agent"


class StubTypes(TypeView):
    def __init__(self, types: Mapping[str, FieldType]) -> None:
        super().__init__(TypeInference(DictStore(), EngineConfig()), NEW, TOOL)
        self._t = dict(types)

    def __call__(self, path: str) -> FieldType:
        return self._t.get(path, FieldType.UNKNOWN)


TYPES = StubTypes(
    {
        "amount": FieldType.NUMERIC,
        "latitude": FieldType.NUMERIC,
        "currency": FieldType.CATEGORICAL,
    }
)


def _pool(cfg: EngineConfig, path: str = "amount", n: int = 300) -> PooledPriors:
    store = DictStore()
    pp = PooledPriors(store, cfg)
    rng = np.random.default_rng(0)
    for x in rng.lognormal(3.6, 0.35, n):
        pp.observe_numeric(TOOL, path, signed_log1p(float(x)))
    return pp


def _score(pp: PooledPriors | None, cfg: EngineConfig, path: str, x: Any) -> DetectorScore:
    det = MagnitudeDetector(DictStore(), cfg, priors=pp)
    out = det.score(make_env(agent_id=NEW, tool_id=TOOL, args={path: x}), TYPES)
    (s,) = [s for s in out if s.field == path]
    return s


def test_cold_agent_typical_value_low_with_prior() -> None:
    cfg = EngineConfig()
    s = _score(_pool(cfg), cfg, "amount", 37)
    assert s.score < 0.1
    assert s.evidence["prior_used"] is True


def test_cold_agent_x50_high_with_prior() -> None:
    cfg = EngineConfig()
    s = _score(_pool(cfg), cfg, "amount", 37 * 50)
    assert s.score > 0.9
    assert s.evidence["prior_used"] is True


def test_prior_z_matches_manually_inflated_iqr() -> None:
    cfg = EngineConfig().replace(mag_score="z")  # the §6.3 rule, whatever the default is
    pp = _pool(cfg)
    s = _score(pp, cfg, "amount", 1850)
    rec = pp.magnitude_ref(TOOL, "amount", "up")
    assert rec is not None
    q25, q50, q75 = (rec.q.quantile(p) for p in (0.25, 0.5, 0.75))
    z = (signed_log1p(1850) - q50) / ((q75 - q25) / IQR_TO_SIGMA * cfg.prior_iqr_inflate)
    assert s.evidence["z"] == pytest.approx(z)
    assert s.score == pytest.approx(squash(z, cfg.z0, cfg.tau))


def test_observed_value_is_pre_transformed_y() -> None:
    cfg = EngineConfig()
    pp = PooledPriors(DictStore(), cfg)
    pp.observe_numeric(TOOL, "amount", 2.0)
    rec = pp._store.get(key_prior_mag(TOOL, "amount"))  # noqa: SLF001
    assert isinstance(rec, MagRecord)
    assert rec.n == 1
    assert rec.q.quantile(0.5) == pytest.approx(2.0)
    assert rec.direction == "up"


def test_off_mode_cold_score_zero() -> None:
    cfg = dataclasses.replace(EngineConfig(), priors_mode="off")
    pp = _pool(cfg)
    assert pp.magnitude_ref(TOOL, "amount", "up") is None
    assert pp.novelty_ref(TOOL, "currency") is None
    s = _score(pp, cfg, "amount", 1850)
    assert s.score == 0.0
    assert s.evidence["prior_used"] is False


def test_money_count_only_gates_direction_and_novelty() -> None:
    cfg = dataclasses.replace(EngineConfig(), priors_mode="money_count_only")
    pp = _pool(cfg, "latitude")
    assert pp.magnitude_ref(TOOL, "latitude", "both") is None
    assert _score(pp, cfg, "latitude", 5000).evidence["prior_used"] is False
    pp2 = _pool(cfg, "amount")
    assert pp2.magnitude_ref(TOOL, "amount", "up") is not None
    for _ in range(40):
        pp2.observe_categorical(TOOL, "currency", "usd")
    assert pp2.novelty_ref(TOOL, "currency") is None


def test_magnitude_ref_requires_cold_n_observations() -> None:
    cfg = EngineConfig()
    pp = PooledPriors(DictStore(), cfg)
    for i in range(cfg.mag_cold_n - 1):
        pp.observe_numeric(TOOL, "amount", float(i))
    assert pp.magnitude_ref(TOOL, "amount", "up") is None
    pp.observe_numeric(TOOL, "amount", 1.0)
    assert pp.magnitude_ref(TOOL, "amount", "up") is not None


def test_pooled_keys_use_d013_prefixes() -> None:
    cfg = EngineConfig()
    store = DictStore()
    pp = PooledPriors(store, cfg)
    pp.observe_numeric(TOOL, "amount", 1.0)
    pp.observe_categorical(TOOL, "currency", "usd")
    assert key_prior_mag(TOOL, "amount").startswith("p:m:")
    assert key_prior_nov(TOOL, "currency").startswith("p:n:")
    assert isinstance(store.get(key_prior_mag(TOOL, "amount")), MagRecord)
    assert isinstance(store.get(key_prior_nov(TOOL, "currency")), NoveltyRecord)


def _nov_score(pp: PooledPriors | None, cfg: EngineConfig, value: str) -> DetectorScore:
    det = NoveltyDetector(DictStore(), cfg, priors=pp)
    out = det.score(make_env(agent_id=NEW, tool_id=TOOL, args={"currency": value}), TYPES)
    (s,) = [s for s in out if s.field == "currency"]
    return s


def _nov_pool(cfg: EngineConfig) -> PooledPriors:
    pp = PooledPriors(DictStore(), cfg)
    for i in range(300):
        pp.observe_categorical(TOOL, "currency", ["usd", "eur", "gbp"][i % 3])
    # a few singletons to make unseen mass non-trivial
    pp.observe_categorical(TOOL, "currency", "chf")
    return pp


def test_novelty_cold_seen_in_pool_scores_zero() -> None:
    cfg = EngineConfig()
    s = _nov_score(_nov_pool(cfg), cfg, "eur")
    assert s.score == 0.0
    assert s.evidence["prior_used"] is True


def test_novelty_cold_unseen_discounted() -> None:
    cfg = EngineConfig()
    pp = _nov_pool(cfg)
    rec = pp.novelty_ref(TOOL, "currency")
    assert rec is not None
    base = (1 - rec.n_singletons / rec.n_total) * min(1.0, rec.n_total / cfg.novelty_full_n)
    s = _nov_score(pp, cfg, "jpy")
    assert s.evidence["prior_used"] is True
    assert s.score == pytest.approx(base * 0.67)
    assert s.score > 0.0


def test_novelty_without_priors_unchanged() -> None:
    cfg = EngineConfig()
    s = _nov_score(None, cfg, "jpy")
    assert s.score == 0.0
    assert s.evidence["prior_used"] is False
