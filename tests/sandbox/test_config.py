"""SandboxConfig safety guards and ApiBudget."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agentfw.sandbox.config import ApiBudget, BudgetExceeded, SandboxConfig, UnsafeConfigError


def _values(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "llm_base_url": "https://llm.example/v1",
        "llm_api_key": "llm-placeholder",
        "llm_model": "model-a",
        "judge_base_url": "https://judge.example/v1",
        "judge_api_key": "judge-placeholder",
        "judge_model": "model-b",
        "stripe_secret_key": "sk_test_placeholder",
        "sf_username": "user@sandbox.example",
        "sf_password": "placeholder",
        "sf_security_token": "placeholder",
        "slack_bot_token": "xoxb-placeholder",
        "slack_channel_id": "C000PLACEHOLDER",
        "llm_usd_budget": 5.0,
        "price_in_per_m": 0.9,
        "price_out_per_m": 0.9,
    }
    base.update(over)
    return base


def test_valid_config_defaults() -> None:
    cfg = SandboxConfig(**_values())
    assert cfg.sf_domain == "login"
    assert cfg.smtp_host == "localhost"
    assert cfg.smtp_port == 1025
    assert cfg.mailhog_http == "http://localhost:8025"
    assert cfg.sf_daily_api_budget == 12_000
    assert cfg.max_concurrency == 4
    assert cfg.data_dir == Path("sandbox_data")


def test_live_stripe_key_refused() -> None:
    with pytest.raises(UnsafeConfigError):
        SandboxConfig(**_values(stripe_secret_key="sk_live_abc123"))


def test_restricted_stripe_key_refused() -> None:
    with pytest.raises(UnsafeConfigError):
        SandboxConfig(**_values(stripe_secret_key="rk_test_abc"))


@pytest.mark.parametrize("host", ["smtp.gmail.com", "mail.example.com", "10.0.0.5", ""])
def test_remote_smtp_host_refused(host: str) -> None:
    with pytest.raises(UnsafeConfigError):
        SandboxConfig(**_values(smtp_host=host))


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "mailhog"])
def test_local_smtp_hosts_allowed(host: str) -> None:
    assert SandboxConfig(**_values(smtp_host=host)).smtp_host == host


def test_from_env_reads_mapping_and_validates() -> None:
    env = {
        "LLM_BASE_URL": "https://llm.example/v1",
        "LLM_API_KEY": "k",
        "LLM_MODEL": "m",
        "JUDGE_BASE_URL": "https://j.example/v1",
        "JUDGE_API_KEY": "k",
        "JUDGE_MODEL": "m",
        "STRIPE_SECRET_KEY": "sk_test_x",
        "SF_USERNAME": "u@x.example",
        "SF_PASSWORD": "p",
        "SF_SECURITY_TOKEN": "t",
        "SLACK_BOT_TOKEN": "xoxb-x",
        "SLACK_CHANNEL_ID": "C1",
        "LLM_USD_BUDGET": "2.5",
        "PRICE_IN_PER_M": "1",
        "PRICE_OUT_PER_M": "2",
        "SF_DAILY_API_BUDGET": "100",
        "SMTP_PORT": "2525",
    }
    cfg = SandboxConfig.from_env(env)
    assert cfg.llm_usd_budget == 2.5
    assert cfg.sf_daily_api_budget == 100
    assert cfg.smtp_port == 2525
    with pytest.raises(UnsafeConfigError):
        SandboxConfig.from_env({**env, "STRIPE_SECRET_KEY": "sk_live_x"})


def test_from_env_uses_process_env_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in {
        "LLM_BASE_URL": "https://llm.example/v1",
        "LLM_API_KEY": "k",
        "LLM_MODEL": "m",
        "JUDGE_BASE_URL": "https://j.example/v1",
        "JUDGE_API_KEY": "k",
        "JUDGE_MODEL": "m",
        "STRIPE_SECRET_KEY": "sk_test_x",
        "SF_USERNAME": "u",
        "SF_PASSWORD": "p",
        "SF_SECURITY_TOKEN": "t",
        "SLACK_BOT_TOKEN": "x",
        "SLACK_CHANNEL_ID": "C1",
        "LLM_USD_BUDGET": "1",
        "PRICE_IN_PER_M": "1",
        "PRICE_OUT_PER_M": "1",
    }.items():
        monkeypatch.setenv(k, v)
    assert SandboxConfig.from_env().stripe_secret_key == "sk_test_x"


class _Clock:
    def __init__(self) -> None:
        self.t = 1_000.0

    def __call__(self) -> float:
        return self.t


def test_budget_raises_after_limit(tmp_path: Path) -> None:
    b = ApiBudget(tmp_path / "b.json", limit=3, clock=_Clock())
    b.charge()
    b.charge(2)
    assert b.remaining() == 0
    with pytest.raises(BudgetExceeded):
        b.charge()
    assert b.remaining() == 0  # failed charge is not recorded


def test_budget_rejects_oversized_charge_without_spending(tmp_path: Path) -> None:
    b = ApiBudget(tmp_path / "b.json", limit=5, clock=_Clock())
    with pytest.raises(BudgetExceeded):
        b.charge(6)
    assert b.remaining() == 5


def test_budget_resets_after_window(tmp_path: Path) -> None:
    clock = _Clock()
    b = ApiBudget(tmp_path / "b.json", limit=2, window_s=100, clock=clock)
    b.charge(2)
    clock.t += 99
    with pytest.raises(BudgetExceeded):
        b.charge()
    clock.t += 2
    assert b.remaining() == 2
    b.charge()
    assert b.remaining() == 1


def test_budget_persists_across_instances(tmp_path: Path) -> None:
    clock = _Clock()
    ApiBudget(tmp_path / "b.json", limit=4, clock=clock).charge(3)
    assert ApiBudget(tmp_path / "b.json", limit=4, clock=clock).remaining() == 1
