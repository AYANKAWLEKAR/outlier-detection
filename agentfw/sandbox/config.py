"""Sandbox configuration, safety guards and the Salesforce API budget."""

from __future__ import annotations

import fcntl
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, ClassVar

import orjson
from pydantic import BaseModel, ConfigDict, SecretStr, field_validator

ALLOWED_SMTP_HOSTS = frozenset({"localhost", "127.0.0.1", "mailhog"})


class UnsafeConfigError(RuntimeError):
    """The configuration would let the sandbox touch something real."""


class BudgetExceeded(RuntimeError):
    """The API call budget for the current window is spent."""


class SandboxConfig(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)

    # Env vars (lower-cased) that have no default; live tests skip unless all are set.
    REQUIRED_ENV: ClassVar[tuple[str, ...]] = (
        "llm_base_url",
        "llm_api_key",
        "llm_model",
        "judge_base_url",
        "judge_api_key",
        "judge_model",
        "stripe_secret_key",
        "sf_username",
        "sf_password",
        "sf_security_token",
        "slack_bot_token",
        "slack_channel_id",
        "llm_usd_budget",
        "price_in_per_m",
        "price_out_per_m",
    )

    llm_base_url: str
    llm_api_key: SecretStr
    llm_model: str
    judge_base_url: str
    judge_api_key: SecretStr
    judge_model: str
    stripe_secret_key: SecretStr
    sf_username: str
    sf_password: SecretStr
    sf_security_token: SecretStr
    sf_domain: str = "login"
    slack_bot_token: SecretStr
    slack_channel_id: str
    smtp_host: str = "localhost"
    smtp_port: int = 1025
    mailhog_http: str = "http://localhost:8025"
    sf_daily_api_budget: int = 12_000
    llm_usd_budget: float
    price_in_per_m: float
    price_out_per_m: float
    max_concurrency: int = 4
    data_dir: Path = Path("sandbox_data")

    @field_validator("stripe_secret_key")
    @classmethod
    def _stripe_test_only(cls, v: SecretStr) -> SecretStr:
        if not v.get_secret_value().startswith("sk_test_"):
            raise UnsafeConfigError("stripe_secret_key must start with 'sk_test_' (test mode only)")
        return v

    @field_validator("smtp_host")
    @classmethod
    def _smtp_local_only(cls, v: str) -> str:
        if v not in ALLOWED_SMTP_HOSTS:
            raise UnsafeConfigError(
                f"smtp_host {v!r} refused: email goes only via Mailhog "
                f"({', '.join(sorted(ALLOWED_SMTP_HOSTS))})"
            )
        return v

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> SandboxConfig:
        """Build from upper-cased field names in `env` (default: the process environment)."""
        src = os.environ if env is None else env
        values: dict[str, Any] = {
            name: src[name.upper()] for name in cls.model_fields if name.upper() in src
        }
        return cls(**values)


class ApiBudget:
    """File-locked JSON counter of API calls over a rolling-reset window."""

    def __init__(
        self,
        path: Path,
        limit: int,
        window_s: int = 86_400,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = Path(path)
        self.limit = limit
        self.window_s = window_s
        self._clock = clock
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict[str, float]:
        """Read the counter file; a corrupt or wrong-shape file fails closed."""
        try:
            raw = orjson.loads(self.path.read_bytes())
        except orjson.JSONDecodeError as exc:
            raise BudgetExceeded(f"budget file {self.path} is corrupt; refusing to spend") from exc
        ok = (
            isinstance(raw, dict)
            and isinstance(raw.get("window_start"), int | float)
            and isinstance(raw.get("used"), int)
            and not isinstance(raw.get("used"), bool)
            and raw["used"] >= 0
        )
        if not ok:
            raise BudgetExceeded(f"budget file {self.path} has an unexpected shape; refusing")
        return {"window_start": float(raw["window_start"]), "used": raw["used"]}

    def _transact(self, n: int) -> int:
        """Charge `n` (0 = peek) under an exclusive lock; return the remaining budget."""
        with open(self.path.with_name(self.path.name + ".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            now = self._clock()
            state: dict[str, float] = {"window_start": now, "used": 0}
            if self.path.exists():
                state = self._load()
            if now - state["window_start"] >= self.window_s:
                state = {"window_start": now, "used": 0}
            used = int(state["used"])
            if n and used + n > self.limit:
                raise BudgetExceeded(f"API budget {self.limit} per {self.window_s}s exceeded")
            used += n
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_bytes(orjson.dumps({"window_start": state["window_start"], "used": used}))
            tmp.replace(self.path)
            return self.limit - used

    def charge(self, n: int = 1) -> None:
        if n < 1:
            raise ValueError("charge amount must be >= 1")
        self._transact(n)

    def remaining(self) -> int:
        return self._transact(0)
