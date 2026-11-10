"""Sandbox configuration, safety guards and the Salesforce API budget."""

from __future__ import annotations

import contextlib
import fcntl
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import orjson
from pydantic import BaseModel, field_validator

ALLOWED_SMTP_HOSTS = frozenset({"localhost", "127.0.0.1", "mailhog"})


class UnsafeConfigError(RuntimeError):
    """The configuration would let the sandbox touch something real."""


class BudgetExceeded(RuntimeError):
    """The API call budget for the current window is spent."""


class SandboxConfig(BaseModel):
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    judge_base_url: str
    judge_api_key: str
    judge_model: str
    stripe_secret_key: str
    sf_username: str
    sf_password: str
    sf_security_token: str
    sf_domain: str = "login"
    slack_bot_token: str
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
    def _stripe_test_only(cls, v: str) -> str:
        if not v.startswith("sk_test_"):
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

    def _transact(self, n: int) -> int:
        """Charge `n` (0 = peek) under an exclusive lock; return the remaining budget."""
        with open(self.path.with_name(self.path.name + ".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            now = self._clock()
            state: dict[str, float] = {"window_start": now, "used": 0}
            if self.path.exists():
                with contextlib.suppress(orjson.JSONDecodeError):
                    state = orjson.loads(self.path.read_bytes())
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
        self._transact(n)

    def remaining(self) -> int:
        return self._transact(0)
