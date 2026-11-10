"""Stripe MCP server. Amounts are dollars at the tool boundary and cents inside backends."""

from __future__ import annotations

import hashlib
import math
import threading
from dataclasses import dataclass, field
from typing import Any, Protocol

from mcp.server.mcpserver import MCPServer

from agentfw.sandbox.config import SandboxConfig, UnsafeConfigError
from agentfw.sandbox.seed.company import Company
from agentfw.sandbox.tools.common import mark_fake, ok, tool_error

_STRIPE_REASONS = frozenset({"duplicate", "fraudulent", "requested_by_customer"})


class StripeBackend(Protocol):
    def retrieve_charge(self, charge_id: str) -> dict[str, Any]: ...
    def list_charges(self, customer_id: str | None, limit: int) -> list[dict[str, Any]]: ...
    def create_refund(self, charge_id: str, amount: float, reason: str) -> dict[str, Any]: ...
    def retrieve_customer(self, customer_id: str) -> dict[str, Any]: ...
    def create_payout(
        self, amount: float, destination: str | None, currency: str
    ) -> dict[str, Any]: ...


def _sid(prefix: str, seed: str, n: int) -> str:
    return f"{prefix}_{hashlib.sha1(f'{prefix}:{seed}'.encode()).hexdigest()[:n]}"


def fake_stripe_ids(company: Company) -> dict[str, str]:
    """Deterministic Stripe-looking ids: {customer key: cus_..., charge key: ch_...}."""
    ids = {c.key: _sid("cus", c.key, 14) for c in company.customers}
    ids.update({c.key: _sid("ch", c.key, 24) for c in company.charges})
    return ids


def _cents(amount: float) -> int:
    if not math.isfinite(amount) or amount <= 0:
        raise tool_error(f"amount must be a positive number of dollars, got {amount}")
    return round(amount * 100)


def _usd(cents: int) -> str:
    return f"{cents / 100:.2f}"


@dataclass(slots=True)
class _Charge:
    id: str
    customer: str
    amount: int
    currency: str
    description: str
    refunded: int = 0


@dataclass(slots=True)
class _State:
    customers: dict[str, dict[str, str]] = field(default_factory=dict)
    charges: dict[str, _Charge] = field(default_factory=dict)
    payouts_total: int = 0
    n_refunds: int = 0
    n_payouts: int = 0


class FakeStripeBackend:
    """In-memory, deterministic Stripe stand-in mirroring the semantics the sandbox relies on."""

    is_fake = True

    def __init__(self, company: Company) -> None:
        ids = fake_stripe_ids(company)
        self._s = _State()
        self._lock = threading.Lock()
        for c in company.customers:
            cid = ids[c.key]
            self._s.customers[cid] = {"id": cid, "email": c.email, "name": c.name}
        for p in company.charges:
            chid = ids[p.key]
            self._s.charges[chid] = _Charge(
                chid, ids[p.customer_key], p.amount_cents, p.currency, p.description
            )

    @staticmethod
    def _charge_dict(c: _Charge) -> dict[str, Any]:
        return {
            "id": c.id,
            "object": "charge",
            "amount": c.amount / 100,
            "amount_refunded": c.refunded / 100,
            "currency": c.currency,
            "customer": c.customer,
            "description": c.description,
            "refunded": c.refunded >= c.amount,
            "status": "succeeded",
        }

    def _charge(self, charge_id: str) -> _Charge:
        c = self._s.charges.get(charge_id)
        if c is None:
            raise tool_error(f"No such charge: {charge_id!r}")
        return c

    def retrieve_charge(self, charge_id: str) -> dict[str, Any]:
        with self._lock:
            return self._charge_dict(self._charge(charge_id))

    def list_charges(self, customer_id: str | None, limit: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = [
                self._charge_dict(c)
                for c in self._s.charges.values()
                if customer_id is None or c.customer == customer_id
            ]
        return rows[: max(1, min(limit, 100))]

    def create_refund(self, charge_id: str, amount: float, reason: str) -> dict[str, Any]:
        cents = _cents(amount)
        with self._lock:
            c = self._charge(charge_id)
            remaining = c.amount - c.refunded
            if cents > remaining:
                raise tool_error(
                    f"Refund amount (${_usd(cents)}) is greater than unrefunded amount "
                    f"on charge (${_usd(remaining)})"
                )
            c.refunded += cents
            self._s.n_refunds += 1
            rid = _sid("re", f"{charge_id}:{self._s.n_refunds}", 24)
        return {
            "id": rid,
            "object": "refund",
            "charge": charge_id,
            "amount": cents / 100,
            "currency": c.currency,
            "reason": reason,
            "status": "succeeded",
        }

    def retrieve_customer(self, customer_id: str) -> dict[str, Any]:
        with self._lock:
            cus = self._s.customers.get(customer_id)
        if cus is None:
            raise tool_error(f"No such customer: {customer_id!r}")
        return {"object": "customer", **cus}

    def _available(self) -> int:
        net = sum(c.amount - c.refunded for c in self._s.charges.values())
        return net - self._s.payouts_total

    def create_payout(
        self, amount: float, destination: str | None, currency: str = "usd"
    ) -> dict[str, Any]:
        cents = _cents(amount)
        with self._lock:
            available = self._available()
            if cents > available:
                raise tool_error(
                    f"Insufficient available balance for payout of ${_usd(cents)} "
                    f"(available: ${_usd(available)})"
                )
            self._s.payouts_total += cents
            self._s.n_payouts += 1
            pid = _sid("po", str(self._s.n_payouts), 24)
        return {
            "id": pid,
            "object": "payout",
            "amount": cents / 100,
            "currency": currency,
            "destination": destination,
            "status": "pending",
        }


def _to_dict(obj: Any) -> dict[str, Any]:
    """Flatten a stripe object to plain JSON-able data (cents -> dollars on money fields)."""
    raw = obj.to_dict() if hasattr(obj, "to_dict") else dict(obj)
    out: dict[str, Any] = {
        k: v for k, v in raw.items() if isinstance(v, str | int | float | bool | type(None))
    }
    for key in ("amount", "amount_refunded"):
        if isinstance(out.get(key), int):
            out[key] = out[key] / 100
    return out


class LiveStripeBackend:
    """Stripe test mode via the `stripe` SDK. Refuses any non-`sk_test_` key."""

    def __init__(self, cfg: SandboxConfig) -> None:
        import stripe

        key = cfg.stripe_secret_key.get_secret_value()
        if not key.startswith("sk_test_"):
            raise UnsafeConfigError("LiveStripeBackend requires an sk_test_ key")
        self._stripe = stripe
        self._client = stripe.StripeClient(key, max_network_retries=0)

    def _call(self, fn: Any, *args: Any) -> Any:
        try:
            return fn(*args)
        except self._stripe.StripeError as exc:
            raise tool_error(getattr(exc, "user_message", None) or str(exc)) from exc

    def retrieve_charge(self, charge_id: str) -> dict[str, Any]:
        return _to_dict(self._call(self._client.charges.retrieve, charge_id))

    def list_charges(self, customer_id: str | None, limit: int) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"limit": max(1, min(limit, 100))}
        if customer_id:
            params["customer"] = customer_id
        res = self._call(self._client.charges.list, params)
        return [_to_dict(c) for c in res.data]

    def create_refund(self, charge_id: str, amount: float, reason: str) -> dict[str, Any]:
        params: dict[str, Any] = {
            "charge": charge_id,
            "amount": _cents(amount),
            "metadata": {"afw_reason": reason[:500]},
        }
        if reason in _STRIPE_REASONS:
            params["reason"] = reason
        return _to_dict(self._call(self._client.refunds.create, params))

    def retrieve_customer(self, customer_id: str) -> dict[str, Any]:
        return _to_dict(self._call(self._client.customers.retrieve, customer_id))

    def create_payout(
        self, amount: float, destination: str | None, currency: str = "usd"
    ) -> dict[str, Any]:
        cents = _cents(amount)
        balance = self._call(self._client.balance.retrieve)
        available = sum(b.amount for b in balance.available if b.currency == currency)
        if cents > available:
            raise tool_error(
                f"Insufficient available balance for payout of ${_usd(cents)} "
                f"(available: ${_usd(available)})"
            )
        params: dict[str, Any] = {"amount": cents, "currency": currency}
        if destination:
            params["destination"] = destination
        return _to_dict(self._call(self._client.payouts.create, params))


def build_server(backend: StripeBackend) -> MCPServer:
    server = MCPServer("stripe")

    @server.tool()
    def retrieve_charge(charge_id: str) -> dict[str, Any]:
        """Retrieve a charge (amounts in dollars)."""
        return ok(backend.retrieve_charge(charge_id))

    @server.tool()
    def list_charges(customer_id: str | None = None, limit: int = 20) -> dict[str, Any]:
        """List charges, optionally for one customer."""
        return ok(backend.list_charges(customer_id, limit))

    @server.tool()
    def create_refund(charge_id: str, amount: float, reason: str) -> dict[str, Any]:
        """Refund `amount` dollars of a charge."""
        return ok(backend.create_refund(charge_id, amount, reason))

    @server.tool()
    def retrieve_customer(customer_id: str) -> dict[str, Any]:
        """Retrieve a customer."""
        return ok(backend.retrieve_customer(customer_id))

    @server.tool()
    def create_payout(
        amount: float, destination: str | None = None, currency: str = "usd"
    ) -> dict[str, Any]:
        """Pay out `amount` dollars from the available balance."""
        return ok(backend.create_payout(amount, destination, currency))

    return mark_fake(server, backend)
