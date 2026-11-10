"""Seed Stripe test mode. Idempotent via metadata markers; never refunds or deletes."""

from __future__ import annotations

from typing import Any

from agentfw.sandbox.config import SandboxConfig, UnsafeConfigError
from agentfw.sandbox.seed.company import Company

MARKER = "afw_seed_key"


def _find(resource: Any, params: dict[str, Any], key: str) -> Any | None:
    for obj in resource.list(params).auto_paging_iter():
        if (getattr(obj, "metadata", None) or {}).get(MARKER) == key:
            return obj
    return None


def seed_stripe(cfg: SandboxConfig, company: Company, client: Any | None = None) -> dict[str, str]:
    """Return {customer key: cus_..., charge key: ch_...}. `client` is injectable for tests."""
    if not cfg.stripe_secret_key.get_secret_value().startswith("sk_test_"):
        raise UnsafeConfigError("refusing to seed Stripe with a non-test key")
    if client is None:
        import stripe

        client = stripe.StripeClient(cfg.stripe_secret_key.get_secret_value())

    ids: dict[str, str] = {}
    for c in company.customers:
        found = _find(client.customers, {"email": c.email, "limit": 100}, c.key)
        if found is None:
            found = client.customers.create(
                {"email": c.email, "name": c.name, "metadata": {MARKER: c.key}}
            )
        ids[c.key] = found.id

    for ch in company.charges:
        cus = ids[ch.customer_key]
        pi = _find(client.payment_intents, {"customer": cus, "limit": 100}, ch.key)
        if pi is None:
            pi = client.payment_intents.create(
                {
                    "amount": ch.amount_cents,
                    "currency": ch.currency,
                    "customer": cus,
                    "description": f"afw-seed:{ch.key}",
                    "metadata": {MARKER: ch.key},
                    "payment_method": "pm_card_bypassPending",
                    "confirm": True,
                    "automatic_payment_methods": {"enabled": True, "allow_redirects": "never"},
                }
            )
        charge = getattr(pi, "latest_charge", None)
        if getattr(pi, "status", None) != "succeeded" or not charge:
            raise RuntimeError(
                f"seed payment {ch.key} did not succeed (status={getattr(pi, 'status', None)!r}, "
                f"charge={charge!r}); refusing to record a charge id"
            )
        ids[ch.key] = str(charge)
    return ids
