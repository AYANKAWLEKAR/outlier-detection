"""Stripe MCP server: fake backend semantics through an in-memory MCP client, plus live."""

from __future__ import annotations

import os

import pytest

from agentfw.core.patterns import is_id_like
from agentfw.sandbox.config import SandboxConfig
from agentfw.sandbox.seed.company import build_company
from agentfw.sandbox.tools.stripe_server import (
    FakeStripeBackend,
    LiveStripeBackend,
    build_server,
    fake_stripe_ids,
)
from tests.sandbox.tools.conftest import call, tool_names

COMPANY = build_company(0)
IDS = fake_stripe_ids(COMPANY)
CHARGE = COMPANY.charges[0]
CHARGE_ID = IDS[CHARGE.key]
CUSTOMER_ID = IDS[CHARGE.customer_key]
DOLLARS = CHARGE.amount_cents / 100


@pytest.fixture
def server():  # type: ignore[no-untyped-def]
    return build_server(FakeStripeBackend(COMPANY))


@pytest.mark.anyio
async def test_tool_names(server) -> None:  # type: ignore[no-untyped-def]
    assert await tool_names(server) == {
        "retrieve_charge",
        "list_charges",
        "create_refund",
        "retrieve_customer",
        "create_payout",
    }


def test_ids_look_like_stripe_ids() -> None:
    for c in COMPANY.charges:
        assert IDS[c.key].startswith("ch_") and is_id_like(IDS[c.key])
    for cust in COMPANY.customers:
        assert IDS[cust.key].startswith("cus_") and is_id_like(IDS[cust.key])
    assert fake_stripe_ids(COMPANY) == IDS  # deterministic


@pytest.mark.anyio
async def test_retrieve_charge_dollars_at_boundary(server) -> None:  # type: ignore[no-untyped-def]
    err, ch = await call(server, "retrieve_charge", charge_id=CHARGE_ID)
    assert not err
    assert ch["id"] == CHARGE_ID
    assert ch["amount"] == DOLLARS
    assert ch["amount_refunded"] == 0
    assert ch["customer"] == CUSTOMER_ID
    assert ch["currency"] == "usd"


@pytest.mark.anyio
async def test_list_charges_wrapped_filtered_limited(server) -> None:  # type: ignore[no-untyped-def]
    err, res = await call(server, "list_charges", customer_id=CUSTOMER_ID)
    assert not err
    assert set(res) == {"data"}
    want = [IDS[c.key] for c in COMPANY.charges if c.customer_key == CHARGE.customer_key]
    assert sorted(x["id"] for x in res["data"]) == sorted(want)
    _, res = await call(server, "list_charges", limit=3)
    assert len(res["data"]) == 3
    assert all(is_id_like(x["id"]) for x in res["data"])


@pytest.mark.anyio
async def test_refund_round_trip_and_accumulates(server) -> None:  # type: ignore[no-untyped-def]
    err, r = await call(
        server, "create_refund", charge_id=CHARGE_ID, amount=1.25, reason="requested_by_customer"
    )
    assert not err
    assert r["id"].startswith("re_") and is_id_like(r["id"])
    assert r["amount"] == 1.25
    assert r["charge"] == CHARGE_ID
    await call(server, "create_refund", charge_id=CHARGE_ID, amount=2.0, reason="duplicate")
    _, ch = await call(server, "retrieve_charge", charge_id=CHARGE_ID)
    assert ch["amount_refunded"] == 3.25


@pytest.mark.anyio
async def test_over_refund_is_validation_error_mentioning_amount(server) -> None:  # type: ignore[no-untyped-def]
    err, msg = await call(
        server, "create_refund", charge_id=CHARGE_ID, amount=DOLLARS + 1, reason="duplicate"
    )
    assert err
    assert f"{DOLLARS + 1:.2f}" in msg and f"{DOLLARS:.2f}" in msg
    # remaining refundable shrinks after a partial refund
    await call(server, "create_refund", charge_id=CHARGE_ID, amount=1.0, reason="duplicate")
    err, msg = await call(
        server, "create_refund", charge_id=CHARGE_ID, amount=DOLLARS, reason="duplicate"
    )
    assert err
    assert f"{DOLLARS - 1:.2f}" in msg


@pytest.mark.anyio
@pytest.mark.parametrize("amount", [0, -5.0, float("nan")])
async def test_bad_refund_amount_is_error(server, amount) -> None:  # type: ignore[no-untyped-def]
    err, _ = await call(server, "create_refund", charge_id=CHARGE_ID, amount=amount, reason="x")
    assert err


@pytest.mark.anyio
async def test_unknown_charge_and_customer_are_errors(server) -> None:  # type: ignore[no-untyped-def]
    err, msg = await call(server, "retrieve_charge", charge_id="ch_doesnotexist0000")
    assert err and "ch_doesnotexist0000" in msg
    err, msg = await call(
        server, "create_refund", charge_id="ch_doesnotexist0000", amount=1.0, reason="x"
    )
    assert err and "ch_doesnotexist0000" in msg
    err, msg = await call(server, "retrieve_customer", customer_id="cus_doesnotexist0000")
    assert err and "cus_doesnotexist0000" in msg


@pytest.mark.anyio
async def test_retrieve_customer(server) -> None:  # type: ignore[no-untyped-def]
    err, cus = await call(server, "retrieve_customer", customer_id=CUSTOMER_ID)
    assert not err
    expected = next(c for c in COMPANY.customers if c.key == CHARGE.customer_key)
    assert cus["id"] == CUSTOMER_ID
    assert cus["email"] == expected.email


@pytest.mark.anyio
async def test_payout_and_insufficient_balance_guard(server) -> None:  # type: ignore[no-untyped-def]
    err, po = await call(server, "create_payout", amount=10.5)
    assert not err
    assert po["id"].startswith("po_") and is_id_like(po["id"])
    assert po["amount"] == 10.5
    err, msg = await call(server, "create_payout", amount=1e9, destination="acct_ops1")
    assert err
    assert "insufficient" in msg.lower()
    err, _ = await call(server, "create_payout", amount=0)
    assert err


def _live_cfg() -> SandboxConfig:
    missing = [n.upper() for n in SandboxConfig.REQUIRED_ENV if n.upper() not in os.environ]
    if missing:
        pytest.skip(f"missing env: {missing}")
    return SandboxConfig.from_env()


@pytest.mark.live
@pytest.mark.anyio
async def test_live_refund_appears_in_stripe() -> None:
    import stripe

    from agentfw.sandbox.seed.stripe_seed import seed_stripe

    cfg = _live_cfg()
    ids = seed_stripe(cfg, COMPANY)
    live_charge = ids[CHARGE.key]
    srv = build_server(LiveStripeBackend(cfg))
    err, refund = await call(
        srv, "create_refund", charge_id=live_charge, amount=1.0, reason="requested_by_customer"
    )
    assert not err
    client = stripe.StripeClient(cfg.stripe_secret_key.get_secret_value())
    listed = client.refunds.list({"charge": live_charge})
    assert refund["id"] in [r.id for r in listed.data]
