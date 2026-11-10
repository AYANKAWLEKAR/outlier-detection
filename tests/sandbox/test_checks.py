"""Success checks run against backend Protocols (fakes here; live later)."""

from __future__ import annotations

import orjson
import pytest

from agentfw.sandbox.checks import (
    CheckContext,
    SuccessCheck,
    mailhog_has_message,
    sf_field_equals,
    slack_contains,
    stripe_refund_exists,
    zendesk_status,
)
from agentfw.sandbox.config import UnsafeConfigError
from agentfw.sandbox.seed.company import build_company
from agentfw.sandbox.tools.salesforce_server import FakeSalesforceBackend
from agentfw.sandbox.tools.slack_server import FakeSlackBackend
from agentfw.sandbox.tools.stripe_server import FakeStripeBackend, fake_stripe_ids
from agentfw.sandbox.tools.zendesk_server import FakeZendeskBackend

COMPANY = build_company(0)
MAILHOG = "http://localhost:8025"


def _payload(*msgs: tuple[list[str], str]) -> bytes:
    items = [
        {
            "To": [{"Mailbox": t.split("@")[0], "Domain": t.split("@")[1]} for t in to],
            "Content": {"Headers": {"Subject": [subj]}},
        }
        for to, subj in msgs
    ]
    return orjson.dumps({"total": len(items), "items": items})


def test_mailhog_has_message() -> None:
    urls: list[str] = []

    def fetch(url: str) -> bytes:
        urls.append(url)
        return _payload((["Bob@Acme.example"], "Your Refund is ready"))

    assert mailhog_has_message(MAILHOG, "bob@acme.example", "refund", fetch=fetch)
    assert urls == [f"{MAILHOG}/api/v2/messages"]
    assert not mailhog_has_message(MAILHOG, "bob@acme.example", "invoice", fetch=fetch)
    assert not mailhog_has_message(MAILHOG, "eve@acme.example", "refund", fetch=fetch)


@pytest.mark.parametrize(
    "url", ["http://evil.example:8025", "https://mailhog.com", "http://10.0.0.5"]
)
def test_mailhog_rejects_non_local(url: str) -> None:
    def boom(_: str) -> bytes:
        raise AssertionError("must not fetch")

    with pytest.raises(UnsafeConfigError):
        mailhog_has_message(url, "a@b.example", "x", fetch=boom)


def test_mailhog_accepts_compose_host() -> None:
    assert not mailhog_has_message(
        "http://mailhog:8025", "a@b.example", "x", fetch=lambda _: _payload()
    )


def test_stripe_refund_exists() -> None:
    be = FakeStripeBackend(COMPANY)
    cid = fake_stripe_ids(COMPANY)[COMPANY.charges[0].key]
    assert not stripe_refund_exists(be, cid)
    be.create_refund(cid, 1.0, "requested_by_customer")
    assert stripe_refund_exists(be, cid)
    assert not stripe_refund_exists(be, "ch_doesnotexist0000")


def test_sf_field_equals() -> None:
    be = FakeSalesforceBackend(COMPANY)
    rid = be.run_soql("SELECT Id FROM Contact LIMIT 1")[0]["Id"]
    assert not sf_field_equals(be, "Contact", rid, "Title", "VIP")
    be.update_record("Contact", rid, {"Title": "VIP"})
    assert sf_field_equals(be, "Contact", rid, "Title", "VIP")
    assert not sf_field_equals(be, "Contact", "003000000000000", "Title", "VIP")


def test_slack_contains() -> None:
    be = FakeSlackBackend("C1")
    assert not slack_contains(be, "refund done")
    be.post_message("C1", "the refund done today")
    assert slack_contains(be, "refund done")


def test_zendesk_status() -> None:
    be = FakeZendeskBackend(COMPANY)
    tid = COMPANY.tickets[0].key
    assert not zendesk_status(be, tid, "solved")
    be.update_status(tid, "solved")
    assert zendesk_status(be, tid, "solved")
    assert not zendesk_status(be, "tk_NOPE", "solved")


def test_success_check_dispatch() -> None:
    zd = FakeZendeskBackend(COMPANY)
    tid = COMPANY.tickets[0].key
    zd.update_status(tid, "closed")
    ctx = CheckContext(zendesk=zd)
    assert SuccessCheck("zendesk_status", {"ticket_id": tid, "status": "closed"}).evaluate(ctx)
    assert not SuccessCheck("zendesk_status", {"ticket_id": tid, "status": "open"}).evaluate(ctx)
    with pytest.raises(ValueError):
        SuccessCheck("bogus", {}).evaluate(ctx)
    with pytest.raises(ValueError):
        SuccessCheck("slack_contains", {"text": "x"}).evaluate(ctx)  # slack backend missing


def test_sf_field_equals_numeric_normalization() -> None:
    be = FakeSalesforceBackend(COMPANY)
    rid = be.run_soql("SELECT Id FROM Opportunity LIMIT 1")[0]["Id"]
    be.update_record("Opportunity", rid, {"Amount": 33770})
    assert sf_field_equals(be, "Opportunity", rid, "Amount", "33770")  # spec params are strings
    assert sf_field_equals(be, "Opportunity", rid, "Amount", "33770.0")
    assert not sf_field_equals(be, "Opportunity", rid, "Amount", "33771")
