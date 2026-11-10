"""Synthetic company builder."""

from __future__ import annotations

from agentfw.sandbox.seed.company import build_company


def test_deterministic() -> None:
    assert build_company(0) == build_company(0)
    assert build_company(0) != build_company(1)


def test_salesforce_record_cap() -> None:
    for seed in (0, 1, 7):
        assert build_company(seed).salesforce_record_count() <= 2000


def test_all_domains_reserved() -> None:
    c = build_company(0)
    emails = [x.email for x in c.customers] + [x.email for x in c.contacts]
    emails += [t.customer_email for t in c.tickets]
    assert emails
    for e in emails:
        assert e.rsplit("@", 1)[1].endswith((".example", ".test")), e
    for a in c.accounts:
        assert a.domain.endswith((".example", ".test")), a.domain


def test_keys_unique_and_references_resolve() -> None:
    c = build_company(3)
    for items in (c.customers, c.contacts, c.accounts, c.opportunities, c.tickets, c.charges):
        keys = [i.key for i in items]
        assert len(keys) == len(set(keys))
    cust = {x.key for x in c.customers}
    acct = {x.key for x in c.accounts}
    charges = {x.key: x for x in c.charges}
    assert all(x.customer_key in cust for x in c.charges)
    assert all(x.account_key in acct for x in c.contacts)
    assert all(x.account_key in acct for x in c.opportunities)
    for t in c.tickets:
        assert t.charge_key in charges
        assert charges[t.charge_key].customer_key == t.customer_key
    emails = [x.email for x in c.customers]
    assert len(emails) == len(set(emails))
    assert all(x.amount_cents > 0 for x in c.charges)
