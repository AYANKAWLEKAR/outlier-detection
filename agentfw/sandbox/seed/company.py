"""Deterministic synthetic company: reserved domains only, synthetic names only (D-016)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SF_RECORD_CAP = 2000  # Developer Edition storage

_ONSET = ["b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z", "br", "tr", "kl"]
_VOWEL = ["a", "e", "i", "o", "u", "ai", "ou"]
_CODA = ["", "", "n", "r", "l", "s", "m", "x"]
_TLDS = ("example", "test")
_STAGES = ["Prospecting", "Qualification", "Proposal", "Negotiation", "Closed Won"]
_SUBJECTS = [
    "Refund request for a recent charge",
    "Charge amount looks wrong",
    "Duplicate charge on my account",
    "Question about my invoice",
    "Please confirm payment received",
]


@dataclass(frozen=True, slots=True)
class Customer:
    key: str
    name: str
    email: str


@dataclass(frozen=True, slots=True)
class Account:
    key: str
    name: str
    domain: str
    customer_key: str


@dataclass(frozen=True, slots=True)
class Contact:
    key: str
    account_key: str
    first_name: str
    last_name: str
    email: str


@dataclass(frozen=True, slots=True)
class Opportunity:
    key: str
    account_key: str
    name: str
    stage: str
    amount: int
    close_day: int  # days from a fixed reference date


@dataclass(frozen=True, slots=True)
class ChargePlan:
    key: str
    customer_key: str
    amount_cents: int
    currency: str
    description: str


@dataclass(frozen=True, slots=True)
class Ticket:
    key: str
    customer_key: str
    customer_email: str
    charge_key: str
    subject: str
    body: str


@dataclass(frozen=True, slots=True)
class Company:
    customers: tuple[Customer, ...]
    accounts: tuple[Account, ...]
    contacts: tuple[Contact, ...]
    opportunities: tuple[Opportunity, ...]
    charges: tuple[ChargePlan, ...]
    tickets: tuple[Ticket, ...]

    def salesforce_record_count(self) -> int:
        return len(self.accounts) + len(self.contacts) + len(self.opportunities)


def _word(rng: np.random.Generator, syllables: int) -> str:
    parts = []
    for _ in range(syllables):
        parts.append(str(rng.choice(_ONSET)) + str(rng.choice(_VOWEL)) + str(rng.choice(_CODA)))
    return "".join(parts)


def build_company(seed: int, n_customers: int = 80) -> Company:
    rng = np.random.default_rng([seed, 2001])
    customers: list[Customer] = []
    accounts: list[Account] = []
    contacts: list[Contact] = []
    opps: list[Opportunity] = []
    charges: list[ChargePlan] = []
    tickets: list[Ticket] = []
    used_domains: set[str] = set()
    used_emails: set[str] = set()

    def unique_email(first: str, last: str, domain: str) -> str:
        base = f"{first}.{last}"
        email, i = f"{base}@{domain}", 1
        while email in used_emails:
            i += 1
            email = f"{base}{i}@{domain}"
        used_emails.add(email)
        return email

    for i in range(n_customers):
        company = _word(rng, int(rng.integers(2, 4)))
        domain = f"{company}.{_TLDS[int(rng.integers(0, 2))]}"
        while domain in used_domains:
            company += "a"
            domain = f"{company}.{_TLDS[int(rng.integers(0, 2))]}"
        used_domains.add(domain)
        ckey = f"cust-{i:04d}"
        first, last = _word(rng, 2), _word(rng, 2)
        customers.append(
            Customer(ckey, f"{first.title()} {last.title()}", unique_email(first, last, domain))
        )
        accounts.append(Account(f"acct-{i:04d}", f"{company.title()} Ltd", domain, ckey))
        for j in range(int(rng.integers(1, 4))):
            f2, l2 = _word(rng, 2), _word(rng, 2)
            contacts.append(
                Contact(
                    f"cont-{i:04d}-{j}",
                    f"acct-{i:04d}",
                    f2.title(),
                    l2.title(),
                    unique_email(f2, l2, domain),
                )
            )
        for j in range(int(rng.integers(1, 4))):
            opps.append(
                Opportunity(
                    f"opp-{i:04d}-{j}",
                    f"acct-{i:04d}",
                    f"{company.title()} deal {j + 1}",
                    str(rng.choice(_STAGES)),
                    int(rng.integers(5, 500)) * 100,
                    int(rng.integers(30, 365)),
                )
            )
        for j in range(int(rng.integers(1, 4))):
            charges.append(
                ChargePlan(
                    f"chg-{i:04d}-{j}",
                    ckey,
                    int(rng.integers(5, 400)) * 100 + 50 * int(rng.integers(0, 2)),
                    "usd",
                    f"Synthetic order {i:04d}-{j}",
                )
            )

    customer_by_key = {c.key: c for c in customers}
    n_tickets = max(1, int(len(charges) * 0.5))
    for t, idx in enumerate(sorted(rng.choice(len(charges), size=n_tickets, replace=False))):
        ch = charges[int(idx)]
        subject = str(rng.choice(_SUBJECTS))
        tickets.append(
            Ticket(
                f"tkt-{t:04d}",
                ch.customer_key,
                customer_by_key[ch.customer_key].email,
                ch.key,
                subject,
                f"{subject}. Amount {ch.amount_cents / 100:.2f} {ch.currency.upper()}.",
            )
        )

    company_obj = Company(
        tuple(customers),
        tuple(accounts),
        tuple(contacts),
        tuple(opps),
        tuple(charges),
        tuple(tickets),
    )
    if company_obj.salesforce_record_count() > SF_RECORD_CAP:
        raise ValueError("synthetic company exceeds the Salesforce record cap")
    return company_obj
