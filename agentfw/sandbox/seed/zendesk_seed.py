"""Seed the local SQLite 'zendesk' store (D-018)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from agentfw.sandbox.seed.company import Company

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    key TEXT PRIMARY KEY,
    subject TEXT NOT NULL,
    body TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    customer_id TEXT NOT NULL,
    customer_email TEXT NOT NULL,
    charge_id TEXT NOT NULL
)
"""


def seed_zendesk(db_path: Path, company: Company, stripe_ids: dict[str, str]) -> None:
    """Upsert one ticket per company ticket, linked to its Stripe charge. Idempotent."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    try:
        con.execute(SCHEMA)
        con.executemany(
            "INSERT INTO tickets (key, subject, body, customer_id, customer_email, charge_id) "
            "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(key) DO UPDATE SET "
            "subject=excluded.subject, body=excluded.body, customer_id=excluded.customer_id, "
            "customer_email=excluded.customer_email, charge_id=excluded.charge_id",
            [
                (
                    t.key,
                    t.subject,
                    t.body,
                    stripe_ids[t.customer_key],
                    t.customer_email,
                    stripe_ids[t.charge_key],
                )
                for t in company.tickets
            ],
        )
        con.commit()
    finally:
        con.close()
