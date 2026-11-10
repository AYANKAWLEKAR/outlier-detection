"""Zendesk MCP server: a local SQLite store (D-018), not a real API."""

from __future__ import annotations

import sqlite3
import tempfile
import threading
from pathlib import Path
from typing import Any, Protocol

from mcp.server.mcpserver import MCPServer

from agentfw.sandbox.config import SandboxConfig
from agentfw.sandbox.seed.company import Company
from agentfw.sandbox.seed.zendesk_seed import SCHEMA, seed_zendesk
from agentfw.sandbox.tools.common import ok, tool_error
from agentfw.sandbox.tools.stripe_server import fake_stripe_ids

STATUSES = ("new", "open", "pending", "hold", "solved", "closed")
DB_NAME = "zendesk.db"

_COMMENTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS comments (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_key TEXT NOT NULL,
    body TEXT NOT NULL,
    public INTEGER NOT NULL
)
"""
_TICKET_COLS = "key, subject, body, status, customer_id, customer_email, charge_id"


class ZendeskBackend(Protocol):
    def get_ticket(self, ticket_id: str) -> dict[str, Any]: ...
    def list_tickets(self, status: str, limit: int) -> list[dict[str, Any]]: ...
    def add_comment(self, ticket_id: str, body: str, public: bool) -> dict[str, Any]: ...
    def update_status(self, ticket_id: str, status: str) -> dict[str, Any]: ...


def _comment_id(seq: int) -> str:
    return f"cm_{seq:010d}"


def _check_status(status: str) -> None:
    if status not in STATUSES:
        raise tool_error(f"invalid status {status!r}; status must be one of {', '.join(STATUSES)}")


class SqliteZendeskBackend:
    """Both Fake and Live use this one code path; they differ only in where the DB lives."""

    def __init__(self, con: sqlite3.Connection) -> None:
        self._con = con
        self._lock = threading.Lock()
        with self._lock:
            self._con.execute(SCHEMA)
            self._con.execute(_COMMENTS_SCHEMA)
            self._con.commit()

    def _row(self, ticket_id: str) -> sqlite3.Row:
        row = self._con.execute(
            f"SELECT {_TICKET_COLS} FROM tickets WHERE key = ?", (ticket_id,)
        ).fetchone()
        if row is None:
            raise tool_error(f"ticket {ticket_id!r} not found")
        assert isinstance(row, sqlite3.Row)
        return row

    @staticmethod
    def _ticket(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["key"],
            "subject": row["subject"],
            "description": row["body"],
            "status": row["status"],
            "customer_id": row["customer_id"],
            "customer_email": row["customer_email"],
            "charge_id": row["charge_id"],
        }

    def snapshot(self) -> bytes:
        """The whole database as bytes (for exact restore after a prompt injection)."""
        with self._lock:
            return self._con.serialize()

    def restore(self, snap: bytes) -> None:
        with self._lock:
            self._con.deserialize(snap)

    def set_ticket_body(self, ticket_id: str, body: str) -> None:
        """Overwrite a ticket's body (test/injection hook; not an MCP tool)."""
        with self._lock:
            self._row(ticket_id)
            self._con.execute("UPDATE tickets SET body = ? WHERE key = ?", (body, ticket_id))
            self._con.commit()

    def get_ticket(self, ticket_id: str) -> dict[str, Any]:
        with self._lock:
            ticket = self._ticket(self._row(ticket_id))
            comments = self._con.execute(
                "SELECT seq, body, public FROM comments WHERE ticket_key = ? ORDER BY seq",
                (ticket_id,),
            ).fetchall()
        ticket["comments"] = [
            {"id": _comment_id(c["seq"]), "body": c["body"], "public": bool(c["public"])}
            for c in comments
        ]
        return ticket

    def list_tickets(self, status: str, limit: int) -> list[dict[str, Any]]:
        _check_status(status)
        with self._lock:
            rows = self._con.execute(
                f"SELECT {_TICKET_COLS} FROM tickets WHERE status = ? ORDER BY key LIMIT ?",
                (status, max(1, min(limit, 100))),
            ).fetchall()
        return [self._ticket(r) for r in rows]

    def add_comment(self, ticket_id: str, body: str, public: bool) -> dict[str, Any]:
        with self._lock:
            self._row(ticket_id)
            cur = self._con.execute(
                "INSERT INTO comments (ticket_key, body, public) VALUES (?, ?, ?)",
                (ticket_id, body, int(public)),
            )
            self._con.commit()
            seq = cur.lastrowid
        assert seq is not None
        return {"id": _comment_id(seq), "ticket_id": ticket_id, "body": body, "public": public}

    def update_status(self, ticket_id: str, status: str) -> dict[str, Any]:
        _check_status(status)
        with self._lock:
            self._row(ticket_id)
            self._con.execute("UPDATE tickets SET status = ? WHERE key = ?", (status, ticket_id))
            self._con.commit()
            return self._ticket(self._row(ticket_id))


def _connect(path: str | Path) -> sqlite3.Connection:
    con = sqlite3.connect(path, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


class FakeZendeskBackend(SqliteZendeskBackend):
    """In-memory, deterministic, built from `Company` via the real zendesk seeder."""

    def __init__(self, company: Company) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / DB_NAME
            seed_zendesk(db, company, fake_stripe_ids(company))
            src = sqlite3.connect(db)
            mem = _connect(":memory:")
            try:
                src.backup(mem)
            finally:
                src.close()
        super().__init__(mem)


class LiveZendeskBackend(SqliteZendeskBackend):
    """The 'live' zendesk is the file-backed SQLite DB under `cfg.data_dir` (no network)."""

    def __init__(self, cfg: SandboxConfig) -> None:
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        super().__init__(_connect(cfg.data_dir / DB_NAME))


def build_server(backend: ZendeskBackend) -> MCPServer:
    server = MCPServer("zendesk")

    @server.tool()
    def get_ticket(ticket_id: str) -> dict[str, Any]:
        """Retrieve a ticket, including its comments."""
        return ok(backend.get_ticket(ticket_id))

    @server.tool()
    def list_tickets(status: str = "open", limit: int = 20) -> dict[str, Any]:
        """List tickets in a status (new, open, pending, hold, solved, closed)."""
        return ok(backend.list_tickets(status, limit))

    @server.tool()
    def add_comment(ticket_id: str, body: str, public: bool = True) -> dict[str, Any]:
        """Add a comment to a ticket."""
        return ok(backend.add_comment(ticket_id, body, public))

    @server.tool()
    def update_status(ticket_id: str, status: str) -> dict[str, Any]:
        """Set a ticket's status."""
        return ok(backend.update_status(ticket_id, status))

    return server
