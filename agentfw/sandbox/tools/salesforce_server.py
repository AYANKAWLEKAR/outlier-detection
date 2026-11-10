"""Salesforce MCP server (Developer Edition only): in-memory Fake, simple_salesforce Live."""

from __future__ import annotations

import copy
import hashlib
import re
import threading
from collections.abc import Mapping
from typing import Any, Protocol

import orjson
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from agentfw.sandbox.config import ApiBudget, BudgetExceeded, SandboxConfig, UnsafeConfigError
from agentfw.sandbox.seed.company import Company
from agentfw.sandbox.tools.common import mark_fake, ok, tool_error

SOQL_MAX_LIMIT = 2000
COLLECTION_CHUNK = 200  # sObject Collections accept at most 200 records per request
OBJECTS = ("Account", "Contact", "Opportunity", "Task", "Lead", "Case")
_KEY_PREFIX = {
    "Account": "001",
    "Contact": "003",
    "Opportunity": "006",
    "Task": "00T",
    "Lead": "00Q",
    "Case": "500",
}
_FORBIDDEN = re.compile(r"\b(delete|update|insert|upsert|merge|undelete)\b", re.IGNORECASE)
_LIMIT_ANY = re.compile(r"\blimit\s+([^\s)]+)", re.IGNORECASE)
_LIMIT_END = re.compile(r"\blimit\s+\d+(\s+offset\s+\d+)?\s*$", re.IGNORECASE)
_STRING = re.compile(r"'(?:[^'\\]|\\.)*'")
_COMMENT = re.compile(r"/\*|\*/|//|--")
_RECORD_ID = re.compile(r"^[A-Za-z0-9]{15,18}$")
_RESERVED_FIELDS = frozenset({"id", "attributes"})


ORG_QUERY = "SELECT OrganizationType, IsSandbox FROM Organization"
# The error code every budget refusal starts with; the runner matches it to mark a session
# ``budget_truncated`` (never a substring of arbitrary tool output).
BUDGET_ERROR_CODE = "SF_API_BUDGET_EXCEEDED"


def budget_error(exc: BudgetExceeded) -> ToolError:
    """The tool error for a spent budget: ``SF_API_BUDGET_EXCEEDED: budget exceeded: ...``."""
    return tool_error(f"{BUDGET_ERROR_CODE}: budget exceeded: {exc}")


DEVELOPER_EDITION = "Developer Edition"


def require_developer_org(sf: Any) -> None:
    """Refuse (``UnsafeConfigError``) unless the logged-in org is a Developer Edition or a sandbox.

    ``sf`` is a ``simple_salesforce.Salesforce`` (or a stand-in with its ``query``). An org that
    cannot be read, or a query error, refuses too (fail closed)."""
    res = sf.query(ORG_QUERY)
    records = res.get("records") if isinstance(res, Mapping) else None
    if not records or not isinstance(records[0], Mapping):
        raise UnsafeConfigError(
            f"Salesforce org refused: cannot read its Organization record; only a "
            f"{DEVELOPER_EDITION} or sandbox org is allowed"
        )
    org_type, is_sandbox = records[0].get("OrganizationType"), records[0].get("IsSandbox")
    if org_type == DEVELOPER_EDITION or is_sandbox is True:
        return
    raise UnsafeConfigError(
        f"Salesforce org refused: OrganizationType={org_type!r}, IsSandbox={is_sandbox!r}; "
        f"only a {DEVELOPER_EDITION} or sandbox org is allowed"
    )


class SalesforceBackend(Protocol):
    def run_soql(self, query: str) -> list[dict[str, Any]]: ...
    def get_record(self, obj: str, record_id: str) -> dict[str, Any]: ...
    def update_record(self, obj: str, record_id: str, fields: dict[str, Any]) -> dict[str, Any]: ...
    def mass_update_batch(self, obj: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """One sObject Collections request (<= 200 records, each {"Id": ..., **fields})."""
        ...

    def create_task(self, what_id: str, subject: str) -> dict[str, Any]: ...


def validate_soql(query: str) -> str:
    """Return the query to run (SELECT only, every LIMIT <= 2000) or raise a ToolError.

    Checks run on a copy with string literals masked; the original text is what is returned
    (plus an appended LIMIT when absent). SOQL has no comments, so comment markers are errors.
    """
    q = query.strip()
    masked = _STRING.sub("''", q)
    if "'" in masked.replace("''", ""):
        raise tool_error("SOQL has an unterminated string literal")
    if _COMMENT.search(masked):
        raise tool_error("SOQL comments are not supported")
    if not re.match(r"select\b", masked, re.IGNORECASE):
        raise tool_error("run_soql accepts SOQL SELECT queries only")
    if ";" in masked:
        raise tool_error("run_soql accepts a single SOQL SELECT statement")
    if _FORBIDDEN.search(masked):
        raise tool_error("run_soql accepts SOQL SELECT queries only (DML keyword found)")
    for m in _LIMIT_ANY.finditer(masked):
        if not m.group(1).isdigit() or int(m.group(1)) > SOQL_MAX_LIMIT:
            raise tool_error(f"SOQL LIMIT must be a number <= {SOQL_MAX_LIMIT}")
    if _LIMIT_END.search(masked) is None:
        q = f"{q} LIMIT {SOQL_MAX_LIMIT}"
    return q


def check_record_id(record_id: str) -> str:
    if _RECORD_ID.match(record_id) is None:
        raise tool_error(f"invalid record id {record_id!r}")
    return record_id


def check_fields(fields: dict[str, Any], what: str) -> None:
    if not fields:
        raise tool_error(f"{what} must not be empty")
    bad = sorted(k for k in fields if k.lower() in _RESERVED_FIELDS)
    if bad:
        raise tool_error(f"{what} must not contain reserved keys: {', '.join(bad)}")


def _check_object(obj: str) -> None:
    if obj not in OBJECTS:
        raise tool_error(f"unsupported object {obj!r}; supported: {', '.join(OBJECTS)}")


def sf_id(obj: str, seed: str) -> str:
    """15-char Salesforce-style id: 3-char key prefix (has a digit) + 12 hex chars."""
    return _KEY_PREFIX[obj] + hashlib.sha1(f"{obj}:{seed}".encode()).hexdigest()[:12].upper()


class FakeSalesforceBackend:
    """In-memory, deterministic org built from `Company` (same shape as the seeder writes)."""

    is_fake = True

    def __init__(self, company: Company, ids: Mapping[str, str] | None = None) -> None:
        """``ids`` ({company key: Salesforce Id}) overrides the fake ids (e.g. a live run's)."""
        self._lock = threading.Lock()
        self._rows: dict[str, dict[str, dict[str, Any]]] = {o: {} for o in OBJECTS}
        self._ids: dict[str, str] = {}
        self._override = dict(ids or {})
        self._n_tasks = 0
        for a in company.accounts:
            self._add("Account", a.key, {"Name": a.name, "Website": a.domain})
        for c in company.contacts:
            self._add(
                "Contact",
                c.key,
                {
                    "AccountId": self._ids[c.account_key],
                    "FirstName": c.first_name,
                    "LastName": c.last_name,
                    "Email": c.email,
                },
            )
        for o in company.opportunities:
            self._add(
                "Opportunity",
                o.key,
                {
                    "AccountId": self._ids[o.account_key],
                    "Name": o.name,
                    "StageName": o.stage,
                    "Amount": o.amount,
                },
            )

    def _add(self, obj: str, key: str, fields: dict[str, Any]) -> None:
        rid = self._override.get(key) or sf_id(obj, key)
        self._ids[key] = rid
        self._rows[obj][rid] = {"Id": rid, **fields}

    def ids(self) -> dict[str, str]:
        """{company record key: Salesforce Id}, like `seed_salesforce` returns."""
        return dict(self._ids)

    def snapshot(self) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, str], int]:
        with self._lock:
            return copy.deepcopy((self._rows, self._ids, self._n_tasks))

    def restore(
        self, snap: tuple[dict[str, dict[str, dict[str, Any]]], dict[str, str], int]
    ) -> None:
        with self._lock:
            self._rows, self._ids, self._n_tasks = copy.deepcopy(snap)

    def _find(self, obj: str, record_id: str) -> dict[str, Any]:
        _check_object(obj)
        row = self._rows[obj].get(record_id)
        if row is None:
            raise tool_error(f"NOT_FOUND: no {obj} with id {record_id!r}")
        return row

    def run_soql(self, query: str) -> list[dict[str, Any]]:
        m = re.match(
            r"select\s+(.+?)\s+from\s+(\w+)(?:\s+where\s+(.+?))?(?:\s+limit\s+(\d+))?\s*$",
            validate_soql(query),
            re.IGNORECASE | re.DOTALL,
        )
        if m is None:
            raise tool_error("unsupported SOQL shape for the fake org")
        fields_s, obj, where, limit = m.groups()
        obj = next((o for o in OBJECTS if o.lower() == obj.lower()), obj)
        _check_object(obj)
        conds = []
        for part in re.split(r"\s+and\s+", where or "", flags=re.IGNORECASE) if where else []:
            cm = re.fullmatch(r"\s*(\w+)\s*=\s*'([^']*)'\s*", part)
            if cm is None:
                raise tool_error("fake org supports only WHERE Field = 'value' [AND ...]")
            conds.append((cm.group(1), cm.group(2)))
        cols = [f.strip() for f in fields_s.split(",")]
        with self._lock:
            rows = [
                r for r in self._rows[obj].values() if all(str(r.get(k)) == v for k, v in conds)
            ]
            rows = rows[: int(limit) if limit else SOQL_MAX_LIMIT]
            return [
                {c: r.get(c) for c in (cols if cols != ["*"] else list(r))} | {"Id": r["Id"]}
                for r in rows
            ]

    def get_record(self, obj: str, record_id: str) -> dict[str, Any]:
        with self._lock:
            return dict(self._find(obj, record_id))

    def update_record(self, obj: str, record_id: str, fields: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._find(obj, record_id).update(fields)
        return {"id": record_id, "success": True}

    def mass_update_batch(self, obj: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        _check_object(obj)
        if len(records) > COLLECTION_CHUNK:
            raise tool_error(f"sObject Collections accept at most {COLLECTION_CHUNK} records")
        out: list[dict[str, Any]] = []
        with self._lock:
            for rec in records:
                row = self._rows[obj].get(str(rec["Id"]))
                if row is None:
                    out.append({"id": rec["Id"], "success": False, "errors": ["NOT_FOUND"]})
                else:
                    row.update({k: v for k, v in rec.items() if k != "Id"})
                    out.append({"id": rec["Id"], "success": True})
        return out

    def create_task(self, what_id: str, subject: str) -> dict[str, Any]:
        with self._lock:
            if not any(what_id in rows for rows in self._rows.values()):
                raise tool_error(f"NOT_FOUND: no record with id {what_id!r}")
            self._n_tasks += 1
            rid = sf_id("Task", str(self._n_tasks))
            self._rows["Task"][rid] = {"Id": rid, "WhatId": what_id, "Subject": subject}
        return {"id": rid, "success": True}


class LiveSalesforceBackend:
    """Developer Edition org via simple_salesforce; every HTTP request is one budget charge.

    Construction checks the org (``require_developer_org``, one charged request) and raises
    ``UnsafeConfigError`` for anything but a Developer Edition or sandbox org."""

    def __init__(
        self, cfg: SandboxConfig, client: Any | None = None, budget: ApiBudget | None = None
    ) -> None:
        self._budget = budget or ApiBudget(
            cfg.data_dir / "sf_api_budget.json", cfg.sf_daily_api_budget
        )
        if client is None:
            from simple_salesforce.api import Salesforce

            client = Salesforce(
                username=cfg.sf_username,
                password=cfg.sf_password.get_secret_value(),
                security_token=cfg.sf_security_token.get_secret_value(),
                domain=cfg.sf_domain,
            )
        self._budget.charge(1)
        require_developer_org(client)
        self._sf = client

    def _request(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            self._budget.charge(1)
        except BudgetExceeded as exc:
            raise budget_error(exc) from exc
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if type(exc).__module__.startswith("simple_salesforce"):
                raise tool_error(f"Salesforce error: {exc}") from exc
            raise

    def run_soql(self, query: str) -> list[dict[str, Any]]:
        res = self._request(self._sf.query, validate_soql(query))
        return [{k: v for k, v in r.items() if k != "attributes"} for r in res["records"]]

    def get_record(self, obj: str, record_id: str) -> dict[str, Any]:
        _check_object(obj)
        check_record_id(record_id)
        rec = self._request(getattr(self._sf, obj).get, record_id)
        return {k: v for k, v in dict(rec).items() if k != "attributes"}

    def update_record(self, obj: str, record_id: str, fields: dict[str, Any]) -> dict[str, Any]:
        _check_object(obj)
        check_record_id(record_id)
        self._request(getattr(self._sf, obj).update, record_id, fields)
        return {"id": record_id, "success": True}

    def mass_update_batch(self, obj: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        _check_object(obj)
        if len(records) > COLLECTION_CHUNK:
            raise tool_error(f"sObject Collections accept at most {COLLECTION_CHUNK} records")
        body = {
            "allOrNone": False,
            "records": [
                {"attributes": {"type": obj}, "id": r["Id"]}
                | {k: v for k, v in r.items() if k != "Id"}
                for r in records
            ],
        }
        res = self._request(
            self._sf.restful,
            "composite/sobjects",
            method="PATCH",
            data=orjson.dumps(body).decode(),
        )
        return [dict(r) for r in res]

    def create_task(self, what_id: str, subject: str) -> dict[str, Any]:
        check_record_id(what_id)
        res = self._request(self._sf.Task.create, {"WhatId": what_id, "Subject": subject})
        return {"id": res["id"], "success": bool(res.get("success", True))}


def build_server(backend: SalesforceBackend) -> MCPServer:
    server = MCPServer("salesforce")

    @server.tool()
    def run_soql(query: str) -> dict[str, Any]:
        """Run a SOQL SELECT query (LIMIT <= 2000; DML is rejected)."""
        return ok(backend.run_soql(validate_soql(query)))

    @server.tool()
    def get_record(object: str, record_id: str) -> dict[str, Any]:
        """Retrieve one record by id."""
        return ok(backend.get_record(object, check_record_id(record_id)))

    @server.tool()
    def update_record(object: str, record_id: str, fields: dict[str, Any]) -> dict[str, Any]:
        """Update fields on one record."""
        check_fields(fields, "fields")
        return ok(backend.update_record(object, check_record_id(record_id), fields))

    @server.tool()
    def mass_update(
        object: str, record_ids: list[str], field_updates: dict[str, Any]
    ) -> dict[str, Any]:
        """Apply the same field updates to many records (sObject Collections, 200 per request)."""
        if not record_ids:
            raise tool_error("record_ids must not be empty")
        check_fields(field_updates, "field_updates")
        _check_object(object)
        for rid in record_ids:
            check_record_id(rid)
        results: list[dict[str, Any]] = []
        batches = 0
        for i in range(0, len(record_ids), COLLECTION_CHUNK):
            chunk = [{**field_updates, "Id": rid} for rid in record_ids[i : i + COLLECTION_CHUNK]]
            try:
                results.extend(backend.mass_update_batch(object, chunk))
            except ToolError as exc:
                done = sum(1 for r in results if r.get("success"))
                raise tool_error(f"{exc} (updated={done} batches={batches})") from exc
            batches += 1
        failed = [r for r in results if not r.get("success")]
        return {"updated": len(results) - len(failed), "failed": failed, "batches": batches}

    @server.tool()
    def create_task(what_id: str, subject: str) -> dict[str, Any]:
        """Create a Task related to a record."""
        return ok(backend.create_task(check_record_id(what_id), subject))

    return mark_fake(server, backend)
