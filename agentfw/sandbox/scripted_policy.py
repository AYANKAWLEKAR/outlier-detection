"""A deterministic stand-in for the agent model, so the whole sandbox pipeline runs offline.

``scripted_calls`` turns one rendered task into the tool calls a competent agent would make: one
call per ``expected_tools`` entry, in order, with arguments derived from the rendered slots and
chosen so the task's own success checks hold (the update writes the field a ``sf_field_equals``
check reads, the email goes to the ``mailhog_has_message`` recipient, the Slack post carries every
``slack_contains`` text, ...). ``compliant_call`` is the call an agent that *obeys* a planted
prompt-level payload would make (it satisfies the payload's ``MatchSpec``). ``build_turns`` wraps
calls as ``ChatTurn``\\s for ``ScriptedChatClient``: one tool call per step, then a final answer.
Each turn reports ``SCRIPTED_USAGE`` tokens so budgets and metering are exercised offline.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import orjson

from agentfw.sandbox.injector import PayloadSpec
from agentfw.sandbox.llm import ChatTurn, ToolCall, Usage, to_fn_name
from agentfw.sandbox.seed.company import Company
from agentfw.sandbox.tasks.schema import RenderedTask
from agentfw.sandbox.tools.salesforce_server import sf_id
from agentfw.sandbox.tools.stripe_server import fake_stripe_ids

__all__ = [
    "SCRIPTED_USAGE",
    "PolicyContext",
    "build_turns",
    "compliant_call",
    "scripted_calls",
]

SCRIPTED_USAGE = (800, 60)  # (prompt, completion) tokens reported per scripted turn
_FALLBACK_TO = "team-leads@helio-ops.example"
_PAYLOAD_ID = re.compile(r"\b(ch_[A-Za-z0-9]+|tk_[A-Za-z0-9]+|00[0-9A-Za-z][0-9A-Za-z]{12,15})\b")

Call = tuple[str, dict[str, Any]]


@dataclass(frozen=True, slots=True)
class PolicyContext:
    company: Company
    stripe_ids: dict[str, str]  # company key -> Stripe id
    sf_ids: dict[str, str]  # company key -> Salesforce id
    channel: str

    @classmethod
    def build(
        cls, company: Company, id_map: Mapping[str, str] | None, channel: str
    ) -> PolicyContext:
        """Fake-backend ids, overridden by a live run's ``id_map`` (seeders' real ids)."""
        overrides = dict(id_map or {})
        stripe = {**fake_stripe_ids(company), **overrides}
        sf = {a.key: sf_id("Account", a.key) for a in company.accounts}
        sf.update({c.key: sf_id("Contact", c.key) for c in company.contacts})
        sf.update({o.key: sf_id("Opportunity", o.key) for o in company.opportunities})
        sf.update({k: v for k, v in overrides.items() if k in sf})
        return cls(company, stripe, sf, channel)


def _coerce(value: str) -> Any:
    """Check values are strings; digit-only ones are numbers on the wire (``Amount``)."""
    return int(value) if value.isdigit() else value


class _TaskArgs:
    """Argument builders for one rendered task."""

    def __init__(self, task: RenderedTask, kinds: Mapping[str, str], ctx: PolicyContext) -> None:
        self.task = task
        self.ctx = ctx
        self.slots: dict[str, dict[str, str]] = {}
        for name, values in task.slot_values.items():
            self.slots.setdefault(kinds[name], values)
        self.checks: dict[str, list[dict[str, str]]] = {}
        for spec in task.success:
            self.checks.setdefault(spec.kind, []).append(dict(spec.params))
        self.occ: Counter[str] = Counter()

    def slot(self, kind: str) -> dict[str, str] | None:
        return self.slots.get(kind)

    # -- record resolution -----------------------------------------------------------------------

    def _ticket_customer_key(self) -> str | None:
        t = self.slot("ticket_open")
        if t is None:
            return None
        return next(x.customer_key for x in self.ctx.company.tickets if x.key == t["key"])

    def charge_id(self) -> str:
        if (t := self.slot("ticket_open")) is not None:
            return t["charge_id"]
        if (c := self.slot("charge_refundable")) is not None:
            return c["id"]
        if (cu := self.slot("customer")) is not None:
            key = next(x.key for x in self.ctx.company.charges if x.customer_key == cu["key"])
            return self.ctx.stripe_ids[key]
        raise ValueError(f"{self.task.template_id}: no charge to act on")

    def customer_id(self) -> str | None:
        if (cu := self.slot("customer")) is not None:
            return cu["stripe_id"]
        if (c := self.slot("charge_refundable")) is not None:
            return c["customer_stripe_id"]
        key = self._ticket_customer_key()
        return None if key is None else self.ctx.stripe_ids[key]

    def ticket_id(self) -> str:
        t = self.slot("ticket_open")
        if t is None:
            raise ValueError(f"{self.task.template_id}: no ticket to act on")
        return t["id"]

    def email(self) -> str:
        for kind, field in (
            ("ticket_open", "customer_email"),
            ("contact", "email"),
            ("customer", "email"),
            ("account", "contact_email"),
            ("opportunity", "contact_email"),
            ("soql_object", "contact_email"),
            ("charge_refundable", "customer_email"),
        ):
            if (s := self.slot(kind)) is not None:
                return s[field]
        return _FALLBACK_TO

    def sf_records(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = [
            (c["object"], c["record_id"]) for c in self.checks.get("sf_field_equals", [])
        ]
        if (o := self.slot("opportunity")) is not None:
            out += [("Opportunity", o["sf_id"]), ("Account", o["account_sf_id"])]
        if (c := self.slot("contact")) is not None:
            out += [("Contact", c["sf_id"]), ("Account", c["account_sf_id"])]
        if (a := self.slot("account")) is not None:
            out += [("Account", a["sf_id"]), ("Contact", a["contact_sf_id"])]
        if (q := self.slot("soql_object")) is not None:
            out.append((q["object"], q["first_id"]))
        out = list(dict.fromkeys(out))
        if not out:
            raise ValueError(f"{self.task.template_id}: no Salesforce record to act on")
        return out

    def sf_updates(self) -> list[tuple[str, str, dict[str, Any]]]:
        groups: dict[tuple[str, str], dict[str, Any]] = {}
        for c in self.checks.get("sf_field_equals", []):
            groups.setdefault((c["object"], c["record_id"]), {})[c["field"]] = _coerce(c["value"])
        if groups:
            return [(obj, rid, fields) for (obj, rid), fields in groups.items()]
        obj, rid = self.sf_records()[0]
        return [(obj, rid, {"Description": "Reviewed by the support agent"})]

    # -- per-tool arguments ----------------------------------------------------------------------

    def args(self, tool: str) -> dict[str, Any]:
        i = self.occ[tool]
        self.occ[tool] += 1
        tid = self.task.template_id
        match tool:
            case "zendesk.get_ticket":
                return {"ticket_id": self.ticket_id()}
            case "zendesk.list_tickets":
                return {"status": "open", "limit": 20}
            case "zendesk.add_comment":
                return {
                    "ticket_id": self.ticket_id(),
                    "body": f"Thanks for your patience, we are on ticket {self.ticket_id()}.",
                    "public": True,
                }
            case "zendesk.update_status":
                want = self.checks.get("zendesk_status", [{"status": "pending"}])
                return {
                    "ticket_id": self.ticket_id(),
                    "status": want[min(i, len(want) - 1)]["status"],
                }
            case "stripe.retrieve_charge":
                return {"charge_id": self.charge_id()}
            case "stripe.list_charges":
                cid = self.customer_id()
                return {"limit": 20} if cid is None else {"customer_id": cid, "limit": 20}
            case "stripe.retrieve_customer":
                cid = self.customer_id()
                if cid is None:
                    raise ValueError(f"{tid}: no customer to look up")
                return {"customer_id": cid}
            case "stripe.create_refund":
                src = self.slot("ticket_open") or self.slot("charge_refundable") or {}
                half = "half" in self.task.prompt.lower() and "half_amount" in src
                amount = src.get("half_amount" if half else "amount")
                if amount is None:
                    raise ValueError(f"{tid}: no amount to refund")
                return {
                    "charge_id": self.charge_id(),
                    "amount": float(amount),
                    "reason": "requested_by_customer",
                }
            case "stripe.create_payout":
                c = self.slot("charge_refundable")
                payout = float(c["amount"]) if c is not None else 50.0
                return {"amount": payout, "currency": "usd"}
            case "mail.send_email":
                want = self.checks.get("mailhog_has_message", [])
                if want:
                    w = want[min(i, len(want) - 1)]
                    to, subject = w["to"], w["subject_contains"]
                else:
                    to, subject = self.email(), "Update on your request"
                body = f"Hello,\n\nHere is the update you asked for ({tid}).\n\nThanks,\nHelio"
                return {"to": [to], "subject": subject, "body": body}
            case "slack.post_message":
                texts = [c["text"] for c in self.checks.get("slack_contains", [])]
                chan = self.slot("channel_topic")
                text = " | ".join(texts) or f"Update {chan['ref'] if chan else tid}"
                return {"channel": self.ctx.channel, "text": text}
            case "slack.read_channel":
                return {"limit": 20}
            case "salesforce.run_soql":
                q = self.slot("soql_object")
                return {"query": q["query"] if q else "SELECT Id, Name FROM Opportunity LIMIT 20"}
            case "salesforce.get_record":
                recs = self.sf_records()
                obj, rid = recs[min(i, len(recs) - 1)]
                return {"object": obj, "record_id": rid}
            case "salesforce.update_record":
                ups = self.sf_updates()
                obj, rid, fields = ups[min(i, len(ups) - 1)]
                return {"object": obj, "record_id": rid, "fields": fields}
            case "salesforce.mass_update":
                ups = self.sf_updates()
                obj, rid, fields = ups[min(i, len(ups) - 1)]
                return {"object": obj, "record_ids": [rid], "field_updates": fields}
            case "salesforce.create_task":
                _, rid = self.sf_records()[0]
                return {"what_id": rid, "subject": f"Follow up ({tid})"}
        raise ValueError(f"{tid}: the scripted policy cannot call {tool!r}")


def scripted_calls(task: RenderedTask, kinds: Mapping[str, str], ctx: PolicyContext) -> list[Call]:
    """One call per expected tool, in order; ``kinds`` maps each slot name to its kind."""
    builder = _TaskArgs(task, kinds, ctx)
    return [(tool, builder.args(tool)) for tool in task.expected_tools]


def compliant_call(payload: PayloadSpec, ctx: PolicyContext, *, charge_id: str) -> Call:
    """The call an agent obeying ``payload`` would make. ``charge_id`` is the carrier's charge."""
    m = payload.match
    planted_ids = _PAYLOAD_ID.findall(payload.text)
    planted = planted_ids[0] if planted_ids else "unknown_0000000000"
    value = m.value
    match (m.tool_id, m.op):
        case ("mail.send_email", "domain_eq"):
            to = [f"inbox@{value}"]
            return m.tool_id, {
                "to": to,
                "subject": "As requested",
                "body": "Attached as requested.",
            }
        case ("mail.send_email", "len_ge"):
            n = int(float(str(value)))
            to = [c.email for c in ctx.company.customers][:n]
            return m.tool_id, {"to": to, "subject": "Notice", "body": "Please read this notice."}
        case ("stripe.list_charges", "ge"):
            return m.tool_id, {"limit": int(float(str(value)))}
        case ("stripe.create_refund", "ge"):
            amount = float(str(value))
            return m.tool_id, {"charge_id": charge_id, "amount": amount, "reason": "goodwill"}
        case ("stripe.create_refund", "id_not_seen"):
            return m.tool_id, {
                "charge_id": planted,
                "amount": 25.0,
                "reason": "requested_by_customer",
            }
        case ("salesforce.update_record", "id_not_seen"):
            fields = {"Description": "Corrected per data-fix note"}
            return m.tool_id, {"object": "Account", "record_id": planted, "fields": fields}
        case ("zendesk.add_comment", "id_not_seen"):
            return m.tool_id, {"ticket_id": planted, "body": "Resolution note.", "public": False}
        case ("salesforce.mass_update", "len_ge"):
            n = int(float(str(value)))
            ids = [ctx.sf_ids[o.key] for o in ctx.company.opportunities][:n]
            return m.tool_id, {
                "object": "Opportunity",
                "record_ids": ids,
                "field_updates": {"StageName": "Closed Won"},
            }
    raise ValueError(f"payload {payload.id}: no compliant call for {m.tool_id} {m.op}")


def build_turns(calls: list[Call]) -> list[ChatTurn]:
    """One tool call per turn, then a final answer; each turn reports ``SCRIPTED_USAGE``."""
    p, c = SCRIPTED_USAGE
    turns = [
        ChatTurn(
            content=None,
            tool_calls=[
                ToolCall(
                    id=f"call_{i:03d}",
                    name=to_fn_name(tool),
                    arguments_json=orjson.dumps(args).decode(),
                )
            ],
            usage=Usage(p, c),
        )
        for i, (tool, args) in enumerate(calls)
    ]
    turns.append(
        ChatTurn(content="Done: worked through the queue.", tool_calls=[], usage=Usage(p, c))
    )
    return turns
