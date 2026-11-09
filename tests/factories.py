"""Test factories shared across the suite."""

from __future__ import annotations

import itertools
from typing import Any

from agentfw.core.envelope import ActionEnvelope, RawCall
from agentfw.core.ids import ulid_from
from agentfw.core.normalize import SessionTracker, normalize

_counter = itertools.count()


def make_raw(**overrides: Any) -> RawCall:
    """Build a valid RawCall; a deterministic ULID envelope_id is generated unless overridden."""
    fields: dict[str, Any] = {
        "session_id": "s1",
        "agent_id": "a1",
        "principal_id": "p1",
        "seq_no": 0,
        "ts_unix_ms": 1_700_000_000_000,
        "tool_id": "stripe.create_refund",
        "args": {"amount": 37.5},
    }
    fields.update(overrides)
    if "envelope_id" not in fields:
        fields["envelope_id"] = ulid_from(int(fields["ts_unix_ms"]), next(_counter))
    return RawCall(**fields)


def make_env(
    raw: RawCall | None = None,
    tracker: SessionTracker | None = None,
    **raw_overrides: Any,
) -> ActionEnvelope:
    """Normalize a raw call (built from overrides if None). Does NOT observe it."""
    if raw is None:
        raw = make_raw(**raw_overrides)
    return normalize(raw, tracker if tracker is not None else SessionTracker())
