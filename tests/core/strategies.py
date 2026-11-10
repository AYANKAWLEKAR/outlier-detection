"""Hypothesis strategies: causally normalized envelope streams for engine property tests.

A stream is built from a few argument *templates* and a list of cheap integer steps (agent,
tool, template, jitter, time gap, session break), so long streams stay fast to generate and
keys actually get warm. ``include_degenerate=True`` adds the Review Focus 3 shapes: empty args
(n = 0 keys), constant streams (IQR = 0), single-tool agents (V = 1), NaN / ±inf, huge ints,
freetext, empty containers and nested lists of dicts.
"""

from __future__ import annotations

import math

from hypothesis import strategies as st

from agentfw.core.envelope import ActionEnvelope, JsonValue, Outcome, RawCall
from agentfw.core.ids import ulid_from
from agentfw.core.normalize import SessionTracker, normalize

__all__ = ["TOOLS", "envelope_streams", "outcomes"]

TOOLS = ("stripe.create_refund", "zendesk.get_ticket", "crm.update_record")
AGENTS = ("agent_a", "agent_b", "agent_c")
KEYS = ("amount", "count", "currency", "customer_id", "email", "url", "flag", "items")
T0 = 1_700_000_000_000
GAPS_MS = (0, 500, 5_000, 60_000, 600_000, 3_600_000, 86_400_000)

_ids = st.sampled_from(["cus_ABCDEFGH12", "cus_QRSTUVWX34", "ch_ZZZZYYYY99", "TCK-0001234"])
_clean_scalars: st.SearchStrategy[JsonValue] = st.one_of(
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    st.integers(min_value=-(10**6), max_value=10**6),
    st.sampled_from(["usd", "eur", "gbp", "37.50"]),
    _ids,
    st.sampled_from(["bob@acme.example", "Bob@ACME.example", "eve@evil.test"]),
    st.sampled_from(["https://acme.example/x", "https://bücher.example/"]),
    st.booleans(),
    st.none(),
)
_degenerate_scalars: st.SearchStrategy[JsonValue] = st.sampled_from(
    [
        math.nan,
        math.inf,
        -math.inf,
        0,
        0.0,
        -1e308,
        2**63,
        10**400,  # beyond float range: try_number saturates to inf
        -(10**400),
        10**5000,  # beyond the interpreter's str() digit limit
        "",
        "SECRET " * 10,  # freetext (>= 40 chars)
    ]
)


def _values(degenerate: bool) -> st.SearchStrategy[JsonValue]:
    scalars = st.one_of(_clean_scalars, _degenerate_scalars) if degenerate else _clean_scalars
    if not degenerate:
        return scalars
    containers: st.SearchStrategy[JsonValue] = st.one_of(
        st.just([]),
        st.just({}),
        st.lists(scalars, max_size=3),
        st.lists(st.fixed_dictionaries({"amount": scalars, "sku": _ids}), max_size=3),
    )
    return st.one_of(scalars, containers)


def _templates(degenerate: bool) -> st.SearchStrategy[list[dict[str, JsonValue]]]:
    min_keys = 0 if degenerate else 1
    one = st.dictionaries(
        st.sampled_from(KEYS), _values(degenerate), min_size=min_keys, max_size=len(KEYS)
    )
    return st.lists(one, min_size=1, max_size=4)


def _jitter(v: JsonValue, factor: float) -> JsonValue:
    """Scale ordinary finite numbers (not bools, not huge ints) so magnitudes vary."""
    if isinstance(v, bool) or not isinstance(v, int | float):
        return v
    if isinstance(v, int) and abs(v) > 10**12:
        return v
    x = float(v) * factor
    if not math.isfinite(x):
        return v
    return round(x) if isinstance(v, int) else x


@st.composite
def envelope_streams(
    draw: st.DrawFn, include_degenerate: bool = False, min_size: int = 1, max_size: int = 120
) -> list[ActionEnvelope]:
    """A non-empty, time-ordered list of envelopes normalized with one causal SessionTracker."""
    n_agents = draw(st.integers(1, len(AGENTS)))
    agent_tools = [
        draw(st.lists(st.sampled_from(TOOLS), min_size=1, max_size=len(TOOLS), unique=True))
        for _ in range(n_agents)
    ]
    templates = draw(_templates(include_degenerate))
    constant = include_degenerate and draw(st.booleans())
    n = draw(st.integers(min_size, max_size))
    steps = draw(
        st.lists(
            st.tuples(
                st.integers(0, n_agents - 1),
                st.integers(0, len(TOOLS) - 1),
                st.integers(0, len(templates) - 1),
                st.integers(0, 31),
                st.integers(0, len(GAPS_MS) - 1),
                st.integers(0, 7),
            ),
            min_size=n,
            max_size=n,
        )
    )
    tracker = SessionTracker()
    out: list[ActionEnvelope] = []
    ts = T0
    session = 0
    seq: dict[str, int] = {}
    for i, (ai, ti, tpl, jit, gap, brk) in enumerate(steps):
        ts += GAPS_MS[gap]
        if brk == 0:
            session += 1
        tools = agent_tools[ai]
        factor = 1.0 if constant else 0.5 + jit / 16.0
        args = {k: _jitter(v, factor) for k, v in templates[tpl].items()}
        sid = f"s{ai}_{session}"
        seq_no = seq.get(sid, 0)
        seq[sid] = seq_no + 1
        raw = RawCall(
            envelope_id=ulid_from(ts, i),
            session_id=sid,
            agent_id=AGENTS[ai],
            principal_id="p1",
            seq_no=seq_no,
            ts_unix_ms=ts,
            tool_id=tools[ti % len(tools)],
            args=args,
        )
        e = normalize(raw, tracker)
        tracker.observe(e)
        out.append(e)
    return out


outcomes = st.sampled_from(list(Outcome))
