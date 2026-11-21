"""Blast radius (spec §6.9, T-17), MCP annotation terms (F3), irreversibility and regime (T-06)."""

from __future__ import annotations

import math
from importlib import resources
from typing import Any

import pytest
import scipy.stats
import yaml

from agentfw.bench.replay import SplitSpec, replay
from agentfw.core.blast import (
    IRREVERSIBLE_VERBS,
    SCOPE_NOUNS,
    blast_radius,
    irreversible,
    lexical,
    regime_of,
    structural,
    tool_blast_radius,
)
from agentfw.core.config import EngineConfig
from agentfw.core.detectors.sequence import SequenceDetector
from agentfw.core.engine import KNOWN_DETECTORS, Engine
from agentfw.core.envelope import Decision, ResponseMeta, TrajectoryInfo
from agentfw.core.hints import FieldHint, ToolAnnotations, ToolHint
from agentfw.core.normalize import SessionTracker
from agentfw.core.state import DictStore
from agentfw.synth.load import load_catalog
from agentfw.synth.schema import ToolSpec
from tests.factories import make_env, make_raw, run_session

LABELS = resources.files("agentfw.bench").joinpath("data/irreversibility_labels.yaml")
REFUND = "stripe.create_refund"
REFUND_HINT = ToolHint(kind="write", field_hints={"amount": "money"})


def ann(spec: ToolSpec) -> ToolAnnotations | None:
    return None if spec.annotations is None else spec.annotations.to_core()


def hints(spec: ToolSpec) -> dict[str, FieldHint]:
    return {name: a.shape for name, a in spec.args.items() if a.shape is not None}


# --- F3 acceptance -----------------------------------------------------------------------------


def test_labels_cover_the_catalog_on_the_0_to_3_scale() -> None:
    labels = yaml.safe_load(LABELS.read_text())
    assert set(labels) == set(load_catalog())
    assert set(labels.values()) <= {0, 1, 2, 3}
    catalog = load_catalog()
    assert all((labels[t] == 0) == (spec.kind == "read") for t, spec in catalog.items())


def test_tool_ranking_matches_hand_labels() -> None:  # F3 acceptance
    labels = yaml.safe_load(LABELS.read_text())  # tool_id -> 0..3, every catalog tool
    catalog = load_catalog()
    b = {t: tool_blast_radius(t, spec.kind, ann(spec), hints(spec)) for t, spec in catalog.items()}
    rho = scipy.stats.spearmanr([labels[t] for t in b], [b[t] for t in b]).statistic
    assert rho >= 0.8


def test_mislabelled_read_only_gets_no_discount() -> None:  # F3 adversarial acceptance
    delete_env = make_env(tool_id="crm.delete_records", args={"ids": ["rec_00000001"]})
    honest = blast_radius(delete_env, ToolHint(kind="write"), is_write=True, has_item_count=False)
    lying = blast_radius(
        delete_env,
        ToolHint(kind="write", annotations=ToolAnnotations(read_only=True)),
        is_write=True,
        has_item_count=False,
    )
    assert lying >= honest
    # a lying read-only flag does not cancel the non-idempotent term either
    liar = ToolAnnotations(read_only=True, idempotent=False)
    plain = ToolAnnotations(idempotent=False)
    assert blast_radius(
        delete_env, ToolHint(kind="write", annotations=liar), is_write=True, has_item_count=False
    ) == blast_radius(
        delete_env, ToolHint(kind="write", annotations=plain), is_write=True, has_item_count=False
    )


def test_read_only_discount_needs_a_read_with_item_counts_and_no_id_list() -> None:
    fh: dict[str, FieldHint] = {"n": "count"}
    ro = ToolHint(field_hints=fh, annotations=ToolAnnotations(read_only=True))
    plain = ToolHint(field_hints=fh)
    e = make_env(tool_id="kb.bulk_send_preview", args={"q": "open", "n": 5})  # lexical 1.0
    bare = blast_radius(e, plain, is_write=False, has_item_count=True)
    assert bare == pytest.approx(0.5)
    assert blast_radius(e, ro, is_write=False, has_item_count=True) == pytest.approx(0.1)
    # no item count: the structural inference does not corroborate the hint
    assert blast_radius(e, ro, is_write=False, has_item_count=False) == blast_radius(
        e, plain, is_write=False, has_item_count=False
    )
    # an id-list argument does not corroborate it either
    lst = make_env(tool_id="kb.search", args={"ids": [f"doc_{i:08d}" for i in range(10)]})
    assert blast_radius(lst, ro, is_write=False, has_item_count=True) == blast_radius(
        lst, None, is_write=False, has_item_count=True
    )


def test_destructive_hint_raises_and_marks_irreversible() -> None:
    e = make_env(tool_id="crm.archive_record", args={"note": "x"})
    plain = ToolHint(kind="write")
    destr = ToolHint(kind="write", annotations=ToolAnnotations(destructive=True))
    b0 = blast_radius(e, plain, is_write=True, has_item_count=False)
    b1 = blast_radius(e, destr, is_write=True, has_item_count=False)
    assert b1 == pytest.approx(min(1.0, b0 + 0.4))
    assert irreversible(b0, plain, is_write=True) is False  # 0.35 < 0.6
    assert irreversible(b1, destr, is_write=True) is True
    # destructive marks a call irreversible whatever b and kind say
    assert irreversible(0.05, destr, is_write=False) is True
    assert irreversible(None, destr, is_write=True) is True


# --- T-17 numbers ------------------------------------------------------------------------------


def test_vocabularies_are_the_spec_lists() -> None:
    verbs = {"refund", "payout", "transfer", "delete", "drop", "void", "cancel", "send", "post"}
    assert frozenset(verbs | {"publish"}) == IRREVERSIBLE_VERBS
    assert frozenset({"bulk", "mass", "all", "batch"}) == SCOPE_NOUNS


@pytest.mark.parametrize(
    ("tool", "paths", "fh", "want"),
    [
        ("stripe.list_charges", (), {}, 0.0),
        ("stripe.create_refund", ("amount",), {}, 0.6),  # verb in the tool name
        ("stripe.create_refund", ("amount",), {"amount": "money"}, 0.8),  # + money hint present
        ("stripe.create_refund", ("reason",), {"amount": "money"}, 0.6),  # hint not in this call
        ("salesforce.mass_update", ("object",), {}, 0.2),  # scope noun in the tool name
        ("crm.update", ("apply_to_all",), {}, 0.2),  # scope noun in a field name
        ("crm.bulk_delete", ("n",), {"n": "count"}, 1.0),  # 0.6 + 0.2 + 0.2
        ("mail.sendEmail", (), {}, 0.6),  # camelCase tool names split too
        ("slack.posts", (), {}, 0.6),  # a plural verb token
        ("mail.postal_lookup", (), {}, 0.0),  # whole tokens only
        ("send.lookup", (), {}, 0.0),  # the server name is not the tool name
        ("x.delete_send_post", ("all", "batch"), {}, 0.8),  # each list counts once
    ],
)
def test_lexical(tool: str, paths: tuple[str, ...], fh: dict[str, FieldHint], want: float) -> None:
    assert lexical(tool, paths, fh) == pytest.approx(want)


@pytest.mark.parametrize(
    ("is_write", "n", "items", "want"),
    [
        (False, 0, False, 0.3),
        (False, 0, True, 0.0),  # a read with item counts
        (True, 0, False, 0.7),
        (True, 0, True, 0.7),  # item counts never discount a write
        (True, 1, False, 0.7),  # log10(1) = 0
        (True, 10, False, 0.95),
        (True, 10_000, False, 1.0),  # the id-list term is capped at 0.3
        (False, 100, True, 0.3 + 0.3 - 0.3),
    ],
)
def test_structural(is_write: bool, n: int, items: bool, want: float) -> None:
    assert structural(is_write, n, items) == pytest.approx(want)


def test_blast_radius_combines_half_and_half_then_annotations() -> None:
    e = make_env()  # stripe.create_refund {"amount": 37.5}
    b = blast_radius(e, REFUND_HINT, is_write=True, has_item_count=False)
    assert b == pytest.approx(0.5 * 0.8 + 0.5 * 0.7)
    full = ToolHint(
        kind="write",
        field_hints={"amount": "money"},
        annotations=ToolAnnotations(idempotent=False, open_world=True),
    )
    assert blast_radius(e, full, is_write=True, has_item_count=False) == pytest.approx(b + 0.2)
    # idempotent=False on a corroborated read-only tool adds nothing
    read = make_env(tool_id="kb.search", args={"q": "x"})
    ro = ToolHint(annotations=ToolAnnotations(read_only=True, idempotent=False))
    assert blast_radius(read, ro, is_write=False, has_item_count=True) == 0.05  # floor


def test_blast_radius_id_list_is_the_longest_list_of_id_like_values() -> None:
    ids = [f"rec_{i:08d}" for i in range(100)]
    e = make_env(tool_id="crm.touch", args={"ids": ids, "tags": ["a", "b", "c"] * 50})
    b = blast_radius(e, None, is_write=True, has_item_count=False)
    assert b == pytest.approx(0.5 * 0.0 + 0.5 * min(1.0, 0.7 + min(0.3, math.log10(100) / 4)))


def test_tool_blast_radius_is_the_static_form() -> None:
    a = ToolAnnotations(read_only=False, destructive=True, idempotent=False, open_world=True)
    assert tool_blast_radius(REFUND, "write", a, {"amount": "money"}) == 1.0
    assert tool_blast_radius("kb.search", "read", None, {}) == 0.05
    assert tool_blast_radius("kb.search", None, None, {}) == pytest.approx(0.15)


def test_regime_four_values() -> None:  # T-06 table
    assert regime_of(True, 30, 30) == "history+schemas"
    assert regime_of(False, 30, 30) == "history_only"
    assert regime_of(True, 29, 30) == "schemas_only"
    assert regime_of(False, 0, 30) == "bare"


def test_n_tool_counts_unigram_observations() -> None:
    store = DictStore()
    eng = Engine(store=store)
    seq = SequenceDetector(store, EngineConfig())
    assert seq.n_tool("a1", REFUND) == 0
    run_session(eng, [make_raw(seq_no=i, ts_unix_ms=1_700_000_000_000 + i) for i in range(3)])
    assert seq.n_tool("a1", REFUND) == 3
    assert seq.n_tool("a1", "other.tool") == 0
    assert seq.n_tool("a2", REFUND) == 0


# --- engine wiring -----------------------------------------------------------------------------


def _warm_stream(n: int = 80) -> list[Any]:
    """Reads with item counts, then refunds: the sequence detector learns both kinds."""
    out = []
    for i in range(n):
        out.append(
            make_raw(
                session_id=f"s{i}",
                seq_no=0,
                ts_unix_ms=1_700_000_000_000 + 10 * i,
                tool_id="stripe.list_charges",
                args={"customer_id": f"cus_{i % 7:08d}", "limit": 10},
                response_meta=ResponseMeta(item_count=3 + i % 4),
            )
        )
        out.append(
            make_raw(
                session_id=f"s{i}",
                seq_no=1,
                ts_unix_ms=1_700_000_000_000 + 10 * i + 1,
                tool_id=REFUND,
                args={"charge_id": f"ch_{i % 9:08d}", "amount": 20.0 + i % 13},
            )
        )
    return out


def test_trajectory_always_filled_and_off_by_default() -> None:
    assert EngineConfig().br_mode == "off"
    r = Engine().score(make_env())
    assert isinstance(r.trajectory, TrajectoryInfo)
    assert r.trajectory.blast_radius is None
    assert r.trajectory.irreversible is False
    assert r.trajectory.threshold_eff == r.threshold
    assert r.trajectory.regime == "bare"
    assert all(x.blast_radius is None for x in r.reasons)


def test_br_off_leaves_decisions_unchanged() -> None:  # Engine(), schemas loaded: same decisions
    stream = _warm_stream()
    schema_hints = {
        "stripe.list_charges": ToolHint(
            annotations=ToolAnnotations(read_only=True, idempotent=True), has_schema=True
        ),
        REFUND: ToolHint(
            annotations=ToolAnnotations(destructive=True, idempotent=False, open_world=True),
            has_schema=True,
        ),
    }
    plain = run_session(Engine(), stream)
    loaded = run_session(Engine(hints=schema_hints), stream)
    for a, b in zip(plain, loaded, strict=True):
        assert (a.fused_score, a.decision, a.threshold) == (b.fused_score, b.decision, b.threshold)
        assert a.per_detector == b.per_detector and a.reasons == b.reasons
        assert a.trajectory is not None and b.trajectory is not None
        assert b.trajectory.threshold_eff == b.threshold
    assert loaded[-1].trajectory is not None
    assert loaded[-1].trajectory.regime == "history+schemas"
    assert plain[-1].trajectory is not None
    assert plain[-1].trajectory.regime == "history_only"
    # off: no b, but a destructive annotation still marks the call irreversible
    assert loaded[-1].trajectory.blast_radius is None
    assert loaded[-1].trajectory.irreversible is True


def test_score_mode_reports_b_without_moving_the_decision() -> None:
    stream = _warm_stream()
    off = run_session(Engine(hints={REFUND: REFUND_HINT}), stream)
    sc = run_session(Engine(EngineConfig(br_mode="score"), hints={REFUND: REFUND_HINT}), stream)
    for a, b in zip(off, sc, strict=True):
        assert (a.fused_score, a.decision, a.threshold) == (b.fused_score, b.decision, b.threshold)
        assert b.trajectory is not None and b.trajectory.blast_radius is not None
        assert b.trajectory.threshold_eff == b.threshold
        assert all(x.blast_radius == b.trajectory.blast_radius for x in b.reasons)
    last = sc[-1].trajectory
    assert last is not None and last.blast_radius == pytest.approx(0.75)
    assert last.irreversible is True  # write with b >= 0.6
    first_read = sc[0].trajectory
    assert first_read is not None and first_read.irreversible is False


def test_modulate_lowers_threshold_for_writes() -> None:
    h = {REFUND: REFUND_HINT}
    refund_env = make_env()
    r = Engine(EngineConfig(br_mode="modulate"), hints=h).score(refund_env)
    assert r.threshold is not None
    assert r.trajectory is not None and r.trajectory.blast_radius is not None
    assert r.trajectory.threshold_eff == pytest.approx(
        r.threshold * (1 - 0.6 * r.trajectory.blast_radius)
    )
    assert r.threshold is not None and r.trajectory.threshold_eff < r.threshold


def test_modulate_decides_against_the_effective_threshold() -> None:
    destr = {
        REFUND: ToolHint(kind="write", annotations=ToolAnnotations(destructive=True)),
    }
    e = make_env()
    off = Engine(EngineConfig(br_k=1.0), hints=destr).score(e)
    mod = Engine(EngineConfig(br_mode="modulate", br_k=1.0), hints=destr).score(e)
    assert off.fused_score == mod.fused_score == 0.0
    assert off.decision is Decision.ALLOW
    assert mod.trajectory is not None and mod.trajectory.blast_radius == 1.0
    assert mod.trajectory.threshold_eff == 0.0
    assert mod.decision is Decision.BLOCK
    assert mod.threshold == off.threshold  # the agent's threshold itself is not modulated


def test_modulate_keeps_state_and_score_pure() -> None:
    stream = _warm_stream(40)
    cfg = EngineConfig(br_mode="modulate")
    det = Engine(cfg, enabled=KNOWN_DETECTORS, hints={REFUND: REFUND_HINT})
    rows = replay(stream, {}, det, SplitSpec(), strict=True)  # LeakError if score() writes
    assert len(rows) == len(stream)
    a, b = Engine(hints={REFUND: REFUND_HINT}), Engine(cfg, hints={REFUND: REFUND_HINT})
    run_session(a, stream, SessionTracker())
    run_session(b, stream, SessionTracker())
    # modulation moves decisions only (allow-all outcomes here), never state
    assert _state(a) == _state(b)


def _state(e: Engine) -> dict[str, object]:
    return {k: r.to_dict() for k, r in sorted(e.store.items())}


def test_unknown_br_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="br_mode"):
        Engine(EngineConfig(br_mode="sometimes"))  # type: ignore[arg-type]
