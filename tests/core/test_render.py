"""Reason rendering (F8, Task 3.4; T-48): one readable line per reason, never a raw id."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from agentfw.bench.ci_regress import CI_DATASET
from agentfw.bench.dataset import load_dir, load_schema_hints
from agentfw.core.config import EngineConfig
from agentfw.core.engine import KNOWN_DETECTORS, Engine
from agentfw.core.envelope import (
    BaselineSummary,
    Decision,
    JsonValue,
    Outcome,
    Reason,
    TrajectoryInfo,
)
from agentfw.core.normalize import SessionTracker, normalize
from agentfw.core.patterns import is_id_like
from agentfw.core.render import render_reason
from agentfw.core.trajectory.decide import HALT_REASON
from agentfw.core.trajectory.gate import GateOutcome
from agentfw.synth.generate import generate

GATE_BLOCK = GateOutcome("block", "stripe.retrieve_charge", "charge_id")


def reason(detector: str, field: str | None = None, **kw: object) -> Reason:
    base: dict[str, object] = {
        "observed": None,
        "baseline_summary": None,
        "percentile": None,
        "blast_radius": None,
        "prereq_missing": None,
    }
    base.update(kw)
    return Reason(detector=detector, field=field, **base)  # type: ignore[arg-type]


def test_render_reason_examples() -> None:
    """Three golden strings (magnitude, prerequisite, novelty); none holds a raw id."""
    magnitude = reason(
        "magnitude",
        "amount",
        observed=48_000.0,
        baseline_summary=BaselineSummary(120.0, 80.0, 1140),
        percentile=0.9997,
        blast_radius=0.8,
        maturity="mature",
        regime="history+schemas",
    )
    info = TrajectoryInfo("history+schemas", 0.8, True, 0.36, maturity="mature", maturity_n=1140)
    assert (
        render_reason(magnitude, info, decision=Decision.BLOCK)
        == "blocked (mature: 1,140 observations); amount at 99.97th percentile "
        "(history+schemas, irreversible)"
    )

    prereq = reason(
        "sequence",
        prereq_missing="stripe.retrieve_charge",
        maturity="cold",
        regime="schemas_only",
    )
    info = TrajectoryInfo(
        "schemas_only", None, True, 0.6, maturity="cold", gate=GATE_BLOCK, ceiling_exempt=True
    )
    assert (
        render_reason(prereq, info, decision=Decision.BLOCK)
        == "blocked (cold, prerequisite exempt from the ceiling); no prior stripe.retrieve_charge; "
        "charge_id has no producer (usually stripe.retrieve_charge) (schemas_only, irreversible)"
    )

    novelty = reason("novelty", "to[]", prior_used=True, maturity="warm", regime="history_only")
    info = TrajectoryInfo(
        "history_only", None, False, 0.6, maturity="warm", capped_from=Decision.BLOCK
    )
    assert (
        render_reason(novelty, info, decision=Decision.REQUIRE_APPROVAL)
        == "held for approval (warm, pooled prior, capped from block); unseen to[] value "
        "(history_only)"
    )


def test_render_notices_and_other_detectors() -> None:
    cold = TrajectoryInfo(
        "history_only", None, False, 0.6, maturity="cold", capped_from=Decision.BLOCK, notice="warn"
    )
    vel = reason("velocity", None, observed=14, maturity="cold", regime="history_only")
    assert (
        render_reason(vel, cold, decision=Decision.ALLOW)
        == "allowed with a warning (cold, capped from block); call burst: 14 calls in the short "
        "window (history_only)"
    )
    bare = TrajectoryInfo(
        "bare", None, False, 0.6, maturity="bare", capped_from=Decision.BLOCK, notice="inform"
    )
    loop = reason("vital", "__loop__", observed=3, maturity="bare", regime="bare")
    assert (
        render_reason(loop, bare, decision=Decision.ALLOW)
        == "allowed, audit only (bare, capped from block); repeated identical calls (bare)"
    )
    info = TrajectoryInfo("history_only", None, False, 0.6, maturity="mature")
    assert render_reason(HALT_REASON, info, decision=Decision.BLOCK) == (
        "blocked (mature); session halted (history_only)"
    )
    # without a decision: the authority only
    trans = reason("sequence", maturity="mature", regime="history_only")
    assert render_reason(trans, info) == "mature; unusual tool sequence (history_only)"


def test_count_is_the_one_that_graded_the_call() -> None:
    """The head's count is ``TrajectoryInfo.maturity_n`` (the contributor that set the
    maturity), never this reason's own baseline: a mature magnitude reason on a call graded warm
    by another contributor must not read "warm: 1,140"."""
    r = reason("magnitude", "amount", baseline_summary=BaselineSummary(1.0, 1.0, 1140))
    info = TrajectoryInfo("history_only", None, False, 0.6, maturity="warm", maturity_n=41)
    assert render_reason(r, info).startswith("warm: 41 observations; ")
    ungraded = TrajectoryInfo("history_only", None, False, 0.6, maturity="mature")
    assert render_reason(r, ungraded).startswith("mature; ")


def test_exemption_shown_only_when_a_cap_was_in_play() -> None:
    """F4's block reads as exempt only when the ceiling would have capped it (``authority`` on,
    maturity below mature): ``TrajectoryInfo.ceiling_exempt``."""
    r = reason("sequence", prereq_missing="stripe.retrieve_charge")
    for exempt in (False, True):
        info = TrajectoryInfo(
            "schemas_only",
            None,
            True,
            0.6,
            maturity="mature",
            gate=GATE_BLOCK,
            ceiling_exempt=exempt,
        )
        out = render_reason(r, info, decision=Decision.BLOCK)
        assert ("exempt from the ceiling" in out) is exempt


def test_overlay_block_over_a_cap_does_not_read_capped() -> None:
    """T-62 (final review m5): ``session_overlay`` keeps ``capped_from`` for the audit and blocks
    the call; the line must not say "blocked (..., capped from block)"."""
    info = TrajectoryInfo(
        "history_only", None, False, 0.6, maturity="cold", capped_from=Decision.BLOCK
    )
    out = render_reason(HALT_REASON, info, decision=Decision.BLOCK)
    assert (
        out == "blocked (cold, session halt over a cap from block); session halted (history_only)"
    )
    assert "capped from" not in out
    held = render_reason(HALT_REASON, info, decision=Decision.ALLOW)
    assert "capped from block" in held


@pytest.mark.parametrize(
    ("p", "text"), [(0.5, "50th"), (0.01, "1st"), (0.22, "22nd"), (0.03, "3rd"), (0.113, "11.3th")]
)
def test_percentile_ordinals(p: float, text: str) -> None:
    r = reason("magnitude", "amount", percentile=p)
    assert f"amount at {text} percentile" in render_reason(
        r, TrajectoryInfo("bare", None, False, 0.6)
    )


def test_string_observations_are_never_rendered() -> None:
    """``observed`` is shown only when it is a number: a string could be a raw value."""
    r = reason("magnitude", "charge_id", observed="ch_000123abcd", maturity="warm")
    out = render_reason(r, TrajectoryInfo("history_only", None, False, 0.6, maturity="warm"))
    assert "ch_000123abcd" not in out and out == "warm; unusual charge_id (history_only)"


def _ids(v: JsonValue) -> Iterator[str]:
    if isinstance(v, str):
        if is_id_like(v):
            yield v
    elif isinstance(v, list):
        for x in v:
            yield from _ids(x)
    elif isinstance(v, dict):
        for k, x in v.items():
            if is_id_like(k):  # a key becomes a path segment of the flat field name
                yield k
            yield from _ids(x)


def test_rendered_reasons_on_the_ci_dataset_hold_no_raw_id(tmp_path: Path) -> None:
    """Every reason of a CI replay (gate on with the dataset's schemas: the gate runs on every
    irreversible call) against every id-like value and dict key the replay saw."""
    generate(CI_DATASET, tmp_path)
    raws, _, _ = load_dir(tmp_path)
    raws = sorted(raws, key=lambda r: (r.ts_unix_ms, r.session_id, r.seq_no))  # replay order
    cfg = EngineConfig(br_mode="score", sessions=True, authority=True, flow_ledger=True, gate=True)
    eng = Engine(cfg, enabled=KNOWN_DETECTORS, hints=load_schema_hints(tmp_path))
    tracker = SessionTracker()
    gated = 0
    ids: set[str] = set()
    rendered: list[str] = []
    for raw in raws:
        ids.update(_ids(raw.args))
        if raw.response_meta is not None:
            ids.update(raw.response_meta.ids)
        ids.update(raw.task_ids)
        e = normalize(raw, tracker)
        res = eng.score(e)
        assert res.trajectory is not None and res.decision is not None
        gated += res.trajectory.gate is not None
        rendered += [render_reason(r, res.trajectory, decision=res.decision) for r in res.reasons]
        eng.update(e, Outcome.ALLOWED)
        tracker.observe(e)
    assert len(rendered) > 100 and len(ids) > 100 and gated > 100
    leaks = [(s, i) for s in rendered for i in ids if i in s]
    assert not leaks, leaks[:5]


def test_integer_observed_beyond_float_range_renders() -> None:
    """An accepted integer too large for a float renders as an integer and does not break the
    line (T-71)."""
    info = TrajectoryInfo("history_only", None, False, 0.6, maturity="mature")
    nov = reason("novelty", "amount", observed=10**400, maturity="mature", regime="history_only")
    assert render_reason(nov, info) == "mature; unseen amount value (history_only)"
    vel = reason("velocity", None, observed=10**30, maturity="mature", regime="history_only")
    assert render_reason(vel, info) == (
        f"mature; call burst: {10**30} calls in the short window (history_only)"
    )
