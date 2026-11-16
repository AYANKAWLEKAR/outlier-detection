"""Output-preserving latency work (Task 4.1, ruling R25): the vital detector's ``arg_tokens`` are
computed once in ``normalize`` (``ActionEnvelope.arg_tokens``) instead of on every ``score``, and
the blast radius's name tokenizer is memoized. Neither may move a score, a decision or any other
replay column; the goldens and ``ci_regress`` cover the default engine, this file covers every
trajectory switch on."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from agentfw.bench import replay as replay_mod
from agentfw.bench.ci_regress import CI_DATASET
from agentfw.bench.dataset import load_dir
from agentfw.bench.registry import make_bench_detector
from agentfw.bench.replay import OutcomeMode, ScoreRow, SplitSpec, replay
from agentfw.core import blast
from agentfw.core.engine import Engine
from agentfw.core.envelope import ActionEnvelope, Outcome, RawCall
from agentfw.core.fingerprint import arg_tokens
from agentfw.core.normalize import SessionTracker, normalize
from agentfw.core.trajectory.session import SessionVerdict
from agentfw.synth.attacks import RUNAWAY_FAMILIES
from agentfw.synth.generate import generate
from tests.factories import make_raw


def test_normalize_precomputes_arg_tokens() -> None:
    tr = SessionTracker()
    e = normalize(make_raw(args={"amount": 37.5, "ids": ["ch_000000000001", "x"]}), tr)
    assert e.arg_tokens == arg_tokens(e.flat_args)


def test_observe_records_the_same_tokens_with_or_without_precompute() -> None:
    raw = make_raw(args={"q": "status open", "limit": 3})
    a, b = SessionTracker(), SessionTracker()
    e = normalize(raw, a)
    a.observe(e)
    b.observe(dataclasses.replace(e, arg_tokens=None))
    probe = make_raw(seq_no=1, ts_unix_ms=raw.ts_unix_ms + 1, args={"q": "x"})
    assert normalize(probe, a).recent == normalize(probe, b).recent


def test_blast_tokens_memo_matches_the_plain_function() -> None:
    for name in ("create_refund", "massUpdate", "records[]", "Bulk-Send_alls", "", "a.b.c"):
        assert blast._tokens(name) == blast._tokens.__wrapped__(name)


@pytest.fixture(scope="module")
def runaway_ci(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The CI dataset plus the runaway and splice families (vital, sessions and halts fire)."""
    d = tmp_path_factory.mktemp("ci_traj")
    families = [*CI_DATASET.families, *sorted(RUNAWAY_FAMILIES), "splice"]
    generate(CI_DATASET.model_copy(update={"families": families, "attack_rate": 0.5}), d)
    return d


def _rows(d: Path, outcomes: OutcomeMode) -> list[ScoreRow]:
    raws, labels, _ = load_dir(d)
    det = make_bench_detector("engine-traj", d)
    return replay(raws, labels, det, SplitSpec(calibration_frac=0.15), outcomes=outcomes)


def _strip(rows: list[ScoreRow]) -> list[dict[str, object]]:
    out = []
    for r in rows:
        d = dataclasses.asdict(r)
        d.pop("latency_us")
        out.append(d)
    return out


@pytest.mark.parametrize("outcomes", ["allow", "review"])
def test_every_replay_column_is_unchanged_with_every_switch_on(
    runaway_ci: Path, outcomes: OutcomeMode, monkeypatch: pytest.MonkeyPatch
) -> None:
    fast = _rows(runaway_ci, outcomes)
    with monkeypatch.context() as m:  # the code path before the optimisation

        def plain_normalize(raw: RawCall, tracker: SessionTracker) -> ActionEnvelope:
            return dataclasses.replace(normalize(raw, tracker), arg_tokens=None)

        m.setattr(replay_mod, "normalize", plain_normalize)
        m.setattr(blast, "_tokens", blast._tokens.__wrapped__)
        slow = _rows(runaway_ci, outcomes)
    assert any(r.flags for r in fast) and any(r.density for r in fast)
    assert any('"__loop__"' in r.per_detector_json for r in fast)
    assert _strip(fast) == _strip(slow)


def test_session_verdict_equals_the_book_verdict_with_every_switch_on(runaway_ci: Path) -> None:
    """The engine computes the density once and the verdict with ``ladder`` (Task 4.1); it must
    equal ``SessionBook.verdict_with`` on the record the call was scored against, call by call."""
    raws, _, _ = load_dir(runaway_ci)
    eng = make_bench_detector("engine-traj", runaway_ci)
    assert isinstance(eng, Engine) and eng._book is not None
    book = eng._book
    tracker = SessionTracker()
    verdicts: set[SessionVerdict] = set()
    for raw in sorted(raws, key=lambda r: (r.ts_unix_ms, r.session_id, r.seq_no)):
        e = normalize(raw, tracker)
        t = eng.score(e).trajectory
        assert t is not None and t.session_verdict is not None
        assert t.session_verdict == book.verdict_with(book.peek(e.session_id), t.flags)
        assert t.density == book.density_with(book.peek(e.session_id), t.flags)
        verdicts.add(t.session_verdict)
        eng.update(e, Outcome.ALLOWED)
        tracker.observe(e)
    assert verdicts == set(SessionVerdict)  # ok, warn and halt all occur
