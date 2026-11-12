"""Session-level labels and metrics (feature spec F2, F10).

Per-call metrics (``bench.metrics``) never see the runaway families (T-10); they are scored here,
per session. A session is ``"attacked"`` when any of its calls carries a non-runaway attack label,
else ``"runaway"`` when any carries a runaway one (``RUNAWAY_FAMILIES``), else ``"benign"``
(a ``benign_unusual`` control is benign). The vital signs are the session-level layer of
arXiv:2610.12375 (paraphrased): a benign session "false-flags" when any of its calls scores a full
``__loop__``, ``__spin__`` or ``__stall__`` (1.0) in ``per_detector_json``.

Decision metrics (Task 2.2) read the ``decision`` column instead of a threshold on the fused
score, so they see what the decision pipeline (``core.trajectory.decide``: provisional flags,
later the gate, escalation and the authority ceiling) did with a call: a row is flagged when its
decision is ``block`` or ``require_approval``. Rows without a decision (unscored) and runaway rows
(T-10) are in no denominator. Recall is per attack (an ``attack_id`` is detected when any of its
rows is flagged; its family is that of its first row by ``(ts_unix_ms, envelope_id)``), as
``metrics.recall_by_family(level="attack")``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Literal

import orjson

from agentfw.bench.metrics import is_runaway
from agentfw.bench.replay import ScoreRow
from agentfw.core.detectors.vital import LOOP, NAME, SPIN, STALL
from agentfw.synth.attacks import RUNAWAY_FAMILIES

__all__ = [
    "SessionLabel",
    "decision_benign_fpr",
    "decision_control_fpr",
    "decision_flagged",
    "decision_recall_by_family",
    "session_labels",
    "vital_false_flag_rate",
]

SessionLabel = Literal["benign", "runaway", "attacked"]
_RANK: dict[SessionLabel, int] = {"benign": 0, "runaway": 1, "attacked": 2}
_VITAL_FIELDS = frozenset({LOOP, SPIN, STALL})
_FLAGGED = frozenset({"block", "require_approval"})


def _row_label(r: ScoreRow) -> SessionLabel:
    if r.label != "attack":
        return "benign"
    return "runaway" if r.attack_family in RUNAWAY_FAMILIES else "attacked"


def session_labels(rows: Sequence[ScoreRow]) -> dict[str, SessionLabel]:
    """``session_id -> label``; attacked beats runaway beats benign."""
    out: dict[str, SessionLabel] = {}
    for r in rows:
        lab = _row_label(r)
        cur = out.get(r.session_id)
        if cur is None or _RANK[lab] > _RANK[cur]:
            out[r.session_id] = lab
    return out


def _vital_flag(r: ScoreRow) -> bool:
    return any(
        it.get("detector") == NAME and it.get("field") in _VITAL_FIELDS and it.get("score") == 1.0
        for it in orjson.loads(r.per_detector_json)
    )


def vital_false_flag_rate(rows: Sequence[ScoreRow]) -> float:
    """Share of benign sessions with any call whose loop, spin or stall score is 1 (NaN when
    there is no benign session)."""
    benign = {s for s, lab in session_labels(rows).items() if lab == "benign"}
    if not benign:
        return math.nan
    flagged = {r.session_id for r in rows if r.session_id in benign and _vital_flag(r)}
    return len(flagged) / len(benign)


# -- decision metrics ----------------------------------------------------------------------------


def decision_flagged(r: ScoreRow) -> bool:
    """The row's decision is ``block`` or ``require_approval``."""
    return r.decision in _FLAGGED


def _decided(rows: Sequence[ScoreRow]) -> list[ScoreRow]:
    return [r for r in rows if r.decision is not None and not is_runaway(r.attack_family)]


def _flag_rate(rows: Sequence[ScoreRow]) -> float:
    return sum(map(decision_flagged, rows)) / len(rows) if rows else math.nan


def decision_recall_by_family(rows: Sequence[ScoreRow]) -> dict[str, float]:
    """``family -> share of its attacks with a flagged row`` (attack level, module docstring)."""
    attacks: dict[str, list[ScoreRow]] = defaultdict(list)
    for r in _decided(rows):
        if r.label == "attack":
            attacks[r.attack_id or r.envelope_id].append(r)
    hits: dict[str, list[bool]] = defaultdict(list)
    for g in attacks.values():
        first = min(g, key=lambda r: (r.ts_unix_ms, r.envelope_id))
        hits[first.attack_family or "unknown"].append(any(map(decision_flagged, g)))
    return {fam: sum(h) / len(h) for fam, h in hits.items()}


def decision_control_fpr(rows: Sequence[ScoreRow]) -> float:
    """Flagged share of the benign ``benign_unusual`` control rows (NaN when there is none)."""
    return _flag_rate(
        [r for r in _decided(rows) if r.label != "attack" and r.control_type == "benign_unusual"]
    )


def decision_benign_fpr(rows: Sequence[ScoreRow]) -> float:
    """Flagged share of all benign rows, controls included (NaN when there is none)."""
    return _flag_rate([r for r in _decided(rows) if r.label != "attack"])
