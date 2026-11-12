"""Amendment A1's gate evidence (plan Task 1.23): the matrix on seeds 0-2 under both protocols.

``docs/decisions/evidence/a1-gate/run.sh`` generates the four datasets, replays each once with
``agentfw.bench.gate`` and writes ``matrix.json`` / ``matrix.md``. This test pins that the matrix
holds every (dataset, seed, engine, protocol) row, the magnitude census of each benchmark seed and
both protocol labels verbatim (spec A1.5 rule 1) beside the original spec §9 criterion.
"""

from __future__ import annotations

from pathlib import Path

import orjson

from agentfw.bench.replay import PROTOCOL_LABELS

EVIDENCE = Path(__file__).resolve().parents[2] / "docs/decisions/evidence/a1-gate"


def test_a1_matrix_is_complete() -> None:
    m = orjson.loads((EVIDENCE / "matrix.json").read_bytes())
    have = {(r["dataset"], r["seed"], r["engine"], r["protocol"]) for r in m["rows"]}
    want = {
        ("sw1", s, e, p) for s in (0, 1, 2) for e in ("a1", "pre-a1") for p in ("review", "allow")
    }
    want |= {("sens", 0, e, p) for e in ("a1", "pre-a1") for p in ("review", "allow")}
    assert have == want
    for s in ("0", "1", "2"):
        assert m["census"][s]["gated_test_n"] >= 50 and m["counts"][s]["magnitude_dropped"] >= 0
    md = (EVIDENCE / "matrix.md").read_text()
    assert (
        PROTOCOL_LABELS["allow"] in md
        and PROTOCOL_LABELS["review"] in md
        and "original spec §9 criterion" in md
    )
