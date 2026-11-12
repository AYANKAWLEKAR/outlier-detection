"""Published gate numbers trace to the Amendment A1 evidence (plan Task 1.26; spec A1.5, D-028).

Every gate number quoted in the README, the writeup and decision 008 is a cell of
``docs/decisions/evidence/a1-gate/matrix.json``, formatted ``0.938 (76/81)``; both protocol labels
stand verbatim in each of them; no paragraph claims the Tier-1 gate is met without both labels or
a negation next to the claim; the README's report block is the oracle-reviewer run with its
one-protocol PASS line, never a bare "gate PASS"; and the clarifications table carries D-021 to
D-028.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import orjson
import pytest

from agentfw.bench.gate import ORIGINAL_CRITERION_LABEL
from agentfw.bench.replay import PROTOCOL_LABELS
from agentfw.bench.report import README_END, README_START

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "docs/decisions/evidence/a1-gate/matrix.json"
DOCS = {
    "README.md": ROOT / "README.md",
    "writeup": ROOT / "docs/writeup.md",
    "008": ROOT / "docs/decisions/008-tier1-gate.md",
}
OLD_SW1_SHA = "e7b3bc2a2f20"


def _matrix() -> dict[str, Any]:
    m: dict[str, Any] = orjson.loads(MATRIX.read_bytes())
    return m


def a1_rows(*, dataset: str, seed: int, engine: str) -> list[dict[str, Any]]:
    """The ``matrix.json`` rows of one (dataset, seed, engine): one per protocol."""
    rows = [
        r
        for r in _matrix()["rows"]
        if (r["dataset"], r["seed"], r["engine"]) == (dataset, seed, engine)
    ]
    assert {r["protocol"] for r in rows} == {"review", "allow"}, (dataset, seed, engine)
    return sorted(rows, key=lambda r: str(r["protocol"]))


def sha12(dataset: str, seed: int) -> str:
    """The 12-character traces sha of a matrix dataset (the same for every row of it)."""
    shas = {
        r["dataset_sha"] for r in _matrix()["rows"] if (r["dataset"], r["seed"]) == (dataset, seed)
    }
    assert len(shas) == 1, (dataset, seed, shas)
    return str(shas.pop())


def cell(c: dict[str, Any]) -> str:
    return f"{c['recall']:.3f} ({c['n_detected']}/{c['n_total']})"


@pytest.mark.parametrize("name", DOCS)
def test_doc_quotes_the_generated_gate_cells(name: str) -> None:
    text = DOCS[name].read_text()
    assert PROTOCOL_LABELS["allow"] in text and PROTOCOL_LABELS["review"] in text
    assert ORIGINAL_CRITERION_LABEL in text
    for row in a1_rows(dataset="sw1", seed=0, engine="a1"):  # both protocols
        for fam, c in row["families"].items():
            assert cell(c) in text, (name, row["protocol"], fam)
    assert sha12("sw1", 0) in text


GATE_CLAIM = re.compile(
    r"(?i)(tier[- ]?1|amended|exit)\b[^.]{0,80}\b(gate|criterion)\b[^.]{0,40}"
    r"\b(met|passe[sd]|pass)\b"
    r"|\bgate\W{0,4}pass\b|\b(passes|meets)\b[^.]{0,40}\bgate\b"
)
NEGATIONS = ("not ", "never ", "neither", "whether")


def unqualified_gate_claims(text: str) -> list[str]:
    """Paragraphs (split on blank lines, whitespace collapsed) with a gate claim that is not
    qualified: the paragraph must contain both protocol labels verbatim, or one of NEGATIONS must
    stand within 12 characters before the match or within 20 characters after it in the same
    sentence, or between the match's "gate" / "criterion" and its claim word ("The Tier-1 gate is
    not met"), or the match must be a quoted mention (a double quote directly before and after
    it)."""
    bad: list[str] = []
    for para in (" ".join(p.split()) for p in re.split(r"\n\s*\n", text)):
        both = PROTOCOL_LABELS["review"] in para and PROTOCOL_LABELS["allow"] in para
        for m in GATE_CLAIM.finditer(para):
            before = para[max(0, m.start() - 12) : m.start()].lower()
            after = para[m.end() : m.end() + 20].split(".")[0].lower()
            between = para[m.end(2) : m.end()].lower() if m.group(2) else ""
            quoted = (
                para[max(0, m.start() - 1) : m.start()] == '"'
                and para[m.end() : m.end() + 1] == '"'
            )
            if not (
                both or quoted or any(n in before or n in after or n in between for n in NEGATIONS)
            ):
                bad.append(para)
    return bad


def test_docs_never_claim_the_gate_is_met_unqualified() -> None:
    for p in (*DOCS.values(), ROOT / "docs/leaderboard.md"):
        assert unqualified_gate_claims(p.read_text()) == [], p


@pytest.mark.parametrize(
    "sentence",
    [
        "Tier 1 gate is met.",
        "The Tier-1 gate **met**",
        "The Tier-1 gate has been met.",
        "The Tier-1 gate is fully met.",
        "now passes the Tier-1 gate",
        "meets the Tier-1 exit gate",
        "Tier-1 gate: PASS",
        "gate PASS",
        "The amended gate passes.",
        "Tier-1 exit criterion met.",
        "The Tier-1 gate is met; allow-all is a diagnostic only.",
        "The Tier-2 exit is not reached and the Tier-1 gate passes.",
    ],
)
def test_gate_claim_checker_flags_violations(sentence: str) -> None:
    assert unqualified_gate_claims(sentence)


@pytest.mark.parametrize(
    "sentence",
    [
        "The Tier-1 gate is met under neither protocol.",
        "Whether the Tier-1 gate is met is the user's open decision.",
        'No document may say "Tier-1 gate met" on the review result alone.',
        "The Tier-1 gate is not met under the spec protocol.",
        "**The Tier-1 gate is still not met.** Under the default protocol, two families fail.",
    ],
)
def test_gate_claim_checker_accepts_legitimate_sentences(sentence: str) -> None:
    assert unqualified_gate_claims(sentence) == []


def readme_report_block() -> str:
    """The text between the README's report markers."""
    text = DOCS["README.md"].read_text()
    s, e = text.index(README_START), text.index(README_END)
    assert s < e
    return text[s + len(README_START) : e]


def test_readme_report_block_is_the_review_run() -> None:
    block = readme_report_block()
    assert f"Protocol: {PROTOCOL_LABELS['review']}." in block
    assert f"dataset `{sha12('sw1', 0)}`" in block
    review = a1_rows(dataset="sw1", seed=0, engine="a1")[1]
    assert review["protocol"] == "review"
    for fam, c in review["families"].items():
        assert cell(c) in block, fam


def test_readme_report_block_never_prints_bare_gate_pass() -> None:
    block = readme_report_block()
    assert "one protocol only" in block and "gate PASS" not in block


def test_readme_dual_protocol_table_sits_directly_above_the_report_block() -> None:
    """The seed-0 dual-protocol table of ``matrix.md``, with the pre-A1 and sensitivity rows, is the
    last table before the report block (spec A1.5 rules 1 and 4)."""
    text = DOCS["README.md"].read_text()
    above = text[: text.index(README_START)]
    tail = above[above.rindex("\n## ") :]  # the section that holds the report block
    head = above[: above.rindex("\n## ")]
    table = head[head.rindex("\n## ") :]
    rows = [
        *a1_rows(dataset="sw1", seed=0, engine="a1"),
        *a1_rows(dataset="sw1", seed=0, engine="pre-a1"),
        *a1_rows(dataset="sens", seed=0, engine="a1"),
        *a1_rows(dataset="sens", seed=0, engine="pre-a1"),
    ]
    for r in rows:
        c = cell(r["families"]["magnitude_spike"])
        assert c in table, (r["dataset"], r["engine"], r["protocol"])
    assert "|" not in tail  # no other table between it and the report block


def test_readme_and_writeup_drop_the_one_protocol_gate_wording() -> None:
    for name in ("README.md", "writeup"):
        text = DOCS[name].read_text()
        assert "gate FAIL" not in text, name


def test_008_appended_section_supersedes_the_one_protocol_lines() -> None:
    text = DOCS["008"].read_text()
    head, sep, a1 = text.partition("\n## Amendment A1 (2026-11-11)\n")
    assert sep, "decision 008 lacks its appended Amendment A1 section"
    assert PROTOCOL_LABELS["allow"] in a1 and PROTOCOL_LABELS["review"] in a1
    assert "supersede" in a1
    for row in (
        *a1_rows(dataset="sw1", seed=1, engine="a1"),
        *a1_rows(dataset="sw1", seed=2, engine="a1"),
    ):
        for fam, c in row["families"].items():
            assert cell(c) in a1, (row["seed"], row["protocol"], fam)
    assert OLD_SW1_SHA in a1 and sha12("sw1", 0) in a1


@pytest.mark.parametrize("name", DOCS)
def test_docs_hold_no_absolute_paths(name: str) -> None:
    assert not re.search(r"(?<![\w.])/(Users|home|private|tmp)/", DOCS[name].read_text()), name


def test_clarifications_table_has_d021_to_d028() -> None:
    text = (ROOT / "docs/decisions/000-plan-clarifications.md").read_text()
    ids = re.findall(r"^\| (D-\d{3}) \|", text, flags=re.M)
    assert ids == [f"D-{i:03d}" for i in range(1, 29)]
    assert "smallest α" in text  # D-021: the per-agent operating point
    assert 'no document may say "Tier-1 gate met" on the review result alone' in text  # D-028
