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

from agentfw.bench.gate import ORIGINAL_CRITERION_LABEL, MagnitudeCensus, render_census
from agentfw.bench.replay import PROTOCOL_LABELS
from agentfw.bench.report import README_END, README_START
from agentfw.synth.generate import sw1_config

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
    sentence, or the match must be a quoted mention (a double quote directly before and after
    it). The window is the Task 1.26 brief's, verbatim (ruling A1-26)."""
    bad: list[str] = []
    for para in (" ".join(p.split()) for p in re.split(r"\n\s*\n", text)):
        both = PROTOCOL_LABELS["review"] in para and PROTOCOL_LABELS["allow"] in para
        for m in GATE_CLAIM.finditer(para):
            before = para[max(0, m.start() - 12) : m.start()].lower()
            after = para[m.end() : m.end() + 20].split(".")[0].lower()
            quoted = (
                para[max(0, m.start() - 1) : m.start()] == '"'
                and para[m.end() : m.end() + 1] == '"'
            )
            if not (both or quoted or any(n in before or n in after for n in NEGATIONS)):
                bad.append(para)
    return bad


def test_docs_never_claim_the_gate_is_met_unqualified() -> None:
    """README, writeup and leaderboard whole; decision 008 in its A1.5 scope (``gate_scope``: the
    Status header and the appended Amendment A1 section). The pre-A1 sentences between them are
    the historical record that section supersedes (ruling A1-26)."""
    texts = {name: DOCS[name].read_text() for name in ("README.md", "writeup")}
    texts["008"] = gate_scope("008")
    texts["leaderboard"] = (ROOT / "docs/leaderboard.md").read_text()
    for name, text in texts.items():
        assert unqualified_gate_claims(text) == [], name


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


FAMILIES = (
    "magnitude_spike",
    "scope_blowup",
    "exfil_volume",
    "prereq_skip",
    "novel_target",
    "velocity_burst",
)
SENS_VERDICT = "companion row (spec A1.5 rule 4), not a gate result"
# Measured numbers: shares and scores (0.938, 0.0058) and percentages (3.52 %); not task numbers.
MEASURED = re.compile(r"\b0\.\d{2,}\b|\b\d+(?:\.\d+)? ?%")
# A protocol named by anything other than its verbatim label (checked after the labels are removed).
SHORT_PROTOCOL = re.compile(
    r"(?i)(?<![\w-])allow-all\b|oracle-reviewer|\breview(?:er)? (?:loop|run)\b"
)
EXPECTED_TABLE_ROWS = {"README.md": 8, "writeup": 16, "008": 16}


def gate_scope(name: str) -> str:
    """The part of a document that A1.5 binds: the README above its generated report block, the
    whole writeup, decision 008's header (Status) and its appended Amendment A1 section."""
    text = DOCS[name].read_text()
    if name == "README.md":
        return text[: text.index(README_START)]
    if name == "008":
        return (
            text[: text.index("\n## Context\n")]
            + text[text.index("\n## Amendment A1 (2026-11-11)\n") :]
        )
    return text


def prose_units(text: str) -> list[str]:
    """Paragraphs (split on blank lines) with table lines dropped and each list item a unit of its
    own, whitespace collapsed: a bullet list is not one block that its first bullet can qualify."""
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        lines = "\n".join(ln for ln in para.splitlines() if not ln.lstrip().startswith("|"))
        for item in re.split(r"\n(?=\s*(?:[-*]|\d+\.) )", lines):
            unit = " ".join(item.split())
            if unit:
                units.append(unit)
    return units


def _row(dataset: str, seed: int, engine: str, protocol: str) -> dict[str, Any]:
    r: dict[str, Any]
    [r] = [
        r
        for r in _matrix()["rows"]
        if (r["dataset"], r["seed"], r["engine"], r["protocol"])
        == (dataset, seed, engine, protocol)
    ]
    return r


def expected_table_cells(r: dict[str, Any]) -> list[str]:
    """A dual-protocol table row as ``matrix.md`` prints it, after the protocol column."""
    out = [
        f"{cell(r['families'][f])}; {r['families'][f]['among']:.2f}; {r['families'][f]['top1']:.2f}"
        for f in FAMILIES
    ]
    out += [f"{r['benign_fpr']:.4f}", f"{r['control_fpr']:.3f}"]
    if r["dataset"] == "sens":
        out.append(SENS_VERDICT)
    else:
        out.append("PASS" if r["passed"] else "FAIL: " + ", ".join(r["failures"]))
    return out


@pytest.mark.parametrize("name", DOCS)
def test_every_quoted_gate_cell_is_a_matrix_cell(name: str) -> None:
    cells = {cell(c) for r in _matrix()["rows"] for c in r["families"].values()}
    quoted = re.findall(r"\d\.\d{3} \(\d+/\d+\)", gate_scope(name))
    assert quoted
    assert [q for q in quoted if q not in cells] == [], name


@pytest.mark.parametrize("name", DOCS)
def test_dual_protocol_table_rows_match_the_matrix(name: str) -> None:
    """Every table line that carries a protocol label is the ``matrix.json`` row it names, cell by
    cell (recall n/N, among, top-1, benign FPR, control FPR, verdict)."""
    proto_of = {label: proto for proto, label in PROTOCOL_LABELS.items()}
    by_sha = {r["dataset_sha"]: (r["dataset"], r["seed"]) for r in _matrix()["rows"]}
    checked = 0
    for line in gate_scope(name).splitlines():
        if not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        at = [i for i, p in enumerate(parts) if p in proto_of]
        if not at or not re.match(r"\d\.\d{3} \(\d+/\d+\);", "".join(parts[at[0] + 1 : at[0] + 2])):
            continue  # not a gate row (e.g. a leaderboard row carries the label too)
        i = at[0]
        key = parts[:i]
        if len(key) == 1 and key[0].isdigit():  # seed tables: the amended engine on SW1
            dataset, seed, engine = "sw1", int(key[0]), "a1"
        else:
            assert len(key) == 2, (name, line)
            sha = re.search(r"`(\w{12})`", key[0])
            assert sha, (name, line)
            dataset, seed = by_sha[sha.group(1)]
            engine = {"pre-A1": "pre-a1", "amended (A1)": "a1"}[key[1]]
        r = _row(dataset, seed, engine, proto_of[parts[i]])
        assert parts[i + 1 :] == expected_table_cells(r), (name, dataset, seed, engine, parts[i])
        checked += 1
    assert checked == EXPECTED_TABLE_ROWS[name]


@pytest.mark.parametrize("name", DOCS)
def test_measured_numbers_name_the_protocol_by_its_verbatim_label(name: str) -> None:
    """Spec A1.5 rule 1: a paragraph or list item that quotes a measured number never names a
    protocol by a shortened name ("under allow-all", "the oracle-reviewer loop")."""
    bad = []
    for unit in prose_units(gate_scope(name)):
        rest = unit.replace(PROTOCOL_LABELS["allow"], "").replace(PROTOCOL_LABELS["review"], "")
        if MEASURED.search(rest) and SHORT_PROTOCOL.search(rest):
            bad.append(unit)
    assert bad == [], name


@pytest.mark.parametrize("name", DOCS)
def test_magnitude_cell_in_prose_stands_with_its_companions(name: str) -> None:
    """Spec A1.5 rule 4 in prose: a list item or paragraph that quotes an amended-engine SW1
    ``magnitude_spike`` cell also quotes both companions (the pre-A1 cell of the same seed and
    protocol, and the sensitivity cell of the same protocol) or points to the companion rows."""
    companions: dict[str, list[tuple[str, str]]] = {}
    for r in _matrix()["rows"]:
        if (r["dataset"], r["engine"]) == ("sw1", "a1"):
            pre = _row("sw1", r["seed"], "pre-a1", r["protocol"])
            sens = _row("sens", 0, "a1", r["protocol"])
            companions.setdefault(cell(r["families"]["magnitude_spike"]), []).append(
                (
                    cell(pre["families"]["magnitude_spike"]),
                    cell(sens["families"]["magnitude_spike"]),
                )
            )
    bad = []
    for unit in prose_units(gate_scope(name)):
        for c, pairs in companions.items():
            if c not in unit or "companion" in unit.lower():
                continue
            if not any(pre in unit and sens in unit for pre, sens in pairs):
                bad.append((c, unit))
    assert bad == [], name


@pytest.mark.parametrize("name", DOCS)
def test_docs_hold_no_absolute_paths(name: str) -> None:
    assert not re.search(r"(?<![\w.])/(Users|home|private|tmp)/", DOCS[name].read_text()), name


def test_clarifications_table_has_d021_to_d028() -> None:
    text = (ROOT / "docs/decisions/000-plan-clarifications.md").read_text()
    ids = re.findall(r"^\| (D-\d{3}) \|", text, flags=re.M)
    assert ids == [f"D-{i:03d}" for i in range(1, 29)]
    assert "smallest α" in text  # D-021: the per-agent operating point
    assert 'no document may say "Tier-1 gate met" on the review result alone' in text  # D-028


# --- Final-review fix list, Part B (rulings A1-28, A1-30 to A1-33): the numbers added to the docs
# are pinned to their sources (matrix.json, docs/leaderboard.md, the README report block, the
# spec), never to themselves.

LEADERBOARD = ROOT / "docs/leaderboard.md"
SPEC = ROOT / "docs/spec.md"
REVIEW, ALLOW = PROTOCOL_LABELS["review"], PROTOCOL_LABELS["allow"]


def flat(text: str) -> str:
    return " ".join(text.split())


def units_with(name: str, needle: str) -> list[str]:
    units = [u for u in prose_units(gate_scope(name)) if needle in u]
    assert units, (name, needle)
    return units


def pct(x: float, nd: int = 2) -> str:
    return f"{100 * x:.{nd}f} %"


def census_line(seed: int) -> str:
    """The census line of ``render_census`` for a seed's ``matrix.json`` census (spec A1.4)."""
    return render_census(MagnitudeCensus(**_matrix()["census"][str(seed)]))


def test_census_line_is_matrix_json_seed0_on_the_leaderboard() -> None:
    """The leaderboard's census (computed from the dataset) is matrix.json's seed-0 census."""
    assert flat(LEADERBOARD.read_text()).count(census_line(0)) == 2


@pytest.mark.parametrize(
    ("name", "start", "end", "seeds"),
    [
        ("README.md", "\n## Tier-1 gate under both protocols, SW1 seed 0\n", "\n| dataset |", (0,)),
        ("writeup", "\n## 6. Tier-1 status\n", "\n**Companion rows", (0, 1, 2)),
        (
            "008",
            "\n### Tier-1 gate under both protocols, SW1 seeds 0, 1 and 2\n",
            "\n### Companion rows",
            (0, 1, 2),
        ),
    ],
)
def test_census_line_stands_with_the_gate_table(
    name: str, start: str, end: str, seeds: tuple[int, ...]
) -> None:
    """Spec A1.4 / ruling A1-31: the README's dual-protocol table intro carries the seed-0 census
    line; the writeup's and decision 008's seed tables carry one per seed."""
    text = gate_scope(name)
    section = text[text.index(start) :]
    section = flat(section[: section.index(end)])
    for seed in seeds:
        assert census_line(seed) in section, (name, seed)


def test_008_names_the_highest_per_agent_velocity_share_under_each_protocol() -> None:
    """Ruling A1-32 docs-4: the maximum of matrix.md's per-agent table under each protocol."""
    best = {
        proto: max(
            (v, a, r["seed"])
            for r in _matrix()["rows"]
            if (r["dataset"], r["engine"], r["protocol"]) == ("sw1", "a1", proto)
            for a, v in r["per_agent_benign_velocity_share_ge_0_5"].items()
        )
        for proto in ("review", "allow")
    }
    (rv, agent, seed), (av, a2, s2) = best["review"], best["allow"]
    assert (agent, seed) == (a2, s2)
    want = (
        f"the highest under the {REVIEW} is {agent} at {pct(rv)} "
        f"(seed {seed}; {pct(av)} under {ALLOW})"
    )
    assert want in flat(gate_scope("008"))


def test_writeup_velocity_share_names_both_protocols() -> None:
    """Ruling A1-32 docs-3: seed 0's benign velocity share and its largest per-agent share."""
    rev, alw = (_row("sw1", 0, "a1", p) for p in ("review", "allow"))
    vr, va = (r["benign_share_ge_0_5"]["velocity"] for r in (rev, alw))
    share = f"{pct(vr)} on seed 0 ({pct(va)} under {ALLOW}"
    per_r, per_a = (r["per_agent_benign_velocity_share_ge_0_5"] for r in (rev, alw))
    agent = max(per_r, key=lambda a: per_r[a])
    assert pct(per_r[agent], 1) == pct(per_a[agent], 1)  # "under both protocols"
    [unit] = units_with("writeup", share)
    assert REVIEW in unit
    assert f"reaches {pct(per_r[agent], 1)} on {agent} (the same under both protocols)" in unit


def leaderboard_table(split: str) -> dict[str, list[str]]:
    """``docs/leaderboard.md``'s ranking table of one split: run name -> cells."""
    text = LEADERBOARD.read_text()
    sec = text[text.index(f"#### Leaderboard: synthetic benchmark (SW1), split `{split}`") :]
    sec = sec[: sec.index("#### Per-family")]
    rows = {}
    for line in sec.splitlines():
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if line.startswith("| ") and parts[0] not in ("Rank", "---"):
            rows[parts[1]] = parts
    return rows


def leaderboard_control_fpr(split: str, run: str) -> str:
    text = LEADERBOARD.read_text()
    sec = text[
        text.index(f"#### Per-family attack recall: synthetic benchmark (SW1), split `{split}`") :
    ]
    [line] = [ln for ln in sec.splitlines() if ln.startswith(f"| {run} | per_agent |")]
    return line.strip().strip("|").split("|")[-1].strip()


def _fpr(split: str, run: str) -> tuple[float, str]:
    """(per-agent FPR@95, "value [lo, hi]") of a leaderboard row."""
    parts = leaderboard_table(split)[run]
    v = parts[5].split()[0]
    return float(v), f"{v} {parts[6]}"


@pytest.mark.parametrize("name", ["writeup", "008"])
def test_ranking_metric_disclosure_is_the_leaderboard(name: str) -> None:
    """Ruling A1-33 critic-1: on the leaderboard's ranking metric the engine before A1 is better
    under the oracle-reviewer loop on both splits (CIs overlap; it ranks first by time); the
    by-agent control FPR rises; under spec §5.2 allow-all the amended engine is lower."""
    [unit] = units_with(name, "ranking metric")
    assert REVIEW in unit and ALLOW in unit
    (at, at_s), (pt, pt_s) = _fpr("time", "engine"), _fpr("time", "engine-pre-a1")
    (ag, ag_s), (pg, pg_s) = _fpr("agent", "engine"), _fpr("agent", "engine-pre-a1")
    assert pt < at and pg < ag  # the engine before A1 is better under the oracle-reviewer loop
    for s1, s2 in ((at_s, pt_s), (ag_s, pg_s)):
        lo1, hi1 = map(float, s1.split("[")[1].rstrip("]").split(","))
        lo2, hi2 = map(float, s2.split("[")[1].rstrip("]").split(","))
        assert lo1 <= hi2 and lo2 <= hi1  # "the 95 % CIs overlap"
    assert f"is {at_s} for the amended engine and {pt_s} for the engine before A1 by time" in unit
    assert f"and {ag_s} against {pg_s} by agent" in unit
    assert "the 95 % CIs overlap" in unit
    assert leaderboard_table("time")["engine-pre-a1"][0] == "1"
    assert "ranks first by time" in unit
    ca, cp = (leaderboard_control_fpr("agent", r) for r in ("engine", "engine-pre-a1"))
    assert (
        float(ca) > float(cp)
        and f"control FPR at that operating point is {ca} against {cp}" in unit
    )
    (aat, _), (apt, _) = _fpr("time", "engine-allow-all"), _fpr("time", "engine-pre-a1-allow-all")
    (aag, _), (apg, _) = _fpr("agent", "engine-allow-all"), _fpr("agent", "engine-pre-a1-allow-all")
    assert aat < apt and aag < apg
    assert f"({aat:.3f} against {apt:.3f} by time, {aag:.3f} against {apg:.3f} by agent)" in unit
    assert "Nothing was retuned" in unit


def readme_report_value(pattern: str) -> re.Match[str]:
    m = re.search(pattern, readme_report_block())
    assert m, pattern
    return m


def test_writeup_review_run_numbers_carry_the_label() -> None:
    """Ruling A1-32 docs-3: 0.490 (twice) and slow_drift's 0.656 (143/218) are the oracle-reviewer
    run's numbers; 0.542 is the spec §5.2 allow-all row."""
    review_fpr = leaderboard_table("time")["engine"][5].split()[0]
    allow_fpr = leaderboard_table("time")["engine-allow-all"][5].split()[0]
    assert review_fpr == "0.490"
    units = units_with("writeup", f"FPR@95 of {review_fpr}") + units_with(
        "writeup", f"per-agent FPR of {review_fpr}"
    )
    assert len(units) == 2
    for u in units:
        assert f"{REVIEW}" in u and f"{allow_fpr} under {ALLOW}" in u, u
    m = readme_report_value(r"\| slow_drift \| all \| (\d\.\d{3}) \((\d+)/(\d+)\) \|")
    [unit] = units_with("writeup", f"({m.group(1)}, {m.group(2)} of {m.group(3)} envelopes")
    assert REVIEW in unit


def test_latency_sentences_name_their_measurement() -> None:
    """Ruling A1-32 docs-2: "stayed below the budget" is the three warm-key perf-test runs of
    matrix.json's latency note; the leaderboard replays' p99 (README report block) is about
    0.3 ms; neither is a latency result."""
    note = _matrix()["latency_note"]
    p99s = [float(x) for x in re.findall(r"\d+\.\d", note.split("p99 =")[1].split("(")[0])]
    assert note.count("Three runs of") == 1 and len(p99s) == 3 and max(p99s) < 200
    p99 = float(readme_report_value(r"Latency: p50 [\d.]+ us, p99 ([\d.]+) us").group(1))
    assert round(p99 / 1000, 1) == 0.3
    text = flat(DOCS["writeup"].read_text())
    below = [m.start() for m in re.finditer("stayed below", text)]
    assert len(below) == 2
    for i in below:
        assert "three runs of the warm-key perf test" in text[i - 120 : i].lower(), text[
            i - 120 : i
        ]
    assert text.count("showed p99 of about 0.3 ms") == 2
    assert (
        text.count("neither is a latency result") + text.count("Neither is a latency result") == 2
    )
    assert "three runs of the warm-key perf test" in flat(gate_scope("008"))


def test_writeup_top_up_is_a_fixed_rate_pass() -> None:
    """Ruling A1-32 docs-5: the top-up rate is sw1_config's, 81 is matrix.json's seed-0 gate cell,
    and the 50 floor is the acceptance test's."""
    rate = sw1_config(0).magnitude_topup_rate
    n = _matrix()["census"]["0"]["gated_test_n"]
    acc = (ROOT / "tests/acceptance/test_tier1_exit.py").read_text()
    m = re.search(r"^MIN_MAGNITUDE_N = (\d+)", acc, flags=re.M)
    assert m
    text = flat(DOCS["writeup"].read_text())
    assert f"fixed-rate pass at {rate}" in text
    assert f"({n} spikes in the seed-0 gate cell" in text
    assert f"fewer than {m.group(1)} gated spikes fails the acceptance test (D-026)" in text
    assert "so the gate cell holds at least" not in text


def test_top_up_rate_disclosure_is_the_specs() -> None:
    """Ruling A1-32 docs-1: decision 008 and writeup §8 repeat spec A1.4's disclosure."""
    spec = flat(SPEC.read_text())
    m = re.search(r"fixed count of (\d+) .{0,60}? a rate of (0\.\d+)", spec)
    assert m and "their recalls were known when 0.015 was chosen" in spec
    for name in ("writeup", "008"):
        [unit] = units_with(name, f"fixed count of {m.group(1)}")
        assert f"a rate of {m.group(2)}" in unit and "recalls were known" in unit, name
        assert str(sw1_config(0).magnitude_topup_rate) in unit, name


def _section(text: str, head: str) -> str:
    s = text[text.index(head) + len(head) :]
    return s[: s.index("\n#")] if "\n#" in s else s


def test_008_copies_spec_a1_13_verbatim() -> None:
    spec = _section(SPEC.read_text(), "### A1.13 Decisions left to the user\n")
    d008 = DOCS["008"].read_text().split("Spec A1.13, copied verbatim:\n", 1)[1]
    assert spec.strip() == d008.strip()


def test_a1_3_limit_is_disclosed_and_pinned() -> None:
    """Ruling A1-30: spec A1.12 and A1.13 item 6, decision 008 and the writeup name the case the
    core test pins."""
    test = "test_pinned_contradiction_not_above_the_learned_range_scores_low"
    assert f"def {test}()" in (ROOT / "tests/core/test_engine.py").read_text()
    spec = SPEC.read_text()
    a1_12 = flat(_section(spec, "### A1.12 What is still not met\n"))
    assert f"tests/core/test_engine.py::{test}" in a1_12
    item6 = flat(_section(spec, "### A1.13 Decisions left to the user\n")).split(" 6. ")[1]
    item6 = item6.split(" 7. ")[0]
    for text in (a1_12, item6, flat(gate_scope("008")), flat(DOCS["writeup"].read_text())):
        assert "`qty = -5`" in text and "fraction between 0 and 1" in text and "about 0.01" in text


def test_writeup_manual_inspection_note_is_narrowed() -> None:
    """Ruling A1-28: §7's (manual inspection) note covers the parquet-derived counts and ranges
    only and names where the other numbers come from."""
    text = DOCS["writeup"].read_text()
    s7 = flat(text[text.index("## 7. Failure cases") : text.index("## 8. Limitations")])
    intro = s7.split(" - **")[0]
    assert "Every count and score range in this section" not in intro
    assert "scores.parquet" in intro and "*(manual inspection)*" in intro
    for src in ("matrix.json", "docs/leaderboard.md", "README"):
        assert src in intro, src
