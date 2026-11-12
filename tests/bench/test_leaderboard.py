from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import orjson
import pytest

from agentfw.bench.gate import MagnitudeCensus, render_census
from agentfw.bench.leaderboard import (
    ALLOW_ALL_CAVEAT,
    ORACLE_CAVEAT,
    ORACLE_DIAGNOSTIC_CAVEAT,
    Leaderboard,
    build_leaderboard,
    render_gate_section,
)
from agentfw.bench.leaderboard import render_leaderboard as render
from agentfw.bench.metrics import GATE_FAMILIES
from agentfw.bench.replay import PROTOCOL_LABELS, SplitSpec
from agentfw.bench.run import run
from agentfw.bench.sweeps import PRE_A1_SWITCHES
from agentfw.core.config import EngineConfig
from agentfw.sandbox.llm import ChatTurn, ScriptedChatClient, Usage
from agentfw.synth.generate import sw1_config
from agentfw.synth.load import load_personas

SHA = "a" * 64
CENSUS = MagnitudeCensus(
    gated_test_n=81,
    by_persona={"lead-router-a": 17, "support-refunds-c": 22, "payouts-ops-a": 1},
    share_strength_ge_10=0.9696969696969697,
    median_strength=48.27679911482127,
    n_spikes=264,
)


def _report(
    root: Path,
    name: str,
    *,
    fpr: float | None = 0.5,
    recall: float | None = 0.95,
    auprc: float | None = 0.3,
    sha: str = SHA,
    split: str = "time",
    detector: str | None = None,
    extras: dict[str, Any] | None = None,
) -> Path:
    """A minimal report.json at ``root/name/report.json`` (only the fields a leaderboard reads)."""
    head = {
        "auprc": auprc,
        "auprc_ci": [0.1, 0.3],
        "fpr95_global": 0.6,
        "fpr95_per_agent": fpr,
        "fpr95_per_agent_ci": [0.2, 0.7],
        "per_agent_recall": recall,
        "warm_only": True,
        "n_rows": 1000,
        "n_attack_rows": 40,
    }
    rep = {
        "headline": head,
        "headline_all": {**head, "warm_only": False},
        "meta": {
            "git_sha": "0123456789abcdef",
            "seed": 0,
            "dataset_sha": sha,
            "detector": detector or name,
            "detector_config": {},
            "split": {"kind": split, "train_frac": 0.7, "seed": 0},
            "created_unix": 0,
        },
        "per_family": {
            "mode": "per_agent",
            "warm_attack": {
                "novel_target": {"all": {"recall": 0.5, "n_detected": 1, "n_total": 2}},
                "magnitude_spike": {
                    "3-10": {"recall": 0.0, "n_detected": 0, "n_total": 1},
                    "all": {"recall": None, "n_detected": 0, "n_total": 0},
                },
            },
        },
        "control_fpr": {"all": 0.2, "warm": 0.125},
        "extras": {"notes": [], "n_boot": 1000, **(extras or {})},
    }
    p = root / name / "report.json"
    p.parent.mkdir(parents=True)
    p.write_bytes(orjson.dumps(rep))
    return p


def test_mixed_dataset_sha_raises(tmp_path: Path) -> None:
    a = _report(tmp_path, "engine")
    b = _report(tmp_path, "random", sha="b" * 64)
    with pytest.raises(ValueError, match="dataset_sha"):
        build_leaderboard([a, b])


def test_mixed_split_raises(tmp_path: Path) -> None:
    a = _report(tmp_path, "engine")
    b = _report(tmp_path, "random", split="agent")
    with pytest.raises(ValueError, match="split"):
        build_leaderboard([a, b])


def test_no_reports_raises() -> None:
    with pytest.raises(ValueError, match="no report"):
        build_leaderboard([])


def test_sorted_by_per_agent_fpr_then_auprc_nan_last(tmp_path: Path) -> None:
    paths = [
        _report(tmp_path, "a", fpr=0.5, auprc=0.3),
        _report(tmp_path, "nan-fpr", fpr=None, recall=0.8, auprc=0.9),
        _report(tmp_path, "b", fpr=0.2, auprc=0.1),
        _report(tmp_path, "e", fpr=0.9, auprc=0.95),
        _report(tmp_path, "c", fpr=0.2, auprc=0.4),
        _report(tmp_path, "nan-both", fpr=None, recall=0.8, auprc=None),
    ]
    lb = build_leaderboard(paths)
    assert [r.name for r in lb.rows] == ["c", "b", "a", "e", "nan-fpr", "nan-both"]
    assert lb.dataset_sha == SHA
    assert lb.split == "time"


def test_rows_read_the_warm_headline_and_run_dir_name(tmp_path: Path) -> None:
    p = _report(tmp_path, "engine-sample", detector="engine", fpr=0.25, recall=0.96, auprc=0.5)
    (row,) = build_leaderboard([p]).rows
    assert row.name == "engine-sample"
    assert row.detector == "engine"
    assert (row.fpr95_per_agent, row.per_agent_recall, row.auprc) == (0.25, 0.96, 0.5)
    assert row.auprc_ci == (0.1, 0.3)
    assert row.fpr95_per_agent_ci == (0.2, 0.7)
    assert row.caveats == []


def test_unreachable_recall_rendered_as_na_with_achieved_recall(tmp_path: Path) -> None:
    md = render(build_leaderboard([_report(tmp_path, "if", fpr=None, recall=0.812)]))
    assert "n/a (recall 0.812)" in md


def test_cis_label_and_dataset_rendered(tmp_path: Path) -> None:
    lb = build_leaderboard([_report(tmp_path, "engine")], label="synthetic benchmark (SW1)")
    md = render(lb)
    assert "synthetic benchmark (SW1)" in md
    assert "[0.100, 0.300]" in md  # AUPRC CI
    assert "[0.200, 0.700]" in md  # FPR@95 per-agent CI
    assert "0.500 (recall 0.950)" in md
    assert SHA[:12] in md
    assert "`time`" in md


def test_sampled_batch_and_oracle_caveats_rendered(tmp_path: Path) -> None:
    paths = [
        _report(tmp_path, "full"),
        _report(tmp_path, "iforest", extras={"batch": True}),
        _report(
            tmp_path,
            "judge-sample",
            extras={"sample": {"spec": "stratified:2000", "n": 2000}},
        ),
        _report(tmp_path, "engine-review", extras={"outcomes": "review"}),
    ]
    lb = build_leaderboard(paths)
    cav = {r.name: r.caveats for r in lb.rows}
    assert cav["full"] == []
    assert any("batch" in c for c in cav["iforest"])
    assert any("sampled (stratified:2000)" in c for c in cav["judge-sample"])
    assert cav["engine-review"] == [ORACLE_CAVEAT]
    assert "optimistic" in ORACLE_CAVEAT and "oracle-reviewer" in ORACLE_CAVEAT
    md = render(lb)
    assert "sampled (stratified:2000)" in md
    assert "batch" in md
    assert ORACLE_CAVEAT in md


def test_oracle_diagnostic_and_other_outcomes_caveats(tmp_path: Path) -> None:
    paths = [
        _report(tmp_path, "engine-oracle", extras={"outcomes": "oracle"}),
        _report(tmp_path, "engine-enforce", extras={"outcomes": "enforce"}),
        _report(tmp_path, "engine-allow", extras={"outcomes": "allow"}),
    ]
    cav = {r.name: r.caveats for r in build_leaderboard(paths).rows}
    assert cav["engine-oracle"] == [ORACLE_DIAGNOSTIC_CAVEAT]
    assert ORACLE_DIAGNOSTIC_CAVEAT == "diagnostic upper bound (reads every label) → optimistic"
    assert cav["engine-enforce"] == ["outcomes=enforce: not the spec 5.2 allow-all protocol"]
    # User decision (decision 008): allow-all is a clearly labelled comparison row.
    assert cav["engine-allow"] == [ALLOW_ALL_CAVEAT] == ["spec §5.2 allow-all"]
    assert ORACLE_CAVEAT == (
        "oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic)"
    )


def test_not_run_rows_listed_after_ranked_rows(tmp_path: Path) -> None:
    lb = build_leaderboard(
        [_report(tmp_path, "engine")], not_run={"llm_judge": "requires a live judge model"}
    )
    md = render(lb)
    assert "not run (requires a live judge model)" in md
    assert md.index("| 1 | engine") < md.index("llm_judge")
    assert [r.name for r in lb.rows] == ["engine"]


def test_family_recall_and_control_fpr_table(tmp_path: Path) -> None:
    lb = build_leaderboard([_report(tmp_path, "engine")])
    (row,) = lb.rows
    assert row.family_recall == {"magnitude_spike": None, "novel_target": 0.5}
    assert row.control_fpr == 0.125
    assert row.family_mode == "per_agent"
    md = render(lb)
    assert "| Run | Thresholds | magnitude_spike | novel_target | Control FPR |" in md
    assert "| engine | per_agent | n/a | 0.500 | 0.125 |" in md


def test_leaderboard_round_trips_as_json(tmp_path: Path) -> None:
    lb = build_leaderboard([_report(tmp_path, "x", fpr=None)], label="L")
    again = Leaderboard.model_validate_json(lb.model_dump_json())
    assert again == lb


def _judge_client(n: int) -> ScriptedChatClient:
    turns = [
        ChatTurn(content=f'{{"suspicion": {(i % 10) / 10}}}', tool_calls=[], usage=Usage(1, 1))
        for i in range(n)
    ]
    return ScriptedChatClient(turns)


def test_real_runs_full_and_sampled_share_one_table(tiny_dataset: Path, tmp_path: Path) -> None:
    """Full-split baselines plus engine and a scripted llm_judge on the same stratified sample."""
    res = tmp_path / "results"
    n_sample = 60
    for det in ("random", "static_threshold"):
        run(tiny_dataset, det, SplitSpec(), res / det, n_boot=20)
    run(
        tiny_dataset,
        "engine",
        SplitSpec(),
        res / "engine-sample",
        n_boot=20,
        sample=f"stratified:{n_sample}",
    )
    run(
        tiny_dataset,
        "llm_judge",
        SplitSpec(),
        res / "llm_judge-sample",
        n_boot=20,
        sample=f"stratified:{n_sample}",
        client=_judge_client(n_sample),
        cache_dir=tmp_path / "judge-cache",
        model_id="scripted",
    )
    lb = build_leaderboard(sorted(res.glob("*/report.json")), label="tiny")
    names = {r.name for r in lb.rows}
    assert names == {"random", "static_threshold", "engine-sample", "llm_judge-sample"}
    cav = {r.name: r.caveats for r in lb.rows}
    assert any(c.startswith(f"sampled (stratified:{n_sample})") for c in cav["engine-sample"])
    assert any(c.startswith(f"sampled (stratified:{n_sample})") for c in cav["llm_judge-sample"])
    # Every real run records its protocol; the default is the oracle-reviewer loop.
    assert cav["random"] == [ORACLE_CAVEAT]
    assert all(ORACLE_CAVEAT in c for c in cav.values())
    md = render(lb)
    for name in names:
        assert f"| {name} |" in md


def test_cli_renders_reports_to_file(tmp_path: Path) -> None:
    from agentfw.bench.leaderboard import main

    paths = [_report(tmp_path, "engine", fpr=0.2), _report(tmp_path, "random", fpr=0.9)]
    out = tmp_path / "lb.md"
    main(["--label", "sandbox dataset v1", "--out", str(out), *map(str, paths)])
    md = out.read_text()
    assert "sandbox dataset v1" in md
    assert md.index("| 1 | engine") < md.index("| 2 | random")


def test_cli_refuses_mixed_datasets(tmp_path: Path) -> None:
    from agentfw.bench.leaderboard import main

    a, b = _report(tmp_path, "engine"), _report(tmp_path, "random", sha="c" * 64)
    with pytest.raises(SystemExit, match="dataset_sha"):
        main(["--out", str(tmp_path / "x.md"), str(a), str(b)])


def test_cli_rejects_malformed_not_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from agentfw.bench.leaderboard import main

    a = _report(tmp_path, "engine")
    with pytest.raises(SystemExit) as exc:
        main(["--out", str(tmp_path / "x.md"), "--not-run", "llm_judge", str(a)])
    assert exc.value.code == 2  # argparse usage error
    assert "NAME=REASON" in capsys.readouterr().err
    assert not (tmp_path / "x.md").exists()


# --- Task 1.25: gate section under both protocols, pre-A1 rows, SW1 header (spec A1.5) ----------

ROOT = Path(__file__).resolve().parents[2]
MATRIX_JSON = ROOT / "docs/decisions/evidence/a1-gate/matrix.json"
REVIEW = PROTOCOL_LABELS["review"]
ALLOW = PROTOCOL_LABELS["allow"]


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_build_leaderboard_script", ROOT / "scripts/build_leaderboard.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # importing must not run the script
    return mod


build_leaderboard_script = _load_script()


def _gate(passed: bool, mag: tuple[int, int]) -> dict[str, Any]:
    """A report.json ``gate`` section (``GatePreview`` as orjson writes it)."""
    cells = {
        fam: {
            "bucket": bucket,
            "detector": det,
            "recall": (mag[0] / mag[1]) if fam == "magnitude_spike" else 1.0,
            "n_detected": mag[0] if fam == "magnitude_spike" else 10,
            "n_total": mag[1] if fam == "magnitude_spike" else 10,
            "top_reason_share": 0.5,
            "among_reasons_share": 1.0,
        }
        for fam, (bucket, det) in GATE_FAMILIES.items()
    }
    return {
        "cells": cells,
        "benign_fpr": 0.0058,
        "control_fpr": 0.035,
        "min_recall": 1.0,
        "passed": passed,
        "missing": [],
        "failures": [] if passed else ["recall:novel_target"],
    }


def _gate_report(outcomes: str, passed: bool, mag: tuple[int, int]) -> dict[str, Any]:
    return {
        "meta": {"dataset_sha": SHA, "seed": 0, "detector": "engine", "split": {"kind": "time"}},
        "extras": {"outcomes": outcomes, "n_boot": 0},
        "gate": _gate(passed, mag),
    }


def _reports(review_pass: bool = True, allow_pass: bool = False) -> dict[str, dict[str, Any]]:
    return {
        "engine": _gate_report("review", review_pass, (77, 78)),
        "engine-allow-all": _gate_report("allow", allow_pass, (71, 78)),
        "engine-pre-a1": _gate_report("review", False, (73, 78)),
        "engine-pre-a1-allow-all": _gate_report("allow", False, (59, 78)),
    }


@pytest.fixture
def reports() -> dict[str, dict[str, Any]]:
    return _reports()


SENS_SHA = "e" * 64


def _sens(mag: tuple[int, int] = (51, 64)) -> dict[str, dict[str, Any]]:
    """The four gate runs on a sensitivity dataset (another dataset sha than ``_reports``)."""
    out = {name: {**rep, "gate": _gate(False, mag)} for name, rep in _reports().items()}
    for rep in out.values():
        rep["meta"] = {**rep["meta"], "dataset_sha": SENS_SHA}
    return out


def _protocol_lines(md: str) -> list[tuple[int, str]]:
    """(line number, protocol) of every table line that starts with a protocol label."""
    out = []
    for i, ln in enumerate(md.splitlines()):
        for mode in ("review", "allow"):
            if ln.startswith(f"| {PROTOCOL_LABELS[mode]} |"):
                out.append((i, mode))
    return out


def test_gate_section_shows_both_protocols_side_by_side(reports: dict[str, Any]) -> None:
    md = render_gate_section(reports)
    assert PROTOCOL_LABELS["review"] in md and PROTOCOL_LABELS["allow"] in md
    assert "original spec §9 criterion" in md and "among" in md.lower()
    assert "engine-pre-a1" in md
    # Rule 1: every review line is directly followed by its allow line, the engine pair first.
    rows = _protocol_lines(md)
    assert [mode for _, mode in rows] == ["review", "allow", "review", "allow"]
    assert rows[1][0] == rows[0][0] + 1 and rows[3][0] == rows[2][0] + 1
    assert md.index("`engine-allow-all`") < md.index("`engine-pre-a1`")
    assert "Tier-1 gate met" not in md


def test_gate_section_refuses_a_single_protocol(reports: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="both protocols"):
        render_gate_section({k: v for k, v in reports.items() if k != "engine-allow-all"})
    with pytest.raises(ValueError, match="both protocols"):
        render_gate_section({k: v for k, v in reports.items() if k != "engine-pre-a1"})
    swapped = {**reports, "engine-allow-all": _gate_report("review", False, (71, 78))}
    with pytest.raises(ValueError, match="both protocols"):
        render_gate_section(swapped)  # a run named allow-all that replayed the review loop
    no_gate = {**reports, "engine": {**reports["engine"], "gate": None}}
    with pytest.raises(ValueError, match="gate"):
        render_gate_section(no_gate)


def test_gate_section_withholds_magnitude_without_the_sensitivity_rows(
    reports: dict[str, Any],
) -> None:
    md = render_gate_section(reports)  # rule 4: the sensitivity companion is missing
    assert "withheld" in md
    assert "(77/78)" not in md and "(71/78)" not in md
    md = render_gate_section(reports, sensitivity=_sens())
    assert "withheld" not in md
    assert "0.987 (77/78)" in md and "0.910 (71/78)" in md and "0.797 (51/64)" in md
    assert md.index("(77/78)") < md.index("(51/64)")  # companion rows after the gate rows
    assert "not a gate result" in md


def test_gate_section_pre_a1_block_shows_the_pre_a1_runs(reports: dict[str, Any]) -> None:
    md = render_gate_section(reports, sensitivity=_sens())
    # engine-pre-a1 is (73/78), engine-pre-a1-allow-all is (59/78) in the fixture.
    assert "0.936 (73/78)" in md and "0.756 (59/78)" in md
    assert (
        md.index("### engine")
        < md.index("(77/78)")
        < md.index("(71/78)")
        < md.index("### pre-A1")
        < md.index("(73/78)")
        < md.index("(59/78)")
        < md.index("(51/64)")
    )


@pytest.mark.parametrize(
    ("run_name", "meta"),
    [
        ("engine-pre-a1", {"dataset_sha": "b" * 64}),  # another dataset
        ("engine-allow-all", {"split": {"kind": "agent"}}),  # another split
    ],
)
def test_gate_section_refuses_runs_mixing_datasets_or_splits(
    reports: dict[str, Any], run_name: str, meta: dict[str, Any]
) -> None:
    mixed = {**reports, run_name: {**reports[run_name]}}
    mixed[run_name]["meta"] = {**reports[run_name]["meta"], **meta}
    with pytest.raises(ValueError, match="mix datasets or splits"):
        render_gate_section(mixed)


@pytest.mark.parametrize("drop", ["meta", "dataset_sha", "split"])
def test_gate_section_refuses_a_report_without_meta(reports: dict[str, Any], drop: str) -> None:
    bad = {**reports, "engine-pre-a1": {**reports["engine-pre-a1"]}}
    if drop == "meta":
        del bad["engine-pre-a1"]["meta"]
    else:
        bad["engine-pre-a1"]["meta"] = {
            k: v for k, v in reports["engine-pre-a1"]["meta"].items() if k != drop
        }
    with pytest.raises(ValueError, match="'engine-pre-a1'"):
        render_gate_section(bad)


def test_gate_section_refuses_a_pair_of_different_engine_configs(
    reports: dict[str, Any],
) -> None:
    """Spec A1.5: both protocols of a pair replay the same engine configuration."""
    pre = orjson.loads(orjson.dumps(dataclasses.asdict(EngineConfig().replace(**PRE_A1_SWITCHES))))
    bad = {**reports, "engine-allow-all": {**reports["engine-allow-all"]}}
    bad["engine-allow-all"]["meta"] = {
        **reports["engine-allow-all"]["meta"],
        "detector_config": {"cfg": pre},
    }
    with pytest.raises(ValueError, match="engine configuration"):
        render_gate_section(bad)
    ok = {name: {**rep} for name, rep in reports.items()}
    for name in ("engine-pre-a1", "engine-pre-a1-allow-all"):
        ok[name]["meta"] = {**reports[name]["meta"], "detector_config": {"cfg": pre}}
    assert "### pre-A1" in render_gate_section(ok)  # a pair sharing its config renders


def test_gate_section_refuses_a_run_that_is_not_the_engine(reports: dict[str, Any]) -> None:
    bad = {**reports, "engine": {**reports["engine"]}}
    bad["engine"]["meta"] = {**reports["engine"]["meta"], "detector": "random"}
    with pytest.raises(ValueError, match="'engine'.*not an engine replay"):
        render_gate_section(bad)


def test_build_leaderboard_uses_sw1_config() -> None:
    assert sw1_config() == build_leaderboard_script.SW1
    assert {n for n, *_ in build_leaderboard_script.RUNS} >= {
        "engine",
        "engine-allow-all",
        "engine-pre-a1",
        "engine-pre-a1-allow-all",
    }
    runs = {name: rest for name, *rest in build_leaderboard_script.RUNS}
    pre = EngineConfig().replace(**PRE_A1_SWITCHES)
    assert runs["engine-pre-a1"] == ["engine", "review", pre]
    assert runs["engine-pre-a1-allow-all"] == ["engine", "allow", pre]
    assert runs["engine"] == ["engine", "review", None]
    assert runs["engine-allow-all"] == ["engine", "allow", None]
    assert all(len(r) == 4 for r in build_leaderboard_script.RUNS)
    # The gate section reads the gate's own split (matrix.json is computed with SplitSpec()).
    assert SplitSpec().kind == build_leaderboard_script.GATE_SPLIT


@pytest.mark.parametrize(
    ("review_pass", "allow_pass", "sentence"),
    [
        (True, False, f"the amended gate passes under {REVIEW} and is not met under {ALLOW}"),
        (True, True, f"the amended gate passes under {REVIEW} and passes under {ALLOW}"),
        (False, False, f"the amended gate is not met under {REVIEW} and is not met under {ALLOW}"),
        (False, True, f"the amended gate is not met under {REVIEW} and passes under {ALLOW}"),
    ],
)
def test_doc_header_states_the_gate_from_the_reports(
    review_pass: bool, allow_pass: bool, sentence: str
) -> None:
    md = build_leaderboard_script._doc(
        ["SECTION"], SHA, _reports(review_pass, allow_pass), census=CENSUS
    )
    assert sentence in md
    assert "Tier-1 gate met" not in md and "Tier-1 gate is met" not in md
    assert "not met under either protocol" not in md  # the old hard-coded sentence


def test_doc_header_names_the_dataset_sw1_and_auprc_caveat(reports: dict[str, Any]) -> None:
    md = build_leaderboard_script._doc(["SECTION-A", "SECTION-B"], SHA, reports, census=CENSUS)
    sw1 = sw1_config()
    assert SHA[:12] in md
    assert f"magnitude_far_mult {sw1.magnitude_far_mult:g}" in md
    assert f"magnitude_topup_rate {sw1.magnitude_topup_rate:g}" in md
    assert f'magnitude_mode "{sw1.magnitude_mode}"' in md
    assert "6451d56" in md and "AUPRC" in md and "not comparable" in md
    assert md.index("SECTION-A") < md.index("SECTION-B")
    # The gate section is in the page with its magnitude cells (both companions present).
    assert "0.987 (77/78)" in md and "withheld" not in md
    personas = len(load_personas(sw1.personas_dir))
    assert f"{personas} personas" in md
    # The SW1 claim names its one exception: the sensitivity companion rows.
    assert "except the companion rows on the sensitivity dataset" in md
    # The leaderboard sections get their own level-2 heading after the companion block, so
    # the outline never puts the SW1 tables under the sensitivity dataset.
    lines = md.splitlines()
    comp = next(i for i, ln in enumerate(lines) if ln.startswith("## Companion"))
    sec_a = lines.index("SECTION-A")
    h2 = [i for i, ln in enumerate(lines) if ln.startswith("## ") and comp < i < sec_a]
    assert h2 and "SW1" in lines[h2[0]], lines[comp:sec_a]


def test_doc_copies_the_sensitivity_rows_from_matrix_json(reports: dict[str, Any]) -> None:
    md = build_leaderboard_script._doc([], SHA, reports, census=CENSUS)
    rows = [r for r in orjson.loads(MATRIX_JSON.read_bytes())["rows"] if r["dataset"] == "sens"]
    assert {(r["engine"], r["protocol"]) for r in rows} == {
        (e, p) for e in ("a1", "pre-a1") for p in ("review", "allow")
    }
    # Rule 3: each sens row's whole table line, family cells (recall, among, top-1) then benign
    # FPR (4 dp) and control FPR (3 dp), under the right engine block and protocol label.
    comp = md.index("## Companion")
    blocks = {"a1": md.index("### engine", comp), "pre-a1": md.index("### pre-A1", comp)}
    assert blocks["a1"] < blocks["pre-a1"]
    for r in rows:
        cells = [
            f"{c['recall']:.3f} ({c['n_detected']}/{c['n_total']}); {c['among']:.2f}; "
            f"{c['top1']:.2f}"
            for c in (r["families"][fam] for fam in GATE_FAMILIES)
        ]
        line = (
            f"| {PROTOCOL_LABELS[r['protocol']]} | "
            + " | ".join(cells)
            + f" | {r['benign_fpr']:.4f} | {r['control_fpr']:.3f} |"
        )
        at = md.find(line, comp)
        assert at != -1, (r["engine"], r["protocol"])
        end = blocks["pre-a1"] if r["engine"] == "a1" else len(md)
        assert blocks[r["engine"]] < at < end, (r["engine"], r["protocol"])
    assert f"`{rows[0]['dataset_sha'][:12]}`" in md


def test_doc_refuses_a_matrix_without_the_sensitivity_rows(
    reports: dict[str, Any], tmp_path: Path
) -> None:
    payload = orjson.loads(MATRIX_JSON.read_bytes())
    payload["rows"] = [
        r for r in payload["rows"] if not (r["dataset"] == "sens" and r["engine"] == "pre-a1")
    ]
    bad = tmp_path / "matrix.json"
    bad.write_bytes(orjson.dumps(payload))
    with pytest.raises(ValueError, match="sensitivity"):
        build_leaderboard_script._doc([], SHA, reports, census=CENSUS, matrix=bad)


class _NoJobs:
    """Stands in for ProcessPoolExecutor: main() must not replay anything in this test."""

    def __init__(self, max_workers: int) -> None:
        pass

    def __enter__(self) -> _NoJobs:
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def map(self, *a: object) -> list[str]:
        raise AssertionError("--reuse with fresh reports must not schedule a run")


def test_main_feeds_the_gate_section_from_the_time_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main(): the gate rows come from results/<GATE_SPLIT>, never from the agent split."""
    mod = build_leaderboard_script
    results = tmp_path / "results"
    # Distinct magnitude cells per split: the time split's engine is (77/78), the agent's (11/78).
    mag = {"time": (77, 78), "agent": (11, 78)}
    for split in mod.SPLITS:
        gates = _reports()
        gates["engine"]["gate"] = _gate(True, mag[split])
        for name, det, oc, cfg in mod.RUNS:
            want = None if cfg is None else orjson.loads(orjson.dumps(dataclasses.asdict(cfg)))
            p = _report(results / split, name, split=split, detector=det, extras={"outcomes": oc})
            rep = orjson.loads(p.read_bytes())
            rep["meta"]["detector_config"] = {} if want is None else {"cfg": want}
            rep["extras"]["n_boot"] = 0
            if name in gates:
                rep["gate"] = gates[name]["gate"]
            p.write_bytes(orjson.dumps(rep))
    monkeypatch.setattr(mod, "_dataset", lambda d: SHA)
    censused: list[Path] = []

    def fake_census(traces_dir: Path) -> MagnitudeCensus:
        censused.append(traces_dir)
        return CENSUS

    monkeypatch.setattr(mod, "dataset_census", fake_census)
    monkeypatch.setattr(mod, "ProcessPoolExecutor", _NoJobs)
    out = tmp_path / "leaderboard.md"
    argv = ["build_leaderboard.py", "--results", str(results), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", [*argv, "--reuse", "--n-boot", "0", "--traces", "tr"])
    mod.main()
    md = out.read_text()
    assert "0.987 (77/78)" in md and "(11/78)" not in md
    assert censused == [Path("tr")]  # the census is of the dataset the page reports
    assert md.count(render_census(CENSUS)) == 2
    assert "split `time`" in md and "split `agent`" in md  # both leaderboard tables


def test_job_and_fresh_carry_the_engine_config(tiny_dataset: Path, tmp_path: Path) -> None:
    mod = build_leaderboard_script
    pre = EngineConfig().replace(**PRE_A1_SWITCHES)
    out = tmp_path / "engine-pre-a1-allow-all"
    mod._job(tiny_dataset, "engine", "time", out, 0, "allow", pre)
    rep = orjson.loads((out / "report.json").read_bytes())
    assert rep["meta"]["detector_config"]["cfg"]["mag_score"] == "z"
    sha = rep["meta"]["dataset_sha"]
    assert mod._fresh(out, sha, 0, "time", "engine", "allow", pre)
    assert not mod._fresh(out, sha, 0, "time", "engine", "allow", None)  # not the default engine
    assert not mod._fresh(out, sha, 0, "time", "engine", "review", pre)


def test_gate_section_reads_real_reports(tiny_dataset: Path, tmp_path: Path) -> None:
    """The report.json gate sections the leaderboard runs write render as they were measured."""
    pre = EngineConfig().replace(**PRE_A1_SWITCHES)
    reports: dict[str, dict[str, Any]] = {}
    for name, outcomes, cfg in (
        ("engine", "review", None),
        ("engine-allow-all", "allow", None),
        ("engine-pre-a1", "review", pre),
        ("engine-pre-a1-allow-all", "allow", pre),
    ):
        kw = {} if cfg is None else {"cfg": cfg}
        run(tiny_dataset, "engine", SplitSpec(), tmp_path / name, n_boot=0, outcomes=outcomes, **kw)  # type: ignore[arg-type]
        reports[name] = orjson.loads((tmp_path / name / "report.json").read_bytes())
    # The same reports as the companion rows, under another dataset sha (one replay suffices).
    sens = {
        name: {**rep, "meta": {**rep["meta"], "dataset_sha": SENS_SHA}}
        for name, rep in reports.items()
    }
    md = render_gate_section(reports, sensitivity=sens)
    assert [m for _, m in _protocol_lines(md)] == ["review", "allow"] * 4
    for rep in reports.values():
        c = rep["gate"]["cells"]["prereq_skip"]
        assert f"({c['n_detected']}/{c['n_total']})" in md


# --- final review: the gate section says where its companion rows come from (ruling A1-27) ------

_COMPANION_PREFIX = (
    "The magnitude_spike cell is shown only with its two companion rows (spec A1.5 rule 4): "
)


def _companion_paragraph(md: str) -> str:
    return next(ln for ln in md.splitlines() if ln.startswith(_COMPANION_PREFIX))


def test_gate_section_names_its_own_companion_sources(reports: dict[str, Any]) -> None:
    md = render_gate_section(
        reports, sensitivity=_sens(), sensitivity_source="rows copied from `x`"
    )
    para = _companion_paragraph(md)
    assert "CLI" not in para and "--sensitivity-traces" not in para
    assert "runs `engine-pre-a1` and `engine-pre-a1-allow-all`" in para
    assert "rows copied from `x`" in para


def test_doc_companion_paragraph_names_matrix_json(reports: dict[str, Any]) -> None:
    para = _companion_paragraph(build_leaderboard_script._doc([], SHA, reports, census=CENSUS))
    assert "CLI" not in para
    assert "runs `engine-pre-a1` and `engine-pre-a1-allow-all`" in para
    assert "copied from `docs/decisions/evidence/a1-gate/matrix.json`" in para


# --- final review: the magnitude census under the gate tables (spec A1.4, ruling A1-31) ----------


def test_gate_section_prints_the_census_under_the_main_tables(reports: dict[str, Any]) -> None:
    sens = _sens()
    md = render_gate_section(reports, sensitivity=sens, census=CENSUS)
    line = render_census(CENSUS)
    assert md.count(line) == 2
    assert md.rindex(line) < md.index("## Companion")
    assert "Census of the magnitude_spike" not in render_gate_section(reports, sensitivity=sens)


def test_doc_prints_the_census(reports: dict[str, Any]) -> None:
    md = build_leaderboard_script._doc([], SHA, reports, census=CENSUS)
    assert md.count(render_census(CENSUS)) == 2
    assert "n 81 (warm test rows, bucket 10-100) by persona: support-refunds-c 22" in md


def test_doc_refuses_gate_reports_of_another_dataset(reports: dict[str, Any]) -> None:
    """The census is of the dataset ``sha``; the gate rows must be of that dataset too."""
    with pytest.raises(ValueError, match="dataset"):
        build_leaderboard_script._doc([], "f" * 64, reports, census=CENSUS)


# --- final review: the companion rows are of another dataset (ruling A1-32 gate-1) ---------------


@pytest.mark.parametrize("sens_sha", [SHA, SHA[:12]])  # matrix.json stores 12 characters
def test_gate_section_refuses_sensitivity_rows_of_the_gate_dataset(
    reports: dict[str, Any], sens_sha: str
) -> None:
    sens = _sens()
    for rep in sens.values():
        rep["meta"] = {**rep["meta"], "dataset_sha": sens_sha}
    with pytest.raises(ValueError, match="sensitivity.*same dataset"):
        render_gate_section(reports, sensitivity=sens)
