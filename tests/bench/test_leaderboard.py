from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson
import pytest

from agentfw.bench.leaderboard import ORACLE_CAVEAT, Leaderboard, build_leaderboard
from agentfw.bench.leaderboard import render_leaderboard as render
from agentfw.bench.replay import SplitSpec
from agentfw.bench.run import run
from agentfw.sandbox.llm import ChatTurn, ScriptedChatClient, Usage

SHA = "a" * 64


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
    assert cav["random"] == []
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
