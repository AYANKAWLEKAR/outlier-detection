"""The benchmark protocol (user decision 2026-11-10, decision 008): ``run`` defaults to the
oracle-reviewer approval loop, records it in ``extras.outcomes`` and labels it in the report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson
import pytest

from agentfw.bench.ablation import run_ablation as real_ablation
from agentfw.bench.replay import BENCH_OUTCOMES, PROTOCOL_LABELS, SplitSpec
from agentfw.bench.replay import replay as real_replay
from agentfw.bench.run import main, run

REVIEW_LABEL = "oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic)"
ALLOW_LABEL = "spec §5.2 allow-all"


def _report(out: Path) -> dict[str, Any]:
    rep = orjson.loads((out / "report.json").read_bytes())
    assert isinstance(rep, dict)
    return rep


def test_protocol_labels_are_verbatim() -> None:
    assert BENCH_OUTCOMES == "review"
    assert PROTOCOL_LABELS["review"] == REVIEW_LABEL
    assert PROTOCOL_LABELS["allow"] == ALLOW_LABEL


def test_run_defaults_to_review_records_and_labels_it(tiny_dataset: Path, tmp_path: Path) -> None:
    run(tiny_dataset, "engine", SplitSpec(), tmp_path, n_boot=0)
    assert _report(tmp_path)["extras"]["outcomes"] == "review"
    md = (tmp_path / "report.md").read_text()
    head = md[: md.index("#### Headline")]
    assert f"Protocol: {REVIEW_LABEL}." in head


def test_run_allow_all_is_recorded_and_labelled(tiny_dataset: Path, tmp_path: Path) -> None:
    run(tiny_dataset, "engine", SplitSpec(), tmp_path, n_boot=0, outcomes="allow")
    assert _report(tmp_path)["extras"]["outcomes"] == "allow"
    md = (tmp_path / "report.md").read_text()
    assert f"Protocol: {ALLOW_LABEL}." in md[: md.index("#### Headline")]


def test_baseline_report_records_the_protocol(tiny_dataset: Path, tmp_path: Path) -> None:
    run(tiny_dataset, "static_threshold", SplitSpec(), tmp_path, n_boot=0)
    assert _report(tmp_path)["extras"]["outcomes"] == "review"


def test_run_passes_outcomes_to_replay_and_ablations(
    tiny_dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    def spy_replay(*a: Any, **kw: Any) -> Any:
        seen["replay"] = kw.get("outcomes")
        return real_replay(*a, **kw)

    def spy_ablation(*a: Any, **kw: Any) -> Any:
        seen["ablation"] = kw.get("outcomes")
        return real_ablation(*a, **kw)

    monkeypatch.setattr("agentfw.bench.run.replay", spy_replay)
    monkeypatch.setattr("agentfw.bench.run.run_ablation", spy_ablation)
    run(tiny_dataset, "random", SplitSpec(), tmp_path, n_boot=0, ablations=["random"])
    assert seen == {"replay": "review", "ablation": "review"}
    run(tiny_dataset, "random", SplitSpec(), tmp_path, n_boot=0, ablations=["random"],
        outcomes="oracle")  # fmt: skip
    assert seen == {"replay": "oracle", "ablation": "oracle"}


def test_review_and_allow_replays_differ_for_the_engine(tiny_dataset: Path, tmp_path: Path) -> None:
    run(tiny_dataset, "engine", SplitSpec(), tmp_path / "r", n_boot=0)
    run(tiny_dataset, "engine", SplitSpec(), tmp_path / "a", n_boot=0, outcomes="allow")
    hashes = {_report(tmp_path / d)["extras"]["score_hash"] for d in ("r", "a")}
    assert len(hashes) == 2


def test_cli_outcomes_flag(
    tiny_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["--traces", str(tiny_dataset), "--detector", "random", "--n-boot", "0"]
    main([*args, "--out", str(tmp_path / "d")])
    assert _report(tmp_path / "d")["extras"]["outcomes"] == "review"
    main([*args, "--out", str(tmp_path / "a"), "--outcomes", "allow"])
    assert _report(tmp_path / "a")["extras"]["outcomes"] == "allow"
    with pytest.raises(SystemExit):
        main([*args, "--out", str(tmp_path / "x"), "--outcomes", "deny"])
    assert "invalid choice" in capsys.readouterr().err


def test_report_unknown_protocol_label_matches_the_leaderboard() -> None:
    from agentfw.bench.replay import is_outcome_mode
    from agentfw.bench.report import _protocol_md

    assert _protocol_md("deny") == ["Protocol: outcomes=deny: unknown replay protocol.", ""]
    assert _protocol_md("review") == [f"Protocol: {REVIEW_LABEL}.", ""]
    assert _protocol_md(None) == []
    assert is_outcome_mode("allow") and not is_outcome_mode("deny") and not is_outcome_mode(1)
