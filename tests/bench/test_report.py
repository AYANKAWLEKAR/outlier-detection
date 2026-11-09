from __future__ import annotations

import math
from pathlib import Path

import orjson
import pytest

from agentfw.bench.replay import SplitSpec
from agentfw.bench.report import build_report, render_markdown, update_readme
from agentfw.bench.run import run

TOP_KEYS = {
    "headline",
    "per_family",
    "control_fpr",
    "cold_start",
    "drift",
    "latency",
    "state_bytes",
    "meta",
    "extras",
}


@pytest.fixture(scope="module")
def out_dir(tiny_dataset: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("run")
    run(tiny_dataset, "static_threshold", SplitSpec(), out, n_boot=20)
    return out


def test_run_writes_three_files(out_dir: Path) -> None:
    for name in ("scores.parquet", "report.json", "report.md"):
        assert (out_dir / name).is_file()


def test_report_has_top_level_keys_and_headline(out_dir: Path) -> None:
    rep = orjson.loads((out_dir / "report.json").read_bytes())
    assert set(rep) >= TOP_KEYS
    for h in (rep["headline"], rep["headline_all"]):
        for f in (
            "auprc",
            "auprc_ci",
            "fpr95_global",
            "fpr95_per_agent",
            "fpr95_per_agent_ci",
            "per_agent_recall",
            "warm_only",
        ):
            assert f in h
    assert rep["headline"]["warm_only"] is True
    assert rep["headline_all"]["warm_only"] is False
    assert rep["meta"]["detector"] == "static_threshold"
    assert len(rep["meta"]["dataset_sha"]) == 64
    assert rep["meta"]["split"]["kind"] == "time"


def test_report_json_is_strict_json_with_nan_as_null() -> None:
    from agentfw.bench.metrics import BucketCell, DriftDelay, Metrics, RecallCell
    from agentfw.bench.report import RunMeta

    nan = math.nan
    m = Metrics(
        auprc=nan,
        auprc_ci=(nan, nan),
        thr95_global=nan,
        fpr95_global=nan,
        recall_at_global_thr=nan,
        fpr95_per_agent=nan,
        fpr95_per_agent_ci=(nan, nan),
        per_agent_alpha=nan,
        per_agent_recall=nan,
        breakdown_mode="per_agent",
        per_family_envelope={"f": {"all": RecallCell(nan, 0, 0)}},
        per_family_attack={"f": {"all": RecallCell(nan, 0, 0)}},
        control_fpr=nan,
        cold_start=[BucketCell(0, math.inf, nan, nan, 0, 0)],
        drift=DriftDelay(nan, nan, nan, 0),
        latency_p50_us=nan,
        latency_p99_us=nan,
        n_rows=0,
        n_attack_rows=0,
    )
    meta = RunMeta(
        git_sha="x",
        seed=0,
        dataset_sha="d",
        detector="d",
        detector_config={},
        split=SplitSpec(),
        created_unix=0,
    )
    rep = build_report(m, m, meta)
    raw = orjson.dumps(rep)  # must not raise
    assert b"NaN" not in raw
    assert orjson.loads(raw)["headline"]["auprc"] is None
    assert render_markdown(rep)  # NaN-safe rendering


def test_markdown_has_tables(out_dir: Path) -> None:
    md = (out_dir / "report.md").read_text()
    assert "| Family |" in md
    assert "FPR@95" in md
    assert "recall" in md.lower()


def test_agent_split_note(tiny_dataset: Path, tmp_path: Path) -> None:
    run(tiny_dataset, "random", SplitSpec(kind="agent"), tmp_path, n_boot=5)
    md = (tmp_path / "report.md").read_text()
    assert "pooled train ECDF" in md
    rep = orjson.loads((tmp_path / "report.json").read_bytes())
    assert "pooled train ECDF" in str(rep["extras"]["notes"])


def test_update_readme_idempotent(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("intro\n<!-- report:start -->\nold\n<!-- report:end -->\noutro\n")
    update_readme(readme, "NEW TABLE")
    once = readme.read_text()
    update_readme(readme, "NEW TABLE")
    assert readme.read_text() == once
    assert "NEW TABLE" in once
    assert "old" not in once
    assert once.startswith("intro\n") and once.endswith("outro\n")


def test_update_readme_requires_markers(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("no markers\n")
    with pytest.raises(ValueError):
        update_readme(readme, "x")


def test_cli_end_to_end_update_readme(
    tiny_dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agentfw.bench.run import main

    (tmp_path / "README.md").write_text("hi\n<!-- report:start -->\nOLDMARK\n<!-- report:end -->\n")
    monkeypatch.chdir(tmp_path)
    args = ["--traces", str(tiny_dataset), "--detector", "random", "--out", "o", "--n-boot", "5"]
    main([*args, "--update-readme"])
    first = (tmp_path / "README.md").read_text()
    assert "FPR@95" in first and "OLDMARK" not in first
    main([*args, "--update-readme"])
    assert (tmp_path / "README.md").read_text() == first
    assert (tmp_path / "o" / "report.json").is_file()
