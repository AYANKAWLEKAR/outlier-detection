from __future__ import annotations

import math
import re
from pathlib import Path

import orjson
import pytest

from agentfw.bench.metrics import GATE_FAMILIES, GateCell, GatePreview, compute_all
from agentfw.bench.replay import PROTOCOL_LABELS, ScoreRow, SplitSpec
from agentfw.bench.report import RunMeta, build_report, render_markdown, update_readme
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
    # wall-clock latency is measured per run, so mask it before comparing the two renderings
    mask = re.compile(r"p50 [\d.]+ us, p99 [\d.]+ us")
    assert mask.sub("", (tmp_path / "README.md").read_text()) == mask.sub("", first)
    assert (tmp_path / "o" / "report.json").is_file()


# --- gate section: one protocol, amended criterion (spec A1.5 to A1.7) ---------------------------

GATE_HEADER = (
    "| Family | Bucket | Detector | Attack recall | Among-the-reasons share "
    "| Top-1 share (original spec §9 criterion) |"
)
SECTION_END = (
    "amended criterion under {label}: {verdict} (one protocol only; this is not the Tier-1 gate "
    "result, see the dual-protocol table)"
)


def _gate(*, passed: bool) -> GatePreview:
    cells = {
        fam: GateCell(bucket, det, 1.0, 10, 10, 0.3 if fam == "velocity_burst" else 1.0, 1.0)
        for fam, (bucket, det) in GATE_FAMILIES.items()
    }
    if not passed:
        cells["exfil_volume"] = GateCell("all", "magnitude", 1.0, 9, 9, 0.5, 8 / 9)
    return GatePreview(
        cells=cells,
        benign_fpr=0.01 if passed else 0.03,
        control_fpr=0.0,
        min_recall=1.0,
        passed=passed,
        missing=(),
        failures=() if passed else ("among:exfil_volume", "benign_fpr"),
    )


def _gate_report_md(gate: GatePreview, outcomes: str) -> str:
    rows = [
        ScoreRow(
            envelope_id=f"e{i}", session_id="s", agent_id="A", tool_id="t", ts_unix_ms=i,
            split="test", fused_score=0.1 * i, per_detector_json="[]",
            label="attack" if i == 1 else "benign", attack_family="prereq_skip" if i == 1 else None,
            attack_id="a" if i == 1 else None, injection_level=None, strength=None, step=None,
            control_type=None, is_warm=True, n_key_obs=100, latency_us=1.0, threshold=0.5,
        )
        for i in (1, 2)
    ]  # fmt: skip
    m = compute_all(rows, warm_only=False, n_boot=0)
    meta = RunMeta(
        git_sha="x",
        seed=0,
        dataset_sha="d",
        detector="engine",
        detector_config={},
        split=SplitSpec(),
        created_unix=0,
    )
    return render_markdown(build_report(m, m, meta, {"outcomes": outcomes}, gate=gate))


def _gate_section(md: str) -> list[str]:
    start = md.index("#### Tier-1 gate preview")
    end = md.index("\n#### ", start + 1)
    return [ln for ln in md[start:end].splitlines()[1:] if ln.strip()]


def test_gate_section_shows_both_shares_and_the_protocol() -> None:
    md = _gate_report_md(_gate(passed=True), "review")
    sec = _gate_section(md)
    assert sec[0] == f"Protocol: {PROTOCOL_LABELS['review']}."
    assert GATE_HEADER in sec
    vel = next(ln for ln in sec if ln.startswith("| velocity_burst |"))
    assert vel == "| velocity_burst | all | velocity | 1.000 (10/10) | 1.000 | 0.300 |"


def test_gate_section_never_prints_bare_gate_pass() -> None:
    for outcomes, passed in [
        ("review", True),
        ("allow", True),
        ("review", False),
        ("allow", False),
    ]:
        md = _gate_report_md(_gate(passed=passed), outcomes)
        sec = _gate_section(md)
        label = PROTOCOL_LABELS[outcomes]  # type: ignore[index]
        assert sec[-1] == SECTION_END.format(label=label, verdict="PASS" if passed else "FAIL")
        assert "gate PASS" not in md
        if not passed:
            assert "Failures: among:exfil_volume, benign_fpr." in sec


def test_run_report_ends_the_gate_section_with_the_amended_line(out_dir: Path) -> None:
    md = (out_dir / "report.md").read_text()
    end = _gate_section(md)[-1]
    label = PROTOCOL_LABELS["review"]  # run's default protocol (BENCH_OUTCOMES)
    assert end in {SECTION_END.format(label=label, verdict=v) for v in ("PASS", "FAIL")}
    assert "gate PASS" not in md
