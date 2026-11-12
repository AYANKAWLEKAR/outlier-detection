"""Task 1.14: ablations, parameter sweeps, quantile tail error and the new report sections."""

from __future__ import annotations

import importlib.util
import math
import re
from pathlib import Path

import numpy as np
import pytest

from agentfw.bench.ablation import AblationRow, run_ablation
from agentfw.bench.metrics import GATE_FAMILIES, Metrics, compute_all, gate_preview
from agentfw.bench.replay import ScoreRow, SplitSpec
from agentfw.bench.report import RunMeta, build_report, render_markdown
from agentfw.bench.sweeps import SweepRow, grid_points, quantile_tail_error, sweep
from agentfw.core.engine import DETECTOR_ORDER

ABLATION_VARIANTS = ["engine", *(f"engine-no-{d}" for d in DETECTOR_ORDER)]


@pytest.fixture(scope="module")
def ablation(tiny_dataset: Path) -> list[AblationRow]:
    return run_ablation(tiny_dataset, ABLATION_VARIANTS, SplitSpec(), outcomes="allow")


def test_ablation_returns_one_row_per_variant_in_order(ablation: list[AblationRow]) -> None:
    assert [r.variant for r in ablation] == ABLATION_VARIANTS
    for r in ablation:
        assert 0.0 <= r.auprc <= 1.0
        assert r.state_bytes > 0
        # A per-agent FPR is only meaningful with its achieved recall.
        assert not math.isnan(r.per_agent_recall)
        assert set(r.per_family_recall) == {"magnitude_spike", "prereq_skip", "novel_target"}


# Task 1.14 found removing magnitude RAISED warm AUPRC on tiny_dataset (a strict xfail). With the
# Amendment A1 scoring as the default the property holds for every single-detector ablation
# (Task 1.21: engine 0.608 >= engine-no-magnitude 0.522) -- see
# docs/decisions/007-ablation-dominance.md, addendum 2026-11-11.
@pytest.mark.parametrize("variant", ABLATION_VARIANTS[1:])
def test_full_engine_auprc_at_least_every_single_detector_ablation(
    ablation: list[AblationRow], variant: str
) -> None:
    by = {r.variant: r for r in ablation}
    assert by["engine"].auprc >= by[variant].auprc - 1e-12


def test_ablation_rejects_unknown_variant(tiny_dataset: Path) -> None:
    with pytest.raises(KeyError):
        run_ablation(tiny_dataset, ["engine-no-such"], SplitSpec(), outcomes="allow")


def test_grid_points_is_the_cartesian_product_in_order() -> None:
    pts = grid_points({"z0": [2.5, 3.0], "tau": [0.5, 0.75, 1.0]})
    assert pts == [{"z0": z, "tau": t} for z in (2.5, 3.0) for t in (0.5, 0.75, 1.0)]
    assert grid_points({}) == [{}]


def test_sweep_returns_full_cartesian_grid_and_applies_params(tiny_dataset: Path) -> None:
    grid = {"z0": [2.5, 3.5], "cal_alpha": [0.005, 0.01]}
    rows = sweep(tiny_dataset, grid, SplitSpec(), outcomes="allow")
    assert all(isinstance(r, SweepRow) for r in rows)
    assert [r.params for r in rows] == grid_points(grid)
    for r in rows:
        for key in ("auprc", "fpr95_per_agent", "per_agent_recall", "benign_fpr", "control_fpr"):
            assert key in r.metrics
        for fam in ("magnitude_spike", "prereq_skip", "novel_target"):
            assert f"recall:{fam}" in r.metrics and f"share:{fam}" in r.metrics
    # The parameter really reaches the engine: a lower z0 scores magnitude higher, so AUPRC moves.
    assert len({round(r.metrics["auprc"], 12) for r in rows}) > 1


def test_sweep_rejects_unknown_config_field(tiny_dataset: Path) -> None:
    with pytest.raises(ValueError, match="not_a_field"):
        sweep(tiny_dataset, {"not_a_field": [1]}, SplitSpec(), outcomes="allow")


def test_quantile_tail_error_is_zero_for_exact() -> None:
    samples = np.random.default_rng(0).lognormal(0.0, 1.0, 5000)
    err = quantile_tail_error(samples, (0.05, 0.95), "exact")
    assert set(err) == {0.05, 0.95}
    assert all(abs(v) < 1e-12 for v in err.values())


def test_quantile_tail_error_p2_is_small_but_measured() -> None:
    samples = np.log1p(np.random.default_rng(1).lognormal(4.0, 0.8, 20_000))
    err = quantile_tail_error(samples, (0.05, 0.25, 0.5, 0.75, 0.95), "p2")
    iqr = float(np.quantile(samples, 0.75) - np.quantile(samples, 0.25))
    assert set(err) == {0.05, 0.25, 0.5, 0.75, 0.95}
    assert all(0.0 <= v < 0.1 * iqr for v in err.values())


def test_quantile_tail_error_tdigest_only_if_installed() -> None:
    samples = np.arange(100.0)
    if importlib.util.find_spec("tdigest") is None:
        with pytest.raises(ImportError):
            quantile_tail_error(samples, (0.05,), "tdigest")
    else:  # pragma: no cover - optional dependency
        assert set(quantile_tail_error(samples, (0.05,), "tdigest")) == {0.05}


# --- gate preview ---------------------------------------------------------------------------------


def _row(i: int, **kw: object) -> ScoreRow:
    base: dict[str, object] = {
        "envelope_id": f"e{i}",
        "session_id": f"s{i}",
        "agent_id": "A",
        "tool_id": "t",
        "ts_unix_ms": i,
        "split": "test",
        "fused_score": 0.0,
        "per_detector_json": "[]",
        "label": "benign",
        "attack_family": None,
        "attack_id": None,
        "injection_level": None,
        "strength": None,
        "step": None,
        "control_type": None,
        "is_warm": True,
        "n_key_obs": 100,
        "latency_us": 1.0,
        "threshold": 0.5,
    }
    base.update(kw)
    return ScoreRow(**base)  # type: ignore[arg-type]


def _mag_json(score: float) -> str:
    return f'[{{"detector":"magnitude","field":"amount","score":{score},"evidence":{{}}}}]'


def test_gate_preview_uses_gate_bucket_warm_test_rows_and_live_thresholds() -> None:
    rows = [
        # magnitude_spike m=20 (gate bucket 10-100): detected, magnitude on top.
        _row(1, label="attack", attack_family="magnitude_spike", attack_id="a1", strength=20.0,
             fused_score=0.9, per_detector_json=_mag_json(0.9)),
        # m=20 again but missed at its live threshold.
        _row(2, label="attack", attack_family="magnitude_spike", attack_id="a2", strength=20.0,
             fused_score=0.4, per_detector_json=_mag_json(0.4)),
        # m=5 (bucket 3-10): outside the gate bucket, must not count.
        _row(3, label="attack", attack_family="magnitude_spike", attack_id="a3", strength=5.0,
             fused_score=0.0),
        # cold attack row: excluded (gate is on warm rows).
        _row(4, label="attack", attack_family="magnitude_spike", attack_id="a4", strength=50.0,
             fused_score=0.0, is_warm=False),
        # train-split attack: excluded.
        _row(5, label="attack", attack_family="magnitude_spike", attack_id="a5", strength=50.0,
             fused_score=0.0, split="train"),
        _row(6, fused_score=0.6),  # benign false positive at live threshold 0.5
        _row(7, fused_score=0.1, control_type="benign_unusual"),
        _row(8, fused_score=0.1),
    ]  # fmt: skip
    g = gate_preview(rows)
    cell = g.cells["magnitude_spike"]
    assert (cell.bucket, cell.detector) == GATE_FAMILIES["magnitude_spike"]
    assert (cell.n_detected, cell.n_total) == (1, 2)
    assert cell.recall == pytest.approx(0.5)
    assert cell.top_reason_share == pytest.approx(1.0)
    assert set(g.cells) == {"magnitude_spike"}  # absent families get no cell ...
    assert "prereq_skip" in g.missing  # ... and are listed as missing
    assert g.benign_fpr == pytest.approx(1 / 3)
    assert g.control_fpr == pytest.approx(0.0)
    assert g.min_recall == pytest.approx(0.5)
    assert g.passed is False


def test_gate_preview_excludes_slow_drift() -> None:
    assert "slow_drift" not in GATE_FAMILIES
    rows = [
        _row(1, label="attack", attack_family="slow_drift", attack_id="d", strength=0.08,
             fused_score=0.0, step=1),
        _row(2, label="attack", attack_family="prereq_skip", attack_id="p", fused_score=0.9),
    ]  # fmt: skip
    g = gate_preview(rows)
    assert set(g.cells) == {"prereq_skip"}
    assert g.cells["prereq_skip"].recall == pytest.approx(1.0)
    # Every other gated family is absent, so the gate cannot pass.
    assert set(g.missing) == set(GATE_FAMILIES) - {"prereq_skip"}
    assert g.passed is False


def test_gate_preview_passes_only_with_every_gated_family_present() -> None:
    rows = [
        _row(i, label="attack", attack_family=fam, attack_id=fam, fused_score=0.9,
             strength=20.0 if fam == "magnitude_spike" else None)
        for i, fam in enumerate(GATE_FAMILIES)
    ]  # fmt: skip
    g = gate_preview(rows)
    assert g.missing == ()
    assert g.passed is True


# --- report sections ------------------------------------------------------------------------------


def _metrics() -> Metrics:
    rows = [
        _row(1, label="attack", attack_family="prereq_skip", attack_id="p", fused_score=0.9),
        _row(2, fused_score=0.1),
        _row(3, fused_score=0.2, split="train"),
    ]
    return compute_all(rows, warm_only=False, n_boot=0)


def test_render_markdown_has_ablation_cold_start_drift_and_gate_sections() -> None:
    m = _metrics()
    meta = RunMeta(
        git_sha="x",
        seed=0,
        dataset_sha="d",
        detector="engine",
        detector_config={},
        split=SplitSpec(),
        created_unix=0,
    )
    abl = [
        AblationRow("engine", 0.81, 0.012, 0.95, {"prereq_skip": 1.0}, 1000),
        AblationRow("engine-no-sequence", 0.42, math.nan, 0.80, {"prereq_skip": 0.2}, 900),
    ]
    gate = gate_preview([_row(1, label="attack", attack_family="prereq_skip", fused_score=0.9)])
    assert gate.passed is False
    rep = build_report(m, m, meta, ablations=abl, gate=gate)
    md = render_markdown(rep)
    for heading in ("Ablations", "Cold start", "Drift", "Tier-1 gate preview"):
        assert re.search(rf"^#+ {heading}$", md, re.M), heading
    assert "engine-no-sequence" in md
    assert "per-agent recall" in md.lower()  # FPR is always shown with its achieved recall
    assert rep["ablations"][1]["fpr95_per_agent"] is None  # type: ignore[index]
    assert rep["gate"]["passed"] is False  # type: ignore[index]
    assert "Missing gated families" in md


def test_render_markdown_without_ablations_says_so() -> None:
    m = _metrics()
    meta = RunMeta(
        git_sha="x",
        seed=0,
        dataset_sha="d",
        detector="static_threshold",
        detector_config={},
        split=SplitSpec(),
        created_unix=0,
    )
    md = render_markdown(build_report(m, m, meta))
    assert re.search(r"^#+ Ablations$", md, re.M)
    assert "No ablations" in md


# --- run(..., ablations=) -------------------------------------------------------------------------


def test_run_with_ablations_feeds_the_report(tiny_dataset: Path, tmp_path: Path) -> None:
    import orjson

    from agentfw.bench.run import main

    variants = ["engine", "engine-no-magnitude"]
    main(
        [
            *("--traces", str(tiny_dataset), "--detector", "engine", "--out", str(tmp_path)),
            *("--n-boot", "0", "--ablations", ",".join(variants)),
        ]
    )
    rep = orjson.loads((tmp_path / "report.json").read_bytes())
    assert [a["variant"] for a in rep["ablations"]] == variants
    md = (tmp_path / "report.md").read_text()
    assert "engine-no-magnitude" in md and "No ablations" not in md


def test_run_without_ablations_keeps_none(tiny_dataset: Path, tmp_path: Path) -> None:
    import orjson

    from agentfw.bench.run import run

    run(tiny_dataset, "random", SplitSpec(), tmp_path, n_boot=0)
    assert orjson.loads((tmp_path / "report.json").read_bytes())["ablations"] is None


# --- Task 1.15: gate candidates (opt-in outcome modes and censoring; decision 008) --------------


def test_replay_detector_passes_the_outcome_mode(tiny_dataset: Path) -> None:
    from agentfw.bench.ablation import replay_detector

    allow, _ = replay_detector(tiny_dataset, "engine", SplitSpec(), None, outcomes="allow")
    oracle, _ = replay_detector(tiny_dataset, "engine", SplitSpec(), None, outcomes="oracle")
    assert [r.envelope_id for r in allow] == [r.envelope_id for r in oracle]
    # attacks never learned: later scores (and thresholds) differ from the allow-all replay
    assert [r.fused_score for r in allow] != [r.fused_score for r in oracle]


def test_gate_suite_runs_every_candidate_with_its_outcome_mode(
    tiny_dataset: Path, tmp_path: Path
) -> None:
    import orjson

    from agentfw.bench.sweeps import GATE_CANDIDATES, main

    assert main(["--traces", str(tiny_dataset), "--out", str(tmp_path), "--suites", "gate"]) == 0
    rep = orjson.loads((tmp_path / "sweeps.json").read_bytes())
    entries = rep["suites"]["gate"]
    assert [e["setting"] for e in entries] == [c[0] for c in GATE_CANDIDATES]
    assert [e["outcomes"] for e in entries] == [c[2] for c in GATE_CANDIDATES]
    assert {"allow", "enforce", "oracle"} <= {e["outcomes"] for e in entries}
    assert all(e["trajectory"] for e in entries)
    md = (tmp_path / "sweeps.md").read_text()
    assert "## gate" in md and "| setting | outcomes |" in md and "ratcheting" in md
    assert "labels act as a perfect human reviewer" in md
    from agentfw.bench.replay import PROTOCOL_LABELS
    from agentfw.bench.sweeps import REVIEW_NOTE

    assert PROTOCOL_LABELS["review"] in REVIEW_NOTE and "->" not in REVIEW_NOTE


# --- user decision 2026-11-10 (decision 008): every bench tool states its protocol ---------------


def test_bench_tools_require_an_explicit_protocol(tiny_dataset: Path) -> None:
    from agentfw.bench.ablation import replay_detector

    with pytest.raises(TypeError, match="outcomes"):
        run_ablation(tiny_dataset, ["engine"], SplitSpec())  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="outcomes"):
        sweep(tiny_dataset, {"z0": [3.0]}, SplitSpec())  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="outcomes"):
        replay_detector(tiny_dataset, "engine", SplitSpec(), None)  # type: ignore[call-arg]


def test_gate_candidates_pin_the_spec_switches() -> None:
    """The 008 evidence suite keeps its meaning: "default" is the spec engine, allow-all."""
    from agentfw.bench.sweeps import GATE_CANDIDATES, PRE_A1_SWITCHES, REVIEW_NOTE, SPEC_SWITCHES
    from agentfw.core.config import EngineConfig

    assert PRE_A1_SWITCHES == {
        "vel_count_stat": "logz",
        "mag_score": "z",
        "mag_bool_numeric": False,
    }
    assert SPEC_SWITCHES == {
        "vel_count_attempts": False,
        "seq_prereq_graded": False,
        "novelty_full_n": 200,
        **PRE_A1_SWITCHES,
    }
    by_name = {n: (p, o) for n, p, o in GATE_CANDIDATES}
    assert by_name["default"] == (SPEC_SWITCHES, "allow")
    assert all(set(SPEC_SWITCHES) <= set(p) for p, _ in by_name.values())
    rec_name = "oracle-reviewer, vel attempts, graded prereq, novelty_full_n=50"
    rec, mode = by_name[rec_name]
    # 008's recommended candidate is the engine before Amendment A1, not today's default.
    assert EngineConfig().replace(**rec) == EngineConfig().replace(**PRE_A1_SWITCHES)
    assert mode == "review"
    assert EngineConfig().replace(**PRE_A1_SWITCHES) != EngineConfig()
    assert f'"{rec_name}"' in REVIEW_NOTE and "before Amendment A1" in REVIEW_NOTE


def test_z0tau_note_is_printed_in_sweeps_md(tiny_dataset: Path, tmp_path: Path) -> None:
    """At the defaults z0 / tau move velocity only (A1.2): sweeps.md says so under z0tau."""
    from agentfw.bench.sweeps import Z0TAU_NOTE, main

    assert "velocity only" in Z0TAU_NOTE and "A1.2" in Z0TAU_NOTE
    args = ["--traces", str(tiny_dataset), "--out", str(tmp_path), "--suites", "z0tau"]
    assert main([*args, "--outcomes", "allow"]) == 0
    md = (tmp_path / "sweeps.md").read_text()
    assert Z0TAU_NOTE in md.split("## z0tau", 1)[1]


def test_sweeps_cli_records_the_protocol_of_the_grid_suites(
    tiny_dataset: Path, tmp_path: Path
) -> None:
    import orjson

    from agentfw.bench.sweeps import main

    base = ["--traces", str(tiny_dataset), "--suites", "saturation"]
    assert main([*base, "--out", str(tmp_path / "d")]) == 0
    assert main([*base, "--out", str(tmp_path / "a"), "--outcomes", "allow"]) == 0
    for d, mode in (("d", "review"), ("a", "allow")):
        rep = orjson.loads((tmp_path / d / "sweeps.json").read_bytes())
        assert rep["outcomes"] == mode
        assert [e["outcomes"] for e in rep["suites"]["saturation"]] == [mode]
    md = (tmp_path / "d" / "sweeps.md").read_text()
    assert "labels act as a perfect human reviewer" in md.split("## ")[0]
