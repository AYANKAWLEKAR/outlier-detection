"""Trajectory benchmark (Task 4.1): rows, threshold selection, paired deltas, attribution groups
on a variant, calibrated.json, and the CLI end to end on a small dataset."""

from __future__ import annotations

import dataclasses
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import pytest

from agentfw.bench import session_metrics as SM
from agentfw.bench.attribution import AttributionGroup, group_engine
from agentfw.bench.calibrated import CALIBRATED_PATH, load_calibrated
from agentfw.bench.dataset import load_dir, load_schema_hints
from agentfw.bench.registry import TRAJ_SWITCHES, make_detector
from agentfw.bench.replay import PROTOCOL_LABELS, ScoreRow, replay
from agentfw.bench.trajectory import (
    ROWS,
    SPLIT,
    THETA_GRID,
    TRAJ_FAMILIES,
    TRAJ_GROUPS,
    Vectors,
    _engine,
    bench_config,
    calibration_curve,
    main,
    paired_deltas,
    select_from_curve,
    select_on_calibration,
    vectors,
    write_calibrated,
)
from agentfw.core.config import EngineConfig
from agentfw.core.engine import ALL_DETECTORS, KNOWN_DETECTORS, Engine
from agentfw.synth.generate import generate, sw1_config

_ids = itertools.count()


def _row(session: str, **kw: Any) -> ScoreRow:
    i = next(_ids)
    base: dict[str, Any] = {
        "envelope_id": f"e{i:06d}",
        "session_id": session,
        "agent_id": "A",
        "tool_id": "zendesk.get_ticket",
        "ts_unix_ms": 1_000 + i,
        "split": "calibration",
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
        "latency_us": 10.0,
        "threshold": None,
        "density": 0.0,
    }
    base.update(kw)
    return ScoreRow(**base)


def _sessions(peaks: list[float], **kw: Any) -> list[ScoreRow]:
    """One calibration session per peak density (two rows: 0, then the peak)."""
    out: list[ScoreRow] = []
    for j, p in enumerate(peaks):
        s = f"s{next(_ids)}-{j}"
        out += [_row(s, **kw), _row(s, density=p, **kw)]
    return out


# -- configuration ---------------------------------------------------------------------------------


def test_bench_config_is_sw1_plus_the_trajectory_families() -> None:
    cfg, sw1 = bench_config(1), sw1_config(1)
    assert cfg.families == [*sw1.families, "loop", "spin", "stall", "splice"]
    assert TRAJ_FAMILIES == ("loop", "spin", "stall", "splice")
    assert cfg.model_copy(update={"families": sw1.families}) == sw1
    assert SPLIT.kind == "time" and SPLIT.calibration_frac == 0.15


def test_registry_engine_traj_turns_every_switch_on() -> None:
    det = make_detector("engine-traj", cfg=EngineConfig(z0=2.0))
    assert isinstance(det, Engine)
    assert det.enabled == KNOWN_DETECTORS and det.cfg.z0 == 2.0
    assert {k: getattr(det.cfg, k) for k in TRAJ_SWITCHES} == dict(TRAJ_SWITCHES)
    assert TRAJ_SWITCHES["br_mode"] == "modulate"
    bools = [k for k in TRAJ_SWITCHES if k != "br_mode"]
    assert all(getattr(EngineConfig(), k) is False for k in bools)  # T-03: off by default
    assert EngineConfig().br_mode == "off"


def _changed(eng: Engine) -> dict[str, object]:
    base = dataclasses.asdict(EngineConfig())
    return {k: v for k, v in dataclasses.asdict(eng.cfg).items() if base[k] != v}


@pytest.mark.parametrize("row", [r for r in ROWS if r.switch is not None], ids=lambda r: r.name)
def test_switch_rows_toggle_exactly_one_switch(row: Any) -> None:
    eng = _engine(row, {})
    if row.name == "vital":
        assert eng.enabled == KNOWN_DETECTORS and _changed(eng) == {}
    else:
        assert eng.enabled == ALL_DETECTORS and _changed(eng) == dict(row.overrides)
        assert len(row.overrides) == 1


def test_traj_rows() -> None:
    rows = {r.name: r for r in ROWS}
    traj, shadow = _engine(rows["engine-traj"], {}), _engine(rows["engine-traj-shadow"], {})
    assert _changed(traj) == dict(TRAJ_SWITCHES)
    assert _changed(shadow) == {k: v for k, v in TRAJ_SWITCHES.items() if k != "session_enforce"}
    assert traj.enabled == shadow.enabled == KNOWN_DETECTORS
    assert _engine(rows["engine"], {}).cfg == EngineConfig()


def test_groups_apply_to_the_variant_and_can_drop_vital() -> None:
    groups = {g.name: g for g in TRAJ_GROUPS}
    assert list(groups) == [
        "magnitude",
        "novelty",
        "velocity",
        "sequence",
        "vital",
        "gate",
        "provisional",
    ]
    no_vital = group_engine("engine-traj", EngineConfig(), groups["vital"], {})
    assert no_vital.enabled == ALL_DETECTORS and _changed(no_vital) == dict(TRAJ_SWITCHES)
    no_gate = group_engine("engine-traj", EngineConfig(), groups["gate"], {})
    assert no_gate.enabled == KNOWN_DETECTORS
    assert _changed(no_gate) == {k: v for k, v in TRAJ_SWITCHES.items() if k != "gate"}
    with pytest.raises(ValueError, match="does not enable"):
        group_engine("engine", EngineConfig(), groups["vital"], {})
    plain = group_engine("engine", EngineConfig(), AttributionGroup("x", "novelty"), {})
    assert plain.enabled == ALL_DETECTORS - {"novelty"} and plain.cfg == EngineConfig()


# -- threshold selection ---------------------------------------------------------------------------


def test_select_on_calibration_picks_the_smallest_qualifying_grid_values() -> None:
    # 100 benign sessions: 9 peak at 0.5, 3 more at 0.7, 1 at 1.0 -> share >= theta:
    # 13% up to 0.5, 4% to 0.7, 1% above -> halt 0.75 (<= 1%), warn 0.55 (<= 5%).
    rows = _sessions([0.5] * 9 + [0.7] * 3 + [1.0] + [0.0] * 87)
    curve = dict((t, s) for t, s, _ in calibration_curve(rows))
    assert curve[0.5] == pytest.approx(0.13) and curve[0.55] == pytest.approx(0.04)
    assert select_on_calibration(rows) == (0.55, 0.75)


def test_select_ignores_other_splits_and_non_benign_sessions() -> None:
    noise = _sessions([1.0] * 50, split="test") + _sessions(
        [1.0] * 50, label="attack", attack_family="loop", attack_id="x"
    )
    rows = _sessions([0.0] * 100) + noise
    # nothing reaches any theta: halt at the grid floor, nothing below it for warn (T-54)
    assert select_on_calibration(rows) == (None, 0.2)


def test_select_caps_warn_below_halt_and_falls_back_to_the_top() -> None:
    # every session reaches 0.4 -> both bars first met at 0.45: warn capped strictly below
    assert select_on_calibration(_sessions([0.4] * 50)) == (0.4, 0.45)
    # 10% reach 1.0: no grid value meets the 1% bar -> halt 0.8, warn 0.75
    assert select_on_calibration(_sessions([1.0] * 5 + [0.0] * 45)) == (0.75, 0.8)
    assert THETA_GRID[0] == 0.2 and THETA_GRID[-1] == 0.8 and len(THETA_GRID) == 13


def test_select_needs_benign_calibration_sessions_with_density() -> None:
    assert select_from_curve([(t, 0.0, 10) for t in THETA_GRID]) == (None, 0.2)
    shares = {0.2: 0.06, 0.25: 0.02}  # warn bar first met at 0.25, halt bar at 0.3
    assert select_from_curve([(t, shares.get(t, 0.0), 10) for t in THETA_GRID]) == (
        0.25,
        0.3,
    )
    with pytest.raises(ValueError, match="no benign session"):
        select_on_calibration(_sessions([0.5], split="test"))
    with pytest.raises(ValueError, match="density column is empty"):
        select_on_calibration([_row("s", density=None), _row("s", density=None)])


# -- paired deltas ---------------------------------------------------------------------------------


def _vec(flagged: list[bool], key: str = "k") -> Vectors:
    # 4 sessions x 2 rows: rows 0-5 benign (row 2-3 controls), rows 6-7 one attack each
    return Vectors(
        key=key,
        session=np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64),
        benign=np.array([True] * 6 + [False] * 2),
        control=np.array([False, False, True, True, False, False, False, False]),
        attack=np.array([-1] * 6 + [0, 1], dtype=np.int64),
        attack_family=("fam", "fam"),
        flagged=np.array(flagged),
    )


def test_paired_deltas_point_estimates_and_identity() -> None:
    a = _vec([True, False, True, False, False, False, True, False])
    b = _vec([False] * 6 + [True, True])
    d = paired_deltas(a, b, n_boot=200)
    assert d["benign_fpr"]["a"] == pytest.approx(2 / 6) and d["benign_fpr"]["b"] == 0.0
    assert d["control_fpr"]["delta"] == pytest.approx(0.5)
    assert d["recall:fam"]["delta"] == pytest.approx(-0.5)
    same = paired_deltas(a, a, n_boot=200)
    assert all(v["delta"] == v["lo"] == v["hi"] == 0.0 for v in same.values())
    with pytest.raises(ValueError, match="different rows"):
        paired_deltas(a, _vec([False] * 8, key="other"))


@pytest.fixture(scope="module")
def tiny_traj(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small bench-config world (all personas, the trajectory families) under ``seed-0``."""
    root = tmp_path_factory.mktemp("traj")
    cfg = bench_config(0).model_copy(
        update={
            "days": 10,
            "max_sessions": 400,
            "attack_rate": 0.3,
            "control_rate": 0.05,
            "warmup_sessions": 5,
        }
    )
    generate(cfg, root / "seed-0")
    return root


def test_vectors_match_the_session_metrics(tiny_traj: Path) -> None:
    d = tiny_traj / "seed-0"
    raws, labels, _ = load_dir(d)
    det = make_detector("engine-traj", hints=load_schema_hints(d))
    rows = replay(raws, labels, det, SPLIT, outcomes="allow")
    v = vectors(rows)
    test = [r for r in rows if r.split == "test"]
    base = paired_deltas(v, dataclasses.replace(v, flagged=np.zeros_like(v.flagged)), n_boot=10)
    assert base["benign_fpr"]["a"] == pytest.approx(SM.decision_benign_fpr(test))
    for fam, rec in SM.decision_recall_by_family(test).items():
        assert base[f"recall:{fam}"]["a"] == pytest.approx(rec)
    ctl = SM.decision_control_fpr(test)
    assert ("control_fpr" in base) == (not np.isnan(ctl))


# -- calibrated.json ------------------------------------------------------------------------------


def test_write_calibrated_records_the_selection_off_engine_keys(tmp_path: Path) -> None:
    """The selection goes under keys that are no EngineConfig field (``*_selected_cal``), with
    the caller's note; an undefined warn leaves its key out; a stale file is refused."""
    p = tmp_path / "c.json"
    p.write_bytes(CALIBRATED_PATH.read_bytes())
    before = load_calibrated(p)
    write_calibrated(0.3, 0.55, "test", p, note="not for deployment")
    got = load_calibrated(p)
    assert got.constants == {
        **before.constants,
        "theta_warn_selected_cal": 0.3,
        "theta_halt_selected_cal": 0.55,
    }
    assert got.history[:-1] == before.history
    assert "theta_halt_selected_cal=0.55" in got.history[-1]
    assert got.history[-1].endswith("not for deployment")
    write_calibrated(None, 0.2, "test", p)
    c = load_calibrated(p).constants
    assert c["theta_halt_selected_cal"] == 0.2 and "theta_warn_selected_cal" not in c
    raw = json.loads(p.read_text())
    raw["normalizer_hash"] = "stale"
    p.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="stale"):
        write_calibrated(0.3, 0.55, "test", p)


def test_calibrated_json_records_the_rejected_selection_honestly() -> None:
    """Task 4.1's calibration-partition selection is on record under non-engine keys: theta_halt
    at the grid floor (0.2), warn undefined (nothing below the floor), with the history line
    saying it is not for deployment. No constant named like an EngineConfig field moves: the
    session thresholds stay 0.33 / 0.6 (decision 010, T-54)."""
    c = load_calibrated()
    fields = {f.name for f in dataclasses.fields(EngineConfig)}
    assert {k for k in c.constants if k not in fields} == {"theta_halt_selected_cal"}
    assert c.constants["theta_halt_selected_cal"] == 0.2
    assert all(c.constants[k] == getattr(EngineConfig(), k) for k in c.constants if k in fields)
    # the latest selection line (restamps for unrelated normalizer changes may follow it)
    last = [h for h in c.history if "calibration-partition selection" in h][-1]
    assert "theta_halt_selected_cal=0.2" in last and "theta_warn undefined" in last
    assert "not for deployment" in last and "0.33/0.6" in last
    assert (EngineConfig().theta_warn, EngineConfig().theta_halt) == (0.33, 0.6)


# -- the CLI end to end ---------------------------------------------------------------------------


def test_cli_end_to_end(tiny_traj: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    rows = "engine,provisional,engine-traj,engine-traj-shadow"
    argv = ["--traces", str(tiny_traj), "--out", str(out), "--seeds", "0", "--workers", "2"]
    argv += ["--rows", rows, "--n-boot", "0", "--n-boot-delta", "50"]
    argv += ["--latency-calls", "300", "--latency-warmup", "100"]
    assert main(argv) == 0
    rep = orjson.loads((out / "trajectory_report.json").read_bytes())
    md = (out / "trajectory_report.md").read_text()
    assert set(rep["protocols"]) == {"allow", "review"}
    for mode, p in rep["protocols"].items():
        assert p["label"] == PROTOCOL_LABELS[mode] and p["optimistic"] is (mode == "review")
        assert f"## Protocol: {PROTOCOL_LABELS[mode]}" in md
        block = p["seeds"]["0"]
        assert list(block["rows"]) == rows.split(",")
        assert set(block["deltas"]) == set(rows.split(",")) - {"engine"}
        assert block["rows"]["engine"]["scores_equal_engine"] is True
        att = block["attribution"]
        assert att["t20_totals_by_construction"] is True
        assert list(att["groups"]) == [g.name for g in TRAJ_GROUPS]
        assert att["literal_exact"] is all(g["gained"] == 0 for g in att["groups"].values())
        halt = block["rows"]["engine-traj-shadow"]["sessions"]["halt"]
        assert [x["theta"] for x in halt["sensitivity"]] == list(THETA_GRID)
        assert halt["selected"]["theta_halt"] in THETA_GRID
        assert block["rows"]["engine"]["sessions"]["halt"] is None  # session gate shut
    assert "OPTIMISTIC" in md
    assert set(rep["latency"]["rows"]) == {"engine", "provisional", "engine-traj"}
    assert rep["meta"]["datasets"]["0"]["is_bench_config"] is False
    # --rerender: the saved report through this code's selection rule, no replay
    again = tmp_path / "again"
    assert main(["--rerender", str(out / "trajectory_report.json"), "--out", str(again)]) == 0
    re = orjson.loads((again / "trajectory_report.json").read_bytes())
    assert (
        re["meta"]["rendered_git_sha"]
        and "Re-rendered (no replay)" in (again / "trajectory_report.md").read_text()
    )
    h = re["protocols"]["allow"]["seeds"]["0"]["rows"]["engine-traj-shadow"]["sessions"]["halt"]
    curve = [(c["theta"], c["benign_share"], c["n_benign"]) for c in h["calibration_curve"]]
    warn, halt = select_from_curve(curve)
    assert h["selected"] == {"theta_warn": warn, "theta_halt": halt}


def test_report_sha_marks_a_dirty_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    """T-63 (final review m6): the run's ``meta.git_sha`` is ``_code_sha()`` (``-dirty`` when
    ``agentfw/`` had uncommitted changes), and the rendered header keeps the marker for both the
    run's sha and the re-render's."""
    import subprocess

    from agentfw.bench import trajectory as T
    from agentfw.bench.trajectory_md import _short_sha

    class _Out:
        stdout = " M agentfw/core/engine.py\n"

    monkeypatch.setattr(T, "git_sha", lambda: "0123456789abcdef0123")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Out())
    assert T._code_sha() == "0123456789abcdef0123-dirty"
    _Out.stdout = ""
    assert T._code_sha() == "0123456789abcdef0123"
    assert _short_sha("0123456789abcdef0123-dirty") == "0123456789ab-dirty"
    assert _short_sha("0123456789abcdef0123") == "0123456789ab"
    src = Path(T.__file__).read_text()
    assert '"git_sha": _code_sha()' in src


def test_paired_deltas_with_zero_bootstrap_samples() -> None:
    a = _vec([True, False, True, False, False, False, True, False])
    b = _vec([False] * 6 + [True, True])
    d = paired_deltas(a, b, n_boot=0)
    assert d["benign_fpr"]["delta"] == pytest.approx(2 / 6)
    assert d["benign_fpr"]["lo"] is None and d["benign_fpr"]["hi"] is None


def test_latency_ratios_are_relative_to_the_engine_row(tiny_traj: Path) -> None:
    from agentfw.bench.trajectory import _ROW_BY_NAME, latency_ab

    d = tiny_traj / "seed-0"
    raws, _, _ = load_dir(d)
    rows = [_ROW_BY_NAME["gate"], _ROW_BY_NAME["engine"]]
    lat = latency_ab(raws, load_schema_hints(d), rows, n_calls=200, warmup=50)
    assert lat["rows"]["engine"]["ratio_p50"] == 1.0
    assert lat["rows"]["engine"]["ratio_p99"] == 1.0


def test_calibration_needs_a_benign_session_not_a_benign_row() -> None:
    from agentfw.bench.trajectory import _has_benign_calibration_session

    setup = _row("s", split="calibration", density=0.1)
    atk = dataclasses.replace(setup, label="attack", attack_family="exfil", attack_id="a1")
    assert not _has_benign_calibration_session([setup, atk])
    assert _has_benign_calibration_session([setup])
    assert not _has_benign_calibration_session([dataclasses.replace(setup, split="test")])


def test_write_calibrated_rejects_a_report_from_another_normalizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentfw.bench import trajectory as T
    from agentfw.bench.calibrated import normalizer_hash

    called: list[object] = []
    monkeypatch.setattr(T, "write_calibrated", lambda *a, **k: called.append(a))
    monkeypatch.setattr(T, "selected_thetas", lambda r: (None, 0.2, "x"))
    for meta in ({}, {"normalizer_hash": "stale"}):
        with pytest.raises(ValueError, match="normalizer"):
            T._calibrate({"meta": meta}, "")
    T._calibrate({"meta": {"normalizer_hash": normalizer_hash()}}, "")
    assert len(called) == 1


# -- CodeRabbit PR #1 ------------------------------------------------------------------------------


def test_session_halts_use_the_configs_theta_halt() -> None:
    """The halt summary evaluates (and records) the replay's own ``theta_halt``, not the default:
    a benign session peaking at 0.7 does not halt at theta_halt 0.8 (T-68)."""
    from agentfw.bench.trajectory import _sessions as sessions_summary

    rows = _sessions([0.7] + [0.0] * 9)
    got = sessions_summary(rows, False, EngineConfig(theta_halt=0.8))["halt"]
    assert got["theta_halt"] == 0.8
    assert got["density_policy"]["false_halt_rate"] == 0.0
    dflt = sessions_summary(rows, False, EngineConfig())["halt"]
    assert dflt["theta_halt"] == 0.6
    assert dflt["density_policy"]["false_halt_rate"] == pytest.approx(0.1)


def test_render_markdown_reads_the_stored_theta_halt() -> None:
    from agentfw.bench.trajectory_md import _halt_theta

    assert _halt_theta({"a": {"theta_halt": 0.8}, "b": {"theta_halt": 0.8}}) == "θ_halt = 0.8"
    assert _halt_theta({"a": {"theta_halt": 0.6}, "b": {"theta_halt": 0.8}}) == "each row's θ_halt"


def test_write_calibrated_marks_unmet_bounds(tmp_path: Path) -> None:
    """A grid-top fallback is not recorded as meeting the bounds (T-69)."""
    p = tmp_path / "c.json"
    p.write_bytes(CALIBRATED_PATH.read_bytes())
    unmet = calibration_curve(_sessions([1.0] * 5 + [0.0] * 45))  # 10% at every theta
    warn, halt = select_from_curve(unmet)
    write_calibrated(warn, halt, "test", p, curve=unmet)
    line = load_calibrated(p).history[-1]
    assert "false-halt 10.0% (bound <= 1%: unmet)" in line
    assert "warn rate 10.0% (bound <= 5%: unmet)" in line
    met = calibration_curve(_sessions([0.5] * 9 + [0.7] * 3 + [1.0] + [0.0] * 87))
    w2, h2 = select_from_curve(met)
    write_calibrated(w2, h2, "test", p, curve=met)
    line = load_calibrated(p).history[-1]
    assert "false-halt 1.0% (bound <= 1%: met)" in line
    assert "warn rate 4.0% (bound <= 5%: met)" in line
    write_calibrated(0.3, 0.55, "test", p)
    assert "bounds not checked" in load_calibrated(p).history[-1]
