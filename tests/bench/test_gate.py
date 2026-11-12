"""Task 1.22: the dual-protocol Tier-1 gate table (spec A1.5 to A1.7, D-027, D-028)."""

from __future__ import annotations

import math
from pathlib import Path

import orjson
import pytest

from agentfw.bench.ablation import replay_detector
from agentfw.bench.gate import ORIGINAL_CRITERION_LABEL, gate_matrix, main, render_gate_matrix
from agentfw.bench.metrics import GATE_FAMILIES, GatePreview, gate_preview
from agentfw.bench.replay import PROTOCOL_LABELS, OutcomeMode, SplitSpec
from agentfw.bench.sweeps import PRE_A1_SWITCHES
from agentfw.core.config import EngineConfig


def _direct(traces: Path, cfg: EngineConfig, mode: OutcomeMode) -> GatePreview:
    rows, _ = replay_detector(traces, "engine", SplitSpec(), cfg, mode)
    return gate_preview(rows)


def _dump(g: GatePreview) -> bytes:
    return orjson.dumps(g)


@pytest.fixture(scope="module")
def matrix(tiny_dataset: Path) -> dict[OutcomeMode, GatePreview]:
    return gate_matrix(tiny_dataset)


def test_gate_matrix_runs_both_protocols(
    tiny_dataset: Path, matrix: dict[OutcomeMode, GatePreview]
) -> None:
    assert list(matrix) == ["review", "allow"]
    with pytest.raises(ValueError, match="both"):
        gate_matrix(tiny_dataset, protocols=("review",))
    with pytest.raises(ValueError, match="both"):
        gate_matrix(tiny_dataset, protocols=("allow", "enforce"))
    # Each entry is gate_preview of the engine replayed under that protocol, same config: every
    # field, floats included, is equal (_dump writes NaN as null, so NaN == NaN here).
    for mode in ("review", "allow"):
        assert _dump(matrix[mode]) == _dump(_direct(tiny_dataset, EngineConfig(), mode))
    # The two protocols give different results on the tiny dataset, so the comparison above
    # would catch a gate_matrix that replayed one protocol twice.
    assert _dump(matrix["review"]) != _dump(matrix["allow"])


def test_render_gate_matrix_labels_verbatim(matrix: dict[OutcomeMode, GatePreview]) -> None:
    md = render_gate_matrix({"engine": matrix})
    assert "spec §5.2 allow-all" in md
    assert (
        "oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic)" in md
    )
    assert "original spec §9 criterion" in md and "among" in md.lower()
    assert ORIGINAL_CRITERION_LABEL == "original spec §9 criterion"
    assert md.index(PROTOCOL_LABELS["review"]) < md.index(PROTOCOL_LABELS["allow"])
    assert "gate PASS" not in md and "Tier-1 gate met" not in md
    # One line per protocol, each with its verdict and its failures (the tiny dataset lacks
    # three gated families, so both protocols fail and name them).
    for mode in ("review", "allow"):
        line = next(ln for ln in md.splitlines() if ln.startswith(f"| {PROTOCOL_LABELS[mode]} |"))
        assert "FAIL" in line and "missing:velocity_burst" in line
    for fam in GATE_FAMILIES:
        assert fam in md


def test_render_gate_matrix_cell_format() -> None:
    from agentfw.bench.metrics import GateCell

    cells = {
        fam: GateCell(bucket, det, 1.0, 10, 10, 0.3, 1.0)
        for fam, (bucket, det) in GATE_FAMILIES.items()
    }
    ok = GatePreview(
        cells=cells, benign_fpr=0.01, control_fpr=0.05, min_recall=1.0, passed=True, failures=()
    )
    md = render_gate_matrix({"engine": {"review": ok, "allow": ok}})
    assert "1.000 (10/10); 1.00; 0.30" in md
    review = next(ln for ln in md.splitlines() if ln.startswith(f"| {PROTOCOL_LABELS['review']} |"))
    assert review.endswith("| 0.0100 | 0.050 | PASS |")
    assert "### engine" in md


def test_gate_cli_writes_md_and_json(tiny_dataset: Path, tmp_path: Path) -> None:
    md_path, js_path = tmp_path / "gate.md", tmp_path / "gate.json"
    args = ["--traces", str(tiny_dataset), "--out", str(md_path), "--json", str(js_path)]
    assert main(args) == 0
    one = orjson.loads(js_path.read_bytes())
    assert list(one["results"]) == ["engine"]
    assert main([*args, "--pre-a1"]) == 0  # --pre-a1 adds a second engine row
    rep = orjson.loads(js_path.read_bytes())
    assert len(rep["results"]) == 2
    pre = list(rep["results"])[1]
    assert "pre-A1" in pre
    for row in rep["results"].values():
        assert list(row) == ["review", "allow"]
        for g in row.values():
            assert set(g) >= {"cells", "benign_fpr", "control_fpr", "passed", "failures"}
    assert rep["labels"] == {m: PROTOCOL_LABELS[m] for m in ("review", "allow")}
    # The pre-A1 row is the engine replayed with PRE_A1_SWITCHES under each protocol (so the CLI
    # and gate_matrix really use that config), and it differs from the default engine row.
    pre_cfg = EngineConfig().replace(**PRE_A1_SWITCHES)
    for mode in ("review", "allow"):
        assert rep["results"][pre][mode] == orjson.loads(
            _dump(_direct(tiny_dataset, pre_cfg, mode))
        )
    assert rep["results"][pre] != rep["results"]["engine"]
    md = md_path.read_text()
    assert f"### {pre}" in md and "### engine" in md
    assert PROTOCOL_LABELS["allow"] in md and PROTOCOL_LABELS["review"] in md


def test_render_gate_matrix_never_shows_one_protocol_alone() -> None:
    ok = GatePreview(
        cells={}, benign_fpr=0.0, control_fpr=0.0, min_recall=math.nan, passed=False,
        missing=tuple(GATE_FAMILIES), failures=tuple(f"missing:{f}" for f in GATE_FAMILIES),
    )  # fmt: skip
    with pytest.raises(ValueError, match="both"):
        render_gate_matrix({"engine": {"review": ok}})
    md = render_gate_matrix({"engine": {"allow": ok, "review": ok}})  # input order does not matter
    assert md.index(PROTOCOL_LABELS["review"]) < md.index(PROTOCOL_LABELS["allow"])
