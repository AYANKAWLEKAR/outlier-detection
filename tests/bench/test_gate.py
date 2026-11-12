"""Task 1.22: the dual-protocol Tier-1 gate table (spec A1.5 to A1.7, D-027, D-028)."""

from __future__ import annotations

import math
from pathlib import Path

import orjson
import pytest

from agentfw.bench import gate as gate_mod
from agentfw.bench.ablation import replay_detector
from agentfw.bench.gate import (
    ENGINE_ROW,
    ORIGINAL_CRITERION_LABEL,
    PRE_A1_ROW,
    gate_matrix,
    load_gate_json,
    main,
    render_gate_matrix,
)
from agentfw.bench.metrics import GATE_FAMILIES, GateCell, GatePreview, gate_preview
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
    assert main(args) == 0  # the pre-A1 row is always computed (spec A1.5 rule 4, ruling A1-17)
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


# --- Task 1.22a: the magnitude cell is never shown without its companion rows (spec A1.5 rule 4,
# ruling A1-17) ---------------------------------------------------------------------------------

_MAG_COL = 1 + list(GATE_FAMILIES).index("magnitude_spike")  # column 0 is the protocol label
_MAG_TEXT = "0.778 (7/9); 0.66; 0.55"  # what the hand-built magnitude cell renders to when shown
_OTHER_TEXT = "1.000 (10/10); 1.00; 0.30"
_COMPANION_HEAD = "## Companion: sensitivity dataset (magnitude_far_mult = 1.0)"
_COMPANION_VERDICT = "companion row (spec A1.5 rule 4), not a gate result"


def _withheld(missing: str) -> str:
    return f"withheld (spec A1.5 rule 4: missing {missing})"


def _preview(*, failing_magnitude: bool = False) -> GatePreview:
    cells = {
        fam: GateCell(bucket, det, 1.0, 10, 10, 0.3, 1.0)
        for fam, (bucket, det) in GATE_FAMILIES.items()
    }
    cells["magnitude_spike"] = GateCell("10-100", "magnitude", 7 / 9, 7, 9, 0.55, 0.66)
    failures = ("recall:magnitude_spike",) if failing_magnitude else ()
    return GatePreview(
        cells=cells, benign_fpr=0.01, control_fpr=0.05, min_recall=7 / 9,
        passed=not failures, failures=failures,
    )  # fmt: skip


def _both(g: GatePreview) -> dict[OutcomeMode, GatePreview]:
    return {"review": g, "allow": g}


def _table_rows(md: str) -> list[list[str]]:
    """Every protocol line of every block, split into its columns."""
    labels = tuple(f"| {PROTOCOL_LABELS[m]} |" for m in ("review", "allow"))
    return [ln.strip().strip("|").split(" | ") for ln in md.splitlines() if ln.startswith(labels)]


def _strip(cols: list[str]) -> list[str]:
    return [c.strip() for c in cols]


def test_magnitude_cell_withheld_without_companions() -> None:
    md = render_gate_matrix({ENGINE_ROW: _both(_preview(failing_magnitude=True))})
    rows = [_strip(r) for r in _table_rows(md)]
    assert len(rows) == 2
    for cols in rows:
        assert cols[_MAG_COL] == _withheld("pre-A1 row and sensitivity row")
        # The other families' cells and the FPRs are printed as before.
        others = [c for i, c in enumerate(cols[1 : 1 + len(GATE_FAMILIES)], 1) if i != _MAG_COL]
        assert others == [_OTHER_TEXT] * (len(GATE_FAMILIES) - 1)
        assert cols[-3:-1] == ["0.0100", "0.050"]
        # The verdict is computed from the data, not from what is displayed.
        assert cols[-1] == "FAIL: recall:magnitude_spike"
    assert _MAG_TEXT not in md and "(7/9)" not in md and "0.778" not in md
    assert _COMPANION_HEAD not in md
    # The output says how to get the magnitude cells.
    assert "--sensitivity-traces" in md


def _withheld_note(md: str) -> str:
    """The paragraph that says why the magnitude cells are withheld and how to get them."""
    return next(ln for ln in md.splitlines() if ln.startswith("Every magnitude_spike cell"))


@pytest.mark.parametrize(
    ("result_rows", "sens_rows", "missing"),
    [
        ((ENGINE_ROW, PRE_A1_ROW), None, "sensitivity row"),
        # The sensitivity companion is the engine row on that dataset, not any sensitivity row.
        ((ENGINE_ROW, PRE_A1_ROW), (PRE_A1_ROW,), "sensitivity row"),
        # The pre-A1 companion is the row named PRE_A1_ROW, not any second row.
        ((ENGINE_ROW, "other engine"), (ENGINE_ROW,), "pre-A1 row"),
    ],
    ids=["pre-A1-present", "sensitivity-lacks-engine-row", "sensitivity-present"],
)
def test_magnitude_cell_withheld_names_the_one_missing_companion(
    result_rows: tuple[str, ...], sens_rows: tuple[str, ...] | None, missing: str
) -> None:
    g = _both(_preview())
    results = dict.fromkeys(result_rows, g)
    sens = None if sens_rows is None else dict.fromkeys(sens_rows, g)
    md = render_gate_matrix(results, sensitivity=sens)
    rows = [_strip(r) for r in _table_rows(md)]
    # Every block, the companion block included, withholds the cell.
    assert len(rows) == 2 * (len(results) + len(sens or {}))
    for cols in rows:
        assert cols[_MAG_COL] == _withheld(missing)
    assert _MAG_TEXT not in md and "(7/9)" not in md
    assert (_COMPANION_HEAD in md) is (sens is not None)
    # The withheld note says how to get what is missing, and only that.
    note = _withheld_note(md)
    assert ("--sensitivity-traces" in note) is (missing == "sensitivity row")
    assert (f"`{PRE_A1_ROW}`" in note) is (missing == "pre-A1 row")


def test_empty_sensitivity_mapping_renders_as_none() -> None:
    # An empty mapping holds no companion row: no empty companion section, same output as None.
    g = _both(_preview())
    results = {ENGINE_ROW: g, PRE_A1_ROW: g}
    md = render_gate_matrix(results, sensitivity={})
    assert md == render_gate_matrix(results)
    assert _COMPANION_HEAD not in md
    assert all(
        cols[_MAG_COL] == _withheld("sensitivity row") for cols in map(_strip, _table_rows(md))
    )


def test_withheld_note_names_both_fixes_when_both_companions_are_missing() -> None:
    note = _withheld_note(render_gate_matrix({ENGINE_ROW: _both(_preview())}))
    assert "missing pre-A1 row and sensitivity row" in note
    assert "--sensitivity-traces" in note and f"`{PRE_A1_ROW}`" in note


def test_magnitude_cell_shown_with_both_companions() -> None:
    g = _both(_preview())
    md = render_gate_matrix(
        {ENGINE_ROW: g, PRE_A1_ROW: g}, sensitivity={ENGINE_ROW: g, PRE_A1_ROW: g}
    )
    assert "withheld" not in md
    rows = [_strip(r) for r in _table_rows(md)]
    assert len(rows) == 8  # two main blocks and two companion blocks, two protocols each
    for cols in rows:
        assert cols[_MAG_COL] == _MAG_TEXT
    assert md.count(_COMPANION_HEAD) == 1
    head = md.index(_COMPANION_HEAD)
    # The companion block follows the main blocks, one table per sensitivity row, same protocol
    # order, and its last column is the companion text, never a verdict.
    assert md.index(f"### {ENGINE_ROW}") < md.index(f"### {PRE_A1_ROW}") < head
    tail = md[head:]
    assert tail.index(f"### {ENGINE_ROW}") < tail.index(f"### {PRE_A1_ROW}")
    assert tail.index(PROTOCOL_LABELS["review"]) < tail.index(PROTOCOL_LABELS["allow"])
    main_rows, comp_rows = rows[:4], rows[4:]
    assert all(cols[-1] == "PASS" for cols in main_rows)
    assert all(cols[-1] == _COMPANION_VERDICT for cols in comp_rows)
    assert "PASS" not in tail and "FAIL" not in tail


def test_sensitivity_row_requires_both_protocols() -> None:
    g = _preview()
    with pytest.raises(ValueError, match="both"):
        render_gate_matrix(
            {ENGINE_ROW: _both(g), PRE_A1_ROW: _both(g)}, sensitivity={ENGINE_ROW: {"review": g}}
        )


def test_cli_always_computes_pre_a1_row(tiny_dataset: Path, tmp_path: Path) -> None:
    md_path, js_path = tmp_path / "gate.md", tmp_path / "gate.json"
    base = ["--traces", str(tiny_dataset), "--out", str(md_path), "--json", str(js_path)]
    for extra in ([], ["--pre-a1"]):  # --pre-a1 is accepted and changes nothing
        assert main([*base, *extra]) == 0
        rep = orjson.loads(js_path.read_bytes())
        assert list(rep) == ["traces", "sensitivity_traces", "labels", "results", "sensitivity"]
        assert list(rep["results"]) == [ENGINE_ROW, PRE_A1_ROW]
        assert rep["sensitivity_traces"] is None and rep["sensitivity"] is None
        md = md_path.read_text()
        assert f"### {PRE_A1_ROW}" in md
        # No sensitivity input: every magnitude cell is withheld and the output names the option.
        rows = [_strip(r) for r in _table_rows(md)]
        assert len(rows) == 4
        assert all(cols[_MAG_COL] == _withheld("sensitivity row") for cols in rows)
        assert "--sensitivity-traces" in md
        # JSON is data: it keeps the magnitude cell's numbers.
        for row in rep["results"].values():
            for g in row.values():
                assert g["cells"]["magnitude_spike"]["n_total"] > 0


def test_cli_sensitivity_traces_shows_magnitude_cells(tiny_dataset: Path, tmp_path: Path) -> None:
    md_path, js_path = tmp_path / "gate.md", tmp_path / "gate.json"
    args = ["--traces", str(tiny_dataset), "--sensitivity-traces", str(tiny_dataset)]
    assert main([*args, "--out", str(md_path), "--json", str(js_path)]) == 0
    rep = orjson.loads(js_path.read_bytes())
    assert rep["sensitivity_traces"] == str(tiny_dataset)
    assert list(rep["sensitivity"]) == [ENGINE_ROW, PRE_A1_ROW]
    # The sensitivity rows are the engine and the pre-A1 engine replayed on that directory (here
    # the same dataset, so they equal the main rows).
    assert rep["sensitivity"] == rep["results"]
    md = md_path.read_text()
    assert "withheld" not in md
    assert _COMPANION_HEAD in md
    rows = [_strip(r) for r in _table_rows(md)]
    assert len(rows) == 8
    eng = rep["results"][ENGINE_ROW]["review"]["cells"]["magnitude_spike"]
    shown = f"({eng['n_detected']}/{eng['n_total']})"
    assert shown in rows[0][_MAG_COL]
    assert all(cols[-1] == _COMPANION_VERDICT for cols in rows[4:])


def _assert_same(a: GatePreview, b: GatePreview) -> None:
    def same(x: object, y: object) -> bool:
        if isinstance(x, float) and isinstance(y, float):
            return (math.isnan(x) and math.isnan(y)) or x == y
        return type(x) is type(y) and x == y

    for f in ("benign_fpr", "control_fpr", "min_recall", "passed", "missing", "failures"):
        assert same(getattr(a, f), getattr(b, f)), f
    assert list(a.cells) == list(b.cells)
    for fam, ca in a.cells.items():
        cb = b.cells[fam]
        assert isinstance(cb, GateCell)
        for f in ca.__slots__:
            assert same(getattr(ca, f), getattr(cb, f)), (fam, f)


def test_load_gate_json_round_trips_nan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    nan = float("nan")
    cells = {
        "magnitude_spike": GateCell("10-100", "magnitude", 0.0, 0, 3, nan, nan),
        "prereq_skip": GateCell("all", "sequence", 0.5, 1, 2, 1.0, 1.0),
    }
    missing = tuple(f for f in GATE_FAMILIES if f not in cells)
    g = GatePreview(
        cells=cells, benign_fpr=0.01, control_fpr=nan, min_recall=0.0, passed=False,
        missing=missing, failures=("recall:magnitude_spike", *(f"missing:{f}" for f in missing)),
    )  # fmt: skip
    seen: list[tuple[Path, EngineConfig]] = []

    def fake_gate_matrix(traces_dir: Path, cfg: EngineConfig) -> dict[OutcomeMode, GatePreview]:
        seen.append((traces_dir, cfg))
        return {"review": g, "allow": g}

    monkeypatch.setattr(gate_mod, "gate_matrix", fake_gate_matrix)
    js = tmp_path / "gate.json"
    main(["--traces", "t", "--json", str(js)])
    results, sens = load_gate_json(js)
    assert sens is None
    assert list(results) == [ENGINE_ROW, PRE_A1_ROW]
    main(["--traces", "t", "--sensitivity-traces", "s", "--json", str(js)])
    results, sens = load_gate_json(js)
    assert sens is not None and list(sens) == [ENGINE_ROW, PRE_A1_ROW]
    # Main and sensitivity rows: the engine and the pre-A1 engine on each directory.
    assert [p for p, _ in seen[2:]] == [Path("t"), Path("t"), Path("s"), Path("s")]
    for loaded in (results, sens):
        for by_mode in loaded.values():
            assert list(by_mode) == ["review", "allow"]
            for back in by_mode.values():
                assert isinstance(back, GatePreview)
                _assert_same(back, g)
                assert math.isnan(back.control_fpr)
                assert math.isnan(back.cells["magnitude_spike"].among_reasons_share)
                assert isinstance(back.missing, tuple) and isinstance(back.failures, tuple)


# --- Task 1.24a: an unknown protocol key is refused (ruling A1-21) -------------------------------


@pytest.fixture(scope="module")
def written_gate_json(tiny_dataset: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One `--json` file with both sections, written once for every case that edits a copy."""
    js = tmp_path_factory.mktemp("gate_json") / "gate.json"
    args = ["--traces", str(tiny_dataset), "--sensitivity-traces", str(tiny_dataset)]
    assert main([*args, "--json", str(js)]) == 0
    return js


@pytest.mark.parametrize("section", ["results", "sensitivity"])
def test_load_gate_json_refuses_an_unknown_protocol(
    written_gate_json: Path, tmp_path: Path, section: str
) -> None:
    load_gate_json(written_gate_json)  # the file as written loads
    payload = orjson.loads(written_gate_json.read_bytes())  # a fresh copy for this case only
    row = payload[section][ENGINE_ROW]
    row["bogus"] = row.pop("allow")  # the same data under a protocol key that does not exist
    bad = tmp_path / "bad.json"
    bad.write_bytes(orjson.dumps(payload))
    with pytest.raises(ValueError, match="unknown replay protocol 'bogus'"):
        load_gate_json(bad)


# --- final review: a public reader of one written gate (ruling A1-25) ---------------------------


def test_preview_from_json_is_public_and_reads_a_written_gate(
    matrix: dict[OutcomeMode, GatePreview],
) -> None:
    assert "preview_from_json" in gate_mod.__all__
    assert not hasattr(gate_mod, "_preview_from")  # one name for the one reader
    for g in matrix.values():
        _assert_same(gate_mod.preview_from_json(orjson.loads(_dump(g))), g)


# --- final review: the companion-source sentence is the caller's (ruling A1-27) ------------------

_COMPANION_PREFIX = (
    "The magnitude_spike cell is shown only with its two companion rows (spec A1.5 rule 4): "
)
#: The CLI's wording, unchanged since Task 1.22a.
_CLI_COMPANION_PARAGRAPH = (
    _COMPANION_PREFIX + f"the engine before Amendment A1 on the same dataset (`{PRE_A1_ROW}`, "
    "always computed by the CLI) and the amended engine on the sensitivity dataset "
    "(`magnitude_far_mult = 1.0`, CLI option `--sensitivity-traces DIR`), printed after the main "
    "blocks."
)


def _companion_paragraph(md: str) -> str:
    return next(ln for ln in md.splitlines() if ln.startswith(_COMPANION_PREFIX))


def test_companion_source_is_the_callers_text() -> None:
    g = _both(_preview())
    rows = {ENGINE_ROW: g, PRE_A1_ROW: g}
    # Without a source the paragraph names the two rows and no caller (it is not CLI-only).
    plain = _companion_paragraph(render_gate_matrix(rows, sensitivity=rows))
    assert "CLI" not in plain and "--sensitivity-traces" not in plain
    assert f"(`{PRE_A1_ROW}`)" in plain and "(`magnitude_far_mult = 1.0`)" in plain
    own = "rows a caller describes, printed after the main blocks."
    md = render_gate_matrix(rows, sensitivity=rows, companion_source=own)
    assert _companion_paragraph(md) == _COMPANION_PREFIX + own
    assert md.count(own) == 1
    # The CLI's own wording is a constant the CLI passes.
    assert _COMPANION_PREFIX + gate_mod.CLI_COMPANION_SOURCE == _CLI_COMPANION_PARAGRAPH


def test_cli_companion_paragraph_is_unchanged(tiny_dataset: Path, tmp_path: Path) -> None:
    md_path = tmp_path / "gate.md"
    assert main(["--traces", str(tiny_dataset), "--out", str(md_path)]) == 0
    assert _companion_paragraph(md_path.read_text()) == _CLI_COMPANION_PARAGRAPH
