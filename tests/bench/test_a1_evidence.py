"""Amendment A1's gate evidence (plan Task 1.23): the matrix on seeds 0-2 under both protocols.

``docs/decisions/evidence/a1-gate/run.sh`` generates the four datasets, replays each once with
``agentfw.bench.gate`` and writes ``matrix.json`` / ``matrix.md``. This test pins that the matrix
holds every (dataset, seed, engine, protocol) row, the magnitude census of each benchmark seed and
both protocol labels verbatim (spec A1.5 rule 1) beside the original spec §9 criterion.

Fix round 1 of Task 1.23 adds: ``run.sh`` stops when a dataset generation fails and runs stage
``seed2`` only when every seed-0/1 cell is written; ``measure.py merge`` refuses an incomplete set
of cells; ``matrix.md`` prints the amended engine's control FPR with two decimals and names any
value above the 5 % Tier-3 bound (A1.12 item 7), labels the sensitivity companion as seed 0, and
shows seed 2's magnitude cell in the comparison table with its pre-A1 companion (A1.5 rule 4).
Importing ``measure.py`` here also puts its imports (private ``agentfw.bench.metrics`` helpers
among them) under the gate.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

import orjson
import pytest

from agentfw.bench.replay import PROTOCOL_LABELS

EVIDENCE = Path(__file__).resolve().parents[2] / "docs/decisions/evidence/a1-gate"
SEEDS01 = ("sw1-seed0", "sw1-seed1", "sw1-sens-seed0")
CELLS = ("a1--review", "a1--allow", "pre-a1--review", "pre-a1--allow")


def test_a1_matrix_is_complete() -> None:
    m = orjson.loads((EVIDENCE / "matrix.json").read_bytes())
    have = {(r["dataset"], r["seed"], r["engine"], r["protocol"]) for r in m["rows"]}
    want = {
        ("sw1", s, e, p) for s in (0, 1, 2) for e in ("a1", "pre-a1") for p in ("review", "allow")
    }
    want |= {("sens", 0, e, p) for e in ("a1", "pre-a1") for p in ("review", "allow")}
    assert have == want
    for s in ("0", "1", "2"):
        assert m["census"][s]["gated_test_n"] >= 50 and m["counts"][s]["magnitude_dropped"] >= 0
    md = (EVIDENCE / "matrix.md").read_text()
    assert (
        PROTOCOL_LABELS["allow"] in md
        and PROTOCOL_LABELS["review"] in md
        and "original spec §9 criterion" in md
    )


# --- run.sh ------------------------------------------------------------------------------------

_FAKE_UV = """#!/usr/bin/env bash
echo "$*" >> "$FAKE_UV_LOG"
case "$*" in *"gen ${FAKE_UV_FAIL:-none}") sleep 0.2; exit 3;; esac
exit 0
"""


def _run_sh(tmp_path: Path, stage: str, fail: str | None = None) -> tuple[int, list[str]]:
    """``run.sh STAGE`` in a scratch repo whose ``uv`` only logs its arguments (and fails the
    generation named by ``fail``). Returns the exit status and the logged ``measure.py`` calls."""
    here = tmp_path / "docs/decisions/evidence/a1-gate"
    here.mkdir(parents=True, exist_ok=True)
    shutil.copy(EVIDENCE / "run.sh", here / "run.sh")
    shutil.copy(EVIDENCE / "matrix.json", here / "matrix.json")  # committed: always present
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "uv").write_text(_FAKE_UV)
    (bin_dir / "uv").chmod(0o755)
    log = tmp_path / "uv.log"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "FAKE_UV_LOG": str(log),
        "FAKE_UV_FAIL": fail or "none",
    }
    proc = subprocess.run(
        ["bash", str(here / "run.sh"), stage], env=env, capture_output=True, text=True
    )
    calls = log.read_text().splitlines() if log.exists() else []
    return proc.returncode, [c.split("measure.py ", 1)[1] for c in calls]


@pytest.mark.parametrize(
    "fail", ["0 traces/sw1-seed0", "1 traces/sw1-seed1", "0 traces/sw1-sens-seed0 1.0"]
)
def test_run_sh_stops_when_any_generation_fails(tmp_path: Path, fail: str) -> None:
    status, calls = _run_sh(tmp_path, "seeds01", fail)
    assert status != 0
    assert [c for c in calls if not c.startswith("gen ")] == []  # no replay, no merge
    assert len([c for c in calls if c.startswith("gen ")]) == 3  # all three ran


def test_run_sh_seed2_needs_every_seed01_cell(tmp_path: Path) -> None:
    status, calls = _run_sh(tmp_path, "seed2")  # matrix.json exists, no cell was replayed
    assert status != 0 and calls == []
    cells = tmp_path / "results/a1-gate"
    cells.mkdir(parents=True, exist_ok=True)  # run.sh made it
    names = [f"{d}--{c}.json" for d in SEEDS01 for c in CELLS]
    for n in names[:-1]:
        (cells / n).write_text("{}")
    status, calls = _run_sh(tmp_path, "seed2")  # one seed-0/1 cell missing
    assert status != 0 and calls == []
    (cells / names[-1]).write_text("{}")
    status, calls = _run_sh(tmp_path, "seed2")
    assert status == 0 and calls[0] == "gen 2 traces/sw1-seed2" and calls[-1].startswith("merge")


# --- measure.py --------------------------------------------------------------------------------


def _measure() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_evidence_a1_measure", EVIDENCE / "measure.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # importing must not run the script
    return mod


@pytest.mark.parametrize(
    "present",
    [
        [],  # nothing replayed: would overwrite the matrix with no rows
        [f"sw1-seed2--{c}.json" for c in CELLS],  # seed 2 alone (a fresh checkout)
        [f"{d}--{c}.json" for d in SEEDS01 for c in CELLS] + ["sw1-seed2--a1--review.json"],
    ],
)
def test_merge_refuses_an_incomplete_set_of_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, present: list[str]
) -> None:
    mod = _measure()
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(mod, "HERE", out)  # never the committed evidence
    cells = tmp_path / "cells"
    cells.mkdir()
    for n in present:
        (cells / n).write_text("{}")
    with pytest.raises(SystemExit, match="missing"):
        mod.merge(cells)
    assert list(out.iterdir()) == []


def _a1_review_over_bound(m: dict[str, Any]) -> list[float]:
    return [
        r["control_fpr"]
        for r in m["rows"]
        if (r["dataset"], r["engine"], r["protocol"]) == ("sw1", "a1", "review")
        and r["control_fpr"] > 0.05
    ]


def test_matrix_md_shows_what_was_measured() -> None:
    m = orjson.loads((EVIDENCE / "matrix.json").read_bytes())
    over = _a1_review_over_bound(m)
    assert over  # seed 2: 0.0501
    labelled = "Companion: sensitivity dataset (seed 0, magnitude_far_mult = 1.0)"
    seed2_pre = (
        "| companion: sw1 dataset, pre-a1 engine, magnitude_spike 10-100 | 2 | none (unused seed) |"
    )
    for md in ((EVIDENCE / "matrix.md").read_text(), _measure().render(m)):
        for x in over:  # visible at two decimals, named against the bound
            assert f"{100 * x:.2f} %" in md
        assert "above the 5 % Tier-3 bound (A1.12 item 7" in md
        assert md.count(labelled) == 3
        assert "Companion: sensitivity dataset (magnitude_far_mult" not in md
        assert seed2_pre in md
