from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from agentfw.bench import calibrated
from agentfw.bench.calibrated import (
    NORMALIZER_FILES,
    load_calibrated,
    main,
    normalizer_hash,
)
from agentfw.core.config import EngineConfig


def test_calibrated_constants_match_normalizer() -> None:
    assert load_calibrated().normalizer_hash == normalizer_hash(), (
        "normalizer changed: recalibrate, or restamp with --reason if no constant depends on it"
    )


def test_normalizer_files_exist_and_hash_is_cwd_independent(tmp_path: Path) -> None:
    assert NORMALIZER_FILES == (
        "agentfw/core/normalize.py",
        "agentfw/core/patterns.py",
        "agentfw/core/detectors/base.py",
    )
    here = normalizer_hash()
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "from agentfw.bench.calibrated import normalizer_hash as h; print(h())",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONPATH": str(Path(calibrated.__file__).parents[2]), "PATH": ""},
    )
    assert out.stdout.strip() == here


def test_seeded_constants_are_engine_defaults() -> None:
    c = EngineConfig()
    got = load_calibrated()
    assert got.constants == {
        "z0": c.z0,
        "tau": c.tau,
        "prereq_rate": c.prereq_rate,
        "cal_alpha": c.cal_alpha,
        "novelty_full_n": float(c.novelty_full_n),
    }
    assert got.history[0] == "2026-11-11 initial stamp: tiers 0-2 constants"


def test_restamp_requires_reason(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["--restamp", "--path", str(tmp_path / "c.json")])


def test_restamp_rewrites_hash_and_appends_history(tmp_path: Path) -> None:
    p = tmp_path / "c.json"
    p.write_bytes(calibrated.CALIBRATED_PATH.read_bytes())
    raw = json.loads(p.read_text())
    raw["normalizer_hash"] = "stale"
    p.write_text(json.dumps(raw))
    assert main(["--restamp", "--reason", "no constant depends on it", "--path", str(p)]) == 0
    got = json.loads(p.read_text())
    assert got["normalizer_hash"] == normalizer_hash()
    assert got["history"][:-1] == raw["history"]
    assert got["history"][-1].endswith(" no constant depends on it")
    assert got["history"][-1][4] == "-"  # "<YYYY-MM-DD> <reason>"
    text = p.read_text()
    assert text.endswith("\n") and list(got) == sorted(got)
