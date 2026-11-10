"""Rewrite ``tests/golden/<detector>.sha256`` from a replay of the CI dataset (D-019).

The replay uses ``CI_OUTCOMES`` (the benchmark protocol, the oracle-reviewer approval loop)."""

from __future__ import annotations

import argparse
from pathlib import Path

from agentfw.bench.ci_regress import CI_OUTCOMES, ci_dataset
from agentfw.bench.registry import make_detector
from agentfw.bench.replay import SplitSpec, canonical_hash, replay

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "tests" / "golden"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--detector", required=True)
    a = p.parse_args()
    rows = replay(*ci_dataset(), make_detector(a.detector), SplitSpec(), outcomes=CI_OUTCOMES)
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    path = GOLDEN_DIR / f"{a.detector}.sha256"
    path.write_text(canonical_hash(rows) + "\n")
    print(f"{path}: {path.read_text().strip()}")


if __name__ == "__main__":
    main()
