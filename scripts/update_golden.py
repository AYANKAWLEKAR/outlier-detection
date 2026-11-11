"""Rewrite ``tests/golden/<name>.sha256`` from a replay of the CI dataset (D-019).

    uv run python scripts/update_golden.py --golden engine
    uv run python scripts/update_golden.py --golden all

Each golden's detector, config and replay protocol come from ``agentfw.bench.ci_regress.GOLDENS``
(``engine`` / ``static_threshold``: the benchmark protocol; ``engine-spec-allow``: the spec engine
under spec §5.2 allow-all).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agentfw.bench.ci_regress import GOLDENS, golden_rows
from agentfw.bench.replay import canonical_hash

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "tests" / "golden"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--golden", required=True, choices=[*GOLDENS, "all"])
    a = p.parse_args()
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for name in GOLDENS if a.golden == "all" else [a.golden]:
        path = GOLDEN_DIR / f"{name}.sha256"
        path.write_text(canonical_hash(golden_rows(name)) + "\n")
        print(f"{path}: {path.read_text().strip()}")


if __name__ == "__main__":
    main()
