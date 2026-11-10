"""CI regression: replay every baselined detector on a fixed dataset and compare to the baseline.

Fails (exit 1) if AUPRC drops by more than 0.01 or per-agent FPR@95 rises by more than 0.002
(0.2 pp). ``baseline_metrics.json`` maps ``detector -> {"auprc", "fpr95"}`` (``null`` = NaN, i.e.
the target recall was unreachable). Rewrite it with ``--write-baseline``.

Replays use :data:`CI_OUTCOMES`, the benchmark protocol (the oracle-reviewer approval loop, user
decision 2026-11-10, decision 008); ``scripts/update_golden.py`` and the golden tests use it too.
"""

from __future__ import annotations

import argparse
import math
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import orjson

from agentfw.bench.dataset import load_dir
from agentfw.bench.metrics import compute_all
from agentfw.bench.registry import make_detector
from agentfw.bench.replay import BENCH_OUTCOMES, OutcomeMode, SplitSpec, replay
from agentfw.core.envelope import Label, RawCall
from agentfw.synth.generate import GenerateConfig, generate
from agentfw.synth.load import CATALOG_DIR, PERSONAS_DIR, TASKS_DIR, TIER0_PERSONA_IDS

__all__ = [
    "BASELINE_PATH",
    "CI_DATASET",
    "CI_OUTCOMES",
    "ci_dataset",
    "compare",
    "main",
    "measure",
]

BASELINE_PATH = Path(__file__).with_name("baseline_metrics.json")
AUPRC_MAX_DROP = 0.01
FPR95_MAX_RISE = 0.002
_TOL = 1e-9  # float slack so a change of exactly the limit passes
_N_BOOT = 0  # the regression compares point estimates only
CI_OUTCOMES: OutcomeMode = BENCH_OUTCOMES  # the benchmark protocol (decision 008, user decision)

# Ruling R2: attack_rate=0.15 leaves enough attacks in 200 sessions (1.5% would leave ~2).
CI_DATASET = GenerateConfig(
    seed=0,
    personas_dir=PERSONAS_DIR,
    persona_ids=list(TIER0_PERSONA_IDS),  # ruling A4: the CI dataset stays the 3 Tier-0 personas
    tasks_dir=TASKS_DIR,
    catalog_dir=CATALOG_DIR,
    days=60,
    max_sessions=200,
    attack_rate=0.15,
    families=["magnitude_spike", "prereq_skip", "novel_target"],
)


def ci_dataset() -> tuple[list[RawCall], dict[str, Label]]:
    """Generate ``CI_DATASET`` (deterministic) and load it."""
    with tempfile.TemporaryDirectory() as d:
        generate(CI_DATASET, Path(d))
        raws, labels, _ = load_dir(d)
    return raws, labels


def _num(v: float | None) -> float:
    return math.nan if v is None else float(v)


def compare(
    current: Mapping[str, float], baseline: Mapping[str, float], *, name: str = ""
) -> list[str]:
    """Failure messages; NaN: nan->nan ok, finite->nan fails, nan->finite ok (improvement)."""
    pre = f"{name}: " if name else ""
    out: list[str] = []
    for key, limit, sign in (("auprc", AUPRC_MAX_DROP, -1.0), ("fpr95", FPR95_MAX_RISE, 1.0)):
        cur, base = _num(current[key]), _num(baseline[key])
        if math.isnan(base):
            continue
        if math.isnan(cur):
            out.append(f"{pre}{key} became NaN (baseline {base:.4f})")
        elif sign * (cur - base) > limit + _TOL:
            verb = "dropped" if sign < 0 else "rose"
            out.append(
                f"{pre}{key} {verb} {abs(cur - base):.4f} (limit {limit}): {base:.4f} -> {cur:.4f}"
            )
    return out


def measure(detector: str, data: tuple[list[RawCall], dict[str, Label]]) -> dict[str, float]:
    rows = replay(*data, make_detector(detector), SplitSpec(), outcomes=CI_OUTCOMES)
    m = compute_all(rows, warm_only=False, n_boot=_N_BOOT)
    return {"auprc": m.auprc, "fpr95": m.fpr95_per_agent}


def _dump(metrics: Mapping[str, Mapping[str, float]]) -> bytes:
    clean = {
        d: {k: (None if math.isnan(v) else v) for k, v in m.items()}
        for d, m in sorted(metrics.items())
    }
    return orjson.dumps(clean, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS) + b"\n"


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m agentfw.bench.ci_regress")
    p.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    p.add_argument("--write-baseline", action="store_true")
    a = p.parse_args(argv)
    baseline: dict[str, dict[str, float | None]] = orjson.loads(a.baseline.read_bytes())
    data = ci_dataset()
    current = {d: measure(d, data) for d in baseline}
    if a.write_baseline:
        a.baseline.write_bytes(_dump(current))
        print(f"wrote {a.baseline}")
        return 0
    failures: list[str] = []
    for d, cur in current.items():
        print(f"{d}: auprc={cur['auprc']:.4f} fpr95={cur['fpr95']:.4f}")
        failures += compare(cur, baseline[d], name=d)  # type: ignore[arg-type]
    for f in failures:
        print(f"FAIL {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
