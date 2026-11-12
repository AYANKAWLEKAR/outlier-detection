"""Build docs/leaderboard.md on the synthetic SW1 benchmark (Task 2.11, Amendment A1 Task 1.25).

SW1 = :func:`agentfw.synth.generate.sw1_config` (seed 0, all 20 personas, 90 days, all 8 attack
families, far-mode magnitude spikes with the top-up; spec A1.4), the Tier-1 acceptance dataset.
Every detector that can run offline is replayed on the full test split, by time and (as a separate
table) by agent, under the benchmark protocol (the oracle-reviewer approval loop, default engine
config; user decision 2026-11-10, decision 008). The engine is also replayed under spec §5.2
allow-all (``engine-allow-all``), and the engine before Amendment A1
(``EngineConfig().replace(**PRE_A1_SWITCHES)``) under both protocols (``engine-pre-a1``,
``engine-pre-a1-allow-all``). The page opens with the Tier-1 gate of those four runs (split
``time``) under both protocols side by side (spec A1.5), with the sensitivity companion rows
copied from ``docs/decisions/evidence/a1-gate/matrix.json``. ``llm_judge`` needs a live judge
model, so it is listed as not run; it is never faked here.

    uv run python scripts/build_leaderboard.py [--traces traces/sw1-seed0] [--reuse] [--jobs 4]

Runs land in ``results/synth-v1/<split>/<detector>/`` (gitignored); the rendered tables go to
``docs/leaderboard.md``. ``--reuse`` keeps a run whose report already matches the dataset, split,
detector, protocol (``extras.outcomes``), engine config (``meta.detector_config.cfg``) and n_boot;
it trusts outputs from older code (no git_sha check).
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import orjson

from agentfw.bench.leaderboard import (
    GATE_RUNS,
    build_leaderboard,
    render_gate_section,
    render_leaderboard,
)
from agentfw.bench.metrics import BOOTSTRAP_N, GATE_FAMILIES
from agentfw.bench.replay import BENCH_OUTCOMES, PROTOCOL_LABELS, OutcomeMode, SplitSpec
from agentfw.bench.run import run
from agentfw.bench.sweeps import PRE_A1_SWITCHES
from agentfw.core.config import EngineConfig
from agentfw.synth.generate import Manifest, generate, sw1_config
from agentfw.synth.load import load_personas

SW1 = sw1_config()
LABEL = "synthetic benchmark (SW1)"
PRE_A1 = EngineConfig().replace(**PRE_A1_SWITCHES)
# (run name, detector, protocol, engine config or None for the default): every detector under the
# benchmark protocol, the engine under spec §5.2 allow-all as the labelled comparison row, and the
# engine before Amendment A1 under both protocols (the companion rows of spec A1.5 rule 4).
RUNS: tuple[tuple[str, str, OutcomeMode, EngineConfig | None], ...] = (
    ("engine", "engine", BENCH_OUTCOMES, None),
    ("engine-allow-all", "engine", "allow", None),
    ("engine-pre-a1", "engine", "review", PRE_A1),
    ("engine-pre-a1-allow-all", "engine", "allow", PRE_A1),
    ("random", "random", BENCH_OUTCOMES, None),
    ("static_threshold", "static_threshold", BENCH_OUTCOMES, None),
    ("isolation_forest", "isolation_forest", BENCH_OUTCOMES, None),
)
#: The split whose engine runs give the gate section (the gate's own split, ``SplitSpec()``).
GATE_SPLIT = "time"
#: Task 1.23's gate evidence; its sensitivity rows are the companion rows of spec A1.5 rule 4.
GATE_EVIDENCE = Path(__file__).resolve().parents[1] / "docs/decisions/evidence/a1-gate/matrix.json"
_EVIDENCE_REL = "docs/decisions/evidence/a1-gate/matrix.json"
# matrix.json engine names -> the leaderboard run names of the same engine.
_MATRIX_RUNS = {
    ("a1", "review"): "engine",
    ("a1", "allow"): "engine-allow-all",
    ("pre-a1", "review"): "engine-pre-a1",
    ("pre-a1", "allow"): "engine-pre-a1-allow-all",
}
PRE_A1_SNAPSHOT = "6451d56"
SPLITS = ("time", "agent")
NOT_RUN = {"llm_judge": "requires a live judge model"}


def _dataset(d: Path) -> str:
    """Generate SW1 into ``d`` if absent; refuse a directory holding anything else."""
    if not (d / "manifest.json").exists():
        generate(SW1, d)
    m = Manifest.model_validate_json((d / "manifest.json").read_bytes())
    if m.config != SW1:
        raise SystemExit(f"{d} is not the SW1 dataset: {m.config!r}")
    return m.files["traces.jsonl"]


def _job(
    traces: Path,
    det: str,
    split: str,
    out: Path,
    n_boot: int,
    outcomes: OutcomeMode,
    cfg: EngineConfig | None,
) -> str:
    t0 = time.time()
    kw = {} if cfg is None else {"cfg": cfg}
    run(traces, det, SplitSpec(kind=split), out, n_boot=n_boot, outcomes=outcomes, **kw)  # type: ignore[arg-type]
    return f"{split}/{out.name}: {time.time() - t0:.0f} s"


def _fresh(
    out: Path,
    sha: str,
    n_boot: int,
    split: str,
    det: str,
    outcomes: OutcomeMode,
    cfg: EngineConfig | None,
) -> bool:
    """An existing report for this dataset, split, detector, protocol, engine config and n_boot
    (code version unchecked)."""
    p = out / "report.json"
    if not p.exists():
        return False
    rep = orjson.loads(p.read_bytes())
    meta = rep["meta"]
    want = None if cfg is None else orjson.loads(orjson.dumps(dataclasses.asdict(cfg)))
    return bool(
        meta["dataset_sha"] == sha
        and meta["split"]["kind"] == split
        and meta["detector"] == det
        and (meta.get("detector_config") or {}).get("cfg") == want
        and rep["extras"].get("n_boot") == n_boot
        and rep["extras"].get("outcomes") == outcomes
    )


def _sensitivity_reports(matrix: Path) -> dict[str, dict[str, Any]]:
    """The sensitivity rows of ``matrix`` (dataset ``sens``, seed ``SW1.seed``) as minimal
    reports keyed by leaderboard run name, so :func:`render_gate_section` prints them as the
    companion rows. Raises ``ValueError`` unless both engines are there under both protocols."""
    rows = orjson.loads(matrix.read_bytes())["rows"]
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        name = _MATRIX_RUNS.get((r["engine"], r["protocol"]))
        if r["dataset"] != "sens" or r["seed"] != SW1.seed or name is None:
            continue
        cells = {
            fam: {
                "bucket": bucket,
                "detector": detector,
                "recall": r["families"][fam]["recall"],
                "n_detected": r["families"][fam]["n_detected"],
                "n_total": r["families"][fam]["n_total"],
                "top_reason_share": r["families"][fam]["top1"],
                "among_reasons_share": r["families"][fam]["among"],
            }
            for fam, (bucket, detector) in GATE_FAMILIES.items()
            if fam in r["families"]
        }
        recalls = [c["recall"] for c in cells.values() if c["recall"] is not None]
        out[name] = {
            "meta": {
                "dataset_sha": r["dataset_sha"],
                "detector": "engine",
                "split": {"kind": GATE_SPLIT},
            },
            "extras": {"outcomes": r["protocol"]},
            "gate": {
                "cells": cells,
                "benign_fpr": r["benign_fpr"],
                "control_fpr": r["control_fpr"],
                "min_recall": min(recalls) if recalls else None,
                "passed": r["passed"],
                "missing": [],
                "failures": r["failures"],
            },
        }
    if len(out) != len(_MATRIX_RUNS):
        raise ValueError(
            f"{matrix} lacks sensitivity rows (dataset 'sens', seed {SW1.seed}) of "
            f"{sorted(set(_MATRIX_RUNS.values()) - set(out))}; spec A1.5 rule 4 needs both "
            "engines under both protocols"
        )
    return out


def _gate_sentence(reports: Mapping[str, Mapping[str, Any]]) -> str:
    """What the amended gate does under each protocol, read from the engine runs' reports."""
    review_run, allow_run = next(iter(GATE_RUNS.values()))
    pairs: tuple[tuple[str, OutcomeMode], ...] = ((review_run, "review"), (allow_run, "allow"))
    parts = []
    for run_name, mode in pairs:
        passed = reports[run_name]["gate"]["passed"]
        parts.append(f"{'passes' if passed else 'is not met'} under {PROTOCOL_LABELS[mode]}")
    return f"the amended gate {parts[0]} and {parts[1]}"


def _doc(
    sections: list[str],
    sha: str,
    reports: Mapping[str, Mapping[str, Any]],
    *,
    matrix: Path = GATE_EVIDENCE,
) -> str:
    """The page: header, the gate section of ``reports`` (run name -> report.json of split
    ``time``) with the sensitivity companion rows of ``matrix``, then ``sections`` under their
    own level-2 heading (so the outline never files the SW1 tables under the companion block)."""
    sens = _sensitivity_reports(matrix)
    sens_sha = str(next(iter(sens.values()))["meta"]["dataset_sha"])
    gate = render_gate_section(reports, sensitivity=sens)
    personas = len(load_personas(SW1.personas_dir))
    head = [
        "# Leaderboard: synthetic benchmark (SW1)",
        "",
        "<!-- generated by scripts/build_leaderboard.py; do not edit by hand -->",
        "",
        "**Every table on this page is the synthetic benchmark (SW1), except the companion rows "
        "on the sensitivity dataset in the gate section**. SW1 is `sw1_config()` after "
        f"Amendment A1 (spec A1.4): generator seed {SW1.seed}, {personas} personas, {SW1.days} "
        "days, all "
        f"{len(SW1.families)} attack families, attack rate {SW1.attack_rate:g}, "
        f'magnitude_mode "{SW1.magnitude_mode}" (a magnitude spike lands at or beyond '
        f"magnitude_far_mult {SW1.magnitude_far_mult:g} × the persona's benign p99 of the field) "
        f"and magnitude_topup_rate {SW1.magnitude_topup_rate:g} (a second, magnitude-only pass) "
        f"(dataset `{sha[:12]}`). It is *not* the live sandbox dataset v1 (2,000+ sessions of real "
        "LLM agents), which has not been built yet; see the README for how to build it. The "
        "Tier-2 exit numbers belong to that dataset.",
        "",
        "AUPRC and every other dataset-wide number on this page are not comparable with the "
        f"pre-A1 leaderboard (snapshot `{PRE_A1_SNAPSHOT}`): the SW1 attack mix changed (magnitude "
        "spikes are now about a third of the attacks; spec A1.5 rule 5).",
        "",
        f"Protocol: {PROTOCOL_LABELS[BENCH_OUTCOMES]}, default engine configuration, full test "
        f"split. The engine is also shown under {PROTOCOL_LABELS['allow']} (row "
        "`engine-allow-all`), and the engine before Amendment A1 (`EngineConfig().replace("
        "**PRE_A1_SWITCHES)`) under both protocols (rows `engine-pre-a1` and "
        "`engine-pre-a1-allow-all`); every row's caveat names its protocol. The baselines do not "
        "learn online, so the protocol does not change their scores. "
        "The by-time and by-agent splits are separate tables and are never averaged (§5.3). "
        "`llm_judge` is listed but was not run: it needs a live judge model and is evaluated on a "
        "stratified 2,000-envelope sample (§5.5), which only the sandbox dataset run will do.",
        "",
        "The leaderboard's per-agent FPR@95 is an operating point fixed after the fact on the "
        "test labels (spec §5.4). It is not the engine's live calibration.",
        "",
        f"Tier-1 gate (spec A1.7; live per-agent thresholds, split `{GATE_SPLIT}`): on this "
        f"dataset {_gate_sentence(reports)}. Which protocol's result closes Tier 1 is the user's "
        f"open decision (D-028); a pass under {PROTOCOL_LABELS['review']} alone is a pass under "
        "that protocol only. See [docs/decisions/008-tier1-gate.md](decisions/008-tier1-gate.md) "
        "and [docs/writeup.md](writeup.md).",
        "",
        "The companion rows on the sensitivity dataset (seed "
        f"{SW1.seed}, magnitude_far_mult = 1.0, dataset `{sens_sha[:12]}`; both engines, both "
        f"protocols) are copied from `{_EVIDENCE_REL}` (Task 1.23); this script does not replay "
        "that dataset.",
        "",
    ]
    boards = [
        f"## Leaderboards: {LABEL}",
        "",
        f"Every table below is on the {LABEL} (dataset `{sha[:12]}`).",
        "",
    ]
    return "\n".join(head) + "\n" + gate + "\n" + "\n".join(boards) + "\n" + "\n".join(sections)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--traces", type=Path, default=Path("traces/sw1-seed0"))
    p.add_argument("--results", type=Path, default=Path("results/synth-v1"))
    p.add_argument("--out", type=Path, default=Path("docs/leaderboard.md"))
    p.add_argument("--n-boot", type=int, default=BOOTSTRAP_N)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument(
        "--reuse",
        action="store_true",
        help="keep a run whose report matches the dataset sha, split, detector, protocol, engine "
        "config and n_boot; it trusts outputs from older code (there is no git_sha check)",
    )
    a = p.parse_args()
    sha = _dataset(a.traces)
    todo = [
        (a.traces, det, split, a.results / split / name, a.n_boot, oc, cfg)
        for split in SPLITS
        for name, det, oc, cfg in RUNS
        if not (a.reuse and _fresh(a.results / split / name, sha, a.n_boot, split, det, oc, cfg))
    ]
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for msg in ex.map(_job, *zip(*todo, strict=True)) if todo else []:
            print(msg, flush=True)
    sections = []
    for split in SPLITS:
        reports = [a.results / split / name / "report.json" for name, *_ in RUNS]
        lb = build_leaderboard(reports, label=LABEL, not_run=NOT_RUN)
        sections.append(render_leaderboard(lb))
    gate_reports = {
        name: orjson.loads((a.results / GATE_SPLIT / name / "report.json").read_bytes())
        for name, *_ in RUNS
    }
    a.out.write_text(_doc(sections, sha, gate_reports))
    print(a.out)


if __name__ == "__main__":
    main()
