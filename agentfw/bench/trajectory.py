"""Trajectory benchmark (Task 4.1, F10): every trajectory switch measured on the bench.

``python -m agentfw.bench.trajectory --traces DIR --out DIR [--seeds 0,1] [--workers N]`` writes
``trajectory_report.json`` and ``trajectory_report.md`` into ``--out``.

**Datasets.** One per seed, ``DIR/seed-{s}``: the generator's bench config (``sw1_config(s)``)
with ``families`` extended by the runaway families and ``splice`` (:func:`bench_config`). A
missing directory is generated; an existing one is used as it is and its config is recorded
(``is_bench_config`` says whether it is this one). Every replay uses the time split with a
calibration partition of :data:`CALIBRATION_FRAC` of the agents (F10's three-way split).

**Rows** (:data:`ROWS`). ``engine``; one row per switch toggled alone on top of it; ``engine-traj``
(every switch on, ``registry.TRAJ_SWITCHES``); and a few labelled composite rows (a switch that
is meant to run with another one, and the shadow configurations the session metrics need). Each
row is replayed under both protocols (:data:`PROTOCOLS`); ``review`` numbers carry
``PROTOCOL_LABELS["review"]`` and are optimistic.

**Per row** (:func:`row_summary`): per-call metrics (``compute_all`` on warm test rows, bootstrap
CIs over sessions), decision metrics (benign and control FPR, attack-level recall per family,
escalation rate, decision counts), precision at the base rate for every operating point, splice
recall (reported separately, T-40), session AUROC on all sessions and on the length-matched set,
the vital false-flag rate, and, for rows with a density column, halt metrics, the sensitivity
curve, the first-flag baseline and the calibration-partition threshold selection.

**Deltas** (:func:`paired_deltas`): each row minus ``engine`` on the same seed and protocol, with
a paired percentile bootstrap over test-split sessions (both rows resampled with the same
sessions), for the decision benign FPR, the decision control FPR and every family's
attack-level decision recall.

**Attribution** (``bench.attribution``): ``engine-traj`` with the groups of :data:`TRAJ_GROUPS`
(the four v1 detectors and ``vital`` dropped, ``gate`` and ``provisional`` switched off),
replayed with the dataset's schemas as hints, as every bench row is. Besides the table, the
literal F10 property is checked per group: removing a group changes exactly the attacks it
carried (no attack is gained either). The T-20 totals are reported too, but ``tabulate`` makes
them hold by construction.

**Latency** (:func:`latency_ab`): every row's engine scores the same calls of the first seed in
one process, the order rotated every call (alternated A/B), all updated with ``ALLOWED``.

**Threshold selection** (:func:`select_on_calibration`): ``theta_halt`` is the smallest value of
the 0.05 grid on [0.2, 0.8] whose benign false-halt rate on the calibration partition is at most
1%; ``theta_warn`` the smallest whose benign warn rate is at most 5%, capped strictly below
``theta_halt`` (undefined when ``theta_halt`` is the grid's floor). ``--write-calibrated`` records
the selection of the ``engine-traj-shadow`` row (spec protocol, first seed) in ``calibrated.json``
with the normalizer hash, under ``*_selected_cal`` keys that no engine reads; it changes no
``EngineConfig`` default (T-03, T-54).

**Re-rendering** (``--rerender JSON --out DIR``): a saved report is re-rendered with this code's
selection rule and labels, without replaying (``meta.rendered_git_sha`` records the code).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Literal

import numpy as np
import orjson
from numpy.typing import NDArray

from agentfw.bench import metrics as M
from agentfw.bench import session_metrics as SM
from agentfw.bench.attribution import (
    REDUNDANT,
    AttributionGroup,
    AttributionRuns,
    detected_attacks,
    group_engine,
    tabulate,
)
from agentfw.bench.calibrated import CALIBRATED_PATH, load_calibrated, normalizer_hash
from agentfw.bench.dataset import load_dir, load_schema_hints
from agentfw.bench.registry import TRAJ_SWITCHES
from agentfw.bench.replay import (
    PROTOCOL_LABELS,
    OutcomeMode,
    ScoreRow,
    SplitSpec,
    canonical_hash,
    replay,
)
from agentfw.bench.report import git_sha
from agentfw.bench.trajectory_md import render_markdown
from agentfw.core.config import EngineConfig
from agentfw.core.engine import Engine
from agentfw.core.envelope import Label, Outcome, RawCall
from agentfw.core.hints import ToolHint
from agentfw.core.normalize import SessionTracker, normalize
from agentfw.synth.attacks import RUNAWAY_FAMILIES, SPLICE_FAMILIES
from agentfw.synth.generate import GenerateConfig, Manifest, generate, sw1_config

__all__ = [
    "CALIBRATION_FRAC",
    "PROTOCOLS",
    "ROWS",
    "SPLIT",
    "THETA_GRID",
    "TRAJ_FAMILIES",
    "TRAJ_GROUPS",
    "BenchRow",
    "bench_config",
    "build_report",
    "calibration_curve",
    "ensure_dataset",
    "latency_ab",
    "main",
    "paired_deltas",
    "row_summary",
    "run_bench",
    "run_jobs",
    "rerender",
    "select_from_curve",
    "select_on_calibration",
    "write_calibrated",
]

#: Families added to the bench config: the runaway families (session metrics only, T-10) and
#: ``splice`` (reported separately, T-40).
TRAJ_FAMILIES: tuple[str, ...] = (*sorted(RUNAWAY_FAMILIES), *sorted(SPLICE_FAMILIES))
CALIBRATION_FRAC = 0.15
SPLIT = SplitSpec(kind="time", calibration_frac=CALIBRATION_FRAC)
PROTOCOLS: tuple[OutcomeMode, ...] = ("allow", "review")
#: F7's threshold grid, 0.2 to 0.8 in steps of 0.05 (exact decimals).
THETA_GRID: tuple[float, ...] = tuple(round(0.2 + 0.05 * i, 2) for i in range(13))
FALSE_HALT_MAX = 0.01
WARN_RATE_MAX = 0.05
N_BOOT_DELTA = 1000
CI_LEVEL = 0.95
REPORT_JSON = "trajectory_report.json"
REPORT_MD = "trajectory_report.md"
#: The row whose calibration-partition selection ``--write-calibrated`` records.
THETA_SOURCE_ROW = "engine-traj-shadow"


@dataclass(frozen=True, slots=True)
class BenchRow:
    """One replay configuration: a registry engine variant, then config overrides set on top of
    the config the variant builds (so they can also turn off one of ``engine-traj``'s switches)."""

    name: str
    detector: str
    overrides: Mapping[str, object] = field(default_factory=dict)
    #: The switch this row toggles alone on top of ``engine`` (None: reference or composite).
    switch: str | None = None
    note: str = ""


ROWS: tuple[BenchRow, ...] = (
    BenchRow("engine", "engine", note="reference: today's default engine"),
    BenchRow("flow_ledger", "engine", {"flow_ledger": True}, "flow_ledger"),
    BenchRow("vital", "engine-vital", {}, "vital", "v1 + vital (enabled=KNOWN_DETECTORS)"),
    BenchRow("br_score", "engine", {"br_mode": "score"}, "br_mode=score"),
    BenchRow("br_modulate", "engine", {"br_mode": "modulate"}, "br_mode=modulate"),
    BenchRow("sessions", "engine", {"sessions": True}, "sessions"),
    BenchRow("provisional", "engine", {"provisional": True}, "provisional"),
    BenchRow("seq_age_discount", "engine", {"seq_age_discount": True}, "seq_age_discount"),
    BenchRow("gate", "engine", {"gate": True}, "gate"),
    BenchRow("escalate", "engine", {"escalate": True}, "escalate"),
    BenchRow("session_enforce", "engine", {"session_enforce": True}, "session_enforce"),
    BenchRow("authority", "engine", {"authority": True}, "authority"),
    BenchRow("engine-traj", "engine-traj", note="every switch on (registry.TRAJ_SWITCHES + vital)"),
    BenchRow(
        "gate+flow_ledger",
        "engine",
        {"gate": True, "flow_ledger": True},
        note="composite: the gate reads ledger history (T-41)",
    ),
    BenchRow(
        "escalate+gate+flow_ledger",
        "engine",
        {"escalate": True, "gate": True, "flow_ledger": True},
        note="composite: escalation of gate escalations (F5 after F4)",
    ),
    BenchRow(
        "sessions+vital",
        "engine-vital",
        {"sessions": True},
        note="composite: the F7 shadow halt configuration (vital flags recorded, nothing enforced)",
    ),
    BenchRow(
        "engine-traj-shadow",
        "engine-traj",
        {"session_enforce": False},
        note="composite: engine-traj with session verdicts recorded, not enforced (exact halt "
        "metrics at every threshold, T-43)",
    ),
)
_ROW_BY_NAME = {r.name: r for r in ROWS}

TRAJ_GROUPS: tuple[AttributionGroup, ...] = (
    AttributionGroup("magnitude", drop_detector="magnitude"),
    AttributionGroup("novelty", drop_detector="novelty"),
    AttributionGroup("velocity", drop_detector="velocity"),
    AttributionGroup("sequence", drop_detector="sequence"),
    AttributionGroup("vital", drop_detector="vital"),
    AttributionGroup("gate", cfg_off={"gate": False}),
    AttributionGroup("provisional", cfg_off={"provisional": False}),
)
ATTRIBUTION_DETECTOR = "engine-traj"


# -- datasets -------------------------------------------------------------------------------------


def bench_config(seed: int) -> GenerateConfig:
    """The bench config (``sw1_config(seed)``) with ``families`` extended by
    :data:`TRAJ_FAMILIES`."""
    base = sw1_config(seed)
    return base.model_copy(update={"families": [*base.families, *TRAJ_FAMILIES]})


def ensure_dataset(root: Path, seed: int) -> Path:
    """``root/seed-{seed}``, generated with :func:`bench_config` when it has no manifest."""
    d = root / f"seed-{seed}"
    if not (d / "manifest.json").exists():
        d.mkdir(parents=True, exist_ok=True)
        generate(bench_config(seed), d)
    return d


def _manifest(d: Path) -> Manifest:
    return Manifest.model_validate_json((d / "manifest.json").read_bytes())


@lru_cache(maxsize=1)
def _load(d: str) -> tuple[list[RawCall], dict[str, Label], dict[str, ToolHint]]:
    raws, labels, _ = load_dir(d)
    return raws, labels, load_schema_hints(d)


# -- jobs -----------------------------------------------------------------------------------------

JobKind = Literal["row", "group"]


@dataclass(frozen=True, slots=True)
class Job:
    kind: JobKind
    traces: str
    seed: int
    outcomes: OutcomeMode
    name: str  # a ROWS name (kind "row") or a TRAJ_GROUPS name (kind "group")
    n_boot: int = 0


@dataclass(frozen=True, slots=True)
class Vectors:
    """Compact per-row columns of a replay's decided test rows (replay order), for the paired
    bootstrap. ``key`` is the sha256 of their envelope ids, so two rows' vectors are only
    compared when they cover the same rows in the same order."""

    key: str
    session: NDArray[np.int64]
    benign: NDArray[np.bool_]
    control: NDArray[np.bool_]
    attack: NDArray[np.int64]  # index into ``attack_family`` (-1: not an attack row)
    attack_family: tuple[str, ...]
    flagged: NDArray[np.bool_]


@dataclass(slots=True)
class JobResult:
    job: Job
    wall_s: float
    summary: dict[str, Any] = field(default_factory=dict)
    vectors: Vectors | None = None
    detected: frozenset[str] = frozenset()


def _engine(row: BenchRow, hints: Mapping[str, ToolHint]) -> Engine:
    """The row's engine: its variant with the dataset's schemas as hints (as
    ``make_bench_detector``), then its overrides on top of the variant's config."""
    over = AttributionGroup(row.name, cfg_off=row.overrides) if row.overrides else None
    return group_engine(row.detector, EngineConfig(), over, hints)


def _run_job(job: Job) -> JobResult:
    t0 = time.time()
    raws, labels, hints = _load(job.traces)
    if job.kind == "row":
        det = _engine(_ROW_BY_NAME[job.name], hints)
    else:
        group = next(g for g in TRAJ_GROUPS if g.name == job.name)
        det = group_engine(ATTRIBUTION_DETECTOR, EngineConfig(), group, hints)
    rows = replay(raws, labels, det, SPLIT, outcomes=job.outcomes)
    res = JobResult(job, 0.0, detected=frozenset(detected_attacks(rows)))
    if job.kind == "row":
        res.summary = row_summary(rows, det, n_boot=job.n_boot)
        res.vectors = vectors(rows)
    res.wall_s = round(time.time() - t0, 1)
    return res


def _progress(r: JobResult, i: int, n: int, t0: float) -> None:
    j = r.job
    print(
        f"[{i}/{n}] {j.kind} {j.name} seed={j.seed} {j.outcomes}: {r.wall_s}s "
        f"(elapsed {time.time() - t0:.0f}s)",
        file=sys.stderr,
        flush=True,
    )


def run_jobs(jobs: Sequence[Job], workers: int, *, progress: bool = False) -> list[JobResult]:
    """Run ``jobs`` (results in job order) in ``workers`` processes, submitted grouped by
    dataset so each worker's loaded dataset is reused; ``progress`` prints each finished job to
    stderr."""
    order = sorted(range(len(jobs)), key=lambda i: (jobs[i].traces, i))
    out: dict[int, JobResult] = {}
    t0 = time.time()
    if workers <= 1:
        for k, i in enumerate(order, 1):
            out[i] = _run_job(jobs[i])
            if progress:
                _progress(out[i], k, len(jobs), t0)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_run_job, jobs[i]): i for i in order}
            for k, fut in enumerate(as_completed(futs), 1):
                out[futs[fut]] = fut.result()
                if progress:
                    _progress(out[futs[fut]], k, len(jobs), t0)
    return [out[i] for i in range(len(jobs))]


# -- per-row metrics ------------------------------------------------------------------------------


def _f(x: float | None) -> float | None:
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) or math.isinf(x) else x


def _decision_hash(rows: Sequence[ScoreRow]) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(orjson.dumps([r.envelope_id, r.decision]))
        h.update(b"\n")
    return h.hexdigest()


def _percall(rows: Sequence[ScoreRow], n_boot: int) -> dict[str, Any]:
    m = M.compute_all(rows, warm_only=True, n_boot=n_boot, seed=0)
    br = m.attack_base_rate
    return {
        "population": "warm test rows; splice, runaway and calibration rows excluded",
        "n_boot": n_boot,
        "auprc": _f(m.auprc),
        "auprc_ci": [_f(x) for x in m.auprc_ci],
        "fpr95_global": _f(m.fpr95_global),
        "fpr95_per_agent": _f(m.fpr95_per_agent),
        "fpr95_per_agent_ci": [_f(x) for x in m.fpr95_per_agent_ci],
        "per_agent_recall": _f(m.per_agent_recall),
        "control_fpr_per_agent_thr": _f(m.control_fpr),
        "attack_base_rate": _f(br),
        "precision_global95": _f(m.precision_global),
        "precision_per_agent95": _f(M.precision_at(m.per_agent_recall, m.fpr95_per_agent, br)),
        "n_rows": m.n_rows,
        "n_attack_rows": m.n_attack_rows,
    }


def _decisions(rows: Sequence[ScoreRow]) -> dict[str, Any]:
    test = [r for r in rows if r.split == "test"]
    dec = [r for r in test if r.decision is not None and not M.is_runaway(r.attack_family)]
    benign = [r for r in dec if r.label != "attack"]
    attacks = [
        r for r in dec if r.label == "attack" and not M.is_reported_separately(r.attack_family)
    ]
    pop = [r for r in dec if not M.is_reported_separately(r.attack_family)]
    env_recall = sum(map(SM.decision_flagged, attacks)) / len(attacks) if attacks else math.nan
    bfpr = SM.decision_benign_fpr(test)
    base = len(attacks) / len(pop) if pop else math.nan
    n_atk: dict[str, set[str]] = defaultdict(set)
    for r in dec:
        if r.label == "attack":
            n_atk[r.attack_family or "unknown"].add(r.attack_id or r.envelope_id)
    counts = Counter(f"{r.label}:{r.decision}" for r in dec)
    flags: Counter[str] = Counter()
    for r in test:
        flags.update(f for f in r.flags.split(",") if f)
    thetas = [r.threshold for r in benign if r.threshold is not None and math.isfinite(r.threshold)]
    return {
        "population": "test split; runaway rows excluded (T-37); splice is its own family",
        "benign_fpr": _f(bfpr),
        "control_fpr": _f(SM.decision_control_fpr(test)),
        "recall_by_family": {
            k: _f(v) for k, v in sorted(SM.decision_recall_by_family(test).items())
        },
        "n_attacks_by_family": {k: len(v) for k, v in sorted(n_atk.items())},
        "envelope_recall": _f(env_recall),
        "decision_base_rate": _f(base),
        "precision_at_base_rate": _f(M.precision_at(env_recall, bfpr, base)),
        "escalation_rate_benign": _f(sum(r.escalated for r in benign) / len(benign))
        if benign
        else None,
        "n_block": sum(r.decision == "block" for r in dec),
        "n_require_approval": sum(r.decision == "require_approval" for r in dec),
        "counts": dict(sorted(counts.items())),
        "flags": dict(sorted(flags.items())),
        "theta_benign_median": _f(float(np.median(thetas))) if thetas else None,
    }


def _halt(h: SM.HaltMetrics) -> dict[str, Any]:
    return {
        "recall_runaway": _f(h.recall_by_label.get("runaway")),
        "recall_attacked": _f(h.recall_by_label.get("attacked")),
        "false_halt_rate": _f(h.false_halt_rate),
        "median_calls_from_onset": _f(h.median_calls_from_onset),
        "precision_at_base_rate": _f(h.precision_at_base_rate),
    }


def calibration_curve(rows: Sequence[ScoreRow]) -> list[tuple[float, float, int]]:
    """``(theta, share, n)`` over :data:`THETA_GRID`: the share of the ``n`` benign sessions of
    the calibration partition whose density reaches ``theta`` on some call. A session reaching
    ``theta`` halts there (``theta`` as ``theta_halt``) or is warned (as ``theta_warn``), so one
    curve gives both rates. ``ValueError`` without a calibration benign session or a density."""
    cal = [r for r in rows if r.split == "calibration"]
    labels = SM.session_labels(cal)
    peak: dict[str, float] = {}
    any_density = False
    for r in cal:
        if labels[r.session_id] != "benign":
            continue
        d = -1.0
        if r.density is not None:
            any_density = True
            d = r.density
        peak[r.session_id] = max(peak.get(r.session_id, -1.0), d)
    if not peak:
        raise ValueError("no benign session in the calibration partition")
    if not any_density:
        raise ValueError("the density column is empty: replay with the session gate open")
    n = len(peak)
    vals = np.fromiter(peak.values(), dtype=np.float64, count=n)
    return [(t, float(np.sum(vals >= t - 1e-12)) / n, n) for t in THETA_GRID]


def select_from_curve(
    curve: Sequence[tuple[float, float, int]],
) -> tuple[float | None, float]:
    """``(theta_warn, theta_halt)`` from a :func:`calibration_curve`. ``theta_halt`` is the
    smallest grid value with share <= 1% (the grid's top when none qualifies); ``theta_warn`` the
    smallest with share <= 5%, capped strictly below ``theta_halt`` at the next grid value down,
    and ``None`` (undefined) when ``theta_halt`` is the grid's floor (T-54)."""
    halt = next((t for t, share, _ in curve if share <= FALSE_HALT_MAX), THETA_GRID[-1])
    warn = next((t for t, share, _ in curve if share <= WARN_RATE_MAX), THETA_GRID[-1])
    below = [t for t in THETA_GRID if t < halt]
    return (min(warn, below[-1]) if below else None), halt


def select_on_calibration(rows: Sequence[ScoreRow]) -> tuple[float | None, float]:
    """``(theta_warn, theta_halt)`` chosen on the calibration partition: :func:`select_from_curve`
    of :func:`calibration_curve` (module docstring; T-54)."""
    return select_from_curve(calibration_curve(rows))


def _sessions(rows: Sequence[ScoreRow], vital: bool, cfg: EngineConfig) -> dict[str, Any]:
    labels = SM.session_labels(rows)
    out: dict[str, Any] = {
        "population": "all splits, session level (T-40: attacked = any non-runaway attack)",
        "n_sessions": dict(sorted(Counter(labels.values()).items())),
        "vital_false_flag_rate": _f(SM.vital_false_flag_rate(rows)) if vital else None,
        "session_auroc_all": _f(SM.session_auroc(rows)),
        "session_auroc_length_matched": _f(SM.session_auroc(SM.length_matched(rows))),
        "n_length_matched_pairs": len(SM.matched_pairs(rows)),
        "session_auroc_runaway": _f(SM.session_auroc(rows, frozenset({"runaway"}))),
    }
    splice: dict[str, Any] = {}
    for warm in (False, True):
        for mode in ("per_agent", "row"):
            sp = M.splice_report(rows, warm_only=warm, breakdown_thresholds=mode)
            splice[f"{'warm' if warm else 'all'}_{mode}"] = {
                "attack_recall": _f(sp.attack.recall),
                "n_detected": sp.attack.n_detected,
                "n_total": sp.attack.n_total,
                "envelope_recall": _f(sp.envelope.recall),
            }
    test = [r for r in rows if r.split == "test"]
    splice["decision_recall"] = _f(SM.decision_recall_by_family(test).get("splice"))
    out["splice"] = splice
    if all(r.density is None for r in rows):
        out["halt"] = None
        return out
    base = 0.015
    h = SM.halt_metrics(rows, EngineConfig().theta_halt, base)
    ff = SM.first_flag_halts(rows, base)
    sens = []
    for t in THETA_GRID:
        m = SM.halt_metrics(rows, t, base)
        sens.append({"theta": t, **_halt(m)})
    steps = [
        abs(a["false_halt_rate"] - b["false_halt_rate"])
        for a, b in zip(sens, sens[1:], strict=False)
        if a["false_halt_rate"] is not None and b["false_halt_rate"] is not None
    ]
    out["halt"] = {
        "theta_halt": EngineConfig().theta_halt,
        "session_base_rate": base,
        "density_policy": _halt(h),
        "first_flag": _halt(ff),
        "sensitivity": sens,
        "sensitivity_max_step": max(steps) if steps else None,
        # T-43: offline halts of an enforcing replay with provisional on are exact only for
        # thresholds at or below the replay's theta_halt; every other replay is exact throughout.
        "exact_at_every_theta": not (cfg.session_enforce and cfg.provisional),
        "calibration_curve": None,
        "selected": None,
    }
    if any(r.split == "calibration" and r.label != "attack" for r in rows):
        warn, halt = select_on_calibration(rows)
        out["halt"]["calibration_curve"] = [
            {"theta": t, "benign_share": s, "n_benign": n} for t, s, n in calibration_curve(rows)
        ]
        out["halt"]["selected"] = {"theta_warn": warn, "theta_halt": halt}
    return out


def row_summary(rows: Sequence[ScoreRow], det: Engine, *, n_boot: int) -> dict[str, Any]:
    """Every per-row number of the report (module docstring)."""
    return {
        "enabled": sorted(det.enabled),
        "config": {
            k: v for k, v in dataclasses.asdict(det.cfg).items() if not isinstance(v, tuple)
        },
        "hints_from_schemas": bool(det.hints),
        "n_rows": len(rows),
        "score_hash": canonical_hash(rows),
        "decision_hash": _decision_hash(rows),
        "detected_attacks": len(detected_attacks(rows)),
        "percall": _percall(rows, n_boot),
        "decisions": _decisions(rows),
        "sessions": _sessions(rows, "vital" in det.enabled, det.cfg),
    }


def vectors(rows: Sequence[ScoreRow]) -> Vectors:
    """The decided test rows' columns for :func:`paired_deltas` (``_decisions``' population)."""
    dec = [
        r
        for r in rows
        if r.split == "test" and r.decision is not None and not M.is_runaway(r.attack_family)
    ]
    sess: dict[str, int] = {}
    fams: list[str] = []
    fam_idx: dict[str, int] = {}
    first: dict[str, ScoreRow] = {}
    for r in dec:
        if r.label == "attack":
            k = r.attack_id or r.envelope_id
            cur = first.get(k)
            if cur is None or (r.ts_unix_ms, r.envelope_id) < (cur.ts_unix_ms, cur.envelope_id):
                first[k] = r
    atk_of: dict[str, int] = {}
    for k in sorted(first):
        fam = first[k].attack_family or "unknown"
        if fam not in fam_idx:
            fam_idx[fam] = len(fams)
            fams.append(fam)
        atk_of[k] = len(atk_of)
    atk_fam = [0] * len(atk_of)
    for k, i in atk_of.items():
        atk_fam[i] = fam_idx[first[k].attack_family or "unknown"]
    h = hashlib.sha256()
    for r in dec:
        h.update(r.envelope_id.encode())
    attack = np.array(
        [atk_of[r.attack_id or r.envelope_id] if r.label == "attack" else -1 for r in dec],
        dtype=np.int64,
    )
    return Vectors(
        key=h.hexdigest(),
        session=np.array([sess.setdefault(r.session_id, len(sess)) for r in dec], dtype=np.int64),
        benign=np.array([r.label != "attack" for r in dec], dtype=np.bool_),
        control=np.array(
            [r.label != "attack" and r.control_type == "benign_unusual" for r in dec],
            dtype=np.bool_,
        ),
        attack=attack,
        attack_family=tuple(fams[i] for i in atk_fam),
        flagged=np.array([SM.decision_flagged(r) for r in dec], dtype=np.bool_),
    )


# -- paired deltas --------------------------------------------------------------------------------


def _per_session(v: Vectors, n_sess: int) -> dict[str, tuple[NDArray[np.float64], ...]]:
    """metric -> (numerator per session, denominator per session) for ``v``."""
    s = v.session
    out: dict[str, tuple[NDArray[np.float64], ...]] = {}
    for name, mask in (("benign_fpr", v.benign), ("control_fpr", v.control)):
        num = np.bincount(s[mask & v.flagged], minlength=n_sess).astype(np.float64)
        den = np.bincount(s[mask], minlength=n_sess).astype(np.float64)
        out[name] = (num, den)
    n_atk = len(v.attack_family)
    if n_atk:
        is_atk = v.attack >= 0
        hit = np.zeros(n_atk, dtype=np.bool_)
        np.logical_or.at(hit, v.attack[is_atk], v.flagged[is_atk])
        atk_sess = np.full(n_atk, -1, dtype=np.int64)
        atk_sess[v.attack[is_atk]] = s[is_atk]  # an attack's rows share one session
        fam = np.array(v.attack_family)
        for f in sorted(set(v.attack_family)):
            m = fam == f
            num = np.bincount(atk_sess[m & hit], minlength=n_sess).astype(np.float64)
            den = np.bincount(atk_sess[m], minlength=n_sess).astype(np.float64)
            out[f"recall:{f}"] = (num, den)
    return out


def paired_deltas(
    a: Vectors, b: Vectors, *, n_boot: int = N_BOOT_DELTA, seed: int = 0, level: float = CI_LEVEL
) -> dict[str, dict[str, float | None]]:
    """``metric -> {a, b, delta, lo, hi}``: ``a - b`` for the decision benign FPR, the decision
    control FPR and each family's attack-level decision recall, with a paired percentile
    bootstrap over sessions (each resample draws the sessions with replacement and recomputes
    both rows' metrics on the same draw). ``ValueError`` when ``a`` and ``b`` do not cover the
    same rows."""
    if a.key != b.key:
        raise ValueError("vectors cover different rows")
    n_sess = int(a.session.max()) + 1 if a.session.size else 0
    pa, pb = _per_session(a, n_sess), _per_session(b, n_sess)
    rng = np.random.default_rng(seed)
    weights = np.stack(
        [np.bincount(rng.integers(0, n_sess, n_sess), minlength=n_sess) for _ in range(n_boot)]
    ).astype(np.float64)
    q = [100 * (1 - level) / 2, 100 * (1 + level) / 2]
    out: dict[str, dict[str, float | None]] = {}
    for name in pa:
        (na, da), (nb, db) = pa[name], pb[name]
        if da.sum() == 0:
            continue
        va, vb = float(na.sum() / da.sum()), float(nb.sum() / db.sum())
        with np.errstate(invalid="ignore", divide="ignore"):
            den = weights @ da
            d = (weights @ na - weights @ nb) / den
        d = d[den > 0]
        lo, hi = np.percentile(d, q) if d.size else (math.nan, math.nan)
        out[name] = {"a": va, "b": vb, "delta": va - vb, "lo": _f(lo), "hi": _f(hi)}
    return out


# -- latency --------------------------------------------------------------------------------------


def latency_ab(
    raws: Sequence[RawCall],
    hints: Mapping[str, ToolHint],
    rows: Sequence[BenchRow],
    *,
    n_calls: int,
    warmup: int,
) -> dict[str, Any]:
    """Alternated A/B ``score`` latency: the first ``n_calls`` calls in replay order, each scored
    by every row's engine in one process (the order rotated by one every call), then every
    engine updated with ``ALLOWED``; the first ``warmup`` calls are not timed. Per row: p50, p99
    and mean in microseconds, and p50 / p99 relative to the first row."""
    engines = [_engine(r, hints) for r in rows]
    calls = sorted(raws, key=lambda r: (r.ts_unix_ms, r.session_id, r.seq_no))[:n_calls]
    tracker = SessionTracker()
    lat: list[list[float]] = [[] for _ in engines]
    k = len(engines)
    t0 = time.time()
    for i, raw in enumerate(calls):
        e = normalize(raw, tracker)
        for j in range(k):
            idx = (i + j) % k
            s = perf_counter_ns()
            engines[idx].score(e)
            d = (perf_counter_ns() - s) / 1000.0
            if i >= warmup:
                lat[idx].append(d)
        for eng in engines:
            eng.update(e, Outcome.ALLOWED)
        tracker.observe(e)
    per: dict[str, dict[str, float]] = {}
    for r, xs in zip(rows, lat, strict=True):
        arr = np.asarray(xs)
        p50, p99 = np.percentile(arr, [50, 99]) if arr.size else (math.nan, math.nan)
        per[r.name] = {"p50_us": float(p50), "p99_us": float(p99), "mean_us": float(arr.mean())}
    ref = per[rows[0].name]
    for v in per.values():
        v["ratio_p50"] = v["p50_us"] / ref["p50_us"]
        v["ratio_p99"] = v["p99_us"] / ref["p99_us"]
    return {
        "method": "alternated A/B: every engine scores each call in one process, the order "
        "rotated by one per call; all updated with ALLOWED (spec allow-all); score() wall time",
        "n_calls": len(calls),
        "warmup_excluded": warmup,
        "n_timed": len(lat[0]),
        "rows": per,
        "wall_s": round(time.time() - t0, 1),
    }


# -- the run --------------------------------------------------------------------------------------


def _attribution(
    full: frozenset[str], without: Mapping[str, frozenset[str]], labels: dict[str, Label]
) -> dict[str, Any]:
    runs = AttributionRuns(full, tuple(without[g.name] for g in TRAJ_GROUPS))
    table = tabulate(runs, labels, TRAJ_GROUPS)
    union: set[str] = set()
    per_group: dict[str, Any] = {}
    for g, det in zip(TRAJ_GROUPS, runs.without, strict=True):
        carried, gained = full - det, det - full
        union |= carried
        per_group[g.name] = {
            "drop_detector": g.drop_detector,
            "cfg_off": dict(g.cfg_off),
            "carried": len(carried),
            "gained": len(gained),
            "detected_without": len(det),
            "changes_exactly_carried": not gained,
        }
    n_red = sum(v.get(REDUNDANT, 0) for v in table.values())
    return {
        "engine": f"{ATTRIBUTION_DETECTOR} (dataset schemas as hints)",
        "table": {f: dict(sorted(v.items())) for f, v in sorted(table.items())},
        "n_detected_full": len(full),
        "groups": per_group,
        # T-20's totals; ``tabulate`` guarantees them by construction (a consistency check, not
        # evidence). ``literal_exact`` is F10's real criterion (T-52).
        "t20_totals_by_construction": n_red + len(union) == len(full)
        and sum(sum(v.values()) for v in table.values()) >= len(full),
        "literal_exact": all(not p["gained"] for p in per_group.values()),
    }


def build_report(
    results: Sequence[JobResult],
    datasets: Mapping[int, Path],
    *,
    n_boot_delta: int,
    latency: Mapping[str, Any] | None,
    wall_s: Mapping[str, float],
) -> dict[str, Any]:
    """The JSON report from finished jobs (module docstring)."""
    by: dict[tuple[OutcomeMode, int], dict[str, JobResult]] = defaultdict(dict)
    groups: dict[tuple[OutcomeMode, int], dict[str, frozenset[str]]] = defaultdict(dict)
    for r in results:
        key = (r.job.outcomes, r.job.seed)
        if r.job.kind == "row":
            by[key][r.job.name] = r
        else:
            groups[key][r.job.name] = r.detected
    protocols: dict[str, Any] = {}
    for mode in PROTOCOLS:
        seeds: dict[str, Any] = {}
        for seed in sorted(datasets):
            rows = by.get((mode, seed))
            if not rows:
                continue
            ref = rows.get("engine")
            block: dict[str, Any] = {"rows": {}, "deltas": {}}
            for name in (r.name for r in ROWS):
                if name not in rows:
                    continue
                res = rows[name]
                s = dict(res.summary)
                if ref is not None:
                    s["scores_equal_engine"] = s["score_hash"] == ref.summary["score_hash"]
                    s["decisions_equal_engine"] = s["decision_hash"] == ref.summary["decision_hash"]
                s["wall_s"] = res.wall_s
                block["rows"][name] = s
                if ref is not None and name != "engine":
                    assert res.vectors is not None and ref.vectors is not None
                    block["deltas"][name] = paired_deltas(
                        res.vectors, ref.vectors, n_boot=n_boot_delta, seed=seed
                    )
            g = groups.get((mode, seed), {})
            traj = rows.get(ATTRIBUTION_DETECTOR)
            if traj is not None and len(g) == len(TRAJ_GROUPS):
                _, labels, _ = _load(str(datasets[seed]))
                block["attribution"] = _attribution(traj.detected, g, labels)
            seeds[str(seed)] = block
        protocols[mode] = {
            "label": PROTOCOL_LABELS[mode],
            "optimistic": mode != "allow",
            "seeds": seeds,
        }
    ds_meta = {}
    for seed, d in sorted(datasets.items()):
        man = _manifest(d)
        ds_meta[str(seed)] = {
            "dir": str(d),
            "traces_sha": man.files.get("traces.jsonl"),
            "counts": man.counts,
            "families": man.config.families,
            "is_bench_config": man.config == bench_config(seed),
        }
    return {
        "meta": {
            "git_sha": git_sha(),
            "created": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            "split": dataclasses.asdict(SPLIT),
            "protocols": {m: PROTOCOL_LABELS[m] for m in PROTOCOLS},
            "datasets": ds_meta,
            "rows": [
                {
                    "name": r.name,
                    "detector": r.detector,
                    "overrides": dict(r.overrides),
                    "switch": r.switch,
                    "note": r.note,
                }
                for r in ROWS
            ],
            "traj_switches": dict(TRAJ_SWITCHES),
            "attribution_groups": [
                {"name": g.name, "drop_detector": g.drop_detector, "cfg_off": dict(g.cfg_off)}
                for g in TRAJ_GROUPS
            ],
            "n_boot_delta": n_boot_delta,
            "delta_method": "paired percentile bootstrap over test-split sessions (row minus "
            f"engine, same resampled sessions for both), {int(CI_LEVEL * 100)}% interval",
            "theta_grid": list(THETA_GRID),
            "theta_source_row": THETA_SOURCE_ROW,
            "wall_s": dict(wall_s),
        },
        "protocols": protocols,
        "latency": dict(latency) if latency is not None else None,
    }


def selected_thetas(report: Mapping[str, Any]) -> tuple[float | None, float, str]:
    """``(theta_warn, theta_halt, where)`` of :data:`THETA_SOURCE_ROW`, spec protocol, first
    seed. ``KeyError`` when the report does not hold that row."""
    seeds = report["protocols"]["allow"]["seeds"]
    seed = min(seeds, key=int)
    sel = seeds[seed]["rows"][THETA_SOURCE_ROW]["sessions"]["halt"]["selected"]
    return sel["theta_warn"], sel["theta_halt"], f"{THETA_SOURCE_ROW}, allow, seed {seed}"


def write_calibrated(
    theta_warn: float | None,
    theta_halt: float,
    where: str,
    path: Path = CALIBRATED_PATH,
    *,
    note: str = "",
) -> None:
    """Record the selection in ``calibrated.json`` under keys that are no ``EngineConfig`` field
    (``theta_warn_selected_cal`` / ``theta_halt_selected_cal``; an undefined warn leaves its key
    out) with a history line ending in ``note``. A recorded value: no engine reads it, and the
    ``EngineConfig`` defaults are unchanged (T-03, T-54). ``ValueError`` when the file's
    normalizer hash is stale: the selection was measured with the current normalizer."""
    c = load_calibrated(path)
    if c.normalizer_hash != normalizer_hash():
        raise ValueError("calibrated.json is stale: restamp or recalibrate before recording")
    if theta_warn is None:
        c.constants.pop("theta_warn_selected_cal", None)
        warn = "theta_warn undefined (theta_halt is the grid floor)"
    else:
        c.constants["theta_warn_selected_cal"] = theta_warn
        warn = f"theta_warn_selected_cal={theta_warn}"
    c.constants["theta_halt_selected_cal"] = theta_halt
    line = (
        f"{datetime.date.today().isoformat()} Task 4.1: calibration-partition selection "
        f"theta_halt_selected_cal={theta_halt}, {warn} ({where}; benign false-halt <= 1%, warn "
        "rate <= 5%); recorded only, EngineConfig defaults unchanged"
    )
    c.history.append(f"{line}; {note}" if note else line)
    path.write_bytes(
        orjson.dumps(c.model_dump(), option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS) + b"\n"
    )


def rerender(report: dict[str, Any]) -> dict[str, Any]:
    """A saved report brought to this code's rules without replaying: each stored calibration
    curve re-selected with :func:`select_from_curve`, the pre-fix attribution key renamed, and
    ``meta.rendered_git_sha`` set (``meta.git_sha`` stays the run's)."""
    for p in report["protocols"].values():
        for block in p["seeds"].values():
            for row in block["rows"].values():
                h = row["sessions"]["halt"]
                if h is not None and h.get("calibration_curve"):
                    cc = h["calibration_curve"]
                    curve = [(c["theta"], c["benign_share"], c["n_benign"]) for c in cc]
                    warn, halt = select_from_curve(curve)
                    h["selected"] = {"theta_warn": warn, "theta_halt": halt}
            att = block.get("attribution")
            if att is not None and "totals_exact_t20" in att:
                att["t20_totals_by_construction"] = att.pop("totals_exact_t20")
    report["meta"]["rendered_git_sha"] = _code_sha()
    return report


def _code_sha() -> str:
    """``git_sha()``, suffixed ``-dirty`` when ``agentfw/`` has uncommitted changes (the code
    that ran is then not that commit's)."""
    sha = git_sha()
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--", "agentfw"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return sha
    return f"{sha}-dirty" if out.stdout.strip() else sha


def run_bench(
    traces: Path,
    seeds: Sequence[int],
    *,
    workers: int,
    rows: Sequence[str] | None = None,
    protocols: Sequence[OutcomeMode] = PROTOCOLS,
    attribution: bool = True,
    n_boot: int = 200,
    n_boot_delta: int = N_BOOT_DELTA,
    latency_calls: int = 40_000,
    latency_warmup: int = 10_000,
    progress: bool = False,
) -> dict[str, Any]:
    """Generate or reuse the datasets, replay every job, measure latency, build the report."""
    t0 = time.time()
    names = [r.name for r in ROWS] if rows is None else list(rows)
    unknown = [n for n in names if n not in _ROW_BY_NAME]
    if unknown or "engine" not in names:
        raise ValueError(f"rows must be ROWS names and include 'engine'; unknown: {unknown}")
    datasets = {s: ensure_dataset(traces, s) for s in seeds}
    t_gen = time.time() - t0
    jobs: list[Job] = []
    for seed, d in datasets.items():
        for mode in protocols:
            jobs += [Job("row", str(d), seed, mode, n, n_boot) for n in names]
            if attribution and ATTRIBUTION_DETECTOR in names:
                jobs += [Job("group", str(d), seed, mode, g.name) for g in TRAJ_GROUPS]
    t1 = time.time()
    results = run_jobs(jobs, workers, progress=progress)
    t_jobs = time.time() - t1
    latency = None
    t_lat = 0.0
    if latency_calls > 0:
        t2 = time.time()
        raws, _, hints = _load(str(datasets[min(datasets)]))
        lat_rows = [_ROW_BY_NAME[n] for n in names if n != "engine-traj-shadow"]
        latency = latency_ab(raws, hints, lat_rows, n_calls=latency_calls, warmup=latency_warmup)
        t_lat = time.time() - t2
    t3 = time.time()
    report = build_report(
        results,
        datasets,
        n_boot_delta=n_boot_delta,
        latency=latency,
        wall_s={},
    )
    report["meta"]["wall_s"] = {
        "datasets": round(t_gen, 1),
        "replays": round(t_jobs, 1),
        "latency": round(t_lat, 1),
        "report": round(time.time() - t3, 1),
        "total": round(time.time() - t0, 1),
        "n_jobs": len(jobs),
        "workers": workers,
    }
    return report


# -- CLI ------------------------------------------------------------------------------------------


def _json(report: Mapping[str, Any]) -> bytes:
    """Keys in build order (rows in ``ROWS`` order), not sorted."""
    return orjson.dumps(report, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m agentfw.bench.trajectory")
    p.add_argument("--traces", type=Path, help="dataset root (seed-N subdirs)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--rows", default=None, help="comma-separated ROWS names (default: all)")
    p.add_argument("--protocols", default=",".join(PROTOCOLS))
    p.add_argument("--no-attribution", action="store_true")
    p.add_argument("--n-boot", type=int, default=200, help="per-call (compute_all) bootstrap")
    p.add_argument("--n-boot-delta", type=int, default=N_BOOT_DELTA)
    p.add_argument("--latency-calls", type=int, default=40_000, help="0: skip latency")
    p.add_argument("--latency-warmup", type=int, default=10_000)
    p.add_argument(
        "--write-calibrated",
        action="store_true",
        help=f"record {THETA_SOURCE_ROW}'s calibration selection in calibrated.json",
    )
    p.add_argument("--calibrated-note", default="", help="appended to calibrated.json's history")
    p.add_argument(
        "--rerender",
        type=Path,
        default=None,
        help="re-render a saved trajectory_report.json with this code (no replay; T-57)",
    )
    a = p.parse_args(argv)
    if a.rerender is not None:
        report = rerender(orjson.loads(a.rerender.read_bytes()))
        _write(report, a.out)
        if a.write_calibrated:
            _calibrate(report, a.calibrated_note)
        return 0
    if a.traces is None:
        p.error("--traces is required (unless --rerender)")
    protocols: list[OutcomeMode] = []
    for m in a.protocols.split(","):
        if m not in PROTOCOLS:
            p.error(f"--protocols: {m!r} is not one of {PROTOCOLS}")
        protocols.append(m)
    report = run_bench(
        a.traces,
        [int(s) for s in a.seeds.split(",")],
        workers=a.workers,
        rows=None if a.rows is None else a.rows.split(","),
        protocols=protocols,
        attribution=not a.no_attribution,
        n_boot=a.n_boot,
        n_boot_delta=a.n_boot_delta,
        latency_calls=a.latency_calls,
        latency_warmup=a.latency_warmup,
        progress=True,
    )
    _write(report, a.out)
    if a.write_calibrated:
        _calibrate(report, a.calibrated_note)
    return 0


def _write(report: Mapping[str, Any], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / REPORT_JSON).write_bytes(_json(report) + b"\n")
    (out / REPORT_MD).write_text(render_markdown(report))
    print(f"wrote {out / REPORT_JSON} and {out / REPORT_MD}")


def _calibrate(report: Mapping[str, Any], note: str) -> None:
    warn, halt, where = selected_thetas(report)
    write_calibrated(warn, halt, where, note=note)
    print(f"calibrated.json: theta_warn={warn} theta_halt={halt} ({where})")


if __name__ == "__main__":
    raise SystemExit(main())
