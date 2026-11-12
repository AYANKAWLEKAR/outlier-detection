"""Engine parameter sweeps and the Open-Decision measurements (Task 1.14, spec §10).

:func:`sweep` replays the engine once per point of a cartesian grid over ``EngineConfig`` fields
and summarises each run with :func:`sweep_metrics`, all on **warm test** rows:

* ``auprc``, ``fpr95_per_agent`` and the ``per_agent_recall`` it was achieved at;
* the Tier-1 gate preview at the engine's **live** per-agent thresholds (``"row"``):
  ``recall:{family}`` (attack level, gate bucket), ``among:{family}`` (among-the-reasons share
  of the targeted detector, gate bucket, spec A1.6), ``share:{family}`` (top-reason share, the
  original spec §9 criterion), ``n:{family}``, ``benign_fpr``, ``control_fpr``,
  ``gate_min_recall``, ``gate_pass`` (the amended criterion, spec A1.7, under the run's one
  protocol); every :data:`GATE_FAMILIES` family gets keys (NaN / 0 when absent);
* ``offcal_*``: the same gate at an *offline* reference calibration (each agent's exact
  ``1 - cal_alpha`` quantile of its warm **train** benign scores) -- a diagnostic that separates
  detector quality from the streaming calibrator; not a deployable number;
* ``prereq_rule_recall`` / ``prereq_benign_rate``: how often the mined-prerequisite rule itself
  fires on ``prereq_skip`` attacks (attack level) and on benign rows (Open Decision 3);
* ``state_bytes``.

The long sweeps run outside pytest: ``python -m agentfw.bench.sweeps --traces DIR --out DIR``
(see :func:`main`), which also measures quantile tail error (Open Decision 1) and velocity
long-window memory (Open Decision 4), and writes ``sweeps.json`` + ``sweeps.md``. Suite ``gate``
(Task 1.15, decision 008) replays :data:`GATE_CANDIDATES`, each with its replay outcome mode and
the spec values of the three user-decision switches and of the three Amendment A1 scoring
switches pinned (:data:`SPEC_SWITCHES`), so it keeps reproducing decision 008's evidence; it
adds the live-threshold trajectory summary (the censoring ratchet). Every other suite replays the
current defaults under ``--outcomes`` (default: the benchmark protocol ``BENCH_OUTCOMES``, the
oracle-reviewer approval loop); :func:`sweep` takes the outcome mode explicitly.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import orjson
import yaml
from numpy.typing import ArrayLike

from agentfw.bench.ablation import (
    AblationRow,
    ablation_row,
    parallel_map,
    replay_detector,
)
from agentfw.bench.dataset import load_dir
from agentfw.bench.diagnostics import (
    RATCHET_DROP,
    SATURATED_THRESHOLD,
    AgentSaturation,
    SaturationReport,
    ThetaTrajectory,
    saturation_md,
    threshold_saturation,
    threshold_trajectory,
)
from agentfw.bench.metrics import GATE_FAMILIES, Thresholds, compute_all, gate_preview
from agentfw.bench.protocol import Detector
from agentfw.bench.replay import (
    BENCH_OUTCOMES,
    OUTCOME_MODES,
    PROTOCOL_LABELS,
    OutcomeMode,
    ScoreRow,
    SplitSpec,
)
from agentfw.core.config import EngineConfig
from agentfw.core.detectors.base import IQR_TO_SIGMA, signed_log1p
from agentfw.core.detectors.velocity import VelocityRecord
from agentfw.core.engine import DETECTOR_ORDER, Engine
from agentfw.core.normalize import SessionTracker, normalize
from agentfw.core.patterns import try_number
from agentfw.core.stats.p2 import P2Quantiles, p2_tail
from agentfw.synth.generate import generate

__all__ = [
    "GATE_CANDIDATES",
    "PRE_A1_SWITCHES",
    "SPEC_SWITCHES",
    "Z0TAU_NOTE",
    "SweepRow",
    "calibration_tail_error",
    "field_tail_errors",
    "grid_points",
    "main",
    "offline_thresholds",
    "quantile_tail_error",
    "sweep",
    "sweep_metrics",
    "velocity_memory",
]

Estimator = Literal["p2", "exact", "tdigest"]
_FIELDS = frozenset(f.name for f in dataclasses.fields(EngineConfig))


@dataclasses.dataclass(slots=True, frozen=True)
class SweepRow:
    params: dict[str, object]
    metrics: dict[str, float]


def grid_points(grid: Mapping[str, Sequence[object]]) -> list[dict[str, object]]:
    """Cartesian product of ``grid`` in key order, last key fastest; ``{}`` -> ``[{}]``."""
    keys = list(grid)
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*grid.values())]


# --- metrics ------------------------------------------------------------------------------------


def offline_thresholds(rows: Sequence[ScoreRow], alpha: float) -> dict[str, float]:
    """Per agent: exact ``1 - alpha`` inverted-CDF quantile of its warm train benign scores."""
    by: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if (
            r.split == "train"
            and r.is_warm
            and r.label != "attack"
            and not math.isnan(r.fused_score)
        ):
            by[r.agent_id].append(r.fused_score)
    return {
        a: float(np.quantile(np.asarray(v), 1.0 - alpha, method="inverted_cdf"))
        for a, v in by.items()
    }


def _prereq_fired(per_detector_json: str) -> bool:
    for item in orjson.loads(per_detector_json):
        if item.get("detector") == "sequence":
            ev = item.get("evidence") or {}
            return ev.get("prereq_missing") is not None
    return False


def _prereq_rates(warm_test: Sequence[ScoreRow]) -> dict[str, float]:
    benign = [r for r in warm_test if r.label != "attack"]
    attacks: dict[str, bool] = {}
    for r in warm_test:
        if r.label == "attack" and r.attack_family == "prereq_skip":
            key = r.attack_id or r.envelope_id
            attacks[key] = attacks.get(key, False) or _prereq_fired(r.per_detector_json)
    n_fire = sum(_prereq_fired(r.per_detector_json) for r in benign)
    return {
        "prereq_benign_rate": n_fire / len(benign) if benign else math.nan,
        "prereq_rule_recall": sum(attacks.values()) / len(attacks) if attacks else math.nan,
    }


def _gate_keys(prefix: str, rows: Sequence[ScoreRow], thresholds: Thresholds) -> dict[str, float]:
    g = gate_preview(rows, thresholds)
    out = {
        f"{prefix}benign_fpr": g.benign_fpr,
        f"{prefix}control_fpr": g.control_fpr,
        f"{prefix}gate_min_recall": g.min_recall,
        f"{prefix}gate_pass": float(g.passed),
    }
    for fam in GATE_FAMILIES:
        cell = g.cells.get(fam)
        out[f"{prefix}recall:{fam}"] = math.nan if cell is None else cell.recall
        out[f"{prefix}among:{fam}"] = math.nan if cell is None else cell.among_reasons_share
        out[f"{prefix}share:{fam}"] = math.nan if cell is None else cell.top_reason_share
        out[f"{prefix}n:{fam}"] = 0.0 if cell is None else float(cell.n_total)
    return out


def sweep_metrics(
    rows: Sequence[ScoreRow], *, state_bytes: int, cal_alpha: float, n_boot: int = 0
) -> dict[str, float]:
    """The per-setting summary described in the module docstring."""
    m = compute_all(rows, warm_only=True, breakdown_thresholds="row", n_boot=n_boot)
    warm_test = [
        r for r in rows if r.split == "test" and r.is_warm and not math.isnan(r.fused_score)
    ]
    out: dict[str, float] = {
        "auprc": m.auprc,
        "fpr95_per_agent": m.fpr95_per_agent,
        "per_agent_recall": m.per_agent_recall,
        "state_bytes": float(state_bytes),
    }
    out |= _gate_keys("", rows, "row")
    out |= _gate_keys("offcal_", rows, offline_thresholds(rows, cal_alpha))
    out |= _prereq_rates(warm_test)
    return out


# --- sweeps -------------------------------------------------------------------------------------


def _check_fields(keys: Sequence[str]) -> None:
    unknown = sorted(set(keys) - _FIELDS)
    if unknown:
        raise ValueError(f"unknown EngineConfig field(s): {unknown}")


_Job = tuple[str, str, SplitSpec, EngineConfig, int, OutcomeMode]


def _sweep_one(job: _Job) -> dict[str, float]:
    traces_dir, detector, split, cfg, n_boot, outcomes = job
    rows, det = replay_detector(traces_dir, detector, split, cfg, outcomes=outcomes)
    return sweep_metrics(
        rows, state_bytes=det.state_bytes(), cal_alpha=cfg.cal_alpha, n_boot=n_boot
    )


def sweep(
    traces_dir: str | Path,
    grid: Mapping[str, Sequence[object]],
    split: SplitSpec,
    *,
    outcomes: OutcomeMode,
    detector: str = "engine",
    base: EngineConfig | None = None,
    n_boot: int = 0,
    workers: int = 1,
) -> list[SweepRow]:
    """One :class:`SweepRow` per grid point (order of :func:`grid_points`), each replayed with
    the caller's ``outcomes`` (the replay protocol; no default).

    Each point is ``base.replace(**point)`` (default ``EngineConfig()``); grid keys that are not
    ``EngineConfig`` fields raise ``ValueError`` before anything is replayed.
    """
    _check_fields(list(grid))
    base = EngineConfig() if base is None else base
    points = grid_points(grid)
    jobs = [(str(traces_dir), detector, split, base.replace(**p), n_boot, outcomes) for p in points]
    results = parallel_map(_sweep_one, jobs, workers)
    return [SweepRow(p, m) for p, m in zip(points, results, strict=True)]


# --- Open Decision 1: quantile tail error -------------------------------------------------------


def quantile_tail_error(
    samples: ArrayLike, probs: Sequence[float], estimator: Estimator
) -> dict[float, float]:
    """``|estimate - exact|`` per probability, in the samples' units.

    ``exact`` is ``numpy.quantile(..., method="linear")`` (so the ``"exact"`` estimator's error is
    0). ``"p2"`` streams the samples in order through :class:`P2Quantiles` (the engine's
    estimator); ``"tdigest"`` uses the optional ``tdigest`` package (``ImportError`` if absent).
    """
    x = np.asarray(samples, dtype=np.float64)
    ps = [float(p) for p in probs]
    exact = {p: float(np.quantile(x, p)) for p in ps}
    if estimator == "exact":
        est = {p: float(np.quantile(x, p)) for p in ps}
    elif estimator == "p2":
        q = P2Quantiles(tuple(ps))
        for v in x.tolist():
            q.add(v)
        est = {p: q.quantile(p) for p in ps}
    elif estimator == "tdigest":
        from tdigest import TDigest  # type: ignore[import-not-found]  # optional

        d = TDigest()
        d.batch_update(x)
        est = {p: float(d.percentile(100.0 * p)) for p in ps}
    else:
        raise ValueError(f"unknown estimator {estimator!r}")
    return {p: abs(est[p] - exact[p]) for p in ps}


def field_tail_errors(
    traces_dir: str | Path,
    probs: Sequence[float],
    estimator: Estimator = "p2",
    *,
    min_n: int = 200,
    eps: float = EngineConfig().mag_eps,
) -> list[dict[str, Any]]:
    """Per numeric (agent, tool, path) key with ``>= min_n`` values: estimator error per prob.

    Values are the normalized flat/derived fields of the replay order, transformed by
    ``signed_log1p`` as magnitude does. A key is numeric when >= 95 % of its values parse as
    finite numbers. Errors are reported raw and in robust-sigma units,
    ``err / max(IQR / 1.349, eps)`` -- the unit of magnitude's z (exact IQR of the key).
    """
    raws, _, _ = load_dir(traces_dir)
    tracker = SessionTracker()
    vals: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    seen: dict[tuple[str, str, str], int] = defaultdict(int)
    for r in sorted(raws, key=lambda r: (r.ts_unix_ms, r.session_id, r.seq_no)):
        env = normalize(r, tracker)
        tracker.observe(env)
        for src in (env.flat_args, env.derived):
            for path, values in src.items():
                key = (env.agent_id, env.tool_id, path)
                for v in values:
                    seen[key] += 1
                    x = try_number(v)
                    if x is not None and math.isfinite(x):
                        vals[key].append(signed_log1p(x))
    out: list[dict[str, Any]] = []
    for key, ys in sorted(vals.items()):
        if len(ys) < min_n or len(ys) < 0.95 * seen[key]:
            continue
        arr = np.asarray(ys)
        q25, q75 = np.quantile(arr, [0.25, 0.75])
        scale = max(float(q75 - q25) / IQR_TO_SIGMA, eps)
        err = quantile_tail_error(arr, probs, estimator)
        out.append(
            {
                "agent": key[0],
                "tool": key[1],
                "path": key[2],
                "n": len(ys),
                "scale": scale,
                "err": err,
                "err_sigma": {p: e / scale for p, e in err.items()},
            }
        )
    return out


def calibration_tail_error(rows: Sequence[ScoreRow], alpha: float) -> dict[str, float]:
    """Per agent: |P² tail estimate - exact| of the ``1 - alpha`` quantile of its fused scores,
    streamed in replay order exactly as the calibrator sees them (every replayed row is allowed)."""
    by: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if not math.isnan(r.fused_score):
            by[r.agent_id].append(r.fused_score)
    p = 1.0 - alpha
    out: dict[str, float] = {}
    for agent, v in sorted(by.items()):
        q = p2_tail(p)
        for s in v:
            q.add(s)
        out[agent] = abs(q.quantile(p) - float(np.quantile(np.asarray(v), p)))
    return out


# --- Open Decision 4: velocity long-window memory -----------------------------------------------


def velocity_memory(det: Detector) -> dict[str, float]:
    """JSON bytes of every velocity record as stored (sparse long window, D-010) versus the same
    record with a dense ``vel_long_min``-bucket ring and with no long window at all; plus the
    binary sizes (dense: 16 B/bucket; sparse: 24 B per active minute)."""
    if not isinstance(det, Engine):
        return {}
    w = det.cfg.vel_long_min
    opts = orjson.OPT_SORT_KEYS
    n = sparse = dense = none = active = 0
    for _, rec in det.store.items():
        if not isinstance(rec, VelocityRecord):
            continue
        d = rec.to_dict()
        n += 1
        active += len(rec.long)
        sparse += len(orjson.dumps(d, option=opts))
        bare = {**d, "long_ago": [], "long_count": []}
        if rec.money:
            bare["long_sum"] = []
        none += len(orjson.dumps(bare, option=opts))
        ring_c = [0] * w
        ring_s = [0.0] * w
        for m, (c, s) in rec.long.items():
            ring_c[m % w], ring_s[m % w] = c, s
        full: dict[str, object] = {**bare, "long_count": ring_c}
        full.pop("long_ago")
        if rec.money:
            full["long_sum"] = ring_s
        dense += len(orjson.dumps(full, option=opts))
    return {
        "vel_keys": float(n),
        "vel_json_sparse": float(sparse),
        "vel_json_dense": float(dense),
        "vel_json_no_long": float(none),
        "vel_bin_sparse": float(24 * active),
        "vel_bin_dense": float(16 * w * n),
        "vel_active_minutes_mean": active / n if n else math.nan,
    }


# --- CLI ----------------------------------------------------------------------------------------

Z0_TAU: dict[str, list[object]] = {"z0": [2.5, 3.0, 3.5], "tau": [0.5, 0.75, 1.0]}
#: Printed under the ``z0tau`` suite (spec A1.2).
Z0TAU_NOTE = (
    "z0 and tau are read by velocity (both signals) and by mag_score = z (spec §6.3). At the "
    "defaults (mag_score = tail, Amendment A1.2) this suite moves velocity only."
)
CAL_ALPHAS = [0.005, 0.01]
SUITE_GRIDS: dict[str, dict[str, list[object]]] = {
    "z0tau": Z0_TAU,
    "direction": {"mag_direction_default": ["both", "up"]},
    "alpha": {"cal_alpha": list(CAL_ALPHAS)},
    "velsum": {"vel_sum_signal": [True, False]},
    "resp": {"mag_resp_signal": [True, False]},
    "vel_long": {"vel_long_min": [1440, 0]},
}
PREREQ_RATES = [0.90, 0.95, 0.98]
SKIP_LOOKUPS = [0.10, 0.05, 0.02]
ABLATION_VARIANTS = [
    "engine",
    *(f"engine-no-{d}" for d in DETECTOR_ORDER),
    *(f"engine-only-{d}" for d in DETECTOR_ORDER),
    "engine-no-priors",
]
#: The scoring of the engine before Amendment A1 (spec A1.9): §6.4 count z, §6.3 magnitude z,
#: BOOL-typed fields skipped by magnitude. On top of the defaults these alone reproduce the
#: pre-amendment engine.
PRE_A1_SWITCHES: dict[str, object] = {
    "vel_count_stat": "logz",
    "mag_score": "z",
    "mag_bool_numeric": False,
}
#: Spec values of the three switches the user turned on by default (decision 008, user decision
#: 2026-11-10) and of the Amendment A1 scoring (:data:`PRE_A1_SWITCHES`). Every gate candidate
#: starts from them, so the suite keeps decision 008's meaning.
SPEC_SWITCHES: dict[str, object] = {
    "vel_count_attempts": False,
    "seq_prereq_graded": False,
    "novelty_full_n": 200,
    **PRE_A1_SWITCHES,
}
#: Task 1.15 gate candidates (docs/decisions/008-tier1-gate.md): ``(setting, params, outcomes)``,
#: params on top of :data:`SPEC_SWITCHES`. "default" is the spec engine under the spec's
#: allow-all replay. "oracle-reviewer" rows use replay outcomes "review": the oracle-reviewer
#: approval loop (labels act as a perfect human reviewer → optimistic). ``oracle`` reads every
#: label (attacks never learned): a diagnostic upper bound only. The last oracle-reviewer row
#: (all three switches) is the benchmark protocol and the engine defaults of the user decision,
#: i.e. the engine before Amendment A1 (its scoring switches stay at :data:`PRE_A1_SWITCHES`).
REVIEW_NOTE = (
    f"oracle-reviewer = {PROTOCOL_LABELS['review']}; oracle = no call labelled attack is ever "
    f'learned (diagnostic). "default" is the spec engine under {PROTOCOL_LABELS["allow"]}. The '
    "benchmark protocol and engine defaults of the user decision (2026-11-10) are the row "
    '"oracle-reviewer, vel attempts, graded prereq, novelty_full_n=50": the engine before '
    "Amendment A1 (every row keeps the pre-amendment scoring)."
)
_W: dict[str, object] = {"cal_censor": "winsorize"}
_W500: dict[str, object] = {**_W, "cal_censor_min_n": 500}
_ATT: dict[str, object] = {"vel_count_attempts": True}
_GRADED: dict[str, object] = {"seq_prereq_graded": True}
_NOV50: dict[str, object] = {"novelty_full_n": 50}
_GATE_CANDIDATES: list[tuple[str, dict[str, object], OutcomeMode]] = [
    ("default", {}, "allow"),
    ("enforce", {}, "enforce"),
    ("cal_censor=drop", {"cal_censor": "drop"}, "allow"),
    ("cal_censor=winsorize", dict(_W), "allow"),
    ("enforce, winsorize", dict(_W), "enforce"),
    ("winsorize after 500", dict(_W500), "allow"),
    ("enforce, winsorize after 500", dict(_W500), "enforce"),
    (
        "enforce, winsorize after 500, attempts, graded, nov50",
        {**_W500, **_ATT, **_GRADED, **_NOV50},
        "enforce",
    ),
    ("oracle-reviewer", {}, "review"),
    ("oracle-reviewer, graded prereq", dict(_GRADED), "review"),
    ("oracle-reviewer, novelty_full_n=50", dict(_NOV50), "review"),
    ("oracle-reviewer, vel attempts", dict(_ATT), "review"),
    ("oracle-reviewer, vel attempts, graded prereq", {**_ATT, **_GRADED}, "review"),
    (
        "oracle-reviewer, vel attempts, graded prereq, novelty_full_n=50",
        {**_ATT, **_GRADED, **_NOV50},
        "review",
    ),
    ("oracle (labels; diagnostic)", {}, "oracle"),
]
GATE_CANDIDATES: list[tuple[str, dict[str, object], OutcomeMode]] = [
    (name, {**SPEC_SWITCHES, **params}, mode) for name, params, mode in _GATE_CANDIDATES
]
ALL_SUITES = [*SUITE_GRIDS, "prereq", "ablation", "tail", "saturation", "gate"]


def skip_dataset(base_dir: Path, skip: float) -> Path:
    """Regenerate ``base_dir``'s dataset with every persona's ``noise.skip_lookup = skip``.

    Written next to it as ``{base_dir}-skip{skip}`` (personas copied into its ``personas/``);
    reused when its manifest already matches."""
    _, _, manifest = load_dir(base_dir)
    out = base_dir.with_name(f"{base_dir.name}-skip{skip:.2f}")
    pdir = out / "personas"
    cfg = manifest.config.model_copy(update={"personas_dir": pdir})
    if (out / "manifest.json").is_file():
        _, _, have = load_dir(out)
        if have.config == cfg:
            return out
    pdir.mkdir(parents=True, exist_ok=True)
    for f in sorted(Path(manifest.config.personas_dir).glob("*.yaml")):
        d = yaml.safe_load(f.read_text())
        d["noise"] = {**(d.get("noise") or {}), "skip_lookup": skip}
        (pdir / f.name).write_text(yaml.safe_dump(d, sort_keys=False))
    generate(cfg, out)
    return out


@dataclasses.dataclass(slots=True, frozen=True)
class _Spec:
    traces: str
    detector: str
    params: tuple[tuple[str, object], ...]
    split: SplitSpec
    n_boot: int
    outcomes: OutcomeMode


def _run_spec(spec: _Spec) -> dict[str, Any]:
    cfg = EngineConfig().replace(**dict(spec.params))
    rows, det = replay_detector(spec.traces, spec.detector, spec.split, cfg, outcomes=spec.outcomes)
    out: dict[str, Any] = {
        "metrics": sweep_metrics(
            rows, state_bytes=det.state_bytes(), cal_alpha=cfg.cal_alpha, n_boot=spec.n_boot
        ),
        "ablation": dataclasses.asdict(
            ablation_row(spec.detector, rows, det.state_bytes(), n_boot=spec.n_boot)
        ),
        "velocity_memory": velocity_memory(det),
        "calibration_tail": {str(a): calibration_tail_error(rows, a) for a in CAL_ALPHAS},
        "saturation": dataclasses.asdict(threshold_saturation(rows)),
        "trajectory": {a: dataclasses.asdict(t) for a, t in threshold_trajectory(rows).items()},
    }
    return out


def _fmt(x: object, nd: int = 3) -> str:
    if isinstance(x, bool) or not isinstance(x, int | float) or math.isnan(x):
        return "n/a"
    return f"{x:.{nd}f}"


def gate_table(rows: Sequence[tuple[str, Mapping[str, float]]], *, offcal: bool = True) -> str:
    """Markdown table: one line per setting with the Tier-1 gate preview.

    Per family: attack recall (among-the-reasons share; top-1 share, the original spec §9
    criterion). ``pass`` is the amended criterion under that setting's one protocol.
    """
    fams = list(GATE_FAMILIES)
    head = ["setting", "AUPRC", "FPR95 pa (recall)", "benign FPR", "control FPR"]
    head += [f"{f} {GATE_FAMILIES[f][0]} rec (among; top-1)" for f in fams]
    head += ["min rec", "pass"]
    if offcal:
        head += ["offcal min rec", "offcal benign FPR", "offcal control FPR"]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for name, m in rows:
        cells = [
            name,
            _fmt(m["auprc"]),
            f"{_fmt(m['fpr95_per_agent'])} ({_fmt(m['per_agent_recall'])})",
            _fmt(m["benign_fpr"], 4),
            _fmt(m["control_fpr"]),
        ]
        cells += [
            f"{_fmt(m[f'recall:{f}'])} ({_fmt(m[f'among:{f}'], 2)}; {_fmt(m[f'share:{f}'], 2)}) "
            f"n={int(m[f'n:{f}'])}"
            for f in fams
        ]
        cells += [_fmt(m["gate_min_recall"]), "yes" if m["gate_pass"] else "no"]
        if offcal:
            cells += [
                _fmt(m["offcal_gate_min_recall"]),
                _fmt(m["offcal_benign_fpr"], 4),
                _fmt(m["offcal_control_fpr"]),
            ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _label(params: Mapping[str, object]) -> str:
    return ", ".join(f"{k}={v}" for k, v in params.items()) or "default"


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m agentfw.bench.sweeps")
    p.add_argument("--traces", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--suites", default="all", help=f"comma list of {ALL_SUITES} or 'all'")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--n-boot", type=int, default=0)
    p.add_argument(
        "--outcomes",
        choices=OUTCOME_MODES,
        default=BENCH_OUTCOMES,
        help="replay protocol of every suite except gate (whose candidates carry their own); "
        "default: the benchmark protocol (oracle-reviewer approval loop)",
    )
    a = p.parse_args(argv)
    suites = ALL_SUITES if a.suites == "all" else [s for s in a.suites.split(",") if s]
    unknown = sorted(set(suites) - set(ALL_SUITES))
    if unknown:
        p.error(f"unknown suites {unknown}")
    split = SplitSpec()
    base = str(a.traces)

    def spec(
        params: Mapping[str, object],
        traces: str = base,
        det: str = "engine",
        outcomes: OutcomeMode | None = None,
    ) -> _Spec:
        mode: OutcomeMode = a.outcomes if outcomes is None else outcomes
        return _Spec(traces, det, tuple(params.items()), split, a.n_boot, mode)

    plan: dict[str, list[tuple[str, _Spec]]] = {}
    for s in suites:
        if s in SUITE_GRIDS:
            plan[s] = [(_label(pt), spec(pt)) for pt in grid_points(SUITE_GRIDS[s])]
        elif s == "ablation":
            plan[s] = [(v, spec({}, det=v)) for v in ABLATION_VARIANTS]
        elif s == "prereq":
            items: list[tuple[str, _Spec]] = []
            for skip in [None, *SKIP_LOOKUPS]:
                traces = base if skip is None else str(skip_dataset(a.traces, skip))
                tag = "persona skip_lookup" if skip is None else f"skip_lookup={skip}"
                for rate in PREREQ_RATES:
                    items.append(
                        (f"{tag}, prereq_rate={rate}", spec({"prereq_rate": rate}, traces))
                    )
            plan[s] = items
        elif s in ("tail", "saturation"):
            plan[s] = [("default", spec({}))]
        elif s == "gate":
            plan[s] = [(n, spec(pt, outcomes=o)) for n, pt, o in GATE_CANDIDATES]

    # Run each distinct replay once (the default point appears in several suites).
    jobs = list(dict.fromkeys(sp for items in plan.values() for _, sp in items))
    results = dict(zip(jobs, parallel_map(_run_spec, jobs, a.workers), strict=True))

    report: dict[str, Any] = {
        "traces": base,
        "split": dataclasses.asdict(split),
        "outcomes": a.outcomes,
        "suites": {},
    }
    md: list[str] = [
        f"# Sweeps on `{base}`",
        "",
        f"Protocol (every suite except gate): {PROTOCOL_LABELS[a.outcomes]}.",
        "",
    ]
    for s, items in plan.items():
        entries = [
            {
                "setting": name,
                "params": dict(sp.params),
                "outcomes": sp.outcomes,
                "traces": sp.traces,
                **results[sp],
            }
            for name, sp in items
        ]
        report["suites"][s] = entries
        md += [f"## {s}", ""]
        if s == "ablation":
            md += _ablation_md([AblationRow(**e["ablation"]) for e in entries])
        else:
            md.append(gate_table([(e["setting"], e["metrics"]) for e in entries]))
        if s == "prereq":
            md += [
                "",
                "| setting | prereq rule recall | prereq benign fire rate |",
                "|---|---|---|",
            ]
            md += [
                f"| {e['setting']} | {_fmt(e['metrics']['prereq_rule_recall'])} "
                f"| {_fmt(e['metrics']['prereq_benign_rate'], 4)} |"
                for e in entries
            ]
        if s == "vel_long":
            md += ["", "| setting | velocity keys | JSON sparse | JSON dense | JSON no long |"]
            md += ["|---|---|---|---|---|"]
            for e in entries:
                vm = e["velocity_memory"]
                md.append(
                    f"| {e['setting']} | {int(vm['vel_keys'])} | {int(vm['vel_json_sparse'])} "
                    f"| {int(vm['vel_json_dense'])} | {int(vm['vel_json_no_long'])} |"
                )
        if s == "z0tau":
            md += ["", Z0TAU_NOTE]
        if s == "gate":
            md += ["", REVIEW_NOTE, "", *_trajectory_summary_md(entries)]
        if s == "saturation":
            rep = entries[0]["saturation"]
            md += ["", *saturation_md(_saturation_from_dict(rep))]
        if s == "tail":
            fte = field_tail_errors(a.traces, EngineConfig().mag_probs)
            report["field_tail"] = fte
            md += ["", *_tail_md(fte, entries[0]["calibration_tail"])]
        md.append("")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "sweeps.json").write_bytes(
        orjson.dumps(report, option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS)
    )
    (a.out / "sweeps.md").write_text("\n".join(md))
    print(a.out / "sweeps.md")
    return 0


def _trajectory_summary_md(entries: Sequence[Mapping[str, Any]]) -> list[str]:
    """One line per candidate: where the live thresholds end up, and whether they ratchet."""
    out = [
        f"Live per-agent thresholds (agents ratcheting = threshold fell by > {RATCHET_DROP} from "
        "the end of train to the end of test; saturated = final threshold >= "
        f"{SATURATED_THRESHOLD}):",
        "",
        "| setting | outcomes | median end of train | median end | median drift | min end "
        "| agents ratcheting | agents saturated |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for e in entries:
        tr = [ThetaTrajectory(**{**t, "test_quarters": tuple(t["test_quarters"])})
              for t in e["trajectory"].values()]  # fmt: skip
        if not tr:
            continue
        ends = np.asarray([t.end for t in tr])
        drift = np.asarray([t.drift for t in tr])
        end_train = float(np.nanmedian([t.end_train for t in tr]))
        out.append(
            f"| {e['setting']} | {e['outcomes']} | {_fmt(end_train)} "
            f"| {_fmt(float(np.median(ends)))} | {_fmt(float(np.nanmedian(drift)))} "
            f"| {_fmt(float(ends.min()))} | {int(np.sum(drift < -RATCHET_DROP))} of {len(tr)} "
            f"| {int(np.sum(ends >= SATURATED_THRESHOLD))} |"
        )
    return out


def _saturation_from_dict(d: Mapping[str, Any]) -> SaturationReport:
    agents = {k: AgentSaturation(**v) for k, v in d["agents"].items()}
    return SaturationReport(agents, int(d["n_saturated"]), dict(d["top1"]))


def _ablation_md(rows: Sequence[AblationRow]) -> list[str]:
    fams = sorted({f for r in rows for f in r.per_family_recall})
    head = ["variant", "AUPRC", "FPR95 per-agent", "per-agent recall", *fams, "state bytes"]
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for r in rows:
        cells = [r.variant, _fmt(r.auprc), _fmt(r.fpr95_per_agent), _fmt(r.per_agent_recall)]
        cells += [_fmt(r.per_family_recall.get(f, math.nan)) for f in fams]
        cells.append(str(r.state_bytes))
        out.append("| " + " | ".join(cells) + " |")
    return out


def _tail_md(fte: Sequence[Mapping[str, Any]], cal: Mapping[str, Mapping[str, float]]) -> list[str]:
    probs = sorted({p for e in fte for p in e["err_sigma"]})
    out = [
        f"P² error per numeric key ({len(fte)} keys with n >= 200), robust-sigma units "
        "(|P² - exact| / max(IQR/1.349, eps)):",
        "",
        "| q | median | p90 | max |",
        "|---|---|---|---|",
    ]
    for p in probs:
        v = np.asarray([e["err_sigma"][p] for e in fte])
        out.append(f"| {p:g} | {np.median(v):.4f} | {np.percentile(v, 90):.4f} | {v.max():.4f} |")
    out += ["", "Worst keys per quantile (robust-sigma error; scale = max(IQR/1.349, eps)):", ""]
    out += [
        "| q | agent | tool | path | n | scale | error (sigma) |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in probs:
        for e in sorted(fte, key=lambda e: -e["err_sigma"][p])[:3]:
            out.append(
                f"| {p:g} | {e['agent']} | {e['tool']} | {e['path']} | {e['n']} "
                f"| {e['scale']:.3f} | {e['err_sigma'][p]:.3f} |"
            )
    out += ["", "Calibrator P² tail (fused-score units), per agent:", ""]
    out += ["| alpha | median | max |", "|---|---|---|"]
    for alpha, per in cal.items():
        v = np.asarray(list(per.values()))
        out.append(f"| {alpha} | {np.median(v):.4f} | {v.max():.4f} |")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
