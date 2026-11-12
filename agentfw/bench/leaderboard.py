"""Leaderboard over ``report.json`` files of one dataset and one split (spec 5.4-5.6).

Rows are ranked by the warm-only per-agent FPR@95% recall (the deployable number, spec 5.4)
ascending, then AUPRC descending. A per-agent FPR is NaN when 95 % recall is unreachable; such
rows rank last and render as ``n/a (recall X)`` with the recall that was achieved. Runs whose
numbers are not comparable with a full online replay carry a caveat: batch fits, stratified
samples, and the replay outcome mode (``extras.outcomes``, labels from
``agentfw.bench.replay.PROTOCOL_LABELS``): the benchmark protocol, the oracle-reviewer loop, and
the label-reading oracle are always labelled optimistic; spec §5.2 allow-all is labelled as the
comparison protocol. Reports without ``extras.outcomes`` (older runs) carry no protocol caveat.
By-time and by-agent splits are never mixed in one table (spec 5.3).

:func:`render_gate_section` prints the Tier-1 gate of the engine runs' ``report.json["gate"]``
under both protocols side by side (spec A1.5): ``engine`` with ``engine-allow-all``, then the
engine before Amendment A1 (``engine-pre-a1`` with ``engine-pre-a1-allow-all``).
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import orjson
from pydantic import BaseModel

from agentfw.bench.gate import ENGINE_ROW, PRE_A1_ROW, preview_from_json, render_gate_matrix
from agentfw.bench.metrics import GatePreview
from agentfw.bench.replay import PROTOCOL_LABELS, OutcomeMode, protocol_label

__all__ = [
    "ALLOW_ALL_CAVEAT",
    "GATE_RUNS",
    "ORACLE_CAVEAT",
    "ORACLE_DIAGNOSTIC_CAVEAT",
    "Leaderboard",
    "LeaderboardRow",
    "build_leaderboard",
    "main",
    "render_gate_section",
    "render_leaderboard",
]

ORACLE_CAVEAT = PROTOCOL_LABELS["review"]
ORACLE_DIAGNOSTIC_CAVEAT = PROTOCOL_LABELS["oracle"]
ALLOW_ALL_CAVEAT = PROTOCOL_LABELS["allow"]
BATCH_CAVEAT = "batch: fit offline on the train split, not an online learner"
#: Gate rows of :func:`render_gate_section`: row name -> (oracle-reviewer run, allow-all run).
#: The run names are those of ``scripts/build_leaderboard.py``; the row names are the engine rows
#: of :func:`agentfw.bench.gate.render_gate_matrix` (the pre-A1 row is a companion, rule 4).
GATE_RUNS: dict[str, tuple[str, str]] = {
    ENGINE_ROW: ("engine", "engine-allow-all"),
    PRE_A1_ROW: ("engine-pre-a1", "engine-pre-a1-allow-all"),
}
_GATE_MODES: tuple[OutcomeMode, OutcomeMode] = ("review", "allow")


class LeaderboardRow(BaseModel):
    name: str  # the run directory's name, e.g. "engine-sample"
    detector: str
    auprc: float | None
    auprc_ci: tuple[float | None, float | None]
    fpr95_per_agent: float | None
    fpr95_per_agent_ci: tuple[float | None, float | None]
    per_agent_recall: float | None
    fpr95_global: float | None
    n_rows: int
    n_attack_rows: int
    family_mode: str  # thresholds of the per-family breakdown ("per_agent" or "row")
    family_recall: dict[str, float | None]  # warm attack-level recall, bucket "all"
    control_fpr: float | None  # warm benign_unusual FPR at the same thresholds
    caveats: list[str]
    git_sha: str
    n_boot: int | None


class Leaderboard(BaseModel):
    label: str
    dataset_sha: str
    split: str
    seed: int
    rows: list[LeaderboardRow]  # ranked
    not_run: dict[str, str]  # name -> reason


def _num(x: object) -> float | None:
    return float(x) if isinstance(x, int | float) and math.isfinite(x) else None


def _pair(x: object) -> tuple[float | None, float | None]:
    if isinstance(x, list | tuple) and len(x) == 2:
        return _num(x[0]), _num(x[1])
    return None, None


def _caveats(extras: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    if extras.get("batch"):
        out.append(BATCH_CAVEAT)
    sample = extras.get("sample")
    if isinstance(sample, Mapping):
        out.append(
            f"sampled ({sample.get('spec')}): attack-enriched test subsample, "
            "AUPRC not comparable with a full replay"
        )
    outcomes = extras.get("outcomes")
    if outcomes is not None:
        out.append(protocol_label(outcomes))
    return out


def _family_recall(per_family: Mapping[str, Any]) -> dict[str, float | None]:
    cells = per_family.get("warm_attack") or {}
    return {fam: _num((buckets.get("all") or {}).get("recall")) for fam, buckets in cells.items()}


def _row(path: Path, rep: Mapping[str, Any]) -> LeaderboardRow:
    h, meta, ex = rep["headline"], rep["meta"], rep.get("extras") or {}
    pf = rep.get("per_family") or {}
    n_boot = ex.get("n_boot")
    return LeaderboardRow(
        name=path.parent.name,
        detector=str(meta["detector"]),
        auprc=_num(h.get("auprc")),
        auprc_ci=_pair(h.get("auprc_ci")),
        fpr95_per_agent=_num(h.get("fpr95_per_agent")),
        fpr95_per_agent_ci=_pair(h.get("fpr95_per_agent_ci")),
        per_agent_recall=_num(h.get("per_agent_recall")),
        fpr95_global=_num(h.get("fpr95_global")),
        n_rows=int(h.get("n_rows") or 0),
        n_attack_rows=int(h.get("n_attack_rows") or 0),
        family_mode=str(pf.get("mode", "n/a")),
        family_recall=dict(sorted(_family_recall(pf).items())),
        control_fpr=_num((rep.get("control_fpr") or {}).get("warm")),
        caveats=_caveats(ex),
        git_sha=str(meta.get("git_sha", "")),
        n_boot=n_boot if isinstance(n_boot, int) else None,
    )


def _rank_key(r: LeaderboardRow) -> tuple[bool, float, bool, float, str]:
    fpr, auprc = r.fpr95_per_agent, r.auprc
    return (
        fpr is None,
        fpr if fpr is not None else 0.0,
        auprc is None,
        -auprc if auprc is not None else 0.0,
        r.name,
    )


def build_leaderboard(
    report_paths: Sequence[Path],
    *,
    label: str = "",
    not_run: Mapping[str, str] | None = None,
) -> Leaderboard:
    """Rank the given ``report.json`` files; raises ``ValueError`` on an empty input or when
    the reports mix ``dataset_sha`` or split kinds. ``not_run`` lists detectors that belong in
    the table but have no report (name -> reason); they are rendered unranked."""
    if not report_paths:
        raise ValueError("no report.json given")
    reps = [(Path(p), orjson.loads(Path(p).read_bytes())) for p in report_paths]
    shas = {str(r["meta"]["dataset_sha"]) for _, r in reps}
    if len(shas) != 1:
        raise ValueError(f"reports mix dataset_sha values: {sorted(s[:12] for s in shas)}")
    splits = {str(r["meta"]["split"]["kind"]) for _, r in reps}
    if len(splits) != 1:
        raise ValueError(f"reports mix split kinds {sorted(splits)}; tabulate them separately")
    return Leaderboard(
        label=label,
        dataset_sha=shas.pop(),
        split=splits.pop(),
        seed=int(reps[0][1]["meta"]["seed"]),  # the dataset's seed; one dataset_sha, one seed
        rows=sorted((_row(p, r) for p, r in reps), key=_rank_key),
        not_run=dict(not_run or {}),
    )


# --- markdown -----------------------------------------------------------------------------------


def _f(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _ci(x: tuple[float | None, float | None]) -> str:
    return "n/a" if None in x else f"[{_f(x[0])}, {_f(x[1])}]"


def _fpr_cell(r: LeaderboardRow) -> str:
    if r.per_agent_recall is None:
        return _f(r.fpr95_per_agent)
    return f"{_f(r.fpr95_per_agent)} (recall {_f(r.per_agent_recall)})"


def render_leaderboard(lb: Leaderboard) -> str:
    lines = [
        f"#### Leaderboard: {_title(lb)}",
        "",
        f"Dataset `{lb.dataset_sha[:12]}` (sha256 of traces.jsonl), seed {lb.seed}. Warm test "
        "rows. Ranked by per-agent FPR@95% recall ascending, then AUPRC descending; a per-agent "
        "FPR is shown with the recall it achieved, and is `n/a` when 95 % recall is unreachable "
        "(ranked last). CIs are 95 % session-bootstrap intervals.",
        "",
        "| Rank | Run | Detector | AUPRC | AUPRC 95% CI | FPR@95 per-agent (recall) "
        "| FPR@95 per-agent 95% CI | FPR@95 global | Test rows (attacks) | Caveats |",
        "|" + "---|" * 10,
    ]
    for i, r in enumerate(lb.rows, 1):
        cells = [
            str(i),
            r.name,
            r.detector,
            _f(r.auprc),
            _ci(r.auprc_ci),
            _fpr_cell(r),
            _ci(r.fpr95_per_agent_ci),
            _f(r.fpr95_global),
            f"{r.n_rows} ({r.n_attack_rows})",
            "; ".join(r.caveats) or "–",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    for name, reason in sorted(lb.not_run.items()):
        lines.append(f"| – | {name} | {name} | – | – | – | – | – | – | not run ({reason}) |")
    return "\n".join([*lines, "", *_family_md(lb)]) + "\n"


def _family_md(lb: Leaderboard) -> list[str]:
    fams = sorted({f for r in lb.rows for f in r.family_recall})
    lines = [
        f"#### Per-family attack recall: {_title(lb)}",
        "",
        "Warm test rows, attack level (an attack counts if any of its envelopes is flagged), all "
        "strength buckets. Thresholds are each run's per-agent FPR@95 operating point "
        "(`per_agent`, benign FPR as in the leaderboard), or the engine's live thresholds (`row`). "
        "Sampled runs "
        "have no train-split scores, so they have no per-agent thresholds and these cells are not "
        "meaningful for them.",
        "",
        "| " + " | ".join(["Run", "Thresholds", *fams, "Control FPR"]) + " |",
        "|" + "---|" * (len(fams) + 3),
    ]
    for r in lb.rows:
        cells = [r.name, r.family_mode, *(_f(r.family_recall.get(f)) for f in fams)]
        lines.append("| " + " | ".join([*cells, _f(r.control_fpr)]) + " |")
    return lines


def _gate_rows(
    reports: Mapping[str, Mapping[str, Any]], what: str
) -> tuple[dict[str, dict[OutcomeMode, GatePreview]], str]:
    """The ``GATE_RUNS`` pairs of ``reports`` as gate rows, and their split kind; refuses a pair
    missing a protocol, a run without a gate section or without ``meta.dataset_sha`` /
    ``meta.split.kind``, a run that is not an engine replay, a pair whose two runs recorded
    different engine configurations (``meta.detector_config.cfg``; spec A1.5: same engine
    configuration, same dataset) and runs of different datasets or splits."""
    rows: dict[str, dict[OutcomeMode, GatePreview]] = {}
    seen: set[tuple[str, str]] = set()
    for row, runs in GATE_RUNS.items():
        by_mode: dict[OutcomeMode, GatePreview] = {}
        cfgs: list[object] = []
        for run, mode in zip(runs, _GATE_MODES, strict=True):
            rep = reports.get(run)
            if rep is None or (rep.get("extras") or {}).get("outcomes") != mode:
                got = (
                    "no report" if rep is None else repr((rep.get("extras") or {}).get("outcomes"))
                )
                raise ValueError(
                    f"{what}: the gate is shown under both protocols {_GATE_MODES}; run {run!r} "
                    f"must be a {mode!r} replay, got {got}"
                )
            gate = rep.get("gate")
            if not isinstance(gate, Mapping):
                raise ValueError(f"{what}: run {run!r} has no gate section in its report")
            meta = rep.get("meta")
            meta = meta if isinstance(meta, Mapping) else {}
            sha, split = meta.get("dataset_sha"), (meta.get("split") or {}).get("kind")
            if sha is None or split is None:
                raise ValueError(
                    f"{what}: run {run!r} has no meta.dataset_sha or meta.split.kind in its report"
                )
            if meta.get("detector") != "engine":
                raise ValueError(
                    f"{what}: run {run!r} is not an engine replay (meta.detector "
                    f"{meta.get('detector')!r})"
                )
            cfgs.append((meta.get("detector_config") or {}).get("cfg"))
            seen.add((str(sha), str(split)))
            by_mode[mode] = preview_from_json(dict(gate))
        if cfgs[0] != cfgs[1]:
            raise ValueError(
                f"{what}: runs {runs[0]!r} and {runs[1]!r} replayed different engine "
                "configurations; both protocols of a gate row need the same engine configuration "
                "(spec A1.5)"
            )
        rows[row] = by_mode
    if len(seen) != 1:
        raise ValueError(f"{what}: the gate runs mix datasets or splits {sorted(seen)}")
    return rows, seen.pop()[1]


def render_gate_section(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    sensitivity: Mapping[str, Mapping[str, Any]] | None = None,
) -> str:
    """The Tier-1 gate of the engine runs under both protocols, side by side (spec A1.5).

    ``reports`` maps run names to ``report.json`` contents and must hold every run of
    :data:`GATE_RUNS` with its ``gate`` section: ``engine`` (oracle-reviewer approval loop) next
    to ``engine-allow-all`` (spec §5.2 allow-all), then the pre-A1 pair, each line under its
    ``PROTOCOL_LABELS`` text verbatim with recall, among-the-reasons share and top-1 share
    (``original spec §9 criterion``) per gated family, as
    :func:`agentfw.bench.gate.render_gate_matrix` prints them.
    ``sensitivity`` holds the same runs on the sensitivity dataset (``magnitude_far_mult = 1.0``);
    without it every ``magnitude_spike`` cell is withheld (spec A1.5 rule 4). Raises
    ``ValueError`` when either protocol of a pair is missing, a run has no gate section or no
    dataset / split in its meta, a run is not an engine replay, the two runs of a pair recorded
    different engine configurations, or the runs of one mapping mix datasets or splits.
    """
    results, split = _gate_rows(reports, "reports")
    sens = None if sensitivity is None else _gate_rows(sensitivity, "sensitivity")[0]
    head, _, rest = render_gate_matrix(results, sensitivity=sens).partition("\n\n")
    runs = "; ".join(
        f"`{row}`: runs `{review}` ({PROTOCOL_LABELS['review']}) and `{allow}` "
        f"({PROTOCOL_LABELS['allow']})"
        for row, (review, allow) in GATE_RUNS.items()
    )
    return f"{head}\n\nRows from the leaderboard runs, split `{split}`: {runs}.\n\n{rest}"


def _title(lb: Leaderboard) -> str:
    return f"{lb.label}, split `{lb.split}`" if lb.label else f"split `{lb.split}`"


def main(argv: Sequence[str] | None = None) -> None:
    """``python -m agentfw.bench.leaderboard --out docs/x.md [--label L] REPORT.json...``"""
    p = argparse.ArgumentParser(prog="python -m agentfw.bench.leaderboard")
    p.add_argument("reports", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--label", default="")
    p.add_argument(
        "--not-run",
        action="append",
        default=[],
        metavar="NAME=REASON",
        help="a detector listed as not run, e.g. 'llm_judge=requires a live judge model'",
    )
    a = p.parse_args(argv)
    not_run: dict[str, str] = {}
    for item in a.not_run:
        name, sep, reason = item.partition("=")
        if not (sep and name and reason):
            p.error(f"--not-run expects NAME=REASON, got {item!r}")
        not_run[name] = reason
    try:
        lb = build_leaderboard(a.reports, label=a.label, not_run=not_run)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    a.out.write_text(render_leaderboard(lb))


if __name__ == "__main__":
    main()
