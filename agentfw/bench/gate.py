"""The Tier-1 gate under both replay protocols (spec A1.5 to A1.7, D-027, D-028).

:func:`gate_matrix` replays one engine configuration on one dataset under the oracle-reviewer
approval loop (``"review"``) and under spec §5.2 allow-all (``"allow"``) and returns the amended
gate (:func:`agentfw.bench.metrics.gate_preview`) of each. :func:`render_gate_matrix` prints the
two results together, each under its protocol label verbatim, so a gate result is never shown for
one protocol alone (spec A1.5 rule 1), and prints the ``magnitude_spike`` cell only with its two
companion rows (spec A1.5 rule 4, ruling A1-17).

    uv run python -m agentfw.bench.gate --traces DIR [--sensitivity-traces DIR] [--pre-a1] [--out FILE.md] [--json FILE.json]

The CLI always replays the engine and the engine before Amendment A1 (the defaults with
:data:`PRE_A1_SWITCHES`) on ``--traces``; ``--pre-a1`` is kept for compatibility and changes
nothing. ``--sensitivity-traces`` (a dataset generated with ``magnitude_far_mult = 1.0``) replays
both engines on that dataset as the companion rows; without it every ``magnitude_spike`` cell is
withheld. The CLI refuses (usage error, exit 2) a ``--sensitivity-traces`` that is not the
sensitivity dataset of ``--traces`` (:func:`sensitivity_refusal`). :func:`load_gate_json` reads a ``--json`` file back.

:func:`magnitude_census` is the composition of the ``magnitude_spike`` gate cell that spec A1.4
shows with every gate table (the gate evidence's census, plan Task 1.23): the CLI prints it from
``--traces`` (:func:`dataset_census`) under each main table, the leaderboard from its dataset, the
acceptance test in its summary, and ``docs/decisions/evidence/a1-gate/measure.py`` writes it.
"""  # noqa: E501 - the CLI usage line is kept whole

from __future__ import annotations

import argparse
import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson

from agentfw.bench.ablation import replay_detector
from agentfw.bench.metrics import (
    GATE_BENIGN_FPR,
    GATE_FAMILIES,
    GATE_RECALL,
    GATE_SHARE,
    GateCell,
    GatePreview,
    _bucket_of,
    gate_preview,
)
from agentfw.bench.replay import (
    PROTOCOL_LABELS,
    OutcomeMode,
    ScoreRow,
    SplitSpec,
    is_outcome_mode,
)
from agentfw.bench.sweeps import PRE_A1_SWITCHES
from agentfw.core.config import EngineConfig
from agentfw.synth.generate import Manifest

__all__ = [
    "CLI_COMPANION_SOURCE",
    "ENGINE_ROW",
    "ORIGINAL_CRITERION_LABEL",
    "PRE_A1_ROW",
    "SENSITIVITY_FAR_MULT",
    "MagnitudeCensus",
    "dataset_census",
    "gate_matrix",
    "load_gate_json",
    "magnitude_census",
    "main",
    "preview_from_json",
    "render_census",
    "render_gate_matrix",
    "sensitivity_refusal",
]

#: The label printed with the top-1 share (spec A1.5 rule 3, A1.6).
ORIGINAL_CRITERION_LABEL = "original spec §9 criterion"
#: Engine row names used by :func:`main`.
ENGINE_ROW = "engine"
PRE_A1_ROW = "pre-A1 engine (defaults + PRE_A1_SWITCHES)"
_REQUIRED: tuple[OutcomeMode, OutcomeMode] = ("review", "allow")
_MAGNITUDE = "magnitude_spike"
_COMPANION_HEADER = "## Companion: sensitivity dataset (magnitude_far_mult = 1.0)"
_COMPANION_VERDICT = "companion row (spec A1.5 rule 4), not a gate result"
#: Where the CLI's companion rows come from (:func:`main` passes it as ``companion_source``).
CLI_COMPANION_SOURCE = (
    f"the engine before Amendment A1 on the same dataset (`{PRE_A1_ROW}`, always computed by the "
    "CLI) and the amended engine on the sensitivity dataset (`magnitude_far_mult = 1.0`, CLI "
    "option `--sensitivity-traces DIR`), printed after the main blocks."
)
_PLAIN_COMPANION_SOURCE = (
    f"the engine before Amendment A1 on the same dataset (`{PRE_A1_ROW}`) and the amended engine "
    "on the sensitivity dataset (`magnitude_far_mult = 1.0`), printed after the main blocks."
)

_Rows = dict[str, dict[OutcomeMode, GatePreview]]


def gate_matrix(
    traces_dir: Path,
    cfg: EngineConfig = EngineConfig(),  # noqa: B008 - frozen dataclass, safe as a default
    *,
    protocols: Sequence[OutcomeMode] = ("review", "allow"),
) -> dict[OutcomeMode, GatePreview]:
    """The amended gate of the engine with ``cfg`` on ``traces_dir``, once per protocol.

    Same engine configuration, same dataset and split for every protocol; keys in the order of
    ``protocols`` (duplicates once). Raises ``ValueError`` unless both ``"review"`` and
    ``"allow"`` are requested (spec A1.5: the gate is always evaluated under both).
    """
    if not set(_REQUIRED) <= set(protocols):
        raise ValueError(
            f"the Tier-1 gate is evaluated under both protocols {_REQUIRED}, got {tuple(protocols)}"
        )
    out: dict[OutcomeMode, GatePreview] = {}
    for mode in dict.fromkeys(protocols):
        rows, _ = replay_detector(traces_dir, "engine", SplitSpec(), cfg, mode)
        out[mode] = gate_preview(rows)
    return out


@dataclass(frozen=True, slots=True)
class MagnitudeCensus:
    """The ``magnitude_spike`` composition shown with every gate table (spec A1.4).

    ``gated_test_n`` / ``by_persona``: the attacks of the gate cell (warm test rows, bucket
    ``"10-100"``; an attack's first warm test row in time order gives its persona and strength),
    in total and per persona. ``share_strength_ge_10`` / ``median_strength`` / ``n_spikes``: over
    every spike of the dataset (``labels.jsonl``, one strength per attack id). The field names
    and values are those of ``matrix.json``'s ``census``.
    """

    gated_test_n: int
    by_persona: dict[str, int]
    share_strength_ge_10: float
    median_strength: float
    n_spikes: int


def magnitude_census(rows: Iterable[ScoreRow], traces_dir: Path) -> MagnitudeCensus:
    """The census of the dataset in ``traces_dir`` from replay ``rows`` of it (all rows, or the
    warm test rows only: both give the same census)."""
    first: dict[str, ScoreRow] = {}
    warm = (r for r in rows if r.split == "test" and r.is_warm and not math.isnan(r.fused_score))
    for r in sorted(warm, key=lambda r: (r.ts_unix_ms, r.envelope_id)):
        if r.label == "attack" and r.attack_family == _MAGNITUDE and r.attack_id:
            first.setdefault(r.attack_id, r)
    bucket = GATE_FAMILIES[_MAGNITUDE][0]
    by_persona: dict[str, int] = {}
    for r in first.values():
        if _bucket_of(_MAGNITUDE, r.strength) == bucket:
            by_persona[r.agent_id] = by_persona.get(r.agent_id, 0) + 1
    strength: dict[str, float] = {}
    for line in (traces_dir / "labels.jsonl").read_bytes().splitlines():
        lab = orjson.loads(line)
        if lab["attack_family"] == _MAGNITUDE and lab["strength"] is not None:
            strength.setdefault(lab["attack_id"], float(lab["strength"]))
    s = list(strength.values())
    return MagnitudeCensus(
        gated_test_n=sum(by_persona.values()),
        by_persona=dict(sorted(by_persona.items())),
        share_strength_ge_10=sum(x >= 10.0 for x in s) / len(s) if s else math.nan,
        median_strength=statistics.median(s) if s else math.nan,
        n_spikes=len(s),
    )


def dataset_census(traces_dir: Path) -> MagnitudeCensus:
    """:func:`magnitude_census` of a dataset directory without an engine replay.

    The census reads labels, the split and the warm flags, which the replay derives from the data
    alone (never from scores), so the rows of the cheapest detector (the ``random`` baseline, a
    few seconds on SW1) give the census of every engine's replay of that dataset.
    """
    rows, _ = replay_detector(traces_dir, "random", SplitSpec(), None, "allow")
    return magnitude_census(rows, traces_dir)


def render_census(c: MagnitudeCensus) -> str:
    """The census line printed under the gate tables (personas by count, then name)."""
    persona = ", ".join(
        f"{a} {n}" for a, n in sorted(c.by_persona.items(), key=lambda kv: (-kv[1], kv[0]))
    )
    return (
        f"Census of the {_MAGNITUDE} gate cell (spec A1.4): n {c.gated_test_n} (warm test rows, "
        f"bucket {GATE_FAMILIES[_MAGNITUDE][0]}) by persona: {persona or 'none'}; over all "
        f"{c.n_spikes} spikes of the dataset, median strength {_num(c.median_strength, 1)} and "
        f"share with strength >= 10 {_num(c.share_strength_ge_10, 3)}."
    )


def _num(x: float, nd: int) -> str:
    return "n/a" if math.isnan(x) else f"{x:.{nd}f}"


def _cell(g: GatePreview, family: str) -> str:
    c = g.cells.get(family)
    if c is None:
        return "missing"
    return (
        f"{_num(c.recall, 3)} ({c.n_detected}/{c.n_total}); {_num(c.among_reasons_share, 2)}; "
        f"{_num(c.top_reason_share, 2)}"
    )


def _verdict(g: GatePreview) -> str:
    return "PASS" if g.passed else f"FAIL: {', '.join(g.failures)}"


def _require_both(results: Mapping[str, Mapping[OutcomeMode, GatePreview]]) -> None:
    for name, by_mode in results.items():
        if not set(_REQUIRED) <= set(by_mode):
            raise ValueError(
                f"row {name!r} must hold both protocols {_REQUIRED}, got {list(by_mode)}"
            )


def _block(
    name: str,
    by_mode: Mapping[OutcomeMode, GatePreview],
    head: list[str],
    withheld: str | None,
    *,
    companion: bool,
    note: str | None = None,
) -> list[str]:
    lines = [f"### {name}", "", "| " + " | ".join(head) + " |"]
    lines.append("|" + "|".join(["---"] * len(head)) + "|")
    for mode in [*_REQUIRED, *(m for m in by_mode if m not in _REQUIRED)]:
        g = by_mode[mode]
        cells = [
            PROTOCOL_LABELS[mode],
            *(
                withheld if fam == _MAGNITUDE and withheld is not None else _cell(g, fam)
                for fam in GATE_FAMILIES
            ),
            _num(g.benign_fpr, 4),
            _num(g.control_fpr, 3),
            _COMPANION_VERDICT if companion else _verdict(g),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    if note is not None:
        lines += [note, ""]
    return lines


def render_gate_matrix(
    results: Mapping[str, Mapping[OutcomeMode, GatePreview]],
    *,
    sensitivity: Mapping[str, Mapping[OutcomeMode, GatePreview]] | None = None,
    companion_source: str | None = None,
    census: MagnitudeCensus | None = None,
) -> str:
    """Markdown: one block per named engine row, one line per protocol under its label verbatim.

    Each family cell is ``recall (n_detected/n_total); among-the-reasons share; top-1 share``,
    the top-1 share labelled :data:`ORIGINAL_CRITERION_LABEL` in the column header; then benign
    FPR, control FPR and the amended criterion's PASS / FAIL with the failures. The oracle-reviewer
    line comes first, allow-all second, any other protocol after them.

    The ``magnitude_spike`` cells are printed only when both companion rows of spec A1.5 rule 4
    are present: ``results`` holds :data:`PRE_A1_ROW` and ``sensitivity`` (the rows replayed on
    the dataset generated with ``magnitude_far_mult = 1.0``) holds :data:`ENGINE_ROW`. Otherwise
    every ``magnitude_spike`` cell of every block reads ``withheld (spec A1.5 rule 4: missing
    ...)``; the other cells, the FPRs and the verdict (computed from the data) are unchanged. The
    ``sensitivity`` rows follow the main blocks under a companion header, their last column saying
    they are companion rows, not gate results. Raises ``ValueError`` when a row of ``results`` or
    ``sensitivity`` lacks either protocol (spec A1.5 rule 1).

    ``companion_source`` ends the intro's sentence "The magnitude_spike cell is shown only with
    its two companion rows (spec A1.5 rule 4): ...": the caller names the two rows and where they
    come from (the CLI passes :data:`CLI_COMPANION_SOURCE`). ``None`` names the rows only.

    ``census`` (:func:`magnitude_census` of the dataset of ``results``) is printed as
    :func:`render_census` under each main table, whether or not the magnitude cells are shown
    (it is the dataset's composition, not a result); the companion tables carry none.
    """
    _require_both(results)
    if sensitivity is not None:
        _require_both(sensitivity)
    # (what is missing, how to supply it); the hint names only the caller's actual gap.
    gaps: list[tuple[str, str]] = []
    if PRE_A1_ROW not in results:
        gaps.append(
            (
                "pre-A1 row",
                f"add the `{PRE_A1_ROW}` row to the results (the engine with `PRE_A1_SWITCHES` "
                "on the same dataset; the CLI always computes it)",
            )
        )
    if sensitivity is None or ENGINE_ROW not in sensitivity:
        gaps.append(
            (
                "sensitivity row",
                f"pass the `{ENGINE_ROW}` row replayed on a dataset generated with "
                "`magnitude_far_mult = 1.0` as `sensitivity` (CLI option "
                "`--sensitivity-traces DIR`)",
            )
        )
    missing = " and ".join(what for what, _ in gaps)
    withheld = f"withheld (spec A1.5 rule 4: missing {missing})" if gaps else None
    head = [
        "Protocol",
        *(
            f"{fam} {bucket}: recall (n/N); among; top-1 ({ORIGINAL_CRITERION_LABEL})"
            for fam, (bucket, _) in GATE_FAMILIES.items()
        ),
        "benign FPR",
        "control FPR",
        "amended criterion",
    ]
    lines = [
        "## Tier-1 gate under both protocols",
        "",
        "Warm test rows at live per-agent thresholds, attack level (slow_drift excluded). Amended "
        f"criterion (spec A1.7): per family attack recall >= {GATE_RECALL:g} and among-the-reasons "
        f"share >= {GATE_SHARE:g}, benign FPR <= {GATE_BENIGN_FPR:g}. Among = share of detected "
        "attacks with the targeted detector among the reasons (score > reason_min) of a flagged "
        f"row; top-1 = the {ORIGINAL_CRITERION_LABEL}, reported and not gated. Each line is the "
        "result under one protocol; the two lines are shown together and neither alone is the "
        "Tier-1 gate result. Which protocol closes Tier 1 is the user's open decision (D-028).",
        "",
        f"The {_MAGNITUDE} cell is shown only with its two companion rows (spec A1.5 rule 4): "
        + (_PLAIN_COMPANION_SOURCE if companion_source is None else companion_source),
        "",
    ]
    if gaps:
        lines += [
            f"Every {_MAGNITUDE} cell is withheld: missing {missing}. To show it, "
            + " and ".join(fix for _, fix in gaps)
            + ".",
            "",
        ]
    for name, by_mode in results.items():
        note = None if census is None else render_census(census)
        lines += _block(name, by_mode, head, withheld, companion=False, note=note)
    if sensitivity:  # an empty mapping holds no companion row: no empty companion section
        lines += [
            _COMPANION_HEADER,
            "",
            "The same engine rows replayed on the sensitivity dataset. They are the companion rows "
            f"of the {_MAGNITUDE} cell, not gate results.",
            "",
        ]
        for name, by_mode in sensitivity.items():
            lines += _block(name, by_mode, head, withheld, companion=True)
    return "\n".join(lines)


def _float(x: float | None) -> float:
    """A float field read back from JSON: orjson writes NaN (and +-inf) as ``null``."""
    return math.nan if x is None else float(x)


def _cell_from(d: dict[str, Any]) -> GateCell:
    return GateCell(
        bucket=str(d["bucket"]),
        detector=str(d["detector"]),
        recall=_float(d["recall"]),
        n_detected=int(d["n_detected"]),
        n_total=int(d["n_total"]),
        top_reason_share=_float(d["top_reason_share"]),
        among_reasons_share=_float(d["among_reasons_share"]),
    )


def preview_from_json(d: dict[str, Any]) -> GatePreview:
    """One ``GatePreview`` as orjson wrote it (a ``--json`` protocol entry, or a ``report.json``
    ``gate`` section): ``null`` floats come back as ``float("nan")``, lists as tuples."""
    return GatePreview(
        cells={fam: _cell_from(c) for fam, c in d["cells"].items()},
        benign_fpr=_float(d["benign_fpr"]),
        control_fpr=_float(d["control_fpr"]),
        min_recall=_float(d["min_recall"]),
        passed=bool(d["passed"]),
        missing=tuple(d["missing"]),
        failures=tuple(d["failures"]),
    )


def _rows_from(d: dict[str, dict[str, Any]]) -> _Rows:
    out: _Rows = {}
    for name, by_mode in d.items():
        row: dict[OutcomeMode, GatePreview] = {}
        for mode, g in by_mode.items():
            if not is_outcome_mode(mode):
                raise ValueError(f"row {name!r}: unknown replay protocol {mode!r}")
            row[mode] = preview_from_json(g)
        out[name] = row
    return out


def load_gate_json(path: Path) -> tuple[_Rows, _Rows | None]:
    """``(results, sensitivity)`` of a file written by ``--json``, as ``GatePreview`` objects.

    ``null`` floats come back as ``float("nan")`` and lists as tuples, so the round trip equals
    the in-memory results field by field; ``sensitivity`` is ``None`` when the run had no
    ``--sensitivity-traces``.
    """
    payload = orjson.loads(path.read_bytes())
    sens = payload["sensitivity"]
    return _rows_from(payload["results"]), None if sens is None else _rows_from(sens)


#: The ``magnitude_far_mult`` of the sensitivity dataset (spec A1.4).
SENSITIVITY_FAR_MULT = 1.0


def sensitivity_refusal(traces_dir: Path, sensitivity_dir: Path) -> str | None:
    """Why ``sensitivity_dir`` is not the sensitivity dataset of ``traces_dir`` (``None``: it is).

    It is when both manifests load, its config has ``magnitude_far_mult = 1.0`` and equals the
    ``traces_dir`` config apart from ``magnitude_far_mult``, and its traces differ (a mode
    ``"own"`` config ignores the multiple, so its "sensitivity" dataset is the same bytes).
    """
    try:
        main_m, sens_m = (
            Manifest.model_validate_json((d / "manifest.json").read_bytes())
            for d in (traces_dir, sensitivity_dir)
        )
    except (OSError, ValueError) as e:
        return f"needs the manifest.json of both datasets ({type(e).__name__}: {e})"
    mult = sens_m.config.magnitude_far_mult
    if mult != SENSITIVITY_FAR_MULT:
        return (
            f"{sensitivity_dir} was generated with magnitude_far_mult = {mult:g}; the sensitivity "
            f"dataset has magnitude_far_mult = {SENSITIVITY_FAR_MULT:g} (spec A1.4)"
        )
    same = sens_m.config.model_copy(update={"magnitude_far_mult": main_m.config.magnitude_far_mult})
    if same != main_m.config:
        return (
            f"the config of {sensitivity_dir} differs from that of {traces_dir} apart from "
            "magnitude_far_mult (the sensitivity dataset changes that field only)"
        )
    sha = main_m.files["traces.jsonl"]
    if sens_m.files["traces.jsonl"] == sha:
        return (
            f"{sensitivity_dir} holds the same traces as {traces_dir} (sha {sha[:12]}): its "
            "magnitude_far_mult changed nothing, so it is not a sensitivity dataset"
        )
    return None


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m agentfw.bench.gate")
    p.add_argument("--traces", type=Path, required=True)
    p.add_argument(
        "--sensitivity-traces",
        type=Path,
        help="the sensitivity dataset (magnitude_far_mult = 1.0): replays the engine and the "
        "pre-A1 engine on it as the companion rows of the magnitude_spike cell (spec A1.5 rule 4); "
        "without it every magnitude_spike cell is withheld. Refused unless its config is the "
        "--traces config with magnitude_far_mult = 1.0 and its traces differ",
    )
    p.add_argument(
        "--pre-a1",
        action="store_true",
        help="always on (spec A1.5 rule 4); kept for compatibility",
    )
    p.add_argument("--out", type=Path, help="write the markdown table here")
    p.add_argument("--json", type=Path, help="write the results as JSON here")
    a = p.parse_args(argv)
    if a.sensitivity_traces is not None:
        why = sensitivity_refusal(a.traces, a.sensitivity_traces)
        if why is not None:
            p.error(f"--sensitivity-traces: {why}")
    engines = {ENGINE_ROW: EngineConfig(), PRE_A1_ROW: EngineConfig().replace(**PRE_A1_SWITCHES)}
    results = {name: gate_matrix(a.traces, cfg) for name, cfg in engines.items()}
    sensitivity = (
        None
        if a.sensitivity_traces is None
        else {name: gate_matrix(a.sensitivity_traces, cfg) for name, cfg in engines.items()}
    )
    head = f"# Tier-1 gate on `{a.traces}`\n\n"
    if a.sensitivity_traces is not None:
        head += f"Sensitivity dataset: `{a.sensitivity_traces}`\n\n"
    md = head + render_gate_matrix(
        results,
        sensitivity=sensitivity,
        companion_source=CLI_COMPANION_SOURCE,
        census=dataset_census(a.traces),
    )
    print(md)
    if a.out is not None:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(md)
    if a.json is not None:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "traces": str(a.traces),
            "sensitivity_traces": (
                None if a.sensitivity_traces is None else str(a.sensitivity_traces)
            ),
            "labels": {m: PROTOCOL_LABELS[m] for m in _REQUIRED},
            # Data, not a rendered summary: every number, the magnitude cells included (ruling R1);
            # orjson writes dataclasses natively, NaN as null (load_gate_json turns it back).
            "results": results,
            "sensitivity": sensitivity,
        }
        a.json.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
