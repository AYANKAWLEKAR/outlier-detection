"""The Tier-1 gate under both replay protocols (spec A1.5 to A1.7, D-027, D-028).

:func:`gate_matrix` replays one engine configuration on one dataset under the oracle-reviewer
approval loop (``"review"``) and under spec §5.2 allow-all (``"allow"``) and returns the amended
gate (:func:`agentfw.bench.metrics.gate_preview`) of each. :func:`render_gate_matrix` prints the
two results together, each under its protocol label verbatim, so a gate result is never shown for
one protocol alone (spec A1.5 rule 1).

    uv run python -m agentfw.bench.gate --traces DIR [--pre-a1] [--out FILE.md] [--json FILE.json]

``--pre-a1`` adds the engine before Amendment A1 (the defaults with :data:`PRE_A1_SWITCHES`) as a
second row on the same dataset.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import orjson

from agentfw.bench.ablation import replay_detector
from agentfw.bench.metrics import (
    GATE_BENIGN_FPR,
    GATE_FAMILIES,
    GATE_RECALL,
    GATE_SHARE,
    GatePreview,
    gate_preview,
)
from agentfw.bench.replay import PROTOCOL_LABELS, OutcomeMode, SplitSpec
from agentfw.bench.sweeps import PRE_A1_SWITCHES
from agentfw.core.config import EngineConfig

__all__ = [
    "ENGINE_ROW",
    "ORIGINAL_CRITERION_LABEL",
    "PRE_A1_ROW",
    "gate_matrix",
    "main",
    "render_gate_matrix",
]

#: The label printed with the top-1 share (spec A1.5 rule 3, A1.6).
ORIGINAL_CRITERION_LABEL = "original spec §9 criterion"
#: Engine row names used by :func:`main`.
ENGINE_ROW = "engine"
PRE_A1_ROW = "pre-A1 engine (defaults + PRE_A1_SWITCHES)"
_REQUIRED: tuple[OutcomeMode, OutcomeMode] = ("review", "allow")


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


def render_gate_matrix(results: Mapping[str, Mapping[OutcomeMode, GatePreview]]) -> str:
    """Markdown: one block per named engine row, one line per protocol under its label verbatim.

    Each family cell is ``recall (n_detected/n_total); among-the-reasons share; top-1 share``,
    the top-1 share labelled :data:`ORIGINAL_CRITERION_LABEL` in the column header; then benign
    FPR, control FPR and the amended criterion's PASS / FAIL with the failures. The oracle-reviewer
    line comes first, allow-all second, any other protocol after them. Raises ``ValueError`` when
    a row lacks either of the two (spec A1.5 rule 1).
    """
    for name, by_mode in results.items():
        if not set(_REQUIRED) <= set(by_mode):
            raise ValueError(
                f"row {name!r} must hold both protocols {_REQUIRED}, got {list(by_mode)}"
            )
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
        "The magnitude_spike cell is read with its companion rows (spec A1.5 rule 4): the engine "
        "before Amendment A1 on the same dataset (`--pre-a1`) and the amended engine on the "
        "sensitivity dataset (`magnitude_far_mult = 1.0`, a separate `--traces`).",
        "",
    ]
    for name, by_mode in results.items():
        lines += [f"### {name}", "", "| " + " | ".join(head) + " |"]
        lines.append("|" + "|".join(["---"] * len(head)) + "|")
        order = [*_REQUIRED, *(m for m in by_mode if m not in _REQUIRED)]
        for mode in order:
            g = by_mode[mode]
            cells = [
                PROTOCOL_LABELS[mode],
                *(_cell(g, fam) for fam in GATE_FAMILIES),
                _num(g.benign_fpr, 4),
                _num(g.control_fpr, 3),
                _verdict(g),
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m agentfw.bench.gate")
    p.add_argument("--traces", type=Path, required=True)
    p.add_argument(
        "--pre-a1",
        action="store_true",
        help="add the engine before Amendment A1 (defaults + PRE_A1_SWITCHES) as a second row",
    )
    p.add_argument("--out", type=Path, help="write the markdown table here")
    p.add_argument("--json", type=Path, help="write the results as JSON here")
    a = p.parse_args(argv)
    engines = {ENGINE_ROW: EngineConfig()}
    if a.pre_a1:
        engines[PRE_A1_ROW] = EngineConfig().replace(**PRE_A1_SWITCHES)
    results = {name: gate_matrix(a.traces, cfg) for name, cfg in engines.items()}
    md = f"# Tier-1 gate on `{a.traces}`\n\n{render_gate_matrix(results)}"
    print(md)
    if a.out is not None:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(md)
    if a.json is not None:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "traces": str(a.traces),
            "labels": {m: PROTOCOL_LABELS[m] for m in _REQUIRED},
            "results": results,  # orjson writes dataclasses natively, NaN as null
        }
        a.json.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
