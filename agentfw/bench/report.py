"""``report.json`` / ``report.md`` assembly (spec 5.6).

The report is a plain JSON-ready dict: NaN and +-inf serialize as ``null``, dataclasses become
dicts, tuples become lists. :func:`render_markdown` consumes that same dict, so the markdown can
only show what the JSON shows.
"""

from __future__ import annotations

import dataclasses
import math
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from agentfw.bench.ablation import AblationRow
from agentfw.bench.metrics import GATE_RECALL, GatePreview, Metrics
from agentfw.bench.replay import PROTOCOL_LABELS, SplitSpec

__all__ = [
    "README_END",
    "README_START",
    "RunMeta",
    "build_report",
    "render_markdown",
    "update_readme",
]

README_START = "<!-- report:start -->"
README_END = "<!-- report:end -->"

AGENT_SPLIT_NOTE = (
    "Split is by agent: per-agent thresholds fall back to the pooled train ECDF for held-out "
    "agents (they have no train-split benign scores of their own)."
)


class RunMeta(BaseModel):
    git_sha: str
    seed: int
    dataset_sha: str  # sha256 of traces.jsonl, taken from the dataset manifest's ``files``
    detector: str
    detector_config: dict[str, Any]
    split: SplitSpec
    created_unix: int


def git_sha() -> str:
    """``git rev-parse HEAD`` or ``"nogit"``."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        )
        return out.stdout.strip() or "nogit"
    except (OSError, subprocess.SubprocessError):
        return "nogit"


def _jsonable(x: Any) -> Any:
    if isinstance(x, BaseModel):
        return _jsonable(x.model_dump(mode="python"))
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {f.name: _jsonable(getattr(x, f.name)) for f in dataclasses.fields(x)}
    if isinstance(x, Mapping):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_jsonable(v) for v in x]
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    return x


def _headline(m: Metrics, warm_only: bool) -> dict[str, Any]:
    return {
        "auprc": m.auprc,
        "auprc_ci": m.auprc_ci,
        "fpr95_global": m.fpr95_global,
        "recall_at_global_thr": m.recall_at_global_thr,
        "fpr95_per_agent": m.fpr95_per_agent,
        "fpr95_per_agent_ci": m.fpr95_per_agent_ci,
        # A per-agent FPR is only meaningful at the recall it was achieved at.
        "per_agent_recall": m.per_agent_recall,
        "per_agent_alpha": m.per_agent_alpha,
        "warm_only": warm_only,
        "n_rows": m.n_rows,
        "n_attack_rows": m.n_attack_rows,
    }


def build_report(
    all_: Metrics,
    warm: Metrics,
    meta: RunMeta,
    extras: Mapping[str, object] | None = None,
    *,
    ablations: Sequence[AblationRow] | None = None,
    gate: GatePreview | None = None,
) -> dict[str, object]:
    """Assemble the report. ``all`` / ``warm`` are the all-rows and warm-only metric sets.

    Per-family, control, cold-start, drift and latency come from the all-rows metrics (the warm-only
    cold-start table would be empty by construction); ``extras["state_bytes"]`` is lifted to the
    top level. ``ablations`` (one row per detector variant) and ``gate`` (the Tier-1 gate preview:
    attack-level recall on warm test rows at live thresholds) are stored as given, ``None`` when
    not run.
    """
    ex = dict(extras or {})
    notes = list(ex.pop("notes", []) or [])  # type: ignore[call-overload]
    if meta.split.kind == "agent" and AGENT_SPLIT_NOTE not in notes:
        notes.append(AGENT_SPLIT_NOTE)
    ex["notes"] = notes
    state_bytes = ex.pop("state_bytes", None)
    report: dict[str, object] = {
        "headline": _headline(warm, True),
        "headline_all": _headline(all_, False),
        "per_family": {
            "mode": all_.breakdown_mode,
            "envelope": all_.per_family_envelope,
            "attack": all_.per_family_attack,
            "warm_envelope": warm.per_family_envelope,
            "warm_attack": warm.per_family_attack,
        },
        "control_fpr": {"all": all_.control_fpr, "warm": warm.control_fpr},
        "cold_start": all_.cold_start,
        "drift": all_.drift,
        "latency": {"p50_us": all_.latency_p50_us, "p99_us": all_.latency_p99_us},
        "state_bytes": state_bytes,
        "gate": gate,
        "ablations": None if ablations is None else list(ablations),
        "meta": meta,
        "extras": ex,
    }
    out = _jsonable(report)
    assert isinstance(out, dict)
    return out


# --- markdown -----------------------------------------------------------------------------------


def _f(x: object, nd: int = 3) -> str:
    return "n/a" if not isinstance(x, int | float) else f"{x:.{nd}f}"


def _ci(x: object) -> str:
    if isinstance(x, list) and len(x) == 2:
        return f"[{_f(x[0])}, {_f(x[1])}]"
    return "n/a"


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _sep(n: int) -> str:
    return "|" + "|".join(["---"] * n) + "|"


def _dict(x: object) -> dict[str, Any]:
    assert isinstance(x, dict)
    return x


def render_markdown(report: Mapping[str, object]) -> str:
    meta = _dict(report["meta"])
    head = _dict(report["headline"])
    lines: list[str] = [
        "### Benchmark report",
        "",
        f"Detector `{meta['detector']}`, split `{_dict(meta['split'])['kind']}`, "
        f"seed {meta['seed']}, git `{str(meta['git_sha'])[:10]}`, "
        f"dataset `{str(meta['dataset_sha'])[:12]}`.",
        "",
        *_protocol_md(_dict(report["extras"]).get("outcomes")),
        "#### Headline",
        "",
        _row(["Metric", "All rows", "Warm only (headline)"]),
        _sep(3),
    ]
    a, w = _dict(report["headline_all"]), head
    for label, key, fmt in [
        ("AUPRC", "auprc", _f),
        ("AUPRC 95% CI", "auprc_ci", _ci),
        ("FPR@95% recall (global)", "fpr95_global", _f),
        ("FPR@95% recall (per-agent)", "fpr95_per_agent", _f),
        ("per-agent recall achieved", "per_agent_recall", _f),
        ("FPR@95 per-agent 95% CI", "fpr95_per_agent_ci", _ci),
        ("test rows", "n_rows", lambda v: str(v)),
        ("test attack rows", "n_attack_rows", lambda v: str(v)),
    ]:
        lines.append(_row([label, fmt(a[key]), fmt(w[key])]))
    lines += [
        "",
        "Per-agent FPR is only meaningful at its achieved recall (shown above).",
    ]
    for n in _dict(report["extras"]).get("notes", []) or []:
        lines += ["", f"Note: {n}"]

    pf = _dict(report["per_family"])
    lines += [
        "",
        "#### Per-family recall",
        "",
        f"Thresholds: `{pf['mode']}`. Cells are recall (detected/total).",
        "",
        _row(["Family", "Bucket", "Envelope recall", "Attack recall"]),
        _sep(4),
    ]
    env, atk = _dict(pf["envelope"]), _dict(pf["attack"])
    for fam in sorted(env):
        for bucket, cell in env[fam].items():
            ac = _dict(atk.get(fam, {})).get(bucket)

            def fmt_cell(c: Any) -> str:
                return (
                    "n/a" if c is None else f"{_f(c['recall'])} ({c['n_detected']}/{c['n_total']})"
                )

            lines.append(_row([fam, bucket, fmt_cell(cell), fmt_cell(ac)]))

    ctrl = _dict(report["control_fpr"])
    lines += [
        "",
        f"Control (benign-unusual) FPR: {_f(ctrl['all'])} all, {_f(ctrl['warm'])} warm.",
        "",
    ]
    lines += _gate_md(report.get("gate"))
    lines += _ablations_md(report.get("ablations"))

    lines += [
        "#### Cold start",
        "",
        _row(["n_key_obs", "Recall", "FPR", "Attacks", "Benign"]),
        _sep(5),
    ]
    for c in report["cold_start"]:  # type: ignore[attr-defined]
        hi = "inf" if c["hi"] is None else f"{c['hi']:g}"
        lines.append(
            _row(
                [
                    f"[{c['lo']:g}, {hi})",
                    _f(c["recall"]),
                    _f(c["fpr"]),
                    str(c["n_attack"]),
                    str(c["n_benign"]),
                ]
            )
        )
    d = _dict(report["drift"])
    lat = _dict(report["latency"])
    lines += [
        "",
        "#### Drift",
        "",
        f"Thresholds: `{pf['mode']}`. Delay in sessions from onset to the first flagged envelope "
        "of each `slow_drift` attack; misses count as k.",
        "",
        _row(["Median delay", "p90 delay", "Miss rate", "Attacks"]),
        _sep(4),
        _row([_f(d["median"], 1), _f(d["p90"], 1), _f(d["miss_rate"]), str(d["n"])]),
        "",
        f"Latency: p50 {_f(lat['p50_us'], 1)} us, p99 {_f(lat['p99_us'], 1)} us; "
        f"detector state {report['state_bytes']} bytes.",
        "",
    ]
    return "\n".join(lines)


def _protocol_md(outcomes: object) -> list[str]:
    """The replay protocol's label (``extras.outcomes``); nothing for reports that lack it."""
    if outcomes is None:
        return []
    label = PROTOCOL_LABELS.get(str(outcomes), f"outcomes={outcomes}")
    return [f"Protocol: {label}.", ""]


def _gate_md(gate: object) -> list[str]:
    lines = ["#### Tier-1 gate preview", ""]
    if not isinstance(gate, dict):
        return [*lines, "Not computed for this run.", ""]
    lines += [
        f"Warm test rows, live per-agent thresholds, attack level; gate = recall >= "
        f"{GATE_RECALL:g} per family (slow_drift excluded). Share = top-reason share of the "
        "targeted detector among detected attacks.",
        "",
        _row(["Family", "Bucket", "Detector", "Attack recall", "Top-reason share"]),
        _sep(5),
    ]
    for fam, c in sorted(_dict(gate["cells"]).items()):
        lines.append(
            _row(
                [
                    fam,
                    c["bucket"],
                    c["detector"],
                    f"{_f(c['recall'])} ({c['n_detected']}/{c['n_total']})",
                    _f(c["top_reason_share"]),
                ]
            )
        )
    lines += [
        "",
        f"Benign FPR {_f(gate['benign_fpr'], 4)}, control FPR {_f(gate['control_fpr'])}; "
        f"min recall {_f(gate['min_recall'])}: gate {'PASS' if gate['passed'] else 'FAIL'}.",
        "",
    ]
    if gate.get("missing"):
        lines += [f"Missing gated families (gate cannot pass): {', '.join(gate['missing'])}.", ""]
    return lines


def _ablations_md(ablations: object) -> list[str]:
    lines = ["#### Ablations", ""]
    if not isinstance(ablations, list) or not ablations:
        return [*lines, "No ablations in this run.", ""]
    fams = sorted({f for a in ablations for f in _dict(a["per_family_recall"])})
    lines += [
        "Warm test rows. FPR@95 per-agent with the per-agent recall it achieved; family recall is "
        "attack level at live thresholds.",
        "",
        _row(["Variant", "AUPRC", "FPR@95 per-agent", "per-agent recall", *fams, "State bytes"]),
        _sep(5 + len(fams)),
    ]
    for a in ablations:
        fr = _dict(a["per_family_recall"])
        lines.append(
            _row(
                [
                    a["variant"],
                    _f(a["auprc"]),
                    _f(a["fpr95_per_agent"]),
                    _f(a["per_agent_recall"]),
                    *(_f(fr.get(f)) for f in fams),
                    str(a["state_bytes"]),
                ]
            )
        )
    return [*lines, ""]


def update_readme(readme: Path, md: str) -> None:
    """Replace the text between the report markers; idempotent. Raises if markers are missing."""
    text = readme.read_text()
    s = text.find(README_START)
    e = text.find(README_END)
    if s < 0 or e < s:
        raise ValueError(f"{readme} lacks {README_START} ... {README_END} markers")
    new = f"{text[: s + len(README_START)]}\n{md.strip()}\n{text[e:]}"
    readme.write_text(new)
