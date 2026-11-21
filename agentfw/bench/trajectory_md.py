"""Markdown rendering of the trajectory benchmark report (``bench.trajectory``, Task 4.1).

:func:`render_markdown` reads only the JSON-ready report dict, so the markdown can only show what
``trajectory_report.json`` holds. Every protocol section is headed by its ``PROTOCOL_LABELS``
label, and the reviewer protocol's is marked optimistic.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import orjson

from agentfw.bench.attribution import REDUNDANT

__all__ = ["render_markdown"]


def _p(x: object, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _ci(lo: object, hi: object, nd: int = 3) -> str:
    return f"[{_p(lo, nd)}, {_p(hi, nd)}]"


def _delta(d: Mapping[str, Any] | None, nd: int = 4) -> str:
    if d is None:
        return "–"
    sig = d["lo"] is not None and d["hi"] is not None and (d["lo"] > 0 or d["hi"] < 0)
    s = f"{d['delta']:+.{nd}f} {_ci(d['lo'], d['hi'], nd)}"
    return f"**{s}**" if sig else s


def _table(head: Sequence[str], body: Sequence[Sequence[str]]) -> list[str]:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(r) + " |" for r in body]
    return out


def _halt_theta(halts: Mapping[str, Mapping[str, Any]]) -> str:
    """The θ_halt the rows' density policy was evaluated at (each row stores its own, T-68)."""
    got = {h["theta_halt"] for h in halts.values()}
    return f"θ_halt = {got.pop()}" if len(got) == 1 else "each row's θ_halt"


def _approx_note(halts: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """The † note under the halt table: an enforcing ``provisional`` replay is exact only up to
    its own θ_halt (T-43), which each row stores (T-68). Empty when every row is exact."""
    got = {h["theta_halt"] for h in halts.values() if not h["exact_at_every_theta"]}
    if not got:
        return []
    above = f"θ = {got.pop()}" if len(got) == 1 else "each † row's θ_halt"
    return [
        "",
        f"† enforcing replay with `provisional`: the sensitivity steps above {above}",
        "are approximate (T-43); the shadow rows are exact at every θ.",
    ]


def _seed_block(seed: str, b: Mapping[str, Any], label: str, theta_row: str) -> list[str]:
    rows: Mapping[str, Any] = b["rows"]
    deltas: Mapping[str, Any] = b["deltas"]
    L = [f"### {label} — seed {seed}", ""]
    L.append("**Per-call (warm test rows) and decision metrics.** Δ = row − engine, paired session")
    L.append("bootstrap 95% CI; bold where the interval excludes 0.")
    L.append("")
    body = []
    for name, s in rows.items():
        pc, dc = s["percall"], s["decisions"]
        dl = deltas.get(name, {})
        body.append(
            [
                name,
                f"{_p(pc['auprc'])} {_ci(*pc['auprc_ci'])}",
                f"{_p(pc['fpr95_per_agent'])} {_ci(*pc['fpr95_per_agent_ci'])}",
                _p(pc["precision_per_agent95"]),
                _p(dc["benign_fpr"], 4),
                _delta(dl.get("benign_fpr")),
                _p(dc["control_fpr"], 4),
                _delta(dl.get("control_fpr")),
                _p(dc["precision_at_base_rate"]),
                _p(dc["escalation_rate_benign"], 4),
                f"{dc['n_block']} / {dc['n_require_approval']}",
                _p(s.get("scores_equal_engine")) + " / " + _p(s.get("decisions_equal_engine")),
            ]
        )
    L += _table(
        [
            "row",
            "AUPRC [CI]",
            "FPR@95 per-agent [CI]",
            "prec@base (per-agent 95)",
            "dec. benign FPR",
            "Δ benign FPR",
            "dec. control FPR",
            "Δ control FPR",
            "dec. prec@base",
            "escalation rate",
            "block / approval",
            "scores / decisions = engine",
        ],
        body,
    )
    fams = sorted({f for s in rows.values() for f in s["decisions"]["recall_by_family"]})
    L += [
        "",
        "**Attack-level decision recall by family** (Δ vs engine with CI below each value).",
        "",
    ]
    body = []
    for name, s in rows.items():
        rec = s["decisions"]["recall_by_family"]
        dl = deltas.get(name, {})
        cells = []
        for f in fams:
            c = _p(rec.get(f))
            d = dl.get(f"recall:{f}")
            if d is not None and abs(d["delta"]) > 0:
                c += f"<br>{_delta(d, 3)}"
            cells.append(c)
        body.append([name, *cells])
    n_atk = rows["engine"]["decisions"]["n_attacks_by_family"] if "engine" in rows else {}
    L += _table(["row", *[f"{f} (N={n_atk.get(f, '?')})" for f in fams]], body)
    halts = {n: s["sessions"]["halt"] for n, s in rows.items() if s["sessions"]["halt"]}
    if halts:
        L += [
            "",
            f"**Session halts** (density policy at {_halt_theta(halts)} vs the first-flag "
            "baseline;",
        ]
        L += ["sessions of all splits; precision at a 1.5% session base rate).", ""]
        body = []
        for n, h in halts.items():
            d, ff = h["density_policy"], h["first_flag"]
            body.append(
                [
                    n if h["exact_at_every_theta"] else f"{n} †",
                    _p(d["false_halt_rate"], 4),
                    _p(d["recall_runaway"]),
                    _p(d["recall_attacked"]),
                    _p(d["median_calls_from_onset"], 1),
                    _p(d["precision_at_base_rate"]),
                    _p(ff["false_halt_rate"], 4),
                    _p(ff["recall_runaway"]),
                    _p(h["sensitivity_max_step"], 4),
                    "–"
                    if h["selected"] is None
                    else f"{_p(h['selected']['theta_warn'])} / {_p(h['selected']['theta_halt'])}",
                ]
            )
        L += _table(
            [
                "row",
                "false-halt",
                "runaway recall",
                "attacked recall",
                "median calls onset→halt",
                "prec@base",
                "first-flag false-halt",
                "first-flag runaway",
                "max sensitivity step",
                "selected θ_warn / θ_halt (calibration; – = undefined)",
            ],
            body,
        )
        L += _approx_note(halts)
        src = halts.get(theta_row)
        if src is not None:
            L += ["", f"**Sensitivity curve, {theta_row}** (false-halt / runaway recall /"]
            L += ["precision at base rate per θ; calibration-partition benign share).", ""]
            cal = {c["theta"]: c for c in src["calibration_curve"] or ()}
            body = [
                [
                    _p(x["theta"], 2),
                    _p(x["false_halt_rate"], 4),
                    _p(x["recall_runaway"]),
                    _p(x["recall_attacked"]),
                    _p(x["precision_at_base_rate"]),
                    _p(cal[x["theta"]]["benign_share"], 4) if cal else "–",
                ]
                for x in src["sensitivity"]
            ]
            L += _table(
                [
                    "θ",
                    "false-halt",
                    "runaway recall",
                    "attacked recall",
                    "prec@base",
                    "calibration benign share",
                ],
                body,
            )
    L += ["", "**Sessions, splice, vital** (splice reported separately, T-40).", ""]
    body = []
    for n, s in rows.items():
        ss = s["sessions"]
        sp = ss["splice"]["warm_per_agent"]
        body.append(
            [
                n,
                _p(ss["session_auroc_all"]),
                _p(ss["session_auroc_length_matched"]),
                str(ss["n_length_matched_pairs"]),
                _p(ss["session_auroc_runaway"]),
                f"{_p(sp['attack_recall'])} ({sp['n_detected']}/{sp['n_total']})",
                _p(ss["splice"]["decision_recall"]),
                _p(ss["vital_false_flag_rate"], 4),
            ]
        )
    L += _table(
        [
            "row",
            "session AUROC (all)",
            "session AUROC (length-matched)",
            "pairs",
            "runaway session AUROC",
            "splice recall, per-agent 95 thr (warm)",
            "splice decision recall",
            "vital false-flag (benign sessions)",
        ],
        body,
    )
    att = b.get("attribution")
    if att is not None:
        gnames = list(att["groups"])
        L += [
            "",
            f"**Attribution** ({att['engine']}; attacks carried per group, leave-one-out).",
            "",
        ]
        body = [
            [f, *[str(v.get(g, 0)) for g in gnames], str(v.get(REDUNDANT, 0))]
            for f, v in att["table"].items()
        ]
        L += _table(["family", *gnames, REDUNDANT], body)
        L += ["", f"Detected with every group: {att['n_detected_full']}. T-20 totals hold: "]
        L[-1] += f"{_p(att['t20_totals_by_construction'])} (by construction of the table). "
        L[-1] += "Removing a group changes exactly its rows (F10, T-52): "
        L[-1] += f"{_p(att['literal_exact'])}."
        gp = att["groups"]
        L += [""]
        L += _table(
            ["group", "carried", "gained when removed", "detected without"],
            [
                [g, str(gp[g]["carried"]), str(gp[g]["gained"]), str(gp[g]["detected_without"])]
                for g in gnames
            ],
        )
    L.append("")
    return L


def _short_sha(sha: object) -> str:
    """12 hex digits, keeping the ``-dirty`` marker of a run on an uncommitted tree (T-63)."""
    text = str(sha)
    return text[:12] + ("-dirty" if text.endswith("-dirty") else "")


def render_markdown(report: Mapping[str, Any]) -> str:
    """The report as markdown; every table carries its protocol label."""
    meta = report["meta"]
    L = ["# Trajectory benchmark", ""]
    L.append(f"Numbers: git `{_short_sha(meta['git_sha'])}`, created {meta['created']}. ")
    if meta.get("rendered_git_sha"):
        L[-1] += f"Re-rendered (no replay) by git `{_short_sha(meta['rendered_git_sha'])}`. "
    L[-1] += "Split: time, "
    L[-1] += f"calibration partition {meta['split']['calibration_frac']} of agents (F10)."
    L.append("")
    L += _table(
        ["seed", "traces sha", "envelopes", "attacks", "families", "bench config"],
        [
            [
                s,
                str(d["traces_sha"])[:12],
                str(d["counts"].get("envelopes")),
                str(d["counts"].get("attacks")),
                ", ".join(d["families"]),
                _p(d["is_bench_config"]),
            ]
            for s, d in meta["datasets"].items()
        ],
    )
    L += ["", "Rows (each switch toggled alone on top of `engine`, plus composites):", ""]
    L += _table(
        ["row", "detector", "overrides", "note"],
        [
            [r["name"], r["detector"], f"`{orjson.dumps(r['overrides']).decode()}`", r["note"]]
            for r in meta["rows"]
        ],
    )
    L += ["", f"Δ method: {meta['delta_method']}, n = {meta['n_boot_delta']}.", ""]
    for p in report["protocols"].values():
        label = p["label"]
        head = f"Protocol: {label}"
        if p["optimistic"]:
            head += " — OPTIMISTIC: labels act as a perfect reviewer"
        L += [f"## {head}", ""]
        for seed, b in p["seeds"].items():
            L += _seed_block(seed, b, label, meta["theta_source_row"])
    lat = report.get("latency")
    if lat:
        L += ["## Latency (alternated A/B, spec §5.2 allow-all updates)", ""]
        L += [f"{lat['method']}; {lat['n_timed']} timed calls after {lat['warmup_excluded']}"]
        L[-1] += " warm-up calls, first seed's dataset."
        L += [""]
        L += _table(
            ["row", "p50 µs", "p99 µs", "mean µs", "p50 / engine", "p99 / engine"],
            [
                [
                    n,
                    _p(v["p50_us"], 1),
                    _p(v["p99_us"], 1),
                    _p(v["mean_us"], 1),
                    _p(v["ratio_p50"], 2),
                    _p(v["ratio_p99"], 2),
                ]
                for n, v in lat["rows"].items()
            ],
        )
        L.append("")
    w = meta["wall_s"]
    L += [f"Wall time: {orjson.dumps(w).decode()}", ""]
    return "\n".join(L)
