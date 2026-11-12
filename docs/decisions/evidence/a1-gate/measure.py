"""Task 1.23: Amendment A1's gate matrix on SW1 seeds 0-2 under both protocols (driven by run.sh).

    uv run python docs/decisions/evidence/a1-gate/measure.py gen SEED OUT_DIR [FAR_MULT]
    uv run python docs/decisions/evidence/a1-gate/measure.py cell TRACES_DIR ENGINE PROTOCOL OUT.json
    uv run python docs/decisions/evidence/a1-gate/measure.py merge CELLS_DIR
    uv run python docs/decisions/evidence/a1-gate/measure.py latency P99_US|null [NOTE]

``gen``: ``generate(sw1_config(SEED))``; with FAR_MULT the same config with that
``magnitude_far_mult`` (the sensitivity dataset uses 1.0). Nothing else of the config changes.

``cell``: one replay of one (dataset, engine, protocol). ENGINE ``a1`` is ``EngineConfig()`` (the
gate CLI's ``ENGINE_ROW``), ``pre-a1`` is ``EngineConfig().replace(**PRE_A1_SWITCHES)`` (its
``PRE_A1_ROW``); the gate numbers are ``gate_preview`` of the rows of
``replay_detector(traces, "engine", SplitSpec(), cfg, protocol)``, which is exactly what
``agentfw.bench.gate.gate_matrix`` computes. The cell is replayed here and not through the gate
CLI because the matrix also needs per-row benign-side numbers (the share of benign scores >= 0.5)
that the CLI's ``--json`` does not carry; one replay per cell gives both, so no dataset is
replayed twice. The cell file keeps the gate CLI's ``results`` shape, so
``agentfw.bench.gate.load_gate_json`` reads it.

Benign-side shares: warm test rows labelled benign that are not ``benign_unusual`` controls; per
row, the detector's highest numeric score; the share is (rows whose score >= 0.5) / (rows where
the detector reported a score). Census: the ``magnitude_spike`` gate cell (warm test rows, bucket
``"10-100"``) of the amended engine under the oracle-reviewer loop, and, over every spike of the
dataset (labels.jsonl, one strength per attack id), the share with strength >= 10 and the median.

``merge``: reads the cell files and each dataset's ``manifest.json`` and writes ``matrix.json``
and ``matrix.md`` next to this file (``matrix.md`` via ``render_gate_matrix``; the sensitivity
dataset's rows are passed as ``sensitivity`` so the ``magnitude_spike`` cells are shown with their
companion rows, spec A1.5 rule 4). ``latency_p99_us`` is kept from an existing ``matrix.json``.
``merge`` refuses an incomplete set: every cell of seeds 0 and 1 and of the sensitivity dataset,
and all four of seed 2 once any is there. ``latency``: writes that number (null: no idle-machine
measurement, spec A1.8) and NOTE (how it was measured, or why it is null) and re-renders
``matrix.md``.

This script imports two private helpers of ``agentfw.bench.metrics`` (``_bucket_of``,
``_numeric_score``) and sits outside the gate's ruff and mypy scope (``docs`` is excluded);
``tests/bench/test_a1_evidence.py`` imports it and renders the committed ``matrix.json`` with it,
so a rename of either helper, or a ``GatePreview`` change that breaks ``_preview``, fails the gate.
"""  # noqa: E501 - the usage lines are kept whole

from __future__ import annotations

import math
import statistics
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import orjson

from agentfw.bench.ablation import replay_detector
from agentfw.bench.gate import (
    ENGINE_ROW,
    ORIGINAL_CRITERION_LABEL,
    PRE_A1_ROW,
    load_gate_json,
    render_gate_matrix,
)
from agentfw.bench.metrics import (
    GATE_FAMILIES,
    GateCell,
    GatePreview,
    _bucket_of,
    _numeric_score,
    gate_preview,
)
from agentfw.bench.replay import (
    PROTOCOL_LABELS,
    OutcomeMode,
    ScoreRow,
    SplitSpec,
    canonical_hash,
    is_outcome_mode,
)
from agentfw.bench.sweeps import PRE_A1_SWITCHES
from agentfw.core.config import EngineConfig
from agentfw.synth.generate import GenerateConfig, generate, sw1_config

HERE = Path(__file__).resolve().parent
ENGINES: dict[str, tuple[str, EngineConfig]] = {
    "a1": (ENGINE_ROW, EngineConfig()),
    "pre-a1": (PRE_A1_ROW, EngineConfig().replace(**PRE_A1_SWITCHES)),
}
PROTOCOLS: tuple[OutcomeMode, ...] = ("review", "allow")
DATASETS = {("sw1", 0): "sw1-seed0", ("sw1", 1): "sw1-seed1", ("sw1", 2): "sw1-seed2"}
DATASETS[("sens", 0)] = "sw1-sens-seed0"
SHARE_DETECTORS = ("magnitude", "velocity")
LABEL = {p: PROTOCOL_LABELS[p] for p in PROTOCOLS}  # verbatim (spec A1.5 rule 1)


# --- gen ---------------------------------------------------------------------------------------


def gen(seed: int, out: Path, far_mult: float | None) -> None:
    cfg = sw1_config(seed)
    if far_mult is not None:  # validated like any config (magnitude_far_mult > 0)
        cfg = GenerateConfig(**{**cfg.model_dump(), "magnitude_far_mult": far_mult})
    m = generate(cfg, out)
    print(f"{out}: traces {m.files['traces.jsonl'][:12]} counts {m.counts}")


# --- cell --------------------------------------------------------------------------------------


def _tops(r: ScoreRow) -> dict[str, float]:
    """Highest numeric score per detector on one row."""
    tops: dict[str, float] = {}
    for item in orjson.loads(r.per_detector_json):
        s = _numeric_score(item)
        d = str(item.get("detector"))
        if s is not None and s > tops.get(d, -math.inf):
            tops[d] = s
    return tops


def _share(hits: int, n: int) -> float:
    return hits / n if n else math.nan


def benign_shares(warm: Sequence[ScoreRow]) -> tuple[dict[str, float], dict[str, float]]:
    """(share of benign scores >= 0.5 per detector, per-agent share for velocity)."""
    tot: dict[str, list[int]] = {d: [0, 0] for d in SHARE_DETECTORS}
    vel: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in warm:
        if r.label == "attack" or r.control_type == "benign_unusual":
            continue
        t = _tops(r)
        for d in SHARE_DETECTORS:
            if d in t:
                tot[d][0] += t[d] >= 0.5
                tot[d][1] += 1
        if "velocity" in t:
            vel[r.agent_id][0] += t["velocity"] >= 0.5
            vel[r.agent_id][1] += 1
    return (
        {d: _share(*v) for d, v in tot.items()},
        {a: _share(*vel[a]) for a in sorted(vel)},
    )


def census(warm: Sequence[ScoreRow], traces: Path) -> dict[str, Any]:
    first: dict[str, ScoreRow] = {}
    for r in sorted(warm, key=lambda r: (r.ts_unix_ms, r.envelope_id)):
        if r.label == "attack" and r.attack_family == "magnitude_spike" and r.attack_id:
            first.setdefault(r.attack_id, r)
    bucket = GATE_FAMILIES["magnitude_spike"][0]
    gated = [r for r in first.values() if _bucket_of("magnitude_spike", r.strength) == bucket]
    by_persona: dict[str, int] = defaultdict(int)
    for r in gated:
        by_persona[r.agent_id] += 1
    strength: dict[str, float] = {}
    for line in (traces / "labels.jsonl").read_bytes().splitlines():
        lab = orjson.loads(line)
        if lab["attack_family"] == "magnitude_spike" and lab["strength"] is not None:
            strength.setdefault(lab["attack_id"], float(lab["strength"]))
    s = list(strength.values())
    return {
        "gated_test_n": len(gated),
        "by_persona": dict(sorted(by_persona.items())),
        "share_strength_ge_10": _share(sum(x >= 10.0 for x in s), len(s)),
        "median_strength": statistics.median(s) if s else math.nan,
        "n_spikes": len(s),
    }


def cell(traces: Path, engine: str, protocol: str, out: Path) -> None:
    name, cfg = ENGINES[engine]
    if not (is_outcome_mode(protocol) and protocol in PROTOCOLS):
        raise SystemExit(f"protocol must be one of {PROTOCOLS}, got {protocol!r}")
    rows, _ = replay_detector(traces, "engine", SplitSpec(), cfg, protocol)
    g = gate_preview(rows)
    warm = [r for r in rows if r.split == "test" and r.is_warm and not math.isnan(r.fused_score)]
    share, per_agent = benign_shares(warm)
    manifest = orjson.loads((traces / "manifest.json").read_bytes())
    payload = {
        "traces": str(traces),
        "engine": engine,
        "protocol": protocol,
        "dataset_sha": manifest["files"]["traces.jsonl"][:12],
        "score_hash": canonical_hash(rows)[:12],
        "labels": {m: PROTOCOL_LABELS[m] for m in PROTOCOLS},
        "results": {name: {protocol: g}},  # the gate CLI's --json shape (load_gate_json)
        "sensitivity": None,
        "benign_share_ge_0_5": share,
        "per_agent_benign_velocity_share_ge_0_5": per_agent,
        "census": census(warm, traces) if (engine, protocol) == ("a1", "review") else None,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    print(f"{out.name}: {'PASS' if g.passed else 'FAIL ' + ','.join(g.failures)}")


# --- merge -------------------------------------------------------------------------------------


def _nan(x: float | None) -> float:
    return math.nan if x is None else float(x)


def _row(dataset: str, seed: int, c: dict[str, Any], path: Path) -> dict[str, Any]:
    results, _ = load_gate_json(path)
    g = results[ENGINES[c["engine"]][0]][c["protocol"]]
    return {
        "dataset": dataset,
        "seed": seed,
        "dataset_sha": c["dataset_sha"],
        "engine": c["engine"],
        "protocol": c["protocol"],
        "protocol_label": PROTOCOL_LABELS[c["protocol"]],
        "score_hash": c["score_hash"],
        "families": {
            fam: {
                "recall": x.recall,
                "n_detected": x.n_detected,
                "n_total": x.n_total,
                "among": x.among_reasons_share,
                "top1": x.top_reason_share,
            }
            for fam, x in g.cells.items()
        },
        "benign_fpr": g.benign_fpr,
        "control_fpr": g.control_fpr,
        "benign_share_ge_0_5": c["benign_share_ge_0_5"],
        "per_agent_benign_velocity_share_ge_0_5": c["per_agent_benign_velocity_share_ge_0_5"],
        "passed": g.passed,
        "failures": list(g.failures),
    }


def _missing_cells(cells_dir: Path) -> list[str]:
    """Cell files a matrix needs that are not written: every cell of seeds 0 and 1 and of the
    sensitivity dataset (stage seeds01), and all four of seed 2 once any is there (stage seed2)."""
    missing: list[str] = []
    for key, d in DATASETS.items():
        paths = [cells_dir / f"{d}--{e}--{p}.json" for e in ENGINES for p in PROTOCOLS]
        if key != ("sw1", 2) or any(p.exists() for p in paths):
            missing += [str(p) for p in paths if not p.exists()]
    return missing


def merge(cells_dir: Path) -> None:
    missing = _missing_cells(cells_dir)
    if missing:  # never overwrite the matrix with part of it
        raise SystemExit("merge: cell files missing: " + ", ".join(missing))
    rows: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    cens: dict[str, dict[str, Any]] = {}
    traces: dict[str, str] = {}
    for (dataset, seed), d in DATASETS.items():
        for engine in ENGINES:
            for protocol in PROTOCOLS:
                path = cells_dir / f"{d}--{engine}--{protocol}.json"
                if not path.exists():  # seed 2 before stage seed2 (_missing_cells checked)
                    continue
                c = orjson.loads(path.read_bytes())
                rows.append(_row(dataset, seed, c, path))
                traces[d] = c["traces"]
                if dataset == "sw1" and c["census"] is not None:
                    cens[str(seed)] = c["census"]
        if dataset == "sw1" and d in traces:
            manifest = orjson.loads((Path(traces[d]) / "manifest.json").read_bytes())
            counts[str(seed)] = manifest["counts"]
    old = HERE / "matrix.json"
    prev = orjson.loads(old.read_bytes()) if old.exists() else {}
    m = {
        "rows": rows,
        "counts": counts,
        "census": cens,
        "latency_p99_us": prev.get("latency_p99_us"),
        "latency_note": prev.get("latency_note"),
        "traces": traces,
    }
    _write(m)


def _write(m: dict[str, Any]) -> None:
    (HERE / "matrix.json").write_bytes(
        orjson.dumps(m, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS) + b"\n"
    )
    (HERE / "matrix.md").write_text(render(m))
    print(f"wrote {HERE / 'matrix.json'} ({len(m['rows'])} rows) and matrix.md")


# --- render ------------------------------------------------------------------------------------


def _f(x: float | None, nd: int = 3) -> str:
    return "n/a" if x is None or math.isnan(x) else f"{x:.{nd}f}"


def _pct(x: float | None, nd: int = 2) -> str:
    return "n/a" if x is None or math.isnan(x) else f"{100 * x:.{nd}f} %"


def _find(m: dict[str, Any], dataset: str, seed: int, engine: str, protocol: str) -> Any:
    for r in m["rows"]:
        if (r["dataset"], r["seed"], r["engine"], r["protocol"]) == (
            dataset,
            seed,
            engine,
            protocol,
        ):
            return r
    return None


def _mag(r: Any) -> tuple[float, int, int]:
    c = r["families"].get("magnitude_spike") if r else None
    if not c:
        return math.nan, 0, 0
    return _nan(c["recall"]), int(c["n_detected"]), int(c["n_total"])


# Spec A1.11 / the Task 1.23 brief (prototype, base f185072): expected values, seeds 0 and 1.
EXP_SHA = {0: "a455b10c705d", 1: "e783c301bf21"}
EXP_SPIKES = {0: (53, 2, 211), 1: (49, 1, 203)}
EXP_GATE_N = {0: 81, 1: 85}
#: family -> seed -> protocol -> (n_detected, n_total)
EXP_A1: dict[str, dict[int, dict[str, tuple[int, int]]]] = {
    "magnitude_spike": {
        0: {"review": (81, 81), "allow": (76, 81)},
        1: {"review": (85, 85), "allow": (70, 85)},
    },
    "scope_blowup": {
        0: {"review": (13, 13), "allow": (13, 13)},
        1: {"review": (24, 24), "allow": (24, 24)},
    },
    "exfil_volume": {
        0: {"review": (19, 19), "allow": (11, 19)},
        1: {"review": (29, 29), "allow": (19, 29)},
    },
    "prereq_skip": {
        0: {"review": (15, 15), "allow": (8, 15)},
        1: {"review": (18, 18), "allow": (12, 18)},
    },
    "novel_target": {
        0: {"review": (49, 49), "allow": (8, 49)},
        1: {"review": (36, 36), "allow": (8, 36)},
    },
    "velocity_burst": {
        0: {"review": (50, 50), "allow": (40, 50)},
        1: {"review": (51, 52), "allow": (42, 52)},
    },
}
EXP_FPR = {
    0: {"review": (0.0058, 0.035), "allow": (0.0014, 0.007)},
    1: {"review": (0.0062, 0.028), "allow": (0.0023, 0.013)},
}
#: (dataset, seed, engine) -> protocol -> expected magnitude recall
EXP_MAG = {
    ("sw1", 0, "pre-a1"): {"review": 0.975, "allow": 0.753},
    ("sw1", 1, "pre-a1"): {"review": 0.988, "allow": 0.788},
    ("sens", 0, "a1"): {"review": 0.797, "allow": 0.656},
    ("sens", 0, "pre-a1"): {"review": 0.688, "allow": 0.578},
}
EXP_MAG_SHARE = {0: 0.0047, 1: 0.0052}
#: the magnitude cell's companion rows in the comparison (spec A1.5 rule 4); seed 2 has no
#: expectation (unused seed) but its magnitude cell is shown there, so its companion is too
COMPANIONS = [
    ("sw1", 0, "pre-a1"),
    ("sw1", 1, "pre-a1"),
    ("sw1", 2, "pre-a1"),
    ("sens", 0, "a1"),
    ("sens", 0, "pre-a1"),
]
TIER3_CONTROL_FPR = 0.05  # the Tier-3 bound A1.12 item 7 cites (recorded, not a stop condition)


def _within_one(recall: float, exp: float, n: int) -> str:
    if math.isnan(recall) or n == 0:
        return "no"
    return "yes" if abs(recall - exp) <= 1.0 / n + 1e-9 else "no"


def _gate_cell(r: Any, fam: str) -> tuple[float, float, int, int]:
    cl = r["families"].get(fam) if r else None
    if not cl:
        return math.nan, math.nan, 0, 0
    return _nan(cl["recall"]), _nan(cl["among"]), int(cl["n_detected"]), int(cl["n_total"])


def _verdict(r: Any) -> str:
    return "PASS" if r["passed"] else "FAIL: " + ", ".join(r["failures"])


def _comparison(m: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Expected (spec A1.11) vs measured lines, and the brief's stop conditions that hold.

    Two tables: dataset checks, then the gate checks with the oracle-reviewer value and the
    allow-all value of each check on one line, each under its label verbatim (spec A1.5 rule 1).
    """
    stops: list[str] = []
    data = [
        "| check | seed | expected | measured | match | stop if | holds |",
        "|---|---|---|---|---|---|---|",
    ]
    gate = [
        "| check | seed | "
        + " | ".join(f"{w}, {LABEL[p]}" for p in PROTOCOLS for w in ("expected", "measured"))
        + " | match (review / allow) | stop if | holds |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    def hold(check: str, seed: Any, meas: str, holds: bool) -> str:
        if holds:
            stops.append(f"{check} (seed {seed}): {meas}")
        return "**YES**" if holds else "no"

    def add_data(
        check: str, seed: Any, exp: str, meas: str, match: str, stop: str, h: bool
    ) -> None:
        holds = hold(check, seed, meas, h)
        data.append(f"| {check} | {seed} | {exp} | {meas} | {match} | {stop} | {holds} |")

    def add_gate(
        check: str, seed: Any, cells: list[tuple[str, str]], match: str, stop: str, h: bool
    ) -> None:
        flat = " | ".join(f"{e} | {v}" for e, v in cells)
        meas = " / ".join(v for _, v in cells)
        gate.append(
            f"| {check} | {seed} | {flat} | {match} | {stop} | {hold(check, seed, meas, h)} |"
        )

    for seed in (0, 1, 2):
        by = {p: _find(m, "sw1", seed, "a1", p) for p in PROTOCOLS}
        if by["review"] is None or by["allow"] is None:
            continue
        sha = by["review"]["dataset_sha"]
        if seed in EXP_SHA:
            add_data(
                "dataset sha (traces)",
                seed,
                f"`{EXP_SHA[seed]}`",
                f"`{sha}`",
                "yes" if sha == EXP_SHA[seed] else "no",
                "different",
                sha != EXP_SHA[seed],
            )
            c = m["counts"][str(seed)]
            got = (c["magnitude_first_pass"], c["magnitude_dropped"], c["magnitude_topup"])
            add_data(
                "spikes: first pass / dropped / top-up",
                seed,
                " / ".join(map(str, EXP_SPIKES[seed])),
                " / ".join(map(str, got)),
                "yes" if got == EXP_SPIKES[seed] else "no",
                "different",
                got != EXP_SPIKES[seed],
            )
        n = m["census"][str(seed)]["gated_test_n"]
        add_data(
            "magnitude gate cell n",
            seed,
            str(EXP_GATE_N.get(seed, "none (unused seed)")),
            str(n),
            ("yes" if n == EXP_GATE_N[seed] else "no") if seed in EXP_GATE_N else "-",
            "below 50",
            n < 50,
        )
        for fam in GATE_FAMILIES:
            cells, matches, holds = [], [], False
            for proto in PROTOCOLS:
                rec, among, nd, nt = _gate_cell(by[proto], fam)
                meas = f"{_f(rec)} ({nd}/{nt}); among {_f(among, 2)}"
                if seed in EXP_SHA:
                    ed, et = EXP_A1[fam][seed][proto]
                    cells.append((f"{ed / et:.3f} ({ed}/{et}); among 1.00", meas))
                    matches.append(_within_one(rec, ed / et, nt))
                else:
                    cells.append(("none (unused seed)", meas))
                    matches.append("-")
                if proto == "review" and seed in EXP_SHA:
                    holds = not (rec >= 0.90) or not (among >= 0.90)
            stop = (
                "review: recall or among below 0.90; allow: expected to fail, recorded"
                if seed in EXP_SHA
                else "see the amended gate line"
            )
            add_gate(fam, seed, cells, " / ".join(matches), stop, holds)
        cells = []
        for proto in PROTOCOLS:
            r = by[proto]
            exp = (
                f"{_pct(EXP_FPR[seed][proto][0])}; {_pct(EXP_FPR[seed][proto][1])}"
                if seed in EXP_FPR
                else "none (unused seed)"
            )
            cells.append((exp, f"{_pct(_nan(r['benign_fpr']))}; {_pct(_nan(r['control_fpr']))}"))
        fpr = _nan(by["review"]["benign_fpr"])
        add_gate(
            "benign FPR; control FPR",
            seed,
            cells,
            "-",
            "review: benign FPR above 2 %",
            not (fpr <= 0.02),
        )
        cells = [
            (
                _pct(EXP_MAG_SHARE[seed])
                if (seed in EXP_MAG_SHARE and p == "review")
                else "none (unused seed)"
                if p == "review"
                else "not stated",
                _pct(_nan(by[p]["benign_share_ge_0_5"]["magnitude"])),
            )
            for p in PROTOCOLS
        ]
        share = _nan(by["review"]["benign_share_ge_0_5"]["magnitude"])
        add_gate(
            "share of benign magnitude scores >= 0.5",
            seed,
            cells,
            "-",
            "review: above 1.0 %",
            not (share <= 0.01),
        )
        exp_v = (
            {"review": "PASS", "allow": "FAIL"}
            if seed in EXP_SHA
            else {"review": "PASS", "allow": "none (unused seed)"}
        )
        cells = [(exp_v[p], _verdict(by[p])) for p in PROTOCOLS]
        add_gate(
            "amended gate (A1.7)",
            seed,
            cells,
            "-",
            "seed 2: fails under review" if seed == 2 else "(covered by the lines above)",
            seed == 2 and not by["review"]["passed"],
        )
    for dataset, seed, engine in COMPANIONS:
        exp_by = EXP_MAG.get((dataset, seed, engine))
        if exp_by is None and all(_find(m, dataset, seed, engine, p) is None for p in PROTOCOLS):
            continue  # seed 2 before stage seed2
        cells, matches = [], []
        for proto in PROTOCOLS:
            rec, nd, nt = _mag(_find(m, dataset, seed, engine, proto))
            if exp_by is None:
                cells.append(("none (unused seed)", f"{_f(rec)} ({nd}/{nt})"))
                matches.append("-")
            else:
                cells.append((f"{exp_by[proto]:.3f}", f"{_f(rec)} ({nd}/{nt})"))
                matches.append(_within_one(rec, exp_by[proto], nt))
        add_gate(
            f"companion: {dataset} dataset, {engine} engine, magnitude_spike 10-100",
            seed,
            cells,
            " / ".join(matches),
            "-",
            False,
        )
    return [*data, "", *gate], stops


def render(m: dict[str, Any]) -> str:
    rv, al = LABEL["review"], LABEL["allow"]
    lines = [
        "<!-- Generated by docs/decisions/evidence/a1-gate/run.sh (measure.py merge); do not edit."
        " -->",
        "",
        "# Amendment A1 gate matrix: SW1 seeds 0-2 under both protocols (Task 1.23)",
        "",
        "First measurement of the implemented combination (A1.1 velocity, A1.2 magnitude, A1.3 BOOL"
        " numerics, A1.4 far spikes with the top-up). No detector, generator rule, constant or "
        "dataset was changed for this measurement. Datasets: `sw1_config(seed)` for seeds 0, 1, 2 "
        "and, as the sensitivity dataset, `sw1_config(0)` with `magnitude_far_mult = 1.0`. Seed 2 "
        "was generated and replayed only after seeds 0 and 1 were written to `matrix.json`.",
        "",
        f"Engines: `a1` = `{ENGINE_ROW}` (`EngineConfig()`), `pre-a1` = `{PRE_A1_ROW}`. Each "
        "(dataset, engine, protocol) was replayed once; gate numbers are "
        "`agentfw.bench.metrics.gate_preview` on warm test rows at live per-agent thresholds, "
        "attack level (as `agentfw.bench.gate.gate_matrix`). Protocols, verbatim:",
        "",
        f"- review: {rv}",
        f"- allow: {al}",
        "",
        "Which protocol's result closes Tier 1 is the user's open decision (D-028); a pass under "
        f"the {rv} is a pass under that protocol only, shown next to the {al} result, and is not "
        "the Tier-1 gate being met.",
        "",
        "## Datasets and the magnitude census",
        "",
        "| dataset | traces sha | envelopes | attacks | spikes: first pass / dropped / top-up | "
        "spikes with strength >= 10 | gate cell n (warm test, 10-100) | share of spikes with "
        "strength >= 10 | median strength | gate cell by persona |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for seed in (0, 1, 2):
        c, cs = m["counts"].get(str(seed)), m["census"].get(str(seed))
        r = _find(m, "sw1", seed, "a1", "review")
        if c is None or cs is None or r is None:
            continue
        by = sorted(cs["by_persona"].items(), key=lambda kv: (-kv[1], kv[0]))
        persona = ", ".join(f"{a} {n}" for a, n in by)
        lines.append(
            f"| sw1 seed {seed} | `{r['dataset_sha']}` | {c['envelopes']} | {c['attacks']} | "
            f"{c['magnitude_first_pass']} / {c['magnitude_dropped']} / {c['magnitude_topup']} | "
            f"{c['magnitude_strength_ge_10']} | {cs['gated_test_n']} | "
            f"{_f(cs['share_strength_ge_10'])} | {_f(cs['median_strength'], 1)} | {persona} |"
        )
    sens = _find(m, "sens", 0, "a1", "review")
    if sens is not None:
        lines.append(
            f"| sensitivity (seed 0, magnitude_far_mult = 1.0) | `{sens['dataset_sha']}` | | | | |"
            f" {_mag(sens)[2]} | | | |"
        )
    lines += [""]
    sp = m["traces"].get(DATASETS[("sens", 0)])
    sens_rows = _rows(m, "sens", 0) or None
    for seed in (0, 1, 2):
        res = _rows(m, "sw1", seed)
        if not res:
            continue
        lines += [
            f"## SW1 seed {seed} (`{m['traces'].get(DATASETS[('sw1', seed)])}`)",
            "",
            "The companion rows (spec A1.5 rule 4) are the engine before Amendment A1 on this "
            f"dataset and, after the main blocks, both engines on the seed-0 sensitivity dataset "
            f"(`{sp}`; the sensitivity dataset exists for seed 0 only).",
            "",
            _label_sens_seed(_demote(render_gate_matrix(res, sensitivity=sens_rows)))
            if sens_rows
            else _demote(render_gate_matrix(res, sensitivity=sens_rows)),
            "",
        ]
    lines += _shares_md(m)
    comp, stops = _comparison(m)
    lines += [
        "## Expected (spec A1.11, prototype) vs measured",
        "",
        "Recalls within one attack of the expectation count as a match; score hashes are not "
        f"expected to match the prototype. Gate cells: recall (n/N); among-the-reasons share "
        f"(the {ORIGINAL_CRITERION_LABEL} top-1 share is in the gate tables above).",
        "",
        *comp,
        "",
        "Stop conditions of the brief (Step 4) that hold: "
        + ("; ".join(stops) if stops else "none")
        + ".",
        "",
        _tier3_note(m),
        "",
        "## Latency",
        "",
    ]
    lat, note = m.get("latency_p99_us"), m.get("latency_note")
    lines.append(
        f"`Engine.score` p99 = {lat:.1f} us (`tests/perf/test_latency.py`; budget 200 us)."
        if lat is not None
        else "`latency_p99_us` is null: not measured on an idle machine (a number measured under "
        "load is not pinned; spec A1.8)."
    )
    if note:
        lines += ["", str(note)]
    lines.append("")
    return "\n".join(lines)


def _tier3_note(m: dict[str, Any]) -> str:
    """The amended engine's control FPRs under the oracle-reviewer loop above the Tier-3 bound,
    at two decimals (the gate tables print 0.050 for 0.0501)."""
    over = [
        f"sw1 seed {r['seed']}: {_pct(_nan(r['control_fpr']))}"
        for r in m["rows"]
        if (r["dataset"], r["engine"], r["protocol"]) == ("sw1", "a1", "review")
        and _nan(r["control_fpr"]) > TIER3_CONTROL_FPR
    ]
    return (
        f"Control FPR of the amended engine under the {LABEL['review']}, above the 5 % Tier-3 "
        "bound (A1.12 item 7, which states seeds 0 and 1 stay below it): "
        + ("; ".join(over) if over else "none")
        + ". Not a stop condition of the brief (Step 4); recorded."
    )


SENS_HEADING = "Companion: sensitivity dataset (magnitude_far_mult = 1.0)"


def _label_sens_seed(md: str) -> str:
    """Name the sensitivity companion's seed under every seed's block (it exists for seed 0)."""
    if md.count(SENS_HEADING) != 1:
        raise SystemExit(f"render: expected one {SENS_HEADING!r} heading in the gate block")
    return md.replace(SENS_HEADING, SENS_HEADING.replace("(", "(seed 0, ", 1))


def _demote(md: str) -> str:
    """Headings one level down, so each seed's gate tables sit under that seed's heading."""
    return "\n".join("#" + ln if ln.startswith("#") else ln for ln in md.splitlines())


def _rows(m: dict[str, Any], dataset: str, seed: int) -> dict[str, dict[Any, GatePreview]]:
    """The gate rows of one dataset keyed like the gate CLI's results (engine row, protocol)."""
    by_name: dict[str, dict[Any, GatePreview]] = {}
    for engine, (name, _) in ENGINES.items():
        for protocol in PROTOCOLS:
            r = _find(m, dataset, seed, engine, protocol)
            if r is not None:
                by_name.setdefault(name, {})[protocol] = _preview(r)
    return by_name


def _preview(r: dict[str, Any]) -> GatePreview:
    """A matrix row back as the ``GatePreview`` it was written from (``render_gate_matrix``)."""
    cells = {
        fam: GateCell(
            bucket=GATE_FAMILIES[fam][0],
            detector=GATE_FAMILIES[fam][1],
            recall=_nan(c["recall"]),
            n_detected=c["n_detected"],
            n_total=c["n_total"],
            top_reason_share=_nan(c["top1"]),
            among_reasons_share=_nan(c["among"]),
        )
        for fam, c in r["families"].items()
    }
    recalls = [c.recall for c in cells.values()]
    return GatePreview(
        cells=cells,
        benign_fpr=_nan(r["benign_fpr"]),
        control_fpr=_nan(r["control_fpr"]),
        min_recall=min(recalls) if recalls else math.nan,
        passed=r["passed"],
        missing=tuple(f for f in GATE_FAMILIES if f not in cells),
        failures=tuple(r["failures"]),
    )


def _shares_md(m: dict[str, Any]) -> list[str]:
    cols = [(s, p) for s in (0, 1, 2) for p in PROTOCOLS if _find(m, "sw1", s, "a1", p)]
    head = [f"sw1 seed {s}, {LABEL[p]}" for s, p in cols]
    lines = [
        "## Benign side: share of benign scores >= 0.5",
        "",
        "Warm test rows labelled benign, `benign_unusual` controls excluded; per row the detector's"
        " highest score; share = rows with score >= 0.5 / rows where the detector reported a score"
        " (the prior is 0.5 %).",
        "",
        "| dataset | engine | detector | " + " | ".join(head) + " |",
        "|---|---|---|" + "---|" * len(head),
    ]
    for engine in ENGINES:
        for d in SHARE_DETECTORS:
            vals = [_pct(_find(m, "sw1", s, engine, p)["benign_share_ge_0_5"][d]) for s, p in cols]
            lines.append(f"| sw1 | {engine} | {d} | " + " | ".join(vals) + " |")
    lines += [
        "",
        "### Per-agent share of benign velocity scores >= 0.5 (amended engine)",
        "",
        "Recorded with no threshold: the open condition of the no-gaming review (A1.12 item 9).",
        "",
        "| agent | " + " | ".join(head) + " |",
        "|---|" + "---|" * len(head),
    ]
    agents = sorted(
        {
            a
            for s, p in cols
            for a in _find(m, "sw1", s, "a1", p)["per_agent_benign_velocity_share_ge_0_5"]
        }
    )
    for a in agents:
        vals = [
            _pct(_find(m, "sw1", s, "a1", p)["per_agent_benign_velocity_share_ge_0_5"].get(a))
            for s, p in cols
        ]
        lines.append(f"| {a} | " + " | ".join(vals) + " |")
    lines.append("")
    return lines


def latency(value: str, note: str | None) -> None:
    m = orjson.loads((HERE / "matrix.json").read_bytes())
    m["latency_p99_us"] = None if value == "null" else float(value)
    m["latency_note"] = note
    _write(m)


def main(argv: Sequence[str]) -> None:
    cmd, *a = argv
    if cmd == "gen":
        gen(int(a[0]), Path(a[1]), float(a[2]) if len(a) > 2 else None)
    elif cmd == "cell":
        cell(Path(a[0]), a[1], a[2], Path(a[3]))
    elif cmd == "merge":
        merge(Path(a[0]))
    elif cmd == "latency":
        latency(a[0], a[1] if len(a) > 1 else None)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
