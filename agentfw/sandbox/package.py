"""Package sandbox run directories into the public benchmark dataset (Task 2.9).

    python -m agentfw.sandbox.package build --runs runs/a [runs/b ...] --out dataset/v1 --seed 0
    python -m agentfw.sandbox.package validate dataset/v1

``package`` takes each run's session list from its ``ledger.jsonl`` (the latest line per session)
and includes **only** ``completed`` sessions: orphan ``raw/`` files, ``failed`` and
``budget_truncated`` sessions are excluded (the manifest's ``source_runs`` counts them). Sessions
are merged, retimed onto the simulated calendar (D-015; ``velocity_burst`` envelopes, found from
the labels, keep their exact gaps), re-sorted by ``(ts, session_id, seq_no)``, checked (exactly one
label per envelope, every envelope normalizes, no secrets, only reserved email domains) and written
as ``traces.jsonl`` / ``labels.jsonl`` (same row shape as ``agentfw.synth.generate``, so
``agentfw.bench.dataset.load_dir`` reads both), ``traces.parquet`` / ``labels.parquet``,
``manifest.json`` (sha256 per file) and ``DATASET_CARD.md``.

Transcripts are **not** part of the public dataset: they hold model prose and tool output that the
benchmark does not need. They are never copied into the output directory.

The Parquet files carry ``args`` and ``response_meta`` as sorted-key JSON strings (their shape is
free-form); every other field is a native column. ``validate_dataset`` checks the minimum counts,
label integrity, manifest hashes and counts, that every envelope normalizes, JSONL == Parquet, zero
``scan_secrets`` hits and that every email domain is ``.example`` or ``.test`` (D-016).
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import orjson
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

from agentfw.core.envelope import Label, RawCall
from agentfw.core.normalize import SessionTracker, normalize
from agentfw.sandbox.retime import retime
from agentfw.sandbox.tasks.schema import AGENTS_DIR, load_agents

__all__ = [
    "DatasetManifest",
    "ManifestCounts",
    "ValidationReport",
    "main",
    "package",
    "scan_secrets",
    "validate_dataset",
]

START_MS = 1_767_225_600_000  # 2026-01-01T00:00:00Z: day 0 of the simulated timeline
CARD_TEMPLATE = Path(__file__).resolve().parents[2] / "docs" / "dataset_card.md"
_OPTS = orjson.OPT_SORT_KEYS
_FILES = ("traces.jsonl", "labels.jsonl", "traces.parquet", "labels.parquet")

_SECRET_PATTERNS = (
    re.compile(r"(?:sk|rk|pk)_(?:test|live)_[A-Za-z0-9]+"),  # Stripe
    re.compile(r"whsec_[A-Za-z0-9]+"),  # Stripe webhook secret
    re.compile(r"\b(?:sk|rk|pk)-[A-Za-z0-9_-]{20,}"),  # OpenAI-style (incl. sk-proj-)
    re.compile(r"xox[abposr]-[A-Za-z0-9-]+"),  # Slack
    re.compile(r"xapp-[A-Za-z0-9-]+"),  # Slack app-level
    # whitespace may be real or still JSON-escaped (\t, \n, \r) in raw serialized text
    re.compile(r"\bbearer(?:\s|\\[tnr])+[^\s\\\"]+", re.IGNORECASE),
    re.compile(r"00D[A-Za-z0-9]{12,15}![A-Za-z0-9._]+"),  # Salesforce session id
)
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*)")
_RESERVED_TLDS = (".example", ".test")


def scan_secrets(text: str) -> list[str]:
    """Every substring of ``text`` that looks like a credential (Stripe, Slack, bearer, SF)."""
    return [m.group(0) for p in _SECRET_PATTERNS for m in p.finditer(text)]


def _bad_domains(text: str) -> list[str]:
    seen = {m.group(1).lower().rstrip(".") for m in _EMAIL.finditer(text)}
    return sorted(d for d in seen if not d.endswith(_RESERVED_TLDS))


def _strings(v: object) -> Iterator[str]:
    """Every string in a decoded JSON value, dict keys included."""
    if isinstance(v, str):
        yield v
    elif isinstance(v, dict):
        for k, x in v.items():
            yield str(k)
            yield from _strings(x)
    elif isinstance(v, list):
        for x in v:
            yield from _strings(x)


def _text_problems(name: str, text: str) -> list[str]:
    hits = sorted(set(scan_secrets(text)))
    out = [f"{name}: secret-like string {s[:40]!r}" for s in hits]
    out += [f"{name}: email domain {d!r} is not .example/.test" for d in _bad_domains(text)]
    return out


def _content_problems(name: str, text: str) -> list[str]:
    """Secret and email findings in ``text``: the raw text and, for JSON / JSON lines, every
    decoded string value (so escapes cannot hide a secret)."""
    found = dict.fromkeys(_text_problems(name, text))
    if name.endswith((".json", ".jsonl")):
        for line in text.splitlines():
            try:
                value = orjson.loads(line) if name.endswith(".jsonl") else orjson.loads(text)
            except orjson.JSONDecodeError:
                continue
            for st in _strings(value):
                found.update(dict.fromkeys(_text_problems(name, st)))
            if not name.endswith(".jsonl"):
                break
    return list(found)


# --- models --------------------------------------------------------------------------------------


class ManifestCounts(BaseModel):
    agents: int
    sessions: int
    actions: int
    attacks: int  # distinct attack_id
    attack_envelopes: int
    by_family: dict[str, int]  # attack envelopes per attack_family
    by_injection_level: dict[str, int]  # attack envelopes per injection_level


class DatasetManifest(BaseModel):
    version: str
    files: dict[str, str]  # file name -> sha256 (DATASET_CARD.md included)
    counts: ManifestCounts
    source_runs: list[dict[str, Any]]
    model: str  # the agent model id (never a secret)
    config: dict[str, Any]  # packaging parameters: seed, start_ms
    created_unix: int


class ValidationReport(BaseModel):
    ok: bool
    problems: list[str]


# --- counting and encoding -----------------------------------------------------------------------


def _counts(raws: Sequence[RawCall], labels: Sequence[Label]) -> ManifestCounts:
    attacks = [lab for lab in labels if lab.label == "attack"]
    return ManifestCounts(
        agents=len({r.agent_id for r in raws}),
        sessions=len({r.session_id for r in raws}),
        actions=len(raws),
        attacks=len({lab.attack_id for lab in attacks}),
        attack_envelopes=len(attacks),
        by_family=dict(sorted(Counter(str(lab.attack_family) for lab in attacks).items())),
        by_injection_level=dict(
            sorted(Counter(str(lab.injection_level) for lab in attacks).items())
        ),
    )


def _jsonl(rows: Sequence[dict[str, object]]) -> bytes:
    return b"".join(orjson.dumps(r, option=_OPTS) + b"\n" for r in rows)


def _dump(model: BaseModel) -> dict[str, object]:
    return model.model_dump(mode="json")


def _json_str(v: object) -> str:
    return orjson.dumps(v, option=_OPTS).decode()


def _traces_table(raws: Sequence[RawCall]) -> pa.Table:
    rows = [_dump(r) for r in raws]
    return pa.table(
        {
            "envelope_id": pa.array([r["envelope_id"] for r in rows], pa.string()),
            "schema_version": pa.array([r["schema_version"] for r in rows], pa.int64()),
            "session_id": pa.array([r["session_id"] for r in rows], pa.string()),
            "agent_id": pa.array([r["agent_id"] for r in rows], pa.string()),
            "principal_id": pa.array([r["principal_id"] for r in rows], pa.string()),
            "seq_no": pa.array([r["seq_no"] for r in rows], pa.int64()),
            "ts_unix_ms": pa.array([r["ts_unix_ms"] for r in rows], pa.int64()),
            "tool_id": pa.array([r["tool_id"] for r in rows], pa.string()),
            "args": pa.array([_json_str(r["args"]) for r in rows], pa.string()),
            "response_meta": pa.array(
                [
                    None if r["response_meta"] is None else _json_str(r["response_meta"])
                    for r in rows
                ],
                pa.string(),
            ),
        }
    )


def _labels_table(labels: Sequence[Label]) -> pa.Table:
    rows = [_dump(lab) for lab in labels]
    strs = ("envelope_id", "label", "attack_family", "injection_level", "attack_id", "control_type")
    cols: dict[str, pa.Array[Any]] = {k: pa.array([r[k] for r in rows], pa.string()) for k in strs}
    cols["strength"] = pa.array([r["strength"] for r in rows], pa.float64())
    cols["step"] = pa.array([r["step"] for r in rows], pa.int64())
    return pa.table(cols)


def _write_parquet(table: pa.Table, path: Path) -> None:
    pq.write_table(table, path, compression="zstd")


# --- reading run directories ---------------------------------------------------------------------


def _read_ledger(run: Path) -> dict[str, dict[str, Any]]:
    path = run / "ledger.jsonl"
    if not path.is_file():
        raise ValueError(f"{run}: no ledger.jsonl (not a sandbox run directory)")
    latest: dict[str, dict[str, Any]] = {}
    for line in path.read_bytes().splitlines():
        try:
            row = orjson.loads(line)
        except orjson.JSONDecodeError:
            continue  # a torn line; the runner repairs these on its next start
        latest[row["session_id"]] = row
    return latest


def _read_meta(run: Path) -> dict[str, Any]:
    meta = run / "run_meta.json"
    if not meta.is_file():
        return {}
    loaded = orjson.loads(meta.read_bytes())
    return loaded if isinstance(loaded, dict) else {}


def _read_rows[T](path: Path, parse: Callable[[bytes], T], sid: str, run: Path) -> list[T]:
    try:
        return [parse(x) for x in path.read_bytes().splitlines()]
    except OSError as e:
        raise ValueError(f"session {sid} (run {run.name}): cannot read {path.name}: {e}") from e


def _check_session(sid: str, run: Path, raws: Sequence[RawCall], labels: Sequence[Label]) -> None:
    where = f"session {sid} (run {run.name})"
    if not raws:
        raise ValueError(f"{where}: completed in the ledger but has no raw calls")
    if {r.agent_id for r in raws} != {raws[0].agent_id} or {r.session_id for r in raws} != {sid}:
        raise ValueError(f"{where}: inconsistent agent_id or session_id across its calls")
    if [lab.envelope_id for lab in labels] != [r.envelope_id for r in raws]:
        raise ValueError(f"{where}: need exactly one label per envelope, in call order")


def _load_run(
    run: Path, sessions: dict[str, list[RawCall]], labels: dict[str, Label], seen: set[str]
) -> dict[str, Any]:
    ledger = _read_ledger(run)
    excluded: Counter[str] = Counter()
    included = 0
    for sid in sorted(ledger):
        status = str(ledger[sid].get("status"))
        if status != "completed":
            excluded[status] += 1
            continue
        if sid in seen:
            raise ValueError(f"duplicate session id {sid!r} across run directories")
        seen.add(sid)
        raws = _read_rows(run / "raw" / f"{sid}.jsonl", RawCall.model_validate_json, sid, run)
        labs = _read_rows(run / "labels" / f"{sid}.jsonl", Label.model_validate_json, sid, run)
        _check_session(sid, run, raws, labs)
        sessions[sid] = raws
        labels.update({lab.envelope_id: lab for lab in labs})
        included += 1
    meta = _read_meta(run)
    return {
        "name": run.name,
        "mode": str(meta.get("mode", "unknown")),
        "included": included,
        "excluded": dict(sorted(excluded.items())),
    }


# --- packaging -----------------------------------------------------------------------------------


def _check_normalizes(raws: Sequence[RawCall]) -> None:
    tracker = SessionTracker()
    for r in raws:
        tracker.observe(normalize(r, tracker))


_FAKE_BANNER = (
    "> **WARNING: FAKE DATA.** At least one source run used fake backends and a scripted agent "
    "(`--fake`). This dataset exercises the pipeline only; it is not a real agent benchmark.\n"
)


def _render_card(manifest: DatasetManifest) -> str:
    c = manifest.counts

    def table(d: dict[str, int]) -> str:
        return "\n".join(f"| {k} | {v} |" for k, v in d.items()) or "| (none) | 0 |"

    fields = {
        "version": manifest.version,
        "agents": str(c.agents),
        "sessions": str(c.sessions),
        "actions": str(c.actions),
        "attacks": str(c.attacks),
        "attack_envelopes": str(c.attack_envelopes),
        "by_family_rows": table(c.by_family),
        "by_injection_level_rows": table(c.by_injection_level),
        "model": manifest.model,
        "seed": str(manifest.config["seed"]),
        "fake_banner": _FAKE_BANNER
        if any(r["mode"] == "fake" for r in manifest.source_runs)
        else "",
    }
    if not CARD_TEMPLATE.is_file():
        raise FileNotFoundError(
            f"dataset card template missing: {CARD_TEMPLATE} (run from a source checkout)"
        )
    text = CARD_TEMPLATE.read_text()
    for k, v in fields.items():
        text = text.replace("{{" + k + "}}", v)
    return text


def package(
    run_dirs: Sequence[Path], out_dir: Path, *, version: str = "v1", seed: int = 0
) -> DatasetManifest:
    """Merge, retime, check and write the dataset; see the module docs."""
    sessions: dict[str, list[RawCall]] = {}
    labels: dict[str, Label] = {}
    seen: set[str] = set()
    sources = [_load_run(Path(r), sessions, labels, seen) for r in run_dirs]
    models = sorted({str(_read_meta(Path(r)).get("model", "unknown")) for r in run_dirs})
    burst = {e for e, lab in labels.items() if lab.attack_family == "velocity_burst"}
    raws = retime(sessions, load_agents(AGENTS_DIR), seed, START_MS, burst_ids=burst)
    final = [labels[r.envelope_id] for r in raws]
    traces_blob = _jsonl([_dump(r) for r in raws])
    labels_blob = _jsonl([_dump(lab) for lab in final])
    problems = _content_problems("traces.jsonl", traces_blob.decode())
    problems += _content_problems("labels.jsonl", labels_blob.decode())
    if problems:
        raise ValueError("refusing to package: " + "; ".join(problems))
    _check_normalizes(raws)

    manifest = DatasetManifest(
        version=version,
        files={},
        counts=_counts(raws, final),
        source_runs=sources,
        model=",".join(models),
        config={"seed": seed, "start_ms": START_MS},
        created_unix=int(time.time()),
    )
    card = _render_card(manifest)
    problems = _content_problems("DATASET_CARD.md", card)
    problems += _content_problems("manifest.json", manifest.model_dump_json())
    if problems:
        raise ValueError("refusing to package: " + "; ".join(problems))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "traces.jsonl").write_bytes(traces_blob)
    (out_dir / "labels.jsonl").write_bytes(labels_blob)
    _write_parquet(_traces_table(raws), out_dir / "traces.parquet")
    _write_parquet(_labels_table(final), out_dir / "labels.parquet")
    (out_dir / "DATASET_CARD.md").write_text(card)
    manifest.files = {
        n: hashlib.sha256((out_dir / n).read_bytes()).hexdigest()
        for n in (*_FILES, "DATASET_CARD.md")
    }
    (out_dir / "manifest.json").write_bytes(
        orjson.dumps(_dump(manifest), option=_OPTS | orjson.OPT_INDENT_2) + b"\n"
    )
    return manifest


# --- validation ----------------------------------------------------------------------------------


def _parquet_matches(d: Path, raws: Sequence[RawCall], labels: Sequence[Label]) -> list[str]:
    from agentfw.bench.dataset import load_labels_list, load_traces

    out: list[str] = []
    try:
        if load_traces(d / "traces.parquet") != list(raws):
            out.append("traces.parquet differs from traces.jsonl")
        if load_labels_list(d / "labels.parquet") != list(labels):
            out.append("labels.parquet differs from labels.jsonl")
    except Exception as e:  # unreadable or malformed Parquet is a validation finding
        out.append(f"parquet unreadable: {e}")
    return out


def validate_dataset(
    dir: Path,
    *,
    min_agents: int = 20,
    min_sessions: int = 2000,
    min_actions: int = 40_000,
    min_attacks: int = 600,
) -> ValidationReport:
    """Check a packaged dataset directory; never raises on bad data, reports problems instead."""
    d = Path(dir)
    problems: list[str] = []
    missing = [n for n in (*_FILES, "manifest.json", "DATASET_CARD.md") if not (d / n).is_file()]
    if missing:
        return ValidationReport(ok=False, problems=[f"missing files: {missing}"])
    try:
        manifest = DatasetManifest.model_validate_json((d / "manifest.json").read_bytes())
    except ValueError as e:
        return ValidationReport(ok=False, problems=[f"manifest.json invalid: {e}"])
    problems += [
        f"manifest.json does not list {n}"
        for n in (*_FILES, "DATASET_CARD.md")
        if n not in manifest.files
    ]
    for name, digest in sorted(manifest.files.items()):
        path = d / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            problems.append(f"{name}: sha256 does not match manifest.json")
    for path in sorted(d.iterdir()):  # every text file: data, manifest, card, anything extra
        if path.is_file() and path.suffix != ".parquet":
            try:
                problems += _content_problems(path.name, path.read_text())
            except UnicodeDecodeError:
                problems.append(f"{path.name}: not UTF-8 text")
    try:
        raws = [
            RawCall.model_validate_json(x) for x in (d / "traces.jsonl").read_bytes().splitlines()
        ]
        labs = [
            Label.model_validate_json(x) for x in (d / "labels.jsonl").read_bytes().splitlines()
        ]
    except ValueError as e:
        return ValidationReport(ok=False, problems=[*problems, f"jsonl rows invalid: {e}"])
    ids = [r.envelope_id for r in raws]
    if len(set(ids)) != len(ids):
        problems.append("traces.jsonl: duplicate envelope_id")
    if sorted(lab.envelope_id for lab in labs) != sorted(ids):
        problems.append("label integrity: need exactly one label per envelope")
    try:
        _check_normalizes(raws)
    except Exception as e:
        problems.append(f"an envelope failed to normalize: {e}")
    problems += _parquet_matches(d, raws, labs)
    by_id = {lab.envelope_id: lab for lab in labs}
    c = _counts(raws, [by_id[i] for i in ids if i in by_id])
    if c != manifest.counts:
        problems.append("manifest.json counts do not match the data")
    for what, have, need in (
        ("agents", c.agents, min_agents),
        ("sessions", c.sessions, min_sessions),
        ("actions", c.actions, min_actions),
        ("attacks", c.attacks, min_attacks),
    ):
        if have < need:
            problems.append(f"too few {what}: {have} < {need}")
    return ValidationReport(ok=not problems, problems=problems)


# --- CLI -----------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m agentfw.sandbox.package", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="package run directories into a dataset directory")
    b.add_argument("--runs", type=Path, nargs="+", required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--version", default="v1")
    b.add_argument("--seed", type=int, default=0)
    v = sub.add_parser("validate", help="validate a packaged dataset directory")
    v.add_argument("dir", type=Path)
    v.add_argument("--min-agents", type=int, default=20)
    v.add_argument("--min-sessions", type=int, default=2000)
    v.add_argument("--min-actions", type=int, default=40_000)
    v.add_argument("--min-attacks", type=int, default=600)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        m = package(a.runs, a.out, version=a.version, seed=a.seed)
        sys.stdout.write(m.counts.model_dump_json(indent=2) + "\n")
        return 0
    rep = validate_dataset(
        a.dir,
        min_agents=a.min_agents,
        min_sessions=a.min_sessions,
        min_actions=a.min_actions,
        min_attacks=a.min_attacks,
    )
    sys.stdout.write(rep.model_dump_json(indent=2) + "\n")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
