from __future__ import annotations

import hashlib
from pathlib import Path

import orjson
import pytest

from agentfw.core.envelope import Label, RawCall
from agentfw.synth.generate import GenerateConfig, generate, main
from agentfw.synth.load import CATALOG_DIR, PERSONAS_DIR, TASKS_DIR, load_personas

START = 1_767_225_600_000
DAY_MS = 86_400_000


def cfg(**kw: object) -> GenerateConfig:
    base: dict[str, object] = {
        "seed": 0,
        "personas_dir": PERSONAS_DIR,
        "tasks_dir": TASKS_DIR,
        "catalog_dir": CATALOG_DIR,
        "days": 3,
        "max_sessions": None,
        "families": [],
    }
    return GenerateConfig.model_validate({**base, **kw})


def _read(path: Path) -> list[dict[str, object]]:
    return [orjson.loads(line) for line in path.read_bytes().splitlines()]


def test_same_seed_byte_identical(tmp_path: Path) -> None:
    m1 = generate(cfg(seed=0, days=3), tmp_path / "a")
    m2 = generate(cfg(seed=0, days=3), tmp_path / "b")
    assert m1.files == m2.files
    for name in ("traces.jsonl", "labels.jsonl", "manifest.json"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
    assert generate(cfg(seed=1, days=3), tmp_path / "c").files != m1.files


def test_manifest_hashes_and_counts(tmp_path: Path) -> None:
    m = generate(cfg(), tmp_path)
    for name, digest in m.files.items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
    assert set(m.files) == {"traces.jsonl", "labels.jsonl"}
    traces = _read(tmp_path / "traces.jsonl")
    assert m.counts["envelopes"] == len(traces) > 0
    assert m.counts["sessions"] == len({t["session_id"] for t in traces})
    assert m.counts["attacks"] == 0
    assert orjson.loads((tmp_path / "manifest.json").read_bytes())["files"] == m.files


def test_traces_sorted_and_well_formed(tmp_path: Path) -> None:
    generate(cfg(days=2), tmp_path)
    raws = [RawCall.model_validate(d) for d in _read(tmp_path / "traces.jsonl")]
    keys = [(r.ts_unix_ms, r.session_id, r.seq_no) for r in raws]
    assert keys == sorted(keys)
    assert len({r.envelope_id for r in raws}) == len(raws)
    hours = {p.agent_id: p.working_hours for p in load_personas(PERSONAS_DIR)}
    by_session: dict[str, list[RawCall]] = {}
    for r in raws:
        by_session.setdefault(r.session_id, []).append(r)
    for calls in by_session.values():
        assert [c.seq_no for c in calls] == list(range(len(calls)))
        lo, hi = hours[calls[0].agent_id]
        assert lo <= ((calls[0].ts_unix_ms - START) % DAY_MS) / 3_600_000 < hi


def test_one_benign_label_per_envelope(tmp_path: Path) -> None:
    generate(cfg(), tmp_path)
    raws = _read(tmp_path / "traces.jsonl")
    labels = [Label.model_validate(d) for d in _read(tmp_path / "labels.jsonl")]
    assert [lab.envelope_id for lab in labels] == [r["envelope_id"] for r in raws]
    assert all(lab.label == "benign" and lab.attack_id is None for lab in labels)


def test_jsonl_uses_sorted_keys(tmp_path: Path) -> None:
    generate(cfg(days=1), tmp_path)
    line = (tmp_path / "traces.jsonl").read_bytes().splitlines()[0]
    d = orjson.loads(line)
    assert line == orjson.dumps(d, option=orjson.OPT_SORT_KEYS)


def test_max_sessions_total_cap_split_in_sorted_order(tmp_path: Path) -> None:
    m = generate(cfg(max_sessions=10, days=30), tmp_path)
    assert m.counts["sessions"] == 10
    traces = _read(tmp_path / "traces.jsonl")
    per: dict[str, set[object]] = {}
    for t in traces:
        per.setdefault(str(t["agent_id"]), set()).add(t["session_id"])
    # 3 personas, 10 total: remainder goes to the first in sorted agent_id order
    assert {k: len(v) for k, v in per.items()} == {
        "outreach-a": 4,
        "sales-ops-a": 3,
        "support-refunds-a": 3,
    }


def test_families_not_yet_supported(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError):
        generate(cfg(families=["magnitude_spike"]), tmp_path)


def test_cli_main(tmp_path: Path) -> None:
    out = tmp_path / "o"
    rc = main(["--seed", "0", "--days", "1", "--out", str(out), "--max-sessions", "6"])
    assert rc == 0
    m = orjson.loads((out / "manifest.json").read_bytes())
    assert m["counts"]["sessions"] == 6
    assert m["config"]["seed"] == 0


def test_cap_smaller_than_persona_count(tmp_path: Path) -> None:
    m = generate(cfg(max_sessions=2, days=30), tmp_path)
    assert m.counts["sessions"] == 2
    per = {str(t["agent_id"]) for t in _read(tmp_path / "traces.jsonl")}
    assert per == {"outreach-a", "sales-ops-a"}


def test_zero_cap_emits_nothing(tmp_path: Path) -> None:
    m = generate(cfg(max_sessions=0), tmp_path)
    assert m.counts["sessions"] == 0
    assert (tmp_path / "traces.jsonl").read_bytes() == b""
    assert (tmp_path / "labels.jsonl").read_bytes() == b""


def test_negative_values_rejected() -> None:
    with pytest.raises(ValueError):
        cfg(days=-1)
    with pytest.raises(ValueError):
        cfg(max_sessions=-1)
