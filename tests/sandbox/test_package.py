"""Dataset packaging, validation and the v1 manifest (Task 2.9)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import orjson
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from agentfw.bench.dataset import load_dir, load_labels, load_traces
from agentfw.bench.registry import make_detector
from agentfw.bench.replay import SplitSpec, replay
from agentfw.core.envelope import Label, RawCall, ResponseMeta
from agentfw.core.normalize import SessionTracker, normalize
from agentfw.sandbox.package import (
    DatasetManifest,
    main,
    package,
    scan_secrets,
    validate_dataset,
)
from agentfw.sandbox.run import main as run_main
from agentfw.sandbox.tasks.schema import AGENTS_DIR, load_agents

AGENT_IDS = [a.agent_id for a in load_agents(AGENTS_DIR)]
WALL = 1_790_000_000_000
LOW = {"min_agents": 1, "min_sessions": 1, "min_actions": 1, "min_attacks": 0}


# -- hand-built run directories -------------------------------------------------------------------


def _calls(
    agent: str, sid: str, gaps_ms: Sequence[int], *, email: str = "bob@acme.example"
) -> list[RawCall]:
    ts = [WALL]
    for g in gaps_ms:
        ts.append(ts[-1] + g)
    return [
        RawCall(
            envelope_id=f"{sid}-e{i:03d}",
            session_id=sid,
            agent_id=agent,
            principal_id="svc",
            seq_no=i,
            ts_unix_ms=t,
            tool_id="zendesk.get_ticket",
            args={"ticket_id": f"T-{i}", "to": email},
            response_meta=ResponseMeta(item_count=1, bytes=10, ids=(f"ch_{sid}{i}",)),
        )
        for i, t in enumerate(ts)
    ]


def _write_run(
    out: Path,
    sessions: dict[str, tuple[str, list[int]]],
    *,
    status: dict[str, str] | None = None,
    labels: dict[str, Label] | None = None,
    meta: dict[str, object] | None = None,
    email: str = "bob@acme.example",
) -> dict[str, list[RawCall]]:
    status = status or {}
    labels = labels or {}
    (out / "raw").mkdir(parents=True)
    (out / "labels").mkdir()
    ledger = b""
    made: dict[str, list[RawCall]] = {}
    for sid, (agent, gaps) in sessions.items():
        calls = _calls(agent, sid, gaps, email=email)
        made[sid] = calls
        (out / "raw" / f"{sid}.jsonl").write_bytes(
            b"".join(c.model_dump_json().encode() + b"\n" for c in calls)
        )
        labs = [
            labels.get(c.envelope_id) or Label(envelope_id=c.envelope_id, label="benign")
            for c in calls
        ]
        (out / "labels" / f"{sid}.jsonl").write_bytes(
            b"".join(lab.model_dump_json().encode() + b"\n" for lab in labs)
        )
        ledger += (
            orjson.dumps(
                {"session_id": sid, "agent_id": agent, "status": status.get(sid, "completed")}
            )
            + b"\n"
        )
    (out / "ledger.jsonl").write_bytes(ledger)
    if meta is not None:
        (out / "run_meta.json").write_bytes(orjson.dumps(meta))
    return made


def _attack(eid: str, aid: str, family: str = "magnitude_spike", level: str = "action") -> Label:
    return Label(
        envelope_id=eid,
        label="attack",
        attack_family=family,
        injection_level=level,
        attack_id=aid,
    )


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [orjson.loads(x) for x in path.read_bytes().splitlines()]


# -- scan_secrets ---------------------------------------------------------------------------------


def test_scan_secrets_patterns() -> None:
    assert scan_secrets("nothing to see, bob@acme.example") == []
    assert scan_secrets("key sk_test_abc123") == ["sk_test_abc123"]
    assert scan_secrets("sk_live_XyZ9") == ["sk_live_XyZ9"]
    assert scan_secrets("t xoxb-123-abc-DEF") == ["xoxb-123-abc-DEF"]
    assert scan_secrets("Authorization: Bearer abc.def") == ["Bearer abc.def"]
    assert scan_secrets("sid 00D5g000000AbCdEFG!AQEAQabc.def_9") == [
        "00D5g000000AbCdEFG!AQEAQabc.def_9"
    ]


# -- package: selection, retiming, outputs --------------------------------------------------------


def test_only_completed_ledger_sessions_are_packaged(tmp_path: Path) -> None:
    run = tmp_path / "run"
    a = AGENT_IDS[0]
    sessions = {f"s{i}": (a, [1000, 1000]) for i in range(4)}
    _write_run(run, sessions, status={"s1": "failed", "s2": "budget_truncated"})
    # an orphan raw file that has no ledger line at all
    (run / "raw" / "s9.jsonl").write_bytes(
        (run / "raw" / "s0.jsonl").read_bytes().replace(b"s0", b"s9")
    )
    man = package([run], tmp_path / "out", seed=0)
    raws = load_traces(tmp_path / "out" / "traces.jsonl")
    assert {r.session_id for r in raws} == {"s0", "s3"}
    assert man.counts.sessions == 2
    assert man.counts.actions == len(raws) == 6
    assert man.source_runs[0]["excluded"] == {"failed": 1, "budget_truncated": 1}


def test_burst_gaps_exact_and_other_gaps_clamped(tmp_path: Path) -> None:
    run = tmp_path / "run"
    sid = "s0"
    burst = {f"{sid}-e{i:03d}" for i in (2, 3)}
    labels = {e: _attack(e, "atk-1", "velocity_burst") for e in burst}
    # gaps: e1 = 5 s (kept), e2/e3 = 100 ms (burst, exact), e4 = 10 min (clamped to 120 s)
    _write_run(run, {sid: (AGENT_IDS[0], [5000, 100, 100, 600_000])}, labels=labels)
    package([run], tmp_path / "out", seed=3)
    ts = [
        r.ts_unix_ms
        for r in sorted(load_traces(tmp_path / "out" / "traces.jsonl"), key=lambda r: r.seq_no)
    ]
    assert [b - a for a, b in zip(ts, ts[1:], strict=False)] == [5000, 100, 100, 120_000]
    assert ts[0] % 86_400_000 >= 8 * 3_600_000  # a working-hours start on the simulated timeline
    assert ts[0] < WALL


def test_outputs_sorted_deterministic_and_manifest_hashes(tmp_path: Path) -> None:
    run = tmp_path / "run"
    sessions = {f"s{i}": (AGENT_IDS[i % 3], [2000, 3000]) for i in range(6)}
    labels = {"s0-e001": _attack("s0-e001", "atk-A", "scope_blowup", "prompt")}
    _write_run(run, sessions, labels=labels, meta={"model": "gpt-test-1", "mode": "fake"})
    m1 = package([run], tmp_path / "o1", seed=1)
    m2 = package([run], tmp_path / "o2", seed=1)
    for name in ("traces.jsonl", "labels.jsonl", "traces.parquet", "labels.parquet"):
        assert (tmp_path / "o1" / name).exists()
    assert (tmp_path / "o1" / "DATASET_CARD.md").exists()
    assert m1.files == m2.files
    assert (tmp_path / "o1" / "traces.jsonl").read_bytes() == (
        tmp_path / "o2" / "traces.jsonl"
    ).read_bytes()
    rows = _jsonl(tmp_path / "o1" / "traces.jsonl")
    assert [(r["ts_unix_ms"], r["session_id"], r["seq_no"]) for r in rows] == sorted(
        (r["ts_unix_ms"], r["session_id"], r["seq_no"]) for r in rows
    )
    import hashlib

    for name, digest in m1.files.items():
        assert hashlib.sha256((tmp_path / "o1" / name).read_bytes()).hexdigest() == digest
    on_disk = DatasetManifest.model_validate_json((tmp_path / "o1" / "manifest.json").read_bytes())
    assert on_disk == m1
    assert m1.model == "gpt-test-1"
    assert m1.version == "v1"
    assert m1.counts.agents == 3 and m1.counts.sessions == 6 and m1.counts.actions == 18
    assert m1.counts.attacks == 1 and m1.counts.attack_envelopes == 1
    assert m1.counts.by_family == {"scope_blowup": 1}
    assert m1.counts.by_injection_level == {"prompt": 1}
    assert validate_dataset(tmp_path / "o1", **LOW).ok


def test_package_rejects_missing_label_and_session_id_collision(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_run(run, {"s0": (AGENT_IDS[0], [1000])})
    lab = run / "labels" / "s0.jsonl"
    lab.write_bytes(lab.read_bytes().splitlines(keepends=True)[0])
    with pytest.raises(ValueError, match="label"):
        package([run], tmp_path / "out")
    run2, run3 = tmp_path / "r2", tmp_path / "r3"
    _write_run(run2, {"s0": (AGENT_IDS[0], [1000])})
    _write_run(run3, {"s0": (AGENT_IDS[0], [1000])})
    with pytest.raises(ValueError, match="duplicate session"):
        package([run2, run3], tmp_path / "out2")


def test_package_refuses_unreserved_email_in_source(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_run(run, {"s0": (AGENT_IDS[0], [1000])}, email="eve@gmail.com")
    with pytest.raises(ValueError, match="gmail.com"):
        package([run], tmp_path / "out")
    assert not (tmp_path / "out" / "manifest.json").exists()


# -- validate_dataset -----------------------------------------------------------------------------


@pytest.fixture
def mini(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    sessions = {f"s{i}": (AGENT_IDS[i % 2], [2000, 3000, 1000]) for i in range(4)}
    labels = {"s1-e001": _attack("s1-e001", "atk-1"), "s2-e002": _attack("s2-e002", "atk-2")}
    _write_run(run, sessions, labels=labels)
    out = tmp_path / "ds"
    package([run], out, seed=0)
    return out


def test_valid_mini_dataset_and_minimums(mini: Path) -> None:
    rep = validate_dataset(mini, min_agents=2, min_sessions=4, min_actions=16, min_attacks=2)
    assert rep.ok, rep.problems
    short = validate_dataset(mini)  # the real v1 minimums
    assert not short.ok
    text = " ".join(short.problems)
    for word in ("agents", "sessions", "actions", "attacks"):
        assert word in text


def test_planted_secret_fails_and_names_file(mini: Path) -> None:
    p = mini / "traces.jsonl"
    p.write_bytes(p.read_bytes().replace(b"T-1", b"sk_test_abc", 1))
    rep = validate_dataset(mini, **LOW)
    assert not rep.ok
    assert any("traces.jsonl" in x and "secret" in x for x in rep.problems)


def test_non_reserved_email_domain_fails(mini: Path) -> None:
    p = mini / "traces.jsonl"
    p.write_bytes(p.read_bytes().replace(b"bob@acme.example", b"bob@gmail.com", 1))
    rep = validate_dataset(mini, **LOW)
    assert not rep.ok
    assert any("gmail.com" in x for x in rep.problems)


def test_jsonl_parquet_mismatch_fails(mini: Path) -> None:
    t = pq.read_table(mini / "labels.parquet")
    cols = t.column_names
    attack = [("attack" if i == 0 else v) for i, v in enumerate(t.column("label").to_pylist())]
    t = t.set_column(cols.index("label"), "label", pa.array(attack))
    pq.write_table(t, mini / "labels.parquet")
    rep = validate_dataset(mini, **LOW)
    assert not rep.ok
    assert any("parquet" in x.lower() for x in rep.problems)


def test_label_integrity_and_hash_problems(mini: Path) -> None:
    p = mini / "labels.jsonl"
    lines = p.read_bytes().splitlines(keepends=True)
    p.write_bytes(b"".join(lines[1:]))  # drop one label
    rep = validate_dataset(mini, **LOW)
    assert not rep.ok
    assert any("label" in x for x in rep.problems)
    assert any("sha256" in x and "labels.jsonl" in x for x in rep.problems)


def test_manifest_counts_match_hand_count(mini: Path) -> None:
    man = DatasetManifest.model_validate_json((mini / "manifest.json").read_bytes())
    traces = _jsonl(mini / "traces.jsonl")
    labels = _jsonl(mini / "labels.jsonl")
    assert man.counts.actions == len(traces) == 16
    assert man.counts.agents == len({t["agent_id"] for t in traces}) == 2
    assert man.counts.sessions == len({t["session_id"] for t in traces}) == 4
    attacks = [x for x in labels if x["label"] == "attack"]
    assert man.counts.attacks == len({x["attack_id"] for x in attacks}) == 2
    assert man.counts.attack_envelopes == len(attacks)
    assert man.counts.by_family == dict(Counter(str(x["attack_family"]) for x in attacks))
    assert man.counts.by_injection_level == dict(
        Counter(str(x["injection_level"]) for x in attacks)
    )


# -- Parquet reading in bench.dataset -------------------------------------------------------------


def test_parquet_equals_jsonl_through_loaders(mini: Path) -> None:
    assert load_traces(mini / "traces.parquet") == load_traces(mini / "traces.jsonl")
    assert load_labels(mini / "labels.parquet") == load_labels(mini / "labels.jsonl")
    # a directory with only Parquet files still loads
    only = mini.parent / "only_parquet"
    only.mkdir()
    for n in ("traces.parquet", "labels.parquet", "manifest.json"):
        (only / n).write_bytes((mini / n).read_bytes())
    a = load_dir(only)
    b = load_dir(mini)
    assert a[0] == b[0] and a[1] == b[1]


# -- end to end on a fake-mode run ----------------------------------------------------------------


def test_fake_run_packages_validates_and_replays(tmp_path: Path) -> None:
    run = tmp_path / "run"
    args = ["--sessions", "40", "--seed", "0", "--out", str(run), "--fake"]
    assert run_main([*args, "--action-rate", "0.3", "--prompt-rate", "0.2"]) == 0
    out = tmp_path / "v1"
    assert main(["build", "--runs", str(run), "--out", str(out), "--seed", "0"]) == 0
    assert (
        main(
            [
                "validate",
                str(out),
                "--min-agents",
                "5",
                "--min-sessions",
                "40",
                "--min-actions",
                "100",
                "--min-attacks",
                "1",
            ]
        )
        == 0
    )
    man = DatasetManifest.model_validate_json((out / "manifest.json").read_bytes())
    assert man.counts.sessions == 40 and man.counts.attacks >= 1
    assert man.model == "scripted"
    assert not scan_secrets((out / "manifest.json").read_text())
    raws, labels, _ = load_dir(out)
    assert set(labels) == {r.envelope_id for r in raws}
    tracker = SessionTracker()
    for r in raws:
        tracker_env = normalize(r, tracker)
        tracker.observe(tracker_env)
    rows = replay(raws, labels, make_detector("engine"), SplitSpec())
    assert len(rows) == len(raws)
    assert any(x.label == "attack" for x in rows)
    assert not (out / "transcripts").exists()  # transcripts are not part of the public dataset


# -- fix round 1: scanning scope, patterns, provenance --------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "sk-proj-abcdefghijklmnopqrstuvwx",
        "key sk-abcdefghijklmnopqrstuv1234",
        "rk_live_abc123",
        "pk_test_abc123",
        "whsec_abc123XYZ",
        "xapp-1-A0B1-123-abc",
        "xoxb-1-2-abc",
        "authorization: bearer abc123",
        "BEARER abc123",
        "Bearer\\tabc123",  # JSON-escaped tab, as in raw serialized text
        "Bearer\\nabc123",
        "Bearer\tabc123",
    ],
)
def test_scan_secrets_extended_patterns(text: str) -> None:
    assert scan_secrets(text), text


def test_scan_secrets_no_false_positive_on_short_sk_dash() -> None:
    assert scan_secrets("sk-short and task-force and whsec") == []


def test_secret_in_model_id_blocks_packaging(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_run(run, {"s0": (AGENT_IDS[0], [1000])}, meta={"model": "sk_test_leak", "mode": "live"})
    with pytest.raises(ValueError, match="secret"):
        package([run], tmp_path / "out")
    assert not (tmp_path / "out" / "traces.jsonl").exists()


def test_secret_in_run_dir_name_blocks_packaging(tmp_path: Path) -> None:
    run = tmp_path / "xoxb-123-leak"
    _write_run(run, {"s0": (AGENT_IDS[0], [1000])})
    with pytest.raises(ValueError, match="secret"):
        package([run], tmp_path / "out")


def test_secret_split_by_json_escape_is_found(mini: Path) -> None:
    p = mini / "traces.jsonl"
    p.write_bytes(p.read_bytes().replace(b"T-1", b"Bearer\\tabc", 1))
    rep = validate_dataset(mini, **LOW)
    assert any("traces.jsonl" in x and "secret" in x for x in rep.problems)


def test_secret_planted_in_card_and_manifest_is_flagged(mini: Path) -> None:
    card = mini / "DATASET_CARD.md"
    card.write_text(card.read_text() + "\nkey sk_live_abc123\n")
    rep = validate_dataset(mini, **LOW)
    assert any("DATASET_CARD.md" in x and "secret" in x for x in rep.problems)
    man = mini / "manifest.json"
    raw = orjson.loads(man.read_bytes())
    raw["model"] = "xoxb-1-leak"
    man.write_bytes(orjson.dumps(raw))
    rep = validate_dataset(mini, **LOW)
    assert any("manifest.json" in x and "secret" in x for x in rep.problems)


def test_manifest_must_list_all_files(mini: Path) -> None:
    man = mini / "manifest.json"
    raw = orjson.loads(man.read_bytes())
    del raw["files"]["traces.parquet"]
    raw["files"].pop("DATASET_CARD.md")
    man.write_bytes(orjson.dumps(raw))
    rep = validate_dataset(mini, **LOW)
    assert not rep.ok
    assert any("traces.parquet" in x and "manifest" in x for x in rep.problems)
    assert any("DATASET_CARD.md" in x and "manifest" in x for x in rep.problems)
    raw["files"] = {}
    man.write_bytes(orjson.dumps(raw))
    assert not validate_dataset(mini, **LOW).ok


def test_dotless_email_domain_rejected(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_run(run, {"s0": (AGENT_IDS[0], [1000])}, email="eve@localhost")
    with pytest.raises(ValueError, match="localhost"):
        package([run], tmp_path / "out")


def test_missing_raw_file_names_session_and_run(tmp_path: Path) -> None:
    run = tmp_path / "myrun"
    _write_run(run, {"s0": (AGENT_IDS[0], [1000])})
    (run / "raw" / "s0.jsonl").unlink()
    with pytest.raises(ValueError, match=r"s0.*myrun|myrun.*s0"):
        package([run], tmp_path / "out")


def test_inconsistent_agent_in_session_rejected(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_run(run, {"s0": (AGENT_IDS[0], [1000, 1000])})
    raw = run / "raw" / "s0.jsonl"
    lines = raw.read_bytes().splitlines(keepends=True)
    lines[1] = lines[1].replace(AGENT_IDS[0].encode(), AGENT_IDS[1].encode())
    raw.write_bytes(b"".join(lines))
    with pytest.raises(ValueError, match="agent"):
        package([run], tmp_path / "out")


def test_card_has_limitations_and_fake_banner(tmp_path: Path) -> None:
    fake = tmp_path / "fake"
    live = tmp_path / "live"
    _write_run(fake, {"s0": (AGENT_IDS[0], [1000])}, meta={"model": "scripted", "mode": "fake"})
    _write_run(live, {"s1": (AGENT_IDS[0], [1000])}, meta={"model": "m-1", "mode": "live"})
    m = package([fake], tmp_path / "o_fake")
    assert [r["mode"] for r in m.source_runs] == ["fake"]
    card = (tmp_path / "o_fake" / "DATASET_CARD.md").read_text()
    assert "## Limitations" in card
    assert "FAKE" in card.split("## Limitations")[0]
    package([live], tmp_path / "o_live")
    live_card = (tmp_path / "o_live" / "DATASET_CARD.md").read_text()
    assert "## Limitations" in live_card
    assert "FAKE" not in live_card
    both = package([fake, live], tmp_path / "o_both")
    assert [r["mode"] for r in both.source_runs] == ["fake", "live"]
    assert "FAKE" in (tmp_path / "o_both" / "DATASET_CARD.md").read_text()


# -- findings never echo the secret (FR I5) -------------------------------------------------------

_TOKEN = "sk_live_" + "Zq9" * 12  # 44 chars


def test_secret_finding_reports_pattern_length_location_not_the_value(mini: Path) -> None:
    p = mini / "traces.jsonl"
    p.write_bytes(p.read_bytes().replace(b"T-1", _TOKEN.encode(), 1))
    rep = validate_dataset(mini, **LOW)
    hits = [x for x in rep.problems if "secret" in x]
    assert hits and all("traces.jsonl" in x for x in hits)
    assert any("stripe_key" in x and f"{len(_TOKEN)} chars" in x and "line 2" in x for x in hits)
    for x in rep.problems:
        assert _TOKEN[:12] not in x and "Zq9Zq9" not in x


def test_refusal_exception_never_contains_the_secret(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_run(run, {"s0": (AGENT_IDS[0], [1000])}, meta={"model": _TOKEN, "mode": "live"})
    with pytest.raises(ValueError, match="secret") as info:
        package([run], tmp_path / "out")
    text = str(info.value)
    assert "stripe_key" in text and f"{len(_TOKEN)} chars" in text
    assert _TOKEN[:12] not in text and "Zq9Zq9" not in text
