from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import orjson
import pytest

from agentfw.core.envelope import Label, RawCall
from agentfw.synth.attacks import CONTROL_P99_MULT
from agentfw.synth.generate import GenerateConfig, Manifest, generate, main
from agentfw.synth.load import (
    CATALOG_DIR,
    PERSONAS_DIR,
    TASKS_DIR,
    TIER0_PERSONA_IDS,
    load_personas,
)

from .conftest import World

START = 1_767_225_600_000
DAY_MS = 86_400_000


def cfg(**kw: object) -> GenerateConfig:
    base: dict[str, object] = {
        "seed": 0,
        "personas_dir": PERSONAS_DIR,
        "persona_ids": list(TIER0_PERSONA_IDS),
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


# --- magnitude_mode "far" (Amendment A1.4) --------------------------------------------------------

ALL_FAMILIES = [
    "magnitude_spike",
    "prereq_skip",
    "novel_target",
    "scope_blowup",
    "velocity_burst",
    "slow_drift",
    "exfil_volume",
    "benign_unusual",
]
SPIKE = "magnitude_spike"
# sha256 of the mode-"own" dataset of ``small_cfg()``, taken at 827cd34 (before magnitude_mode).
SMALL_OWN_FILES = {
    "traces.jsonl": "dbdef7b8ec5a6b46129eca21e22e92605d5d8885b3489cf133e721d6e0e84ec1",
    "labels.jsonl": "db4e21b005e6ad3bf712d0900ba289e93d2bc49dbf9f9182f56a3ea7ef6dacdc",
}


def small_cfg(**kw: object) -> GenerateConfig:
    base: dict[str, object] = {
        "days": 40,
        "attack_rate": 0.15,
        "control_rate": 0.05,
        "families": ALL_FAMILIES,
    }
    return cfg(**{**base, **kw})


@dataclass
class Data:
    """A generated dataset: raw JSONL lines and their parsed rows, in file order."""

    manifest: Manifest
    trace_lines: list[bytes]
    label_lines: list[bytes]
    traces: list[dict[str, object]]
    labels: list[dict[str, object]]

    def session_of(self) -> dict[object, object]:
        return {t["envelope_id"]: t["session_id"] for t in self.traces}

    def spikes(self) -> list[dict[str, object]]:
        return [lab for lab in self.labels if lab["attack_family"] == SPIKE]

    def attack_ids(self, family: str | None = None) -> set[object]:
        return {
            lab["attack_id"]
            for lab in self.labels
            if lab["attack_id"] is not None and family in (None, lab["attack_family"])
        }


def gen(out: Path, config: GenerateConfig) -> Data:
    m = generate(config, out)
    tl, ll = (
        (out / "traces.jsonl").read_bytes().splitlines(),
        (out / "labels.jsonl").read_bytes().splitlines(),
    )
    return Data(m, tl, ll, [orjson.loads(x) for x in tl], [orjson.loads(x) for x in ll])


@dataclass
class Trio:
    benign: Data  # no injection at all: the calls as the personas made them
    own: Data
    far: Data


@pytest.fixture(scope="module")
def trio(tmp_path_factory: pytest.TempPathFactory) -> Trio:
    root = tmp_path_factory.mktemp("magnitude-mode")
    return Trio(
        gen(root / "benign", small_cfg(families=[])),
        gen(root / "own", small_cfg()),
        gen(root / "far", small_cfg(magnitude_mode="far")),
    )


def _num(v: object) -> float:
    assert isinstance(v, int | float) and not isinstance(v, bool)
    return float(v)


def _spiked_values(d: Data, benign: Data) -> list[tuple[str, str, str, float, float]]:
    """``(agent, tool, arg, new value, own value)`` per magnitude spike of ``d``."""
    orig = {t["envelope_id"]: t for t in benign.traces}
    now = {t["envelope_id"]: t for t in d.traces}
    out = []
    for lab in d.spikes():
        new, old = now[lab["envelope_id"]], orig[lab["envelope_id"]]
        na, oa = new["args"], old["args"]
        assert isinstance(na, dict) and isinstance(oa, dict)
        (name,) = [k for k in na if na[k] != oa[k]]
        assert {**new, "args": None} == {**old, "args": None}  # only that one arg changed
        assert lab["strength"] == _num(na[name]) / _num(oa[name])  # the realized multiplier
        out.append(
            (str(new["agent_id"]), str(new["tool_id"]), name, _num(na[name]), _num(oa[name]))
        )
    return out


def test_far_mode_leaves_other_families_and_controls_byte_identical(
    trio: Trio, world: World
) -> None:
    own, far = trio.own, trio.far
    sess = own.session_of()
    assert [t["envelope_id"] for t in far.traces] == [t["envelope_id"] for t in own.traces]
    # the sessions mode "own" spiked are the ones mode "far" spiked or dropped
    spiked_sessions = {sess[lab["envelope_id"]] for lab in own.spikes()}
    spiked_or_dropped = {env for env, sid in sess.items() if sid in spiked_sessions}
    assert {lab["envelope_id"] for lab in far.spikes()} <= spiked_or_dropped

    def keep(lines: list[bytes], rows: list[dict[str, object]]) -> list[bytes]:
        return [
            line
            for line, r in zip(lines, rows, strict=True)
            if r["envelope_id"] not in spiked_or_dropped
        ]

    kept_labels = keep(own.label_lines, own.labels)
    assert kept_labels == keep(far.label_lines, far.labels)  # raw JSON lines
    assert keep(own.trace_lines, own.traces) == keep(far.trace_lines, far.traces)
    assert own.trace_lines != far.trace_lines and own.label_lines != far.label_lines
    # ... and that kept part holds every other family and the controls
    kept = [orjson.loads(x) for x in kept_labels]
    assert {r["attack_family"] for r in kept if r["label"] == "attack"} == set(ALL_FAMILIES) - {
        SPIKE,
        "benign_unusual",
    }
    assert sum(r["control_type"] == "benign_unusual" for r in kept) >= 5
    assert far.attack_ids() - far.attack_ids(SPIKE) == own.attack_ids() - own.attack_ids(SPIKE)
    assert far.attack_ids(SPIKE) <= own.attack_ids(SPIKE)

    far_spikes = far.spikes()
    assert len(far_spikes) >= 20
    assert all(3.0 <= _num(lab["strength"]) <= 100.0 + 1e-9 for lab in far_spikes)
    # every far spike is above the largest benign value of its field
    benign_max: dict[tuple[str, str, str], float] = {}
    for t in trio.benign.traces:
        args = t["args"]
        assert isinstance(args, dict)
        for name, v in args.items():
            if isinstance(v, int | float) and not isinstance(v, bool):
                key = (str(t["agent_id"]), str(t["tool_id"]), name)
                benign_max[key] = max(benign_max.get(key, 0.0), float(v))
    far_values = _spiked_values(far, trio.benign)
    assert min(new / benign_max[(a, t, n)] for a, t, n, new, _ in far_values) > 1.0
    assert len({(t, n) for _, t, n, _, _ in far_values}) >= 2  # money and count fields

    # no attack-labelled refund at or below the benign-labelled control of the same field
    control = {
        _num(t["args"]["amount"])  # type: ignore[index]
        for t, lab in zip(far.traces, far.labels, strict=True)
        if lab["control_type"] == "benign_unusual" and t["tool_id"] == "stripe.create_refund"
    }
    # 5 x the persona's declared p99
    assert control == {round(CONTROL_P99_MULT * world.declared_p99("support-refunds-a"), 2)}
    (level,) = control

    def at_or_below_control(d: Data) -> int:
        return sum(
            new <= level
            for a, t, n, new, _ in _spiked_values(d, trio.benign)
            if (a, t, n) == ("support-refunds-a", "stripe.create_refund", "amount")
        )

    assert at_or_below_control(far) == 0 < at_or_below_control(own)
    assert set(far.manifest.counts) == {
        "sessions",
        "envelopes",
        "attacks",
        "magnitude_first_pass",
        "magnitude_dropped",
        "magnitude_topup",
    }
    assert far.manifest.counts["magnitude_first_pass"] == len(far.attack_ids(SPIKE))
    assert far.manifest.counts["magnitude_topup"] == 0
    dropped_ids = own.attack_ids(SPIKE) - far.attack_ids(SPIKE)
    assert far.manifest.counts["magnitude_dropped"] == len(dropped_ids) == 1
    assert far.manifest.counts["attacks"] == len(far.attack_ids())
    assert far.manifest.config.magnitude_mode == "far"


def test_far_infeasible_session_stays_benign_and_is_counted(tmp_path: Path) -> None:
    # One refund persona, a control on EVERY session the attack draw leaves alone, and a far bound
    # high enough (25 x p99) that many refunds cannot reach it within a multiplier of 100.
    def run(name: str, **kw: object) -> Data:
        base: dict[str, object] = {
            "persona_ids": ["support-refunds-a"],
            "days": 5,
            "attack_rate": 0.15,
            "control_rate": 1.0,
            "families": [SPIKE, "benign_unusual"],
            "magnitude_far_mult": 25.0,
        }
        return gen(tmp_path / name, cfg(**{**base, **kw}))

    own, far = run("own"), run("far", magnitude_mode="far")
    benign, controls = run("benign", families=[]), run("controls", attack_rate=0.0)
    sess = own.session_of()
    own_spiked = {sess[lab["envelope_id"]]: lab["attack_id"] for lab in own.spikes()}
    far_spiked = {sess[lab["envelope_id"]] for lab in far.spikes()}
    dropped = set(own_spiked) - far_spiked
    assert far.manifest.counts["magnitude_dropped"] == len(dropped) >= 1
    assert far.manifest.counts["magnitude_first_pass"] == len(far_spiked) >= 1
    assert far.manifest.counts["attacks"] == len(far_spiked)
    assert "magnitude_dropped" not in own.manifest.counts

    def of(d: Data, lines: list[bytes], sids: set[object]) -> list[bytes]:
        return [x for x, t in zip(lines, d.traces, strict=True) if t["session_id"] in sids]

    # a dropped session is exactly the benign one: no spike, no attack label and no control label
    assert of(far, far.trace_lines, dropped) == of(benign, benign.trace_lines, dropped)
    assert of(far, far.label_lines, dropped) == of(benign, benign.label_lines, dropped)
    # ... although, left alone by the attack draw, some of those sessions do take the control
    taken = [orjson.loads(x) for x in of(controls, controls.label_lines, dropped)]
    assert any(r["control_type"] == "benign_unusual" for r in taken)
    # the dropped attacks' ids are consumed: the persona's ids show exactly those gaps
    assert far.attack_ids() == own.attack_ids() - {own_spiked[s] for s in dropped}
    # everything outside the spiked-or-dropped sessions is byte-identical, controls included
    rest = set(sess.values()) - set(own_spiked)
    assert of(far, far.trace_lines, rest) == of(own, own.trace_lines, rest)
    assert of(far, far.label_lines, rest) == of(own, own.label_lines, rest)
    assert any(orjson.loads(x)["control_type"] for x in of(far, far.label_lines, rest))


def test_far_mode_is_deterministic(trio: Trio, tmp_path: Path) -> None:
    again = gen(tmp_path, small_cfg(magnitude_mode="far"))
    assert again.manifest.files == trio.far.manifest.files  # sha256 of both files
    assert again.manifest.counts == trio.far.manifest.counts
    for name, digest in again.manifest.files.items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
    assert again.manifest.files != trio.own.manifest.files


def test_default_generator_is_unchanged(trio: Trio, tmp_path: Path) -> None:
    c = small_cfg()
    assert c.magnitude_mode == "own" and c.magnitude_far_mult == CONTROL_P99_MULT == 5.0
    m = trio.own.manifest
    assert set(m.counts) == {"sessions", "envelopes", "attacks"}
    assert m.files == SMALL_OWN_FILES
    # mode "own" never reads the multiple
    tiny = {"days": 5, "attack_rate": 0.5, "families": [SPIKE, "benign_unusual"]}
    a = generate(cfg(**tiny), tmp_path / "a")
    b = generate(cfg(**tiny, magnitude_mode="own", magnitude_far_mult=1.0), tmp_path / "b")
    assert a.files == b.files and a.counts == b.counts and a.counts["attacks"] > 0


def test_magnitude_config_is_validated() -> None:
    assert cfg(magnitude_mode="far", magnitude_far_mult=1.0).magnitude_far_mult == 1.0
    for bad in (
        {"magnitude_mode": "near"},
        {"magnitude_far_mult": 0.0},
        {"magnitude_far_mult": -5},
        {"magnitude_far_mult": math.inf},  # the bound would be inf and the manifest value null
        {"magnitude_far_mult": math.nan},
    ):
        with pytest.raises(ValueError):
            cfg(**bad)


def test_far_manifest_census_is_present_without_any_family(tmp_path: Path) -> None:
    m = generate(cfg(days=1, magnitude_mode="far"), tmp_path)
    census = {"magnitude_first_pass": 0, "magnitude_dropped": 0, "magnitude_topup": 0}
    assert m.counts == {**m.counts, **census} and len(m.counts) == 6


def test_cli_magnitude_flags(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base = ["--seed", "0", "--days", "1", "--max-sessions", "6", "--families", SPIKE]
    assert main([*base, "--out", str(tmp_path / "d")]) == 0
    m = orjson.loads((tmp_path / "d" / "manifest.json").read_bytes())
    assert m["config"]["magnitude_mode"] == "own" and m["config"]["magnitude_far_mult"] == 5.0
    assert set(m["counts"]) == {"sessions", "envelopes", "attacks"}
    far = [*base, "--magnitude-mode", "far", "--magnitude-far-mult", "1.5"]
    assert main([*far, "--out", str(tmp_path / "f")]) == 0
    m = orjson.loads((tmp_path / "f" / "manifest.json").read_bytes())
    assert m["config"]["magnitude_mode"] == "far" and m["config"]["magnitude_far_mult"] == 1.5
    assert m["counts"]["magnitude_topup"] == 0 and "magnitude_dropped" in m["counts"]
    with pytest.raises(SystemExit):
        main([*base, "--out", str(tmp_path / "x"), "--magnitude-mode", "near"])
    assert "--magnitude-mode" in capsys.readouterr().err


def test_cli_rejects_a_bad_config_with_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = ["--seed", "0", "--days", "1", "--max-sessions", "6", "--families", SPIKE]
    out = tmp_path / "x"
    for bad in ("0", "-1", "inf", "nan"):
        with pytest.raises(SystemExit) as exc:
            main([*base, "--out", str(out), "--magnitude-mode", "far", "--magnitude-far-mult", bad])
        assert exc.value.code == 2  # an argparse usage error, not a pydantic traceback
        err = capsys.readouterr().err
        assert "magnitude_far_mult" in err and "Traceback" not in err
        assert not out.exists()  # nothing is written
    with pytest.raises(SystemExit) as exc:
        main(["--days", "-1", "--out", str(out)])
    assert exc.value.code == 2 and "days" in capsys.readouterr().err


def test_cli_magnitude_defaults_are_the_config_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fields = GenerateConfig.model_fields
    assert fields["magnitude_mode"].default == "own"
    assert fields["magnitude_far_mult"].default == CONTROL_P99_MULT
    # the CLI holds no copy of the defaults: it reads the config fields when it builds its parser
    monkeypatch.setattr(fields["magnitude_mode"], "default", "far")
    monkeypatch.setattr(fields["magnitude_far_mult"], "default", 7.5)
    assert main(["--days", "1", "--max-sessions", "3", "--out", str(tmp_path)]) == 0
    m = orjson.loads((tmp_path / "manifest.json").read_bytes())
    assert m["config"]["magnitude_mode"] == "far" and m["config"]["magnitude_far_mult"] == 7.5
