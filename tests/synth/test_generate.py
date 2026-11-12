from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import orjson
import pytest

from agentfw.core.envelope import Label, RawCall
from agentfw.synth.attacks import (
    ATTACKS,
    CONTROL_P99_MULT,
    M_MAX,
    M_MIN,
    AttackContext,
    AttackFn,
    AttackResult,
    far_bounds,
)
from agentfw.synth.generate import (
    GenerateConfig,
    Manifest,
    attack_rng,
    generate,
    main,
    sw1_config,
    topup_rng,
)
from agentfw.synth.load import (
    CATALOG_DIR,
    PERSONAS_DIR,
    TASKS_DIR,
    TIER0_PERSONA_IDS,
    load_personas,
)
from agentfw.synth.sampler import stable_hash

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
        "magnitude_strength_ge_10",
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
    census = {
        "magnitude_first_pass": 0,
        "magnitude_dropped": 0,
        "magnitude_topup": 0,
        "magnitude_strength_ge_10": 0,
    }
    assert m.counts == {**m.counts, **census} and len(m.counts) == 7


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


# --- magnitude top-up pass, sw1_config, manifest census (Amendment A1.4) --------------------------

TOPUP_ID = re.compile(r"atk-(.+)-m(\d{5})")
WARMUP = 30  # GenerateConfig.warmup_sessions
MONEY_COUNT_NAMES = {"amount", "count", "qty", "total"}
CENSUS_KEYS = {
    "sessions",
    "envelopes",
    "attacks",
    "magnitude_first_pass",
    "magnitude_dropped",
    "magnitude_topup",
    "magnitude_strength_ge_10",
}
# sha256 of ``small_cfg(magnitude_mode="far")``, taken at 335354f (Task 1.19, before the top-up).
SMALL_FAR_FILES = {
    "traces.jsonl": "1c4ed3825e25b2e31082837602648dab7cf1b1911d2f69c5d59f64048eb6bddf",
    "labels.jsonl": "55ad0117e741f2170529231de47cd1a2e12df5c66d3e10f1e518cd8ce65c7313",
}
Cand = tuple[str, str, int | float]  # (tool, arg, the call's own value)


def topup_cfg(rate: float, mode: str = "far") -> GenerateConfig:
    return small_cfg(magnitude_mode=mode, magnitude_topup_rate=rate)


def _is_topup(lab: dict[str, object]) -> bool:
    aid = lab["attack_id"]
    return isinstance(aid, str) and TOPUP_ID.fullmatch(aid) is not None


def _topups(d: Data) -> dict[str, str]:
    """``session id -> attack id`` of the top-up spikes of ``d`` (ids ``atk-{agent}-m{k:05d}``)."""
    sess = d.session_of()
    out: dict[str, str] = {}
    for lab in d.labels:
        if _is_topup(lab):
            sid = str(sess[lab["envelope_id"]])
            assert sid not in out  # one spiked call per top-up session
            out[sid] = str(lab["attack_id"])
    return out


def _without(lines: list[bytes], rows: list[dict[str, object]], envs: set[object]) -> list[bytes]:
    return [x for x, r in zip(lines, rows, strict=True) if r["envelope_id"] not in envs]


def gen_spying(out: Path, config: GenerateConfig) -> tuple[Data, set[str]]:
    """``gen`` plus the ids of the sessions the FIRST pass took: every session an attack or a
    control returned, the later sessions of a multi-session attack included (labelled or not)."""
    touched: set[str] = set()

    def spy(fn: AttackFn) -> AttackFn:
        def wrapped(
            session: list[RawCall], rng: np.random.Generator, ctx: AttackContext
        ) -> AttackResult | None:
            res = fn(session, rng, ctx)
            if res is not None and not TOPUP_ID.fullmatch(ctx.attack_id):
                i = ctx.session_index
                window = ctx.persona_sessions[i : i + 1 + len(res.extra_sessions)]
                touched.update(s[0].session_id for s in window)
            return res

        return wrapped

    with pytest.MonkeyPatch.context() as mp:
        for name, fn in list(ATTACKS.items()):
            mp.setitem(ATTACKS, name, spy(fn))
        data = gen(out, config)
    return data, touched


def _session_order(benign: Data) -> dict[str, list[str]]:
    """Each persona's session ids in time order (the file is sorted by time)."""
    out: dict[str, list[str]] = {}
    seen: set[object] = set()
    for t in benign.traces:
        if t["session_id"] not in seen:
            seen.add(t["session_id"])
            out.setdefault(str(t["agent_id"]), []).append(str(t["session_id"]))
    return out


def _calls(benign: Data) -> dict[str, list[dict[str, object]]]:
    out: dict[str, list[dict[str, object]]] = {}
    for t in benign.traces:
        out.setdefault(str(t["session_id"]), []).append(t)
    return out


def _spike_candidates(world: World, calls: list[dict[str, object]]) -> list[Cand]:
    """``(tool, arg, value)`` of every positive money/count argument of a write call (spec 4.6)."""
    out: list[Cand] = []
    for t in calls:
        tool, args = str(t["tool_id"]), t["args"]
        assert isinstance(args, dict)
        spec = world.catalog[tool]
        if spec.kind != "write":
            continue
        for name, a in spec.args.items():
            v = args.get(name)
            money_count = a.shape in ("money", "count") or (
                a.type == "numeric" and name in MONEY_COUNT_NAMES
            )
            if money_count and isinstance(v, int | float) and not isinstance(v, bool) and v > 0:
                out.append((tool, name, v))
    return out


def _reach(v: int | float) -> float:
    """The largest value a spike of ``v`` writes: ``v x 100`` as a whole number or in cents."""
    return int(round(v * M_MAX)) if isinstance(v, int) else round(v * M_MAX, 2)


def _bounds(world: World, benign: Data) -> dict[str, dict[tuple[str, str], float]]:
    """Each persona's far bounds, from its own benign sessions (``far_bounds``, Task 1.19)."""
    calls = _calls(benign)
    return {
        agent: far_bounds(
            [[RawCall.model_validate(t) for t in calls[s]] for s in sids],
            world.persona(agent),
            world.catalog,
        )
        for agent, sids in _session_order(benign).items()
    }


def _usable(world: World, benign: Data, mode: str) -> dict[str, tuple[int, int]]:
    """``session id -> (spike candidates, candidates the top-up can use)``: in mode "far" only a
    candidate whose value x 100 reaches the field's far bound is usable (A1.4)."""
    bounds = _bounds(world, benign)
    out: dict[str, tuple[int, int]] = {}
    for sid, calls in _calls(benign).items():
        cands = _spike_candidates(world, calls)
        far = bounds[str(calls[0]["agent_id"])]
        ok = [c for c in cands if mode == "own" or _reach(c[2]) >= far[(c[0], c[1])]]
        out[sid] = (len(cands), len(ok))
    return out


def replay_topup(
    world: World, benign: Data, touched: set[str], *, mode: str, rate: float, seed: int = 0
) -> dict[str, str]:
    """A1.4's top-up pass replayed on the benign dataset, ``session id -> attack id``: personas in
    sorted order, sessions in time order from the warm-up, a touched session or one with no usable
    candidate skipped WITHOUT a draw, else one draw on the top-up stream and, below the rate, the
    spike's own two draws on that same stream."""
    rng = np.random.default_rng([seed, stable_hash("attacks:magnitude_topup")])
    order, usable = _session_order(benign), _usable(world, benign, mode)
    out: dict[str, str] = {}
    for agent in sorted(order):
        k = 0
        for sid in order[agent][WARMUP:]:
            n_cands, n_ok = usable[sid]
            if sid in touched or n_ok == 0:
                continue
            if rng.random() < rate:
                rng.integers(n_cands)
                rng.uniform(np.log(M_MIN), np.log(M_MAX))
                out[sid] = f"atk-{agent}-m{k:05d}"
                k += 1
    return out


@dataclass
class Topup:
    far_half: Data  # mode "far", top-up rate 0.5
    far_one: Data  # mode "far", top-up rate 1.0
    own_one: Data  # mode "own", top-up rate 1.0
    far_touched: set[str]  # the sessions the first pass took (a dropped spike takes none)
    own_touched: set[str]


@pytest.fixture(scope="module")
def topup(tmp_path_factory: pytest.TempPathFactory) -> Topup:
    root = tmp_path_factory.mktemp("magnitude-topup")
    far_one, far_touched = gen_spying(root / "far-one", topup_cfg(1.0))
    own_one, own_touched = gen_spying(root / "own-one", topup_cfg(1.0, "own"))
    far_half = gen(root / "far-half", topup_cfg(0.5))
    return Topup(far_half, far_one, own_one, far_touched, own_touched)


def test_topup_does_not_change_the_first_pass(trio: Trio, topup: Topup) -> None:
    pairs = [(trio.far, topup.far_half), (trio.far, topup.far_one), (trio.own, topup.own_one)]
    for a, b in pairs:  # the same config without (a) and with (b) the top-up
        assert not _topups(a)
        topped = {lab["envelope_id"] for lab in b.labels if _is_topup(lab)}
        assert len(topped) == len(_topups(b)) >= 50  # one spiked call per top-up attack
        assert [t["envelope_id"] for t in a.traces] == [t["envelope_id"] for t in b.traces]
        # every other label row and call is byte-identical (raw JSON lines) ...
        assert _without(a.label_lines, a.labels, topped) == _without(
            b.label_lines, b.labels, topped
        )
        assert _without(a.trace_lines, a.traces, topped) == _without(
            b.trace_lines, b.traces, topped
        )
        # ... and that rest holds every first-pass attack; what the top-up took was benign
        assert b.attack_ids() - set(_topups(b).values()) == a.attack_ids()
        was = {lab["envelope_id"]: lab for lab in a.labels}
        assert all(was[e]["label"] == "benign" and was[e]["attack_id"] is None for e in topped)
        assert a.manifest.files != b.manifest.files


def test_topup_rate_one_spikes_every_eligible_untouched_session(
    trio: Trio, topup: Topup, world: World
) -> None:
    d, touched = topup.far_one, topup.far_touched
    got = _topups(d)
    order = _session_order(trio.benign)
    index = {sid: i for sids in order.values() for i, sid in enumerate(sids)}
    usable = _usable(world, trio.benign, "far")
    # eligible: post-warm-up, untouched by the first pass, with a feasible candidate
    eligible = {
        sid for sid, (_, ok) in usable.items() if index[sid] >= WARMUP and sid not in touched and ok
    }
    assert set(got) == eligible and len(eligible) >= 100
    assert all(TOPUP_ID.fullmatch(a) for a in got.values())
    labs = [lab for lab in d.labels if _is_topup(lab)]
    assert all(lab["attack_family"] == SPIKE and lab["label"] == "attack" for lab in labs)
    assert not set(got) & touched
    assert all(index[sid] >= WARMUP for sid in got)
    # the first pass took more than the sessions that carry an attack label: its controls too
    sess = trio.far.session_of()
    attacked = {str(sess[lab["envelope_id"]]) for lab in trio.far.labels if lab["attack_id"]}
    assert attacked < touched
    # sessions with a candidate but no feasible one, and warm-up sessions, were really skipped
    assert any(n and not ok and sid not in touched for sid, (n, ok) in usable.items())
    assert any(ok and index[sid] < WARMUP for sid, (_, ok) in usable.items())
    # ids number each persona's top-ups from 0 in session order
    assert got == replay_topup(world, trio.benign, touched, mode="far", rate=1.0)
    for agent, sids in order.items():
        ids = [got[sid] for sid in sids if sid in got]
        assert ids == [f"atk-{agent}-m{k:05d}" for k in range(len(ids))]
    assert {a for a, sids in order.items() if any(s in got for s in sids)} == {
        "sales-ops-a",
        "support-refunds-a",
    }
    # each top-up is a far spike: at or beyond its field's bound, strength the realized multiplier
    bounds = _bounds(world, trio.benign)
    topped = {lab["envelope_id"] for lab in labs}
    first = {lab["envelope_id"] for lab in trio.far.spikes()}
    values = _spiked_values(d, trio.benign)
    assert len(values) == len(topped) + len(first)
    assert all(new >= bounds[a][(t, n)] for a, t, n, new, _ in values)
    assert all(3.0 <= _num(lab["strength"]) <= 100.0 + 1e-9 for lab in labs)


def test_topup_counts_in_manifest(trio: Trio, topup: Topup) -> None:
    for d, first in ((topup.far_half, trio.far), (topup.far_one, trio.far)):
        m = d.manifest
        assert set(m.counts) == CENSUS_KEYS
        n_topup = len(_topups(d))
        assert m.counts["magnitude_topup"] == n_topup > 0
        assert m.counts["magnitude_first_pass"] + n_topup == len(d.attack_ids(SPIKE))
        # the first pass's own counts are those of the dataset without the top-up
        for key in ("magnitude_first_pass", "magnitude_dropped"):
            assert m.counts[key] == first.manifest.counts[key] > 0
        assert m.counts["attacks"] == len(d.attack_ids())
        assert m.counts["attacks"] == first.manifest.counts["attacks"] + n_topup
        assert m.config.magnitude_topup_rate > 0
    assert len(_topups(topup.far_half)) < len(_topups(topup.far_one))


def test_manifest_counts_magnitude_strength_ge_10(trio: Trio, topup: Topup) -> None:
    seen = []
    for d in (trio.far, topup.far_half, topup.far_one, topup.own_one):
        ge_10 = sum(_num(lab["strength"]) >= 10.0 for lab in d.spikes())
        assert d.manifest.counts["magnitude_strength_ge_10"] == ge_10
        assert len(d.spikes()) == len(d.attack_ids(SPIKE))  # one label per spike attack
        seen.append((ge_10, len(d.spikes())))
    assert all(0 < ge_10 <= n for ge_10, n in seen)
    own_ge_10, own_n = seen[-1]
    assert own_ge_10 < 0.8 * own_n  # mode "own": a good share of the spikes is below 10 x
    # the default dataset (mode "own", no top-up) carries no census at all
    assert "magnitude_strength_ge_10" not in trio.own.manifest.counts


def test_topup_is_deterministic_and_uses_its_own_stream(
    trio: Trio, topup: Topup, world: World, tmp_path: Path
) -> None:
    again = gen(tmp_path, topup_cfg(0.5))
    assert again.manifest == topup.far_half.manifest  # config, sha256 of both files, counts
    for name, digest in again.manifest.files.items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
    # rate 0: no top-up id, and the bytes of Task 1.19's far dataset
    assert topup_cfg(0.0) == small_cfg(magnitude_mode="far") == trio.far.manifest.config
    assert trio.far.manifest.files == SMALL_FAR_FILES != again.manifest.files
    assert not _topups(trio.far) and trio.far.manifest.counts["magnitude_topup"] == 0
    # the third stream (A1.4): its own, a function of the seed
    want = np.random.default_rng([0, stable_hash("attacks:magnitude_topup")])
    draws = topup_rng(0).random(8).tolist()
    assert draws == want.random(8).tolist()
    assert draws != attack_rng(0).random(8).tolist() and draws != topup_rng(1).random(8).tolist()
    # ... and the pass draws exactly what A1.4 says from it: replaying those draws on that stream
    # (one per eligible untouched session, then the spike's two) picks the same sessions and ids
    got = _topups(topup.far_half)
    assert got == replay_topup(world, trio.benign, topup.far_touched, mode="far", rate=0.5)
    n_eligible = len(_topups(topup.far_one))
    assert 0.4 * n_eligible < len(got) < 0.6 * n_eligible


def test_topup_in_own_mode_uses_all_candidates(trio: Trio, topup: Topup, world: World) -> None:
    d, touched = topup.own_one, topup.own_touched
    got = _topups(d)
    assert got == replay_topup(world, trio.benign, touched, mode="own", rate=1.0)
    # a session whose candidates are all infeasible in mode "far" is still topped up ...
    usable = _usable(world, trio.benign, "far")
    assert sum(usable[sid][1] == 0 for sid in got) >= 1
    assert set(_topups(topup.far_one)) - topup.own_touched < set(got)
    # ... and the spike is spec 4.6's: the call's own value x m, wherever that lands
    bounds = _bounds(world, trio.benign)
    topped = {lab["envelope_id"] for lab in d.labels if _is_topup(lab)}
    strengths = [_num(lab["strength"]) for lab in d.labels if _is_topup(lab)]
    assert all(3.0 <= s <= 100.0 + 1e-9 for s in strengths) and min(strengths) < 4.0
    values = _spiked_values(d, trio.benign)  # also checks strength == new / own, one arg changed
    assert len(values) == len(topped) + len(trio.own.spikes())
    assert sum(new < bounds[a][(t, n)] for a, t, n, new, _ in values) > len(trio.own.spikes())
    m = d.manifest
    assert set(m.counts) == CENSUS_KEYS  # the census is there in mode "own" with a top-up
    assert m.counts["magnitude_topup"] == len(got) and m.counts["magnitude_dropped"] == 0
    assert m.counts["magnitude_first_pass"] == len(trio.own.attack_ids(SPIKE))


def test_topup_needs_the_spike_family(tmp_path: Path) -> None:
    # The top-up adds magnitude spikes to a dataset that has the family; without it (or with no
    # family at all) nothing is topped up, and the census says so.
    for name, families in (("none", []), ("other", ["novel_target", "benign_unusual"])):
        d = gen(tmp_path / name, small_cfg(days=5, families=families, magnitude_topup_rate=1.0))
        assert not _topups(d) and not d.spikes()
        assert set(d.manifest.counts) == CENSUS_KEYS
        assert {k: v for k, v in d.manifest.counts.items() if k.startswith("magnitude_")} == {
            "magnitude_first_pass": 0,
            "magnitude_dropped": 0,
            "magnitude_topup": 0,
            "magnitude_strength_ge_10": 0,
        }
    with_family = gen(tmp_path / "spike", small_cfg(days=5, magnitude_topup_rate=1.0))
    assert len(_topups(with_family)) >= 10


def test_topup_rate_is_validated() -> None:
    assert cfg().magnitude_topup_rate == 0.0
    assert GenerateConfig.model_fields["magnitude_topup_rate"].default == 0.0
    for ok in (0.0, 0.015, 1.0):
        assert cfg(magnitude_topup_rate=ok).magnitude_topup_rate == ok
    for bad in (-0.001, 1.001, math.inf, -math.inf, math.nan):
        with pytest.raises(ValueError):
            cfg(magnitude_topup_rate=bad)


def test_sw1_config() -> None:
    c = sw1_config()
    assert (c.seed, c.days, c.attack_rate) == (0, 90, 0.015)
    assert c.magnitude_mode == "far" and c.magnitude_topup_rate == c.attack_rate
    assert c.magnitude_far_mult == 5.0
    assert c.families == sorted(ATTACKS) and len(c.families) == 8
    assert sw1_config(seed=2).seed == 2
    assert sw1_config(seed=2) == c.model_copy(update={"seed": 2})
    # all 20 personas of the shipped world, uncapped
    assert (c.personas_dir, c.tasks_dir, c.catalog_dir) == (PERSONAS_DIR, TASKS_DIR, CATALOG_DIR)
    assert c.persona_ids is None and c.max_sessions is None
    assert len(load_personas(c.personas_dir)) == 20
    # without the two opt-ins it is the pre-amendment SW1, field for field
    legacy = GenerateConfig(
        seed=0,
        personas_dir=PERSONAS_DIR,
        tasks_dir=TASKS_DIR,
        catalog_dir=CATALOG_DIR,
        days=90,
        families=sorted(ATTACKS),
        attack_rate=0.015,
    )
    assert c.model_copy(update={"magnitude_mode": "own", "magnitude_topup_rate": 0.0}) == legacy


def test_pre_a1_sw1_manifest_is_refused() -> None:
    old = sw1_config().model_dump(mode="json")  # a manifest config written before the amendment
    for k in ("magnitude_mode", "magnitude_far_mult", "magnitude_topup_rate"):
        del old[k]
    # so the SW1 guards refuse a stale traces/ directory
    assert GenerateConfig.model_validate(old) != sw1_config()
    assert GenerateConfig.model_validate(sw1_config().model_dump(mode="json")) == sw1_config()


def test_cli_magnitude_topup_rate(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base = ["--seed", "0", "--days", "3", "--families", SPIKE]
    assert main([*base, "--out", str(tmp_path / "d")]) == 0
    m = orjson.loads((tmp_path / "d" / "manifest.json").read_bytes())
    assert m["config"]["magnitude_topup_rate"] == 0.0
    assert set(m["counts"]) == {"sessions", "envelopes", "attacks"}
    assert main([*base, "--out", str(tmp_path / "t"), "--magnitude-topup-rate", "1.0"]) == 0
    t = orjson.loads((tmp_path / "t" / "manifest.json").read_bytes())
    assert t["config"]["magnitude_topup_rate"] == 1.0 and t["config"]["magnitude_mode"] == "own"
    assert set(t["counts"]) == CENSUS_KEYS
    assert t["counts"]["magnitude_topup"] >= 10
    assert t["counts"]["attacks"] == m["counts"]["attacks"] + t["counts"]["magnitude_topup"]
    out = tmp_path / "x"
    for bad in ("-0.1", "1.5", "nan"):
        with pytest.raises(SystemExit) as exc:
            main([*base, "--out", str(out), "--magnitude-topup-rate", bad])
        assert exc.value.code == 2  # a usage error
        err = capsys.readouterr().err
        assert "magnitude_topup_rate" in err and "Traceback" not in err
        assert not out.exists()


def test_cli_topup_rate_default_is_the_config_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    field = GenerateConfig.model_fields["magnitude_topup_rate"]
    monkeypatch.setattr(field, "default", 0.25)  # the CLI holds no copy of the default
    assert main(["--days", "1", "--max-sessions", "3", "--out", str(tmp_path)]) == 0
    m = orjson.loads((tmp_path / "manifest.json").read_bytes())
    assert m["config"]["magnitude_topup_rate"] == 0.25
