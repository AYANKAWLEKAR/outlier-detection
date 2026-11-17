"""Captured tool schemas (F3): ``schemas.jsonl`` rows, loading, hints, generator and bench."""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Mapping
from pathlib import Path

import orjson
import pytest

from agentfw.bench.ablation import replay_detector
from agentfw.bench.ci_regress import CI_DATASET
from agentfw.bench.dataset import load_schema_hints
from agentfw.bench.registry import ENGINE_VARIANTS, make_bench_detector
from agentfw.bench.replay import BENCH_OUTCOMES, ScoreRow, SplitSpec, canonical_hash
from agentfw.core.detectors.vital import LOOP
from agentfw.core.engine import VITAL, Engine
from agentfw.core.hints import ToolAnnotations, ToolHint
from agentfw.core.schemas import (
    ToolSchema,
    hints_from_schemas,
    load_schemas,
    schema_hash,
)
from agentfw.synth.generate import GenerateConfig, generate
from agentfw.synth.load import CATALOG_DIR, PERSONAS_DIR, TASKS_DIR, load_catalog

RO = {"readOnlyHint": True, "idempotentHint": True}


def _row(tool: str, annotations: dict[str, bool], at: int, schema: Mapping[str, object]) -> bytes:
    h = schema_hash(schema, annotations)
    row = {
        "tool_id": tool,
        "input_schema": schema,
        "annotations": annotations,
        "captured_at": at,
        "hash": h,
    }
    return orjson.dumps(row) + b"\n"


def test_load_schemas_last_row_wins(tmp_path: Path) -> None:
    p = tmp_path / "schemas.jsonl"
    s1 = {"type": "object", "properties": {"q": {"type": "string"}}}
    s2 = {"type": "object", "properties": {}}
    p.write_bytes(
        _row("kb.search", RO, 1, s1)
        + b"\n"  # blank lines are skipped
        + _row("crm.delete", {"destructiveHint": True}, 2, s2)
        + _row("kb.search", {"readOnlyHint": False}, 3, s2)
    )
    got = load_schemas(p)
    assert sorted(got) == ["crm.delete", "kb.search"]
    kb = got["kb.search"]
    assert isinstance(kb, ToolSchema)
    assert kb.captured_at == 3 and kb.input_schema == s2
    assert kb.annotations == ToolAnnotations(read_only=False)
    assert kb.hash == schema_hash(s2, {"readOnlyHint": False})
    assert got["crm.delete"].annotations == ToolAnnotations(destructive=True)


def test_schema_rows_round_trip_and_hash_is_canonical() -> None:
    s = ToolSchema(
        tool_id="kb.search",
        input_schema={"type": "object", "properties": {"b": {}, "a": {}}},
        annotations=ToolAnnotations(read_only=True, open_world=False),
        captured_at=7,
        hash=schema_hash({"properties": {"a": {}, "b": {}}, "type": "object"}, {}),
    )
    row = s.to_row()
    assert row["annotations"] == {"readOnlyHint": True, "openWorldHint": False}
    assert ToolSchema.from_row(orjson.loads(orjson.dumps(row))) == s
    # key order does not change the hash; content does
    assert schema_hash({"b": 1, "a": 2}, {"readOnlyHint": True}) == schema_hash(
        {"a": 2, "b": 1}, {"readOnlyHint": True}
    )
    assert schema_hash({"a": 2}, {}) != schema_hash({"a": 3}, {})


def test_load_schemas_rejects_unknown_annotation_keys(tmp_path: Path) -> None:
    p = tmp_path / "schemas.jsonl"
    p.write_bytes(_row("kb.search", {"readOnlyHnt": True}, 1, {}))
    with pytest.raises(ValueError, match="readOnlyHnt"):
        load_schemas(p)


def test_load_schemas_rejects_a_row_missing_a_key(tmp_path: Path) -> None:
    p = tmp_path / "schemas.jsonl"
    p.write_bytes(orjson.dumps({"tool_id": "kb.search", "input_schema": {}, "hash": "h"}) + b"\n")
    with pytest.raises(ValueError, match="captured_at"):
        load_schemas(p)


def test_hints_from_schemas_keeps_kind_and_field_hints() -> None:
    schemas = {
        "kb.search": ToolSchema("kb.search", {}, ToolAnnotations(read_only=True), 1, "h"),
        "crm.delete": ToolSchema("crm.delete", {}, ToolAnnotations(destructive=True), 1, "h"),
    }
    base = {
        "crm.delete": ToolHint(
            kind="write",
            field_hints={"n": "count"},
            annotations=ToolAnnotations(destructive=False),
        ),
        "mail.send": ToolHint(kind="write"),
    }
    got = hints_from_schemas(schemas, base)
    assert got["crm.delete"] == ToolHint(
        kind="write",
        field_hints={"n": "count"},
        annotations=ToolAnnotations(destructive=True),
        has_schema=True,
    )
    assert got["kb.search"] == ToolHint(
        annotations=ToolAnnotations(read_only=True), has_schema=True
    )
    assert got["mail.send"] is base["mail.send"]  # no schema: the base hint as it was
    assert ToolHint().has_schema is False


# --- generator ---------------------------------------------------------------------------------


def _cfg(**kw: object) -> GenerateConfig:
    base: dict[str, object] = {
        "seed": 0,
        "personas_dir": PERSONAS_DIR,
        "persona_ids": ["sales-ops-a"],
        "tasks_dir": TASKS_DIR,
        "catalog_dir": CATALOG_DIR,
        "days": 1,
    }
    return GenerateConfig.model_validate({**base, **kw})


def test_catalog_annotations_are_complete_and_match_kind() -> None:
    for tool, spec in load_catalog().items():
        assert spec.annotations is not None, tool
        a = spec.annotations.to_core()
        assert None not in (a.read_only, a.destructive, a.idempotent, a.open_world), tool
        assert a.read_only is (spec.kind == "read"), tool
        if a.read_only:
            assert a.destructive is False and a.idempotent is True, tool


def test_generate_writes_one_schema_row_per_catalog_tool(tmp_path: Path) -> None:
    m = generate(_cfg(), tmp_path / "a")
    data = (tmp_path / "a" / "schemas.jsonl").read_bytes()
    assert m.files["schemas.jsonl"] == hashlib.sha256(data).hexdigest()
    lines = data.splitlines()
    rows = [orjson.loads(x) for x in lines]
    catalog = load_catalog()
    assert [r["tool_id"] for r in rows] == sorted(catalog)  # every tool, sorted
    assert lines == [orjson.dumps(r, option=orjson.OPT_SORT_KEYS) for r in rows]  # sorted keys
    assert {r["captured_at"] for r in rows} == {_cfg().start_ms}  # the generation epoch
    schemas = load_schemas(tmp_path / "a" / "schemas.jsonl")
    for tool, spec in catalog.items():
        s = schemas[tool]
        assert spec.annotations is not None
        assert s.annotations == spec.annotations.to_core()
        assert s.hash == schema_hash(s.input_schema, s.to_row()["annotations"])  # type: ignore[arg-type]
        props = s.input_schema["properties"]
        assert isinstance(props, dict) and sorted(props) == sorted(spec.args)
    refund = schemas["stripe.create_refund"].input_schema["properties"]
    assert isinstance(refund, dict)
    assert refund["amount"] == {"type": "number"}
    assert refund["reason"] == {
        "type": "string",
        "enum": ["duplicate", "fraudulent", "requested_by_customer", "manager_approved"],
    }
    mass = schemas["salesforce.mass_update"].input_schema["properties"]
    assert isinstance(mass, dict)
    assert mass["record_ids"] == {"type": "array", "items": {"type": "string"}}
    # deterministic: a second run (other day, same config) writes the same bytes
    generate(_cfg(), tmp_path / "b")
    assert (tmp_path / "b" / "schemas.jsonl").read_bytes() == data
    generate(_cfg(start_ms=_cfg().start_ms + 1), tmp_path / "c")
    assert (tmp_path / "c" / "schemas.jsonl").read_bytes() != data


# --- bench auto-load ---------------------------------------------------------------------------


def test_load_schema_hints(tiny_dataset: Path, tmp_path: Path) -> None:
    hints = load_schema_hints(tiny_dataset)
    assert set(hints) == set(load_catalog())
    assert all(h.has_schema and h.kind is None and not h.field_hints for h in hints.values())
    bare = tmp_path / "bare"
    bare.mkdir()
    assert load_schema_hints(bare) == {}


def test_make_bench_detector_gives_engines_the_dataset_schemas(tiny_dataset: Path) -> None:
    assert "engine" in ENGINE_VARIANTS and "engine-vital" in ENGINE_VARIANTS
    assert "static_threshold" not in ENGINE_VARIANTS
    det = make_bench_detector("engine", tiny_dataset)
    assert isinstance(det, Engine) and det.hints == load_schema_hints(tiny_dataset)
    own = {"x.y": ToolHint(kind="read")}
    det2 = make_bench_detector("engine", tiny_dataset, hints=own)
    assert isinstance(det2, Engine) and det2.hints is own  # explicit hints are kept
    assert not isinstance(make_bench_detector("static_threshold", tiny_dataset), Engine)


def test_schemas_change_no_bench_score_with_default_config(
    tiny_dataset: Path, tmp_path: Path
) -> None:
    bare = tmp_path / "bare"
    shutil.copytree(tiny_dataset, bare)
    (bare / "schemas.jsonl").unlink()
    with_rows, _ = replay_detector(tiny_dataset, "engine", SplitSpec(), None, outcomes="review")
    without, det = replay_detector(bare, "engine", SplitSpec(), None, outcomes="review")
    assert isinstance(det, Engine) and det.hints == {}
    assert canonical_hash(with_rows) == canonical_hash(without)
    assert [r.decision for r in with_rows] == [r.decision for r in without]
    assert [r.per_detector_json for r in with_rows] == [r.per_detector_json for r in without]


def _vital_scores(r: ScoreRow) -> dict[str, float]:
    items = orjson.loads(r.per_detector_json)
    return {d["field"]: d["score"] for d in items if d["detector"] == VITAL}


def test_engine_vital_under_schema_auto_load_on_the_ci_dataset(tmp_path: Path) -> None:
    """Pins current behaviour (review round 1, T-32): auto-loaded annotations never move a fused
    score or decision, but they do move vital scores. Every catalog read is read-only +
    idempotent, so it polls (T-26: 32-call loop window instead of 8). On the CI dataset exactly one
    row changes: a benign ``salesforce.run_soql`` whose ``__loop__`` goes 0.7 -> 1.0 (a new
    full LOOP; F2's polling demotion is Task 2.1)."""
    with_dir, bare = tmp_path / "with", tmp_path / "bare"
    generate(CI_DATASET, with_dir)
    shutil.copytree(with_dir, bare)
    (bare / "schemas.jsonl").unlink()
    a, _ = replay_detector(bare, "engine-vital", SplitSpec(), None, outcomes=BENCH_OUTCOMES)
    b, det = replay_detector(with_dir, "engine-vital", SplitSpec(), None, outcomes=BENCH_OUTCOMES)
    assert isinstance(det, Engine) and det.hints  # the schemas were loaded
    assert [(r.fused_score, r.decision) for r in a] == [(r.fused_score, r.decision) for r in b]
    changed = [(x, y) for x, y in zip(a, b, strict=True) if _vital_scores(x) != _vital_scores(y)]
    assert len(changed) == 1
    x, y = changed[0]
    assert (x.tool_id, x.label) == ("salesforce.run_soql", "benign")
    assert (_vital_scores(x)[LOOP], _vital_scores(y)[LOOP]) == (0.7, 1.0)


def test_captured_schema_is_independent_of_caller_mutations() -> None:
    from agentfw.core.hints import ToolAnnotations
    from agentfw.core.schemas import ToolSchema, schema_hash

    src = {"type": "object", "properties": {"id": {"type": "string"}}}
    s = ToolSchema.capture("t", src, ToolAnnotations(), 1)  # type: ignore[arg-type]
    src["properties"]["id"]["type"] = "integer"  # type: ignore[index]
    row = s.to_row()
    assert row["input_schema"]["properties"]["id"]["type"] == "string"  # type: ignore[index]
    row["input_schema"]["properties"]["id"]["type"] = "integer"  # type: ignore[index]
    assert s.to_row()["hash"] == schema_hash(s.to_row()["input_schema"], {})  # type: ignore[arg-type]
    back = ToolSchema.from_row(s.to_row())
    raw = s.to_row()
    again = ToolSchema.from_row(raw)
    raw["input_schema"]["properties"]["id"]["type"] = "integer"  # type: ignore[index]
    assert again.to_row() == back.to_row()
