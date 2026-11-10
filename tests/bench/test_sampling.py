from __future__ import annotations

import math
from pathlib import Path

import orjson
import pytest

from agentfw.bench.replay import SplitSpec, assign_split, read_score_table
from agentfw.bench.run import run
from agentfw.bench.sampling import sample_counts, stratified_sample
from agentfw.core.envelope import Label, RawCall
from agentfw.sandbox.llm import ChatTurn, ScriptedChatClient, Usage
from tests.factories import make_raw

FAMILIES = ("magnitude_spike", "novel_target", "velocity_burst")


def _data(n: int = 1000) -> tuple[list[RawCall], dict[str, Label]]:
    raws: list[RawCall] = []
    labels: dict[str, Label] = {}
    for i in range(n):
        raw = make_raw(
            agent_id=f"agent{i % 4}",
            session_id=f"s{i // 10}",
            seq_no=i % 10,
            ts_unix_ms=1_700_000_000_000 + i * 1000,
        )
        raws.append(raw)
        if i % 20 == 7:
            fam = FAMILIES[(i // 20) % 3]
            labels[raw.envelope_id] = Label(
                envelope_id=raw.envelope_id, label="attack", attack_family=fam, attack_id=f"x{i}"
            )
        else:
            labels[raw.envelope_id] = Label(envelope_id=raw.envelope_id, label="benign")
    return raws, labels


def test_sample_size_families_and_test_split_only() -> None:
    raws, labels = _data()
    ids = stratified_sample(raws, labels, n=100, seed=0)
    assert len(ids) == 100
    split = assign_split(raws, SplitSpec())
    assert all(split[i] == "test" for i in ids)
    counts = sample_counts(raws, labels, ids)
    assert set(counts["attack_by_family"]) == set(FAMILIES)
    assert sum(counts["attack_by_family"].values()) <= 50
    assert counts["n"] == 100
    assert counts["n_attack"] + counts["n_benign"] == 100
    assert set(counts["benign_by_agent"]) == {f"agent{i}" for i in range(4)}


def test_sample_deterministic_per_seed() -> None:
    raws, labels = _data()
    a = stratified_sample(raws, labels, n=80, seed=1)
    assert a == stratified_sample(raws, labels, n=80, seed=1)
    assert a != stratified_sample(raws, labels, n=80, seed=2)


def test_sample_caps_at_available_and_attack_half() -> None:
    raws, labels = _data()
    ids = stratified_sample(raws, labels, n=10_000, seed=0)
    split = assign_split(raws, SplitSpec())
    assert len(ids) == sum(1 for r in raws if split[r.envelope_id] == "test")
    few = stratified_sample(raws, labels, n=6, seed=0)
    assert len(few) == 6
    assert sample_counts(raws, labels, few)["n_attack"] <= 3


def test_run_sample_flag_records_spec_counts_and_caveats(
    tiny_dataset: Path, tmp_path: Path
) -> None:
    out = run(
        tiny_dataset,
        "static_threshold",
        SplitSpec(),
        tmp_path / "o",
        n_boot=5,
        sample="stratified:40",
    )
    rep = orjson.loads((out / "report.json").read_bytes())
    ex = rep["extras"]
    assert ex["sample"]["spec"] == "stratified:40"
    assert ex["sample"]["n"] == ex["sample"]["n_attack"] + ex["sample"]["n_benign"] <= 40
    assert ex["batch"] is True
    joined = " ".join(ex["notes"])
    assert "sampled" in joined and "batch" in joined
    rows = read_score_table(out / "scores.parquet")
    assert sum(1 for r in rows if not math.isnan(r.fused_score)) == ex["sample"]["n"]


def test_run_rejects_bad_sample_spec(tiny_dataset: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="stratified:N"):
        run(tiny_dataset, "random", SplitSpec(), tmp_path / "o", n_boot=5, sample="random:5")


def test_run_llm_judge_offline_completes_and_records_safe_config(
    tiny_dataset: Path, tmp_path: Path
) -> None:
    client = ScriptedChatClient(
        [ChatTurn('{"suspicion": 0.4}', [], Usage(1, 1)) for _ in range(500)]
    )
    out = run(
        tiny_dataset,
        "llm_judge",
        SplitSpec(),
        tmp_path / "o",
        n_boot=5,
        sample="stratified:10",
        client=client,
        cache_dir=tmp_path / "cache",
        model_id="stub-model",
    )
    rep = orjson.loads((out / "report.json").read_bytes())
    cfg = rep["meta"]["detector_config"]
    assert cfg == {
        "model_id": "stub-model",
        "window": 8,
        "prompt_version": "v1",
        "cache_dir": str(tmp_path / "cache"),
    }
    assert "batch" not in rep["extras"]
    assert len(client.calls) >= 1
    assert any("splits multi-envelope attacks" in n for n in rep["extras"]["notes"])


def test_cli_judge_requires_env_and_builds_live_client_only_from_env(
    tiny_dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agentfw.bench import run as run_mod

    for k in ("JUDGE_BASE_URL", "JUDGE_API_KEY", "JUDGE_MODEL"):
        monkeypatch.delenv(k, raising=False)
    base = ["--traces", str(tiny_dataset), "--detector", "llm_judge", "--out", str(tmp_path / "o")]
    with pytest.raises(SystemExit):
        run_mod.main(base)

    seen: dict[str, object] = {}

    class _Stub:
        def __init__(self, base_url: str, api_key: str, model: str) -> None:
            seen.update(base_url=base_url, model=model)

        async def complete(
            self, messages: list[dict[str, object]], tools: list[dict[str, object]]
        ) -> ChatTurn:
            return ChatTurn('{"suspicion": 0.2}', [], Usage(1, 1))

    monkeypatch.setattr(run_mod, "OpenAICompatClient", _Stub)
    monkeypatch.setenv("JUDGE_BASE_URL", "http://judge.test/v1")
    monkeypatch.setenv("JUDGE_API_KEY", "sk-should-never-be-recorded")
    run_mod.main(
        [
            *base,
            "--judge-model",
            "m1",
            "--judge-cache",
            str(tmp_path / "c"),
            "--sample",
            "stratified:6",
            "--n-boot",
            "5",
        ]
    )
    assert seen == {"base_url": "http://judge.test/v1", "model": "m1"}
    text = (tmp_path / "o" / "report.json").read_text()
    assert "sk-should-never-be-recorded" not in text and "judge.test" not in text
