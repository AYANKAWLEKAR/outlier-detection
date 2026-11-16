"""A call is scored before its response exists (T-58, final review I1).

The sandbox proxy scores a call, forwards it, then updates with the response; the bench replays
envelopes that already carry their response. ``score`` must read nothing of the current call's
own response, so both paths give the same results and the same state. Before T-58 the spin score
read the current call's ``delta_i``: proxy-style SPIN at score time 543 -> 0 on this dataset.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from pathlib import Path

import pytest

from agentfw.bench.ci_regress import CI_DATASET
from agentfw.bench.dataset import load_dir, load_schema_hints
from agentfw.bench.registry import TRAJ_SWITCHES
from agentfw.core.config import EngineConfig
from agentfw.core.engine import KNOWN_DETECTORS, Engine
from agentfw.core.envelope import DetectorResult, Outcome, RawCall
from agentfw.core.hints import ToolHint
from agentfw.core.normalize import SessionTracker, normalize
from agentfw.synth.generate import generate


@pytest.fixture(scope="module")
def runaway(tmp_path_factory: pytest.TempPathFactory) -> tuple[list[RawCall], dict[str, ToolHint]]:
    """The CI dataset with the runaway families, half the sessions attacked (probe3's set)."""
    d = tmp_path_factory.mktemp("runaway")
    families = [*CI_DATASET.families, "loop", "spin", "stall"]
    generate(CI_DATASET.model_copy(update={"families": families, "attack_rate": 0.5}), d)
    raws, _, _ = load_dir(d)
    raws.sort(key=lambda r: (r.ts_unix_ms, r.session_id, r.seq_no))
    return raws, load_schema_hints(d)


def _run(
    raws: list[RawCall], hints: dict[str, ToolHint], cfg: EngineConfig, *, proxy: bool, tmp: Path
) -> tuple[list[DetectorResult], bytes]:
    eng = Engine(cfg, enabled=KNOWN_DETECTORS, hints=hints)
    tr = SessionTracker()
    out = []
    for r in raws:
        pre = r.model_copy(update={"response_meta": None}) if proxy else r
        res = eng.score(normalize(pre, tr))
        out.append(dataclasses.replace(res, latency_us=0.0))
        e = normalize(r, tr)  # the call with its response, for update and the tracker
        eng.update(e, Outcome.ALLOWED)
        tr.observe(e)
    p = tmp / ("proxy.json" if proxy else "bench.json")
    eng.store.dump(p)
    return out, p.read_bytes()


@pytest.mark.parametrize(
    "cfg",
    [EngineConfig(sessions=True), EngineConfig().replace(**TRAJ_SWITCHES)],
    ids=["sessions", "every-switch"],
)
def test_proxy_style_scoring_equals_replay(
    runaway: tuple[list[RawCall], dict[str, ToolHint]], cfg: EngineConfig, tmp_path: Path
) -> None:
    raws, hints = runaway
    bench, bench_store = _run(raws, hints, cfg, proxy=False, tmp=tmp_path)
    proxy, proxy_store = _run(raws, hints, cfg, proxy=True, tmp=tmp_path)
    assert proxy == bench
    assert proxy_store == bench_store
    flags = Counter(f.name for r in bench if r.trajectory is not None for f in r.trajectory.flags)
    assert flags["SPIN"] > 0  # non-vacuous: spin fires on the proxy path too
