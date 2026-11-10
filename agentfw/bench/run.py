"""Bench run CLI: replay a detector over a dataset and write scores + report.

``dataset_sha`` is the sha256 of ``traces.jsonl`` as recorded in the dataset manifest. The replay
protocol (``--outcomes``) defaults to the benchmark protocol, the oracle-reviewer approval loop
(``BENCH_OUTCOMES``, user decision 2026-11-10, decision 008); ``--outcomes allow`` is spec §5.2
allow-all. Baselines and the ablation table use the same protocol as the run; the report records
it in ``extras.outcomes`` and labels it in its header.
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import orjson

from agentfw.bench.ablation import run_ablation
from agentfw.bench.dataset import load_dir
from agentfw.bench.metrics import BOOTSTRAP_N, compute_all, gate_preview
from agentfw.bench.protocol import BatchFit
from agentfw.bench.registry import make_detector
from agentfw.bench.replay import (
    BENCH_OUTCOMES,
    OUTCOME_MODES,
    OutcomeMode,
    SplitSpec,
    canonical_hash,
    replay,
    write_score_table,
)
from agentfw.bench.report import RunMeta, build_report, git_sha, render_markdown, update_readme
from agentfw.bench.sampling import sample_counts, stratified_sample
from agentfw.sandbox.llm import OpenAICompatClient

__all__ = ["main", "run"]


def run(
    traces_dir: Path,
    detector: str,
    split: SplitSpec,
    out_dir: Path,
    *,
    n_boot: int = BOOTSTRAP_N,
    ablations: Sequence[str] | None = None,
    sample: str | None = None,
    outcomes: OutcomeMode = BENCH_OUTCOMES,
    **detector_kw: Any,
) -> Path:
    """Write ``scores.parquet``, ``report.json``, ``report.md`` into ``out_dir``; returns it.

    ``ablations``: registry variants to replay on the same dataset and split for the report's
    ablation table (``None``: no table). They use the default config, not ``detector_kw``, and the
    run's ``outcomes``.
    ``sample``: ``"stratified:N"`` scores only a stratified N-envelope test subsample (others get
    NaN and drop out of the metrics); the spec, actual counts and caveats go in the report.
    ``outcomes``: the replay protocol (:func:`agentfw.bench.replay.replay`), recorded as
    ``extras.outcomes``.
    """
    raws, labels, manifest = load_dir(traces_dir)
    det = make_detector(detector, **detector_kw)
    seed = manifest.config.seed
    sample_ids: set[str] | None = None
    extras: dict[str, object] = {"outcomes": outcomes}
    notes: list[str] = []
    if sample is not None:
        n = _parse_sample(sample)
        sample_ids = stratified_sample(raws, labels, n=n, seed=seed, split=split)
        extras["sample"] = {"spec": sample, "seed": seed, **sample_counts(raws, labels, sample_ids)}
        notes.append(
            f"sampled: scores cover a stratified test subsample ({sample}); attack prevalence is "
            "far above the dataset's, so AUPRC is not comparable with a full replay. Sampling "
            "splits multi-envelope attacks, so per-attack recall counts only sampled envelopes."
        )
    if isinstance(det, BatchFit):
        extras["batch"] = True
        notes.append(
            "batch: fit offline on the whole train split, not an online/causal learner. "
            "Isolation forest cannot score extrapolation above the train tail (a point far "
            "outside the train range scores like the extremes already seen), which depresses "
            "its recall on magnitude spikes."
        )
    if notes:
        extras["notes"] = notes
    rows = replay(raws, labels, det, split, sample_ids=sample_ids, outcomes=outcomes)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_score_table(rows, out_dir / "scores.parquet")
    all_ = compute_all(rows, warm_only=False, n_boot=n_boot, seed=seed)
    warm = compute_all(rows, warm_only=True, n_boot=n_boot, seed=seed)
    meta = RunMeta(
        git_sha=git_sha(),
        seed=seed,
        dataset_sha=manifest.files["traces.jsonl"],
        detector=detector,
        detector_config=_config_record(det, detector_kw),
        split=split,
        created_unix=int(time.time()),
    )
    report = build_report(
        all_,
        warm,
        meta,
        {
            "state_bytes": det.state_bytes(),
            "score_hash": canonical_hash(rows),
            "n_boot": n_boot,
            **extras,
        },
        gate=gate_preview(rows),
        ablations=None
        if ablations is None
        else run_ablation(traces_dir, ablations, split, outcomes=outcomes),
    )
    (out_dir / "report.json").write_bytes(orjson.dumps(report, option=orjson.OPT_INDENT_2))
    (out_dir / "report.md").write_text(render_markdown(report))
    return out_dir


_SECRET_HINTS = ("key", "secret", "token", "password")


def _config_record(det: object, kw: dict[str, Any]) -> dict[str, Any]:
    """JSON-safe, credential-free record of the detector config for the report."""
    rec = getattr(det, "config_record", None)
    if callable(rec):
        return dict(rec())
    out: dict[str, Any] = {}
    for k, v in kw.items():
        if any(h in k.lower() for h in _SECRET_HINTS):
            continue
        out[k] = v if isinstance(v, str | int | float | bool | type(None)) else type(v).__name__
    return out


def _live_judge_kwargs(model: str | None, cache: Path) -> dict[str, Any]:
    """Build the judge's live client from JUDGE_BASE_URL / JUDGE_API_KEY / JUDGE_MODEL.

    Only called when actually running ``--detector llm_judge``; the key is never recorded.
    """
    env = os.environ
    base, key = env.get("JUDGE_BASE_URL"), env.get("JUDGE_API_KEY")
    model = model or env.get("JUDGE_MODEL")
    if not (base and key and model):
        raise SystemExit(
            "llm_judge needs JUDGE_BASE_URL and JUDGE_API_KEY in the environment and a model "
            "(--judge-model or JUDGE_MODEL)"
        )
    client = OpenAICompatClient(base_url=base, api_key=key, model=model)
    return {"client": client, "cache_dir": cache, "model_id": model}


def _parse_sample(spec: str) -> int:
    kind, _, n = spec.partition(":")
    if kind != "stratified" or not n.isdigit() or int(n) < 1:
        raise ValueError(f"bad --sample {spec!r}; expected stratified:N")
    return int(n)


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m agentfw.bench.run")
    p.add_argument("--traces", type=Path, required=True)
    p.add_argument("--detector", required=True)
    p.add_argument("--split", choices=["time", "agent"], default="time")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-boot", type=int, default=BOOTSTRAP_N)
    p.add_argument(
        "--sample", default=None, help="stratified:N test subsample, e.g. stratified:2000"
    )
    p.add_argument("--judge-model", default=None, help="model id for --detector llm_judge")
    p.add_argument("--judge-cache", type=Path, default=Path("bench/cache"))
    p.add_argument(
        "--outcomes",
        choices=OUTCOME_MODES,
        default=BENCH_OUTCOMES,
        help="replay protocol: review = oracle-reviewer approval loop (default; labels act as a "
        "perfect human reviewer, optimistic); allow = spec §5.2 allow-all",
    )
    p.add_argument("--update-readme", action="store_true")
    p.add_argument(
        "--ablations",
        type=lambda s: [v for v in s.split(",") if v],
        default=None,
        help="registry variants for the ablation table, e.g. engine,engine-no-novelty",
    )
    a = p.parse_args(argv)
    kw: dict[str, Any] = {}
    if a.detector == "llm_judge":
        kw = _live_judge_kwargs(a.judge_model, a.judge_cache)
    out = run(
        a.traces,
        a.detector,
        SplitSpec(kind=a.split),
        a.out,
        n_boot=a.n_boot,
        ablations=a.ablations,
        sample=a.sample,
        outcomes=a.outcomes,
        **kw,
    )
    if a.update_readme:
        update_readme(Path("README.md"), (out / "report.md").read_text())
    print(out)


if __name__ == "__main__":
    main()
