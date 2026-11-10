"""Bench run CLI: replay a detector over a dataset and write scores + report.

``dataset_sha`` is the sha256 of ``traces.jsonl`` as recorded in the dataset manifest.
"""

from __future__ import annotations

import argparse
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
from agentfw.bench.replay import SplitSpec, canonical_hash, replay, write_score_table
from agentfw.bench.report import RunMeta, build_report, git_sha, render_markdown, update_readme
from agentfw.bench.sampling import sample_counts, stratified_sample

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
    **detector_kw: Any,
) -> Path:
    """Write ``scores.parquet``, ``report.json``, ``report.md`` into ``out_dir``; returns it.

    ``ablations``: registry variants to replay on the same dataset and split for the report's
    ablation table (``None``: no table). They use the default config, not ``detector_kw``.
    ``sample``: ``"stratified:N"`` scores only a stratified N-envelope test subsample (others get
    NaN and drop out of the metrics); the spec, actual counts and caveats go in the report.
    """
    raws, labels, manifest = load_dir(traces_dir)
    det = make_detector(detector, **detector_kw)
    seed = manifest.config.seed
    sample_ids: set[str] | None = None
    extras: dict[str, object] = {}
    notes: list[str] = []
    if sample is not None:
        n = _parse_sample(sample)
        sample_ids = stratified_sample(raws, labels, n=n, seed=seed, split=split)
        extras["sample"] = {"spec": sample, "seed": seed, **sample_counts(raws, labels, sample_ids)}
        notes.append(
            f"sampled: scores cover a stratified test subsample ({sample}); attack prevalence is "
            "far above the dataset's, so AUPRC is not comparable with a full replay."
        )
    if isinstance(det, BatchFit):
        extras["batch"] = True
        notes.append("batch: fit offline on the whole train split, not an online/causal learner.")
    if notes:
        extras["notes"] = notes
    rows = replay(raws, labels, det, split, sample_ids=sample_ids)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_score_table(rows, out_dir / "scores.parquet")
    all_ = compute_all(rows, warm_only=False, n_boot=n_boot, seed=seed)
    warm = compute_all(rows, warm_only=True, n_boot=n_boot, seed=seed)
    meta = RunMeta(
        git_sha=git_sha(),
        seed=seed,
        dataset_sha=manifest.files["traces.jsonl"],
        detector=detector,
        detector_config=dict(detector_kw),
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
        ablations=None if ablations is None else run_ablation(traces_dir, ablations, split),
    )
    (out_dir / "report.json").write_bytes(orjson.dumps(report, option=orjson.OPT_INDENT_2))
    (out_dir / "report.md").write_text(render_markdown(report))
    return out_dir


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
    p.add_argument("--update-readme", action="store_true")
    p.add_argument(
        "--ablations",
        type=lambda s: [v for v in s.split(",") if v],
        default=None,
        help="registry variants for the ablation table, e.g. engine,engine-no-novelty",
    )
    a = p.parse_args(argv)
    out = run(
        a.traces,
        a.detector,
        SplitSpec(kind=a.split),
        a.out,
        n_boot=a.n_boot,
        ablations=a.ablations,
        sample=a.sample,
    )
    if a.update_readme:
        update_readme(Path("README.md"), (out / "report.md").read_text())
    print(out)


if __name__ == "__main__":
    main()
