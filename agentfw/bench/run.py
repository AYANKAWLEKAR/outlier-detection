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

from agentfw.bench.dataset import load_dir
from agentfw.bench.metrics import BOOTSTRAP_N, compute_all, gate_preview
from agentfw.bench.registry import make_detector
from agentfw.bench.replay import SplitSpec, canonical_hash, replay, write_score_table
from agentfw.bench.report import RunMeta, build_report, git_sha, render_markdown, update_readme

__all__ = ["main", "run"]


def run(
    traces_dir: Path,
    detector: str,
    split: SplitSpec,
    out_dir: Path,
    *,
    n_boot: int = BOOTSTRAP_N,
    **detector_kw: Any,
) -> Path:
    """Write ``scores.parquet``, ``report.json``, ``report.md`` into ``out_dir``; returns it."""
    raws, labels, manifest = load_dir(traces_dir)
    det = make_detector(detector, **detector_kw)
    rows = replay(raws, labels, det, split)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_score_table(rows, out_dir / "scores.parquet")
    seed = manifest.config.seed
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
        {"state_bytes": det.state_bytes(), "score_hash": canonical_hash(rows), "n_boot": n_boot},
        gate=gate_preview(rows),
    )
    (out_dir / "report.json").write_bytes(orjson.dumps(report, option=orjson.OPT_INDENT_2))
    (out_dir / "report.md").write_text(render_markdown(report))
    return out_dir


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m agentfw.bench.run")
    p.add_argument("--traces", type=Path, required=True)
    p.add_argument("--detector", required=True)
    p.add_argument("--split", choices=["time", "agent"], default="time")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-boot", type=int, default=BOOTSTRAP_N)
    p.add_argument("--update-readme", action="store_true")
    a = p.parse_args(argv)
    out = run(a.traces, a.detector, SplitSpec(kind=a.split), a.out, n_boot=a.n_boot)
    if a.update_readme:
        update_readme(Path("README.md"), (out / "report.md").read_text())
    print(out)


if __name__ == "__main__":
    main()
