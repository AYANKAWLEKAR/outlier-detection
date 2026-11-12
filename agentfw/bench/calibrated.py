"""Recalibration guard (F10): constants tuned on the bench are stamped with a normalizer hash.

``calibrated.json`` records the constants calibrated against the current normalizer, the sha256
of the normalizer source files, and a history of (re)stamps. When a normalizer file changes, the
hash no longer matches and ``test_calibrated_constants_match_normalizer`` fails: either recalibrate,
or, if no constant depends on the change, ``python -m agentfw.bench.calibrated --restamp --reason
TEXT``. The recorded constants are informational; nothing reads them into the engine.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
from collections.abc import Sequence
from pathlib import Path

import orjson
from pydantic import BaseModel

import agentfw

__all__ = [
    "CALIBRATED_PATH",
    "NORMALIZER_FILES",
    "CalibratedConstants",
    "load_calibrated",
    "main",
    "normalizer_hash",
]

#: Repo-relative paths (rooted next to the ``agentfw`` package) whose bytes the hash covers.
NORMALIZER_FILES: tuple[str, ...] = (
    "agentfw/core/normalize.py",
    "agentfw/core/patterns.py",
    "agentfw/core/detectors/base.py",
    "agentfw/core/fingerprint.py",
    "agentfw/core/ledger.py",
)
CALIBRATED_PATH = Path(__file__).with_name("calibrated.json")
_ROOT = Path(agentfw.__file__).resolve().parent.parent


class CalibratedConstants(BaseModel):
    normalizer_hash: str
    constants: dict[str, float]
    history: list[str]


def normalizer_hash() -> str:
    """sha256 over the bytes of :data:`NORMALIZER_FILES`, in order."""
    h = hashlib.sha256()
    for rel in NORMALIZER_FILES:
        h.update((_ROOT / rel).read_bytes())
    return h.hexdigest()


def load_calibrated(path: Path = CALIBRATED_PATH) -> CalibratedConstants:
    return CalibratedConstants.model_validate_json(path.read_bytes())


def _write(c: CalibratedConstants, path: Path) -> None:
    path.write_bytes(
        orjson.dumps(c.model_dump(), option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS) + b"\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m agentfw.bench.calibrated")
    p.add_argument("--restamp", action="store_true", help="rewrite the hash; needs --reason")
    p.add_argument("--reason", help="why no constant needs recalibrating (required by --restamp)")
    p.add_argument("--path", type=Path, default=CALIBRATED_PATH)
    a = p.parse_args(argv)
    if not a.restamp:
        c = load_calibrated(a.path)
        ok = c.normalizer_hash == normalizer_hash()
        print("calibrated.json matches the normalizer" if ok else "STALE: normalizer changed")
        return 0 if ok else 1
    if not a.reason or not a.reason.strip():
        p.error("--restamp requires --reason TEXT")
    c = load_calibrated(a.path)
    c.normalizer_hash = normalizer_hash()
    c.history.append(f"{datetime.date.today().isoformat()} {a.reason.strip()}")
    _write(c, a.path)
    print(f"restamped {a.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
