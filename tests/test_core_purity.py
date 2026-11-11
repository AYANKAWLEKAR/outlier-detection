"""Guard: response content never enters the scoring core.

Only ``ResponseMeta`` fields (item_count, bytes, error, ids, fp) may cross into
``agentfw/core/detectors/`` and ``agentfw/core/trajectory/``.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCANNED_DIRS = (ROOT / "agentfw" / "core" / "detectors", ROOT / "agentfw" / "core" / "trajectory")
FORBIDDEN = ("response_text", "CallToolResult", ".content", "TextContent")


def scan_for_response_content(paths: Iterable[Path]) -> list[str]:
    """Return one ``path:line: token`` hit per forbidden token occurrence."""
    hits: list[str] = []
    for path in paths:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for token in FORBIDDEN:
                if token in line:
                    hits.append(f"{path}:{lineno}: {token}")
    return hits


def _scoring_core_files() -> list[Path]:
    files: list[Path] = []
    for directory in SCANNED_DIRS:
        if directory.is_dir():
            files.extend(sorted(directory.glob("*.py")))
    return files


def test_no_response_content_in_scoring_core() -> None:
    files = _scoring_core_files()
    assert files, "expected at least one scoring-core module to scan"
    assert scan_for_response_content(files) == []


def test_guard_sees_a_violation(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text("x = response_text\n", encoding="utf-8")
    assert len(scan_for_response_content([bad])) == 1
