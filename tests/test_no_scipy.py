"""``agentfw.core`` never imports scipy, pandas or sklearn in the scoring path."""

from __future__ import annotations

import subprocess
import sys

_CHECK = (
    "import agentfw.core.engine, sys; "
    'sys.exit(int(any(m in sys.modules for m in ("scipy", "pandas", "sklearn"))))'
)


def test_engine_import_pulls_no_scipy_pandas_sklearn() -> None:
    proc = subprocess.run([sys.executable, "-c", _CHECK], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
