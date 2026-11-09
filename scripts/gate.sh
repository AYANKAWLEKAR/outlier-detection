#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

MARK='not slow and not live'
if [[ "${1:-}" == "--full" ]]; then
    MARK='not live'
fi

uv run ruff check .
uv run ruff format --check .
uv run mypy
# exit code 5 = no tests collected (e.g. an empty marker selection); not a failure
rc=0
uv run pytest -q -m "$MARK" || rc=$?
if [[ $rc -ne 0 && $rc -ne 5 ]]; then
    exit $rc
fi
