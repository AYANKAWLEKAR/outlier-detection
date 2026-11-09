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
uv run pytest -q -m "$MARK"
