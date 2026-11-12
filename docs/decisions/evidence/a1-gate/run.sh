#!/usr/bin/env bash
# Task 1.23: Amendment A1's gate matrix on SW1 seeds 0-2 under both protocols.
#
#   docs/decisions/evidence/a1-gate/run.sh [seeds01|seed2|all]    (default: all)
#
# seeds01: generate traces/sw1-seed0, traces/sw1-seed1 (sw1_config(seed)) and traces/sw1-sens-seed0
#          (sw1_config(0) with magnitude_far_mult = 1.0), replay each (engine, protocol) once
#          (12 replays), write matrix.json / matrix.md.
# seed2:   only after seeds01 is written, with nothing changed in between: generate
#          traces/sw1-seed2, replay its four cells, rewrite matrix.json / matrix.md with all rows.
# At most 4 processes. Datasets go under the git-ignored traces/; per-cell results under
# results/a1-gate/ (git-ignored). The latency number is written separately:
#   measure.py latency P99_US|null [NOTE]   (tests/perf/test_latency.py on an idle machine; A1.8)
set -euo pipefail
cd "$(dirname "$0")/../../../.."

M=docs/decisions/evidence/a1-gate/measure.py
CELLS=results/a1-gate
STAGE="${1:-all}"
mkdir -p traces "$CELLS"

cells() {  # replay every (engine, protocol) of the given datasets, 4 at a time
    for d in "$@"; do
        for e in a1 pre-a1; do
            for p in review allow; do
                echo "traces/$d $e $p $CELLS/$d--$e--$p.json"
            done
        done
    done | xargs -P 4 -L 1 sh -c 'uv run python '"$M"' cell "$0" "$1" "$2" "$3"'
}

if [[ "$STAGE" == "seeds01" || "$STAGE" == "all" ]]; then
    uv run python "$M" gen 0 traces/sw1-seed0 & a=$!
    uv run python "$M" gen 1 traces/sw1-seed1 & b=$!
    uv run python "$M" gen 0 traces/sw1-sens-seed0 1.0 & c=$!
    wait "$a" && wait "$b" && wait "$c"  # each generation must succeed
    cells sw1-seed0 sw1-seed1 sw1-sens-seed0
    uv run python "$M" merge "$CELLS"
fi

if [[ "$STAGE" == "seed2" || "$STAGE" == "all" ]]; then
    test -f docs/decisions/evidence/a1-gate/matrix.json  # seeds 0 and 1 are written first
    uv run python "$M" gen 2 traces/sw1-seed2
    cells sw1-seed2
    uv run python "$M" merge "$CELLS"
fi
