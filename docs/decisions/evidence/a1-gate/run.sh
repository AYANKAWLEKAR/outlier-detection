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

cell_files() {  # the cell file of every (engine, protocol) of the given datasets
    for d in "$@"; do
        for e in a1 pre-a1; do
            for p in review allow; do
                echo "$d $e $p $CELLS/$d--$e--$p.json"
            done
        done
    done
}

cells() {  # replay every (engine, protocol) of the given datasets, 4 at a time
    cell_files "$@" | xargs -P 4 -L 1 sh -c 'uv run python '"$M"' cell "traces/$0" "$1" "$2" "$3"'
}

require_cells() {  # every cell of the given datasets is written (a committed matrix.json is not)
    local d e p f
    while read -r d e p f; do
        [[ -f "$f" ]] || { echo "run.sh: $f is missing; run stage seeds01 first" >&2; return 1; }
    done < <(cell_files "$@")
}

if [[ "$STAGE" == "seeds01" || "$STAGE" == "all" ]]; then
    uv run python "$M" gen 0 traces/sw1-seed0 & a=$!
    uv run python "$M" gen 1 traces/sw1-seed1 & b=$!
    uv run python "$M" gen 0 traces/sw1-sens-seed0 1.0 & c=$!
    # Wait for all three, then stop if any failed. Explicit exit: errexit ignores a failure
    # inside an && list, and bash 3.2 (macOS /bin/bash) ignores a failing (( )) or [[ ]].
    failed=0
    for pid in "$a" "$b" "$c"; do
        wait "$pid" || failed=1
    done
    if [[ "$failed" != 0 ]]; then
        echo "run.sh: a dataset generation failed; nothing was replayed" >&2
        exit 1
    fi
    cells sw1-seed0 sw1-seed1 sw1-sens-seed0
    uv run python "$M" merge "$CELLS"
fi

if [[ "$STAGE" == "seed2" || "$STAGE" == "all" ]]; then
    require_cells sw1-seed0 sw1-seed1 sw1-sens-seed0  # seeds 0 and 1 are written first
    uv run python "$M" gen 2 traces/sw1-seed2
    cells sw1-seed2
    uv run python "$M" merge "$CELLS"
fi
