"""Task 1.15 root-cause replays (evidence for docs/decisions/008-tier1-gate.md; not package code).

    uv run python docs/decisions/evidence/008-root-causes/replay_variants.py TRACES_DIR OUT_DIR

Writes three score tables of the default EngineConfig on TRACES_DIR:
  default.parquet   -- spec replay (every call ALLOWED, spec §5.2)
  oraclecal.parquet -- the calibrator never learns attack-labelled calls; detectors learn all
  oracle.parquet    -- replay(outcomes="oracle"): nothing learns attack-labelled calls
The two oracle variants read the labels: diagnostics only, never reported numbers.
"""

import sys
from pathlib import Path

from agentfw.bench.dataset import load_dir
from agentfw.bench.replay import SplitSpec, replay, write_score_table
from agentfw.core.config import EngineConfig
from agentfw.core.engine import Engine
from agentfw.core.envelope import ActionEnvelope, DetectorResult, Outcome


class OracleCal:
    """Engine whose calibrator skips attack-labelled calls (detectors still learn them)."""

    def __init__(self, attacks: set[str]) -> None:
        self.e = Engine(EngineConfig())
        self.attacks = attacks
        self._observe = self.e.calibrator.observe

    def reset(self) -> None:
        self.e.reset()

    def score(self, env: ActionEnvelope) -> DetectorResult:
        return self.e.score(env)

    def update(self, env: ActionEnvelope, outcome: Outcome) -> None:
        if env.envelope_id in self.attacks:
            self.e.calibrator.observe = lambda *a, **k: None  # type: ignore[method-assign]
        try:
            self.e.update(env, outcome)
        finally:
            self.e.calibrator.observe = self._observe  # type: ignore[method-assign]

    def state_bytes(self) -> int:
        return self.e.state_bytes()


def main() -> None:
    traces, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    raws, labels, _ = load_dir(traces)
    attacks = {e for e, lab in labels.items() if lab.label == "attack"}
    split = SplitSpec()
    write_score_table(replay(raws, labels, Engine(), split), out / "default.parquet")
    write_score_table(replay(raws, labels, OracleCal(attacks), split), out / "oraclecal.parquet")
    oracle = replay(raws, labels, Engine(), split, outcomes="oracle")
    write_score_table(oracle, out / "oracle.parquet")


if __name__ == "__main__":
    main()
