"""Task 1.15 root-cause replays (evidence for docs/decisions/008-tier1-gate.md; not package code).

    uv run python docs/decisions/evidence/008-root-causes/replay_variants.py TRACES_DIR OUT_DIR \
        [OUTCOMES ['JSON EngineConfig params']]

Writes three score tables of one configuration (default: the spec engine, outcomes "allow"):
  primary.parquet   -- replay(outcomes=OUTCOMES) with EngineConfig().replace(**params)
  oraclecal.parquet -- the same, but the calibrator never learns attack-labelled calls
  oracle.parquet    -- the same config with replay(outcomes="oracle"): nothing learns attacks
The two oracle variants read every label: diagnostics only, never reported numbers.
"""

import json
import sys
from pathlib import Path

from agentfw.bench.dataset import load_dir
from agentfw.bench.replay import SplitSpec, replay, write_score_table
from agentfw.core.config import EngineConfig
from agentfw.core.engine import Engine
from agentfw.core.envelope import ActionEnvelope, DetectorResult, Outcome


class OracleCal:
    """Engine whose calibrator skips attack-labelled calls (detectors still learn them)."""

    def __init__(self, attacks: set[str], cfg: EngineConfig) -> None:
        self.e = Engine(cfg)
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
    outcomes = sys.argv[3] if len(sys.argv) > 3 else "allow"
    cfg = EngineConfig().replace(**(json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}))
    out.mkdir(parents=True, exist_ok=True)
    raws, labels, _ = load_dir(traces)
    attacks = {e for e, lab in labels.items() if lab.label == "attack"}
    split = SplitSpec()
    primary = replay(raws, labels, Engine(cfg), split, outcomes=outcomes)  # type: ignore[arg-type]
    write_score_table(primary, out / "primary.parquet")
    oraclecal = replay(raws, labels, OracleCal(attacks, cfg), split, outcomes=outcomes)  # type: ignore[arg-type]
    write_score_table(oraclecal, out / "oraclecal.parquet")
    oracle = replay(raws, labels, Engine(cfg), split, outcomes="oracle")
    write_score_table(oracle, out / "oracle.parquet")


if __name__ == "__main__":
    main()
