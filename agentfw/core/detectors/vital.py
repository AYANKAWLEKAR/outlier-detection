"""Vital signs (feature spec F2): loop, spin and stall over a session's recent calls.

These are the reference-free per-step signals of the L1 layer in arXiv:2610.12375, paraphrased
and rebuilt on fingerprints instead of embeddings: is the agent repeating a call, are its calls
returning nothing new, is the session producing no new artifacts. They need no per-agent state
and no configuration per tool; everything is read from the envelope (``call_fp``, ``recent``,
``flat_args``), never from response content. Every response-derived input (``RecentCall.error``,
``delta_i``, ``new_artifacts``) is a *completed* call's, from the ring: the current call's own
response does not exist yet when a proxy scores it, so it is never read (T-58).

Each call gets three scores, fields ``__loop__``, ``__spin__`` and ``__stall__``:

* **Loop.** The window is the last ``vital_w_loop`` ring entries (``vital_w_loop_polling`` when
  the tool's annotations declare it read-only and idempotent: polling is expected there). Exact
  ``call_fp`` matches in the window score 0.7 for the first and 1.0 from the second on (T-08).
  Retry exemption: when the most recent match errored, the call scores 0 while the run of
  errored matches (most recent first) is at most ``vital_r_max``. With no exact match, a call
  of the same tool whose arg-token Jaccard is >= ``vital_jaccard`` is a near-duplicate: 0.7,
  or 0 when that call errored (a corrected retry).
* **Spin.** Over the last ``vital_w_spin`` completed calls (0 while there are fewer): 1.0 when
  each has a fingerprinted response with ``delta_i < vital_spin_di`` and none errored; else 0.
  Like stall, it lags the call that completes the window by one call (T-58).
* **Stall.** Over the last ``vital_w_stall`` completed calls (0 while there are fewer): 1.0 when
  none created a ledger artifact and none is a write; else 0.5 when every one is a read with
  ``delta_i < vital_stall_di`` (busy re-reading, even if an id trickles in); else 0.

The engine never fuses these scores (T-07): they are reported in ``per_detector`` and reasons
only, so a loop is a session-level finding, never a single-call block. ``update`` is a no-op.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from agentfw.core.config import EngineConfig
from agentfw.core.envelope import ActionEnvelope, DetectorScore, RecentCall
from agentfw.core.fingerprint import arg_tokens
from agentfw.core.hints import ToolHint
from agentfw.core.schema_infer import TypeView

__all__ = ["LOOP", "NAME", "SPIN", "STALL", "VitalDetector"]

NAME = "vital"
LOOP, SPIN, STALL = "__loop__", "__spin__", "__stall__"
_DUP_FIRST = 0.7


def _jaccard(a: frozenset[int], b: frozenset[int]) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 0.0


class VitalDetector:
    name: str = NAME

    def __init__(
        self,
        cfg: EngineConfig,
        hints: Mapping[str, ToolHint],
        is_write: Callable[[str, str], bool],
    ) -> None:
        self._cfg = cfg
        self._hints = hints
        self._is_write = is_write

    def _polling(self, tool: str) -> bool:
        hint = self._hints.get(tool)
        ann = None if hint is None else hint.annotations
        return ann is not None and ann.read_only is True and ann.idempotent is True

    def _loop(self, e: ActionEnvelope) -> tuple[float, dict[str, object]]:
        cfg = self._cfg
        w = cfg.vital_w_loop_polling if self._polling(e.tool_id) else cfg.vital_w_loop
        window = e.recent[-w:]
        matches = [rc for rc in reversed(window) if rc.call_fp == e.call_fp]
        ev: dict[str, object] = {"window": w, "observed": len(matches), "exempt": False}
        if matches:
            run = 0
            for rc in matches:
                if not rc.error:
                    break
                run += 1
            if 0 < run <= cfg.vital_r_max:
                ev["exempt"] = True
                return 0.0, ev
            return (_DUP_FIRST if len(matches) == 1 else 1.0), ev
        toks = arg_tokens(e.flat_args) if e.arg_tokens is None else e.arg_tokens
        best: RecentCall | None = None
        best_j = 0.0
        for rc in reversed(window):
            if rc.tool_id != e.tool_id:
                continue
            j = _jaccard(toks, rc.arg_tokens)
            if j > best_j:
                best, best_j = rc, j
        ev["jaccard"] = best_j
        if best is None or best_j < cfg.vital_jaccard:
            return 0.0, ev
        if best.error:
            ev["exempt"] = True
            return 0.0, ev
        return _DUP_FIRST, ev

    def _spin(self, e: ActionEnvelope) -> tuple[float, dict[str, object]]:
        cfg = self._cfg
        w = cfg.vital_w_spin
        window = e.recent[-w:]
        ev: dict[str, object] = {"observed": window[-1].delta_i if window else None}
        if len(window) < w:
            return 0.0, ev
        for rc in window:
            if rc.error or rc.delta_i is None or rc.delta_i >= cfg.vital_spin_di:
                return 0.0, ev
        return 1.0, ev

    def _stall(self, e: ActionEnvelope) -> tuple[float, dict[str, object]]:
        cfg = self._cfg
        w = cfg.vital_w_stall
        window = e.recent[-w:]
        ev: dict[str, object] = {"window": w, "observed": len(window)}
        if len(window) < w:
            return 0.0, ev
        agent = e.agent_id
        reads = all(not self._is_write(agent, rc.tool_id) for rc in window)
        created = sum(rc.new_artifacts for rc in window)
        ev["new_artifacts"] = created
        if reads and created == 0:
            return 1.0, ev
        if reads and all(
            rc.delta_i is not None and rc.delta_i < cfg.vital_stall_di for rc in window
        ):
            return 0.5, ev
        return 0.0, ev

    def score(self, e: ActionEnvelope, types: TypeView) -> list[DetectorScore]:
        out: list[DetectorScore] = []
        for field, part in ((LOOP, self._loop), (SPIN, self._spin), (STALL, self._stall)):
            s, ev = part(e)
            out.append(DetectorScore(NAME, field, s, ev))
        return out

    def update(self, e: ActionEnvelope, types: TypeView) -> None:
        """No state: everything the vital signs need travels on the envelope."""
