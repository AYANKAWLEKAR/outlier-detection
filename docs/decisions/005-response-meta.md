# 005 — Score on response metadata in v1?

Spec §10 open decision 5. Status: decided (Task 1.14).

## Context

§2.3: the previous call's `response_meta.item_count` is scored on the *next* call (magnitude field
`__resp__.item_count`, ruling R-RESP). It adds proxy complexity (a second, small envelope update) and
only the `exfil_volume` family needs it. Task 1.14 adds `EngineConfig.mag_resp_signal` (default `True`)
to switch it off for the sweep.

## Options

1. Score response metadata (current).
2. Do not score it in v1.

## Evidence

`evidence/sweeps-seed0.md`, suite `resp` (warm test rows, live thresholds):

| setting | exfil_volume recall (share) | AUPRC | benign FPR | control FPR | min gate recall |
|---|---|---|---|---|---|
| on | 0.579 (0.73) n=19 | 0.550 | 0.0027 | 0.062 | 0.388 |
| off | 0.158 (0.00) n=19 | 0.547 | 0.0027 | 0.062 | 0.158 |

Every other gate cell is unchanged. With it off, exfil recall drops to what the outbound write's own
fields catch (3/19), and magnitude is never the top reason. It costs nothing measurable in benign or
control FPR. Ablation: removing magnitude entirely drops exfil recall to 0.158 as well, so all of
magnitude's exfil contribution is the response signal.

## Decision

**Score response metadata in v1** (keep `mag_resp_signal = True`). It is the only signal that catches
`exfil_volume`, at no measured FPR cost. Note that exfil recall (0.579) is itself far below the Tier-1
gate; see the Task 1.14 report.
