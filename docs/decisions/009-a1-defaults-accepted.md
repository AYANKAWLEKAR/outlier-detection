# 009 — Amendment A1 defaults accepted; ablation backlog

Status: **decided by the user (2026-11-12).** The user's words: "For now accept default answers for each
decision but note it down as something to ablate on."

Spec A1.13 lists seven decisions that Amendment A1 left to the user, each with a default. Every default
below is now the user's accepted answer. Each one is also an ablation candidate: a later run should measure
what the alternative does before the answer is treated as settled. Nothing here changes code, a constant or
a reported number.

| A1.13 item | accepted default | what to ablate |
|---|---|---|
| 1. Which protocol's result closes Tier 1 | left open; both protocols' results are always shown side by side (A1.5), and a pass under the oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic) alone is not called the Tier-1 gate being met | none to run; revisit once the spec §5.2 allow-all result changes |
| 2. "Far" = 5 × the persona's benign p99 | `magnitude_far_mult = 5.0` | gate numbers at `magnitude_far_mult` 1, 2, 3, 5, 10 (the 1.0 point is the existing sensitivity dataset) |
| 3. Top-up rate and the 50-attack floor | `magnitude_topup_rate = 0.015`; fewer than 50 gated spikes fails the acceptance test | top-up rate 0, 0.01, 0.015, 0.03; how the magnitude cell's n and recall move |
| 4. Acceptance test green on `review`, `allow` a strict xfail | yes | none to run; the xfail flips when the spec §5.2 allow-all result passes |
| 5. ×10 on the widest SW1 field scores 0.575 | accepted | `tail_prior` and the log-scale spread term on the widest fields; recall of ×10 spikes by field width |
| 6. A1.3 lowers some BOOL-field scores, including a negative number or a fraction in a 0/1 field (about 0.01 instead of 1.0) | accepted (`mag_bool_numeric = True`) | `mag_bool_numeric` off; and a variant that keeps novelty's FR2 N1 score when magnitude's rule scores z = 0 |
| 7. Calibrators keep old-formula scores after an upgrade | kept; thresholds adapt over time | reset vs keep on upgrade; how long the per-agent thresholds take to settle |
