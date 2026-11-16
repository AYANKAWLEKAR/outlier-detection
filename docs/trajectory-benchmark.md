<!-- Numbers: one run of `python -m agentfw.bench.trajectory --traces traces/bench --out results/trajectory-final --seeds 0,1 --workers 4` on committed sha c7c123f (clean tree), 2026-11-16, 39 min wall; latency run last, load average 4.2 / 8.2 / 14.3 (1 / 5 / 15 min) at its end, so the ratios to `engine` are the comparable latency figures. Ruling R28 then reverted the T-59 ledger change (an echoed args-first id no longer becomes grounded). The rows that read ledger refs were re-measured on that code (`--rows engine,flow_ledger,provisional,seq_age_discount,gate,engine-traj,gate+flow_ledger,escalate+gate+flow_ledger,engine-traj-shadow --latency-calls 0`, both seeds, both protocols, with attribution; agentfw/ byte-identical to the R28 commit; the run meta says 9413905-dirty because the revert was not yet committed): those rows and the attribution tables below are from that run, every other row and the latency table are carried over from c7c123f, and the report was re-rendered from the merged JSON (no replay; hence "Re-rendered by 94139055db32-dirty", the rendering code is unchanged). Effect of the revert: on seed 0 one more benign gate prerequisite block in every gate row (gate, gate+flow_ledger 6 -> 7 blocks; escalate+gate+flow_ledger 426 -> 427 under spec 5.2 allow-all, 657 -> 658 optimistic; engine-traj 2,795 -> 2,796 and 2,862 -> 2,863; engine-traj-shadow 2,548 -> 2,549 and 2,588 -> 2,589) and engine-traj false halts at theta <= 0.30 up by two benign sessions (of 40,840); seed 1, all scores, the other rows and the attribution are identical. `results/` is not committed (ruling R24, T-57). Summary and recommendations: docs/decisions/010-trajectory-defaults.md. -->

# Trajectory benchmark

Numbers: git `c7c123fd77e1`, created 2026-11-16T21:19:49+00:00. Re-rendered (no replay) by git `94139055db32-dirty`. Split: time, calibration partition 0.15 of agents (F10).

| seed | traces sha | envelopes | attacks | families | bench config |
|---|---|---|---|---|---|
| 0 | 1d4c1873497b | 177460 | 855 | benign_unusual, exfil_volume, magnitude_spike, novel_target, prereq_skip, scope_blowup, slow_drift, velocity_burst, loop, spin, stall, splice | yes |
| 1 | 27e7e5c04f94 | 174995 | 829 | benign_unusual, exfil_volume, magnitude_spike, novel_target, prereq_skip, scope_blowup, slow_drift, velocity_burst, loop, spin, stall, splice | yes |

Rows (each switch toggled alone on top of `engine`, plus composites):

| row | detector | overrides | note |
|---|---|---|---|
| engine | engine | `{}` | reference: today's default engine |
| flow_ledger | engine | `{"flow_ledger":true}` |  |
| vital | engine-vital | `{}` | v1 + vital (enabled=KNOWN_DETECTORS) |
| br_score | engine | `{"br_mode":"score"}` |  |
| br_modulate | engine | `{"br_mode":"modulate"}` |  |
| sessions | engine | `{"sessions":true}` |  |
| provisional | engine | `{"provisional":true}` |  |
| seq_age_discount | engine | `{"seq_age_discount":true}` |  |
| gate | engine | `{"gate":true}` |  |
| escalate | engine | `{"escalate":true}` |  |
| session_enforce | engine | `{"session_enforce":true}` |  |
| authority | engine | `{"authority":true}` |  |
| engine-traj | engine-traj | `{}` | every switch on (registry.TRAJ_SWITCHES + vital) |
| gate+flow_ledger | engine | `{"gate":true,"flow_ledger":true}` | composite: the gate reads ledger history (T-41) |
| escalate+gate+flow_ledger | engine | `{"escalate":true,"gate":true,"flow_ledger":true}` | composite: escalation of gate escalations (F5 after F4) |
| sessions+vital | engine-vital | `{"sessions":true}` | composite: the F7 shadow halt configuration (vital flags recorded, nothing enforced) |
| engine-traj-shadow | engine-traj | `{"session_enforce":false}` | composite: engine-traj with session verdicts recorded, not enforced (exact halt metrics at every threshold, T-43) |

Δ method: paired percentile bootstrap over test-split sessions (row minus engine, same resampled sessions for both), 95% interval, n = 1000.

## Protocol: spec §5.2 allow-all

### spec §5.2 allow-all — seed 0

**Per-call (warm test rows) and decision metrics.** Δ = row − engine, paired session
bootstrap 95% CI; bold where the interval excludes 0.

| row | AUPRC [CI] | FPR@95 per-agent [CI] | prec@base (per-agent 95) | dec. benign FPR | Δ benign FPR | dec. control FPR | Δ control FPR | dec. prec@base | escalation rate | block / approval | scores / decisions = engine |
|---|---|---|---|---|---|---|---|---|---|---|---|
| engine | 0.659 [0.541, 0.756] | 0.578 [0.539, 0.660] | 0.032 | 0.0106 | – | 0.0265 | – | 0.586 | 0.0000 | 510 / 575 | yes / yes |
| flow_ledger | 0.658 [0.535, 0.755] | 0.601 [0.539, 0.665] | 0.031 | 0.0106 | +0.0000 [0.0000, 0.0000] | 0.0265 | +0.0000 [0.0000, 0.0000] | 0.586 | 0.0000 | 509 / 575 | no / no |
| vital | 0.659 [0.541, 0.756] | 0.578 [0.539, 0.660] | 0.032 | 0.0106 | +0.0000 [0.0000, 0.0000] | 0.0265 | +0.0000 [0.0000, 0.0000] | 0.586 | 0.0000 | 510 / 575 | yes / yes |
| br_score | 0.659 [0.541, 0.756] | 0.578 [0.539, 0.660] | 0.032 | 0.0106 | +0.0000 [0.0000, 0.0000] | 0.0265 | +0.0000 [0.0000, 0.0000] | 0.586 | 0.0000 | 510 / 575 | yes / yes |
| br_modulate | 0.659 [0.541, 0.756] | 0.578 [0.539, 0.660] | 0.032 | 0.1508 | **+0.1403 [0.1365, 0.1438]** | 0.2640 | **+0.2375 [0.2042, 0.2703]** | 0.103 | 0.0000 | 4350 / 2781 | yes / no |
| sessions | 0.659 [0.541, 0.756] | 0.578 [0.539, 0.660] | 0.032 | 0.0106 | +0.0000 [0.0000, 0.0000] | 0.0265 | +0.0000 [0.0000, 0.0000] | 0.586 | 0.0000 | 510 / 575 | yes / yes |
| provisional | 0.659 [0.541, 0.756] | 0.578 [0.539, 0.660] | 0.032 | 0.0095 | **-0.0011 [-0.0014, -0.0008]** | 0.0265 | +0.0000 [0.0000, 0.0000] | 0.597 | 0.0000 | 426 / 575 | yes / no |
| seq_age_discount | 0.659 [0.541, 0.756] | 0.578 [0.539, 0.660] | 0.032 | 0.0106 | +0.0000 [0.0000, 0.0000] | 0.0265 | +0.0000 [0.0000, 0.0000] | 0.586 | 0.0000 | 510 / 575 | no / no |
| gate | 0.659 [0.541, 0.756] | 0.578 [0.539, 0.660] | 0.032 | 0.0106 | +0.0000 [0.0000, 0.0001] | 0.0265 | +0.0000 [0.0000, 0.0000] | 0.586 | 0.0000 | 7 / 1079 | yes / no |
| escalate | 0.659 [0.541, 0.756] | 0.578 [0.539, 0.660] | 0.032 | 0.0106 | +0.0000 [0.0000, 0.0000] | 0.0265 | +0.0000 [0.0000, 0.0000] | 0.586 | 0.0067 | 510 / 575 | yes / yes |
| session_enforce | 0.659 [0.541, 0.756] | 0.578 [0.539, 0.660] | 0.032 | 0.0133 | **+0.0027 [0.0019, 0.0037]** | 0.0339 | +0.0074 [0.0000, 0.0176] | 0.530 | 0.0000 | 799 / 405 | yes / no |
| authority | 0.659 [0.541, 0.756] | 0.578 [0.539, 0.660] | 0.032 | 0.0106 | +0.0000 [0.0000, 0.0000] | 0.0265 | +0.0000 [0.0000, 0.0000] | 0.586 | 0.0000 | 485 / 600 | yes / no |
| engine-traj | 0.658 [0.535, 0.755] | 0.601 [0.539, 0.665] | 0.031 | 0.1187 | **+0.1081 [0.1045, 0.1117]** | 0.2699 | **+0.2434 [0.2076, 0.2795]** | 0.123 | 0.0653 | 2796 / 2935 | no / no |
| gate+flow_ledger | 0.658 [0.535, 0.755] | 0.601 [0.539, 0.665] | 0.031 | 0.0106 | +0.0000 [0.0000, 0.0001] | 0.0265 | +0.0000 [0.0000, 0.0000] | 0.585 | 0.0000 | 7 / 1078 | no / no |
| escalate+gate+flow_ledger | 0.658 [0.535, 0.755] | 0.601 [0.539, 0.665] | 0.031 | 0.0106 | +0.0000 [0.0000, 0.0001] | 0.0265 | +0.0000 [0.0000, 0.0000] | 0.585 | 0.0066 | 427 / 658 | no / no |
| sessions+vital | 0.659 [0.541, 0.756] | 0.578 [0.539, 0.660] | 0.032 | 0.0106 | +0.0000 [0.0000, 0.0000] | 0.0265 | +0.0000 [0.0000, 0.0000] | 0.586 | 0.0000 | 510 / 575 | yes / yes |
| engine-traj-shadow | 0.658 [0.535, 0.755] | 0.601 [0.539, 0.665] | 0.031 | 0.1147 | **+0.1041 [0.1005, 0.1075]** | 0.2566 | **+0.2301 [0.1967, 0.2635]** | 0.119 | 0.0653 | 2549 / 2965 | no / no |

**Attack-level decision recall by family** (Δ vs engine with CI below each value).

| row | exfil_volume (N=10) | magnitude_spike (N=50) | novel_target (N=22) | prereq_skip (N=3) | scope_blowup (N=11) | slow_drift (N=9) | splice (N=26) | velocity_burst (N=20) |
|---|---|---|---|---|---|---|---|---|
| engine | 0.900 | 1.000 | 1.000 | 1.000 | 1.000 | 0.333 | 0.115 | 1.000 |
| flow_ledger | 0.900 | 1.000 | 1.000 | 1.000 | 1.000 | 0.333 | 0.115 | 1.000 |
| vital | 0.900 | 1.000 | 1.000 | 1.000 | 1.000 | 0.333 | 0.115 | 1.000 |
| br_score | 0.900 | 1.000 | 1.000 | 1.000 | 1.000 | 0.333 | 0.115 | 1.000 |
| br_modulate | 1.000<br>+0.100 [0.000, 0.333] | 1.000 | 1.000 | 1.000 | 1.000 | 1.000<br>**+0.667 [0.333, 1.000]** | 0.500<br>**+0.385 [0.205, 0.586]** | 1.000 |
| sessions | 0.900 | 1.000 | 1.000 | 1.000 | 1.000 | 0.333 | 0.115 | 1.000 |
| provisional | 0.800<br>-0.100 [-0.333, 0.000] | 0.700<br>**-0.300 [-0.431, -0.173]** | 1.000 | 0.667<br>-0.333 [-1.000, 0.000] | 1.000 | 0.333 | 0.077<br>-0.038 [-0.136, 0.000] | 0.950<br>-0.050 [-0.154, 0.000] |
| seq_age_discount | 0.900 | 1.000 | 1.000 | 1.000 | 1.000 | 0.333 | 0.115 | 1.000 |
| gate | 0.900 | 1.000 | 1.000 | 1.000 | 1.000 | 0.333 | 0.115 | 1.000 |
| escalate | 0.900 | 1.000 | 1.000 | 1.000 | 1.000 | 0.333 | 0.115 | 1.000 |
| session_enforce | 0.900 | 1.000 | 1.000 | 1.000 | 1.000 | 0.333 | 0.115 | 1.000 |
| authority | 0.900 | 1.000 | 1.000 | 1.000 | 1.000 | 0.333 | 0.115 | 1.000 |
| engine-traj | 0.800<br>-0.100 [-0.444, 0.250] | 0.660<br>**-0.340 [-0.473, -0.212]** | 1.000 | 0.667<br>-0.333 [-1.000, 0.000] | 1.000 | 1.000<br>**+0.667 [0.333, 1.000]** | 0.346<br>**+0.231 [0.077, 0.400]** | 1.000 |
| gate+flow_ledger | 0.900 | 1.000 | 1.000 | 1.000 | 1.000 | 0.333 | 0.115 | 1.000 |
| escalate+gate+flow_ledger | 0.900 | 1.000 | 1.000 | 1.000 | 1.000 | 0.333 | 0.115 | 1.000 |
| sessions+vital | 0.900 | 1.000 | 1.000 | 1.000 | 1.000 | 0.333 | 0.115 | 1.000 |
| engine-traj-shadow | 0.800<br>-0.100 [-0.444, 0.250] | 0.660<br>**-0.340 [-0.473, -0.212]** | 1.000 | 0.667<br>-0.333 [-1.000, 0.000] | 1.000 | 1.000<br>**+0.667 [0.333, 1.000]** | 0.346<br>**+0.231 [0.077, 0.400]** | 0.900<br>-0.100 [-0.238, 0.000] |

**Session halts** (density policy at θ_halt = 0.6 vs the first-flag baseline;
sessions of all splits; precision at a 1.5% session base rate).

| row | false-halt | runaway recall | attacked recall | median calls onset→halt | prec@base | first-flag false-halt | first-flag runaway | max sensitivity step | selected θ_warn / θ_halt (calibration; – = undefined) |
|---|---|---|---|---|---|---|---|---|---|
| sessions | 0.0029 | 0.819 | 0.071 | 7.0 | 0.502 | 0.0377 | 0.935 | 0.0047 | – / 0.200 |
| provisional | 0.0029 | 0.815 | 0.071 | 7.0 | 0.499 | 0.0350 | 0.918 | 0.0043 | – / 0.200 |
| session_enforce | 0.0029 | 0.819 | 0.071 | 7.0 | 0.502 | 0.0377 | 0.935 | 0.0047 | – / 0.200 |
| engine-traj † | 0.0040 | 0.948 | 0.074 | 7.0 | 0.449 | 0.2178 | 1.000 | 0.0168 | – / 0.200 |
| sessions+vital | 0.0030 | 0.944 | 0.071 | 7.0 | 0.520 | 0.0417 | 1.000 | 0.0050 | – / 0.200 |
| engine-traj-shadow | 0.0040 | 0.948 | 0.074 | 7.0 | 0.449 | 0.2178 | 1.000 | 0.0168 | – / 0.200 |

† enforcing replay with `provisional`: the sensitivity steps above θ = 0.6
are approximate (T-43); the shadow rows are exact at every θ.

**Sensitivity curve, engine-traj-shadow** (false-halt / runaway recall /
precision at base rate per θ; calibration-partition benign share).

| θ | false-halt | runaway recall | attacked recall | prec@base | calibration benign share |
|---|---|---|---|---|---|
| 0.20 | 0.0285 | 0.996 | 0.135 | 0.127 | 0.0072 |
| 0.25 | 0.0271 | 0.996 | 0.129 | 0.131 | 0.0072 |
| 0.30 | 0.0271 | 0.996 | 0.129 | 0.131 | 0.0072 |
| 0.35 | 0.0103 | 0.987 | 0.082 | 0.252 | 0.0011 |
| 0.40 | 0.0082 | 0.974 | 0.081 | 0.295 | 0.0010 |
| 0.45 | 0.0082 | 0.974 | 0.081 | 0.295 | 0.0010 |
| 0.50 | 0.0082 | 0.974 | 0.081 | 0.295 | 0.0010 |
| 0.55 | 0.0040 | 0.948 | 0.074 | 0.449 | 0.0003 |
| 0.60 | 0.0040 | 0.948 | 0.074 | 0.449 | 0.0003 |
| 0.65 | 0.0040 | 0.948 | 0.074 | 0.449 | 0.0003 |
| 0.70 | 0.0022 | 0.897 | 0.068 | 0.581 | 0.0000 |
| 0.75 | 0.0020 | 0.888 | 0.068 | 0.605 | 0.0000 |
| 0.80 | 0.0020 | 0.888 | 0.068 | 0.605 | 0.0000 |

**Sessions, splice, vital** (splice reported separately, T-40).

| row | session AUROC (all) | session AUROC (length-matched) | pairs | runaway session AUROC | splice recall, per-agent 95 thr (warm) | splice decision recall | vital false-flag (benign sessions) |
|---|---|---|---|---|---|---|---|
| engine | 0.760 | 0.746 | 1141 | 0.978 | 0.962 (25/26) | 0.115 | – |
| flow_ledger | 0.760 | 0.745 | 1141 | 0.979 | 0.962 (25/26) | 0.115 | – |
| vital | 0.760 | 0.746 | 1141 | 0.978 | 0.962 (25/26) | 0.115 | 0.0385 |
| br_score | 0.760 | 0.746 | 1141 | 0.978 | 0.962 (25/26) | 0.115 | – |
| br_modulate | 0.760 | 0.746 | 1141 | 0.978 | 0.962 (25/26) | 0.500 | – |
| sessions | 0.760 | 0.746 | 1141 | 0.978 | 0.962 (25/26) | 0.115 | – |
| provisional | 0.760 | 0.746 | 1141 | 0.978 | 0.962 (25/26) | 0.077 | – |
| seq_age_discount | 0.760 | 0.746 | 1141 | 0.978 | 0.962 (25/26) | 0.115 | – |
| gate | 0.760 | 0.746 | 1141 | 0.978 | 0.962 (25/26) | 0.115 | – |
| escalate | 0.760 | 0.746 | 1141 | 0.978 | 0.962 (25/26) | 0.115 | – |
| session_enforce | 0.760 | 0.746 | 1141 | 0.978 | 0.962 (25/26) | 0.115 | – |
| authority | 0.760 | 0.746 | 1141 | 0.978 | 0.962 (25/26) | 0.115 | – |
| engine-traj | 0.760 | 0.746 | 1141 | 0.979 | 0.962 (25/26) | 0.346 | 0.0385 |
| gate+flow_ledger | 0.760 | 0.745 | 1141 | 0.979 | 0.962 (25/26) | 0.115 | – |
| escalate+gate+flow_ledger | 0.760 | 0.745 | 1141 | 0.979 | 0.962 (25/26) | 0.115 | – |
| sessions+vital | 0.760 | 0.746 | 1141 | 0.978 | 0.962 (25/26) | 0.115 | 0.0385 |
| engine-traj-shadow | 0.760 | 0.746 | 1141 | 0.979 | 0.962 (25/26) | 0.346 | 0.0385 |

**Attribution** (engine-traj (dataset schemas as hints); attacks carried per group, leave-one-out).

| family | magnitude | novelty | velocity | sequence | vital | gate | provisional | redundant |
|---|---|---|---|---|---|---|---|---|
| exfil_volume | 7 | 0 | 0 | 7 | 0 | 0 | 0 | 0 |
| magnitude_spike | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 32 |
| novel_target | 0 | 20 | 0 | 10 | 0 | 0 | 0 | 1 |
| prereq_skip | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| scope_blowup | 8 | 0 | 0 | 11 | 0 | 0 | 0 | 0 |
| slow_drift | 0 | 3 | 0 | 4 | 0 | 0 | 0 | 2 |
| splice | 0 | 1 | 1 | 7 | 0 | 0 | 0 | 2 |
| velocity_burst | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 20 |

Detected with every group: 114. T-20 totals hold: yes (by construction of the table). Removing a group changes exactly its rows (F10, T-52): no.

| group | carried | gained when removed | detected without |
|---|---|---|---|
| magnitude | 16 | 2 | 100 |
| novelty | 24 | 0 | 90 |
| velocity | 1 | 1 | 114 |
| sequence | 39 | 1 | 76 |
| vital | 0 | 0 | 114 |
| gate | 0 | 0 | 114 |
| provisional | 0 | 24 | 138 |

### spec §5.2 allow-all — seed 1

**Per-call (warm test rows) and decision metrics.** Δ = row − engine, paired session
bootstrap 95% CI; bold where the interval excludes 0.

| row | AUPRC [CI] | FPR@95 per-agent [CI] | prec@base (per-agent 95) | dec. benign FPR | Δ benign FPR | dec. control FPR | Δ control FPR | dec. prec@base | escalation rate | block / approval | scores / decisions = engine |
|---|---|---|---|---|---|---|---|---|---|---|---|
| engine | 0.652 [0.478, 0.736] | 0.682 [0.573, 0.758] | 0.029 | 0.0125 | – | 0.0163 | – | 0.560 | 0.0000 | 631 / 543 | yes / yes |
| flow_ledger | 0.651 [0.478, 0.736] | 0.583 [0.496, 0.761] | 0.033 | 0.0125 | +0.0000 [0.0000, 0.0000] | 0.0163 | +0.0000 [0.0000, 0.0000] | 0.560 | 0.0000 | 630 / 543 | no / no |
| vital | 0.652 [0.478, 0.736] | 0.682 [0.573, 0.758] | 0.029 | 0.0125 | +0.0000 [0.0000, 0.0000] | 0.0163 | +0.0000 [0.0000, 0.0000] | 0.560 | 0.0000 | 631 / 543 | yes / yes |
| br_score | 0.652 [0.478, 0.736] | 0.682 [0.573, 0.758] | 0.029 | 0.0125 | +0.0000 [0.0000, 0.0000] | 0.0163 | +0.0000 [0.0000, 0.0000] | 0.560 | 0.0000 | 631 / 543 | yes / yes |
| br_modulate | 0.652 [0.478, 0.736] | 0.682 [0.573, 0.758] | 0.029 | 0.1568 | **+0.1443 [0.1405, 0.1487]** | 0.1993 | **+0.1830 [0.1380, 0.2203]** | 0.102 | 0.0000 | 4576 / 2542 | yes / no |
| sessions | 0.652 [0.478, 0.736] | 0.682 [0.573, 0.758] | 0.029 | 0.0125 | +0.0000 [0.0000, 0.0000] | 0.0163 | +0.0000 [0.0000, 0.0000] | 0.560 | 0.0000 | 631 / 543 | yes / yes |
| provisional | 0.652 [0.478, 0.736] | 0.682 [0.573, 0.758] | 0.029 | 0.0105 | **-0.0020 [-0.0026, -0.0014]** | 0.0163 | +0.0000 [0.0000, 0.0000] | 0.560 | 0.0000 | 446 / 543 | yes / no |
| seq_age_discount | 0.652 [0.478, 0.736] | 0.682 [0.573, 0.758] | 0.029 | 0.0125 | +0.0000 [0.0000, 0.0000] | 0.0163 | +0.0000 [0.0000, 0.0000] | 0.560 | 0.0000 | 631 / 543 | no / no |
| gate | 0.652 [0.478, 0.736] | 0.682 [0.573, 0.758] | 0.029 | 0.0125 | +0.0000 [0.0000, 0.0000] | 0.0163 | +0.0000 [0.0000, 0.0000] | 0.560 | 0.0000 | 13 / 1161 | yes / no |
| escalate | 0.652 [0.478, 0.736] | 0.682 [0.573, 0.758] | 0.029 | 0.0125 | +0.0000 [0.0000, 0.0000] | 0.0163 | +0.0000 [0.0000, 0.0000] | 0.560 | 0.0073 | 631 / 543 | yes / yes |
| session_enforce | 0.652 [0.478, 0.736] | 0.682 [0.573, 0.758] | 0.029 | 0.0166 | **+0.0041 [0.0029, 0.0055]** | 0.0163 | +0.0000 [0.0000, 0.0000] | 0.491 | 0.0000 | 934 / 410 | yes / no |
| authority | 0.652 [0.478, 0.736] | 0.682 [0.573, 0.758] | 0.029 | 0.0125 | +0.0000 [0.0000, 0.0000] | 0.0163 | +0.0000 [0.0000, 0.0000] | 0.560 | 0.0000 | 588 / 585 | yes / no |
| engine-traj | 0.651 [0.478, 0.736] | 0.583 [0.496, 0.761] | 0.033 | 0.1186 | **+0.1061 [0.1025, 0.1102]** | 0.1961 | **+0.1797 [0.1335, 0.2161]** | 0.125 | 0.0720 | 2792 / 2733 | no / no |
| gate+flow_ledger | 0.651 [0.478, 0.736] | 0.583 [0.496, 0.761] | 0.033 | 0.0125 | +0.0000 [0.0000, 0.0000] | 0.0163 | +0.0000 [0.0000, 0.0000] | 0.560 | 0.0000 | 13 / 1160 | no / no |
| escalate+gate+flow_ledger | 0.651 [0.478, 0.736] | 0.583 [0.496, 0.761] | 0.033 | 0.0125 | +0.0000 [0.0000, 0.0000] | 0.0163 | +0.0000 [0.0000, 0.0000] | 0.560 | 0.0071 | 449 / 724 | no / no |
| sessions+vital | 0.652 [0.478, 0.736] | 0.682 [0.573, 0.758] | 0.029 | 0.0125 | +0.0000 [0.0000, 0.0000] | 0.0163 | +0.0000 [0.0000, 0.0000] | 0.560 | 0.0000 | 631 / 543 | yes / yes |
| engine-traj-shadow | 0.651 [0.478, 0.736] | 0.583 [0.496, 0.761] | 0.033 | 0.1150 | **+0.1025 [0.0989, 0.1062]** | 0.1961 | **+0.1797 [0.1335, 0.2161]** | 0.114 | 0.0720 | 2514 / 2774 | no / no |

**Attack-level decision recall by family** (Δ vs engine with CI below each value).

| row | exfil_volume (N=7) | magnitude_spike (N=55) | novel_target (N=19) | prereq_skip (N=5) | scope_blowup (N=5) | slow_drift (N=4) | splice (N=23) | velocity_burst (N=20) |
|---|---|---|---|---|---|---|---|---|
| engine | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.750 | 0.522 | 1.000 |
| flow_ledger | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.750 | 0.522 | 1.000 |
| vital | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.750 | 0.522 | 1.000 |
| br_score | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.750 | 0.522 | 1.000 |
| br_modulate | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000<br>+0.250 [0.000, 1.000] | 0.652<br>+0.130 [0.000, 0.292] | 1.000 |
| sessions | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.750 | 0.522 | 1.000 |
| provisional | 0.857<br>-0.143 [-0.500, 0.000] | 0.636<br>**-0.364 [-0.500, -0.226]** | 1.000 | 1.000 | 1.000 | 0.500<br>-0.250 [-1.000, 0.000] | 0.522 | 0.950<br>-0.050 [-0.160, 0.000] |
| seq_age_discount | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.750 | 0.522 | 1.000 |
| gate | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.750 | 0.522 | 1.000 |
| escalate | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.750 | 0.522 | 1.000 |
| session_enforce | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.750 | 0.522 | 1.000 |
| authority | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.750 | 0.522 | 1.000 |
| engine-traj | 0.857<br>-0.143 [-0.500, 0.000] | 0.545<br>**-0.455 [-0.600, -0.323]** | 0.947<br>-0.053 [-0.177, 0.000] | 1.000 | 1.000 | 1.000<br>+0.250 [0.000, 1.000] | 0.565<br>+0.043 [-0.105, 0.195] | 1.000 |
| gate+flow_ledger | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.750 | 0.522 | 1.000 |
| escalate+gate+flow_ledger | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.750 | 0.522 | 1.000 |
| sessions+vital | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.750 | 0.522 | 1.000 |
| engine-traj-shadow | 0.857<br>-0.143 [-0.500, 0.000] | 0.545<br>**-0.455 [-0.600, -0.323]** | 0.947<br>-0.053 [-0.177, 0.000] | 1.000 | 1.000 | 1.000<br>+0.250 [0.000, 1.000] | 0.565<br>+0.043 [-0.105, 0.195] | 0.800<br>**-0.200 [-0.409, -0.043]** |

**Session halts** (density policy at θ_halt = 0.6 vs the first-flag baseline;
sessions of all splits; precision at a 1.5% session base rate).

| row | false-halt | runaway recall | attacked recall | median calls onset→halt | prec@base | first-flag false-halt | first-flag runaway | max sensitivity step | selected θ_warn / θ_halt (calibration; – = undefined) |
|---|---|---|---|---|---|---|---|---|---|
| sessions | 0.0029 | 0.794 | 0.084 | 7.0 | 0.539 | 0.0366 | 0.942 | 0.0051 | – / 0.200 |
| provisional | 0.0030 | 0.786 | 0.084 | 7.0 | 0.529 | 0.0338 | 0.930 | 0.0049 | – / 0.200 |
| session_enforce | 0.0029 | 0.794 | 0.084 | 7.0 | 0.539 | 0.0366 | 0.942 | 0.0051 | – / 0.200 |
| engine-traj † | 0.0037 | 0.877 | 0.090 | 7.0 | 0.502 | 0.2031 | 1.000 | 0.0186 | – / 0.200 |
| sessions+vital | 0.0030 | 0.877 | 0.084 | 7.0 | 0.554 | 0.0399 | 1.000 | 0.0055 | – / 0.200 |
| engine-traj-shadow | 0.0037 | 0.877 | 0.090 | 7.0 | 0.502 | 0.2031 | 1.000 | 0.0186 | – / 0.200 |

† enforcing replay with `provisional`: the sensitivity steps above θ = 0.6
are approximate (T-43); the shadow rows are exact at every θ.

**Sensitivity curve, engine-traj-shadow** (false-halt / runaway recall /
precision at base rate per θ; calibration-partition benign share).

| θ | false-halt | runaway recall | attacked recall | prec@base | calibration benign share |
|---|---|---|---|---|---|
| 0.20 | 0.0315 | 1.000 | 0.184 | 0.143 | 0.0052 |
| 0.25 | 0.0296 | 1.000 | 0.183 | 0.151 | 0.0052 |
| 0.30 | 0.0296 | 1.000 | 0.183 | 0.151 | 0.0052 |
| 0.35 | 0.0111 | 0.975 | 0.108 | 0.279 | 0.0009 |
| 0.40 | 0.0087 | 0.971 | 0.104 | 0.326 | 0.0009 |
| 0.45 | 0.0087 | 0.971 | 0.104 | 0.326 | 0.0009 |
| 0.50 | 0.0087 | 0.971 | 0.104 | 0.326 | 0.0009 |
| 0.55 | 0.0038 | 0.889 | 0.090 | 0.501 | 0.0002 |
| 0.60 | 0.0037 | 0.877 | 0.090 | 0.502 | 0.0002 |
| 0.65 | 0.0037 | 0.877 | 0.090 | 0.502 | 0.0002 |
| 0.70 | 0.0023 | 0.848 | 0.084 | 0.615 | 0.0000 |
| 0.75 | 0.0022 | 0.848 | 0.084 | 0.620 | 0.0000 |
| 0.80 | 0.0022 | 0.848 | 0.084 | 0.620 | 0.0000 |

**Sessions, splice, vital** (splice reported separately, T-40).

| row | session AUROC (all) | session AUROC (length-matched) | pairs | runaway session AUROC | splice recall, per-agent 95 thr (warm) | splice decision recall | vital false-flag (benign sessions) |
|---|---|---|---|---|---|---|---|
| engine | 0.815 | 0.785 | 906 | 0.971 | 0.957 (22/23) | 0.522 | – |
| flow_ledger | 0.815 | 0.786 | 906 | 0.972 | 0.957 (22/23) | 0.522 | – |
| vital | 0.815 | 0.785 | 906 | 0.971 | 0.957 (22/23) | 0.522 | 0.0363 |
| br_score | 0.815 | 0.785 | 906 | 0.971 | 0.957 (22/23) | 0.522 | – |
| br_modulate | 0.815 | 0.785 | 906 | 0.971 | 0.957 (22/23) | 0.652 | – |
| sessions | 0.815 | 0.785 | 906 | 0.971 | 0.957 (22/23) | 0.522 | – |
| provisional | 0.815 | 0.785 | 906 | 0.971 | 0.957 (22/23) | 0.522 | – |
| seq_age_discount | 0.815 | 0.785 | 906 | 0.971 | 0.957 (22/23) | 0.522 | – |
| gate | 0.815 | 0.785 | 906 | 0.971 | 0.957 (22/23) | 0.522 | – |
| escalate | 0.815 | 0.785 | 906 | 0.971 | 0.957 (22/23) | 0.522 | – |
| session_enforce | 0.815 | 0.785 | 906 | 0.971 | 0.957 (22/23) | 0.522 | – |
| authority | 0.815 | 0.785 | 906 | 0.971 | 0.957 (22/23) | 0.522 | – |
| engine-traj | 0.815 | 0.786 | 906 | 0.972 | 0.957 (22/23) | 0.565 | 0.0363 |
| gate+flow_ledger | 0.815 | 0.786 | 906 | 0.972 | 0.957 (22/23) | 0.522 | – |
| escalate+gate+flow_ledger | 0.815 | 0.786 | 906 | 0.972 | 0.957 (22/23) | 0.522 | – |
| sessions+vital | 0.815 | 0.785 | 906 | 0.971 | 0.957 (22/23) | 0.522 | 0.0363 |
| engine-traj-shadow | 0.815 | 0.786 | 906 | 0.972 | 0.957 (22/23) | 0.565 | 0.0363 |

**Attribution** (engine-traj (dataset schemas as hints); attacks carried per group, leave-one-out).

| family | magnitude | novelty | velocity | sequence | vital | gate | provisional | redundant |
|---|---|---|---|---|---|---|---|---|
| exfil_volume | 4 | 0 | 0 | 4 | 0 | 0 | 0 | 2 |
| magnitude_spike | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 29 |
| novel_target | 0 | 17 | 0 | 13 | 0 | 0 | 0 | 0 |
| prereq_skip | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 |
| scope_blowup | 3 | 0 | 0 | 5 | 0 | 0 | 0 | 0 |
| slow_drift | 0 | 2 | 0 | 2 | 0 | 0 | 0 | 2 |
| splice | 0 | 1 | 0 | 9 | 0 | 0 | 0 | 3 |
| velocity_burst | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 20 |

Detected with every group: 101. T-20 totals hold: yes (by construction of the table). Removing a group changes exactly its rows (F10, T-52): no.

| group | carried | gained when removed | detected without |
|---|---|---|---|
| magnitude | 8 | 4 | 97 |
| novelty | 20 | 0 | 81 |
| velocity | 0 | 5 | 106 |
| sequence | 33 | 0 | 68 |
| vital | 0 | 0 | 101 |
| gate | 0 | 0 | 101 |
| provisional | 0 | 29 | 130 |

## Protocol: oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic) — OPTIMISTIC: labels act as a perfect reviewer

### oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic) — seed 0

**Per-call (warm test rows) and decision metrics.** Δ = row − engine, paired session
bootstrap 95% CI; bold where the interval excludes 0.

| row | AUPRC [CI] | FPR@95 per-agent [CI] | prec@base (per-agent 95) | dec. benign FPR | Δ benign FPR | dec. control FPR | Δ control FPR | dec. prec@base | escalation rate | block / approval | scores / decisions = engine |
|---|---|---|---|---|---|---|---|---|---|---|---|
| engine | 0.678 [0.563, 0.764] | 0.572 [0.517, 0.632] | 0.033 | 0.0396 | – | 0.0737 | – | 0.287 | 0.0000 | 889 / 1472 | yes / yes |
| flow_ledger | 0.678 [0.563, 0.764] | 0.568 [0.511, 0.648] | 0.033 | 0.0397 | **+0.0001 [0.0000, 0.0003]** | 0.0737 | +0.0000 [0.0000, 0.0000] | 0.286 | 0.0000 | 887 / 1480 | no / no |
| vital | 0.678 [0.563, 0.764] | 0.572 [0.517, 0.632] | 0.033 | 0.0396 | +0.0000 [0.0000, 0.0000] | 0.0737 | +0.0000 [0.0000, 0.0000] | 0.287 | 0.0000 | 889 / 1472 | yes / yes |
| br_score | 0.678 [0.563, 0.764] | 0.572 [0.517, 0.632] | 0.033 | 0.0396 | +0.0000 [0.0000, 0.0000] | 0.0737 | +0.0000 [0.0000, 0.0000] | 0.287 | 0.0000 | 889 / 1472 | yes / yes |
| br_modulate | 0.688 [0.577, 0.772] | 0.586 [0.508, 0.632] | 0.032 | 0.1643 | **+0.1247 [0.1211, 0.1285]** | 0.2832 | **+0.2094 [0.1798, 0.2391]** | 0.097 | 0.0000 | 6336 / 1384 | no / no |
| sessions | 0.678 [0.563, 0.764] | 0.572 [0.517, 0.632] | 0.033 | 0.0396 | +0.0000 [0.0000, 0.0000] | 0.0737 | +0.0000 [0.0000, 0.0000] | 0.287 | 0.0000 | 889 / 1472 | yes / yes |
| provisional | 0.672 [0.554, 0.761] | 0.558 [0.531, 0.614] | 0.034 | 0.0101 | **-0.0295 [-0.0313, -0.0277]** | 0.0265 | **-0.0472 [-0.0687, -0.0296]** | 0.587 | 0.0000 | 517 / 516 | no / no |
| seq_age_discount | 0.678 [0.563, 0.764] | 0.572 [0.517, 0.632] | 0.033 | 0.0396 | +0.0001 [0.0000, 0.0002] | 0.0737 | +0.0000 [0.0000, 0.0000] | 0.286 | 0.0000 | 885 / 1479 | no / no |
| gate | 0.678 [0.563, 0.764] | 0.572 [0.517, 0.632] | 0.033 | 0.0396 | +0.0000 [-0.0001, 0.0001] | 0.0723 | -0.0015 [-0.0048, 0.0000] | 0.287 | 0.0000 | 7 / 2354 | no / no |
| escalate | 0.678 [0.563, 0.764] | 0.572 [0.517, 0.632] | 0.033 | 0.0396 | +0.0000 [0.0000, 0.0000] | 0.0737 | +0.0000 [0.0000, 0.0000] | 0.287 | 0.0147 | 889 / 1472 | yes / yes |
| session_enforce | 0.678 [0.563, 0.764] | 0.572 [0.518, 0.632] | 0.033 | 0.0421 | **+0.0026 [0.0018, 0.0035]** | 0.0811 | +0.0074 [0.0000, 0.0176] | 0.274 | 0.0000 | 1015 / 1455 | no / no |
| authority | 0.678 [0.563, 0.764] | 0.571 [0.517, 0.633] | 0.033 | 0.0395 | -0.0000 [-0.0001, 0.0000] | 0.0737 | +0.0000 [0.0000, 0.0000] | 0.287 | 0.0000 | 817 / 1543 | no / no |
| engine-traj | 0.687 [0.580, 0.771] | 0.583 [0.537, 0.657] | 0.032 | 0.1143 | **+0.0747 [0.0710, 0.0787]** | 0.2699 | **+0.1962 [0.1623, 0.2294]** | 0.129 | 0.0638 | 2863 / 2697 | no / no |
| gate+flow_ledger | 0.678 [0.563, 0.764] | 0.568 [0.511, 0.648] | 0.033 | 0.0397 | **+0.0001 [0.0000, 0.0003]** | 0.0723 | -0.0015 [-0.0048, 0.0000] | 0.286 | 0.0000 | 7 / 2360 | no / no |
| escalate+gate+flow_ledger | 0.678 [0.563, 0.764] | 0.568 [0.511, 0.648] | 0.033 | 0.0397 | **+0.0001 [0.0000, 0.0003]** | 0.0723 | -0.0015 [-0.0048, 0.0000] | 0.286 | 0.0146 | 658 / 1709 | no / no |
| sessions+vital | 0.678 [0.563, 0.764] | 0.572 [0.517, 0.632] | 0.033 | 0.0396 | +0.0000 [0.0000, 0.0000] | 0.0737 | +0.0000 [0.0000, 0.0000] | 0.287 | 0.0000 | 889 / 1472 | yes / yes |
| engine-traj-shadow | 0.672 [0.552, 0.760] | 0.584 [0.521, 0.678] | 0.032 | 0.1123 | **+0.0728 [0.0689, 0.0765]** | 0.2566 | **+0.1829 [0.1503, 0.2144]** | 0.122 | 0.0640 | 2589 / 2829 | no / no |

**Attack-level decision recall by family** (Δ vs engine with CI below each value).

| row | exfil_volume (N=10) | magnitude_spike (N=50) | novel_target (N=22) | prereq_skip (N=3) | scope_blowup (N=11) | slow_drift (N=9) | splice (N=26) | velocity_burst (N=20) |
|---|---|---|---|---|---|---|---|---|
| engine | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.444 | 0.500 | 1.000 |
| flow_ledger | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.444 | 0.500 | 1.000 |
| vital | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.444 | 0.500 | 1.000 |
| br_score | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.444 | 0.500 | 1.000 |
| br_modulate | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000<br>**+0.556 [0.200, 0.889]** | 0.769<br>**+0.269 [0.111, 0.455]** | 1.000 |
| sessions | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.444 | 0.500 | 1.000 |
| provisional | 0.900<br>-0.100 [-0.333, 0.000] | 0.680<br>**-0.320 [-0.451, -0.190]** | 1.000 | 0.667<br>-0.333 [-1.000, 0.000] | 1.000 | 0.444 | 0.077<br>**-0.423 [-0.632, -0.227]** | 0.950<br>-0.050 [-0.154, 0.000] |
| seq_age_discount | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.444 | 0.500 | 1.000 |
| gate | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.444 | 0.500 | 1.000 |
| escalate | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.444 | 0.500 | 1.000 |
| session_enforce | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.444 | 0.500 | 1.000 |
| authority | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.444 | 0.500 | 1.000 |
| engine-traj | 0.800<br>-0.200 [-0.500, 0.000] | 0.660<br>**-0.340 [-0.473, -0.212]** | 1.000 | 0.667<br>-0.333 [-1.000, 0.000] | 1.000 | 1.000<br>**+0.556 [0.200, 0.889]** | 0.615<br>+0.115 [-0.100, 0.333] | 1.000 |
| gate+flow_ledger | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.444 | 0.500 | 1.000 |
| escalate+gate+flow_ledger | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.444 | 0.500 | 1.000 |
| sessions+vital | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.444 | 0.500 | 1.000 |
| engine-traj-shadow | 0.800<br>-0.200 [-0.500, 0.000] | 0.660<br>**-0.340 [-0.473, -0.212]** | 1.000 | 0.667<br>-0.333 [-1.000, 0.000] | 1.000 | 1.000<br>**+0.556 [0.200, 0.889]** | 0.385<br>-0.115 [-0.419, 0.172] | 0.900<br>-0.100 [-0.238, 0.000] |

**Session halts** (density policy at θ_halt = 0.6 vs the first-flag baseline;
sessions of all splits; precision at a 1.5% session base rate).

| row | false-halt | runaway recall | attacked recall | median calls onset→halt | prec@base | first-flag false-halt | first-flag runaway | max sensitivity step | selected θ_warn / θ_halt (calibration; – = undefined) |
|---|---|---|---|---|---|---|---|---|---|
| sessions | 0.0028 | 0.836 | 0.072 | 7.0 | 0.511 | 0.0422 | 0.944 | 0.0051 | – / 0.200 |
| provisional | 0.0029 | 0.815 | 0.072 | 7.0 | 0.500 | 0.0326 | 0.918 | 0.0048 | – / 0.200 |
| session_enforce | 0.0028 | 0.836 | 0.072 | 7.0 | 0.511 | 0.0422 | 0.944 | 0.0051 | – / 0.200 |
| engine-traj † | 0.0040 | 0.948 | 0.075 | 7.0 | 0.450 | 0.2190 | 1.000 | 0.0173 | – / 0.200 |
| sessions+vital | 0.0029 | 0.948 | 0.072 | 7.0 | 0.526 | 0.0462 | 1.000 | 0.0055 | – / 0.200 |
| engine-traj-shadow | 0.0039 | 0.948 | 0.075 | 7.0 | 0.453 | 0.2133 | 1.000 | 0.0171 | – / 0.200 |

† enforcing replay with `provisional`: the sensitivity steps above θ = 0.6
are approximate (T-43); the shadow rows are exact at every θ.

**Sensitivity curve, engine-traj-shadow** (false-halt / runaway recall /
precision at base rate per θ; calibration-partition benign share).

| θ | false-halt | runaway recall | attacked recall | prec@base | calibration benign share |
|---|---|---|---|---|---|
| 0.20 | 0.0288 | 0.996 | 0.140 | 0.127 | 0.0051 |
| 0.25 | 0.0274 | 0.996 | 0.134 | 0.131 | 0.0051 |
| 0.30 | 0.0274 | 0.996 | 0.134 | 0.131 | 0.0051 |
| 0.35 | 0.0104 | 0.987 | 0.084 | 0.251 | 0.0011 |
| 0.40 | 0.0083 | 0.974 | 0.083 | 0.292 | 0.0011 |
| 0.45 | 0.0083 | 0.974 | 0.083 | 0.292 | 0.0011 |
| 0.50 | 0.0083 | 0.974 | 0.083 | 0.292 | 0.0011 |
| 0.55 | 0.0039 | 0.948 | 0.075 | 0.453 | 0.0003 |
| 0.60 | 0.0039 | 0.948 | 0.075 | 0.453 | 0.0003 |
| 0.65 | 0.0039 | 0.948 | 0.075 | 0.453 | 0.0003 |
| 0.70 | 0.0022 | 0.897 | 0.069 | 0.582 | 0.0000 |
| 0.75 | 0.0020 | 0.888 | 0.069 | 0.606 | 0.0000 |
| 0.80 | 0.0020 | 0.888 | 0.069 | 0.606 | 0.0000 |

**Sessions, splice, vital** (splice reported separately, T-40).

| row | session AUROC (all) | session AUROC (length-matched) | pairs | runaway session AUROC | splice recall, per-agent 95 thr (warm) | splice decision recall | vital false-flag (benign sessions) |
|---|---|---|---|---|---|---|---|
| engine | 0.780 | 0.764 | 1141 | 0.981 | 0.923 (24/26) | 0.500 | – |
| flow_ledger | 0.780 | 0.764 | 1141 | 0.982 | 0.923 (24/26) | 0.500 | – |
| vital | 0.780 | 0.764 | 1141 | 0.981 | 0.923 (24/26) | 0.500 | 0.0385 |
| br_score | 0.780 | 0.764 | 1141 | 0.981 | 0.923 (24/26) | 0.500 | – |
| br_modulate | 0.783 | 0.768 | 1141 | 0.981 | 0.923 (24/26) | 0.769 | – |
| sessions | 0.780 | 0.764 | 1141 | 0.981 | 0.923 (24/26) | 0.500 | – |
| provisional | 0.773 | 0.759 | 1141 | 0.979 | 0.923 (24/26) | 0.077 | – |
| seq_age_discount | 0.780 | 0.765 | 1141 | 0.981 | 0.923 (24/26) | 0.500 | – |
| gate | 0.780 | 0.764 | 1141 | 0.981 | 0.923 (24/26) | 0.500 | – |
| escalate | 0.780 | 0.764 | 1141 | 0.981 | 0.923 (24/26) | 0.500 | – |
| session_enforce | 0.780 | 0.764 | 1141 | 0.981 | 0.923 (24/26) | 0.500 | – |
| authority | 0.780 | 0.764 | 1141 | 0.981 | 0.923 (24/26) | 0.500 | – |
| engine-traj | 0.777 | 0.762 | 1141 | 0.980 | 0.923 (24/26) | 0.615 | 0.0385 |
| gate+flow_ledger | 0.780 | 0.764 | 1141 | 0.982 | 0.923 (24/26) | 0.500 | – |
| escalate+gate+flow_ledger | 0.780 | 0.764 | 1141 | 0.982 | 0.923 (24/26) | 0.500 | – |
| sessions+vital | 0.780 | 0.764 | 1141 | 0.981 | 0.923 (24/26) | 0.500 | 0.0385 |
| engine-traj-shadow | 0.778 | 0.763 | 1141 | 0.979 | 0.923 (24/26) | 0.385 | 0.0385 |

**Attribution** (engine-traj (dataset schemas as hints); attacks carried per group, leave-one-out).

| family | magnitude | novelty | velocity | sequence | vital | gate | provisional | redundant |
|---|---|---|---|---|---|---|---|---|
| exfil_volume | 7 | 0 | 0 | 6 | 0 | 0 | 0 | 0 |
| magnitude_spike | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 32 |
| novel_target | 0 | 21 | 0 | 10 | 0 | 0 | 0 | 0 |
| prereq_skip | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| scope_blowup | 7 | 0 | 0 | 11 | 0 | 0 | 0 | 0 |
| slow_drift | 0 | 3 | 0 | 2 | 0 | 0 | 0 | 4 |
| splice | 0 | 2 | 5 | 12 | 0 | 0 | 0 | 3 |
| velocity_burst | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 20 |

Detected with every group: 121. T-20 totals hold: yes (by construction of the table). Removing a group changes exactly its rows (F10, T-52): no.

| group | carried | gained when removed | detected without |
|---|---|---|---|
| magnitude | 15 | 2 | 108 |
| novelty | 26 | 0 | 95 |
| velocity | 5 | 0 | 116 |
| sequence | 41 | 0 | 80 |
| vital | 0 | 2 | 123 |
| gate | 0 | 0 | 121 |
| provisional | 0 | 24 | 145 |

### oracle-reviewer approval loop (labels act as a perfect human reviewer → optimistic) — seed 1

**Per-call (warm test rows) and decision metrics.** Δ = row − engine, paired session
bootstrap 95% CI; bold where the interval excludes 0.

| row | AUPRC [CI] | FPR@95 per-agent [CI] | prec@base (per-agent 95) | dec. benign FPR | Δ benign FPR | dec. control FPR | Δ control FPR | dec. prec@base | escalation rate | block / approval | scores / decisions = engine |
|---|---|---|---|---|---|---|---|---|---|---|---|
| engine | 0.693 [0.536, 0.766] | 0.553 [0.457, 0.663] | 0.035 | 0.0466 | – | 0.0343 | – | 0.271 | 0.0000 | 955 / 1669 | yes / yes |
| flow_ledger | 0.692 [0.536, 0.765] | 0.546 [0.445, 0.711] | 0.035 | 0.0472 | **+0.0006 [0.0003, 0.0010]** | 0.0343 | +0.0000 [0.0000, 0.0000] | 0.269 | 0.0000 | 952 / 1698 | no / no |
| vital | 0.693 [0.536, 0.766] | 0.553 [0.457, 0.663] | 0.035 | 0.0466 | +0.0000 [0.0000, 0.0000] | 0.0343 | +0.0000 [0.0000, 0.0000] | 0.271 | 0.0000 | 955 / 1669 | yes / yes |
| br_score | 0.693 [0.536, 0.766] | 0.553 [0.457, 0.663] | 0.035 | 0.0466 | +0.0000 [0.0000, 0.0000] | 0.0343 | +0.0000 [0.0000, 0.0000] | 0.271 | 0.0000 | 955 / 1669 | yes / yes |
| br_modulate | 0.714 [0.570, 0.783] | 0.509 [0.460, 0.644] | 0.038 | 0.1700 | **+0.1235 [0.1198, 0.1277]** | 0.2173 | **+0.1830 [0.1375, 0.2190]** | 0.099 | 0.0000 | 6237 / 1466 | no / no |
| sessions | 0.693 [0.536, 0.766] | 0.553 [0.457, 0.663] | 0.035 | 0.0466 | +0.0000 [0.0000, 0.0000] | 0.0343 | +0.0000 [0.0000, 0.0000] | 0.271 | 0.0000 | 955 / 1669 | yes / yes |
| provisional | 0.680 [0.507, 0.757] | 0.578 [0.502, 0.700] | 0.033 | 0.0129 | **-0.0337 [-0.0357, -0.0319]** | 0.0163 | **-0.0180 [-0.0319, -0.0071]** | 0.519 | 0.0000 | 529 / 577 | no / no |
| seq_age_discount | 0.693 [0.536, 0.766] | 0.552 [0.457, 0.663] | 0.035 | 0.0466 | +0.0000 [-0.0001, 0.0001] | 0.0343 | +0.0000 [0.0000, 0.0000] | 0.271 | 0.0000 | 955 / 1669 | no / no |
| gate | 0.693 [0.536, 0.766] | 0.553 [0.457, 0.663] | 0.035 | 0.0466 | +0.0000 [0.0000, 0.0000] | 0.0343 | +0.0000 [0.0000, 0.0000] | 0.271 | 0.0000 | 13 / 2611 | yes / no |
| escalate | 0.693 [0.536, 0.766] | 0.553 [0.457, 0.663] | 0.035 | 0.0466 | +0.0000 [0.0000, 0.0000] | 0.0343 | +0.0000 [0.0000, 0.0000] | 0.271 | 0.0167 | 955 / 1669 | yes / yes |
| session_enforce | 0.693 [0.536, 0.766] | 0.551 [0.457, 0.664] | 0.035 | 0.0503 | **+0.0037 [0.0027, 0.0050]** | 0.0343 | +0.0000 [0.0000, 0.0000] | 0.257 | 0.0000 | 1145 / 1632 | no / no |
| authority | 0.690 [0.530, 0.764] | 0.554 [0.457, 0.663] | 0.035 | 0.0452 | **-0.0014 [-0.0017, -0.0010]** | 0.0343 | +0.0000 [0.0000, 0.0000] | 0.276 | 0.0000 | 835 / 1730 | no / no |
| engine-traj | 0.709 [0.559, 0.780] | 0.554 [0.478, 0.715] | 0.035 | 0.1135 | **+0.0669 [0.0632, 0.0710]** | 0.1993 | **+0.1650 [0.1130, 0.2060]** | 0.136 | 0.0724 | 2847 / 2507 | no / no |
| gate+flow_ledger | 0.692 [0.536, 0.765] | 0.546 [0.445, 0.711] | 0.035 | 0.0472 | **+0.0006 [0.0003, 0.0010]** | 0.0343 | +0.0000 [0.0000, 0.0000] | 0.269 | 0.0000 | 13 / 2637 | no / no |
| escalate+gate+flow_ledger | 0.692 [0.536, 0.765] | 0.546 [0.445, 0.711] | 0.035 | 0.0472 | **+0.0006 [0.0003, 0.0010]** | 0.0343 | +0.0000 [0.0000, 0.0000] | 0.269 | 0.0165 | 665 / 1985 | no / no |
| sessions+vital | 0.693 [0.536, 0.766] | 0.553 [0.457, 0.663] | 0.035 | 0.0466 | +0.0000 [0.0000, 0.0000] | 0.0343 | +0.0000 [0.0000, 0.0000] | 0.271 | 0.0000 | 955 / 1669 | yes / yes |
| engine-traj-shadow | 0.702 [0.547, 0.776] | 0.552 [0.482, 0.717] | 0.035 | 0.1149 | **+0.0684 [0.0646, 0.0725]** | 0.1961 | **+0.1618 [0.1090, 0.2037]** | 0.120 | 0.0720 | 2457 / 2864 | no / no |

**Attack-level decision recall by family** (Δ vs engine with CI below each value).

| row | exfil_volume (N=7) | magnitude_spike (N=55) | novel_target (N=19) | prereq_skip (N=5) | scope_blowup (N=5) | slow_drift (N=4) | splice (N=23) | velocity_burst (N=20) |
|---|---|---|---|---|---|---|---|---|
| engine | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.750 | 0.783 | 1.000 |
| flow_ledger | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.750 | 0.783 | 1.000 |
| vital | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.750 | 0.783 | 1.000 |
| br_score | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.750 | 0.783 | 1.000 |
| br_modulate | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000<br>+0.250 [0.000, 1.000] | 0.870<br>+0.087 [0.000, 0.235] | 1.000 |
| sessions | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.750 | 0.783 | 1.000 |
| provisional | 0.857<br>-0.143 [-0.500, 0.000] | 0.618<br>**-0.382 [-0.523, -0.250]** | 1.000 | 1.000 | 1.000 | 0.500<br>-0.250 [-1.000, 0.000] | 0.565<br>**-0.217 [-0.409, -0.067]** | 0.950<br>-0.050 [-0.160, 0.000] |
| seq_age_discount | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.750 | 0.783 | 1.000 |
| gate | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.750 | 0.783 | 1.000 |
| escalate | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.750 | 0.783 | 1.000 |
| session_enforce | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.750 | 0.783 | 1.000 |
| authority | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.750 | 0.783 | 1.000 |
| engine-traj | 0.857<br>-0.143 [-0.500, 0.000] | 0.545<br>**-0.455 [-0.600, -0.323]** | 0.947<br>-0.053 [-0.177, 0.000] | 1.000 | 1.000 | 1.000<br>+0.250 [0.000, 1.000] | 0.739<br>-0.043 [-0.143, 0.000] | 1.000 |
| gate+flow_ledger | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.750 | 0.783 | 1.000 |
| escalate+gate+flow_ledger | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.750 | 0.783 | 1.000 |
| sessions+vital | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.750 | 0.783 | 1.000 |
| engine-traj-shadow | 0.857<br>-0.143 [-0.500, 0.000] | 0.545<br>**-0.455 [-0.600, -0.323]** | 0.947<br>-0.053 [-0.177, 0.000] | 1.000 | 1.000 | 1.000<br>+0.250 [0.000, 1.000] | 0.565<br>**-0.217 [-0.400, -0.059]** | 0.800<br>**-0.200 [-0.409, -0.043]** |

**Session halts** (density policy at θ_halt = 0.6 vs the first-flag baseline;
sessions of all splits; precision at a 1.5% session base rate).

| row | false-halt | runaway recall | attacked recall | median calls onset→halt | prec@base | first-flag false-halt | first-flag runaway | max sensitivity step | selected θ_warn / θ_halt (calibration; – = undefined) |
|---|---|---|---|---|---|---|---|---|---|
| sessions | 0.0030 | 0.835 | 0.086 | 7.0 | 0.545 | 0.0443 | 0.963 | 0.0054 | – / 0.200 |
| provisional | 0.0030 | 0.794 | 0.083 | 7.0 | 0.529 | 0.0363 | 0.934 | 0.0052 | – / 0.200 |
| session_enforce | 0.0030 | 0.835 | 0.086 | 7.0 | 0.543 | 0.0443 | 0.963 | 0.0055 | – / 0.200 |
| engine-traj † | 0.0038 | 0.885 | 0.091 | 7.0 | 0.498 | 0.2124 | 1.000 | 0.0181 | – / 0.200 |
| sessions+vital | 0.0030 | 0.905 | 0.086 | 6.0 | 0.557 | 0.0475 | 1.000 | 0.0057 | – / 0.200 |
| engine-traj-shadow | 0.0037 | 0.877 | 0.091 | 7.0 | 0.505 | 0.2002 | 1.000 | 0.0176 | – / 0.200 |

† enforcing replay with `provisional`: the sensitivity steps above θ = 0.6
are approximate (T-43); the shadow rows are exact at every θ.

**Sensitivity curve, engine-traj-shadow** (false-halt / runaway recall /
precision at base rate per θ; calibration-partition benign share).

| θ | false-halt | runaway recall | attacked recall | prec@base | calibration benign share |
|---|---|---|---|---|---|
| 0.20 | 0.0308 | 1.000 | 0.197 | 0.150 | 0.0050 |
| 0.25 | 0.0288 | 1.000 | 0.196 | 0.158 | 0.0050 |
| 0.30 | 0.0288 | 1.000 | 0.196 | 0.158 | 0.0050 |
| 0.35 | 0.0112 | 0.975 | 0.112 | 0.278 | 0.0008 |
| 0.40 | 0.0088 | 0.967 | 0.108 | 0.325 | 0.0008 |
| 0.45 | 0.0088 | 0.967 | 0.108 | 0.325 | 0.0008 |
| 0.50 | 0.0088 | 0.967 | 0.108 | 0.325 | 0.0008 |
| 0.55 | 0.0038 | 0.889 | 0.091 | 0.504 | 0.0002 |
| 0.60 | 0.0037 | 0.877 | 0.091 | 0.505 | 0.0002 |
| 0.65 | 0.0037 | 0.877 | 0.091 | 0.505 | 0.0002 |
| 0.70 | 0.0023 | 0.848 | 0.086 | 0.614 | 0.0000 |
| 0.75 | 0.0022 | 0.848 | 0.086 | 0.619 | 0.0000 |
| 0.80 | 0.0022 | 0.848 | 0.086 | 0.619 | 0.0000 |

**Sessions, splice, vital** (splice reported separately, T-40).

| row | session AUROC (all) | session AUROC (length-matched) | pairs | runaway session AUROC | splice recall, per-agent 95 thr (warm) | splice decision recall | vital false-flag (benign sessions) |
|---|---|---|---|---|---|---|---|
| engine | 0.830 | 0.804 | 906 | 0.976 | 0.957 (22/23) | 0.783 | – |
| flow_ledger | 0.830 | 0.804 | 906 | 0.977 | 0.957 (22/23) | 0.783 | – |
| vital | 0.830 | 0.804 | 906 | 0.976 | 0.957 (22/23) | 0.783 | 0.0363 |
| br_score | 0.830 | 0.804 | 906 | 0.976 | 0.957 (22/23) | 0.783 | – |
| br_modulate | 0.835 | 0.809 | 906 | 0.977 | 0.957 (22/23) | 0.870 | – |
| sessions | 0.830 | 0.804 | 906 | 0.976 | 0.957 (22/23) | 0.783 | – |
| provisional | 0.822 | 0.795 | 906 | 0.971 | 0.957 (22/23) | 0.565 | – |
| seq_age_discount | 0.830 | 0.804 | 906 | 0.976 | 0.957 (22/23) | 0.783 | – |
| gate | 0.830 | 0.804 | 906 | 0.976 | 0.957 (22/23) | 0.783 | – |
| escalate | 0.830 | 0.804 | 906 | 0.976 | 0.957 (22/23) | 0.783 | – |
| session_enforce | 0.830 | 0.804 | 906 | 0.976 | 0.957 (22/23) | 0.783 | – |
| authority | 0.829 | 0.803 | 906 | 0.975 | 0.957 (22/23) | 0.783 | – |
| engine-traj | 0.828 | 0.801 | 906 | 0.975 | 0.957 (22/23) | 0.739 | 0.0363 |
| gate+flow_ledger | 0.830 | 0.804 | 906 | 0.977 | 0.957 (22/23) | 0.783 | – |
| escalate+gate+flow_ledger | 0.830 | 0.804 | 906 | 0.977 | 0.957 (22/23) | 0.783 | – |
| sessions+vital | 0.830 | 0.804 | 906 | 0.976 | 0.957 (22/23) | 0.783 | 0.0363 |
| engine-traj-shadow | 0.828 | 0.801 | 906 | 0.972 | 0.957 (22/23) | 0.565 | 0.0363 |

**Attribution** (engine-traj (dataset schemas as hints); attacks carried per group, leave-one-out).

| family | magnitude | novelty | velocity | sequence | vital | gate | provisional | redundant |
|---|---|---|---|---|---|---|---|---|
| exfil_volume | 4 | 0 | 0 | 4 | 0 | 0 | 0 | 2 |
| magnitude_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 30 |
| novel_target | 0 | 17 | 0 | 13 | 0 | 0 | 0 | 0 |
| prereq_skip | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 |
| scope_blowup | 2 | 0 | 0 | 5 | 0 | 0 | 0 | 0 |
| slow_drift | 2 | 2 | 0 | 2 | 0 | 0 | 0 | 2 |
| splice | 0 | 2 | 0 | 14 | 0 | 0 | 0 | 2 |
| velocity_burst | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 20 |

Detected with every group: 105. T-20 totals hold: yes (by construction of the table). Removing a group changes exactly its rows (F10, T-52): no.

| group | carried | gained when removed | detected without |
|---|---|---|---|
| magnitude | 8 | 1 | 98 |
| novelty | 21 | 0 | 84 |
| velocity | 0 | 1 | 106 |
| sequence | 38 | 0 | 67 |
| vital | 0 | 0 | 105 |
| gate | 0 | 0 | 105 |
| provisional | 0 | 30 | 135 |

## Latency (alternated A/B, spec §5.2 allow-all updates)

alternated A/B: every engine scores each call in one process, the order rotated by one per call; all updated with ALLOWED (spec allow-all); score() wall time; 30000 timed calls after 10000 warm-up calls, first seed's dataset.

| row | p50 µs | p99 µs | mean µs | p50 / engine | p99 / engine |
|---|---|---|---|---|---|
| engine | 78.9 | 154.9 | 83.5 | 1.00 | 1.00 |
| flow_ledger | 76.4 | 149.9 | 81.1 | 0.97 | 0.97 |
| vital | 84.8 | 157.4 | 89.7 | 1.08 | 1.02 |
| br_score | 83.5 | 166.9 | 118.0 | 1.06 | 1.08 |
| br_modulate | 82.5 | 161.5 | 87.0 | 1.05 | 1.04 |
| sessions | 86.1 | 167.5 | 91.2 | 1.09 | 1.08 |
| provisional | 90.0 | 178.0 | 95.7 | 1.14 | 1.15 |
| seq_age_discount | 78.4 | 149.8 | 82.8 | 0.99 | 0.97 |
| gate | 84.0 | 173.5 | 89.2 | 1.06 | 1.12 |
| escalate | 82.1 | 163.3 | 108.4 | 1.04 | 1.05 |
| session_enforce | 85.6 | 166.1 | 90.5 | 1.09 | 1.07 |
| authority | 81.4 | 153.3 | 85.8 | 1.03 | 0.99 |
| engine-traj | 108.4 | 208.3 | 115.3 | 1.37 | 1.34 |
| gate+flow_ledger | 87.5 | 176.3 | 92.9 | 1.11 | 1.14 |
| escalate+gate+flow_ledger | 84.8 | 173.4 | 90.0 | 1.08 | 1.12 |
| sessions+vital | 95.5 | 183.6 | 101.0 | 1.21 | 1.19 |

Wall time: {"datasets":0.0,"replays":2180.6,"latency":123.9,"report":25.3,"total":2329.8,"n_jobs":96,"workers":4}
