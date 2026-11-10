# 003 — Prerequisite threshold (`prereq_rate`, `prereq_min_n`)

Spec §10 open decision 3. Status: decided (Task 1.14) — **default changed 0.95 → 0.90**.

## Context

The sequence detector mines X as a prerequisite of tool Y once `n_Y ≥ prereq_min_n` (50) and X appeared
before Y in ≥ `prereq_rate` of Y's observations (§6.6b); `s_prereq = 1` when a mined prerequisite is
absent. The synthetic grammar skips a skippable lookup with probability `noise.skip_lookup` (personas use
0.02–0.06), so a lookup's base rate before its write is ≈ 1 − skip_lookup ≈ 0.94–0.98 — right at the
0.95 threshold.

## Options

`prereq_rate` ∈ {0.90, 0.95, 0.98}, swept against grammars with `skip_lookup` ∈ {0.10, 0.05, 0.02} applied
to every persona (datasets regenerated from the seed-0 config by `sweeps.skip_dataset`), plus the
as-configured per-persona rates. `prereq_min_n` stays 50 (not swept).

## Evidence

`evidence/sweeps-seed0.md`, suite `prereq`. "Rule recall" = share of `prereq_skip` attacks on which the mined
rule itself fires; "rule benign" = share of warm benign rows on which it fires; gate = attack-level
prereq_skip recall at live thresholds; benign FPR = all warm benign rows at live thresholds.

| grammar | rate | rule recall | rule benign | gate recall (prereq_skip) | benign FPR | AUPRC |
|---|---|---|---|---|---|---|
| persona (0.02–0.06) | 0.90 | 0.933 | 0.0017 | 0.933 | 0.0039 | 0.575 |
| persona | 0.95 | 0.400 | 0.0004 | 0.867 | 0.0027 | 0.550 |
| persona | 0.98 | 0.267 | 0.0000 | 0.800 | 0.0023 | 0.499 |
| skip 0.10 | 0.90 | 1.000 | 0.0003 | 1.000 | 0.0027 | 0.694 |
| skip 0.10 | 0.95 | 0.667 | 0.0001 | 0.889 | 0.0026 | 0.638 |
| skip 0.10 | 0.98 | 0.333 | 0.0000 | 0.778 | 0.0026 | 0.509 |
| skip 0.05 | 0.90 | 1.000 | 0.0018 | 1.000 | 0.0039 | 0.597 |
| skip 0.05 | 0.95 | 0.375 | 0.0003 | 0.812 | 0.0024 | 0.577 |
| skip 0.05 | 0.98 | 0.125 | 0.0000 | 0.625 | 0.0022 | 0.519 |
| skip 0.02 | 0.90 | 0.867 | 0.0013 | 1.000 | 0.0040 | 0.672 |
| skip 0.02 | 0.95 | 0.467 | 0.0008 | 0.933 | 0.0034 | 0.646 |
| skip 0.02 | 0.98 | 0.400 | 0.0002 | 0.867 | 0.0029 | 0.615 |

prereq_skip has 9–16 warm test attacks per dataset, so single cells move in steps of 6–11 pp; the
ordering 0.90 > 0.95 > 0.98 holds in **every** grammar for rule recall, gate recall and AUPRC. Control
FPR does not move with the rate (0.062 / 0.068 / 0.062 / 0.078 per dataset, flat across rates). Side
effect: at 0.90 the rule also fires on many velocity bursts (duplicated writes), so velocity_burst
recall rises (0.60 → 0.70 on the persona grammar) while velocity's top-reason share falls (0.63 → 0.26).

**Evidence limits.** Each grammar has only 9–16 warm test prereq_skip attacks, so one attack moves
gate recall by 6–11 pp. The rate was selected on the same time split's test rows that the Tier-1
gate preview uses. There is no separate selection split and no second seed, so this is not an
out-of-sample result. What argues against plain overfitting:
- the ordering is the same in four independently regenerated grammars;
- the effect is on the rule itself (rule recall moves 25–60 pp, far larger than one attack);
- it has a mechanistic cause (the threshold sat on the lookup base rate).

A held-out seed should confirm it before Tier 2 publishes numbers.

## Decision

Set **`prereq_rate = 0.90`** (spec: 0.95); keep `prereq_min_n = 50`. A threshold has to sit below the
lookup's base rate with margin; at 0.95 it sits on top of it, so mining flaps and the rule catches only
37.5–66.7 % of skips. 0.90 is the best setting in all four grammars, not just the default one. Cost, stated plainly: live benign FPR rises from 0.27 % to 0.39 % on
the persona grammar (+0.12 pp; the rule fires on 0.17 % of benign rows), and the velocity-burst
top-reason share drops. `EngineConfig`, `tests/core/test_config.py` (as a recorded override of the spec
value) and the prereq boundary tests were updated; the engine golden and the CI baseline were
regenerated in the same commit and are byte-identical (the 3-persona, 200-session CI dataset mines no
prerequisite whose rate falls in [0.90, 0.95)). Pinned at engine level by
`tests/core/test_engine.py::test_default_engine_mines_a_92_percent_prerequisite_that_095_would_not`.
Later finding (fix round 1, evidence `saturation` section): at 0.90 the prereq rule supplies the top-1
score of 47 % of benign fused == 1.0 rows (302 of 638). Those rows are part of the benign tail that
the calibrator learns (decision 006).
