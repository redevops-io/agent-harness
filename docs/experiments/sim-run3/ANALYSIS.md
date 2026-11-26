# Run 3: egress-vetting -> command-vetting, Qwen3.8-27B (local), feedback=v2, record_rejected=true, 6 rounds, 2 candidates per round

Planned before it ran (`PLAN_run3.md`); kept whole. Pair reversed from runs 1 and 2 because the egress task is the one prompt edits can improve.

## What is different from runs 1 and 2

This is the first run in which the source run did its job: 2 of its 5 recorded attempts were accepted, so the carried harness differs from the seed and the carried memory
is non-empty (one problem, "the model over-analyses the DNS answers and domain names", 5 attempts: 2 accepted, 3 rejected). The source-evolved harness scored 0.700 on the
source held-out set (seed 0.667) and 0.500 on the target held-out set (seed 0.500): it neither helped nor hurt the target before target optimization.

## Results (target = command-vetting; 30 held-out tasks, one task = 3.3 points; one run per arm)

| Arm | target held-out start -> final | change | source held-out after target optimization (retention) | promotions | missed predictions | final evolve |
|---|---|---|---|---|---|---|
| cold start (seed, no memory) | 0.500 -> 0.500 | 0.000 | 0.667 | 1 | 27/35 | 0.833 |
| harness transfer (evolved harness, no memory) | 0.500 -> 0.767 | +0.267 | 0.633 | 3 | 2/6 | 1.000 |
| problem-only (evolved harness, root cause only) | 0.500 -> 0.600 | +0.100 | 0.833 | 2 | 3/10 | 0.967 |
| full SIM (evolved harness, complete record) | 0.500 -> 0.633 | +0.133 | 0.833 | 2 | 1/5 | 1.000 |
| existing learning (bandit, 4 prewritten variants) | 0.500 -> 0.500 | 0.000 | 0.633 | - | - | - |

## What the files support

1. **Cold start did not improve and its edits were mostly wrong:** 27 of 35 predicted fixes failed, and its one promotion regressed 34 task-evaluations. Starting from the evolved harness made the optimizer's predictions far better (2/6, 1/5 and 3/10 missed in the three arms that carried it).
2. **Target improvement is largest for harness transfer without memory (+0.267), not for full SIM (+0.133).** The difference is 4 tasks, one run each, and sampling in the engineers is not controlled: this run does not establish an ordering between these two arms.
3. **Both arms that carried memory retained the source task better (0.833 against 0.633 and 0.667 for the two without memory) and made no or almost no regressions in promoted attempts (0 for full SIM, 1 for problem-only; 7 for harness transfer, 34 for cold start).** The gap is 5 to 6 tasks of 30 and holds for both memory arms, which makes it less likely to be noise than a one-task difference, but it is still one run and was not tested against a replicate.
4. **The evolve set saturated.** Harness transfer and full SIM reached 1.000 on the 30 evolve tasks by round 3 and then had no failing task to learn from (full SIM proposed only 4 candidates in 6 rounds). Evolve 1.000 against held-out 0.767 and 0.633 is a 7 to 12 task generalization gap: the optimizer fit its evolve tasks. A larger evolve set or a stop rule on saturation is needed before the held-out numbers mean much.
5. **No evidence about repeated failed interventions here**: full SIM stopped proposing once nothing failed, so the question cannot be answered from this run.

## Not shown

Variance (no replicate), transfer across models, real workspace, retrieval or legal tasks, retrieval-based versus complete memory (the other session's selector), and whether the pattern in 3 survives a different task pair. The proposer, selector and policy are one model, and the tasks are small synthetic sets labelled by this package's own policy code.

## Next

Replicate this exact configuration (same pair, same tasks, new sampling) to measure how much of 2 and 3 is sampling; then enlarge the evolve set and add a stop rule on saturation.
