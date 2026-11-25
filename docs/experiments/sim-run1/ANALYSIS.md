# Run 1: command-vetting -> egress-vetting, Qwen3.8-27B (local), configuration feedback=v1, record_rejected=false

Everything the run produced is in this directory: per round `rounds/round-NNN.manifest.json` (written before any candidate was evaluated) and `.outcome.json`, each arm's `sim.json` and `state.json`, every proposer and selector reply (`proposer_replies.json`, `selector_replies.json`), the bandit trace, and `results.json`. Nothing was selected or edited after the fact.

## What the run does and does not show

1. **It does not test transfer.** The source run (4 rounds, 8 proposed edits) promoted nothing, so the source-evolved harness is byte-identical to the seed (same artifact id `2415b94ffc...`) and the carried memory has 0 problems. Cold start, harness transfer, problem-only and full SIM therefore began from the same harness with the same (empty) memory. Their differences are proposer sampling (temperature 0.3) and evaluation noise, not memory.
2. **Evaluation noise is about one task.** The same seed harness scored 0.667 (cold start, round 1) and 0.633 (the other three arms, round 1) on the same 30 evolve tasks at temperature 0. Held-out sets have 30 tasks, so one task is 3.3 points. The held-out deltas (-0.10 to +0.067) are 2 to 3 tasks, one run per arm: no ranking is supported.
3. **The existing-learning arm** (bandit over four prewritten variants) chose `stepwise` and scored 0.733 on target held-out, equal to cold start. Same caveat.

| Arm | target held-out start -> final | source held-out final | promotions | predictions missed |
|---|---|---|---|---|
| cold start | 0.667 -> 0.733 | 0.567 | 2 | 5/10 |
| harness transfer | 0.667 -> 0.600 | 0.600 | 2 | 10/14 |
| problem-only | 0.667 -> 0.567 | 0.433 | 2 | 8/12 |
| full SIM | 0.667 -> 0.600 | 0.467 | 2 | 3/6 |
| existing learning (bandit) | 0.667 -> 0.733 | 0.533 | - | - |
| seed (reference) | 0.667 | 0.500 | | |

## Separating the causes (evidence from the saved files)

- **Evaluator accuracy.** In the source baseline, 5 of the 6 failing evolve tasks were replies with no verdict at all: the policy model reasoned at length and was cut off at 400 tokens. The evaluator graded these as failures (correct under its rule) but v1 feedback showed the proposer only the tail of the reply, not that no verdict existed. The scorer extracts the last ALLOW/DENY in the reply; no case of a correct reply being scored wrong was found, but replies that end undecided are indistinguishable from wrong ones in v1.
- **Diagnostic accuracy.** The proposer's first diagnosis (round 1) was that the model "conflates the command allowlist with sandbox/path checks" and should ignore paths. The oracle (`CommandPolicy`) denies commands whose file arguments leave the workspace, so the diagnosis was the opposite of the rule. It predicted 6 fixes; 1 came true; 3 tasks that had passed failed afterwards (all 8 candidates in the source run scored 0.733 against the current 0.800).
- **Intervention quality.** Edits that rewrote the system prompt wholesale (the proposer must return complete files) regressed previously passing tasks in every promoted attempt: 6 regressions across cold start's two promotions, 4 in full SIM, 3 in problem-only, 2 in harness transfer.
- **Memory utilization.** Under the paper's rule only accepted edits enter the memory. The source run accepted none, so its memory stayed empty even though 4 of its 8 proposals were near-repeats of an earlier round. The memory could not discourage repetition because nothing about rejected attempts was recorded. This is a property of the rule and of this run, not evidence against SIM; it is the reason for the `record_rejected` configuration and the next run.

## Next (run 2, separately versioned)

`feedback=v2` (the evaluator's observed verdict and cut-off replies are shown to the proposer) and `record_rejected=true`, 6 rounds, same models and tasks, so that the source run has a chance to produce a non-empty memory. Held-out results stay outside every loop in both runs.
