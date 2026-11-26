# Run 2: command-vetting -> egress-vetting, Qwen3.8-27B (local), configuration feedback=v2, record_rejected=true, 6 rounds, 2 candidates per round

Everything the run produced is in this directory (round manifests written before evaluation, outcomes, each arm's `sim.json` and `state.json`, every proposer and selector reply, `results.json`). Nothing was selected or edited afterwards. Run 1 (`../sim-run1`) used `feedback=v1`, `record_rejected=false`, 4 rounds.

## What the run is, exactly

The source run again accepted no edit in six rounds, so the "source-evolved harness" is the seed and **this is not a transfer experiment**. What the arms differ in is the memory the proposer and selector are shown, from an identical starting harness and identical budgets:

| Arm | memory shown | target held-out start -> final | final evolve score | candidate scores (mean / best, evaluated) |
|---|---|---|---|---|
| cold start | none | 0.633 -> 0.867 | 0.933 | 0.807 / 0.933 (9) |
| harness transfer | none | 0.667 -> 0.867 | 0.967 | 0.842 / 0.967 (8) |
| problem-only | root causes of the failed source run | 0.667 -> 0.667 | 0.667 | 0.493 / 0.667 (9) |
| full SIM | the failed source run in full, 8 rejected attempts | 0.667 -> 0.700 | 0.667 | 0.613 / 0.700 (5) |
| existing learning (bandit over 4 prewritten variants) | - | 0.633 -> 0.533 | - | - |

The seed scored 0.633 and 0.667 on the same target held-out tasks in two evaluations (noise of one task, 3.3 points). One run per arm, 30 held-out tasks: the direction below is consistent across the two arms in each group, which is not the same as a measured effect.

## What the files show

1. **Both arms that were given the source's memory accepted nothing in six rounds; both arms without memory accepted edits and reached evolve 0.93 and 0.97.** Proposals were made and evaluated in every arm (no parse failures, no refusals), so the memory arms did not stall for lack of output: their candidates scored lower (mean 0.49 and 0.61 against 0.81 and 0.84).
2. **The memory they were given was the failed source run's own diagnoses.** All 8 source attempts were rejected and regressed 24 tasks between them; the two problems recorded ("the model conflates the command policy with sandbox/path checks"; "the model is confused by 'sandboxed agent' and 'workspace root'") were the proposer's guesses about the command task, never confirmed, and the first of them is the opposite of what the policy code enforces (see run 1's analysis). In the egress task the memory arms' proposals read like the same diagnosis moved across: "check ONLY the host against the allowlist", "ignore external security context (DNS validity, IP reputation)", "treat DNS answers as irrelevant". The egress oracle denies a connection when any DNS answer is private, so those edits are wrong there too (candidate scores 0.40 to 0.57 against a current 0.667).
3. **Hypothesis (not shown by this run):** unconfirmed diagnoses recorded as problems anchor the proposer when carried into another task, even next to the outcomes that refuted them (full SIM saw the 22/30 results and 24 regressions and still proposed the same kind of edit). The paper's design assumes recorded problems come from edits that were selected on measured gain; recording rejected attempts under their original diagnoses departs from that.
4. **Repetition.** Near-repeat counts (text similarity above 0.6 with an earlier round's proposal): cold start 3 of 12, harness transfer 2 of 12, problem-only 5 of 10, full SIM 3 of 8. The memory arms repeated at least as often as the arms without memory, so in this run SIM did not prevent repeated unsuccessful interventions. The measure is crude.
5. **Retention.** Source held-out after target optimization: seed 0.50 and 0.53; cold start 0.433, harness transfer 0.567, problem-only 0.50, full SIM 0.50. Target edits cost the source task up to 0.1 in one arm.
6. **Evaluator.** v2 feedback told the proposer which source failures were replies cut off before a verdict; the source run still accepted nothing (all 8 candidates scored 0.733 against 0.800, each regressing 12 passing tasks). The identical score suggests the proposer re-issued essentially one edit.

## Not shown

Whether SIM helps when the source run succeeds (this one did not); whether the memory effect repeats (one run per arm); cross-model or real workspace, retrieval and legal tasks. The proposer, the selector and the policy are one model.

## Next

A source run that actually promotes something (a better proposer prompt, or a task pair where the seed fails for reasons a rewrite can fix), and an arm that shows the same rejected attempts under a "refuted" label instead of as open problems, so the two readings of item 3 can be separated.
