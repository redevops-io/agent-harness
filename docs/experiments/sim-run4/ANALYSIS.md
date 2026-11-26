# Run 4: a replicate of run 3 (same pair, same configuration, new sampling)

Run 3 found that the arm without memory improved most on the target and the two arms with memory kept the source task best. Run 4 repeats the whole pipeline with the same configuration (egress-vetting source, command-vetting target, feedback v2, rejected attempts recorded, 6 rounds, 2 candidates, same model and sampling settings, the same task sets): the source run, the four transfer arms and the bandit. Nothing was changed between the runs. Reported side by side, whichever looked better.

## Result: run 3 is not reproduced

Target held-out (command-vetting, 30 tasks, one task = 3.3 points), start -> final:

| Arm | run 3 | run 4 |
|---|---|---|
| cold start | 0.500 -> 0.500 | 0.533 -> 0.933 |
| harness transfer (no memory) | 0.500 -> 0.767 | 0.700 -> 0.933 |
| problem-only | 0.500 -> 0.600 | 0.700 -> 0.967 |
| full SIM | 0.500 -> 0.633 | 0.700 -> 0.967 |
| existing learning (bandit) | 0.500 -> 0.500 | 0.533 -> 0.567 |

Source held-out after target optimization (retention): run 3 cold 0.667, transfer 0.633, problem-only 0.833, full SIM 0.833, bandit 0.633; run 4 cold 0.567, transfer 0.633, problem-only 0.700, full SIM 0.733, bandit 0.667.

- **The spread between the two runs of one arm (0.00 to 0.43) is larger than the spread between arms inside either run.** In run 4 the four optimized arms end within one task of each other (0.933 to 0.967) and cold start, which did nothing in run 3, is among them. The ordering reported for run 3 (harness transfer best on the target) does not appear in run 4.
- **Retention:** full SIM and problem-only kept the source task better than the others in both runs (0.833 and 0.733 for full SIM, 0.833 and 0.700 for problem-only, against 0.567 to 0.667 for cold start, transfer and the bandit), but the margins are 1 to 5 tasks and run 4's are smaller. Two runs, in the same direction, are not an established effect.
- **The bandit** (four prewritten variants) did not improve in either run (0.500, 0.567).
- **The starting point is not shared.** The carried harness differs between runs because each run has its own source run: run 3's source accepted 2 edits (source held-out 0.667 seed, 0.700 evolved), run 4's accepted 3 (0.767 seed, 0.700 evolved). Run 3's evolved harness scored 0.500 on the target held-out before optimization, run 4's 0.700. So the arms in run 4 started higher, and the improvement column mixes arm effect with starting point. The final scores are the comparable quantity.
- **The evaluator is not deterministic at temperature 0.** The same seed harness on the same 30 held-out tasks scored 0.667 on the source set in run 3 and 0.767 in run 4, and 0.500 against 0.533 on the target set. The only difference between the two evaluations is the server's own nondeterminism (batching). That is up to 3 tasks of measurement noise on one set before any optimization effect.
- **The evolve set saturates.** Every optimized arm reached 1.000 (or 0.967) on its 30 evolve tasks in both runs. Held-out scores of 0.567 to 0.967 against evolve scores of 1.0 are a generalization gap that varies by arm and run.
- **Memory.** Run 4's source memory holds 2 problems (P1: 6 attempts, 3 accepted; P2: 1 attempt), run 3's one problem. Both are far too small for a question about choosing among problems.

## What this supports

On this task pair, one model and 30-task held-out sets, differences between memory conditions are not distinguishable from run-to-run variation. Run 3's statement that the arm without memory improved most on the target, and its description of the memory arms' retention as the more reliable finding, are superseded by this replicate: the retention direction repeated, its size did not. No claim that SIM, or any arm, is better or worse follows from runs 3 and 4.

## Not recorded

Runs 3 and 4 were run before token counting was added: they have no per-call token or latency record (wall seconds per arm are in `results.json`). The model server is vLLM 0.34.0 serving `Qwen/Qwen3.8-27B-FP8` (FP8 weights) as `qwen3.8-27b` with a 65,536-token context; thinking was disabled for the policy, proposer and selector; policy temperature 0 with 400 output tokens, proposer and selector temperature 0.3 with 4,000.
