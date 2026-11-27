# G1: does real work make a memory worth selecting from? (plan, written before the run)

Follows `NEXT_EXPERIMENT.md`. The G0 pilot (`~/qualification/jobbench/pilot-g0/RESULTS.md`, copied into `docs/experiments/jobbench-pilot-g0/`) measured one JobBench rollout at 5 to 10 minutes and 250,000 to 570,000 prompt tokens, one judging at 4.5 to 13 minutes, and
the same task scoring 0.10 and 0.37 on two rollouts. The design (21 evolve tasks, 6 rounds of 2 candidates, one rollout) would take 60 to 120 hours and could not tell a better harness from a lucky rollout. The cuts below are the ones
NEXT_EXPERIMENT.md allows ("the evolve set is cut by stratum, never the held-out set"); they are fixed here, before any G1 result exists.

| Setting | G1 | Why |
|---|---|---|
| Evolve tasks | 8 of the 21 train tasks, `random.Random(20261126).sample(sorted(train), 8)` (`jobbench-seed42/g1_evolve_tasks.txt`) | cost; a seeded draw, not a choice. Train has one task per occupation, so every task is its own stratum |
| Held-out | `unseen.txt`, not evaluated in G1 | G1 asks only whether a record accumulates; the held-out set stays untouched for the comparison, if the gate passes |
| Rollouts per task | 2, score = mean | one rollout moved by 0.27 between two identical runs |
| Rounds, candidates | 4 rounds of 2 candidates, rejected attempts recorded | cost: 1 baseline + up to 8 candidate evaluations of 16 rollouts each |
| Parallel tasks | 3 | the server's capacity is the limit and is not measured under load; the run records its own wall-clock |
| Memory | complete record, no selection (the paper's full condition) | G1 produces the record that selection would be tested on |
| Feedback to the proposer | score, how each rollout stopped, and the criteria the judge found UNMET for the worst rollout, in the judge's words (evolve tasks only); never the rubric file | NEXT_EXPERIMENT.md; the held-out set is never shown to an evaluator that feeds a proposer |
| Actor, proposer, selector, judge | Qwen3.8-27B FP8 (vLLM 0.34.0, 65,536 context), thinking off; actor temperature 0.55, 16,384 output tokens; judge validated in `jobbench-judge-validation` (94.8 percent criterion agreement with Kimi on low-scoring outputs) | owner decision: local, no paid judge |
| Context manager | Context Runtime (commit 3b55bec), web off, sandbox network none | as the pilot |

**Gate (unchanged, applied to what accumulates):** the comparison of complete against selected memory runs only if the record holds at least 6 distinct problems and is larger than the selected arm's budget, fixed at one half of the record's size. The run computes both numbers and states them.
If the gate fails, G1 reports exactly that; no record is added, copied or split, and no threshold moves.

**Estimated cost:** 9 evaluations x 16 rollouts x about 7.5 minutes plus judging, with 3 in parallel: about 10 to 16 hours of local inference. Token use and seconds are recorded.

**What a result would not show:** a better harness (the held-out set is not evaluated), anything about another model, or a rate: one run.

## Amendment after the smoke run, before the real run (2026-11-26)

A one-task smoke run (`1 baseline + 1 candidate evaluation`, 53 minutes, plumbing only) found that the optimizer accepted a candidate whose judge score fell from 16/74 to 6/74. Cause: the loop's accept guard compared the *pass fraction* (tasks at or above the 0.5 pass line), and both harnesses were below the line, so 0.0 was not less than 0.0. No threshold of the gate changes. What changes is the optimizer's own accept rule, fixed here before any G1 result exists:

* `Feedback` also carries `mean_score`, the mean of the per-task scores (for a pass/fail evaluator it equals the pass fraction, so earlier experiments behave as before).
* A candidate is refused when its pass fraction is lower than the current harness's, **or** when the pass fractions are equal and its `mean_score` is lower. Tests: `tests/sim/test_loop.py` (a worse graded harness is not promoted; a better one is).
* The smoke candidate's rollout ran 200 turns and 7.4 million prompt tokens and delivered one placeholder-blocked file. That is a real failure class (`max_turns`, `invalid_deliverable`) and is reported as one; the turn limit is not lowered, because that would change the task. Cost is local inference time, recorded per evaluation.
* The proposer's root-cause text in the smoke run was generic ("the system prompt was empty"). With real unmet-criteria feedback and 8 tasks that may improve; if the record is generic, G1 reports that and the gate decides.

## Amendment 2: the evaluator's failure in run 1 and its repair (2026-11-27, prospective: no run 2 result exists)

**What happened.** Run 1 (started 2026-11-26, harness commit 00e1db4) stopped in its second evaluation (a candidate) with `EvaluationIncomplete`: for `social_science_research_assistants/task3` (rollout 1) 18 rubric criteria got no judge reply, so the harness had no score and the run raised, as designed. **It is classified as an evaluation/infrastructure failure, not a result about SIM.** Nothing about it is a negative or positive finding; its artifacts are preserved read-only with a checksum manifest in `~/qualification/jobbench/g1-run1-failed-evaluation-2026-11-26/` and none of its scores are used.

**Cause (measured, not assumed).** The judge prompt for that deliverable is 61,716 tokens of the server's 65,536-token context, leaving 3,820 tokens for the reply. With the server's default (thinking on) the model spends them on reasoning (13,228 characters, empty content, `finish_reason=length`) on rubrics 0, 1, 3 and 4, on every one of three repeats and on a single instrumented request (so not contention, a timeout or a malformed reply); rubrics 2 and 5, whose replies need about 2,200 tokens, fit. With thinking disabled the same rubric replies in 1,197 tokens. The truncated-JSON cases are the same cut-off. Server logs were not reachable from here; the evidence is the client side (`~/qualification/jobbench/g1-eval-reliability/`).

**What changes (evaluator only; nothing about the experiment).**
* Judge verdicts are kept: a rubric the judge answered, whatever it decided, is never asked again and never replaced; a stored verdict is used only for the same files, rubrics and judge (fingerprint). Previously the harness deleted the detail file before every attempt, so one missing rubric discarded the others.
* Recovery is bounded to 3 passes (was 1) and only for rubrics with no usable reply. A pass after a cut-off reply, and the last pass, asks with thinking disabled (the same model, prompt, rubric and parser; `judge_shim.py`). Rubrics answered that way are marked `recovered_with: thinking_off` and counted in the run's report. After 3 passes a rubric without a verdict still makes the evaluation `ungraded`, never a 0.
* A finished agent answer for exactly the same harness, task and rollout is not generated again (`.rollout.sealed`); a genuine task failure keeps its recorded failure and is not retried.

**Unchanged:** model, tasks, rubrics, judge prompt and parser, harness configuration, rollouts, rounds, candidates, the gate and its thresholds.

**Qualification criteria were fixed before they ran** (`QUALIFICATION_CRITERIA.md`, sha256 077889af…). Result:
1. Regression (the failing output, through the new flow): PASS. 6 of 6 rubrics have a verdict (4 were missing); the answered ones kept their first-pass verdicts; 3 were recovered and marked.
2. Agreement of the thinking-off judge with the existing verdicts on three outputs the judge had answered (23 rubrics): **FAIL**. Criterion level 65 of 77 = 84.4 percent (threshold 90); rubric level 19 of 22 = 86.4 percent (threshold 85, met); no leniency (50 criteria passed in each mode); one rubric got no reply at all in the repeat.
3. Determinism: PASS (3 of 3 recovered rubrics repeated the same verdict).
4. Unit tests: PASS (`tests/test_evalkit.py`, `tests/test_bench_eval_resume.py`).
Context measured afterwards (changes no verdict and no threshold): the same three outputs judged a second time by the unchanged thinking-on judge agree with their first verdicts on 23 of 23 rubrics and 77 of 81 criteria (95.1 percent), with every rubric answered. So the thinking-off mode differs from the normal judge by more than the judge differs from itself; the miss in criterion 2 is real, not noise.

**Status: the evaluator is NOT qualified under its own criteria. No run 2 is authorized and none was started.** The criteria are not changed after seeing this result.

## Amendment 3: the judge gets the context it needs; thinking-off recovery is disabled for G1 (2026-11-27, owner decision "Option 1", recorded BEFORE the repeat qualification)

Amendment 2's thinking-off recovery mode failed its qualification (84.4 percent criterion agreement against 90; the unchanged judge agrees with itself at 95.1 percent). The owner chose to remove the cause instead: the judge prompt for the failing output is 61,716 tokens and the server's context was 65,536.
* The judge server (`qwen-reasoning`, the same FP8 weights and quantization, vLLM) is restarted with `--max-model-len=98304` and nothing else changed (model native context 262,144; KV cost about 64 KiB per token, 16 full-attention layers). The evaluated agent's context budget stays 65,536 tokens, enforced by the harness (`BenchConfig.max_context_tokens`), not by the server; this is checked after the run from the recorded prompt sizes.
* The judge is the unchanged thinking-on judge: same model, prompt, rubric, parser, temperature 0, scoring. Its completion cap stays at the upstream 65,536 (the server clamps to the room left: 36,588 tokens for the largest prompt seen, against a need of roughly 1,000 to 4,000).
* `recovery_thinking_off` is **off** for G1 (code retained, default off). Durable per-rubric verdicts, the 3-pass bounded recovery of rubrics with no reply (same request), the sealed agent rollouts and the process-status verification stay.
* Unchanged: model, tasks, rubrics, harness configuration, rollouts, rounds, candidates, the gate and its thresholds, and the qualification thresholds (`QUALIFICATION_CRITERIA_v2.md`, sha256 fc8c1683..., same numbers, fixed before the server change, 2026-11-27T13:40Z).
* If the repeat qualification fails, G1 run 2 does not start. If it passes, one run starts; whatever it yields against the original gate is accepted.
