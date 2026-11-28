# G1 run 2: what happened, what is recoverable, and whether it can be resumed

Run 2 (2026-11-27 to 2026-11-28, amendments 2 and 3) ended with exit 1 after 19.2 hours. Its artifacts are preserved read-only, with a checksum manifest of 9,377 files, in
`~/qualification/jobbench/g1-run2-failed-evaluation-2026-11-28/` (run 1's are in `g1-run1-failed-evaluation-2026-11-26/`). Nothing below changes a score, a threshold or the protocol.

## The failure

`EvaluationIncomplete` in the fourth evaluation (round 2, the only screened-in candidate): `mechanical_engineering_technicians/task2`, rollout 2, 29 criteria with no judge reply.
The judge's own log shows why, for all 8 rubrics on both recovery passes: HTTP 400, "maximum context length is 98304 tokens ... your prompt contains at least 98305 input tokens".

That "at least" is the server's lower bound, not a near miss. Measured with the server's tokenizer, the extracted text of that deliverable alone is **144,169 tokens** (the upstream judge caps a file
at 200,000 characters, and the 286 KB report reaches that cap), before the rubric and the two attached images. It is 46,000 or more tokens too large for the qualified judge.
So this is not run 1's failure. Run 1's prompt fitted (61,716 of 65,536) and the reply was cut off; amendment 3 removed that cause and it did not recur. Run 2's prompt cannot be sent at all,
the same request is the same answer on every pass, and the three bounded passes only added about 80 seconds of waiting.

## What exists and what does not

| | state |
|---|---|
| Round 1 (baseline plus two candidates, 48 rollouts, all judged) | complete and valid under the qualified evaluator; its outcome and the record (2 problems, 1,363 tokens) are written |
| Round 2, candidate 1 (eval-0004): 16 rollouts | all 16 finished and sealed; 14 judged (one had no deliverable, one is the oversize deliverable above) |
| That candidate's harness text (the edit to `systemprompt.md`) | **not stored anywhere.** The round manifest holds the file names, the rationale and the predictions; the edit text lived in memory; the proposer's replies are written only when a run ends normally |
| Round 2 candidate 2 | screened out by the selector |
| Rounds 3 and 4 | not started |

## Can run 2 be resumed validly? No.

1. The candidate whose 16 rollouts exist cannot be reproduced: its edit text is gone. The sealed rollouts cannot be attached to a harness, so they cannot be reused and the candidate cannot be evaluated again as itself.
2. Drawing round 2's proposals again is a different experiment, and not a neutral one: the reason the candidate stopped the run is a property of what it made (a deliverable the judge cannot read), so replacing it
   is selection after seeing an outcome.
3. Even with the text, the oversize deliverable has no verdict under the qualified judge, and the evaluator may not invent one.
4. The loop could not restart from disk at all (no state restore, no stored proposals, evaluation folders named by a counter that restarts at 1 and would have overwritten round 1's rollouts).

Run 2 therefore stays what it is: an incomplete run, a negative result about the evaluator (not about SIM), with round 1 as measured. A new run is required. This was established without any GPU use beyond the
two checks below.

## What was fixed (mechanics only; model, tasks, rubrics, judge prompt and parser, rollouts, rounds, candidates, the gate and its thresholds are unchanged)

* **Overflow is its own class.** A judge prompt larger than the context is recognised from the server's message, is not retried, and is reported with the limit and the prompt size
  (`JudgeResult.context_exceeded`, `EvaluationIncomplete.kind == "judge_context_exceeded"`). Real judge, real server, run 2's deliverable: recognised after 1 pass in 9.8 s (was 3 passes).
* **A partial judgment is never a grade.** The upstream judge writes each rubric as it finishes, with totals over what it finished. A judge process that died part way (killed, out of memory) left such a file
  and `classify_details` would have graded it. It now requires every rubric (`total_count`) to have a verdict. Found while writing the interruption tests; reproduced red before the fix.
* **Checkpoints and restart.** Evaluation folders are named by the content they answer (harness text and files, tasks, settings), so a restarted process finds its finished rollouts and judged rubrics and never
  lands in another harness's folder. A rollout that raises no longer abandons its siblings; every evaluation writes `evaluation.json` (what was settled, what is ungraded and why, what it consumed) whether it finished or not.
* **The optimizer can restart.** Each round's manifest now stores the proposals in full before any candidate runs; `HarnessOptimizer.resume` restores the stored harness, the memory and the completed rounds, and an
  interrupted round continues with exactly its stored proposals (the proposer and the screening are not asked again). A manifest without stored proposals (run 2's) is refused, not re-drawn.
  `experiment_jobbench --resume` uses it and runs only the rounds that remain.
* **Tokens are recorded although the cost is 0.0.** Actor tokens come from the rollout summaries; the judge's from a per-request log written by the same wrapper as the thinking-off recovery, run with thinking left
  as it is (`AGENT_HARNESS_JUDGE_THINKING_OFF=0`: no request, prompt or parser differs); the proposer's and selector's from a persistent meter. A run that stops still writes `results.json` with
  `status: incomplete`, the kind, the usage and the gate marked not assessable.
  **Run 2, counted from its artifacts:** 64 rollouts, 151,261,748 actor prompt tokens and 2,568,831 actor completion tokens. Its judge tokens were not recorded (the judge was not metered); the proposer's and selector's
  were in memory only. Those three numbers cannot be recovered and are reported as missing, not estimated.

Verified: 19 new tests (`tests/test_g1_recovery.py`, `tests/sim/test_resume.py`), each shown red without its fix (partial judgment, overflow classification); full suite 3,826 passed. On copies of run 2's artifacts with the real
judge and server: a finished judgment makes **0** requests; with two rubrics removed it makes exactly **2** (59,120 and 59,197 prompt tokens), the two verdicts come back identical to the originals and the score is unchanged (0.32).

## The gate and the budget

The gate (`NEXT_EXPERIMENT.md`, unchanged): at least 6 distinct problems, **and** a record whose rendered size exceeds the selected arm's token budget, "fixed now at one half of that size".
Read literally the second condition is `size > size / 2`, which is true of every non-empty record, so as written it cannot fail, and `experiment_jobbench.gate()` implements exactly that (`tokens > 0`). Only the
problem count binds. This is now stated in the gate's own output (`size_condition.can_fail: false`) so nobody reads the size test as a result.

The selected-memory runner (run 5) used an absolute budget of 3,000 tokens (`--budget-tokens`). Run 2's record is 1,363 tokens, so under that budget the record does not exceed it and selection would have had nothing to
select from: with 2 of 6 problems and 1,363 of 3,000 tokens, the gate is not met on either reading. The owner decides which reading is the experiment's (see amendment 4). Neither reading was changed or applied here.

## What a new run needs before any GPU window

Owner decisions (recorded in `G1_PLAN.md`, amendment 4, proposed and not in force): the rule for a deliverable whose judge prompt cannot fit, and which budget reading is the gate. Then one run of the same protocol.
A new run is not expected to avoid this by luck: one deliverable in 64 was oversize, and a run has up to 144.
