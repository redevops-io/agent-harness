# The next experiment: real office work, failure patterns that accumulate on their own

Status: design, written before any of it ran. Nothing here has been measured. Runs 1 to 5 use two small synthetic policy-vetting sets labelled by this package's own policy code; they are smoke tests of the machinery. Their memories hold one problem each, so they cannot test selection among problems (see `sim-run3/ANALYSIS.md` and the run 5 notes).

## Workload

JobBench (35 occupations, working directory plus weighted rubric per task, LLM judge), run through `BenchEvaluator` (`agent_harness/sim/bench_eval.py`: `jobbench.run_task` in the Docker sandbox, `evalkit.judge_output`). It is real multi-source work (reconcile contradictory data, cross-reference records, trace citations) and its tasks differ enough that more than one kind of failure appears without anyone arranging it: wrong or missing deliverable files, unit and date handling, citation tracing, spreadsheet structure, stopping early.

Split: the seed-42 split in `docs/experiments/jobbench-seed42` (task lists copied from the working clone where they were made; not the paper's 31/34 split): `train.txt` (21 tasks) is the evolve set; `unseen.txt` (22) is the held-out set, evaluated in its two stages (`stage1.txt` once at the end of the whole experiment, `stage2.txt` only if stage 1 is read and reported first); `seen_during_development.txt` is excluded from every claim. Train and held-out are disjoint by task id (checked), but most occupations appear on both sides with different tasks, so occupation-level similarity carries over and the held-out score is not a score on unseen kinds of work; the analysis says so. The optimizer's evaluator holds only the evolve ids. Rubrics are never inside a harness tree or a workspace (`RUBRICS.json` is read by the judge outside the sandbox).

Legal-document, retrieval, QA and cross-app workloads follow the same way once each has an evaluator that grades a deliverable without the agent seeing the key: court_clerks and licensing_examiners (held-out here) are the nearest legal tasks in JobBench; the `redevops-qa` suites and the desktop's cross-app tasks need their own graders first. They are not part of this experiment because they have none yet.

## Decisions of 2026-11-26 (owner), and what follows from them

- **The judge is local.** The Qwen3.8-27B already serving locally scores the deliverables; no paid judge is used at this stage, and no budget is needed. The same model family acts and judges, so its agreement with an independent judge is measured first (below) and published next to every score.
- **Deterministic first.** The shipped JobBench rubrics are semantic (each asks whether a deliverable states or does something) and are scored by an LLM judge; none is machine-checkable. What is checked deterministically, and reported apart from the judge's score, is: a deliverable exists, is non-empty, opens as its file type and is within the size the judge can read. A task with no usable deliverable scores 0 without a judge call.
- **The judge is isolated from the optimizer.** The judge process receives the rubric file and the deliverables directory and nothing else: no harness tree, memory, prediction, rationale or proposal (`evalkit.judge_output` runs the upstream judge script as a subprocess with only those paths). The optimizer receives scores and failing-criterion texts as feedback, never the rubric file itself, and the held-out half is evaluated outside every loop.
- **Judge consistency is validated before any score is relied on.** Independent scores exist: the pre-registered four-harness experiment judged 95 stored outputs with Kimi k2.7-code at criterion level. The local judge re-scores a seeded, score-stratified sample of those stored outputs with the unchanged upstream judge (plan fixed in `run_validation.py` before it ran), 4 of them twice for test-retest. Reported: criterion-level agreement and Cohen's kappa with the independent verdicts, test-retest agreement, outputs the local judge could not grade (a deliverable beyond its 65,536-token context is reported as ungraded, not dropped), and the range of scores the sample covers (the stored outputs score 0 to 0.62 on Kimi's scale, so agreement on high-quality outputs is NOT tested by it). If agreement is poor, the experiment does not proceed on this judge.
- **Every run records its conditions**: model name and weights (`Qwen/Qwen3.8-27B-FP8`), serving software and version, context length, sampling settings for the actor, proposer, selector and judge, tokens in and out per client as the server reports them, and seconds per call.

## Gates, fixed before running

- **G0, pilot (2 tasks, 2 rollouts each):** wall-clock time and judge cost per evaluation. These numbers, not estimates, set rounds, candidates and budget. If one evolve pass over 21 tasks does not fit the budget, the evolve set is cut by stratum and the cut is recorded, never the held-out set.
- **G1, accumulation (the gate that decides whether selection is tested at all):** run the source optimization (full memory, no selection) on the evolve set until the budget for it is used. The selected-versus-complete comparison runs only if the record then holds at least 6 distinct problems and its rendered size exceeds the selected arm's token budget, fixed now at one half of that size. If the record is smaller or has fewer problems, the experiment reports exactly that and stops. No record is added, copied, split or invented to pass the gate.
- **G2, relevance (only if G1 passed):** a decision where relevance matters is one whose failing tasks bear on some problems and not others. The selector's choices are scored against the regression evidence in the record (did the shown problems include those whose edits broke a currently failing task), computed from the record, not from a model.

## Arms and measures (if G1 passes)

All arms start from the same evolved harness and the same source record and see the memory under the same condition; they differ only in what the proposer and selector read: complete record; selected view under the fixed budget; problem-only; no memory. At least 3 replicates per arm, arms interleaved, the same held-out tasks, the same judge.

Per arm and replicate: held-out score per task (not only the mean), evolve score, input and output tokens as reported by the server for proposer, selector, policy and judge separately, wall-clock seconds, the number of problems and attempts in the record against the text shown (parsed back from the prompt, as in `experiment_selected.history_report`), `BudgetTooSmall` events, and the number of proposed edits that repeat a direction the record already shows as refuted.

## Analysis

Pre-registered: the primary comparison is per-task paired differences on the held-out set, complete against selected, with all replicates shown, not a best run. No difference smaller than the replicate-to-replicate spread is reported as a difference. Negative results (selection worse, `BudgetTooSmall`, a gate failing, judge disagreement) are published with the same prominence as positive ones, with `RUN_NOTES.json` stating comparability.

## Budget (owner decision, not taken here)

Local inference needs no purchase. A paid judge, a second paid acting model or any cross-model run needs a stated spend limit; the pilot (G0) produces the numbers to decide it with. Until then the experiment can proceed with a local judge and its measured agreement.

## What a positive result would still not show

One benchmark family; one acting model; an LLM judge with its own error; a selector that is lexical and structural, so it says nothing about learned selection; and, whatever the numbers, no evidence of recursive self-improvement beyond selecting and applying harness edits under evaluation.
