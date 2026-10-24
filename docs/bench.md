# Benchmark runner and evaluation checks

`python -m agent_harness.bench.jobbench` runs [JobBench](https://github.com/Job-Bench/job-bench-eval) tasks with this
package's own executor (chat client, tool loop, Docker sandbox with no network, context management) and scores the
deliverables with JobBench's own judge script, so results use the same rubrics as other runners. It is not a
reproduction of any published result.

## Executor
* Tools: `bash` (inside the sandbox), confined `read_file` / `write_file` / `list_dir`, and `web_fetch` (host side; refuses
  huggingface.co, hf.co, github.com, raw.githubusercontent.com; gists allowed). Plus `search_context` / `read_archive`
  from the context manager ([context.md](context.md)).
* Limits per attempt: turns, wall clock, context window. A second attempt runs only if the first produced no deliverables.
* Guards: grader files (`RUBRICS.json`, `rubric*.json`, `eval_result`, `*_judge.json`) in the agent's workspace abort the run;
  six consecutive failures of one tool add a note; the same call with the same result three times in a row is not re-run.
  an action carrying a context-manager placeholder is refused ([context.md](context.md#keeping-actions-and-instructions-intact)).
* Options of `run_task`: `pin_instructions` (default on) keeps the task statement out of compaction; `trace` (default off) records
  what the model was shown at every call.

## Evaluation rules (`bench/evalkit.py`; each exists because the opposite happened)
* A judge failure (rate limit, 5xx, timeout, unparsable reply) is **ungraded**, never a score of 0; infrastructure failures
  are retried with backoff and a stale details file is deleted before every attempt.
* "No output" is 0 only for a rollout that **finished** with no deliverables. An unfinished or missing rollout cannot be
  judged at all (`NotReady`).
* The scoreboard is append-only; paired statistics use only tasks graded in both arms and report how many were dropped.

## Preflight: `python -m agent_harness.bench.selfcheck`
Nothing in it calls a paid API. It checks, in order: sandbox isolation (no network, non-root, libraries present); a scripted
end-to-end run (real sandbox and context manager; a scripted model forces compaction and must recover archived content
through `search_context`); each model endpoint (answers, served window, no hidden reasoning); judge controls (irrelevant text
scores low, empty output is a zero without a judge call); and judge calibration against labelled outputs (rank correlation
must reach 0.7). Exit code is non-zero if anything fails.

Local judges can be slow: a single JobBench output can produce a prompt of >100k tokens, sent once per rubric. Use
`--judge-workers 1` (so the server's prefix cache is reused) and a large `--judge-rubric-timeout`.

## Spend and providers
See [spend.md](spend.md). The price table also lists self-hosted endpoints as free providers (`local`, `proxmox48`,
`proxmox32`); a run with no `--cap` arguments has no paid provider at all.
