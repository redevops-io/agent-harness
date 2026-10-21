# Google's Harness² (adapted for the experiment)

`agent_harness/bench/harnesses/harness2.py`: [google-research/harness2](https://github.com/google-research/harness2)
(Apache-2.0; code and paper by Google researchers, [arXiv 2610.14224](https://arxiv.org/abs/2610.14224)) as one arm of the
harness x precision x thinking experiment ([bench.md](bench.md)). Upstream's pipeline runs **unmodified**; nothing is vendored.

## What runs
Per task, with `--k 1 --mode parallel` on the OpenCode substrate: a reference rollout with the starting harness (`base1`); an
editor writes an *invariant* and a *domain* candidate harness, each probed by a rollout (`inv1`, `dom1`); a composer merges
them; the composed harness runs once more (`agg1`) and a reflector stores lessons in a per-domain library. At k=1 selection is
skipped, so **`agg1` is the Harness² result**. That is 4 executor rollouts plus 3-4 editor sessions per task. The editor, composer
and reflector use a fixed model that is the same in every cell, so only the executor varies between cells.

## What is pinned
Upstream commit `1cfe254e8d25b15f37048c3730134686abe60335` (3 commits on `main`, no release tag when reviewed), OpenCode
`23fb5e0516c99ac04a1aa46c193efda2e1b9bb24` with upstream's own patch, set up with `harness2-setup jb`. Requires Python 3.12-3.13;
macOS also needs GNU `timeout` (`brew install coreutils`).

## Adaptations (routing and bookkeeping only)
Executor and editor traffic go through the spend proxy (`OPENCODE_CONFIG_CONTENT`); the proxy label `think` switches the
executor's reasoning on; web goes through the egress filter; each cell gets its own runs directory, so libraries and state are
never shared; the splits file is written from the experiment's task list; scoring happens afterwards, outside the pipeline.

## Differences from the paper (this is NOT a reproduction)
| Paper | Here |
|---|---|
| Executors Gemini 3.5 Flash and GLM-5 | Qwen3.8-27B at two precisions (FP8, NVFP4), thinking off and on |
| Editor and selector Claude Opus 4.8 | a local model, fixed across cells |
| Judges Claude Opus 4.8 / Sonnet 4.6 | a local judge that must pass a calibration first |
| JobBench 31/34 split, 3 runs | a stratified seed-42 split (21/44); one run per task |

## Recorded per rollout
Peak prompt size (so a report can show whether the 65,536-token window held), the label of the rollout, the pipeline status.
With `h2-base` in the experiment's arms, the **reference rollout** is recorded too: a second, independent sample of the
stock OpenCode harness on the same tasks, which gives a run-to-run noise estimate for free.

## Cost warning
About 4x the executor time of a stock run, plus editor sessions. A task whose refined rollout produces nothing is *finished with
no output* and scored 0 by the judge step, as the method's own failure.
