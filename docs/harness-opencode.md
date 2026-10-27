# OpenCode stock harness (adapted for the experiment)

`agent_harness/bench/harnesses/opencode.py`: the **unrefined** OpenCode agent as one arm of the harness x precision x thinking
experiment ([bench.md](bench.md)). It uses upstream [Harness²](https://github.com/google-research/harness2)'s JobBench adapter on
its OpenCode substrate with an empty starting harness ("bare stock OpenCode"); none of Harness²'s refinement runs.

## What is pinned
* Harness² upstream commit `1cfe254e8d25b15f37048c3730134686abe60335` (the package must be importable and its checkout is
  `harness2_root` in the experiment config). Its own setup pins and patches OpenCode (`anomalyco/opencode` @ `23fb5e0516c99ac04a1aa46c193efda2e1b9bb24`
  plus upstream's `opencode-compat.diff`); run `harness2-setup jb` once. Nothing is vendored here.
* macOS needs GNU `timeout` (`brew install coreutils`, put `gnubin` on PATH); upstream assumes Linux.

## What is adapted (routing and bookkeeping only)
| Adaptation | Why |
|---|---|
| Providers declared in `OPENCODE_CONFIG_CONTENT`, base URL `<proxy>/<provider>/<label>/v1`, dummy key | all model traffic goes through the spend proxy; real keys and addresses stay out of the subprocess |
| Proxy label `oc`, or `think` for thinking cells | the label is the only difference between a thinking and a non-thinking cell (the proxy sets `enable_thinking`) |
| `limit.context` = the server's window, `limit.output` = the experiment's common output ceiling | OpenCode plans its own compaction from these |
| `HTTP(S)_PROXY` through the egress filter | benchmark hosts (huggingface.co, github.com, ...) blocked; other web allowed |
| One child process per task | a hung rollout cannot stall the run |
| Non-zero exit after time and retries -> *finished with no output*; any other failure -> *crash* | a crash is never judged and never scored 0 |

## Diagnostics recorded per rollout
Peak prompt size (`input + cache read + cache write` of the largest step) and step count, so a report can show whether the
context window held without relying on OpenCode's own compaction being invisible.

## Not claimed
OpenCode manages its own context; this adapter does not add context-runtime to it by design. Results are for this model,
these endpoints and these tasks only.
