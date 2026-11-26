# G0 pilot of the JobBench experiment (2026-11-26): what one evaluation costs

Setup (all local, no paid service): actor and judge Qwen3.8-27B FP8 (`Qwen/Qwen3.8-27B-FP8`, vLLM 0.34.0, 65,536-token context), thinking off, temperature 0.55, 16,384 output tokens, context manager Context Runtime (commit 3b55bec, loaded from the pinned checkout), sandbox `agent-harness-office:1`, no web, up to 200 turns / 3,600 s, 2 attempts.
Judge: the upstream judge script with the same local model (agreement with an independent judge validated: `docs/experiments/jobbench-judge-validation`). Two train tasks, two rollouts each.

| Task | Rollout | Turns | Rollout seconds | Prompt tokens | Completion tokens | Judge score (rubrics passed of 8) |
|---|---|---|---|---|---|---|
| data_entry_keyers/task2 | 1 | 26 | 532 | 567,637 | 18,939 | 0.60 (5) |
| data_entry_keyers/task2 | 2 | 20 | 297 | 463,387 | 10,445 | 0.60 (5) |
| technical_writers/task2 | 1 | 15 | 442 | 245,027 | 16,233 | 0.10 (1) |
| technical_writers/task2 | 2 | 16 | 611 | 259,231 | 22,437 | 0.37 (3) |

What it says:
- **A rollout takes 5 to 10 minutes and 250,000 to 570,000 prompt tokens** (completion 10,000 to 22,000); judging one output took 4.5 to 13 minutes in the judge validation. About 15 to 25 minutes per task evaluated, on one shared GPU.
- **The same task and settings scored 0.10 and 0.37 (technical_writers) and 0.60 twice (data_entry)**: one rollout of one task is a noisy measurement. A 21-task evolve set measured once per candidate would carry that noise into every accept decision.
- **Budget implication (the gate this pilot exists for):** one pass over the 21 train tasks is roughly 5 to 9 hours sequentially. The design's 6 rounds of 2 candidates (13 passes plus the baseline) would be 60 to 120 hours. That does not fit; the evolve set has to be cut by stratum (the cut is recorded, never the held-out set), candidates per round reduced, tasks run concurrently (the server's capacity is the limit and has not been measured under concurrent load), or repeated rollouts per task used to control the noise. This pilot does not decide which; it fixes the numbers to decide with.
- Not measured: concurrency throughput, judge time inside this pilot (not separately recorded), and any effect of the harness edits (nothing was optimized).
