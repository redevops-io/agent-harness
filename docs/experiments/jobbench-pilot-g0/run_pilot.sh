#!/bin/sh
# G0 pilot of NEXT_EXPERIMENT.md: 2 train tasks x 2 rollouts, local Qwen3.8-27B FP8 acting and judging (judge validated in docs/experiments/jobbench-judge-validation).
cd $HOME/qualification/public-review/src/agent-harness
for rep in 1 2; do
  env -i PATH=/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:$HOME/.local/bin HOME=$HOME PYTHONPATH=$HOME/qualification/public-review/src/agent-harness:$HOME/qualification/ctx-runtime-3b55bec python3 -m agent_harness.bench.jobbench \
    --tasks-base $HOME/projects/harness2-upstream/dependencies/job-bench-eval/dataset/main --task data_entry_keyers/task2 --task technical_writers/task2 --label pilot-rep$rep \
    --base-url http://192.168.40.105:30807/v1 --model qwen3.8-27b --out $HOME/qualification/jobbench/pilot-g0/runs --no-thinking \
    --temperature 0.55 --max-tokens 16384 --context 49152 --max-turns 200 --max-seconds 3600 --attempts 2 --no-web \
    --judge-python $HOME/projects/harness2-upstream/dependencies/job-bench-eval/.venv/bin/python --judge-script $HOME/projects/harness2-upstream/dependencies/job-bench-eval/eval/judge.py --judge-model qwen3.8-27b --judge-base-url http://192.168.40.105:30807/v1
done
echo PILOT-DONE
