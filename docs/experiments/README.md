# SIM experiments

Each directory is one complete run, kept as produced: every round's manifest (written before evaluation) and outcome,
every arm's memory (`sim.json`) and state, every proposer and selector reply, and `results.json`. `ANALYSIS.md` in each
says what the run does and does not show. A run is reproduced with `python -m agent_harness.sim.experiment` and the
configuration recorded in its `results.json`; the model endpoint is not part of this repository.

- `sim-run1`: configuration `feedback=v1`, `record_rejected=false`. A negative result on the experiment's own design: the
  source run promoted nothing, so the transfer arms started from the seed with an empty memory.

## RUN_NOTES.json

Every run directory carries `RUN_NOTES.json` (`agent-harness.sim-run-notes/1`): `comparable` (false means no tool may put this
run's arms side by side), `caveat` (the exact sentence to show with the run), `reasons`, and the `configuration` used. It is
written by the person who read the run, after the run, and the experiment code never reads it.
- `sim-run2`: configuration `feedback=v2`, `record_rejected=true`, 6 rounds. The source run again accepted nothing, so it is a comparison of the memory shown, not of transfer. The two arms shown the failed source run's memory accepted no edit; the two arms without memory improved. See its `ANALYSIS.md` for what that does and does not show.
- `sim-run3`: configuration `feedback=v2`, `record_rejected=true`, 6 rounds, with the pair reversed (egress-vetting source, command-vetting target). The source run accepted 2 edits, so the transfer arms start from a real evolved harness and a non-empty memory. The arm without memory improved most on the target; the two arms with memory kept more of the source task. One run per arm, and the evolve set saturated in two arms. See its `ANALYSIS.md` and `PLAN_run3.md`.
- `sim-run4`: an unchanged replicate of run 3. The ordering of the arms in run 3 was not reproduced (all optimized arms end at 0.93 to 0.97; the same arm differs between runs by up to 0.43), and the evaluator is not deterministic at temperature 0. See its `ANALYSIS.md`.
