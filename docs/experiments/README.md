# SIM experiments

Each directory is one complete run, kept as produced: every round's manifest (written before evaluation) and outcome,
every arm's memory (`sim.json`) and state, every proposer and selector reply, and `results.json`. `ANALYSIS.md` in each
says what the run does and does not show. A run is reproduced with `python -m agent_harness.sim.experiment` and the
configuration recorded in its `results.json`; the model endpoint is not part of this repository.

- `sim-run1`: configuration `feedback=v1`, `record_rejected=false`. A negative result on the experiment's own design: the
  source run promoted nothing, so the transfer arms started from the seed with an empty memory.
