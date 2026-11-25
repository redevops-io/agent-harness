# Self-Improvement Memory and the harness optimization loop

Specification: [specs/SELF_IMPROVEMENT_MEMORY_SPEC.md](specs/SELF_IMPROVEMENT_MEMORY_SPEC.md). Method: "Self-Improvement
Memory Enables Continual Recursive Self-Improvement" (arXiv:2611.18856), sections 3.1-3.3 and appendix D.

`agent_harness.sim` is memory **of the harness optimizer**: which problem an edit addressed, why it was made, which tasks it
was predicted to fix, and what was measured. It is separate from conversational memory and from the memory an agent uses
to solve tasks. A bandit that learns to pick one of several existing strategies is not this, and nothing here calls it
recursive self-improvement.

## What exists

| Paper | Here | Test |
|---|---|---|
| 3.1 a round: evaluate on the evolve set, propose candidates, screen, evaluate, select or keep | `sim/loop.py` `HarnessOptimizer.step` | `tests/sim/test_loop.py` |
| 3.1 only evolve-set feedback is available to the loop; held-out stays outside | constructor refuses an overlap; every evaluator call is checked; `evaluate_heldout` is a free function | `test_held_out_tasks_cannot_enter_the_loop` |
| 3.2 entries grouped by root-cause hypothesis; attempts with edit, rationale, predicted and observed outcome, provenance | `sim/memory.py` `Problem`, `Attempt`, `Provenance` | `tests/sim/test_memory.py` |
| 3.2 predictions recorded before evaluation, checked afterwards | the round's change manifest is written before any candidate runs; `fixed`/`missed` are computed from the prediction on record | `test_the_change_manifest_...`, `test_fixed_and_missed_...` |
| 3.2 only the chosen candidate is recorded, including when it regresses or misses | `step` | `test_a_missed_prediction_and_a_regression_...` |
| 3.2 no accepted edit: harness and memory unchanged | `step` | `test_no_accepted_candidate_...` |
| 3.2 / D.1 identical root-cause text is one entry; merges keep all attempts; aliases | `open_problem`, `merge`, `merge_identical` | `test_identical_root_cause_text_...`, `test_there_is_no_way_to_delete_history_...` |
| 3.2 proposer and selector both read the complete record, no retrieval, no truncation | `render(FULL)` given to both ports | `test_the_proposer_and_the_selector_both_receive_...` |
| 3.3 / C.3 four conditions: cold start, harness transfer, problem-only, full SIM | `TransferCondition`, `start_target_run` | same test |
| D.3 record format (`next_id`, `carried_from`, `aliases`, `problems`, attempt fields) | `to_json` / `from_json`, one JSON file per run | `test_a_saved_record_round_trips_...` |

Beyond the paper, required by the specification: the optimizer cannot edit protected harness files (guardrails, hooks,
plugins) or anything outside the harness surface; a candidate scoring below the current harness is not promoted unless the
run allows it; every promoted harness is a hash-addressed artifact with its parent, and a rollback is recorded and removes
nothing from the memory.

## What does not exist yet

- **No LLM proposer or selector is included.** They are ports; the tests use scripts. The prompt guidance of appendix D.2 is not written.
- **No experiment has been run.** No comparison of the conditions, no number, on any model or benchmark. The five-arm
  benchmark of the specification (the four conditions and the existing learning implementation) is not built.
- **No task evaluator is wired in.** `Evaluator` is a port; connecting it to `agent_harness.bench` (isolated runs, the
  calibrated judge) is the next step.
- Dashboards, memory selection through Context Runtime, and continuation across model generations are not started.
