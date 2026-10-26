# Execution supervision, completion control and image input

Who owns what in a run (`run_loop`), after the cross-repository audit of 2026-10-25. Every row is a contract that changed; tests are named in `tests/`.

| concern | owner | where | contract |
|---|---|---|---|
| dispatch, the message list, stop reasons | agent-harness loop | `loop.py` | `run_loop(..., supervision=Supervision(), check_integrity=False, keep_images=2)`; stops: `done`, `verified_complete`, `blocked`, `max_turns`, `max_seconds`, `model_error`, `context_overflow`, `integrity_error` |
| repeated operations, stalls, recovery, budget notices, completion checkpoint, verified-complete ending | **`Supervisor`** (one authority; replaces the `[repeat]` guard, the consecutive-failure note, `progress.py`) | `supervisor.py` | `before()` / `after()` / `on_finish_attempt()` / `stop_reason()`; config in `SupervisorConfig` |
| what the model can see of old content; archive; recovery reads | Context Runtime (`ContextManager`) | `context_runtime/compression/transcript.py` | `prepare`, `on_context_error`, `admit`, `tools` plus the optional `ContextAware` extension below |
| stand-in text for removed content | the manager that wrote it | `TranscriptCompactor.is_placeholder` | **new**: the loop asks `cm.is_placeholder(value)`; the harness no longer contains Context Runtime's stub formats |
| visible / clipped / archived / unavailable result | the manager | `TranscriptCompactor.visibility(messages, tool_call_id)` returns `ResultView(state, archive_id, chars)` | **new**; for a clipped result archived later it names the id that holds the complete text |
| tool output size | the tool, against the budget the loop admits | `parity_registry(output_budget=...)` | **new**: pages, bash output and hints fit `max_tool_chars`, so `admit` is only a safety net and never cuts a page or its continuation hint |
| deliverable structure | `completion.py` | `check_deliverables`, `expected_artifacts` | statuses `none < draft_only < invalid < partial < structurally_valid`; never "correct" |
| image input | `multimodal.py`, tool `view_image` | `tools_parity.py` | capability is configuration or a measured probe, never assumed; blank/faint images are reported, not attached; images arrive in a `user` message after the tool results |

## The supervisor in one page
* **Families**: same tool; shell commands at least 0.90 alike after digits/whitespace are normalised; web tools by URL and offset; file tools by exact arguments.
* **Progress** (resets a family's stall): output-directory content changed; a file the operation refers to changed by ANOTHER call; an error turned into a different error or into success; a new family ran. Not progress: scratch files the operation writes itself, identical bytes rewritten, hidden/input directories, background noise. The workspace is compared by content hash, attributed per call.
* **States**: `normal` -> one concise note (changing, successful results: 6th run) -> `suppressed` (identical result twice, or 10 changing runs) -> a probe run after a TTL (15 turns, doubling to 60) -> `normal` on relevant change. A different operation is never blocked.
* **Exhaustion** (ends the run as `blocked`): a suppressed family requested `exhaust_after` (6) times with no different operation between AND `exhaust_min_idle_turns` (30) turns without progress. `end_on_exhaustion=False` records it and keeps going (for controlled comparisons).
* **Messages**: the first two suppression answers per family are informative, later ones are one-line stubs; an archived or clipped earlier result is answered with a pointer to the archive (asked from the manager), mentioning `offset` only if the tool has it; an unavailable result is allowed to run again.
* **Budget**: notices at 50% and 80% of the budget while no valid deliverable exists, an urgent notice in the delivery reserve (`max(12 turns, 8%)` or 300 s), refusal of new web exploration inside the reserve until a valid deliverable exists. Notices say "structural check", never "success".
* **Completion**: a text-only reply with missing/invalid deliverables is turned back at most twice (budget permitting); the status stays truthful if it is accepted anyway. When the task text names its outputs and all pass the structural check, nothing changes for 12 turns, the run ends as `verified_complete` (a notice comes at 6).

## Known limits
Structural checks cannot judge correctness. The supervisor cannot tell pixel-peeping from polling except by the thresholds above. Exhaustion can end a run that would have recovered on its own (see the replay in the audit notes). Image support depends on the endpoint; OCR remains an alternative. Hash-based progress ignores files above 4 MB (size and mtime only).
