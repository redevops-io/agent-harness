# Context management

`agent_harness.loop.run_loop` keeps a conversation inside the model's window through a `ContextManager`
(`agent_harness/context.py`). The loop calls it at four points:

| Hook | When | Purpose |
|---|---|---|
| `admit(tool, text, max_chars)` | every tool result, before it enters the conversation | return the text to append; a storing manager keeps the full text |
| `prepare(messages, target_tokens, estimated_tokens)` | the next prompt is estimated to pass `compact_at` of the window | shrink the conversation toward `target_tokens` |
| `on_context_error(messages, attempt)` | the server rejected the prompt as too long (1, 2, 3) | compact harder; the loop retries |
| `tools()` | once, at the start | extra read-only tools for the model, such as retrieval |

## Two implementations

* `BasicContextManager` (default, in this package, no dependencies): clips large outputs, replaces old tool
  outputs with a stub, shortens old tool-call arguments. **Lossy**: what it removes is gone.
* `ContextRuntimeManager` (in [context-runtime](https://github.com/redevops-io/context-runtime),
  `context_runtime.integrations.agent_harness`): **lossless**. Anything it removes is first written to an archive
  (one file per item, indexed). The stub carries the id, the size and the first characters, and the model gets
  `search_context(query)` and `read_archive(id, offset)` to get the content back. Offline BM25 by default;
  `use_redevops_rag=True` backs retrieval with [redevops-rag](https://github.com/redevops-io/redevops-rag)
  (hybrid vector + BM25).

The benchmark runner (`python -m agent_harness.bench.jobbench`) uses `ContextRuntimeManager` by default
(`--context-manager basic` opts out; `--rag` selects redevops-rag). If context-runtime is not installed the
runner stops with an error instead of falling back, so a run cannot silently use the lossy manager.

## Keeping actions and instructions intact

* **Placeholder guard.** `run_loop` refuses to run an action whose string argument is, or contains, text a manager put in
  the history in place of removed content. The model receives an error saying nothing was run. A model was seen to copy
  such a stub into a new `write_file`; the file then held the stub. What a stub looks like is the manager's business: the
  loop asks `cm.is_placeholder(value)` (the optional `ContextAware` extension), so Context Runtime's own formats
  (`[archived A0007: ...`, `<<ARCHIVED-PLACEHOLDER ...`) are recognised by Context Runtime and a manager that adds a new
  format only has to report it there. `placeholder_in` / `placeholder_arg` in `context.py` cover only the markers this
  package writes itself (`... chars omitted to save context]` and the elision notice).
* **Pins.** `BasicContextManager.pin_text(text)` keeps any tool result that contains the start of `text` (whitespace
  normalised) out of `prepare`'s elision. The last-resort clipping at `on_context_error` attempt 3 still applies, so a
  pin cannot make a run fail. The benchmark runner pins the task's `TASK_INSTRUCTIONS*` file (`run_task(pin_instructions=True)`,
  the default), because the task statement is read from a file and is not part of the pinned prompt.
* **Archive ids.** The sandbox `read_file` answers an archive-id-shaped path (`A0007`) with an error that names
  `read_archive`, instead of a missing-file error.
* **Run trace (opt-in).** `run_loop(..., trace=TraceRecorder(dir))`, or `run_task(trace=True)`, writes `trace.jsonl` and a
  content-addressed `blobs/` folder: the canonical text of every message as it was appended, the projection sent at
  each model call (hash and length shown versus canonical, which messages a manager rewrote, which archive ids they
  name), and the full text of tool results, tool-call arguments and assistant replies. It changes nothing in the run; a
  write failure disables tracing with one message and never ends the run.

## Dependency direction

`context-runtime` depends on `agent-harness` (its tool contract and approval gate are re-exported from
`agent_harness.tools`, replacing a copy). `agent-harness` never imports `context-runtime`, `redevops-rag` or any
other ReDevOps package; the integration is a structural interface (`ContextManager`) implemented on the other
side. That keeps the dependency graph acyclic.

## What is and is not established

Unit tests cover losslessness (every archived item is recoverable byte for byte), stub and argument validity,
recovery through `search_context`, and the loop integration. The redevops-rag backend has an opt-in recall check
(`CR_TEST_RAG=1` in context-runtime). Whether lossless compaction improves **benchmark scores** has not been
measured yet. The guard, pins, archive-id error and trace are covered by `tests/test_context_correctness.py`
(including that a traced run produces the same conversation as an untraced one).
