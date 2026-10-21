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

## Dependency direction

`context-runtime` depends on `agent-harness` (its tool contract and approval gate are re-exported from
`agent_harness.tools`, replacing a copy). `agent-harness` never imports `context-runtime`, `redevops-rag` or any
other ReDevOps package; the integration is a structural interface (`ContextManager`) implemented on the other
side. That keeps the dependency graph acyclic.

## What is and is not established

Unit tests cover losslessness (every archived item is recoverable byte for byte), stub and argument validity,
recovery through `search_context`, and the loop integration. The redevops-rag backend has an opt-in recall check
(`CR_TEST_RAG=1` in context-runtime). Whether lossless compaction improves **benchmark scores** has not been
measured yet.
