# Architecture

- LLM client: a minimal OpenAI-compatible client stub — real providers/fallbacks are supplied by the agent layer that wires this harness
- Tools (`agent_harness.tools`): the `ToolSpec` / `ToolPlugin` / `ToolResult` contract, a `ToolRegistry` that exports OpenAI-style specs, an audit log, an optional authorizer, and the tool-level `ApprovalPolicy`. Same contract as context-runtime's `context_runtime.tools.base`; tool arguments are not schema-validated at call time
- Command policy (`agent_harness.command_policy`): vets shell commands and edits for the Agent's built-in tools — modes `allowlist` / `edits-only` / `bypass` / `never`
- Sandbox: a lightweight restricted-execution wrapper (documented as *not* full isolation)
- Guardrails: input/output filtering, redaction, and a loop guard
- Eval harness: reproducible test runs (no invented benchmarks)
- Agent loop: `agent_harness.loop.run_loop` is a multi-turn tool-calling loop. It sends the conversation to a `ChatClient`, runs requested tools through a `ToolRegistry` (approval gate, authorizer and audit log apply), and repeats until the model answers without tool calls or a limit is hit (`max_turns`, `max_seconds`, `max_context_tokens`; old tool outputs are elided when the prompt grows), with an optional JSONL trajectory and context manager (`agent_harness.context`). `Agent.run` in `agent_harness.agent` is a separate, minimal single-step stub that only finishes
- QA module (`agent_harness.qa`, optional, Python 3.11+): neutral records, ports, `canon-v1` hashing, a deterministic verifier and classifier, a bounded single-run executor, a fail-closed action policy, recovery, an egress policy, redaction and evidence integrity checks. It imports the standard library and harness core only and is never imported by `agent_harness/__init__.py`. See [qa.md](qa.md)

The core remains provider-agnostic; concrete targets (llama.cpp/ollama, vLLM, Moonshot/Kimi, xAI/Grok, OpenAI) are configured via environment variables.
