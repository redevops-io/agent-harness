# Architecture

- LLM client: a minimal OpenAI-compatible client stub — real providers/fallbacks are supplied by the agent layer that wires this harness
- Tools (`agent_harness.tools`): the `ToolSpec` / `ToolPlugin` / `ToolResult` contract, a `ToolRegistry` that exports OpenAI-style specs, an audit log, an optional authorizer, and the tool-level `ApprovalPolicy`. Same contract as context-runtime's `context_runtime.tools.base`; tool arguments are not schema-validated at call time
- Command policy (`agent_harness.command_policy`): vets shell commands and edits for the Agent's built-in tools — modes `allowlist` / `edits-only` / `bypass` / `never`
- Sandbox: a lightweight restricted-execution wrapper (documented as *not* full isolation)
- Guardrails: input/output filtering, redaction, and a loop guard
- Eval harness: reproducible test runs (no invented benchmarks)
- Agent loop: observe → plan → act → observe (the intended design; the shipped loop is a minimal single-step stub)

The core remains provider-agnostic; concrete targets (llama.cpp/ollama, vLLM, Moonshot/Kimi, xAI/Grok, OpenAI) are configured via environment variables.
