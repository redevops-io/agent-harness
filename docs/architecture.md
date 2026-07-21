# Architecture

- LLM client: a minimal OpenAI-compatible client stub — real providers/fallbacks are supplied by the agent layer that wires this harness
- Tool registry: register tools and export their OpenAI-style schemas (schemas are stored for export, not validated at call time)
- Approval: policy-based allow/deny — modes `allowlist` / `edits-only` / `bypass` (+ `never` on the `Approver` policy)
- Sandbox: a lightweight restricted-execution wrapper (documented as *not* full isolation)
- Guardrails: input/output filtering, redaction, and a loop guard
- Eval harness: reproducible test runs (no invented benchmarks)
- Agent loop: observe → plan → act → observe (the intended design; the shipped loop is a minimal single-step stub)

The core remains provider-agnostic; concrete targets (llama.cpp/ollama, vLLM, Moonshot/Kimi, xAI/Grok, OpenAI) are configured via environment variables.
