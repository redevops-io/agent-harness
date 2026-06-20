# Architecture

- LLM client: unified OpenAI-compatible interface with provider fallbacks
- Tool registry: register and invoke tools with schema validation
- Approval: human-in-the-loop modes (always, on-risk, never)
- Sandbox: restricted execution environment for tool actions
- Guardrails: input/output filtering and policy enforcement
- Eval harness: reproducible test runs (no invented benchmarks)
- Agent loop: observe → plan → act → observe (consumes the above)

The core remains provider-agnostic; concrete targets (llama.cpp/ollama, vLLM, Moonshot/Kimi, xAI/Grok, OpenAI) are configured via environment variables.
