# Configuration

Environment variables:

- `OPENAI_BASE_URL` – base URL for OpenAI-compatible endpoint
- `OPENAI_API_KEY` – API key or provider token
- `MODEL` – model identifier

Any OpenAI-compatible endpoint (llama.cpp / vLLM / Ollama / a cloud provider) is reached through
`OPENAI_BASE_URL` + `MODEL`; the harness core reads no other endpoint env vars.

Approval is configured **in code**, not via an env var — pass `ApprovalPolicy(mode=...)` when you
construct the policy. Modes:

- `allowlist` – only tools/commands on the allowlist are allowed
- `edits-only` – allow edits, gate other actions
- `bypass` – allow everything (use with care)
- `never` – deny all (available on the `Approver` policy)
