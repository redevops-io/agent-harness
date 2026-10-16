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

## Approval allowlist semantics

Allowlist entries are resolved to absolute paths through a fixed trusted search
path (`/usr/local/bin`, `/usr/bin`, `/bin`), never the caller's `PATH` or
working directory. Command names containing `/` are rejected unless that exact
literal is allowlisted. File-like arguments of allowlisted commands must stay
inside the workspace root (`AGENT_HARNESS_ROOT` or the cwd; symlinks and `..`
are resolved). Files written via `write_file` during a session cannot be
executed, or passed to an interpreter, until re-approved
(`ApprovalPolicy.clear_written`). `run_bash` uses the Agent's policy and root and
runs with `cwd` set to the root.

This is a policy check, not isolation: allowed commands still run as the host
user. Use an OS-level sandbox or container for untrusted workloads.
