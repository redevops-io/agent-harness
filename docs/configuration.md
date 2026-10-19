# Configuration

Environment variables:

- `OPENAI_BASE_URL` – base URL for OpenAI-compatible endpoint
- `OPENAI_API_KEY` – API key or provider token
- `MODEL` – model identifier

Any OpenAI-compatible endpoint (llama.cpp / vLLM / Ollama / a cloud provider) is reached through
`OPENAI_BASE_URL` + `MODEL`; the harness core reads no other endpoint env vars.

Approval is configured **in code**, not via an env var. There are two layers.

**Tool level: `agent_harness.tools.ApprovalPolicy`** decides whether a side-effecting or gated tool may run
at all (same contract as context-runtime). Pass it to `ToolRegistry(policy)` or `Agent(approval=...)`:

- `deny_side_effects` (default) – read-only tools run; side-effecting tools need an `approver` callback
  returning true
- `allowlist` – only side-effecting tools named in `allow` run
- `bypass` – allow everything (use with care)
- any other value denies every gated call

A `ToolRegistry` also takes an optional `authorizer(principal, spec, args)` (or the process-wide
`set_default_authorizer`) that runs before the approval gate, and keeps an `audit` list of gated decisions.

**Command level: `agent_harness.command_policy.CommandPolicy`** vets the content of a shell command or edit
for the Agent's `run_bash` / `write_file` tools. Pass it as `Agent(policy=...)`. Modes:

- `allowlist` – only commands on the allowlist are allowed
- `edits-only` – allow edits, deny shell commands
- `bypass` – allow everything (use with care)
- `never` – deny all

By default the Agent allows its own `write_file` and `run_bash` at the tool level and leaves the real vetting
to `CommandPolicy`.

## Command allowlist semantics

Allowlist entries are resolved to absolute paths through a fixed trusted search
path (`/usr/local/bin`, `/usr/bin`, `/bin`), never the caller's `PATH` or
working directory. Command names containing `/` are rejected unless that exact
literal is allowlisted. File-like arguments of allowlisted commands must stay
inside the workspace root (`AGENT_HARNESS_ROOT` or the cwd; symlinks and `..`
are resolved). Files written via `write_file` during a session cannot be
executed, or passed to an interpreter, until re-approved
(`CommandPolicy.clear_written`). `run_bash` uses the Agent's policy and root and
runs with `cwd` set to the root.

This is a policy check, not isolation: allowed commands still run as the host
user. Use an OS-level sandbox or container for untrusted workloads.
