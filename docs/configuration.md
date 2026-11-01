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

## Process execution

Commands run by `agent_harness.sandbox` (and the judge and benchmark launches) go through `agent_harness.process_runner`,
which owns the child's whole process tree:

- **POSIX (Linux, macOS):** the child starts in a new session and process group. On a timeout or a cancel the group gets
  `SIGTERM`, then `SIGKILL` after a grace period (2 s), and is reaped.
- **Windows:** the child is created suspended, assigned to a Job Object with kill-on-close, then resumed; stopping
  terminates the job.
- **Descendants are killed at the end of every call.** A command that starts a background process does not leave it
  running: a service started in one tool call is gone in the next. Tell the model so if it matters for your tools.
- A process that starts its own session (`setsid`, `git gc --auto`, `tmux -d`, `ssh-agent`) leaves the group on POSIX and can
  hold the output pipes; the call then waits up to about 2.5 s for the output to drain, returns what it has, and reports
  `drain_complete=False`. On Windows a descendant cannot leave the job.
- `max_seconds` and `cancel` of `run_loop` apply while a tool runs (a contextvar, `run_control`): a tool that runs
  `run_shell` in a new thread does not inherit it unless the thread runs in a copied context
  (`contextvars.copy_context().run`, `asyncio.to_thread`).
- Output is decoded as UTF-8 with replacement characters; `\r\n` is preserved; the sandbox keeps at most 4 MiB per
  stream (the excess is counted, not stored). The child's stdin is closed.
- A process group and a Job Object are process-tree control, not isolation: files, network and system calls are not
  restricted.

## Child environment

A child process started by the sandbox, the judge launch or a benchmark launch does **not** inherit the parent's
environment (`agent_harness.env_policy`). It starts from an allowlist, so a credential in the launching shell does not reach
it. Names are matched case-sensitively on POSIX and case-insensitively on Windows.

| Child | Gets |
|---|---|
| Sandboxed command (`preset="tool"`) | `PATH` (POSIX: `/usr/local/bin:/usr/bin:/bin`; Windows: the system directories under `SYSTEMROOT`), `LANG`, `LANGUAGE`, `LC_*`, `TZ`, `TERM`, and `TMPDIR` (POSIX) or `TEMP`, `TMP` (Windows). Windows also: `PATHEXT`, `SYSTEMROOT`, `SYSTEMDRIVE`, `WINDIR`, `COMSPEC`, `USERPROFILE`, `APPDATA`, `LOCALAPPDATA`, `PROGRAMDATA`, `PROGRAMFILES*`, `COMMONPROGRAMFILES*`, `ALLUSERSPROFILE`, `PUBLIC`, `HOMEDRIVE`, `HOMEPATH`, `OS`, `PROCESSOR_ARCHITECTURE`, `NUMBER_OF_PROCESSORS` |
| Judge and benchmark launches (`preset="bench"`, the real `PATH` is kept) | the above plus `HOME`, `XDG_*_HOME`, `PYTHONPATH`, `PYTHONUTF8`, `PYTHONIOENCODING`, `HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY`, `ALL_PROXY` (both cases), `SSL_CERT_FILE`, `SSL_CERT_DIR`, `REQUESTS_CA_BUNDLE`, `CURL_CA_BUNDLE`, `NODE_EXTRA_CA_CERTS`; the judge also gets `JUDGE_API_KEY` and `JUDGE_REASONING_EFFORT` from its configuration and `JUDGE_MAX_COMPLETION_TOKENS` when set |

What a sandboxed command therefore **loses**: `HOME`, `USER`/`LOGNAME`, `VIRTUAL_ENV`, proxy variables, CA bundle variables,
`XDG_*`, `SSH_AUTH_SOCK`, `DOCKER_HOST`, `JAVA_HOME`, `NODE_*`, `PYTHONUTF8`/`PYTHONIOENCODING`, and every credential.
Practical effects: a command behind a corporate proxy or a private CA fails unless the variables are passed; `git commit`
authors with a guessed `user@host` identity and ignores `~/.gitconfig` (pass `HOME`, or set the identity in the command);
on Windows a Python child gets the legacy code page unless `PYTHONUTF8` is passed.

- **Opt in by name:** `sandbox.run_shell(..., env_pass=("HOME", "HTTPS_PROXY"))` (a sequence; a bare string is refused), or
  process-wide `AGENT_HARNESS_ENV_PASS="HOME,HTTPS_PROXY"`. Each opt-in is logged by `agent_harness.env_policy` (names only,
  never values); a WARNING is logged when an ambient variable that looks like a credential (`*KEY*`, `*TOKEN*`, `*SECRET*`,
  `*PASS*`, `*AUTH*`, `*DSN*`, `DATABASE*`, cloud and provider prefixes ...) is passed, and when a passed value embeds
  credentials in a URL (`scheme://user:password@host`).
- **Explicit values from code:** `child_env(extra={"NAME": "value"})`.
- **Opt out entirely:** `AGENT_HARNESS_ENV_INHERIT=1` (exactly `1`) or `child_env(inherit_all=True)` passes the whole
  environment, with a WARNING on every call. `PATH` stays pinned to the trusted directories unless `inherit_path=True` is
  given.

