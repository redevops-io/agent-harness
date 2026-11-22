# Changelog

Notable changes to `agent-harness`. The package version in `pyproject.toml` is `0.2.0` (tag `v0.2.0`; the previous tag is `v0.1.0`).
The 0.2.0 release was checked against [QA_RELEASE_CHECKLIST.md](QA_RELEASE_CHECKLIST.md).

## Next release (not yet tagged)

### Changed (behaviour changes)

- **Tool authorization is fail-closed (`agent_harness.tools`).** The authorizer seam used to read `None` as *allow*,
  so an authorizer that returned early, missed a branch, or was never installed let the tool run — the
  deny-by-default registry was fail-open for every tool, read-only tools included. Now: a registry with no authorizer
  denies every call (`missing_explicit_grant`), an authorizer that raises or returns a non-decision denies
  (`no_resource_plane` / `malformed_request`), and clearing the process-wide default closes the registries instead of
  opening them. `None` is no longer an allow; `True` and a non-empty `str` still work and are audited as legacy
  decisions. Two named opt-outs replace the old silence: `local_trusted_authorizer()` for local/direct execution
  (attributed to a plane and warned once) and `ToolRegistry(authorizer_required=False)` (warns at construction, audits
  every call as a wildcard grant). `Agent`, `bench_registry` and `parity_registry` now pass local trust explicitly and
  accept `authorizer=` to run behind a real policy plane; `Agent(model=None)` remains ungated by design. New:
  `Authorization` / `Decision` / `PolicyReason` (the `authz/v1` subset — spelled locally, pinned against the contract
  by `tests/test_authz_contract_align.py`, still zero dependencies), `ToolRegistry.authorize`,
  `ToolRegistry.unattributed_allows` and `default_authorizer_installed`. **Consumers that pass an authorizer returning
  `None`, or that construct a bare `ToolRegistry()`, start denying: measured against `context-runtime` `c255945`, 17 of
  its 367 tests fail (12 `integrations/*` adapters construct a bare registry) until it wires its own authorizer.**

- **Tool registration carries a risk class, and the registry checks itself at boot (`agent_harness.tools`).** A tool
  now declares the `authz/v1` action it needs — `read`, `write` or `delete` (`function_tool(..., action=...)`, or
  `ToolPlugin.action`) — and the registry turns that into the capability the call requires
  (`ToolRegistry.required_capability(name)` -> `tool:<name>:<action>`), so "what may this run?" is answered by the code
  under review instead of by the caller. A registration that contradicts itself (`action="read"` on a side-effecting or
  approval-gated tool — no grant plane hands out `tool:<name>:read` for a call that writes) raises
  `ToolClassificationError` where the old `permission` string agreed to anything; re-registering a tool cannot lower its
  class; an undeclared class stays legal at registration and is fatal at boot. `ToolRegistry.validate()` returns a
  `ValidationReport` — problems: no authorizer, tools with no declared class; warnings: gated tools the configured
  `ApprovalPolicy` can never run — and `validate(strict=True)` raises instead. `Agent`, `bench_registry` and
  `parity_registry` run that check at construction: a misconfiguration stops the service there rather than surfacing
  as a denial in front of a user. New: `ACTIONS`, `TOOL_RESOURCE_KIND`, `ValidationReport`, `ToolClassificationError`,
  `RegistryConfigurationError`, `ToolRegistry.classification` / `.unclassified` / `.validate`. `Agent`'s built-ins are
  classified (`read_file` and `finish` -> `read`; `write_file` and `run_bash` -> `write` — a shell is classified by
  what running a command accomplishes, not by whether this particular command looks harmless) and the evaluation
  toolsets declare their classes, so everything this repository ships passes its own startup check. Approval bypass
  is unchanged and still does not stand in for authorization: `ToolRegistry(ApprovalPolicy("bypass"))` with no
  authorizer denies every call
  (`tests/test_authz_failclosed.py::test_approval_bypass_does_not_substitute_for_authorization`).
  **Measured consumer impact: `context-runtime` `c255945` fails the same 17 of its 364 tests — 7 files: the
  `integrations/*` module and security suites, `test_tools.py`, `test_tool_authorizer.py`, `test_mcp.py`,
  `test_outreach.py`, `test_transcript_compaction.py` — against the fail-closed commit and against this one, with identical failure
  sets, because those registries stop at the missing authorizer before classification is ever consulted. The undeclared
  classes surface the moment its adapters wire an authorizer, which is why the consumer-side fix does both at once.**

- **Process execution (`agent_harness.process_runner`, used by `sandbox.run_shell`, the judge launches and the OpenCode and
  Harness2 launches).** Each child now runs in its own process group (POSIX) or Job Object (Windows), and the whole tree is
  stopped on a timeout, on a cancel, and when the call ends. See [configuration.md](configuration.md#process-execution).
  - Processes a command leaves running are **killed when the call returns**: a service started in one call (for example
    `python -m http.server &` or `nohup ... &`) is gone in the next. A process that starts its own session escapes the group;
    the call then waits up to about 2.5 s for the output to drain and reports `drain_complete=False`.
  - The child's stdin is closed (`DEVNULL`) instead of inherited.
  - Output is read as bytes and decoded as UTF-8 with `errors="replace"` (it was the locale codec, with newline translation):
    `\r\n` from a command is now preserved, undecodable bytes become U+FFFD, and the sandbox keeps at most 4 MiB per stream.
  - `sandbox.run_shell` and `sandbox.run` take `cancel=`; their result gains `tree_gone`; a cancelled run returns code 130
    (a timeout is still 124). `agent_harness.sandbox` no longer imports `subprocess`.
  - `run_loop` takes `cancel=`; `max_seconds` and a cancel now also apply while a tool is running, and the run can end
    with `stop="cancelled"`.
  - SIGTERM and SIGHUP sent to the parent stop its running children before the signal takes its normal course. A handler is
    installed on the first run, only if that run is on the main thread and the application does not ignore the signal, and
    it is never removed; an application whose runs are all on other threads gets only the `atexit` hook (normal exit). The
    handler **SIGKILLs every live group first** and then chains to the application's own handler (or restores the default
    and re-raises); a command that was running therefore returns code -9. SIGKILL of the parent cannot be handled on POSIX;
    on Windows the Job Object ends the children. A forked child does not inherit the parent's live groups.
  - A failure of an output reader while the child is running (for example EIO or EMFILE) stops the tree at once and is
    reported as an error (`read_error`; `run_shell` returns a non-zero code with `[output read error: ...]`, `run_subprocess`
    raises `OSError`), ahead of a timeout or cancel; output is never silently dropped.
- **Child environment (`agent_harness.env_policy`): children no longer inherit the parent's environment.** Before, the
  sandbox passed the whole of `os.environ` to every command and the judge and evaluation-runner launches passed `{**os.environ, ...}`,
  so a credential exported in the launching shell reached every child. Now a child starts from an allowlist. This breaks any
  child that relied on inheritance. See [configuration.md](configuration.md#child-environment) for exactly what is kept,
  what a sandboxed command loses, and how to opt in.
  - Sandboxed commands keep only `PATH` (pinned to the trusted directories; on Windows the system directories), locale
    (`LANG`, `LANGUAGE`, `LC_*`), `TZ`, `TERM` and the temp directory; on Windows also the system and profile variables.
  - Opt in by name with `run_shell(env_pass=("NAME",))` or `AGENT_HARNESS_ENV_PASS=NAME,OTHER` (logged by name, never value);
    opt out entirely with `AGENT_HARNESS_ENV_INHERIT=1` (a warning on every call; `PATH` stays pinned unless the caller asks).


## [0.2.0] - 2026-10-31

### Added

- **`agent_harness.qa`**, an optional QA module (Python 3.11 or newer; the `qa` extra is empty and the floor is enforced
  by an import guard). Standard library and harness core only; `import agent_harness` never imports it. See [qa.md](qa.md).
  - Neutral records, ports, typed errors, schema identifiers and `canon-v1` hashing, mirroring runtime-contracts `qa/v1`
    independently (`canon-v1:<hex>` equals `rcv1:<hex>`), with the 97 frozen canonical-hash vectors under `tests/qa/golden/`.
  - A deterministic assertion verifier that loads frozen assertions by pinned hash and recomputes the hash, a pure run
    classifier, and a read-only spec store.
  - A bounded single-run executor with an explicit state machine, an append-only receipt chain, a fail-closed `ui_action`
    policy, idempotency-aware retries, deadlines and budgets.
  - Deterministic recovery (semantic locator, checkpoint and fixture modes) and model-candidate recovery behind an injected
    client, off by default. A run that used a recovered locator never ends `PASS` or `PRODUCT_FAIL`.
  - An egress policy (default deny, metadata hard-deny, resolve-then-pin, per-hop redirect checks), a fail-closed redactor,
    full-width digest checks and an evidence integrity verifier.
- `tests/qa/`: the QA test suite, including mutation tests for the immutable-assertion rule, import-boundary and
  licence-boundary checks, and the Python floor tests.
- `.github/workflows/qa.yml`: runs `tests/qa` on Python 3.11 and 3.12 with `QA_STRICT_ENV=1` and fails if no QA test is
  collected or any is skipped (`tests/qa/ci_gate.py`, allow-list `tests/qa/ci_skip_allowlist.txt`, currently empty); a
  Python 3.10 job checks that the core imports and that `import agent_harness.qa` fails with the guard message. The
  existing `tests` workflow is unchanged.
- Documentation: [qa.md](qa.md) (module map, ports and what a deployment supplies, the immutable-assertion rule,
  recovery, policy, egress and redaction, evidence integrity, what the module does not do, the compatibility statement and
  the licence statement) and [QA_RELEASE_CHECKLIST.md](QA_RELEASE_CHECKLIST.md).

### Changed

- `README.md`, `STATUS.md` and `docs/architecture.md` no longer describe the agent loop as a single-step stub: that
  describes `Agent.run`; `agent_harness.loop.run_loop` is a multi-turn tool-calling loop with hard limits. They also
  describe the Context Runtime integration as it is documented in `agent_harness/context.py`.
- `agent_harness.qa` now follows the contracts `qa/v1` text where it had derived its own definition (compatibility
  statement in [qa.md](qa.md)): the digest of an action value is `rcv1` of the value's canonical JSON text alone (a
  `SecretRef` digests as its fingerprint); each set-valued field is ordered by its own rule (element key, canonical text
  or value); the integrity refusals have distinct codes (`ASSERTION_PIN_MALFORMED`, `ASSERTION_PIN_NOT_FOUND`,
  `ASSERTION_PROGRAM_UNREGISTERED`, `ASSERTION_PROGRAM_MISMATCH`, `BATCH_HALTED`) and an unknown code classifies as an
  integrity refusal; `FaultRecovery` carries an optional `reason_code` from a closed vocabulary; `Check.detail` is not
  hashed, `Check.evidence_refs` is sorted when hashed, an `EvidenceRef` and a `SecretRef` hash with the contracts'
  defaults; and `ACTION_TIMEOUT` is recoverable (recovery stays guarded). Hashes of records that carry those values
  change accordingly. The golden data under `tests/qa/golden/` gains the added vectors, the clarification fixtures, the
  field inventory and the JSON Schema of the contracts commit named in the compatibility statement.
- No behaviour of the existing modules changed, and no `LICENSE` file changed. The harness remains AGPL-3.0-or-later.
