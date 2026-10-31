# Status

| | |
|---|---|
| Maturity | alpha / experimental |
| Version | 0.1.0 (`pyproject.toml`); tag `v0.1.0` |
| Last git activity | 2026-09-25 (`git log -1 --format=%cs`; clone is shallow) |
| Activity | low: 10 commits between 2026-07-15 and 2026-10-15 (GitHub API) |
| Distribution | not published by ReDevOps on PyPI; install from git only (PyPI `agent-harness` is an unrelated project) |
| Python | `requires-python >=3.10`; the optional QA module (`agent_harness.qa`) needs 3.11+ |

## Supported

- Modules: `loop` (`run_loop`: multi-turn tool-calling loop with `max_turns` / `max_seconds` / context limits and a JSONL trajectory), `context` (context-manager interface), `llm` (minimal OpenAI-compatible client stub), `tools` (tool contract, registry, approval gate; same contract as context-runtime), `command_policy` (shell/edit vetting), `sandbox`, `guardrails` (plus `guardrails_nemo`), `evals`, `agent`, `nim`, `openshell`.
- `agent_harness.qa` (optional QA module, Python 3.11+): records, ports, `canon-v1`, verifier, classifier, single-run executor, recovery, egress policy and redaction; see [docs/qa.md](docs/qa.md). It makes no model calls, opens no network connection and drives no browser; the adapter, evidence store, egress enforcement and model client are supplied by the deployment. Its tests are in `tests/qa`.
- A local example (`python -m examples.tiny_agent`) and a pytest suite (159 tests passed locally on 2026-10-19, including 69 conformance cases that run only when upstream harness2 and context-runtime are installed).

## Not supported / not established

- No real LLM provider is wired in this repo; the agent layer that does so is not included.
- `Agent.run` (`agent_harness.agent`) is a minimal single-step stub that only finishes; the multi-turn loop is `agent_harness.loop.run_loop`. The sandbox is soft confinement, not full isolation (see `docs/architecture.md`).
- A tests workflow is defined (`.github/workflows/tests.yml`; run history not checked) alongside the secret scan; no benchmarks or performance claims.
- `agent_harness.h2` (Harness² file-tree spec, validators, artifacts) is groundwork only: see [docs/harness2.md](docs/harness2.md).
- The consumer side of the integration is not in this repository: `agent_harness/context.py` documents Context Runtime's `context_runtime.integrations.agent_harness.ContextRuntimeManager` as depending on this package ("the dependency points from context-runtime to here"), and `agent_harness/bench/jobbench.py` loads that manager lazily. No import of `agent_harness` by another package is demonstrated here.

## Known issues

- Wording "OSS core" appears in `pyproject.toml` (description) and AGENTS.md; this PR neutralised it in the README only.
- GitHub topics (`rag`, `semantic-search`, `context-runtime`) are generic relative to the package's role.

Related repositories and what each owns: see OVERLAP in [README.md](README.md#overlap).
