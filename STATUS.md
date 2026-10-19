# Status

| | |
|---|---|
| Maturity | alpha / experimental |
| Version | 0.1.0 (`pyproject.toml`); tag `v0.1.0` |
| Last git activity | 2026-09-25 (`git log -1 --format=%cs`; clone is shallow) |
| Activity | low: 10 commits between 2026-07-15 and 2026-10-15 (GitHub API) |
| Distribution | not published by ReDevOps on PyPI; install from git only (PyPI `agent-harness` is an unrelated project) |
| Python | `requires-python >=3.10` |

## Supported

- Modules: `llm` (minimal OpenAI-compatible client stub), `tools` (tool contract, registry, approval gate; same contract as context-runtime), `command_policy` (shell/edit vetting), `sandbox`, `guardrails` (plus `guardrails_nemo`), `evals`, `agent`, `nim`, `openshell`.
- A local example (`python -m examples.tiny_agent`) and a pytest suite (159 tests passed locally on 2026-10-19, including 69 conformance cases that run only when upstream harness2 and context-runtime are installed).

## Not supported / not established

- No real LLM provider is wired in this repo; the agent layer that does so is not included.
- The agent loop is a minimal single-step stub; the sandbox is soft confinement, not full isolation (see `docs/architecture.md`).
- A tests workflow is defined (`.github/workflows/tests.yml`; run history not checked) alongside the secret scan; no benchmarks or performance claims.
- `agent_harness.h2` (Harness² file-tree spec, validators, artifacts) is groundwork only: see [docs/harness2.md](docs/harness2.md).
- No code in the other ReDevOps clones checked imports `agent_harness`; its consumer is described in docs rather than demonstrated by an import.

## Known issues

- Wording "OSS core" appears in `pyproject.toml` (description) and AGENTS.md; this PR neutralised it in the README only.
- GitHub topics (`rag`, `semantic-search`, `context-runtime`) are generic relative to the package's role.

Related repositories and what each owns: see OVERLAP in [README.md](README.md#overlap).
