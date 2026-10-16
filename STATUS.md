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

- Modules: `llm` (minimal OpenAI-compatible client stub), `tools` (registry), `approval`, `sandbox`, `guardrails` (plus `guardrails_nemo`), `evals`, `agent`, `nim`, `openshell`.
- A local example (`python -m examples.tiny_agent`) and a pytest suite (44 tests passed locally on 2026-10-15).

## Not supported / not established

- No real LLM provider is wired in this repo; the agent layer that does so is not included.
- The agent loop is a minimal single-step stub; the sandbox is soft confinement, not full isolation (see `docs/architecture.md`).
- No CI test workflow (only a secret scan); no benchmarks or performance claims.
- No code in the other ReDevOps clones checked imports `agent_harness`; its consumer is described in docs rather than demonstrated by an import.

## Known issues

- Wording "OSS core" appears in `pyproject.toml` (description) and AGENTS.md; this PR neutralised it in the README only.
- GitHub topics (`rag`, `semantic-search`, `context-runtime`) are generic relative to the package's role.

Related repositories and what each owns: see OVERLAP in [README.md](README.md#overlap).
