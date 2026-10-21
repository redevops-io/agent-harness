# agent-harness

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE) ![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg) [![NVIDIA Inception](https://img.shields.io/badge/NVIDIA-Inception%20Program%20Member-76B900.svg)](https://www.nvidia.com/en-us/startups/)

> **🚀 NVIDIA Inception Program Member** — ReDevOps is a member of the NVIDIA Inception Program, supporting startups advancing AI and accelerated computing. Membership provides access to NVIDIA technology, technical resources, and the startup ecosystem. It does not imply product endorsement by NVIDIA.

**What it does.** `agent_harness` is a small Python package of building blocks for a safe, tool-using agent: an LLM client
(a minimal OpenAI-compatible stub), a tool registry with an approval gate (the context-runtime tool contract), a shell/edit command policy, a sandbox wrapper, guardrails and an eval harness.
It is the foundation an agent layer consumes; concrete providers (llama.cpp/ollama, vLLM, Moonshot/Kimi, xAI/Grok, OpenAI)
are wired by that layer, not here. It is the developer toolkit that the [ReDevOps reference applications](https://github.com/redevops-io) build on; they pair it with [Context Runtime](https://github.com/redevops-io/context-runtime), which decides what context each agent sees:

```
Context Runtime  ->  ReDevOps RAG  ->  Sidekick  ->  Application logic
```

**Who it's for.** Developers building or reading the ReDevOps reference applications, and anyone who wants a minimal
approval/sandbox/guardrail skeleton. Not a Mission runtime; see [OVERLAP](#overlap).

**Maturity: alpha / experimental.** Version 0.1.0, one tag (`v0.1.0`). The shipped agent loop is a minimal single-step stub,
the example runs offline against a stub client, and the sandbox is documented as *not* full isolation. See [STATUS.md](STATUS.md).

**Ownership, overlap and support status.** Owned by ReDevOps. Status: **experimental, low activity** (last commit 2026-09-25;
10 commits between 2026-07-15 and 2026-10-15 per the GitHub API). Primary consumer: in the clones checked
(`agentic-os` and the other repos in this org's working set) no code does `import agent_harness`. [`agentic-os`](https://github.com/redevops-io/agentic-os)
describes it in `docs/architecture.md` and `agentic_os/stack_knowledge.py`, mentions it in app comments, and its NVIDIA NIM profile
refers to `agent_harness.nim`; the "agent layer" that wires it is not in this repository. Questions and bug reports: GitHub Issues
([CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md)).

## Install

This project is **not published on PyPI by ReDevOps**. The name `agent-harness` on PyPI belongs to an unrelated third-party
project (version 0.1.10 when checked); **never run `pip install agent-harness`**. Install from git or from a clone:

```bash
pip install "git+https://github.com/redevops-io/agent-harness@v0.1.0"
# or, from a clone
git clone https://github.com/redevops-io/agent-harness && cd agent-harness && pip install -e .
```

## 3-minute example

```bash
git clone https://github.com/redevops-io/agent-harness && cd agent-harness
pip install -e ".[dev]"
python -m examples.tiny_agent     # offline, stub client; prints {'model': None, 'tools': ['add'], 'result': 'What is 2+3?'}
python -m pytest -q               # 159 passed locally on 2026-10-19 (Python 3.13; 90 without the optional upstream and context-runtime conformance tests)
```

The bundled `tiny_agent` example runs fully **offline** against the stub LLM client (it echoes the prompt rather than calling a model).
The `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `MODEL` environment variables only take effect once an agent layer wires a real provider.

**Evidence.** [`tests/`](tests) (Internally reproduced: run locally, 159 passed with the optional conformance extras installed; a tests workflow is defined in `.github/workflows/tests.yml`, plus a secret scan; its run history was not checked).
No benchmark or performance numbers are claimed.

**Limitations.** Stub LLM client; single-step agent loop; sandbox is soft confinement; tool schemas are stored for export, not validated at call time.

**Docs.** [docs/architecture.md](docs/architecture.md), [docs/configuration.md](docs/configuration.md), [docs/harness2.md](docs/harness2.md) (Harness² file-tree spec and validators; groundwork only, no benchmark results), [docs/spend.md](docs/spend.md) (per-provider dollar caps and a metering proxy for paid calls), [docs/context.md](docs/context.md) (context management interface; lossless compaction via context-runtime).

## Quickstart (original text)

```bash
pip install -e .
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_API_KEY=ollama
export MODEL=llama3.1
python -m examples.tiny_agent
```

## Using the modules

```python
from agent_harness import llm, tools, command_policy, sandbox, guardrails
```

## Architecture

`agent_harness` (this package, the core building blocks) + a thin agent layer that wires real providers: llama.cpp/ollama, vLLM, Moonshot/Kimi, xAI/Grok, OpenAI. See docs/architecture.md and docs/configuration.md.

## License

AGPL-3.0 (`LICENSE` is the "GNU AFFERO GENERAL PUBLIC LICENSE, Version 3" text; `pyproject.toml` declares `AGPL-3.0-or-later`).

## OVERLAP

Several ReDevOps repositories sound alike. This table is identical in `redevops-connectors`, `redevops-integrations` and `agent-harness`.

| Repo | Owns | Does not own | Licence file (see its own repo) |
|---|---|---|---|
| [`redevops-connectors`](https://github.com/redevops-io/redevops-connectors) | The `IntegrationAdapter` contract, OAuth/credential-reference plumbing, and 14 provider adapters (Gmail, Google Calendar, HubSpot, Stripe, Polar, Slack, WhatsApp Cloud API, WhatsApp via WAHA, Klaviyo, Postiz, Ayrshare, Blotato, Apollo, Web). Tested against canned fixtures only. | Governance decisions, Missions, framework benchmarks. | `LICENSE.md` (AGPL-3.0-or-later with Commons Clause) |
| [`redevops-integrations`](https://github.com/redevops-io/redevops-integrations) | Eight framework/cloud conformance slices (NVIDIA NeMo, PydanticAI, LlamaIndex, CrewAI, Azure, Google, AWS, DigitalOcean) that wrap a real framework agent with the Mission SDK and classify native vs native+ReDevOps results. | The connector adapter contract, the Mission SDK itself. | Apache-2.0 |
| [`agent-harness`](https://github.com/redevops-io/agent-harness) | Single-step agent building blocks: LLM client, tool registry and approval gate, command policy, sandbox, guardrails, eval harness. | Mission lifecycle, provider integrations, context planning. | AGPL-3.0 text (`LICENSE`) |
| [`mission-sdk`](https://github.com/redevops-io/mission-sdk) | The `redevops-mission` package and `rdo` CLI: the developer-facing way to author and run governed Missions. | Provider adapters, framework conformance slices. | Apache-2.0 |
| [`agentic-os`](https://github.com/redevops-io/agentic-os) | The control plane and Mission kernel, plus the reference apps; it is the main consumer of `redevops-connectors` (optional `connectors` / `projects` extras). | Connector contract source, SDK packaging. | See that repo |

Note on the word "harness": here it means the toolkit in this repo. In [Harness²](docs/harness2.md) it means an agent's editable file tree, and `HarnessProfile` in `agentic-os` is an evaluation-profile selector.

Common confusions: `redevops-connectors` is the live-adapter SDK, `redevops-integrations` is a conformance harness (not an adapter library), and `agent-harness` is not a Mission runtime. Do not `pip install agent-harness` from PyPI: that name belongs to an unrelated third-party project.
