> ### Developer toolkit for [Context Runtime](https://github.com/redevops-io/context-runtime)
>
> The agent toolkit the [ReDevOps reference applications](https://github.com/redevops-io) build on — LLM client, tool registry, approval flow, sandboxed execution, guardrails and an eval harness. The reference applications pair it with **Context Runtime**, which decides *what context each agent sees* before it runs.
>
> ```
> Context Runtime  →  ReDevOps RAG  →  Sidekick  →  Application logic
> ```

---

# agent_harness

agent_harness is the OSS core that provides the building blocks for a safe, tool-using agent: LLM client, tool registry, approval flow, sandboxed execution, guardrails, and an eval harness. It is the foundation that an agent layer consumes.

## Quickstart

```bash
pip install -e .
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_API_KEY=ollama
export MODEL=llama3.1
python -m examples.tiny_agent
```

## Using the modules

```python
from agent_harness import llm, tools, approval, sandbox, guardrails
```

## Architecture

agent_harness (OSS core) + thin agent layer that wires real providers: llama.cpp/ollama, vLLM, Moonshot/Kimi, xAI/Grok, OpenAI. See docs/architecture.md and docs/configuration.md.

## License

AGPL-3.0
