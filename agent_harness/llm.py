"""Minimal LLM client shim.

The OSS core ships without a real provider wired in; a thin agent layer supplies
one (ollama/vLLM/OpenAI/etc.). This ``Client`` is just enough for the example
and tests to run: it resolves a registered tool and dispatches it via the
approval + sandbox path, without contacting any network model.
"""
import os
from typing import Any, Dict, List, Optional


class Client:
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None,
                 model: Optional[str] = None):
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model or os.environ.get("MODEL")

    def chat(self, prompt: str, tools: Optional[List[str]] = None,
             approver: Any = None, sandbox: Any = None, **kwargs: Any) -> Dict[str, Any]:
        """Offline stub.

        A real provider would call the model, parse a tool call, run it through
        ``approver`` + ``sandbox``, and return the result. Without a provider this
        just echoes the prompt and the tools that were made available, so the
        example/wiring can run end to end.
        """
        return {"model": self.model, "tools": list(tools or []), "result": prompt}
