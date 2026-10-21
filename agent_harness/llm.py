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


# --- real OpenAI-compatible chat client with tool calling (stdlib only) ----------------------------------------
import json as _json
import time as _time
import urllib.error as _urlerr
import urllib.request as _urlreq


class ChatError(RuntimeError):
    pass


class ChatClient:
    """Minimal ``/chat/completions`` client. ``extra_body`` is merged into every request (e.g. provider switches);
    transient failures (429, 5xx, network) are retried with backoff, other 4xx are raised immediately."""

    def __init__(self, base_url: str, model: str, api_key: str = "dummy", temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None, extra_body: Optional[Dict[str, Any]] = None,
                 timeout: float = 900, retries: int = 3):
        self.base_url, self.model, self.api_key = base_url.rstrip("/"), model, api_key
        self.temperature, self.max_tokens = temperature, max_tokens
        self.extra_body, self.timeout, self.retries = dict(extra_body or {}), timeout, retries

    def complete(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"model": self.model, "messages": messages, **self.extra_body}
        if tools:
            body["tools"] = tools
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens
        data = _json.dumps(body).encode()
        last: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            req = _urlreq.Request(self.base_url + "/chat/completions", data,
                                  {"Content-Type": "application/json", "Authorization": "Bearer " + self.api_key})
            try:
                with _urlreq.urlopen(req, timeout=self.timeout) as r:
                    out = _json.load(r)
                choice = out["choices"][0]
                return {"message": choice["message"], "finish_reason": choice.get("finish_reason"),
                        "usage": out.get("usage") or {}}
            except _urlerr.HTTPError as e:
                detail = e.read().decode(errors="replace")[:400]
                if e.code != 429 and e.code < 500:
                    raise ChatError(f"HTTP {e.code}: {detail}") from e
                last = ChatError(f"HTTP {e.code}: {detail}")
            except (OSError, ValueError, KeyError) as e:
                last = e
            _time.sleep(min(2 ** attempt * 2, 30))
        raise ChatError(f"chat request failed after {self.retries + 1} attempts: {last}")
