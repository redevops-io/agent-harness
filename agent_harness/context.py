"""Context management interface for the agent loop.

The loop asks a ``ContextManager`` to keep the conversation inside the model's window and may expose extra
tools it provides (for example retrieval over content that was compacted away). Two implementations:

* ``BasicContextManager`` (default, dependency-free): clips big tool outputs, elides old ones and shortens old
  tool-call arguments. Lossy: nothing is stored anywhere, so elided content is gone.
* ``context_runtime.integrations.agent_harness.ContextRuntimeManager`` (separate package): compacts against
  a token budget, archives everything it removes, indexes the archive and gives the model a ``search_context``
  tool to get it back. This package never imports it; the dependency points from context-runtime to here.

A manager works on the OpenAI-style message list in place. ``target_tokens`` is where the prompt should end
up after ``prepare`` (not just below the hard limit), so the loop does not compact on every turn.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

from .multimodal import content_chars
from .tools import ToolPlugin

ELIDED = "[earlier tool output removed to save context]"


# -- placeholders --------------------------------------------------------------------------------------------------
# A manager that shortens old content leaves a stub in the history. A model can imitate a stub and emit it as the argument of a NEW action (seen: a write_file
# whose content was a placeholder, in 44 of 52 cases with an id that appeared nowhere in the request). Such text is never real content, so the loop refuses to run an
# action that contains one. WHAT a stub looks like is each manager's own business: the loop asks ``cm.is_placeholder(value)`` (ContextAware). The patterns below
# cover only the markers THIS package writes (BasicContextManager).
_PLACEHOLDER_ANYWHERE = ("chars omitted to save context]", "[earlier tool output removed to save context]")


def placeholder_in(value: Any) -> bool:
    return isinstance(value, str) and any(m in value for m in _PLACEHOLDER_ANYWHERE)


def placeholder_arg(args: Dict[str, Any], is_placeholder: Optional[Callable[[Any], bool]] = None) -> Optional[str]:
    """Name of the first string argument that is (or contains) a context placeholder, else None. ``is_placeholder`` is the context manager's own
    detector for ITS stand-ins (Context Runtime owns its formats); the patterns in this module cover only this package's own markers."""
    for k, v in (args or {}).items():
        if placeholder_in(v) or (is_placeholder is not None and is_placeholder(v)):
            return str(k)
    return None


@dataclass(frozen=True)
class ResultView:
    """What the model can currently see of one tool result. ``state``: visible | clipped (head and tail shown, middle stored) | archived (replaced by a
    stub, recoverable) | unavailable (gone: lossy elision, or never in the conversation). Managers return their own object with the same fields."""
    state: str
    archive_id: Optional[str] = None
    chars: int = 0

    @property
    def recoverable(self) -> bool:
        return self.state in ("archived", "clipped") and self.archive_id is not None


def basic_visibility(messages: List[Dict[str, Any]], tool_call_id: str) -> ResultView:
    """Visibility for managers that store nothing: elided means unavailable, a clipped result is visible but cut."""
    for m in reversed(messages):
        if m.get("role") == "tool" and m.get("tool_call_id") == tool_call_id:
            c = m.get("content")
            if not isinstance(c, str):
                return ResultView("visible", None, 0)
            if c == ELIDED:
                return ResultView("unavailable", None, len(c))
            return ResultView("clipped" if "chars omitted ...]" in c else "visible", None, len(c))
    return ResultView("unavailable")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


class PinSet:
    """Texts that must stay verbatim in the prompt (for example the task instructions). A tool result that contains the start of a
    pinned text is never elided or archived by ``prepare``. Matching is on whitespace-normalised text."""

    def __init__(self, key_chars: int = 160):
        self.key_chars = key_chars
        self._keys: List[str] = []

    def add(self, text: str) -> None:
        key = _norm(text)[:self.key_chars]
        if len(key) >= 20 and key not in self._keys:
            self._keys.append(key)

    def matches(self, content: Any) -> bool:
        if not self._keys or not isinstance(content, str):
            return False
        head = _norm(content[:self.key_chars * 4 + 4000])
        return any(k in head for k in self._keys)

    def __bool__(self) -> bool:
        return bool(self._keys)


def clip_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    h = limit * 2 // 3
    return text[:h] + f"\n[... {len(text) - limit} chars omitted ...]\n" + text[-(limit - h):]


def estimate_tokens(chars: int) -> int:
    return chars // 3 + 1


def messages_chars(messages: List[Dict[str, Any]]) -> int:
    n = 0
    for m in messages:
        n += content_chars(m.get("content") or "")
        for c in m.get("tool_calls") or []:
            n += len(c["function"].get("arguments") or "")
    return n


@runtime_checkable
class ContextManager(Protocol):
    def prepare(self, messages: List[Dict[str, Any]], *, target_tokens: int, estimated_tokens: int) -> int:
        """Shrink ``messages`` in place toward ``target_tokens``. Return how many messages were changed."""
        ...

    def on_context_error(self, messages: List[Dict[str, Any]], attempt: int) -> int:
        """The server rejected the prompt as too long. Compact harder (``attempt`` is 1, 2, 3). Return changes."""
        ...

    def admit(self, tool: str, text: str, max_chars: int) -> str:
        """Called for every tool result before it enters the conversation. Return the text to append. A manager
        that stores content should keep the full ``text`` and return a shortened version that says how to get it."""
        ...

    def tools(self) -> List[ToolPlugin]:
        """Extra read-only tools to give the model (may be empty)."""
        ...


@runtime_checkable
class ContextAware(Protocol):
    """Optional extension a manager may implement; the loop uses it when present (``getattr``) and works without it."""

    def is_placeholder(self, value: Any) -> bool:
        """Whether ``value`` is this manager's own stand-in for removed content (the loop refuses to run an action built from one)."""
        ...

    def visibility(self, messages: List[Dict[str, Any]], tool_call_id: str) -> ResultView:
        """What the model can see, in the CURRENT messages, of the result of ``tool_call_id``."""
        ...


def elide_old_tool_outputs(messages: List[Dict[str, Any]], keep_last: int = 4,
                           keep_if: Optional[Callable[[str], bool]] = None) -> int:
    idx = [i for i, m in enumerate(messages) if m.get("role") == "tool" and m.get("content") != ELIDED
           and not (keep_if and keep_if(m.get("content")))]
    n = 0
    for i in idx[:-keep_last] if keep_last else idx:
        messages[i] = {**messages[i], "content": ELIDED}
        n += 1
    return n


def shrink_tool_call_args(messages: List[Dict[str, Any]], keep_last: int = 2, limit: int = 400) -> int:
    """Shorten big string arguments (e.g. a whole document passed to write_file) in all but the last
    ``keep_last`` assistant messages. The arguments stay valid JSON with their keys."""
    idx = [i for i, m in enumerate(messages) if m.get("role") == "assistant" and m.get("tool_calls")]
    n = 0
    for i in idx[:-keep_last] if keep_last else idx:
        calls = []
        for c in messages[i]["tool_calls"]:
            try:
                args = json.loads(c["function"]["arguments"] or "{}")
            except ValueError:
                calls.append(c)
                continue
            small = ({k: (v[:limit // 2] + f"...[{len(v) - limit // 2} chars omitted to save context]"
                          if isinstance(v, str) and len(v) > limit else v) for k, v in args.items()}
                     if isinstance(args, dict) else args)
            if small != args:
                n += 1
            calls.append({**c, "function": {**c["function"], "arguments": json.dumps(small)}})
        messages[i] = {**messages[i], "tool_calls": calls}
    return n


class NoCompactionManager:
    """No proactive context management: nothing is elided, archived or summarised while the prompt fits. Only when the server
    rejects a prompt as too long (or the last prompt used >95% of the window) does the loop fall back to the same last-resort
    clipping ``BasicContextManager`` uses, so a run does not die where a stock harness would carry on. Tool output is still
    capped per call (``admit``), identically to ``BasicContextManager``. Used to measure what proactive management costs or buys."""

    def __init__(self):
        self.pins = PinSet()

    def pin_text(self, text: str) -> None:
        self.pins.add(text)

    def prepare(self, messages, *, target_tokens, estimated_tokens):
        return 0

    def on_context_error(self, messages, attempt):
        return BasicContextManager(self.pins).on_context_error(messages, attempt)

    def admit(self, tool, text, max_chars):
        return clip_text(text, max_chars)

    def tools(self):
        return []

    def is_placeholder(self, value):
        return placeholder_in(value)

    def visibility(self, messages, tool_call_id):
        return basic_visibility(messages, tool_call_id)


class BasicContextManager:
    """Default manager: lossy elision, no storage, no extra tools. ``pin_text`` protects named texts from elision (not from the
    last-resort clipping at attempt 3, which exists so a run does not die)."""

    def __init__(self, pins: Optional[PinSet] = None):
        self.pins = pins if pins is not None else PinSet()

    def pin_text(self, text: str) -> None:
        self.pins.add(text)

    def prepare(self, messages, *, target_tokens, estimated_tokens):
        return shrink_tool_call_args(messages) + elide_old_tool_outputs(messages, keep_if=self.pins.matches)

    def on_context_error(self, messages, attempt):
        n = elide_old_tool_outputs(messages, keep_last=4 if attempt <= 1 else 1,
                                   keep_if=self.pins.matches if attempt <= 2 else None)
        n += shrink_tool_call_args(messages, keep_last=2 if attempt <= 1 else 1)
        if attempt >= 3:
            for i, m in enumerate(messages):
                if i > 1 and isinstance(m.get("content"), str) and len(m["content"]) > 1500:
                    messages[i] = {**m, "content": clip_text(m["content"], 1500)}
                    n += 1
        return n

    def admit(self, tool, text, max_chars):
        return clip_text(text, max_chars)

    def tools(self):
        return []

    def is_placeholder(self, value):
        return placeholder_in(value)

    def visibility(self, messages, tool_call_id):
        return basic_visibility(messages, tool_call_id)
