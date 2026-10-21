"""Multi-turn tool-calling agent loop with hard limits, context management and a JSONL trajectory.

``run_loop`` sends the conversation to a ``ChatClient``, runs requested tools through a ``ToolRegistry`` (so the
approval gate, authorizer and audit log apply), appends the results and repeats until the model answers without
tool calls, or a limit is hit. Limits: ``max_turns`` model calls, ``max_seconds`` wall clock, ``max_context_tokens``
(old tool outputs are elided, oldest first, when the last prompt grows past ``compact_at`` of it).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .guardrails import LoopGuard
from .llm import ChatClient, ChatError
from .tools import ToolRegistry

ELIDED = "[earlier tool output removed to save context]"


@dataclass
class LoopResult:
    stop: str                     # done | max_turns | max_seconds | model_error | context_overflow
    turns: int = 0
    seconds: float = 0.0
    final_text: str = ""
    prompt_tokens: int = 0        # summed over turns
    completion_tokens: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    messages: List[Dict[str, Any]] = field(default_factory=list)


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    h = limit * 2 // 3
    return text[:h] + f"\n[... {len(text) - limit} chars omitted ...]\n" + text[-(limit - h):]


def elide_old_tool_outputs(messages: List[Dict[str, Any]], keep_last: int = 4) -> int:
    """Replace the oldest tool messages with a stub, keeping the last ``keep_last``. Returns how many changed."""
    idx = [i for i, m in enumerate(messages) if m.get("role") == "tool" and m.get("content") != ELIDED]
    n = 0
    for i in idx[:-keep_last] if keep_last else idx:
        messages[i] = {**messages[i], "content": ELIDED}
        n += 1
    return n


def is_context_error(text: str) -> bool:
    t = text.lower()
    return "context length" in t or "context_length" in t or "too many tokens" in t or "maximum context" in t


def compact(messages: List[Dict[str, Any]], level: int) -> None:
    """Shrink the conversation. 1: elide old tool outputs (keep 4). 2: keep 1. 3: also clip every remaining
    message body to 1,500 chars (the system prompt and the task statement are never touched)."""
    elide_old_tool_outputs(messages, keep_last=4 if level <= 1 else 1)
    shrink_tool_call_args(messages, keep_last=2 if level <= 1 else 1)
    if level >= 3:
        for i, m in enumerate(messages):
            if i > 1 and isinstance(m.get("content"), str) and len(m["content"]) > 1500:
                messages[i] = {**m, "content": _clip(m["content"], 1500)}


def shrink_tool_call_args(messages: List[Dict[str, Any]], keep_last: int = 2, limit: int = 400) -> int:
    """Shorten big string arguments (e.g. the whole document passed to write_file) in all but the last ``keep_last``
    assistant messages. The arguments stay valid JSON with their keys, so the transcript still parses."""
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
            small = {k: (v[:limit // 2] + f"...[{len(v) - limit // 2} chars omitted to save context]" if isinstance(v, str) and len(v) > limit else v)
                     for k, v in args.items()} if isinstance(args, dict) else args
            if small != args:
                n += 1
            calls.append({**c, "function": {**c["function"], "arguments": json.dumps(small)}})
        messages[i] = {**messages[i], "tool_calls": calls}
    return n


def estimate_tokens(chars: int) -> int:
    return chars // 3 + 1


def missing_args(registry: ToolRegistry, name: str, args: Dict[str, Any]) -> Optional[str]:
    try:
        spec = registry.get(name).spec()
    except Exception:
        return None            # unknown tool: the registry produces the error
    params = spec.parameters or {}
    required = [r for r in params.get("required", []) if r not in args]
    if not required:
        return None
    props = params.get("properties", {})
    expected = ", ".join(f"{k} ({(v or {}).get('type', 'any')})" for k, v in props.items())
    return (f"missing required argument(s): {', '.join(required)}. {name} expects: {expected}. "
            f"You sent: {', '.join(args) or 'no arguments'}.")


def run_loop(client: ChatClient, registry: ToolRegistry, system: str, task: str, *, max_turns: int = 200,
             max_seconds: float = 3600, max_context_tokens: int = 40960, compact_at: float = 0.7,
             max_tool_chars: Optional[int] = None, trajectory: Optional[Path] = None,
             on_turn: Optional[Callable[[int, Dict[str, Any]], None]] = None) -> LoopResult:
    if max_tool_chars is None:
        max_tool_chars = min(12000, int(0.2 * max_context_tokens))
    messages: List[Dict[str, Any]] = [{"role": "system", "content": system}, {"role": "user", "content": task}]
    res = LoopResult(stop="done", messages=messages)
    guard = LoopGuard(max_turns)
    t0 = time.time()
    specs = registry.specs()
    log = open(trajectory, "a", encoding="utf-8") if trajectory else None

    def emit(kind: str, **kw: Any) -> None:
        if log:
            log.write(json.dumps({"t": round(time.time() - t0, 1), "kind": kind, **kw}) + "\n")
            log.flush()

    last_prompt, sent_len = 0, 0
    fails: Dict[str, int] = {}
    try:
        while True:
            if not guard.check():
                res.stop = "max_turns"
                break
            if time.time() - t0 > max_seconds:
                res.stop = "max_seconds"
                break
            # size of the next prompt: the last measured one plus everything appended since that reply
            new_chars = sum(len(m.get("content") or "") for m in messages[sent_len:])
            if last_prompt + estimate_tokens(new_chars) > compact_at * max_context_tokens:
                shrunk = shrink_tool_call_args(messages)
                if elide_old_tool_outputs(messages) == 0 and shrunk == 0 and last_prompt > 0.95 * max_context_tokens:
                    res.stop = "context_overflow"
                    break
            out, attempt = None, 0
            while out is None:
                try:
                    out = client.complete(messages, specs)
                except ChatError as e:
                    if is_context_error(str(e)) and attempt < 3:
                        attempt += 1
                        compact(messages, attempt)
                        emit("compact", level=attempt, reason="context error from server")
                        continue
                    emit("model_error", error=str(e)[:300])
                    res.stop = "model_error"
                    break
            if out is None:
                break
            res.turns += 1
            msg, usage = out["message"], out["usage"]
            last_prompt = int(usage.get("prompt_tokens") or 0)
            res.prompt_tokens += last_prompt
            res.completion_tokens += int(usage.get("completion_tokens") or 0)
            keep = {k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")}
            keep["role"] = "assistant"
            keep["content"] = msg.get("content") or ""
            messages.append(keep)
            sent_len = len(messages)
            emit("assistant", turn=res.turns, content=(keep["content"] or "")[:2000], usage=usage,
                 finish=out.get("finish_reason"), tool_calls=[(c["function"]["name"], c["function"]["arguments"][:300])
                                                              for c in msg.get("tool_calls") or []])
            if on_turn:
                on_turn(res.turns, out)
            calls = msg.get("tool_calls") or []
            if not calls:
                res.final_text = keep["content"]
                break
            for c in calls:
                name = c["function"]["name"]
                try:
                    args = json.loads(c["function"]["arguments"] or "{}")
                    if not isinstance(args, dict):
                        raise ValueError("arguments must be an object")
                    bad = missing_args(registry, name, args)
                    if bad:
                        text, res.tool_errors = f"ERROR: {bad}", res.tool_errors + 1
                    else:
                        r = registry.run(name, args)
                        err = (r.error or "").removeprefix("ERROR: ").removeprefix("ERROR:").strip()
                        text = r.text if r.ok else f"ERROR: {err}"
                        res.tool_errors += 0 if r.ok else 1
                except ValueError as e:
                    text, res.tool_errors = f"ERROR: bad tool arguments: {e}", res.tool_errors + 1
                res.tool_calls += 1
                fails[name] = fails.get(name, 0) + 1 if text.startswith("ERROR:") else 0
                if fails[name] >= 6 and text.startswith("ERROR:"):
                    text += (f"\n[note: {name} has failed {fails[name]} times in a row. Stop retrying it; work with the "
                             f"information and files you already have and finish the task.]")
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": _clip(text, max_tool_chars)})
                emit("tool", name=name, ok=not text.startswith("ERROR:"), out=text[:400])
    finally:
        res.seconds = round(time.time() - t0, 1)
        emit("end", stop=res.stop, turns=res.turns, seconds=res.seconds)
        if log:
            log.close()
    return res
