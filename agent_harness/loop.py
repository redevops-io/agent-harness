"""Multi-turn tool-calling agent loop with hard limits, context management and a JSONL trajectory.

``run_loop`` sends the conversation to a ``ChatClient``, runs requested tools through a ``ToolRegistry`` (so the
approval gate, authorizer and audit log apply), appends the results and repeats until the model answers without
tool calls, or a limit is hit. Limits: ``max_turns`` model calls, ``max_seconds`` wall clock, ``max_context_tokens``
(old tool outputs are compacted by the context manager when the last prompt grows past ``compact_at`` of it).

Ownership: the loop owns dispatch and the message list; the ``ContextManager`` (Context Runtime in the benchmark) owns what the model sees of old content;
the ``Supervisor`` (agent_harness.supervisor) is the single authority on repeated operations, recovery, budget and completion; tools own their own output size.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .guardrails import LoopGuard
from .llm import ChatClient, ChatError
from .context import (ELIDED, BasicContextManager, ContextManager, clip_text as _clip, elide_old_tool_outputs,  # noqa: F401
                      estimate_tokens, messages_chars, placeholder_arg, shrink_tool_call_args)
from .integrity import check_message_integrity
from .multimodal import image_message, prune_images
from .supervisor import Decision, Supervisor, SupervisorConfig
from .trace import TraceRecorder
from .tools import ToolRegistry

@dataclass
class Supervision:
    """How to supervise a run: the policy, how to observe the workspace, and how to check deliverables. ``probe() -> {relative path: content digest}`` makes progress
    detection content-based; ``checker() -> DeliverableReport`` enables the completion checkpoint. Both are optional."""
    cfg: SupervisorConfig = field(default_factory=SupervisorConfig)
    probe: Optional[Callable[[], Dict[str, str]]] = None
    checker: Optional[Callable[[], Any]] = None


@dataclass
class LoopResult:
    stop: str                     # done | verified_complete | max_turns | max_seconds | model_error | context_overflow | blocked | integrity_error
    turns: int = 0
    seconds: float = 0.0
    final_text: str = ""
    prompt_tokens: int = 0        # summed over turns
    completion_tokens: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    messages: List[Dict[str, Any]] = field(default_factory=list)
    supervisor: Optional[Dict[str, Any]] = None   # Supervisor.report(): families, suppressions, budget notices, completion status


def is_context_error(text: str) -> bool:
    t = text.lower()
    return "context length" in t or "context_length" in t or "too many tokens" in t or "maximum context" in t


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
             on_turn: Optional[Callable[[int, Dict[str, Any]], None]] = None,
             context_manager: Optional[ContextManager] = None, compact_to: float = 0.5,
             trace: Optional[TraceRecorder] = None, supervision: Optional[Supervision] = Supervision(),
             check_integrity: bool = False, keep_images: int = 2) -> LoopResult:
    """``supervision=None`` turns the supervisor off (a bare loop, for ablations). ``check_integrity`` validates the message list before every model call and
    ends the run as ``integrity_error`` rather than send a corrupt history."""
    if max_tool_chars is None:
        max_tool_chars = min(12000, int(0.2 * max_context_tokens))
    messages: List[Dict[str, Any]] = [{"role": "system", "content": system}, {"role": "user", "content": task}]
    res = LoopResult(stop="done", messages=messages)
    if trace:
        trace.add(0, messages[0])
        trace.add(1, messages[1])
    guard = LoopGuard(max_turns)
    t0 = time.time()
    cm = context_manager or BasicContextManager()
    for extra in cm.tools():            # e.g. a retrieval tool over content the manager compacted away
        registry.register(extra)
    specs = registry.specs()
    log = open(trajectory, "a", encoding="utf-8") if trajectory else None

    def emit(kind: str, **kw: Any) -> None:
        if log:
            log.write(json.dumps({"t": round(time.time() - t0, 1), "kind": kind, **kw}) + "\n")
            log.flush()

    sup: Optional[Supervisor] = None
    if supervision is not None:
        tool_params = {(sp.get("function") or sp).get("name"): (sp.get("function") or sp).get("parameters") or {} for sp in specs}
        sup = Supervisor(supervision.cfg, max_turns=max_turns, max_seconds=max_seconds, context=cm, probe=supervision.probe, checker=supervision.checker,
                         specs=tool_params, emit=emit)
    is_ph = getattr(cm, "is_placeholder", None)
    last_prompt, sent_len = 0, 0
    try:
        while True:
            if not guard.check():
                res.stop = "max_turns"
                break
            if time.time() - t0 > max_seconds:
                res.stop = "max_seconds"
                break
            if sup:
                sup.begin_turn(res.turns + 1, time.time() - t0)
            # size of the next prompt: the last measured one plus everything appended since that reply
            new_chars = messages_chars(messages[sent_len:])
            est = last_prompt + estimate_tokens(new_chars)
            if est > compact_at * max_context_tokens:
                pruned = prune_images(messages, keep_images)
                changed = cm.prepare(messages, target_tokens=int(compact_to * max_context_tokens), estimated_tokens=est) + pruned
                if changed:
                    emit("compact", level=0, changed=changed, estimated_before=est)
                elif last_prompt > 0.95 * max_context_tokens and cm.on_context_error(messages, 3) == 0:
                    res.stop = "context_overflow"
                    break
            if check_integrity:
                bad = check_message_integrity(messages)
                if bad:
                    emit("integrity_violation", problems=bad[:5])
                    res.stop = "integrity_error"
                    break
            if trace:
                trace.projection(res.turns + 1, messages, est, last_prompt)
            out, attempt = None, 0
            while out is None:
                try:
                    out = client.complete(messages, specs)
                except ChatError as e:
                    if is_context_error(str(e)) and attempt < 3:
                        attempt += 1
                        emit("compact", level=attempt, reason="context error from server",
                             changed=cm.on_context_error(messages, attempt))
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
            if trace:
                trace.add(len(messages) - 1, keep)
                trace.assistant(res.turns, keep, usage, out.get("finish_reason"))
            sent_len = len(messages)
            emit("assistant", turn=res.turns, content=(keep["content"] or "")[:2000], usage=usage,
                 finish=out.get("finish_reason"), tool_calls=[(c["function"]["name"], c["function"]["arguments"][:300])
                                                              for c in msg.get("tool_calls") or []])
            if on_turn:
                on_turn(res.turns, out)
            calls = msg.get("tool_calls") or []
            if not calls:
                res.final_text = keep["content"]
                fd = sup.on_finish_attempt() if sup else None
                if fd is not None and fd.action == "push":
                    messages.append({"role": "user", "content": fd.text})
                    if trace:
                        trace.add(len(messages) - 1, messages[-1])
                    sent_len = len(messages) - 1
                    continue
                break
            images: List[Any] = []
            for c in calls:
                name = c["function"]["name"]
                call_id = c["id"]
                args: Dict[str, Any] = {}
                try:
                    args = json.loads(c["function"]["arguments"] or "{}")
                    if not isinstance(args, dict):
                        raise ValueError("arguments must be an object")
                    bad = missing_args(registry, name, args)
                    ph = None if bad else placeholder_arg(args, is_ph)
                    decision = Decision(True)
                    if bad:
                        text, res.tool_errors = f"ERROR: {bad}", res.tool_errors + 1
                    elif ph:
                        text, res.tool_errors = (f"ERROR: argument '{ph}' is a context placeholder, not real content (older content was "
                                                 f"shortened in your history to save space). Nothing was run. Write the real content, or "
                                                 f"get the original text back first (read_archive / re-read the source file)."), res.tool_errors + 1
                        emit("placeholder_blocked", name=name, arg=ph)
                    elif sup and not (decision := sup.before(call_id, name, args, messages)).run:
                        text, res.tool_errors = sup.attach_pending(decision.text), res.tool_errors + 1
                    else:
                        r = registry.run(name, args)
                        err = (r.error or "").removeprefix("ERROR: ").removeprefix("ERROR:").strip()
                        text = r.text if r.ok else f"ERROR: {err}"
                        res.tool_errors += 0 if r.ok else 1
                        images += list(getattr(r, "attachments", None) or [])
                        if sup:
                            text = sup.after(call_id, name, args, text, r.ok)
                except ValueError as e:
                    text, res.tool_errors = f"ERROR: bad tool arguments: {e}", res.tool_errors + 1
                res.tool_calls += 1
                admitted = cm.admit(name, text, max_tool_chars)
                messages.append({"role": "tool", "tool_call_id": call_id, "content": admitted})
                if trace:
                    trace.add(len(messages) - 1, messages[-1])
                    trace.tool_result(res.turns, name, args, text, admitted)
                emit("tool", name=name, ok=not text.startswith("ERROR:"), out=text[:400])
            if images:
                messages.append(image_message(images))
                if trace:
                    trace.add(len(messages) - 1, messages[-1])
                emit("images", n=len(images), paths=[a.path for a in images])
            if sup:
                reason = sup.stop_reason()
                if reason:
                    res.stop = reason
                    emit("supervisor", action="stop", reason=reason)
                    break
    finally:
        if sup:
            res.supervisor = sup.report()
        res.seconds = round(time.time() - t0, 1)
        emit("end", stop=res.stop, turns=res.turns, seconds=res.seconds)
        if log:
            log.close()
        if trace:
            trace.close()
    return res
