"""Multi-turn tool-calling agent loop with hard limits, context management and a JSONL trajectory.

``run_loop`` sends the conversation to a ``ChatClient``, runs requested tools through a ``ToolRegistry`` (so the
approval gate, authorizer and audit log apply), appends the results and repeats until the model answers without
tool calls, or a limit is hit. Limits: ``max_turns`` model calls, ``max_seconds`` wall clock, ``max_context_tokens``
(old tool outputs are compacted by the context manager when the last prompt grows past ``compact_at`` of it).

Ownership: the loop owns dispatch and the message list; the ``ContextManager`` (Context Runtime in the benchmark) owns what the model sees of old content;
the ``Supervisor`` (agent_harness.supervisor) is the single authority on repeated operations, recovery, budget and completion; tools own their own output size.
"""
from __future__ import annotations

import contextlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Set, Any, Callable, Dict, List, Optional

from .guardrails import LoopGuard
from .llm import ChatClient, ChatError
from .context import (ELIDED, BasicContextManager, ContextManager, clip_text as _clip, elide_old_tool_outputs,  # noqa: F401
                      estimate_tokens, messages_chars, placeholder_arg, shrink_tool_call_args)
from .integrity import check_message_integrity
from .process_runner import run_control
from .multimodal import ImageCapError, enforce_image_cap, image_message, parse_image_limit, prune_images
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
    inputs: Sequence[str] = ()                  # names of the task's input files (for the opt-in re-read check)


@dataclass
class LoopResult:
    stop: str                     # done | verified_complete | max_turns | max_seconds | cancelled | model_error | context_overflow | blocked | integrity_error | request_invalid
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


REFUSED_ARGS_NOTE = "repeat of an earlier call, refused, not run"


def shrink_refused_calls(messages: List[Dict[str, Any]], refused_ids: Set[str], keep_full: int = 2, min_chars: int = 160) -> int:
    """Replace the arguments of OLD refused tool calls by a one-line note. A refusal loop otherwise grows the prompt by the size of the repeated call every turn (about
    380 tokens a turn in the statisticians run, 130 turns, until the window overflowed) and nothing else can compact it. Only assistant messages all of whose calls were
    refused are touched; the newest ``keep_full`` stay intact so the model still sees what was refused; indices, roles and call ids are unchanged. Returns messages changed."""
    idx = [i for i, m in enumerate(messages) if m.get("role") == "assistant" and m.get("tool_calls") and all(c.get("id") in refused_ids for c in m["tool_calls"])]
    changed = 0
    for i in (idx[:-keep_full] if keep_full else idx):
        new, hit = [], False
        for c in messages[i]["tool_calls"]:
            if len(c["function"].get("arguments") or "") > min_chars:
                c = {**c, "function": {**c["function"], "arguments": json.dumps({"note": REFUSED_ARGS_NOTE})}}
                hit = True
            new.append(c)
        if hit:
            messages[i] = {**messages[i], "tool_calls": new}
            changed += 1
    return changed


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
             check_integrity: bool = False, keep_images: int = 2, max_images: Optional[int] = None,
             cancel: Optional[Callable[[], bool]] = None) -> LoopResult:
    """``supervision=None`` turns the supervisor off (a bare loop, for ablations). ``check_integrity`` validates the message list before every model call and
    ends the run as ``integrity_error`` rather than send a corrupt history. ``max_images`` is the endpoint's limit of images per prompt: enforced on every request
    (oldest images become labelled text notes), never more than that many images attached in one turn, and learned from the endpoint's own error if it is not given.
    ``max_seconds`` and ``cancel`` (a cooperative ``() -> bool``) apply during a tool call too, not only between turns: a process a tool starts through
    ``agent_harness.sandbox`` is stopped, with everything it started, when the time is up or ``cancel()`` is true, and the run ends as ``max_seconds`` or ``cancelled``."""
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
                         specs=tool_params, emit=emit, inputs=supervision.inputs)
    is_ph = getattr(cm, "is_placeholder", None)
    last_prompt, sent_len = 0, 0
    refused: Set[str] = set()                       # call ids the supervisor refused (their old arguments are shrunk, see shrink_refused_calls)
    shrunk = 0
    controls = contextlib.ExitStack()
    controls.enter_context(run_control(max_seconds=max_seconds, cancel=cancel))
    try:
        while True:
            if not guard.check():
                res.stop = "max_turns"
                break
            if cancel is not None and cancel():
                res.stop = "cancelled"
                break
            if time.time() - t0 > max_seconds:
                res.stop = "max_seconds"
                break
            if sup:
                sup.begin_turn(res.turns + 1, time.time() - t0)
            # size of the next prompt: the last measured one plus everything appended since that reply
            n_shrunk = shrink_refused_calls(messages, refused) if refused else 0
            if n_shrunk:
                shrunk += n_shrunk
                if shrunk <= 3 or shrunk % 10 == 0:
                    emit("refusal_shrink", changed=n_shrunk, total=shrunk)
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
                if max_images:
                    try:
                        dropped = enforce_image_cap(messages, max_images)
                    except ImageCapError as e:
                        emit("model_error", error=f"request refused locally: {e}"[:300])
                        res.stop = "request_invalid"
                        break
                    if dropped:
                        emit("image_cap", cap=max_images, dropped=[d["label"] for d in dropped])
                try:
                    out = client.complete(messages, specs)
                except ChatError as e:
                    limit = parse_image_limit(str(e))
                    if limit and (not max_images or limit < max_images):
                        emit("image_cap", cap=limit, learned_from="server error", was=max_images)
                        max_images = limit
                        continue
                    if is_context_error(str(e)) and attempt < 3:
                        attempt += 1
                        cut = shrink_refused_calls(messages, refused, keep_full=0) if refused else 0
                        emit("compact", level=attempt, reason="context error from server",
                             changed=cm.on_context_error(messages, attempt) + cut, refusal_pairs_shrunk=cut)
                        continue
                    emit("model_error", error=str(e)[:300])
                    res.stop = "context_overflow" if is_context_error(str(e)) else "model_error"
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
                    elif cancel is not None and cancel():
                        text, res.tool_errors = "ERROR: cancelled; not run.", res.tool_errors + 1
                    elif sup and not (decision := sup.before(call_id, name, args, messages)).run:
                        text, res.tool_errors = sup.attach_pending(decision.text), res.tool_errors + 1
                        refused.add(call_id)
                    else:
                        r = registry.run(name, args)
                        err = (r.error or "").removeprefix("ERROR: ").removeprefix("ERROR:").strip()
                        text = r.text if r.ok else f"ERROR: {err}"
                        res.tool_errors += 0 if r.ok else 1
                        atts = list(getattr(r, "attachments", None) or [])
                        if max_images and atts and len(images) + len(atts) > max_images:
                            text = (f"NOT attached: at most {max_images} images can be shown at once and this turn already attaches {len(images)}. "
                                    f"Write down what you read from the images you have seen (a notes file in /work), then view {', '.join(a.path for a in atts)} in a later call.")
                            emit("image_cap", cap=max_images, refused_attachments=[a.path for a in atts])
                        else:
                            images += atts
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
                messages.append(image_message(images, max_images))
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
        controls.close()
        if sup:
            res.supervisor = sup.report()
        res.seconds = round(time.time() - t0, 1)
        emit("end", stop=res.stop, turns=res.turns, seconds=res.seconds)
        if log:
            log.close()
        if trace:
            trace.close()
    return res
