"""Structural integrity of the message list that is about to be sent to a model. Pure function, no I/O.

Violations are returned as strings (empty list = sound). Checked: the first messages are system then user; every assistant tool call has exactly one tool message
with its id, directly after the assistant message and before the next assistant/user message (image messages from ``view_image`` may follow the tool results);
no tool message without a preceding call; unique call ids; tool content is text; images only in user messages; roles from the known set.
"""
from __future__ import annotations

from typing import Any, Dict, List


def check_message_integrity(messages: List[Dict[str, Any]]) -> List[str]:
    bad: List[str] = []
    if len(messages) < 2 or messages[0].get("role") != "system" or messages[1].get("role") != "user":
        bad.append("conversation must start with a system and a user message")
    seen_ids: set = set()
    pending: List[str] = []
    for i, m in enumerate(messages):
        role = m.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            bad.append(f"message {i}: unknown role {role!r}")
            continue
        if role in ("assistant", "user", "system") and pending:
            if role == "user" and isinstance(m.get("content"), list) and not pending:
                pass
            if not (role == "user" and isinstance(m.get("content"), list)):
                bad.append(f"message {i}: {role} message while tool results are missing for {pending}")
                pending = []
        if role == "assistant":
            calls = m.get("tool_calls") or []
            ids = [c.get("id") for c in calls]
            for cid in ids:
                if cid in seen_ids:
                    bad.append(f"message {i}: duplicate tool call id {cid}")
                seen_ids.add(cid)
            if any(not c.get("function", {}).get("name") for c in calls):
                bad.append(f"message {i}: tool call without a name")
            pending = [c for c in ids if c]
            if isinstance(m.get("content"), list):
                bad.append(f"message {i}: assistant content must be text")
        elif role == "tool":
            cid = m.get("tool_call_id")
            if cid not in pending:
                bad.append(f"message {i}: tool result {cid!r} has no matching pending call")
            else:
                pending.remove(cid)
            if not isinstance(m.get("content"), str):
                bad.append(f"message {i}: tool content must be text")
        elif role == "system" and i > 0:
            bad.append(f"message {i}: system message after the start")
        if role != "user" and isinstance(m.get("content"), list):
            bad.append(f"message {i}: content parts are only allowed in user messages")
    if pending:
        bad.append(f"end of conversation: no tool result for {pending}")
    return bad
