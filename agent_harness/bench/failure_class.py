"""Deterministic failure classification of one attempt, from its trajectory and summary (no model, no judge).

Classes (first match wins): ``delivered`` (files present and structurally valid); ``invalid_deliverable`` (files present but empty, unreadable, draft-only or incomplete); ``blocked`` (the supervisor's recovery was exhausted); ``verified_complete`` is a ``delivered`` run the supervisor ended; ``identical_call_loop`` (at least a fifth of the turns were refused or
repeated identical calls); ``time_limit``; ``turn_limit``; ``model_error``; ``request_invalid`` (the harness refused to send a request the endpoint would reject); ``context_overflow``; ``finished_without_deliverable``. Flags are independent signals:
``files_outside_output``, ``network_errors``, ``placeholder_blocked``, ``capability_notes`` and the recovery blockers.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

_NET = re.compile(r"Temporary failure in name resolution|Tunnel connection failed|Network is unreachable|blocked by the egress policy|Name or service not known")


def classify(trajectory: Path, summary: Dict[str, Any]) -> Dict[str, Any]:
    turns = repeats = recovery_blocked = placeholder = net = notes = 0
    stop = summary.get("stop")
    try:
        rows: List[dict] = [json.loads(l) for l in Path(trajectory).read_text().splitlines() if l.strip()]
    except (OSError, ValueError):
        rows = []
    for r in rows:
        k = r.get("kind")
        if k == "assistant":
            turns += 1
        elif k == "supervisor" and r.get("action") in ("suppressed", "pointer"):
            recovery_blocked += 1
        elif k == "placeholder_blocked":
            placeholder += 1
        elif k == "tool":
            out = str(r.get("out") or "")
            net += bool(_NET.search(out))
            notes += "[environment:" in out
        elif k == "end":
            stop = r.get("stop", stop)
    loop = repeats + recovery_blocked
    flags = {"files_outside_output": list(summary.get("outside_output") or []), "network_errors": net, "placeholder_blocked": placeholder,
             "capability_notes": notes, "refused_or_repeated_calls": loop, "turns": turns,
             "supervisor_events": [e.get("kind") for e in ((summary.get("supervisor") or {}).get("events") or [])][:20]}
    comp = (summary.get("completion") or {}).get("status")
    flags["completion_status"] = comp
    if summary.get("deliverables") and comp in (None, "structurally_valid"):
        cls = "delivered"
    elif summary.get("deliverables"):
        cls = "invalid_deliverable"
    elif stop == "blocked":
        cls = "blocked"
    elif turns and loop >= max(5, turns // 5):
        cls = "identical_call_loop"
    elif stop == "max_seconds":
        cls = "time_limit"
    elif stop == "max_turns":
        cls = "turn_limit"
    elif stop in ("model_error", "request_invalid"):
        cls = "model_error" if stop == "model_error" else "request_invalid"
    elif stop == "context_overflow":
        cls = "context_overflow"
    else:
        cls = "finished_without_deliverable"
    return {"class": cls, **flags}
