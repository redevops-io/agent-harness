"""Deterministic failure classification of one attempt, from its trajectory and summary (no model, no judge).

Classes (first match wins): ``delivered``; ``blocked`` (the recovery budget ended the run); ``identical_call_loop`` (at least a fifth of the turns were refused or
repeated identical calls); ``time_limit``; ``turn_limit``; ``model_error``; ``context_overflow``; ``finished_without_deliverable``. Flags are independent signals:
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
        elif k == "repeat":
            repeats += 1
        elif k == "recovery" and r.get("blocked"):
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
             "recovery_blockers": [b.get("kind") for b in ((summary.get("recovery") or {}).get("blockers") or [])]}
    if summary.get("deliverables"):
        cls = "delivered"
    elif stop == "blocked":
        cls = "blocked"
    elif turns and loop >= max(5, turns // 5):
        cls = "identical_call_loop"
    elif stop == "max_seconds":
        cls = "time_limit"
    elif stop == "max_turns":
        cls = "turn_limit"
    elif stop == "model_error":
        cls = "model_error"
    elif stop == "context_overflow":
        cls = "context_overflow"
    else:
        cls = "finished_without_deliverable"
    return {"class": cls, **flags}
