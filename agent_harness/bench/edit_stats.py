"""Edit failure and retry categories from the per-attempt edit log (``/work/.artifacts/edits.jsonl``, written by the parity tools).

Each row: ``seq`` (order of file-changing tool calls), ``tool`` (edit_file | write_file), ``path``, ``ok``, ``category`` (see ``edit.edit_text_ex``).
Every failed edit is followed through the later calls on the same path and given an outcome: ``retried_ok`` (a later edit_file on that path succeeded),
``retried_failed`` (later edit_file calls on it failed again and none succeeded), ``rewritten`` (a write_file replaced the file before any later edit succeeded), ``abandoned`` (nothing further). Order matters: the first later event on the path decides.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def load(path: Path) -> List[dict]:
    try:
        return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    except (OSError, ValueError):
        return []


def edit_stats(rows: List[dict]) -> Dict[str, Any]:
    edits = [r for r in rows if r.get("tool") == "edit_file"]
    failures = [r for r in edits if not r.get("ok")]
    outcomes: Counter = Counter()
    for f in failures:
        later = [r for r in rows if r["seq"] > f["seq"] and r.get("path") == f.get("path")]
        outcome, failed_again = "abandoned", False
        for r in later:                                  # what happened to this file next, in order
            if r["tool"] == "write_file":
                outcome = "rewritten"
                break
            if r["ok"]:
                outcome = "retried_ok"
                break
            failed_again = True
        else:
            outcome = "retried_failed" if failed_again else "abandoned"
        outcomes[outcome] += 1
    return {"edit_calls": len(edits), "edit_ok": len(edits) - len(failures), "edit_failed": len(failures),
            "failure_categories": dict(Counter(r.get("category") for r in failures)), "failure_outcomes": dict(outcomes),
            "write_file_calls": sum(1 for r in rows if r.get("tool") == "write_file")}
