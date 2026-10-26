"""Offline replay of recorded runs through the execution supervisor (no model calls, no tools).

Reads a run's ``trace<N>/`` directory (full tool-call arguments and results from the trace blobs) and feeds the recorded operations to a ``Supervisor`` in order.
It reports where the supervisor would first have intervened, how many requests it would have refused, and whether recovery would have been EXHAUSTED, which is the
only thing that ends a run. It is a counterfactual on a recorded sequence: the model's behaviour after an intervention is unknown, and the workspace is
reconstructed from the recorded calls (``write_file``/``edit_file``/``web_download`` change the named file; a shell command with a write marker changes an
unknown scratch file, or an output file if the command names ``output/``), so progress is approximated. Use it to look for FALSE STOPS (exhaustion before a
run that later delivered), not to estimate recovery.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..supervisor import Supervisor, SupervisorConfig

_WRITE_MARK = re.compile(r"(>>?\s*['\"]?(?:/work|\.?/?[\w.-]+\.(?:csv|txt|json|md|xlsx|docx|py|sh|log))|open\([^)]*['\"][wax]b?['\"]|\.write(?:text|bytes)?\(|\.save\(|to_excel|to_csv|to_json|savefig|\bmkdir\b|\bcp\b|\bmv\b|\brm\b|\btee\b)")


@dataclass
class Step:
    turn: int
    name: str
    args: Dict[str, Any]
    text: str
    ok: bool
    call_id: str = ""


def _blob(d: Path, sha: Optional[str]) -> str:
    if not sha:
        return ""
    p = d / "blobs" / f"{sha}.txt"
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


def load_steps(trace_dir: Path) -> List[Step]:
    d = Path(trace_dir)
    rows = [json.loads(l) for l in (d / "trace.jsonl").read_text().splitlines() if l.strip()]
    steps = []
    for r in rows:
        if r.get("kind") != "tool_full":
            continue
        raw = _blob(d, r.get("args_sha"))
        try:
            args = json.loads(raw) if raw else {}
        except ValueError:
            args = {"_raw": raw}
        text = _blob(d, r.get("out_sha"))
        steps.append(Step(r.get("turn", len(steps) + 1), r.get("name", "?"), args if isinstance(args, dict) else {"_raw": raw}, text, not text.startswith("ERROR:"), f"s{len(steps)}"))
    return steps


def _rel(path: str) -> str:
    return re.sub(r"^/work/", "", str(path)).lstrip("/")


def replay(steps: List[Step], cfg: Optional[SupervisorConfig] = None, *, max_turns: int = 200, max_seconds: float = 3600.0) -> Dict[str, Any]:
    snap: Dict[str, str] = {}
    sup = Supervisor(cfg or SupervisorConfig(), max_turns=max_turns, max_seconds=max_seconds, probe=lambda: dict(snap))
    refused = first = None
    n_refused = 0
    for i, s in enumerate(steps):
        sup.begin_turn(s.turn, 0.0)
        d = sup.before(s.call_id, s.name, s.args, [])
        if not d.run:
            n_refused += 1
            first = s.turn if first is None else first
            if d.exhausted:
                refused = s.turn
                break
            continue
        if s.ok and s.name in ("write_file", "edit_file", "web_download") and s.args.get("path"):
            snap[_rel(s.args["path"])] = hashlib.sha1(json.dumps(s.args, sort_keys=True).encode()).hexdigest()[:12]
        elif s.name == "bash" and _WRITE_MARK.search(str(s.args.get("command", ""))):
            where = "output" if "output/" in str(s.args.get("command", "")) else "scratch"
            snap[f"{where}/_bash_{i}"] = hashlib.sha1(str(i).encode()).hexdigest()[:12]
        sup.after(s.call_id, s.name, s.args, s.text, s.ok)
    return {"steps": len(steps), "first_refusal_turn": first, "refusals": n_refused, "exhausted_at_turn": refused, "report": sup.report()}


def replay_dir(trace_dir: Path, cfg: Optional[SupervisorConfig] = None, **kw) -> Dict[str, Any]:
    return replay(load_steps(trace_dir), cfg, **kw)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Replay recorded runs through the execution supervisor")
    ap.add_argument("trace_dirs", nargs="+")
    a = ap.parse_args(argv)
    for t in a.trace_dirs:
        r = replay_dir(Path(t))
        print(json.dumps({"trace": t, **{k: r[k] for k in ("steps", "first_refusal_turn", "refusals", "exhausted_at_turn")}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
