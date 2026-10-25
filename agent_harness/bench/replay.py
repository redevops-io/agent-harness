"""Offline replay of recorded runs through the progress tracker (no model calls, no tools).

Reads a run's ``trace<N>/`` directory (``trace.jsonl`` plus content-addressed ``blobs/``, written by ``TraceRecorder``) and feeds the recorded tool calls
and results to ``ProgressTracker`` in order. It answers: where would the policy first have intervened, how many calls would it have blocked, and would
the recovery budget have ended the run, and how many turns earlier. It is a **counterfactual on the recorded sequence**: the model's later behaviour
after an intervention is unknown, and the workspace is not available, so file changes are approximated from the recorded calls (a successful write_file,
or a bash command with a write marker, counts as the workspace changing; ``side_effect_bash=False`` turns the bash heuristic off for a stricter reading).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..progress import ProgressTracker, RecoveryPolicy

_WRITE_MARK = re.compile(r"(>>?\s*['\"]?(?:/work|\.?/?[\w.-]+\.(?:csv|txt|json|md|xlsx|docx|py|sh|log))|open\([^)]*['\"][wax]b?['\"]|\.write(?:text|bytes)?\(|\.save\(|to_excel|to_csv|to_json|savefig|\bmkdir\b|\bcp\b|\bmv\b|\brm\b|\btee\b|pip install)")


@dataclass
class Step:
    turn: int
    name: str
    args: Dict[str, Any]
    text: str
    ok: bool


def _blob(d: Path, sha: Optional[str]) -> str:
    if not sha:
        return ""
    p = d / "blobs" / f"{sha}.txt"
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


def load_steps(trace_dir: Path) -> List[Step]:
    """Tool calls in order with their full arguments and full results. Calls the legacy guard refused carry its ``[repeat]`` text."""
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
        steps.append(Step(r.get("turn", len(steps) + 1), r.get("name", "?"), args if isinstance(args, dict) else {"_raw": raw}, text, not text.startswith("ERROR:")))
    return steps


def replay(steps: List[Step], policy: RecoveryPolicy, *, side_effect_bash: bool = True) -> Dict[str, Any]:
    state = {"epoch": 0}
    pol = RecoveryPolicy(**{**policy.__dict__, "progress_probe": lambda: state["epoch"]})
    tr = ProgressTracker(pol)
    last_real: Dict[str, str] = {}
    first_block = ended_at = None
    blocked = 0
    from ..progress import normalize_call
    for i, s in enumerate(steps):
        key = normalize_call(s.name, s.args)
        text = s.text
        if text.startswith("[repeat]") and key in last_real:       # the legacy guard refused this call: it meant "same result as before"
            text = last_real[key]
        if tr.before(s.name, s.args):
            blocked += 1
            first_block = i + 1 if first_block is None else first_block
        else:
            tr.after(s.name, s.args, text, s.ok)
            last_real[key] = text
            writes = s.ok and (s.name in ("write_file", "web_download") or (side_effect_bash and s.name == "bash" and _WRITE_MARK.search(json.dumps(s.args))))
            if writes:
                state["epoch"] += 1
        if tr.exhausted and ended_at is None:
            ended_at = i + 1
            break
    rep = tr.report()
    return {"steps": len(steps), "first_blocked_step": first_block, "blocked_calls": blocked, "ended_at_step": ended_at,
            "steps_saved": (len(steps) - ended_at) if ended_at else 0, "counts": rep["counts"], "blockers": rep["blockers"]}


def replay_dir(trace_dir: Path, policy: Optional[RecoveryPolicy] = None, **kw) -> Dict[str, Any]:
    from .capabilities import capabilities
    pol = policy or RecoveryPolicy(capabilities=tuple(capabilities(web_paging=False, web_downloads=False)))
    return replay(load_steps(trace_dir), pol, **kw)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Replay recorded runs through the progress tracker")
    ap.add_argument("trace_dirs", nargs="+")
    ap.add_argument("--no-bash-heuristic", action="store_true")
    a = ap.parse_args(argv)
    for t in a.trace_dirs:
        r = replay_dir(Path(t), side_effect_bash=not a.no_bash_heuristic)
        print(json.dumps({"trace": t, **{k: r[k] for k in ("steps", "first_blocked_step", "blocked_calls", "ended_at_step", "steps_saved")},
                          "blockers": [b["kind"] + ":" + b["tool"] for b in r["blockers"]]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
