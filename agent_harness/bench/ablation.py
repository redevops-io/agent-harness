"""Summary of a repeated-rollout ablation: arms ``family`` and ``family-r2``, ``family-r3`` ... are repeats of one setting.

For every (family, cell) it reports the original weighted group score AND the share of individual criteria passed, with mean, spread
and range over the repeats, plus indicators read from each run's trace (messages rewritten by compaction, tool results cut at
admission, actions refused because they carried a placeholder). Repeats of ONE task are not a benchmark; the table is for seeing
whether an effect is larger than the run-to-run spread."""
from __future__ import annotations

import argparse
import json
import re
import statistics as S
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPEAT = re.compile(r"-r\d+$")


def family(arm: str) -> str:
    return _REPEAT.sub("", arm)


def trace_stats(run_dir: Path) -> Dict[str, Any]:
    out = {"traced": False, "max_rewritten": 0, "cut_results": 0, "placeholder_blocked": 0, "compact_events": 0}
    for t in sorted(Path(run_dir).glob("trace*/trace.jsonl")):
        out["traced"] = True
        for line in t.read_text().splitlines():
            e = json.loads(line)
            if e["kind"] == "projection":
                out["max_rewritten"] = max(out["max_rewritten"], e.get("rewritten", 0))
            elif e["kind"] == "tool_full" and e.get("cut"):
                out["cut_results"] += 1
    for t in sorted(Path(run_dir).glob("trajectory*.jsonl")):
        for line in t.read_text().splitlines():
            k = json.loads(line)["kind"]
            out["placeholder_blocked"] += k == "placeholder_blocked"
            out["compact_events"] += k == "compact"
    return out


def _stats(xs: List[float]) -> str:
    if not xs:
        return "-"
    sd = S.pstdev(xs) if len(xs) > 1 else 0.0
    return f"{S.mean(xs):.2f} ±{sd:.2f} [{min(xs):.2f}-{max(xs):.2f}]"


def summarize(states: Dict[Any, Dict[str, Any]], task: str, cells: List[str]) -> str:
    """``states``: {(arm_id, task): scoreboard record}; arm ids look like ``family-r2@cell``."""
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for (aid, t), d in states.items():
        if t != task or "@" not in aid:
            continue
        arm, cell = aid.split("@", 1)
        if cell in cells:
            groups.setdefault((family(arm), cell), []).append(d)
    lines = [f"Task {task}. Mean ±sd [min-max] over repeats; 'criteria' = share of individual criteria passed, 'group' = the all-or-nothing score.",
             "", "| setting | cell | runs | judged | group score | criteria passed | stopped at limit | compact events (mean) | rewritten msgs (max) | cut results (mean) | blocked placeholders |",
             "|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|"]
    for (fam, cell), rows in sorted(groups.items()):
        fin = [r for r in rows if r.get("finished")]
        graded = [r for r in fin if r.get("status") == "graded"]
        crit = [r["criteria_passed"] / r["criteria_total"] for r in graded if r.get("criteria_total")]
        limit = sum(1 for r in fin if r.get("stop") in ("max_seconds", "max_turns"))
        ts = [trace_stats(Path(r["dir"]).parent) for r in fin if r.get("dir")]
        lines.append(f"| {fam} | {cell} | {len(fin)} | {len(graded)} | {_stats([r['score'] for r in graded])} | {_stats(crit)} | {limit} | "
                     f"{S.mean([t['compact_events'] for t in ts]) if ts else 0:.1f} | {max([t['max_rewritten'] for t in ts], default=0)} | "
                     f"{S.mean([t['cut_results'] for t in ts]) if ts else 0:.1f} | {sum(t['placeholder_blocked'] for t in ts)} |")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    from .experiment import Experiment
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--task", required=True)
    a = ap.parse_args(argv)
    ex = Experiment(json.loads(Path(a.config).expanduser().read_text()))
    print(summarize(ex.board._state, a.task, [c.name for c in ex.cells]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
