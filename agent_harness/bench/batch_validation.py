"""Validate the four-answers-per-call judge against the per-output judge on outputs that were ALREADY judged one at a time.

The batched protocol is a different measurement procedure, so it is not trusted until it reproduces the validated one. For one model
(cell) and the tasks where every harness's old output has a per-output judgement, the same outputs are judged again in shared passes and
the two are compared. Thresholds are fixed here, before any result is seen:

* rank agreement of the per-candidate scores (Spearman) >= 0.80,
* agreement of the individual criterion verdicts >= 0.80,
* at most 1 candidate left ungraded.

Passing writes a report in the selfcheck format with ``protocol: batched`` so the experiment's judge step accepts the protocol.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .cells import arm_id, task_dir_name
from .evalkit import JudgeConfig, Scoreboard, classify_details, judge_batch, spearman

THRESHOLDS = {"min_spearman": 0.80, "min_criterion_agreement": 0.80, "max_ungraded": 1}


def verdicts(d: Dict[str, Any]) -> Dict[Tuple[int, str], bool]:
    return {(int(r.get("index", i)), str(c.get("criterion", c.get("index")))): bool(c.get("passed"))
            for i, r in enumerate(d.get("rubrics", [])) for c in (r.get("result", {}).get("criteria_results") or [])}


def compare(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, int]:
    a, b = verdicts(old), verdicts(new)
    common = [k for k in a if k in b]
    groups = {}
    for d, tag in ((old, "o"), (new, "n")):
        groups[tag] = {int(r.get("index", i)): bool(r.get("result", {}).get("passed")) for i, r in enumerate(d.get("rubrics", []))}
    gc = [i for i in groups["o"] if i in groups["n"]]
    return {"criteria": len(common), "criteria_agree": sum(1 for k in common if a[k] == b[k]),
            "groups": len(gc), "groups_agree": sum(1 for i in gc if groups["o"][i] == groups["n"][i])}


def _load(p: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def validate(old_root: Path, arms: Sequence[str], cell: str, tasks: Sequence[str], tasks_base: Path, jcfg: JudgeConfig, details_dir: Path,
             judge_name: str, *, thresholds: Dict[str, float] = THRESHOLDS, max_tasks: Optional[int] = None,
             judge: Callable[..., Dict[str, Any]] = judge_batch, sleep=time.sleep) -> Dict[str, Any]:
    board = Scoreboard(Path(old_root) / "scoreboard.jsonl")
    rows: List[Dict[str, Any]] = []
    agree = {"criteria": 0, "criteria_agree": 0, "groups": 0, "groups_agree": 0}
    ungraded, used = 0, 0
    for t in tasks:
        old = {}
        for arm in arms:
            a = arm_id(arm, cell)
            d = _load(Path(old_root) / "judge_details" / a / f"{task_dir_name(t)}.json")
            info = board.info(a, t)
            if d and board.graded(a, t) and info.get("dir") and classify_details(d, 1).status == "graded":
                old[arm] = (d, Path(info["dir"]))
        if len(old) < 2:
            continue                                              # a shared pass over one candidate says nothing about the protocol
        if max_tasks is not None and used >= max_tasks:
            break
        used += 1
        res = judge(t, tasks_base, {arm: p for arm, (_, p) in old.items()}, details_dir / task_dir_name(t), jcfg, seed=f"validate|{t}|{cell}", sleep=sleep)
        for arm, (od, _) in old.items():
            r = res.get(arm)
            nd = _load(details_dir / task_dir_name(t) / f"{arm}.json")
            if r is None or r.status != "graded" or nd is None:
                ungraded += 1
                continue
            c = compare(od, nd)
            for k in agree:
                agree[k] += c[k]
            rows.append({"task": t, "arm": arm, "old_score": od["total_score"] / od["max_score"], "new_score": r.score, **c})
    sp = spearman([r["old_score"] for r in rows], [r["new_score"] for r in rows]) if len(rows) >= 5 else float("nan")
    ca = agree["criteria_agree"] / agree["criteria"] if agree["criteria"] else float("nan")
    ga = agree["groups_agree"] / agree["groups"] if agree["groups"] else float("nan")
    ok = (len(rows) >= 5 and sp >= thresholds["min_spearman"] and ca >= thresholds["min_criterion_agreement"]
          and ungraded <= thresholds["max_ungraded"])
    detail = (f"batched vs per-output on {len(rows)} outputs of {used} tasks ({cell}): score Spearman {sp:.2f} (need >= {thresholds['min_spearman']}), "
              f"criterion agreement {ca:.2f} (need >= {thresholds['min_criterion_agreement']}), group agreement {ga:.2f}, ungraded {ungraded} "
              f"(max {thresholds['max_ungraded']})")
    return {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "judge_model": jcfg.model, "judge_reasoning_effort": jcfg.reasoning_effort,
            "protocol": "batched", "checks": [{"name": "judge calibration", "status": "PASS" if ok else "FAIL", "detail": detail}],
            "validation": {"thresholds": thresholds, "rows": rows, "agreement": agree, "ungraded": ungraded}}


def main(argv: Optional[Sequence[str]] = None) -> int:
    from .experiment import Experiment
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--old-root", required=True, help="results folder with per-output judgements (scoreboard.jsonl + judge_details/)")
    ap.add_argument("--cell", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--max-tasks", type=int)
    a = ap.parse_args(argv)
    ex = Experiment(json.loads(Path(a.config).expanduser().read_text()))
    j = ex.cfg["judge"]
    jc = JudgeConfig(j["python"], j["script"], j["model"], j["base_url"], j.get("reasoning_effort", "default"), workers=j.get("workers", 1),
                     rubric_timeout_s=j.get("rubric_timeout", 3600), timeout_s=j.get("timeout", 10800))
    rep = validate(Path(a.old_root).expanduser(), ex.arms, a.cell, ex.tasks, ex.tasks_base, jc, ex.root / "batch_validation" / a.cell, j["model"],
                   max_tasks=a.max_tasks)
    Path(a.report).expanduser().write_text(json.dumps(rep, indent=1))
    for c in rep["checks"]:
        print(f"[{c['status']}] {c['name']}: {c['detail']}")
    return 0 if rep["checks"][0]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
