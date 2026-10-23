"""The harness x precision (x thinking) experiment: generation, judging and the report as separate, gated steps.

Arms (harnesses), each an adapter module in ``bench/harnesses``:
  ours       agent-harness executor + context-runtime context management
  opencode   OpenCode's stock harness (its own rollouts, no refinement)
  harness2   Google's Harness² on the OpenCode substrate: a refined harness per task (k=1, parallel)
Cells: the same model served one way (precision, thinking on/off) on one endpoint. Cells are never mixed. The
Harness² editor/composer and the judge are fixed models, identical in every cell, so only the executor varies.

    python -m agent_harness.bench.experiment generate --config exp.json [--only ARM] [--cell NAME]
    python -m agent_harness.bench.experiment judge    --config exp.json     # refuses a judge without a passed calibration
    python -m agent_harness.bench.experiment report   --config exp.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .cells import Cell, arm_id, task_dir_name
from .evalkit import JudgeConfig, Scoreboard, judge_batch, judge_output
from . import harnesses

DEFAULT_ARMS = ("ours", "opencode", "harness2")


class Experiment:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.root = Path(cfg["root"]).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.tasks: List[str] = [t.strip() for t in Path(cfg["tasks_file"]).expanduser().read_text().splitlines() if t.strip()]
        self.tasks_base = Path(cfg["tasks_base"]).expanduser()
        self.cells = [Cell(n, c["provider"], c["model"], int(c["window"]), bool(c.get("think", False))) for n, c in cfg["cells"].items()]
        self.arms: Sequence[str] = tuple(cfg.get("arms", DEFAULT_ARMS))
        self.proxy = cfg.get("proxy_base", "http://127.0.0.1:8765")
        self.board = Scoreboard(self.root / "scoreboard.jsonl")

    def generate(self, only: Optional[str] = None, cell: Optional[str] = None, *, skip_audit: bool = False) -> None:
        if not skip_audit:                  # refuse to spend hours on a run whose settings differ from the declared spec
            from . import audit
            findings = audit.audit_config(self)
            for f in findings:
                print(f, flush=True)
            if not audit.passed(findings):
                raise SystemExit("config audit failed: fix the configuration (or pass --skip-audit knowingly); nothing was started")
        for c in self.cells:
            if cell and c.name != cell:
                continue
            for arm in self.arms:
                if (only and arm != only) or arm in harnesses.PASSIVE:
                    continue
                harnesses.load(arm).generate(self, c, int(self.cfg.get(arm, {}).get("workers", 1)))

    def judge_pending(self, jcfg: JudgeConfig, judge_name: str, details_root: Optional[Path] = None, *, sleep=time.sleep,
                      limit: Optional[int] = None, complete_only: bool = False, parallel: int = 1) -> int:
        """Judge finished rollouts TASK BY TASK (all cells and arms of a task before the next task), so that if the money or
        the time runs out every arm has the same tasks judged. ``complete_only`` skips a task until EVERY arm and cell has a
        finished rollout for it (a fair, balanced spend while generation is still running). ``parallel`` judges that many outputs
        of a task at once. Stops when the spend proxy reports the budget is gone (nothing is recorded for refused outputs)."""
        from concurrent.futures import ThreadPoolExecutor
        n = 0
        details_root = details_root or self.root / "judge_details"
        for t in self.tasks:
            arms = [arm_id(arm, c.name) for c in self.cells for arm in self.arms]
            if complete_only and not all(self.board.finished(a, t) for a in arms):
                continue
            todo = [a for a in arms if t in self.board.pending(a, [t])]
            if limit is not None:
                todo = todo[:max(0, limit - n)]
            if not todo:
                continue

            def one(a: str):
                info = self.board.info(a, t)
                return a, judge_output(t, self.tasks_base, Path(info.get("dir") or "/nonexistent"),
                                       details_root / a / f"{task_dir_name(t)}.json", jcfg, finished=True, sleep=sleep)
            with ThreadPoolExecutor(max(1, parallel)) as pool:
                results = list(pool.map(one, todo))
            exhausted = False
            for a, r in results:
                if r.budget_exhausted:
                    exhausted = True
                    continue
                self.board.record_judgement(a, t, r, judge_name)
                n += 1
            if exhausted:
                print("spend budget exhausted: stopping the judge step (nothing recorded for refused outputs)", flush=True)
                return n
            if limit is not None and n >= limit:
                return n
        return n

    def judge_pending_batched(self, jcfg: JudgeConfig, judge_name: str, details_root: Optional[Path] = None, *, sleep=time.sleep,
                              complete_only: bool = False, parallel: int = 3, limit_tasks: Optional[int] = None) -> int:
        """Four-answers-per-call judging: for each task, each MODEL (cell) is judged in one shared pass per rubric group over the answers
        of all its harnesses (blinded, shuffled by task and cell). ``complete_only`` waits until every harness has finished the task in that
        cell. Cells of a task run in parallel. Stops everything when the provider reports the money is gone."""
        from concurrent.futures import ThreadPoolExecutor
        details_root = details_root or self.root / "judge_details_batched"
        n, stopped = 0, False
        for ti, t in enumerate(self.tasks):
            if limit_tasks is not None and ti >= limit_tasks:
                break
            jobs = []
            for c in self.cells:
                ids = {arm: arm_id(arm, c.name) for arm in self.arms}
                if complete_only and not all(self.board.finished(a, t) for a in ids.values()):
                    continue
                cand = {arm: Path(self.board.info(a, t).get("dir") or "/nonexistent") for arm, a in ids.items() if t in self.board.pending(a, [t])}
                if cand:
                    jobs.append((c, ids, cand))

            def one(job):
                c, ids, cand = job
                return job, judge_batch(t, self.tasks_base, cand, details_root / c.name / task_dir_name(t), jcfg, seed=f"{t}|{c.name}", sleep=sleep)
            with ThreadPoolExecutor(max(1, min(parallel, len(jobs) or 1))) as pool:
                outcomes = list(pool.map(one, jobs))
            for (c, ids, cand), res in outcomes:
                for arm, r in res.items():
                    if r.budget_exhausted:
                        stopped = True
                        continue
                    self.board.record_judgement(ids[arm], t, r, judge_name, protocol="batched")
                    n += 1
            if stopped:
                print("judge money exhausted: stopping the judge step (nothing recorded for refused outputs)", flush=True)
                return n
        return n

    def cell_pairs(self) -> List[Sequence[str]]:
        names = [c.name for c in self.cells]
        return self.cfg.get("cell_pairs") or ([names[:2]] if len(names) >= 2 else [])

    def report(self) -> Dict[str, Any]:
        cells = [c.name for c in self.cells]
        arms = [arm_id(a, c) for c in cells for a in self.arms]
        common = [t for t in self.tasks if all(self.board.score(a, t) is not None for a in arms)]
        out: Dict[str, Any] = {"tasks_total": len(self.tasks), "tasks_graded_in_all_arms": len(common), "table": {}, "paired": {}, "diagnostics": {}}
        win = {c.name: c.window for c in self.cells}
        for c in cells:
            for a in self.arms:
                aid = arm_id(a, c)
                graded = [t for t in self.tasks if self.board.score(aid, t) is not None]
                sc = [self.board.score(aid, t) for t in common]
                out["table"][aid] = {"mean_on_common_tasks": sum(sc) / len(sc) if sc else None, "n_common": len(sc),
                                     "n_graded": len(graded), "n_ungraded_or_missing": len(self.tasks) - len(graded)}
                infos = [self.board.info(aid, t) for t in self.tasks if self.board.finished(aid, t)]
                peaks = [i["peak_prompt"] for i in infos if i.get("peak_prompt")]
                out["diagnostics"][aid] = {"finished": len(infos), "no_output": sum(1 for i in infos if not i.get("deliverables")),
                                           "max_peak_prompt": max(peaks) if peaks else None, "window": win[c],
                                           "runs_at_90pct_of_window": sum(1 for p in peaks if p >= 0.9 * win[c]),
                                           "runs_over_window": sum(1 for p in peaks if p > win[c])}
            for i, a in enumerate(self.arms):
                for b in self.arms[i + 1:]:
                    out["paired"][f"{a} - {b} @ {c}"] = self.board.paired(arm_id(a, c), arm_id(b, c), self.tasks)
        for x, y in self.cell_pairs():
            for a in self.arms:
                out["paired"][f"{a}: {x} - {y}"] = self.board.paired(arm_id(a, x), arm_id(a, y), self.tasks)
        return out

    def report_markdown(self) -> str:
        r = self.report()
        cells = [c.name for c in self.cells]
        f = lambda v: "-" if v is None else f"{100 * v:.1f}"
        lines = [f"Tasks: {r['tasks_total']} total, {r['tasks_graded_in_all_arms']} graded in every arm (the table uses those only).", "",
                 "| harness | " + " | ".join(cells) + " |", "|---|" + "---|" * len(cells)]
        for a in self.arms:
            lines.append(f"| {a} | " + " | ".join(f(r["table"][arm_id(a, c)]["mean_on_common_tasks"]) for c in cells) + " |")
        lines += ["", "Paired differences (points; mean, se, wins/ties/losses, sign-test p):"]
        for k, p in r["paired"].items():
            lines.append(f"- {k}: n={p['n']}" + (f", {100 * p['mean_delta']:+.1f} (se {100 * p['se']:.1f}), {p['wins']}/{p['ties']}/{p['losses']}, p={p['sign_test_p']:.3f}"
                                                   if "mean_delta" in p else " (too few)") + f", dropped {p['dropped_ungraded_or_missing']}")
        lines += ["", "Context and completion (does the window hold?):"]
        for aid, d in r["diagnostics"].items():
            lines.append(f"- {aid}: finished {d['finished']}, no output {d['no_output']}, peak prompt {d['max_peak_prompt']} of {d['window']}, "
                         f"runs >=90% of window {d['runs_at_90pct_of_window']}, over window {d['runs_over_window']}")
        return "\n".join(lines)


def calibration_ok(report: Path, judge_model: str, reasoning_effort: str, protocol: str = "per_output") -> bool:
    """The judge may be used only if a report for THIS judge setting AND protocol records a passed calibration (or, for the batched
    protocol, a passed validation against the per-output judge)."""
    try:
        d = json.loads(report.read_text())
    except (OSError, ValueError):
        return False
    return any(c["name"] == "judge calibration" and c["status"] == "PASS" and d.get("judge_model") == judge_model
               and d.get("judge_reasoning_effort") == reasoning_effort and d.get("protocol", "per_output") == protocol
               for c in d.get("checks", []))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=["generate", "judge", "report"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--only")
    ap.add_argument("--cell")
    ap.add_argument("--allow-uncalibrated-judge", action="store_true")
    ap.add_argument("--limit", type=int, help="judge at most this many outputs in this call")
    ap.add_argument("--skip-audit", action="store_true", help="generate even if the config audit fails (the run is then not comparable)")
    ap.add_argument("--protocol", choices=["per_output", "batched"], default="per_output",
                    help="per_output: one answer per judge call; batched: the answers of all harnesses of one model together")
    ap.add_argument("--complete-only", action="store_true", help="judge a task only once every arm and cell has finished it")
    ap.add_argument("--parallel", type=int, default=1, help="outputs of one task judged at the same time")
    a = ap.parse_args(argv)
    ex = Experiment(json.loads(Path(a.config).expanduser().read_text()))
    if a.step == "generate":
        ex.generate(a.only, a.cell, skip_audit=a.skip_audit)
    elif a.step == "judge":
        j = ex.cfg["judge"]
        if not a.allow_uncalibrated_judge and not calibration_ok(Path(ex.cfg["selfcheck_report"]).expanduser(), j["model"], j.get("reasoning_effort", "default"), a.protocol):
            print("REFUSING to judge: no passed calibration for this judge setting in the selfcheck report. "
                  "Run selfcheck with --report, or pass --allow-uncalibrated-judge (results then do not count).")
            return 2
        jc = JudgeConfig(j["python"], j["script"], j["model"], j["base_url"], j.get("reasoning_effort", "default"),
                         workers=j.get("workers", 1), rubric_timeout_s=j.get("rubric_timeout", 3600), timeout_s=j.get("timeout", 10800))
        if a.protocol == "batched":
            print(f"judged {ex.judge_pending_batched(jc, j['model'], complete_only=a.complete_only, parallel=a.parallel)} outputs (batched)")
        else:
            print(f"judged {ex.judge_pending(jc, j['model'], limit=a.limit, complete_only=a.complete_only, parallel=a.parallel)} outputs")
    else:
        print(ex.report_markdown())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
