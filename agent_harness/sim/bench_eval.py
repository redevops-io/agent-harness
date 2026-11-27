# SPDX-License-Identifier: AGPL-3.0-or-later
"""The SIM evaluator over the existing benchmark tooling: real workspace tasks, isolated rollouts, the rubric judge.

Each task is rolled with :func:`agent_harness.bench.jobbench.run_task` (a fresh temporary workspace per attempt, no
network unless the run says otherwise) and scored with :func:`agent_harness.bench.evalkit.judge_output`. The harness
under evaluation contributes two things and nothing else: its always-on text, appended to the system prompt, and its
skill files, placed in the workspace at their tree paths.

What a harness cannot reach from here: the task folders and their ``RUBRICS.json`` (read by the judge from
``tasks_base``, never copied into what the proposer sees), the judge configuration, and the pass threshold. The feedback
given back per task is the score, how the rollout stopped and its failure class; with ``feedback_detail="criteria"`` (the default, as NEXT_EXPERIMENT.md
decides) it also names the criteria the judge found unmet, in the judge's words, for the EVOLVE tasks only (the held-out half is never evaluated
by an object that feeds a proposer). The rubric FILE is never shown. ``feedback_detail="score"`` gives no rubric text at all.

A task may be rolled several times (``rollouts``): its score is the mean and the evaluation records each rollout, because one rollout of one task
was measured to move by 0.27 between two identical runs (pilot-g0).

A task the judge could not grade (an infrastructure failure) is not a pass and not a fail: the evaluation raises
``EvaluationIncomplete`` and the round is not scored, instead of a guess entering the memory as a measured outcome.
"""
from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

from ..h2.spec import HarnessSpec
from .loop import Feedback, TaskResult


class EvaluationIncomplete(RuntimeError):
    """At least one task could not be graded; the harness has no score for this evaluation."""


@dataclass(frozen=True)
class BenchConfig:
    tasks_base: Path
    workdir: Path                       # rollouts and judge details are written here, one folder per evaluation
    pass_score: float = 0.5             # a task passes when the judge's score is at least this
    max_turns: int = 200
    max_seconds: float = 3600
    max_context_tokens: int = 65536
    attempts: int = 2
    network: str = "none"
    context: str = "context-runtime"
    web: bool = False                   # the rollout's web tools; off: an evaluation must not depend on what the internet says that day
    rollouts: int = 1                   # rollouts per task; the score is their mean
    workers: int = 1                    # tasks rolled in parallel (the model server's capacity is the limit)
    feedback_detail: str = "criteria"   # "criteria": unmet criteria in the judge's words; "score": none
    max_failed_criteria: int = 6


class BenchEvaluator:
    def __init__(self, client: Any, judge: Any, config: BenchConfig, *, run_task: Optional[Callable[..., Dict[str, Any]]] = None,
                 judge_output: Optional[Callable[..., Any]] = None) -> None:
        if run_task is None or judge_output is None:                      # imported late: the bench stack is heavy and optional
            from ..bench import evalkit, jobbench
            run_task = run_task or jobbench.run_task
            judge_output = judge_output or evalkit.judge_output
        self.client, self.judge, self.cfg = client, judge, config
        self._run_task, self._judge_output = run_task, judge_output
        self.evaluations = 0

    @staticmethod
    def harness_inputs(harness: HarnessSpec) -> Dict[str, Any]:
        """What of a harness reaches a rollout: the always-on text and the skill and script files."""
        files = {p: t for p, t in harness.files.items() if p.startswith(("skills/", "scripts/")) and t}
        return {"extra_system": harness.always_on_text().strip(), "extra_files": files}

    def _unmet(self, details_file: Path) -> str:
        """The criteria the judge found unmet in one graded output, in its own words (at most ``max_failed_criteria``, 220 characters each)."""
        try:
            d = json.loads(Path(details_file).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return ""
        out = []
        for r in d.get("rubrics", []):
            for c in (r.get("result") or {}).get("criteria_results", []) or []:
                if not c.get("passed"):
                    out.append(str(c.get("criterion", ""))[:220])
        return " | ".join(f"unmet: {x}" for x in out[: max(0, int(self.cfg.max_failed_criteria))])

    def _rollout_key(self, task: str, harness_inputs: Dict[str, Any], rollout: int) -> str:
        """What a finished rollout is the answer TO: the harness text and files, the task, which rollout, and every setting that shapes it."""
        c = self.cfg
        blob = json.dumps({"inputs": harness_inputs, "task": task, "rollout": rollout, "max_turns": c.max_turns, "max_seconds": c.max_seconds, "max_context_tokens": c.max_context_tokens,
                           "attempts": c.attempts, "network": c.network, "context": c.context, "web": c.web}, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _finished_rollout(self, out: Path, task: str, harness_inputs: Dict[str, Any], rollout: int) -> Optional[Dict[str, Any]]:
        seal, summary = out / ".rollout.sealed", out / "summary.json"
        try:
            if seal.read_text().strip() == self._rollout_key(task, harness_inputs, rollout):
                return json.loads(summary.read_text())
        except (OSError, ValueError):
            pass
        return None

    def _seal_rollout(self, out: Path, task: str, harness_inputs: Dict[str, Any], rollout: int) -> None:
        if (out / "summary.json").is_file():
            (out / ".rollout.sealed").write_text(self._rollout_key(task, harness_inputs, rollout))

    def _one(self, run_dir: Path, task: str, harness_inputs: Dict[str, Any], rollout: int):
        safe = task.replace("/", "__") + (f"__r{rollout}" if self.cfg.rollouts > 1 else "")
        out = run_dir / safe
        summary = self._finished_rollout(out, task, harness_inputs, rollout)
        if summary is None:                                                # an agent answer already finished for exactly this harness and task is never generated again
            summary = self._run_task(self.cfg.tasks_base, task, out, self.client, max_turns=self.cfg.max_turns, max_seconds=self.cfg.max_seconds,
                                     max_context_tokens=self.cfg.max_context_tokens, attempts=self.cfg.attempts, context=self.cfg.context,
                                     network=self.cfg.network, web=self.cfg.web, **harness_inputs)
            self._seal_rollout(out, task, harness_inputs, rollout)
        verdict = self._judge_output(task, self.cfg.tasks_base, out / "output", run_dir / "judge" / f"{safe}.json", self.judge, finished=True)
        return summary, verdict, run_dir / "judge" / f"{safe}.json"

    def evaluate(self, harness: HarnessSpec, tasks: Sequence[str]) -> Feedback:
        self.evaluations += 1
        run_dir = Path(self.cfg.workdir) / f"eval-{self.evaluations:04d}"
        inputs = self.harness_inputs(harness)
        t0 = time.monotonic()
        jobs = [(t, k) for t in tasks for k in range(1, max(1, int(self.cfg.rollouts)) + 1)]
        with ThreadPoolExecutor(max(1, int(self.cfg.workers))) as pool:
            done = list(pool.map(lambda j: (j, self._one(run_dir, j[0], inputs, j[1])), jobs))
        results, ungraded = [], []
        for task in tasks:
            mine = [(summary, verdict, f) for (t, _k), (summary, verdict, f) in done if t == task]
            bad = [m for m in mine if getattr(m[1], "status", "ungraded") == "ungraded"]
            if bad:
                ungraded.append(f"{task}: {getattr(bad[0][1], 'reason', '')}"[:160])
                continue
            scores = [float(getattr(v, "score", 0.0) or 0.0) for _, v, _ in mine]
            score = sum(scores) / len(scores)
            notes = []
            for n, (summary, verdict, f) in enumerate(mine, 1):
                failure = (summary.get("failure") or {}).get("class", "") if isinstance(summary.get("failure"), dict) else ""
                notes.append(f"rollout {n}: score {scores[n - 1]:.2f}; stopped: {summary.get('stop')}; turns {summary.get('turns')}" + (f"; failure class: {failure}" if failure else "")
                             + ("; no deliverable was produced" if getattr(verdict, "status", "") == "no_output" else ""))
            detail = f"mean score {score:.2f} over {len(mine)} rollout(s) [" + "; ".join(notes) + "]"
            if self.cfg.feedback_detail == "criteria":
                worst = min(mine, key=lambda m: float(getattr(m[1], "score", 0.0) or 0.0))
                unmet = self._unmet(worst[2])
                if unmet:
                    detail += " " + unmet
            results.append(TaskResult(task, score >= self.cfg.pass_score, score, detail))
        if ungraded:
            raise EvaluationIncomplete(f"{len(ungraded)} task(s) could not be graded, so this harness has no score: " + "; ".join(ungraded[:3]))
        return Feedback(harness_id="", results=tuple(results), seconds=round(time.monotonic() - t0, 1))
