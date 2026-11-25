# SPDX-License-Identifier: AGPL-3.0-or-later
"""The SIM evaluator over the existing benchmark tooling: real workspace tasks, isolated rollouts, the rubric judge.

Each task is rolled with :func:`agent_harness.bench.jobbench.run_task` (a fresh temporary workspace per attempt, no
network unless the run says otherwise) and scored with :func:`agent_harness.bench.evalkit.judge_output`. The harness
under evaluation contributes two things and nothing else: its always-on text, appended to the system prompt, and its
skill files, placed in the workspace at their tree paths.

What a harness cannot reach from here: the task folders and their ``RUBRICS.json`` (read by the judge from
``tasks_base``, never copied into what the proposer sees), the judge configuration, and the pass threshold. The feedback
given back per task is the score, how the rollout stopped and its failure class: no rubric text.

A task the judge could not grade (an infrastructure failure) is not a pass and not a fail: the evaluation raises
``EvaluationIncomplete`` and the round is not scored, instead of a guess entering the memory as a measured outcome.
"""
from __future__ import annotations

import time
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

    def evaluate(self, harness: HarnessSpec, tasks: Sequence[str]) -> Feedback:
        self.evaluations += 1
        run_dir = Path(self.cfg.workdir) / f"eval-{self.evaluations:04d}"
        inputs = self.harness_inputs(harness)
        t0, results, ungraded = time.monotonic(), [], []
        for task in tasks:
            safe = task.replace("/", "__")
            out = run_dir / safe
            summary = self._run_task(self.cfg.tasks_base, task, out, self.client, max_turns=self.cfg.max_turns, max_seconds=self.cfg.max_seconds,
                                     max_context_tokens=self.cfg.max_context_tokens, attempts=self.cfg.attempts, context=self.cfg.context,
                                     network=self.cfg.network, **inputs)
            verdict = self._judge_output(task, self.cfg.tasks_base, out / "output", run_dir / "judge" / f"{safe}.json", self.judge, finished=True)
            status = getattr(verdict, "status", "ungraded")
            if status == "ungraded":
                ungraded.append(f"{task}: {getattr(verdict, 'reason', '')}"[:160])
                continue
            score = float(getattr(verdict, "score", 0.0) or 0.0)
            failure = (summary.get("failure") or {}).get("class", "") if isinstance(summary.get("failure"), dict) else ""
            detail = f"score {score:.2f}; rollout stopped: {summary.get('stop')}; turns {summary.get('turns')}" + (f"; failure class: {failure}" if failure else "") \
                     + ("; no deliverable was produced" if status == "no_output" else "")
            results.append(TaskResult(task, score >= self.cfg.pass_score, score, detail))
        if ungraded:
            raise EvaluationIncomplete(f"{len(ungraded)} task(s) could not be graded, so this harness has no score: " + "; ".join(ungraded[:3]))
        return Feedback(harness_id="", results=tuple(results), seconds=round(time.monotonic() - t0, 1))
