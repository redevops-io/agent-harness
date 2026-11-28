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
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..h2.spec import HarnessSpec
from .loop import Feedback, TaskResult


class EvaluationIncomplete(RuntimeError):
    """At least one task could not be graded; the harness has no score for this evaluation.

    ``kind`` says what to do about it: ``judge_infrastructure`` (replies missing after the bounded passes: run the evaluation again, the answered rubrics and the
    finished rollouts are kept) or ``judge_context_exceeded`` (the judge's prompt for a deliverable is larger than the judge's context: the same request can never be answered,
    so running again changes nothing until the judge's context is changed, which is a decision about the experiment, not a retry). ``ungraded`` lists each rollout that
    has no verdict, ``evaluation_dir`` is where everything settled so far is, and ``usage`` is what the evaluation consumed up to the failure."""

    def __init__(self, message: str, *, kind: str = "judge_infrastructure", ungraded: Sequence[Dict[str, Any]] = (), evaluation_dir: Optional[Path] = None,
                 usage: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.kind, self.ungraded, self.evaluation_dir, self.usage = kind, list(ungraded), evaluation_dir, dict(usage or {})


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

    def _eval_dir(self, harness_inputs: Dict[str, Any], tasks: Sequence[str]) -> Path:
        """One directory per (harness, tasks, settings), found again by content: a restarted process evaluating the same harness continues in the directory that
        holds its finished rollouts and judged rubrics, and a different harness never lands in (and overwrites) another's."""
        c = self.cfg
        tag = hashlib.sha256(json.dumps({"inputs": harness_inputs, "tasks": list(tasks), "rollouts": c.rollouts, "max_turns": c.max_turns, "max_seconds": c.max_seconds,
                                         "max_context_tokens": c.max_context_tokens, "attempts": c.attempts, "network": c.network, "context": c.context, "web": c.web},
                                        sort_keys=True, default=str).encode()).hexdigest()[:12]
        root = Path(c.workdir)
        found = sorted(root.glob(f"eval-*-{tag}")) if root.is_dir() else []
        if found:
            return found[0]
        taken = [int(p.name.split("-")[1]) for p in root.glob("eval-*") if p.name.split("-")[1].isdigit()] if root.is_dir() else []
        return root / f"eval-{max(taken, default=0) + 1:04d}-{tag}"

    @staticmethod
    def _usage(summary: Dict[str, Any], details_file: Path) -> Dict[str, Any]:
        """Tokens one rollout and its judging consumed: the actor's from its summary, the judge's from the per-request log when the judge was run metered."""
        u = {"actor_prompt_tokens": int(summary.get("prompt_tokens") or 0), "actor_completion_tokens": int(summary.get("completion_tokens") or 0),
             "judge_prompt_tokens": 0, "judge_completion_tokens": 0, "judge_requests": 0, "judge_failed_requests": 0, "judge_measured": False}
        log = details_file.with_name(details_file.name + ".usage.jsonl")
        try:
            for line in log.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                u["judge_measured"] = True
                if r.get("error"):
                    u["judge_failed_requests"] += 1
                    continue
                u["judge_requests"] += 1
                u["judge_prompt_tokens"] += int(r.get("prompt_tokens") or 0)
                u["judge_completion_tokens"] += int(r.get("completion_tokens") or 0)
        except (OSError, ValueError):
            pass
        return u

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

    def _one_safely(self, run_dir: Path, task: str, harness_inputs: Dict[str, Any], rollout: int):
        """A rollout that raised is a failure of THIS rollout, recorded and reported at the end; the others are finished and settled first, so a restart has less left to do."""
        try:
            return self._one(run_dir, task, harness_inputs, rollout)
        except Exception as e:  # noqa: BLE001
            return {}, _Failed(f"{type(e).__name__}: {e}"[:200]), run_dir / "judge" / "none.json"

    def evaluate(self, harness: HarnessSpec, tasks: Sequence[str]) -> Feedback:
        self.evaluations += 1
        inputs = self.harness_inputs(harness)
        run_dir = self._eval_dir(inputs, tasks)
        t0 = time.monotonic()
        jobs = [(t, k) for t in tasks for k in range(1, max(1, int(self.cfg.rollouts)) + 1)]
        with ThreadPoolExecutor(max(1, int(self.cfg.workers))) as pool:
            done = list(pool.map(lambda j: (j, self._one_safely(run_dir, j[0], inputs, j[1])), jobs))
        usage: Dict[str, Any] = {}
        statuses: List[Dict[str, Any]] = []
        for (t, k), (summary, verdict, f) in done:
            u = self._usage(summary, f) if summary else {}
            for key, val in u.items():
                usage[key] = (usage.get(key, 0) + val) if isinstance(val, int) and not isinstance(val, bool) else (usage.get(key, False) or val)
            statuses.append({"task": t, "rollout": k, "status": getattr(verdict, "status", "ungraded"), "reason": getattr(verdict, "reason", ""),
                             "score": getattr(verdict, "score", None), "context_exceeded": bool(getattr(verdict, "context_exceeded", False)),
                             "prompt_limit": getattr(verdict, "prompt_limit", None), "prompt_tokens_at_least": getattr(verdict, "prompt_tokens_at_least", None), "usage": u})
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
        seconds = round(time.monotonic() - t0, 1)
        bad_rows = [s for s in statuses if s["status"] == "ungraded"]
        self._write_status(run_dir, tasks, statuses, usage, seconds, complete=not ungraded)
        if ungraded:
            kind = "judge_context_exceeded" if bad_rows and all(s["context_exceeded"] for s in bad_rows) else "judge_infrastructure"
            raise EvaluationIncomplete(f"{len(ungraded)} task(s) could not be graded, so this harness has no score: " + "; ".join(ungraded[:3]),
                                       kind=kind, ungraded=bad_rows, evaluation_dir=run_dir, usage=usage)
        return Feedback(harness_id="", results=tuple(results), seconds=seconds, usage=usage)

    @staticmethod
    def _write_status(run_dir: Path, tasks: Sequence[str], statuses: List[Dict[str, Any]], usage: Dict[str, Any], seconds: float, *, complete: bool) -> None:
        """What this evaluation settled and what it consumed, on disk whether it finished or not (``evaluation.json``)."""
        tmp = run_dir / "evaluation.json.tmp"
        try:
            tmp.write_text(json.dumps({"complete": complete, "tasks": list(tasks), "rollouts": statuses, "usage": usage, "seconds": seconds}, indent=1, sort_keys=True), encoding="utf-8")
            os.replace(tmp, run_dir / "evaluation.json")
        except OSError:
            pass


class _Failed:
    """The stand-in verdict for a rollout that raised: ungraded, with the reason."""
    status = "ungraded"
    score = None

    def __init__(self, reason: str) -> None:
        self.reason = reason


def tally_usage(workdir: "str | Path") -> Dict[str, Any]:
    """What a run consumed, counted from what is on disk (so it also works for a run that stopped, and counts each finished rollout once however many times it was served again):
    the actor's tokens from every rollout summary, the judge's from the per-request usage logs (present only when the judge ran metered; ``judge_measured`` says so)."""
    root = Path(workdir)
    t = {"rollouts": 0, "actor_prompt_tokens": 0, "actor_completion_tokens": 0, "judge_requests": 0, "judge_failed_requests": 0, "judge_prompt_tokens": 0,
         "judge_completion_tokens": 0, "judge_measured": False, "evaluations": 0}
    for ev in sorted(root.glob("eval-*")):
        if not ev.is_dir():
            continue
        t["evaluations"] += 1
        for sm in ev.glob("*/summary.json"):
            try:
                d = json.loads(sm.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            t["rollouts"] += 1
            t["actor_prompt_tokens"] += int(d.get("prompt_tokens") or 0)
            t["actor_completion_tokens"] += int(d.get("completion_tokens") or 0)
        for log in (ev / "judge").glob("*.usage.jsonl") if (ev / "judge").is_dir() else []:
            u = BenchEvaluator._usage({}, log.with_name(log.name[: -len(".usage.jsonl")]))
            t["judge_measured"] = t["judge_measured"] or u["judge_measured"]
            for k in ("judge_requests", "judge_failed_requests", "judge_prompt_tokens", "judge_completion_tokens"):
                t[k] += u[k]
    return t
