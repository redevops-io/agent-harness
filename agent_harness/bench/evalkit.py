"""Judging and scoring with the failure modes of earlier runs designed out.

Every rule below exists because the opposite happened in a real run:

* A judge failure (rate limit, 5xx, timeout, unparsable reply) is ``ungraded``. It is never a score of 0.
* Stale judge detail files are deleted before each attempt; a leftover failed file made retries return instantly.
* Infrastructure failures are retried with backoff; a task is ``graded`` only when every rubric got a real reply.
* "No output" is a score of 0 only for a rollout that FINISHED and delivered nothing; a rollout that is missing or
  unfinished cannot be judged at all (``NotReady``), so a missing directory is never read as "no output".
* The scoreboard is append-only; "ungraded" never counts as done and never as 0; paired statistics only use tasks
  graded in both arms and report how many were dropped.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import time
from dataclasses import dataclass, field
from math import comb
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

FAIL_PREFIX = "Failed to get judge response"


class NotReady(Exception):
    """The rollout has not finished (or does not exist): refuse to judge it."""


@dataclass
class JudgeConfig:
    python: str
    script: str
    model: str
    base_url: str
    reasoning_effort: str = "default"      # "default" = send no reasoning_effort (needed for models that reject temperature 0)
    workers: int = 4
    max_retries: int = 3                   # per rubric, inside the upstream judge
    attempts: int = 3                      # whole-output attempts when infrastructure failures remain
    rubric_timeout_s: int = 240            # per rubric inside the upstream judge; large deliverables on a local model need far more
    backoff_s: float = 20.0
    timeout_s: int = 1800
    api_key: str = "dummy"


@dataclass
class JudgeResult:
    status: str                            # graded | ungraded | no_output
    score: Optional[float] = None
    total: Optional[float] = None
    max: Optional[float] = None
    passed: Optional[int] = None
    count: Optional[int] = None
    failures: int = 0
    attempts: int = 0
    reason: str = ""
    budget_exhausted: bool = False        # the spend proxy refused the calls: stop judging, do not retry
    criteria_passed: Optional[int] = None  # individual criteria passed (the group score is all-or-nothing per rubric group)
    criteria_total: Optional[int] = None


def has_deliverables(deliver: Path) -> bool:
    return deliver.is_dir() and any(p.is_file() for p in deliver.rglob("*"))


# Our own spend proxy says "budget_exceeded"/"budget exhausted"; the PROVIDER says it when the account has no balance left, in a
# 429 that looks like a rate limit ("exceeded_current_quota_error ... suspended due to insufficient balance"). Retrying cannot help.
_NO_MONEY = ("budget_exceeded", "budget exhausted", "exceeded_current_quota", "insufficient balance", "insufficient_quota",
             "account is suspended")


def _budget_exhausted(details: Dict[str, Any]) -> bool:
    return any(any(m in str(c.get("reasoning", "")).lower() for m in _NO_MONEY)
               for r in details.get("rubrics", []) for c in (r.get("result", {}).get("criteria_results") or []))


def _failures(details: Dict[str, Any]) -> int:
    return sum(1 for r in details.get("rubrics", []) for c in (r.get("result", {}).get("criteria_results") or [])
               if str(c.get("reasoning", "")).startswith(FAIL_PREFIX))


def judge_output(task_id: str, tasks_base: Path, deliver: Path, details_file: Path, cfg: JudgeConfig, *,
                 finished: bool = True, sleep=time.sleep) -> JudgeResult:
    """Score one deliverables directory with the upstream JobBench judge script."""
    if not finished:
        raise NotReady(f"{task_id}: rollout not finished")
    if not has_deliverables(deliver):
        return JudgeResult("no_output", 0.0, 0.0, None, reason="finished with no deliverables")
    rubrics = Path(tasks_base) / task_id / "RUBRICS.json"
    if not rubrics.is_file():
        return JudgeResult("ungraded", reason=f"no rubrics at {rubrics}")
    details_file.parent.mkdir(parents=True, exist_ok=True)
    last = JudgeResult("ungraded", reason="not attempted")
    for attempt in range(1, cfg.attempts + 1):
        details_file.unlink(missing_ok=True)                      # never reuse a previous (possibly failed) attempt
        cmd = [cfg.python, cfg.script, "--output-dir", str(deliver), "--rubrics-file", str(rubrics),
               "--details-file", str(details_file), "--judge-model", cfg.model, "--api-base", cfg.base_url,
               "--max-workers", str(cfg.workers), "--max-retries", str(cfg.max_retries),
               "--timeout-per-rubric", str(cfg.rubric_timeout_s), "--evaluated-model", "own"]
        env = {**os.environ, "JUDGE_API_KEY": cfg.api_key, "JUDGE_REASONING_EFFORT": cfg.reasoning_effort}
        try:
            subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=cfg.timeout_s)
        except (subprocess.SubprocessError, OSError) as e:
            last = JudgeResult("ungraded", attempts=attempt, reason=f"judge process failed: {type(e).__name__}")
            if attempt < cfg.attempts:
                sleep(cfg.backoff_s * attempt)
            continue
        try:
            d = json.loads(details_file.read_text())
            total, mx = float(d["total_score"]), float(d["max_score"])
        except (OSError, ValueError, KeyError, TypeError):
            last = JudgeResult("ungraded", attempts=attempt, reason="judge wrote no readable details")
            if attempt < cfg.attempts:
                sleep(cfg.backoff_s * attempt)
            continue
        fails = _failures(d)
        if fails and _budget_exhausted(d):
            return JudgeResult("ungraded", failures=fails, attempts=attempt, reason="spend budget exhausted", budget_exhausted=True)
        if fails or mx <= 0:
            last = JudgeResult("ungraded", failures=fails, attempts=attempt,
                               reason=f"{fails} rubric criteria got no judge reply" if fails else "max score is 0")
            if attempt < cfg.attempts:
                sleep(cfg.backoff_s * attempt)
            continue
        rubrics = d.get("rubrics") or []
        cp = sum(int((r.get("result") or {}).get("criteria_passed") or 0) for r in rubrics)
        ct = sum(int((r.get("result") or {}).get("criteria_count") or 0) for r in rubrics)
        return JudgeResult("graded", total / mx, total, mx, d.get("passed_count"), d.get("total_count"), 0, attempt,
                           criteria_passed=cp if ct else None, criteria_total=ct or None)
    return last


class Scoreboard:
    """Append-only JSONL of what happened to each (arm, task). Last event wins; nothing is overwritten."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._state: Dict[Tuple[str, str], Dict[str, Any]] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    e = json.loads(line)
                    self._state.setdefault((e["arm"], e["task"]), {}).update(e["data"])

    def _append(self, arm: str, task: str, **data: Any) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps({"ts": round(time.time(), 1), "arm": arm, "task": task, "data": data}, sort_keys=True) + "\n")
        self._state.setdefault((arm, task), {}).update(data)

    def mark_generated(self, arm: str, task: str, *, deliverables: Sequence[str], stop: str, finished: bool = True,
                       **extra: Any) -> None:
        extra.setdefault("error", None)         # a rollout that now finished supersedes an earlier crash record
        self._append(arm, task, generated=True, finished=finished, deliverables=list(deliverables), stop=stop, **extra)

    def info(self, arm: str, task: str) -> Dict[str, Any]:
        return dict(self._state.get((arm, task), {}))

    def arms(self) -> List[str]:
        return sorted({a for a, _ in self._state})

    def finished(self, arm: str, task: str) -> bool:
        return bool(self._state.get((arm, task), {}).get("finished"))

    def graded(self, arm: str, task: str) -> bool:
        return self._state.get((arm, task), {}).get("status") in ("graded", "no_output")

    def record_judgement(self, arm: str, task: str, r: JudgeResult, judge: str) -> None:
        if not self.finished(arm, task):
            raise NotReady(f"{arm}/{task}: no finished rollout recorded")
        self._append(arm, task, status=r.status, score=r.score, judge=judge, failures=r.failures, reason=r.reason,
                     criteria_passed=r.criteria_passed, criteria_total=r.criteria_total)

    def score(self, arm: str, task: str) -> Optional[float]:
        s = self._state.get((arm, task), {})
        return s.get("score") if s.get("status") in ("graded", "no_output") else None

    def pending(self, arm: str, tasks: Sequence[str]) -> List[str]:
        """Finished rollouts that still need a (successful) judgement. Unfinished ones are not listed."""
        return [t for t in tasks if self.finished(arm, t) and not self.graded(arm, t)]

    def paired(self, a: str, b: str, tasks: Sequence[str]) -> Dict[str, Any]:
        both = [t for t in tasks if self.score(a, t) is not None and self.score(b, t) is not None]
        d = [self.score(a, t) - self.score(b, t) for t in both]
        out: Dict[str, Any] = {"n": len(both), "dropped_ungraded_or_missing": len(tasks) - len(both), "tasks": both}
        if len(d) >= 2:
            m, sd = sum(d) / len(d), math.sqrt(sum((x - sum(d) / len(d)) ** 2 for x in d) / (len(d) - 1))
            w, l = sum(x > 0 for x in d), sum(x < 0 for x in d)
            n = w + l
            p = min(1.0, 2 * sum(comb(n, k) for k in range(max(w, l), n + 1)) / 2 ** n) if n else 1.0
            out.update(mean_delta=m, sd=sd, se=sd / math.sqrt(len(d)), wins=w, ties=len(d) - n, losses=l, sign_test_p=p,
                       mean_a=sum(self.score(a, t) for t in both) / len(both), mean_b=sum(self.score(b, t) for t in both) / len(both))
        return out


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):                       # average ranks for ties
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx)) * math.sqrt(sum((b - my) ** 2 for b in ry))
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / sx if sx else float("nan")
