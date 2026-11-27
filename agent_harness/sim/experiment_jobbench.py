# SPDX-License-Identifier: AGPL-3.0-or-later
"""G1 of NEXT_EXPERIMENT.md: optimize a harness on real JobBench work with a local model, and measure whether the memory it produces is big enough to need selecting.

    python -m agent_harness.sim.experiment_jobbench --out DIR --base-url URL --model NAME --tasks FILE --heldout FILE
        [--rounds 4] [--candidates 2] [--rollouts 2] [--workers 3] --judge-python P --judge-script S

One optimization run with the COMPLETE memory (the condition the paper calls full), on the evolve tasks listed in ``--tasks``; the held-out list is read only to refuse any overlap and is
never evaluated here. Actor, proposer, selector and judge are the local model. The judge is the upstream JobBench judge subprocess: it receives a rubric file and a deliverables directory,
nothing of the optimizer's memory, predictions or rationale. At the end the gate is computed from the record that accumulated, with thresholds fixed in NEXT_EXPERIMENT.md: at least 6
distinct problems, and a record larger than the budget of the selected arm (one half of its size). Nothing is added, copied or split to pass it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

from ..bench import evalkit
from ..h2.spec import OPENCODE, HarnessSpec, stock_spec
from ..llm import ChatClient
from .bench_eval import BenchConfig, BenchEvaluator
from .experiment_selected import Meter, MeteredClient
from .llm import LLMProposer, LLMSelector
from .loop import HarnessOptimizer
from .memory import MemoryCondition, Provenance, SelfImprovementMemory

MIN_PROBLEMS = 6                      # fixed in NEXT_EXPERIMENT.md before anything ran
BUDGET_FRACTION = 0.5                 # the selected arm's budget is this fraction of the record's size


def _ids(path: Path) -> List[str]:
    return [l.strip() for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]


def task_texts(tasks_base: Path, tasks: Sequence[str], limit: int = 1500) -> Dict[str, str]:
    """What the proposer is shown of each evolve task: its card, never its rubric."""
    out = {}
    for t in tasks:
        card = Path(tasks_base) / t / "task_card.md"
        out[t] = (card.read_text(encoding="utf-8", errors="replace")[:limit] if card.is_file() else t)
    return out


def seed_harness() -> HarnessSpec:
    return stock_spec(OPENCODE)             # nothing added to the system prompt: the starting point is what the benchmark's own runner gives the model


def gate(memory: SelfImprovementMemory) -> Dict[str, Any]:
    from .memory import MemoryCondition as MC
    text = memory.render(MC.FULL)
    tokens = (len(text) + 3) // 4
    problems = len(memory.problems)
    return {"problems": problems, "attempts": sum(len(p.attempts) for p in memory.problems), "record_tokens": tokens, "selected_budget_tokens": int(tokens * BUDGET_FRACTION),
            "min_problems": MIN_PROBLEMS, "passed": problems >= MIN_PROBLEMS and tokens > 0,
            "note": "the comparison of complete against selected memory runs only if passed; no record is added, copied or split to pass it"}


def run(out: Path, *, base_url: str, model: str, tasks_file: Path, heldout_file: Path, tasks_base: Path, rounds: int, candidates: int, rollouts: int, workers: int,
        judge_python: str, judge_script: str, pass_score: float = 0.5) -> Dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    tasks, heldout = _ids(tasks_file), _ids(heldout_file)
    if set(tasks) & set(heldout):
        raise SystemExit(f"evolve and held-out tasks overlap: {sorted(set(tasks) & set(heldout))[:3]}")
    extra = {"chat_template_kwargs": {"enable_thinking": False}}
    actor = ChatClient(base_url, model, temperature=0.55, max_tokens=16384, timeout=900, retries=2, extra_body=extra)
    engineer_meter = Meter()
    engineer = MeteredClient(ChatClient(base_url, model, temperature=0.3, max_tokens=4000, timeout=600, retries=2, extra_body=extra), engineer_meter)
    judge = evalkit.JudgeConfig(judge_python, judge_script, model, base_url, "default", workers=2, max_retries=2, attempts=3, recovery_workers=1, recovery_thinking_off=False, backoff_s=30.0, rubric_timeout_s=900, timeout_s=3 * 3600)
    cfg = BenchConfig(tasks_base=tasks_base, workdir=out / "rollouts", pass_score=pass_score, rollouts=rollouts, workers=workers, attempts=2, context="context-runtime", web=False)
    evaluator = BenchEvaluator(actor, judge, cfg)
    proposer = LLMProposer(engineer, editable=("systemprompt.md",), task_text=task_texts(tasks_base, tasks), feedback="v2", max_failures_shown=len(tasks))
    selector = LLMSelector(engineer)
    prov = Provenance(domain="jobbench", benchmark="jobbench-seed42-train", policy=model, optimizer=model, evaluator="jobbench-judge:" + model, run="g1")
    memory = SelfImprovementMemory(run="g1")
    opt = HarnessOptimizer(evaluator=evaluator, proposer=proposer, selector=selector, workdir=out / "g1", harness=seed_harness(), memory=memory, condition=MemoryCondition.FULL,
                           evolve_tasks=tasks, heldout_tasks=heldout, provenance=prov, candidates_per_round=candidates, record_rejected=True)
    t0 = time.monotonic()
    recs = opt.run(rounds)
    (out / "g1" / "proposer_replies.json").write_text(json.dumps(proposer.raw, indent=1), encoding="utf-8")
    (out / "g1" / "selector_replies.json").write_text(json.dumps(selector.raw, indent=1), encoding="utf-8")
    result = {"schema": "agent-harness.sim-g1/1", "model": model, "tasks": tasks, "rounds": rounds, "candidates_per_round": candidates, "rollouts_per_task": rollouts, "workers": workers,
              "evolve_scores": [r.current["score"] for r in recs] + [opt._current.score if opt._current else None], "promotions": sum(1 for r in recs if r.promoted),
              "evaluations": evaluator.evaluations, "engineer_usage": engineer_meter.total(), "seconds": round(time.monotonic() - t0, 1), "gate": gate(opt.memory),
              "configuration": {"actor_temperature": 0.55, "actor_max_tokens": 16384, "context_manager": "context-runtime", "web": False, "feedback": "v2 + unmet criteria", "pass_score": pass_score}}
    (out / "results.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--heldout", type=Path, required=True)
    ap.add_argument("--tasks-base", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--candidates", type=int, default=2)
    ap.add_argument("--rollouts", type=int, default=2)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--judge-python", required=True)
    ap.add_argument("--judge-script", required=True)
    a = ap.parse_args(argv)
    r = run(a.out, base_url=a.base_url, model=a.model, tasks_file=a.tasks, heldout_file=a.heldout, tasks_base=a.tasks_base, rounds=a.rounds, candidates=a.candidates,
            rollouts=a.rollouts, workers=a.workers, judge_python=a.judge_python, judge_script=a.judge_script)
    print(json.dumps({k: r[k] for k in ("evolve_scores", "promotions", "evaluations", "seconds", "gate")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
