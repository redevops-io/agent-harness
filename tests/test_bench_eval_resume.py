"""A finished agent answer is never generated again, and a judging failure keeps what was already settled (see docs/experiments/G1_PLAN.md, amendment 1)."""
import json
from pathlib import Path

import pytest

from agent_harness.sim.bench_eval import BenchConfig, BenchEvaluator, EvaluationIncomplete
from agent_harness.sim.experiment_jobbench import seed_harness


class Verdict:
    def __init__(self, status="graded", score=0.8, reason=""):
        self.status, self.score, self.reason = status, score, reason


def make(tmp_path, judge_results):
    calls = {"run": [], "judge": []}

    def run_task(base, task, out, client, **kw):
        calls["run"].append((task, kw["extra_system"]))
        Path(out, "output").mkdir(parents=True, exist_ok=True)
        Path(out, "output", "a.md").write_text("x")
        s = {"task": task, "stop": "done", "turns": 3, "deliverables": ["a.md"]}
        Path(out, "summary.json").write_text(json.dumps(s))
        return s

    def judge_output(task, base, deliver, details, judge, finished=True):
        calls["judge"].append(task)
        return judge_results.pop(0) if len(judge_results) > 1 else judge_results[0]

    cfg = BenchConfig(tasks_base=tmp_path / "base", workdir=tmp_path / "w", rollouts=1, workers=1)
    return BenchEvaluator(None, None, cfg, run_task=run_task, judge_output=judge_output), calls


def test_a_second_evaluation_of_the_same_harness_does_not_generate_the_finished_answer_again(tmp_path):
    ev, calls = make(tmp_path, [Verdict("ungraded", None, "1 rubric criteria got no judge reply"), Verdict()])
    with pytest.raises(EvaluationIncomplete):
        ev.evaluate(seed_harness(), ["x/task1"])
    assert calls["run"] == [("x/task1", "")]
    ev.evaluations = 0                                             # the same process, or a new one over the same workdir, evaluating the same harness again
    fb = ev.evaluate(seed_harness(), ["x/task1"])
    assert len(calls["run"]) == 1 and calls["judge"] == ["x/task1", "x/task1"] and fb.results[0].score == 0.8


def test_another_harness_is_a_different_question_and_gets_its_own_answer(tmp_path):
    ev, calls = make(tmp_path, [Verdict()])
    ev.evaluate(seed_harness(), ["x/task1"])
    h2 = seed_harness()
    h2.set(next(iter(h2.surface.always_on)), "be brief")
    ev.evaluations = 0
    ev.evaluate(h2, ["x/task1"])
    assert [c[1] for c in calls["run"]] == ["", "be brief"]


def test_a_rollout_that_never_finished_is_not_reused(tmp_path):
    ev, calls = make(tmp_path, [Verdict()])
    ev.evaluate(seed_harness(), ["x/task1"])
    (next((tmp_path / "w").glob("eval-0001-*")) / "x__task1" / ".rollout.sealed").unlink()
    ev.evaluations = 0
    ev.evaluate(seed_harness(), ["x/task1"])
    assert len(calls["run"]) == 2
