# SPDX-License-Identifier: AGPL-3.0-or-later
"""The SIM evaluator over the benchmark tooling, with the rollout and the judge replaced by recorders."""
from types import SimpleNamespace

import pytest

from agent_harness.h2.spec import OPENCODE, stock_spec
from agent_harness.sim.bench_eval import BenchConfig, BenchEvaluator, EvaluationIncomplete


def harness():
    s = stock_spec(OPENCODE)
    s.set("systemprompt.md", "Check every deliverable before finishing.")
    s.set("skills/verify/SKILL.md", "---\nname: verify\ndescription: verify outputs\n---\nRun the checks.")
    return s


def make(tmp_path, verdicts):
    rolled, judged = [], []

    def run_task(tasks_base, task, out, client, **kw):
        rolled.append((task, kw))
        return {"stop": "finish", "turns": 7, "failure": {"class": "none"}, "deliverables": ["a.docx"]}

    def judge_output(task, tasks_base, deliver, details, cfg, finished=True):
        judged.append((task, str(deliver), str(details)))
        return verdicts[task]

    ev = BenchEvaluator(client=object(), judge=object(), config=BenchConfig(tasks_base=tmp_path / "tasks", workdir=tmp_path / "work", pass_score=0.6),
                        run_task=run_task, judge_output=judge_output)
    return ev, rolled, judged


def test_a_rollout_gets_the_harness_text_and_skills_only_and_a_task_passes_at_the_threshold(tmp_path):
    ev, rolled, judged = make(tmp_path, {"occ/task1": SimpleNamespace(status="graded", score=0.6), "occ/task2": SimpleNamespace(status="graded", score=0.59),
                                         "occ/task3": SimpleNamespace(status="no_output", score=0.0)})
    fb = ev.evaluate(harness(), ["occ/task1", "occ/task2", "occ/task3"])
    assert fb.passed == {"occ/task1"} and fb.failed == {"occ/task2", "occ/task3"}
    kw = rolled[0][1]
    assert kw["extra_system"] == "Check every deliverable before finishing." and list(kw["extra_files"]) == ["skills/verify/SKILL.md"]
    assert kw["network"] == "none" and kw["attempts"] == 2
    assert "no deliverable" in fb.results[2].detail and "rubric" not in " ".join(r.detail for r in fb.results).lower()
    assert all("/eval-0001/" in d and "/judge/" in j for _, d, j in judged)                # each evaluation has its own folder


def test_an_ungraded_task_makes_the_evaluation_incomplete_instead_of_a_pass_or_a_fail(tmp_path):
    ev, _, _ = make(tmp_path, {"a": SimpleNamespace(status="graded", score=1.0), "b": SimpleNamespace(status="ungraded", reason="judge endpoint down")})
    with pytest.raises(EvaluationIncomplete, match="could not be graded"):
        ev.evaluate(harness(), ["a", "b"])
