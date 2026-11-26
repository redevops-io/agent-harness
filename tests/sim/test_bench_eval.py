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
    from pathlib import Path
    assert all("eval-0001" in Path(d).parts and "judge" in Path(j).parts for _, d, j in judged)      # each evaluation has its own folder (any separator)


def test_an_ungraded_task_makes_the_evaluation_incomplete_instead_of_a_pass_or_a_fail(tmp_path):
    ev, _, _ = make(tmp_path, {"a": SimpleNamespace(status="graded", score=1.0), "b": SimpleNamespace(status="ungraded", reason="judge endpoint down")})
    with pytest.raises(EvaluationIncomplete, match="could not be graded"):
        ev.evaluate(harness(), ["a", "b"])


def _details(path, unmet):
    from pathlib import Path
    import json
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"rubrics": [{"index": 0, "result": {"criteria_results": [{"criterion": c, "passed": ok} for c, ok in unmet]}}]}))


def test_several_rollouts_are_averaged_and_each_is_in_the_detail_and_the_web_is_off(tmp_path):
    scores = iter([0.1, 0.5])
    rolled, ev = [], None

    def run_task(tasks_base, task, out, client, **kw):
        rolled.append(kw)
        return {"stop": "finish", "turns": 5}

    def judge_output(task, tasks_base, deliver, details, cfg, finished=True):
        return SimpleNamespace(status="graded", score=next(scores))

    ev = BenchEvaluator(client=object(), judge=object(), config=BenchConfig(tasks_base=tmp_path, workdir=tmp_path / "w", pass_score=0.5, rollouts=2),
                        run_task=run_task, judge_output=judge_output)
    fb = ev.evaluate(harness(), ["occ/task1"])
    r = fb.results[0]
    assert abs(r.score - 0.3) < 1e-9 and not r.passed                       # the mean of two rollouts, not the best one
    assert "over 2 rollout(s)" in r.detail and "rollout 1: score 0.10" in r.detail and "rollout 2: score 0.50" in r.detail
    assert all(kw["web"] is False for kw in rolled) and len(rolled) == 2


def test_the_unmet_criteria_of_the_worst_rollout_reach_the_feedback_in_the_judges_words_and_not_with_score_only(tmp_path):
    def run_task(tasks_base, task, out, client, **kw):
        return {"stop": "finish", "turns": 5}

    def judge_output(task, tasks_base, deliver, details, cfg, finished=True):
        _details(details, [("The guide says API keys stay valid until v3.", False), ("It names SUP-78234.", True), ("It lists the four webhook events.", False)])
        return SimpleNamespace(status="graded", score=0.2)

    for mode, expect in (("criteria", True), ("score", False)):
        ev = BenchEvaluator(client=object(), judge=object(), config=BenchConfig(tasks_base=tmp_path, workdir=tmp_path / mode, feedback_detail=mode, max_failed_criteria=5),
                            run_task=run_task, judge_output=judge_output)
        detail = ev.evaluate(harness(), ["occ/task1"]).results[0].detail
        assert ("unmet: The guide says API keys stay valid until v3." in detail and "unmet: It lists the four webhook events." in detail) is expect
        assert "SUP-78234" not in detail                                    # a criterion that was met is not shown


def test_tasks_can_be_rolled_in_parallel_and_the_results_keep_the_order_asked_for(tmp_path):
    import threading, time
    live, peak, lock = [0], [0], threading.Lock()

    def run_task(tasks_base, task, out, client, **kw):
        with lock:
            live[0] += 1; peak[0] = max(peak[0], live[0])
        time.sleep(0.05)
        with lock:
            live[0] -= 1
        return {"stop": "finish", "turns": 1}

    def judge_output(task, tasks_base, deliver, details, cfg, finished=True):
        return SimpleNamespace(status="graded", score=1.0)

    ev = BenchEvaluator(client=object(), judge=object(), config=BenchConfig(tasks_base=tmp_path, workdir=tmp_path / "w", workers=3), run_task=run_task, judge_output=judge_output)
    fb = ev.evaluate(harness(), ["t1", "t2", "t3", "t4"])
    assert [r.task_id for r in fb.results] == ["t1", "t2", "t3", "t4"] and peak[0] >= 2
