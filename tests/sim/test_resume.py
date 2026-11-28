# SPDX-License-Identifier: AGPL-3.0-or-later
"""A run that stopped is continued from what it wrote: nothing finished is generated, judged or proposed again, and a failure to grade says what it is (G1 amendment 4)."""
import json
from types import SimpleNamespace

import pytest

from agent_harness.sim.bench_eval import BenchConfig, BenchEvaluator, EvaluationIncomplete, tally_usage
from agent_harness.sim.loop import HarnessOptimizer, LoopRefused
from agent_harness.sim import SelfImprovementMemory

from tests.sim.test_loop import EVOLVE, HELDOUT, PROV, Evaluator, Proposer, Selector, edit, opt, seed
from tests.sim.test_bench_eval import harness as bench_harness


# ── the evaluator ─────────────────────────────────────────────────────────────────────────────────────────
def make(tmp_path, verdict_for):
    rolled = []

    def run_task(tasks_base, task, out, client, **kw):
        from pathlib import Path
        rolled.append(task)
        Path(out, "output").mkdir(parents=True, exist_ok=True)
        Path(out, "output", "a.md").write_text("x")
        s = {"task": task, "stop": "done", "turns": 3, "deliverables": ["a.md"], "prompt_tokens": 1000, "completion_tokens": 50}
        Path(out, "summary.json").write_text(json.dumps(s))
        return s

    def judge_output(task, tasks_base, deliver, details, cfg, finished=True):
        from pathlib import Path
        Path(details).parent.mkdir(parents=True, exist_ok=True)
        Path(str(details) + ".usage.jsonl").write_text(json.dumps({"prompt_tokens": 70000, "completion_tokens": 900}) + "\n" + json.dumps({"error": "BadRequestError"}) + "\n")
        return verdict_for(task)

    ev = BenchEvaluator(None, None, BenchConfig(tasks_base=tmp_path / "t", workdir=tmp_path / "w", rollouts=1, workers=2), run_task=run_task, judge_output=judge_output)
    return ev, rolled


GRADED = SimpleNamespace(status="graded", score=0.7, reason="")


def too_big(task):
    return SimpleNamespace(status="ungraded", score=None, reason="prompt too large", context_exceeded=True, prompt_limit=98304, prompt_tokens_at_least=98305)


def test_an_evaluation_that_cannot_be_graded_says_it_is_a_context_problem_and_settles_the_rest(tmp_path):
    ev, rolled = make(tmp_path, lambda t: too_big(t) if t == "x/task2" else GRADED)
    with pytest.raises(EvaluationIncomplete) as e:
        ev.evaluate(bench_harness(), ["x/task1", "x/task2", "x/task3"])
    x = e.value
    assert x.kind == "judge_context_exceeded" and [u["task"] for u in x.ungraded] == ["x/task2"] and x.ungraded[0]["prompt_limit"] == 98304
    status = json.loads((x.evaluation_dir / "evaluation.json").read_text())
    assert status["complete"] is False and {r["task"]: r["status"] for r in status["rollouts"]} == {"x/task1": "graded", "x/task2": "ungraded", "x/task3": "graded"}
    assert sorted(rolled) == ["x/task1", "x/task2", "x/task3"]              # every rollout finished and is sealed; the one that cannot be graded did not stop the others


def test_another_kind_of_failure_is_reported_as_judge_infrastructure(tmp_path):
    ev, _ = make(tmp_path, lambda t: SimpleNamespace(status="ungraded", score=None, reason="3 rubric criteria got no judge reply"))
    with pytest.raises(EvaluationIncomplete) as e:
        ev.evaluate(bench_harness(), ["x/task1"])
    assert e.value.kind == "judge_infrastructure"


def test_a_restarted_process_continues_in_the_same_directory_and_generates_and_judges_only_what_is_missing(tmp_path):
    ev, rolled = make(tmp_path, lambda t: too_big(t) if t == "x/task2" else GRADED)
    with pytest.raises(EvaluationIncomplete):
        ev.evaluate(bench_harness(), ["x/task1", "x/task2"])
    ev2, rolled2 = make(tmp_path, lambda t: GRADED)                            # a new process: no counter, no memory, only the directory
    fb = ev2.evaluate(bench_harness(), ["x/task1", "x/task2"])
    assert rolled2 == [] and len(fb.results) == 2                              # no answer was generated again
    assert len(list((tmp_path / "w").glob("eval-*"))) == 1                     # the same evaluation directory


def test_a_different_harness_never_lands_in_another_harness_directory(tmp_path):
    ev, _ = make(tmp_path, lambda t: GRADED)
    ev.evaluate(bench_harness(), ["x/task1"])
    other = bench_harness()
    other.set("systemprompt.md", "something else")
    ev2, rolled = make(tmp_path, lambda t: GRADED)
    ev2.evaluate(other, ["x/task1"])
    assert rolled == ["x/task1"] and len(list((tmp_path / "w").glob("eval-*"))) == 2


def test_tokens_are_recorded_for_a_local_model_with_no_cost(tmp_path):
    ev, _ = make(tmp_path, lambda t: GRADED)
    fb = ev.evaluate(bench_harness(), ["x/task1", "x/task2"])
    assert fb.cost_usd == 0.0
    assert fb.usage["actor_prompt_tokens"] == 2000 and fb.usage["actor_completion_tokens"] == 100
    assert fb.usage["judge_prompt_tokens"] == 140000 and fb.usage["judge_requests"] == 2 and fb.usage["judge_failed_requests"] == 2 and fb.usage["judge_measured"] is True
    assert fb.to_json()["usage"] == dict(fb.usage)
    t = tally_usage(tmp_path / "w")
    assert t["rollouts"] == 2 and t["actor_prompt_tokens"] == 2000 and t["judge_prompt_tokens"] == 140000


def test_tokens_are_tallied_for_a_run_whose_judge_was_not_metered_and_say_so(tmp_path):
    ev, _ = make(tmp_path, lambda t: GRADED)
    ev.evaluate(bench_harness(), ["x/task1"])
    for f in (tmp_path / "w").rglob("*.usage.jsonl"):
        f.unlink()
    t = tally_usage(tmp_path / "w")
    assert t["actor_prompt_tokens"] == 1000 and t["judge_measured"] is False and t["judge_prompt_tokens"] == 0


# ── the optimizer ─────────────────────────────────────────────────────────────────────────────────────────
class Breaks(Evaluator):
    """Fails to grade the n-th evaluation, like a judge that cannot answer."""

    def __init__(self, fail_on):
        super().__init__()
        self.fail_on = fail_on

    def evaluate(self, harness, tasks):
        if len(self.calls) + 1 == self.fail_on:
            self.calls.append(tuple(tasks))
            raise EvaluationIncomplete("1 task(s) could not be graded", kind="judge_infrastructure")
        return super().evaluate(harness, tasks)


def resumed(tmp_path, proposer, evaluator=None, selector=None):
    return HarnessOptimizer.resume(tmp_path / "run", evaluator=evaluator or Evaluator(), proposer=proposer, selector=selector or Selector(), evolve_tasks=EVOLVE,
                                   heldout_tasks=HELDOUT, provenance=PROV)


def test_an_interrupted_round_is_continued_with_the_proposals_it_already_made(tmp_path):
    prop = Proposer([[edit("You are an agent. [e1]", "markers", ["e1"])], [edit("You are an agent. [e1][e2]", "markers2", ["e2"])]])
    o = opt(tmp_path, prop, evaluator=Breaks(fail_on=99))
    o.step()                                                      # round 1 completes and promotes
    done_calls = len(o._evaluator.calls)
    o._evaluator.fail_on = done_calls + 1                         # the next evaluation (round 2's candidate) cannot be graded
    with pytest.raises(EvaluationIncomplete):
        o.step()
    assert len(prop.seen) == 2                                    # the proposer was asked for round 2 before the failure
    manifest = json.loads((tmp_path / "run" / "rounds" / "round-002.manifest.json").read_text())
    assert manifest["proposals"][0]["edits"] == {"systemprompt.md": "You are an agent. [e1][e2]"}

    prop2, ev2 = Proposer([[edit("DIFFERENT", "drawn again", ["e3"])]]), Evaluator()
    o2 = resumed(tmp_path, prop2, evaluator=ev2)
    assert o2.round == 1 and len(o2.history) == 1 and o2.harness.get("systemprompt.md") == "You are an agent. [e1]"
    assert o2.memory.problem("P1").name == "markers"
    rec = o2.step()
    assert prop2.seen == []                                       # the round-2 proposals are the ones on disk: the proposer is not asked again
    assert rec.round == 2 and rec.promoted and "[e2]" in o2.harness.get("systemprompt.md")
    assert o2.memory.problem("P2").name == "markers2"
    assert json.loads((tmp_path / "run" / "state.json").read_text())["round"] == 2
    assert (tmp_path / "run" / "rounds" / "round-002.outcome.json").is_file()


def test_resuming_with_nothing_interrupted_starts_the_next_round_normally(tmp_path):
    prop = Proposer([[edit("You are an agent. [e1]", "markers", ["e1"])]])
    opt(tmp_path, prop).step()
    prop2 = Proposer([[edit("You are an agent. [e1][e2]", "more markers", ["e2"])]])
    o2 = resumed(tmp_path, prop2)
    o2.step()
    assert len(prop2.seen) == 1 and o2.round == 2


def test_resume_refuses_an_interrupted_round_whose_manifest_has_no_proposals(tmp_path):
    prop = Proposer([[edit("You are an agent. [e1]", "markers", ["e1"])], [edit("x [e2]", "m2", ["e2"])]])
    o = opt(tmp_path, prop, evaluator=Breaks(fail_on=99))
    o.step()
    o._evaluator.fail_on = len(o._evaluator.calls) + 1
    with pytest.raises(EvaluationIncomplete):
        o.step()
    mf = tmp_path / "run" / "rounds" / "round-002.manifest.json"
    d = json.loads(mf.read_text())
    d.pop("proposals")                                            # a manifest written before proposals were stored (G1 run 2's)
    mf.write_text(json.dumps(d))
    with pytest.raises(LoopRefused, match="predates"):
        resumed(tmp_path, Proposer([]))


def test_resume_does_not_touch_the_stored_state_until_a_round_completes(tmp_path):
    opt(tmp_path, Proposer([[edit("You are an agent. [e1]", "markers", ["e1"])]])).step()
    before = {f.name: f.read_bytes() for f in (tmp_path / "run").iterdir() if f.is_file()}
    resumed(tmp_path, Proposer([]))
    assert {f.name: f.read_bytes() for f in (tmp_path / "run").iterdir() if f.is_file()} == before
