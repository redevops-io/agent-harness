# SPDX-License-Identifier: AGPL-3.0-or-later
"""The proposer-evaluator-selector loop: scripted proposer and selector, a deterministic evaluator, real harness trees."""
import json

import pytest

from agent_harness.h2.spec import OPENCODE, stock_spec
from agent_harness.sim import MemoryCondition, Provenance, SelfImprovementMemory
from agent_harness.sim.loop import (Feedback, HarnessOptimizer, LoopRefused, Proposal, TaskResult, TransferCondition,
                                    apply_edits, evaluate_heldout, start_target_run)

EVOLVE = ["e1", "e2", "e3", "e4"]
HELDOUT = ["h1", "h2"]
PROV = Provenance(domain="coding", benchmark="bench-a", policy="model-x@1", optimizer="opt@1", evaluator="tests@1", run="src")


def seed():
    s = stock_spec(OPENCODE)
    s.set("systemprompt.md", "You are an agent.")
    return s


class Evaluator:
    """A task passes when the system prompt contains its marker; e4 breaks when the prompt says BREAK."""

    def __init__(self, on_call=None):
        self.calls, self.on_call = [], on_call

    def evaluate(self, harness, tasks):
        self.calls.append(tuple(tasks))
        if self.on_call:
            self.on_call()
        text = harness.get("systemprompt.md")
        res = []
        for t in tasks:
            ok = ("BREAK" not in text) if t == "e4" else (f"[{t}]" in text)
            res.append(TaskResult(t, ok, 1.0 if ok else 0.0, "" if ok else f"{t}: marker missing"))
        return Feedback(harness_id="", results=tuple(res))


class Proposer:
    def __init__(self, script):
        self.script, self.seen = list(script), []

    def propose(self, *, harness, feedback, history, memory, candidates):
        self.seen.append({"memory": memory, "history": len(history), "failed": sorted(feedback.failed)})
        return self.script.pop(0) if self.script else []


class Selector:
    def __init__(self, picks=None, screen=None):
        self.picks, self.screens, self.seen = list(picks or []), screen, []

    def screen(self, proposals, *, harness, memory):
        self.seen.append(memory)
        return [True] * len(proposals) if self.screens is None else self.screens(proposals)

    def choose(self, evaluated, *, current, memory):
        if self.picks:
            return self.picks.pop(0)
        best = max(range(len(evaluated)), key=lambda i: evaluated[i][1].score, default=None)
        return best if best is not None and evaluated[best][1].score > current.score else None


def edit(text, cause, predicted, what="edit the prompt"):
    return Proposal(root_cause=cause, what=what, rationale="because " + cause, edits={"systemprompt.md": text}, predicted=tuple(predicted))


def opt(tmp_path, proposer, selector=None, evaluator=None, **kw):
    mem = kw.pop("memory", None) or SelfImprovementMemory(run="src")
    return HarnessOptimizer(evaluator=evaluator or Evaluator(), proposer=proposer, selector=selector or Selector(), workdir=tmp_path / "run",
                            harness=kw.pop("harness", None) or seed(), memory=mem, condition=kw.pop("condition", MemoryCondition.FULL),
                            evolve_tasks=EVOLVE, provenance=PROV, heldout_tasks=HELDOUT, **kw)


def test_a_round_promotes_the_chosen_edit_and_records_its_prediction_outcome_and_provenance(tmp_path):
    p = Proposer([[edit("You are an agent. [e1]", "missing task markers", ["e1", "e2"]), edit("You are an agent. [e1][e2]", "missing task markers", ["e1", "e2"])]])
    o = opt(tmp_path, p)
    rec = o.step()
    assert rec.chosen == 1 and rec.promoted == o.harness_id and "[e2]" in o.harness.get("systemprompt.md")
    prob = o.memory.problem("P1")
    assert prob.name == "missing task markers" and len(prob.attempts) == 1                # only the chosen candidate is recorded
    a = prob.attempts[0]
    assert a.accepted and a.fixed == ("e1", "e2") and a.missed == () and (a.passed, a.total) == (3, 4)
    assert a.provenance.benchmark == "bench-a" and a.provenance.harness_version == o.promotions[0]["harness"] and a.candidate_harness == o.harness_id
    assert json.loads((tmp_path / "run" / "sim.json").read_text())["problems"][0]["attempts"][0]["what"] == "edit the prompt"


def test_the_change_manifest_with_every_prediction_is_on_disk_before_any_candidate_is_evaluated(tmp_path):
    seen = []
    manifest = tmp_path / "run" / "rounds" / "round-001.manifest.json"
    ev = Evaluator(on_call=lambda: seen.append(manifest.is_file()))
    o = opt(tmp_path, Proposer([[edit("x [e1]", "missing task markers", ["e1"])]]), evaluator=ev)
    o.step()
    assert seen == [False, True]                           # the first call scores the current harness; the candidate runs after the manifest
    doc = json.loads(manifest.read_text())
    assert doc["candidates"][0]["predicted"] == ["e1"] and "outcome" not in doc["candidates"][0]


def test_a_missed_prediction_and_a_regression_are_recorded_like_a_success(tmp_path):
    o = opt(tmp_path, Proposer([[edit("x [e1][e2][e3] BREAK", "missing task markers", ["e1", "e2", "e3"])]]), selector=Selector(picks=[0]), allow_score_drop=True)
    o._current = None
    start = o._evaluate(o.harness)
    assert start.passed == {"e4"}
    o.step()
    a = o.memory.problem("P1").attempts[0]
    assert a.fixed == ("e1", "e2", "e3") and a.regressed == ("e4",) and a.accepted


def test_no_accepted_candidate_leaves_the_harness_and_the_memory_unchanged(tmp_path):
    o = opt(tmp_path, Proposer([[edit("no markers at all", "wording", ["e1"])]]))
    before = o.harness_id
    rec = o.step()
    assert rec.chosen is None and o.harness_id == before and o.memory.problems == [] and "unchanged" in rec.note


def test_a_choice_that_scores_below_the_current_harness_is_not_promoted_whatever_the_selector_says(tmp_path):
    o = opt(tmp_path, Proposer([[edit("BREAK", "wording", [])]]), selector=Selector(picks=[0]))
    before = o.harness_id
    rec = o.step()
    assert rec.promoted is None and o.harness_id == before and "not promoted" in rec.note and o.memory.problems == []


def test_held_out_tasks_cannot_enter_the_loop(tmp_path):
    ev = Evaluator()
    o = opt(tmp_path, Proposer([[edit("x [e1]", "missing task markers", ["e1"])], [edit("x [e1][h1]", "missing task markers", ["h1"])]]), evaluator=ev)
    o.step()
    rec = o.step()
    assert "outside the evolve set" in rec.candidates[0]["refused"] and rec.promoted is None
    assert all(set(c) <= set(EVOLVE) for c in ev.calls)                                   # no call ever carried a held-out task
    with pytest.raises(LoopRefused, match="overlap"):
        HarnessOptimizer(evaluator=ev, proposer=Proposer([]), selector=Selector(), workdir=tmp_path / "bad", harness=seed(),
                         memory=SelfImprovementMemory(run="r"), condition=MemoryCondition.FULL, evolve_tasks=["e1", "h1"], provenance=PROV, heldout_tasks=HELDOUT)
    assert not hasattr(o, "evaluate_heldout") and evaluate_heldout(ev, o.harness, HELDOUT).results[0].task_id == "h1"


def test_the_optimizer_cannot_edit_guardrails_hooks_plugins_or_files_outside_the_harness(tmp_path):
    base = seed()
    for path in ("guardrails.md", "plugin/evil.ts", "../../etc/passwd", "/abs/path", "random.txt"):
        with pytest.raises(LoopRefused):
            apply_edits(base, Proposal("c", "w", "r", {path: "x"}))
    with pytest.raises(LoopRefused, match="changes nothing"):
        apply_edits(base, Proposal("c", "w", "r", {"systemprompt.md": "You are an agent."}))
    o = opt(tmp_path, Proposer([[Proposal("weaken the rules", "delete the guardrails", "they block me", {"guardrails.md": ""}, ("e1",))]]))
    rec = o.step()
    assert "protected" in rec.candidates[0]["refused"] and rec.promoted is None and "outcome" not in rec.candidates[0]


def test_a_rollback_restores_an_earlier_version_keeps_the_memory_and_is_recorded(tmp_path):
    o = opt(tmp_path, Proposer([[edit("x [e1]", "missing task markers", ["e1"])], [edit("x [e1][e2]", "missing task markers", ["e2"])]]))
    o.run(2)
    first, second = o.promotions[1]["harness"], o.promotions[2]["harness"]
    assert o.harness_id == second and len(o.memory.problem("P1").attempts) == 2
    o.rollback(first, reason="e2 marker confused a downstream tool")
    assert o.harness_id == first and "[e2]" not in o.harness.get("systemprompt.md")
    assert len(o.memory.problem("P1").attempts) == 2 and o.promotions[-1]["kind"] == "rollback" and o.promotions[-1]["from"] == second
    with pytest.raises(LoopRefused):
        o.rollback("0" * 64, reason="x")
    with pytest.raises(LoopRefused):
        o.rollback(first, reason=" ")


def test_the_proposer_and_the_selector_both_receive_the_memory_under_the_runs_condition(tmp_path):
    src = opt(tmp_path, Proposer([[edit("x [e1]", "missing task markers", ["e1"])]]))
    src.step()
    shown = {}
    for cond in TransferCondition:
        harness, mem, view = start_target_run(cond, seed=seed(), source_harness=src.harness, source_memory=src.memory, target_run="tgt-" + cond.value, domain="coding")
        p, s = Proposer([[edit("y [e1][e2]", "missing task markers", ["e2"])]]), Selector()
        o = HarnessOptimizer(evaluator=Evaluator(), proposer=p, selector=s, workdir=tmp_path / cond.value, harness=harness, memory=mem,
                             condition=view, evolve_tasks=EVOLVE, provenance=PROV)
        o.step()
        assert p.seen[0]["memory"] == s.seen[0]                                            # the same record for both
        shown[cond] = p.seen[0]["memory"]
        assert ("[e1]" in harness.get("systemprompt.md")) == (cond is not TransferCondition.COLD_START)
    assert shown[TransferCondition.COLD_START] == "" and shown[TransferCondition.HARNESS_TRANSFER] == ""
    assert "rationale" not in shown[TransferCondition.PROBLEM_ONLY] and "missing task markers" in shown[TransferCondition.PROBLEM_ONLY]
    assert "because missing task markers" in shown[TransferCondition.FULL_SIM] and '"carried_from"' in shown[TransferCondition.FULL_SIM]
