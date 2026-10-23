import json
from types import SimpleNamespace

import pytest

from agent_harness.bench import ablation, experiment as ex, harnesses
from agent_harness.bench.cells import arm_id
from agent_harness.bench.evalkit import JudgeResult, Scoreboard
from agent_harness.bench.harnesses import ours_cr, ours_cr_pin


def make_exp(tmp_path, arms):
    (tmp_path / "tasks.txt").write_text("a/task1\n")
    return ex.Experiment({"root": str(tmp_path / "exp"), "tasks_file": str(tmp_path / "tasks.txt"), "tasks_base": str(tmp_path / "base"),
                          "cells": {"q8": {"provider": "px", "model": "M", "window": 65536}}, "arms": arms, "trace": True})


def test_repeats_are_separate_arms_of_the_same_setting(tmp_path):
    e = make_exp(tmp_path, ["ours-cr", "ours-cr-r2", "ours-cr-pin-r3", "ours-nocr-r2"])
    seen = []

    def run_task(tasks_base, task, out, client, **kw):
        seen.append((out.parts[-3], kw["context"], kw.get("context_options"), kw["trace"]))
        return {"deliverables": ["r.md"], "stop": "done", "archived_items": 0, "retrieval_calls": 0, "seconds": 1.0}
    for arm in e.arms:
        harnesses.load(arm).generate(e, e.cells[0], 1, run_task=run_task, client_factory=lambda: SimpleNamespace(model="M"))
    assert seen == [("ours-cr", "context-runtime", {"policy": "v2"}, True),
                    ("ours-cr-r2", "context-runtime", {"policy": "v2"}, True),
                    ("ours-cr-pin-r3", "context-runtime", {"policy": "v2", "pin_input_reads": True}, True),
                    ("ours-nocr-r2", "none", None, True)]
    for arm in e.arms:
        assert e.board.finished(arm_id(arm, "q8"), "a/task1")


def test_arms_that_cannot_be_repeated_or_do_not_exist_are_refused():
    with pytest.raises(ValueError):
        harnesses.load("nonsense-r2")
    with pytest.raises(ValueError):
        harnesses.load("harness2-r2")                 # its generate() takes no name=


def test_the_pin_arm_differs_from_the_plain_arm_only_by_the_pin_option():
    assert ours_cr_pin.OPTIONS == {**ours_cr.OPTIONS, "pin_input_reads": True}


def test_judging_records_criteria_passed_beside_the_group_score(tmp_path):
    b = Scoreboard(tmp_path / "s.jsonl")
    b.mark_generated("a@q8", "t", deliverables=["f"], stop="done")
    b.record_judgement("a@q8", "t", JudgeResult("graded", 0.0, 0, 72, 0, 10, criteria_passed=13, criteria_total=33), "J")
    info = Scoreboard(tmp_path / "s.jsonl").info("a@q8", "t")
    assert info["score"] == 0.0 and info["criteria_passed"] == 13 and info["criteria_total"] == 33


def test_summary_reports_both_metrics_with_spread_and_trace_indicators(tmp_path):
    run = tmp_path / "run1"
    (run / "output").mkdir(parents=True)
    (run / "trace1").mkdir()
    (run / "trace1" / "trace.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"kind": "projection", "turn": 1, "rewritten": 0}, {"kind": "projection", "turn": 2, "rewritten": 7},
        {"kind": "tool_full", "cut": True}, {"kind": "tool_full", "cut": False}]) + "\n")
    (run / "trajectory.jsonl").write_text(json.dumps({"kind": "compact"}) + "\n" + json.dumps({"kind": "placeholder_blocked"}) + "\n")
    states = {("ours-cr@q8", "t"): {"finished": True, "status": "graded", "score": 0.0, "criteria_passed": 13, "criteria_total": 33,
                                    "stop": "done", "dir": str(run / "output")},
              ("ours-cr-r2@q8", "t"): {"finished": True, "status": "graded", "score": 0.1, "criteria_passed": 20, "criteria_total": 33,
                                       "stop": "max_seconds", "dir": str(run / "output")},
              ("other@q8", "x"): {"finished": True}}
    md = ablation.summarize(states, "t", ["q8"])
    row = [l for l in md.splitlines() if l.startswith("| ours-cr |")][0]
    assert "| 2 | 2 |" in row and "0.05 ±0.05 [0.00-0.10]" in row and "0.50 ±0.11 [0.39-0.61]" in row     # both metrics, spread over the two repeats
    assert "| 1 |" in row and "7" in row                                                         # one run hit a limit; max rewritten 7


def test_the_corrected_h2_arm_uses_the_same_options_as_the_corrected_base_arm(tmp_path):
    from agent_harness.bench.harnesses import ours_h2_cr
    from tests.test_harness_ours_h2 import put, trees
    assert ours_h2_cr.OPTIONS == ours_cr.OPTIONS == {"policy": "v2"}
    e = make_exp(tmp_path, ["ours-h2-cr", "ours-h2-cr-r2"])
    put(e, e.cells[0], "a/task1", *trees())
    got = []

    def run_task(tasks_base, task, out, client, **kw):
        got.append((out.parts[-3], kw["context"], kw["context_options"], kw["trace"], "extra_system" in kw and bool(kw["extra_system"])))
        return {"deliverables": ["r.md"], "stop": "done", "archived_items": 0, "retrieval_calls": 0, "seconds": 1.0}
    for arm in e.arms:
        harnesses.load(arm).generate(e, e.cells[0], 1, run_task=run_task, client_factory=lambda: SimpleNamespace(model="M"))
    assert got == [("ours-h2-cr", "context-runtime", {"policy": "v2"}, True, True), ("ours-h2-cr-r2", "context-runtime", {"policy": "v2"}, True, True)]
