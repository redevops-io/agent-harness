"""Arms A/B/C for the tool-parity experiment, and the deterministic failure classification."""
import json
from types import SimpleNamespace

import pytest

from agent_harness.bench import experiment as ex, harnesses
from agent_harness.bench.cells import arm_id
from agent_harness.bench.failure_class import classify
from agent_harness.bench.harnesses import ours_cr, ours_cr_tools, ours_cr_v2


def make_exp(tmp_path, arms):
    (tmp_path / "tasks.txt").write_text("a/task1\n")
    return ex.Experiment({"root": str(tmp_path / "exp"), "tasks_file": str(tmp_path / "tasks.txt"), "tasks_base": str(tmp_path / "base"),
                          "cells": {"q8": {"provider": "px", "model": "M", "window": 65536}}, "arms": arms, "trace": True})


def run_arms(tmp_path, arms):
    e = make_exp(tmp_path, arms)
    seen = {}

    def run_task(tasks_base, task, out, client, **kw):
        seen[out.parts[-3]] = kw
        return {"deliverables": ["r.md"], "stop": "done", "archived_items": 0, "retrieval_calls": 0, "seconds": 1.0, "failure": {"class": "delivered"}}
    for arm in e.arms:
        harnesses.load(arm).generate(e, e.cells[0], 1, run_task=run_task, client_factory=lambda: SimpleNamespace(model="M"))
    return e, seen


def test_b_and_v2_differ_from_the_baseline_only_in_the_declared_options(tmp_path):
    e, seen = run_arms(tmp_path, ["ours-cr", "ours-cr-tools", "ours-cr-v2"])
    base = seen["ours-cr"]
    for k in ("tools", "network", "supervise", "vision", "check_integrity"):
        assert k not in base                                              # the baseline is called exactly as before
    b, v2 = seen["ours-cr-tools"], seen["ours-cr-v2"]
    assert {k: v for k, v in b.items() if k not in ("tools", "network")} == base       # same context, limits, window, attempts, trace
    assert b["tools"] == "parity" and b["network"] == "egress" and "supervise" not in b
    assert v2 == {**b, "supervise": True, "vision": "probe", "check_integrity": True, "max_images": 5}
    for arm in e.arms:
        assert e.board.finished(arm_id(arm, "q8"), "a/task1")
    info = e.board.info(arm_id("ours-cr-v2", "q8"), "a/task1")
    assert info["run_options"] == {"tools": "parity", "network": "egress", "supervise": True, "vision": "probe", "check_integrity": True, "max_images": 5} and info["failure"] == "delivered"


def test_arm_options_are_declared_once_and_repeatable():
    assert ours_cr_tools.RUN_OPTIONS == {"tools": "parity", "network": "egress"}
    assert ours_cr_v2.RUN_OPTIONS == {**ours_cr_tools.RUN_OPTIONS, "supervise": True, "vision": "probe", "check_integrity": True, "max_images": 5}
    assert harnesses.load("ours-cr-v2-r2").NAME == "ours-cr-v2-r2"
    assert ours_cr_v2.OPTIONS == ours_cr.OPTIONS


def traj(tmp_path, rows):
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p


def asst(n):
    return [{"kind": "assistant", "turn": i} for i in range(n)]


@pytest.mark.parametrize("rows,summary,expected", [
    (asst(10), {"deliverables": ["a.xlsx"], "stop": "done"}, "delivered"),
    (asst(30) + [{"kind": "end", "stop": "blocked"}], {"deliverables": [], "stop": "blocked"}, "blocked"),
    (asst(30) + [{"kind": "supervisor", "action": "suppressed"}] * 8, {"deliverables": [], "stop": "max_turns"}, "identical_call_loop"),
    (asst(10), {"deliverables": ["a.xlsx"], "stop": "done", "completion": {"status": "invalid"}}, "invalid_deliverable"),
    (asst(10), {"deliverables": ["a.xlsx"], "stop": "verified_complete", "completion": {"status": "structurally_valid"}}, "delivered"),
    (asst(30) + [{"kind": "end", "stop": "max_seconds"}], {"deliverables": [], "stop": "max_seconds"}, "time_limit"),
    (asst(30) + [{"kind": "end", "stop": "max_turns"}], {"deliverables": [], "stop": "max_turns"}, "turn_limit"),
    (asst(3) + [{"kind": "end", "stop": "model_error"}], {"deliverables": [], "stop": "model_error"}, "model_error"),
    (asst(3) + [{"kind": "end", "stop": "context_overflow"}], {"deliverables": [], "stop": "context_overflow"}, "context_overflow"),
    (asst(5) + [{"kind": "end", "stop": "done"}], {"deliverables": [], "stop": "done"}, "finished_without_deliverable"),
])
def test_failure_classes(tmp_path, rows, summary, expected):
    assert classify(traj(tmp_path, rows), summary)["class"] == expected


def test_failure_flags_count_network_errors_placeholders_stray_files_and_notes(tmp_path):
    rows = asst(40) + [{"kind": "tool", "name": "bash", "out": "URLError: Temporary failure in name resolution"},
                       {"kind": "tool", "name": "web_fetch", "out": "ERROR: fetch failed: Tunnel connection failed: 403 Forbidden\n[environment: not allowed]"},
                       {"kind": "placeholder_blocked"}, {"kind": "end", "stop": "max_turns"}]
    r = classify(traj(tmp_path, rows), {"deliverables": [], "stop": "max_turns", "outside_output": ["scratch/draft.docx"],
                                         "supervisor": {"events": [{"kind": "suppress"}, {"kind": "watch"}]}})
    assert r["class"] == "turn_limit" and r["network_errors"] == 2 and r["placeholder_blocked"] == 1 and r["capability_notes"] == 1
    assert r["files_outside_output"] == ["scratch/draft.docx"] and r["supervisor_events"] == ["suppress", "watch"]


def test_a_missing_trajectory_still_classifies_from_the_summary(tmp_path):
    assert classify(tmp_path / "none.jsonl", {"deliverables": [], "stop": "max_seconds"})["class"] == "time_limit"
