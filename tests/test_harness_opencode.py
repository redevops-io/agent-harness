import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_harness.bench import experiment as ex, harnesses
from agent_harness.bench.cells import arm_id
from agent_harness.bench.harnesses import opencode as oc


def make_exp(tmp_path, think=False):
    (tmp_path / "tasks.txt").write_text("a/task1\nb/task1\n")
    cfg = {"root": str(tmp_path / "exp"), "tasks_file": str(tmp_path / "tasks.txt"), "tasks_base": str(tmp_path / "base"),
           "cells": {"q8": {"provider": "px48", "model": "M", "window": 65536, "think": think}}, "arms": ["opencode"],
           "harness2_root": str(tmp_path / "h2"), "egress_proxy": "http://127.0.0.1:8766", "limits": {"max_seconds": 123}}
    return ex.Experiment(cfg)


def test_the_adapter_is_discoverable_by_the_runner():
    assert harnesses.load("opencode") is oc


def test_child_environment_routes_the_executor_through_the_proxy_with_the_cells_label(tmp_path):
    for think, label in ((False, "oc"), (True, "think")):
        e = make_exp(tmp_path, think)
        env = oc.child_env(e, e.cells[0])
        cfg = json.loads(env["OPENCODE_CONFIG_CONTENT"])["provider"]["px48"]
        assert cfg["options"]["baseURL"] == f"http://127.0.0.1:8765/px48/{label}/v1" and cfg["models"]["M"]["limit"] == {"context": 65536, "output": 16384}
        assert env["HARNESS2_SOLVER_MODEL"] == "px48/M" and Path(env["HARNESS2_RUNS_DIR"]) == tmp_path / "exp" / "opencode" / "q8"
        assert env["HTTPS_PROXY"] == "http://127.0.0.1:8766"


def test_child_command_runs_this_module_with_the_task_and_time_limit():
    cmd = oc.child_command("a/task1", "lbl", 123)
    assert cmd[:3] == [sys.executable, "-m", "agent_harness.bench.harnesses.opencode"] and cmd[cmd.index("--timeout") + 1] == "123"


def test_result_line_is_found_among_upstream_noise():
    assert oc.parse_result("noise\nRESULT " + json.dumps({"out": "/x"}) + "\n") == {"out": "/x"}
    with pytest.raises(ValueError):
        oc.parse_result("no result here")


def test_generate_registers_outputs_no_output_and_crashes_differently(tmp_path):
    e = make_exp(tmp_path)
    cell = e.cells[0]
    out = tmp_path / "exp" / "opencode" / "q8" / "jb" / "dataset" / "a/task1" / "model_output" / "oc-q8__a__task1__base1"
    out.mkdir(parents=True)
    (out / "r.md").write_text("x")
    traj = tmp_path / "exp" / "opencode" / "q8" / "jb" / "dataset" / "a/task1" / "model_traj" / out.name
    traj.mkdir(parents=True)
    (traj / "t.jsonl").write_text(json.dumps({"type": "step_finish", "part": {"tokens": {"input": 700, "cache": {"read": 300, "write": 0}}}}) + "\n")
    calls = []

    def runner(cmd, cwd, env, capture_output, text):
        task = cmd[cmd.index("--task") + 1]
        calls.append((task, str(cwd), env["HARNESS2_SOLVER_MODEL"]))
        if task == "a/task1":
            return SimpleNamespace(returncode=0, stdout="RESULT " + json.dumps({"out": str(out)}), stderr="")
        return SimpleNamespace(returncode=1, stdout="", stderr="boom")

    oc.generate(e, cell, 1, runner=runner)
    a = arm_id("opencode", "q8")
    assert e.board.finished(a, "a/task1") and e.board.info(a, "a/task1")["deliverables"] == ["r.md"] and e.board.info(a, "a/task1")["peak_prompt"] == 1000
    assert not e.board.finished(a, "b/task1") and "boom" in e.board.info(a, "b/task1")["error"]            # crash: not finished, not 0
    assert Path(calls[0][1]).name == "h2" and calls[0][2] == "px48/M"
    # a runner that exhausted its time with nothing delivered is a FINISHED rollout with no output
    calls.clear()
    oc.generate(e, cell, 1, runner=lambda *a, **k: SimpleNamespace(returncode=0, stdout="RESULT " + json.dumps({"out": None}), stderr=""))
    assert e.board.finished(a, "b/task1") and e.board.info(a, "b/task1")["deliverables"] == [] and e.board.info(a, "b/task1")["stop"] == "no_output"
    assert e.board.pending(a, e.tasks) == ["a/task1", "b/task1"]                                           # both finished and awaiting judgement


def test_the_child_runs_this_checkout_not_an_installed_copy(tmp_path, monkeypatch):
    from pathlib import Path
    monkeypatch.setenv("PYTHONPATH", "/somewhere/else")
    e, seen = make_exp(tmp_path), []

    def runner(cmd, cwd, env, capture_output, text):
        seen.append(env["PYTHONPATH"])
        return SimpleNamespace(returncode=1, stdout="", stderr="x")
    oc.generate(e, e.cells[0], 1, runner=runner)
    root = str(Path(oc.__file__).resolve().parents[3])
    assert (Path(root) / "agent_harness" / "bench" / "harnesses" / "opencode.py").is_file()
    assert seen and all(p.split(os.pathsep)[0] == root and p.endswith(os.pathsep + "/somewhere/else") for p in seen)    # ours first, the caller's kept
